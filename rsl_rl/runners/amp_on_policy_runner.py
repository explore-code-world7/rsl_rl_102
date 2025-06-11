# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""

"""


from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
import rsl_rl.utils
from rsl_rl.env import VecEnv
from rsl_rl.modules import *
from rsl_rl.utils import store_code_state

from rsl_rl.utils import Normalizer
from rsl_rl.utils import AMPLoader
from rsl_rl.algorithms import AMP_PPO, PPO
from rsl_rl.networks import Discriminator
from rsl_rl.utils import export_policy_as_onnx


class AMPOnPolicyRunner:
    """
    AMPOnPolicyRunner is a high-level orchestrator that manages the training and evaluation
    of a policy using Adversarial Motion Priors (AMP) combined with on-policy reinforcement learning (PPO).

    It brings together multiple components:
    - Environment (`VecEnv`)
    - Policy (`ActorCritic`, `ActorCriticRecurrent`)
    - Discriminator (Discriminator)
    - Expert dataset (AMPLoader)
    - Reward combination (task + style)
    - Logging and checkpointing

    ---
    🔧 Configuration
    ----------------
    The class expects a `train_cfg` dictionary structured with keys:
    - "policy": configuration for the policy network, including `"class_name"`
    - "algorithm": configuration for PPO/AMP_PPO, including `"class_name"`
    - "discriminator": configuration for the AMP discriminator
    - "amp_data_path": path to folder containing expert dataset(s)
    - "dataset_names": list of dataset filenames (without `.npy`)
    - "dataset_weights": list of float weights used to sample from datasets
    - "slow_down_factor": slowdown applied to real motion data to match sim dynamics
    - "num_steps_per_env": rollout horizon per environment
    - "save_interval": frequency (in iterations) for model checkpointing
    - "empirical_normalization": whether to apply running observation normalization
    - "logger": one of "tensorboard", "wandb", or "neptune"

    ---
    📦 Dataset format
    ------------------
    The expert motion datasets loaded via `AMPLoader` must be `.npy` files with a dictionary containing:

    - `"joints_list"`: List[str] — ordered list of joint names
    - `"joint_positions"`: List[np.ndarray] — joint configurations per timestep (1D arrays)
    - `"root_position"`: List[np.ndarray] — base position in world coordinates
    - `"root_quaternion"`: List[np.ndarray] — base orientation in **`xyzw`** format (SciPy default)
    - `"fps"`: float — original dataset frame rate

    Internally:
    - Quaternions are interpolated via SLERP and converted to **`wxyz`** format before being used by the model (to match Isaac Gym convention).
    - Velocities are estimated with finite differences.
    - All data is converted to torch tensors and placed on the desired device.

    ---
    🎓 AMP Reward
    -------------
    During each training step, the runner collects AMP-specific observations and computes
    a discriminator-based "style reward" from the expert dataset. This is combined
    with the environment reward as:

        `reward = 0.5 * task_reward + 0.5 * style_reward`

    This can be later generalized into a weighted or learned reward mixing policy.

    ---
    🔁 Training loop
    ----------------
    The `learn()` method performs:
    - `rollout`: collects data via `self.alg.act()` and `env.step()`
    - `style_reward`: computed from discriminator via `predict_reward(...)`
    - `storage update`: via `process_env_step()` and `process_amp_step()`
    - `return computation`: via `compute_returns()`
    - `update`: performs backpropagation with `self.alg.update()`
    - Logging via TensorBoard/WandB/Neptune

    ---
    💾 Saving and ONNX export
    --------------------------
    At each `save_interval`, the runner:
    - Saves the full state (`model`, `optimizer`, `discriminator`, `normalizer`, etc.)
    - Optionally exports the policy as an ONNX model for deployment
    - Uploads checkpoints to logging services if enabled

    ---
    📤 Inference policy
    -------------------
    `get_inference_policy()` returns a callable that takes an observation and returns an action.
    If empirical normalization is enabled, observations are normalized before inference.

    ---
    🛠️ Additional tools
    -------------------
    - Git integration via `store_code_state()` to track code changes
    - Logging of learning statistics, reward breakdown, discriminator metrics
    - Compatible with multi-task setups via dataset weights

    ---
    📚 Notes
    --------
    - This runner assumes an AMP-compatible VecEnv, providing `observations["amp"]`
    - AMP uses both current and next state to train the discriminator
    - Logging behavior is separated from core logic (WandB, Neptune, TensorBoard)
    - The Discriminator and AMP_PPO must follow expected APIs

    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.discriminator_cfg = train_cfg["discriminator"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # resolve training type depending on the algorithm
        if (self.alg_cfg["class_name"] == "PPO") or (self.alg_cfg["class_name"] == "AMP_PPO"):
            self.training_type = "rl"
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")


        # Get the size of the observation space
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]

        # resolve type of privileged observations
        if self.training_type == "rl":
            if "critic" in extras["observations"]:
                self.privileged_obs_type = "critic"  # actor-critic reinforcement learnig, e.g., PPO
            else:
                self.privileged_obs_type = None
        if self.training_type == "distillation":
            if "teacher" in extras["observations"]:
                self.privileged_obs_type = "teacher"  # policy distillation
            else:
                self.privileged_obs_type = None

        # resolve dimensions of privileged observations
        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs

        # actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic
        # actor_critic: ActorCritic | ActorCriticRecurrent | ActorCriticMoE = (
        #     actor_critic_class(
        #         num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg
        #     ).to(self.device)
        # )

        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: ActorCritic | ActorCriticRecurrent | StudentTeacher | StudentTeacherRecurrent = policy_class(
            num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # actuated_joint_names = self.env.cfg.actions.joint_positions.joint_names

        delta_t = self.env.cfg.sim.dt * self.env.cfg.decimation

        # Initilize all the ingredients required for AMP (discriminator, dataset loader)
        # num_amp_obs = extras["observations"]["amp_obs"].shape[1]
        # num_amp_obs = self.env.unwrapped.extras["amp_obs"].shape[1]//2
        amp_obs_dim = self.env.amp_observation_dim
        # amp_data = AMPLoader(
        #     self.device,
        #     self.cfg["amp_data_path"],
        #     self.cfg["dataset_names"],
        #     self.cfg["dataset_weights"],
        #     delta_t,
        #     self.cfg["slow_down_factor"],
        #     actuated_joint_names,
        # )

        # amp_data = AMPLoader(num_amp_obs, self.device)
        print(f"amp_obs_dim = {amp_obs_dim}, type={type(amp_obs_dim)}")
        self.amp_normalizer = Normalizer(amp_obs_dim, device=self.device)
        self.discriminator = Discriminator(
            amp_obs_dim * 2,  # the discriminator takes in the concatenation of the current and next observation
            self.discriminator_cfg["hidden_dims"],
            self.discriminator_cfg["reward_scale"],
            device=self.device,
        ).to(self.device)

        ## setting for rnd + symmetry resolution
        # resolve dimension of rnd gated state
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # check if rnd gated state is present
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            # get dimension of rnd gated state
            num_rnd_state = rnd_state.shape[1]
            # add rnd gated state to config
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # if using symmetry then pass the environment config object
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # this is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # Initialize the PPO algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))  # AMP_PPO

        self.alg: AMP_PPO = alg_class(
            policy=policy,
            discriminator=self.discriminator,
            # amp_data=amp_data,
            collect_reference_motions = self.env.unwrapped.collect_reference_motions,
            amp_normalizer=self.amp_normalizer,
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg = self.multi_gpu_cfg
        )
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(
                shape=[num_obs], until=1.0e8
            ).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(
                shape=[num_privileged_obs], until=1.0e8
            ).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()  # no normalization
            self.privileged_obs_normalizer = torch.nn.Identity()  # no normalization
        # init storage and model
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]       # 获取全局路径

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                self.writer.log_config(
                    self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg
                )
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                import wandb

                # Update the run name with a sequence number. This function is useful to
                # replicate the same behaviour of rsl-rl-lib before v2.3.0
                def update_run_name_with_sequence(prefix: str) -> None:
                    # Retrieve the current wandb run details (project and entity)
                    project = wandb.run.project
                    entity = wandb.run.entity

                    # Use wandb's API to list all runs in your project
                    api = wandb.Api()
                    runs = api.runs(f"{entity}/{project}")

                    max_num = 0
                    # Iterate through runs to extract the numeric suffix after the prefix.
                    for run in runs:
                        if run.name.startswith(prefix):
                            # Extract the numeric part from the run name.
                            numeric_suffix = run.name[
                                len(prefix) :
                            ]  # e.g., from "prefix564", get "564"
                            try:
                                run_num = int(numeric_suffix)
                                if run_num > max_num:
                                    max_num = run_num
                            except ValueError:
                                continue

                    # Increment to get the new run number
                    new_num = max_num + 1
                    new_run_name = f"{prefix}{new_num}"

                    # Update the wandb run's name
                    wandb.run.name = new_run_name
                    print("Updated run name to:", wandb.run.name)

                self.writer = WandbSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                update_run_name_with_sequence(prefix=self.cfg["wandb_project"])

                wandb.gym.monitor()
                self.writer.log_config(
                    self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg
                )
            elif self.logger_type == "tensorboard":
                # self.writer = TensorboardSummaryWriter(
                #     log_dir=self.log_dir, flush_secs=10
                # )
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        amp_obs = extras["observations"]["amp_obs"]
        # amp_obs = self.env.unwrapped.extras["amp_obs"][::2]
        # print(f"obs.shape = {obs.shape}, amp.shape = {amp_obs.shape}")
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        amp_obs = amp_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            # TODO: Do we need to synchronize empirical normalizers?
            #   Right now: No, because they all should converge to the same values "asymptotically".

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        self.save_interval = num_learning_iterations//20
        for it in range(start_iter+1, tot_iter+1):
            start = time.time()
            # Rollout

            mean_style_reward_log = 0
            mean_task_reward_log = 0

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_obs)
                    # 获取上一次的amp_obs
                    self.alg.act_amp(amp_obs)   
                    obs, rewards, dones, infos = self.env.step(actions)
                    _, extras = self.env.get_observations()
                    # next_amp_obs = extras["observations"]["policy"]
                    # next_amp_obs = extras["observations"]["amp_obs"]
                    # next_amp_obs = self.env.unwrapped.extras["amp_obs"][::2]
                    
                    # for special humanoid
                    # next_amp_obs = infos["amp_obs"][:, ::2]
                    # for h1
                    next_amp_obs = infos["amp_obs"]
                    obs = self.obs_normalizer(obs)
                    if "critic" in infos["observations"]:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"]["critic"]
                        )
                    else:
                        privileged_obs = obs
                    obs, privileged_obs, rewards, dones = (
                        obs.to(self.device),
                        privileged_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    next_amp_obs = next_amp_obs.to(self.device)

                    # print(f"amp_obs.shape = {amp_obs.shape}, {next_amp_obs.shape}")
                    # Process the AMP reward
                    # print(f"amp_obs.shape = {amp_obs.shape}, {next_amp_obs.shape}")
                    style_rewards = self.discriminator.predict_reward(
                        amp_obs, next_amp_obs, normalizer=self.amp_normalizer
                    )

                    mean_task_reward_log += rewards.mean().item()
                    # print(self.discriminator_cfg)
                    mean_style_reward_log += style_rewards.mean().item()

                    # Combine the task and style rewards (TODO this can be a hyperparameters)
                    # rewards = 0.5 * rewards + 0.5 * style_rewards
                    rewards += self.alg_cfg["discriminator_reward_scale"] * style_rewards

                    self.alg.process_env_step(rewards, dones, infos)
                    self.alg.process_amp_step(next_amp_obs)

                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # The next observation becomes the current observation for the next step
                    amp_obs = torch.clone(next_amp_obs)

                    if self.log_dir is not None:
                        # Book keeping
                        # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                        # different types of logging data (rewards, curriculum, etc.)
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])

                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                        # -- intrinsic and extrinsic rewards
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

            mean_style_reward_log /= self.num_steps_per_env
            mean_task_reward_log /= self.num_steps_per_env
            mean_total_reward_log = mean_style_reward_log + mean_task_reward_log

            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_entropy,
                mean_amp_loss,
                mean_grad_pen_loss,
                mean_policy_pred,
                mean_expert_pred,
                mean_accuracy_policy,
                mean_accuracy_expert,
                mean_rnd_loss,
                mean_symmetry_loss,
            ) = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(
                    os.path.join(self.log_dir, f"model_{it}.pt"), 
                    # save_onnx=True,
                    save_onnx=False,
                )
            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)
            
            print(f"amp_sample_num = {self.alg.amp_storage.num_samples}")

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(
                os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"),
                # save_onnx=True,
                save_onnx=False,
            )

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        if self.alg.policy.noise_std_type=="log":
            mean_std = torch.exp(self.alg.policy.log_std.mean())
        else:
            mean_std = self.alg.policy.std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/entropy", locs["mean_entropy"], locs["it"]
        )

        # Adding logging due to AMP
        self.writer.add_scalar("Loss/amp_loss", locs["mean_amp_loss"], locs["it"])
        self.writer.add_scalar(
            "Loss/grad_pen_loss", locs["mean_grad_pen_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/policy_pred", locs["mean_policy_pred"], locs["it"])
        self.writer.add_scalar("Loss/expert_pred", locs["mean_expert_pred"], locs["it"])
        self.writer.add_scalar(
            "Loss/accuracy_policy", locs["mean_accuracy_policy"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/accuracy_expert", locs["mean_accuracy_expert"], locs["it"]
        )

        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            if self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])

            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_style_reward", locs["mean_style_reward_log"], locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_task_reward", locs["mean_task_reward_log"], locs["it"]
            )
            if (
                self.logger_type != "wandb"
            ):  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar(
                    "Train/mean_reward/time",
                    statistics.mean(locs["rewbuffer"]),
                    self.tot_time,
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length/time",
                    statistics.mean(locs["lenbuffer"]),
                    self.tot_time,
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        # writer和terminal都print一次
        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Entropy:':>{pad}} {locs['mean_entropy']:.4f}\n"""
                f"""{'Amp loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                f"""{'Grad penalty loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                f"""{'Mean policy Pred loss:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                f"""{'Mean expert Pred loss:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                f"""{'Mean policy Accuracy:':>{pad}} {locs['mean_accuracy_policy']:.4f}\n"""
                f"""{'Mean expert Accuracy:':>{pad}} {locs['mean_accuracy_expert']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                f"""{'Mean style reward:':>{pad}} {locs['mean_style_reward_log']:.4f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Entropy:':>{pad}} {locs['mean_entropy']:.4f}\n"""
                f"""{'Amp loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                f"""{'Grad penalty loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                f"""{'Mean policy Pred loss:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                f"""{'Mean expert Pred loss:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                f"""{'Mean policy Accuracy:':>{pad}} {locs['mean_accuracy_policy']:.4f}\n"""
                f"""{'Mean expert Accuracy:':>{pad}} {locs['mean_accuracy_expert']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean style reward:':>{pad}} {locs['mean_style_reward_log']:.4f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string

        # # make the eta in H:M:S
        # eta_seconds = (
        #     self.tot_time
        #     / (locs["it"] + 1)
        #     * (locs["num_learning_iterations"] - locs["it"])
        # )

        # # Convert seconds to H:M:S
        # eta_h, rem = divmod(eta_seconds, 3600)
        # eta_m, eta_s = divmod(rem, 60)

        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            # f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            # f"""{'ETA:':>{pad}} {int(eta_h)}h {int(eta_m)}m {int(eta_s)}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                               locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n"""
        )
        print(log_string)

    def save(self, path, infos=None, save_onnx=False):
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "amp_normalizer": self.alg.amp_normalizer,
            "iter": self.current_learning_iteration,
            "infos": infos,
        }

        # -- Save RND model if used
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = (
                self.privileged_obs_normalizer.state_dict()
            )
        # save model
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

        if save_onnx:
            # Save the model in ONNX format
            # extract the folder path
            onnx_folder = os.path.dirname(path)

            # extract the iteration number from the path. The path is expected to be in the format
            # model_{iteration}.pt
            iteration = int(os.path.basename(path).split("_")[1].split(".")[0])
            onnx_model_name = f"policy_{iteration}.onnx"

            export_policy_as_onnx(
                self.alg.policy,
                normalizer=self.obs_normalizer,
                path=onnx_folder,
                filename=onnx_model_name,
            )

            if self.logger_type in ["neptune", "wandb"]:
                self.writer.save_model(
                    os.path.join(onnx_folder, onnx_model_name),
                    self.current_learning_iteration,
                )

    def load(self, path, load_optimizer=True, weights_only=False):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=weights_only)
        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        self.alg.amp_normalizer = loaded_dict["amp_normalizer"]
        # -- Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])

        # -- Load RND model if used
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])

        if self.empirical_normalization:
            if resumed_training:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(
                    loaded_dict["privileged_obs_norm_state_dict"]
                )
            else:
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])

        if resumed_training:

            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(
                self.obs_normalizer(x)
            )  # noqa: E731
        return policy

    def train_mode(self):
        self.alg.policy.train()
        self.alg.discriminator.train()
        if self.alg.rnd:
            self.alg.rnd.train()
                    
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

    def eval_mode(self):
        self.alg.policy.eval()
        self.alg.discriminator.eval()
        if self.alg.rnd:
            self.alg.rnd.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # rank of the main process
            "local_rank": self.gpu_local_rank,  # rank of the current process
            "world_size": self.gpu_world_size,  # total number of processes
        }

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'.")
        # validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'.")
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'.")

        # initialize torch distributed
        torch.distributed.init_process_group(
            backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size
        )
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)
