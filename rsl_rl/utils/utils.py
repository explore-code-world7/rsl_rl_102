# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import git
import importlib
import os
import pathlib
import torch
from typing import Callable


def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    if act_name == "elu":
        return torch.nn.ELU()
    elif act_name == "selu":
        return torch.nn.SELU()
    elif act_name == "relu":
        return torch.nn.ReLU()
    elif act_name == "crelu":
        return torch.nn.CELU()
    elif act_name == "lrelu":
        return torch.nn.LeakyReLU()
    elif act_name == "tanh":
        return torch.nn.Tanh()
    elif act_name == "sigmoid":
        return torch.nn.Sigmoid()
    elif act_name == "identity":
        return torch.nn.Identity()
    else:
        raise ValueError(f"Invalid activation function '{act_name}'.")


def split_and_pad_trajectories(tensor, dones):
    """Splits trajectories at done indices. Then concatenates them and pads with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example:
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]

    Assumes that the inputy has the following dimension order: [time, number of envs, additional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
    # add at least one full length trajectory
    trajectories = trajectories + (torch.zeros(tensor.shape[0], *tensor.shape[2:], device=tensor.device),)
    # pad the trajectories to the length of the longest trajectory
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)
    # remove the added tensor
    padded_trajectories = padded_trajectories[:, :-1]

    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
    """Does the inverse operation of  split_and_pad_trajectories()"""
    # Need to transpose before and after the masking to have proper reshaping
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


def store_code_state(logdir, repositories) -> list:
    git_log_dir = os.path.join(logdir, "git")
    os.makedirs(git_log_dir, exist_ok=True)
    file_paths = []
    for repository_file_path in repositories:
        try:
            repo = git.Repo(repository_file_path, search_parent_directories=True)
            t = repo.head.commit.tree
        except Exception:
            print(f"Could not find git repository in {repository_file_path}. Skipping.")
            # skip if not a git repository
            continue
        # get the name of the repository
        repo_name = pathlib.Path(repo.working_dir).name
        diff_file_name = os.path.join(git_log_dir, f"{repo_name}.diff")
        # check if the diff file already exists
        if os.path.isfile(diff_file_name):
            continue
        # write the diff file
        print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
        with open(diff_file_name, "x", encoding="utf-8") as f:
            content = f"--- git status ---\n{repo.git.status()} \n\n\n--- git diff ---\n{repo.git.diff(t)}"
            f.write(content)
        # add the file path to the list of files to be uploaded
        file_paths.append(diff_file_name)
    return file_paths


def string_to_callable(name: str) -> Callable:
    """Resolves the module and function names to return the function.

    Args:
        name (str): The function name. The format should be 'module:attribute_name'.

    Raises:
        ValueError: When the resolved attribute is not a function.
        ValueError: When unable to resolve the attribute.

    Returns:
        Callable: The function loaded from the module.
    """
    try:
        mod_name, attr_name = name.split(":")
        mod = importlib.import_module(mod_name)
        callable_object = getattr(mod, attr_name)
        # check if attribute is callable
        if callable(callable_object):
            return callable_object
        else:
            raise ValueError(f"The imported object is not callable: '{name}'")
    except AttributeError as e:
        msg = (
            "We could not interpret the entry as a callable object. The format of input should be"
            f" 'module:attribute_name'\nWhile processing input '{name}', received the error:\n {e}."
        )
        raise ValueError(msg)

# For amp

from typing import Tuple, Union

import torch


class RunningMeanStd:
    """
    Calculates the running mean and standard deviation of a data stream.
    Based on the parallel algorithm for calculating variance:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    Args:
        epsilon (float): Small constant to initialize the count for numerical stability.
        shape (Tuple[int, ...]): Shape of the data (e.g., observation shape).
    """

    def __init__(
        self,
        epsilon: float = 1e-4,
        shape: Tuple[int, ...] = (),
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.mean = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.var = torch.ones(shape, dtype=torch.float32, device=self.device)
        self.count = torch.tensor(epsilon, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def update(self, arr: torch.Tensor) -> None:
        """
        Updates the running statistics using a new batch of data.

        Args:
            arr (torch.Tensor): Batch of data (batch_size, *shape).
        """
        batch = arr.to(self.device, dtype=torch.float32)
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        batch_count = torch.tensor(
            batch.shape[0], dtype=torch.float32, device=self.device
        )
        self._update_from_moments(batch_mean, batch_var, batch_count)

    @torch.no_grad()
    def _update_from_moments(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: torch.Tensor,
    ) -> None:
        """
        Updates statistics using precomputed batch mean, variance, and count.

        Args:
            batch_mean (torch.Tensor): Mean of the batch.
            batch_var (torch.Tensor): Variance of the batch.
            batch_count (torch.Tensor): Number of samples in the batch.
        """
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total_count)


class Normalizer(RunningMeanStd):
    """
    A normalizer that uses running statistics to normalize inputs, with optional clipping.

    Args:
        input_dim (Tuple[int, ...]): Shape of the input observations.
        epsilon (float): Small constant added to variance to avoid division by zero.
        clip_obs (float): Maximum absolute value to clip the normalized observations.
    """

    def __init__(
        self,
        input_dim: Union[int, Tuple[int, ...]],
        epsilon: float = 1e-4,
        clip_obs: float = 10.0,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        shape = (input_dim,) if isinstance(input_dim, int) else tuple(input_dim)
        super().__init__(epsilon=epsilon, shape=shape, device=device)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize(self, input: torch.Tensor) -> torch.Tensor:
        """
        Normalizes input using running mean and std, and clips the result.

        Args:
            input (torch.Tensor): Input tensor to normalize.

        Returns:
            torch.Tensor: Normalized and clipped tensor.
        """
        x = input.to(self.device, dtype=torch.float32)
        std = (self.var + self.epsilon).sqrt()
        # print(f"utils/utils.py: x.shape = {x.shape}, {self.mean.shape}, {std.shape}")
        y = (x - self.mean) / std
        return torch.clamp(y, -self.clip_obs, self.clip_obs)

    @torch.no_grad()
    def update_normalizer(self, rollouts, expert_loader) -> None:
        """
        Updates running statistics using samples from both policy and expert trajectories.

        Args:
            rollouts: Object with method `feed_forward_generator_amp(...)`.
            expert_loader: Dataloader or similar object providing expert batches.
        """
        policy_generator = rollouts.feed_forward_generator_amp(
            None, mini_batch_size=expert_loader.batch_size
        )
        expert_generator = expert_loader.dataset.feed_forward_generator_amp(
            expert_loader.batch_size
        )

        for expert_batch, policy_batch in zip(expert_generator, policy_generator):
            batch = torch.cat((*expert_batch, *policy_batch), dim=0)
            self.update(batch)