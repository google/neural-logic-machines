#! /usr/bin/env python3
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for tensor masking."""

import torch

import torch.autograd as ag
from jactorch.functional import meshgrid, meshgrid_exclude_self

__all__ = ['meshgrid', 'meshgrid_exclude_self', 'exclude_mask', 'mask_value']


def exclude_mask(inputs, cnt=2, dim=1):
  """Produce an exclusive mask.

  Specifically, for cnt=2, given an array a[i, j] of n * n, it produces
  a mask with size n * n where only a[i, j] = 1 if and only if (i != j).

  Args:
    inputs: The tensor to be masked.
    cnt: The operation is performed over [dim, dim + cnt) axes.
    dim: The starting dimension for the exclusive mask.

  Returns:
    A mask that make sure the coordinates are mutually exclusive.
  """
  assert cnt > 0
  if dim < 0:
    dim += inputs.dim()
  n = inputs.size(dim)
  for i in range(1, cnt):
    assert n == inputs.size(dim + i)

  rng = torch.arange(0, n, dtype=torch.long, device=inputs.device)
  q = []
  for i in range(cnt):
    p = rng
    for j in range(cnt):
      if i != j:
        p = p.unsqueeze(j)
    p = p.expand((n,) * cnt)
    q.append(p)
  mask = q[0] == q[0]
  # Mutually Exclusive
  for i in range(cnt):
    for j in range(cnt):
      if i != j:
        mask *= q[i] != q[j]
  for i in range(dim):
    mask.unsqueeze_(0)
  for j in range(inputs.dim() - dim - cnt):
    mask.unsqueeze_(-1)

  return mask.expand(inputs.size()).float()


def mask_value(inputs, mask, value):
  assert inputs.size() == mask.size()
  return inputs * mask + value * (1 - mask)
