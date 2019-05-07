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
"""The environment class for sorting tasks."""

import numpy as np

import jacinle.random as random
from jacinle.utils.meta import notnone_property
from jaclearn.rl.env import SimpleRLEnvBase


class ListSortingEnv(SimpleRLEnvBase):
  """Environment for sorting a random permutation.

  Args:
    nr_numbers: The number of numbers in the array.
  """

  def __init__(self, nr_numbers):
    super().__init__()
    self._nr_numbers = nr_numbers
    self._array = None

  @notnone_property
  def array(self):
    return self._array

  @property
  def nr_numbers(self):
    return self._nr_numbers

  def get_state(self):
    """Compute the state given the array."""
    x, y = np.meshgrid(self.array, self.array)
    number_relations = np.stack([x < y, x == y, x > y], axis=-1).astype('float')
    index = np.array(list(range(self._nr_numbers)))
    x, y = np.meshgrid(index, index)
    position_relations = np.stack([x < y, x == y, x > y],
                                  axis=-1).astype('float')
    return np.concatenate([number_relations, position_relations], axis=-1)

  def _calculate_optimal(self):
    """Calculate the optimal number of steps for sorting the array."""
    a = self._array
    b = [0 for i in range(len(a))]
    cnt = 0
    for i, x in enumerate(a):
      if b[i] == 0:
        j = x
        b[i] = 1
        while b[j] == 0:
          b[j] = 1
          j = a[j]
        assert i == j
        cnt += 1
    return len(a) - cnt

  def _restart(self):
    """Restart: Generate a random permutation."""
    self._array = random.permutation(self._nr_numbers)
    self._set_current_state(self.get_state())
    self.optimal = self._calculate_optimal()

  def _action(self, action):
    """action is a tuple (i, j), perform this action leads to the swap."""
    a = self._array
    i, j = action
    x, y = a[i], a[j]
    a[i], a[j] = y, x
    self._set_current_state(self.get_state())
    for i in range(self._nr_numbers):
      if a[i] != i:
        return 0, False
    return 1, True
