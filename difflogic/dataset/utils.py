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
"""Utility functions for customer datasets."""

import collections
import copy
import jacinle.random as random
import numpy as np

__all__ = [
    'ValidActionDataset',
    'RandomlyIterDataset',
]


class ValidActionDataset(object):
  """Collect data and sample batches of whether actions are valid or not.

  The data are collected into different bins according to the number of objects
  in the cases. At most $maxn bins are maintained. The capacity of each bin
  should be set, the newly added ones exceeding the capacity would replace the
  earliest ones in the bin (implemented using deque).

  Args:
    capacity: upper bound of the number of training instances in each bin.
    maxn: The maximum number of objects in the collected cases.
  """

  def __init__(self, capacity=5000, maxn=30):
    super().__init__()
    self.largest_n = 0
    self.maxn = maxn
    # Better to use defaultdict to replace array
    self.data = [[collections.deque(maxlen=capacity)
                  for i in range(2)]
                 for j in range(maxn + 1)]

  def append(self, n, state, action, valid):
    """add a new data point of n objects, given $state, the $action is $valid."""
    assert n <= self.maxn
    valid = int(valid)
    self.data[n][valid].append((state, action))
    self.largest_n = max(self.largest_n, n)

  def _sample(self, data, num, label):
    """Sample a batch of size $num from the data, with already determined label."""
    # assert num <= len(data)
    states, actions = [], []
    for _ in range(num):
      ind = random.randint(len(data))
      state, action = data[ind]
      states.append(state)
      actions.append([action])
    return np.array(states), np.array(actions), np.ones((num,)) * label

  def sample_batch(self, batch_size, n=None):
    """Sample a batch of data for $n objects."""
    # use the data from the bin with largest number of objects in default.
    if n is None:
      n = self.largest_n
    data = self.data[n]
    # The pos/neg ones are not strict equal if batch_size % 2 != 0.
    # Should add warning.
    num = batch_size // 2
    # if no negative ones, using all positive ones.
    c = 1 - int(len(data[0]) > 0)
    states1, actions1, labels1 = self._sample(data[c], num, c)
    # if no positive ones, using all negative ones.
    c = int(len(data[1]) > 0)
    states2, actions2, labels2 = self._sample(data[c], batch_size - num, c)
    return (np.vstack([states1, states2]),
            np.vstack([actions1, actions2]).squeeze(axis=-1),
            np.concatenate([labels1, labels2], axis=0))


class RandomlyIterDataset(object):
  """Collect data and iterate the dataset in random order."""

  def __init__(self):
    super().__init__()
    self.data = []
    self.ind = 0

  def append(self, data):
    self.data.append(data)

  @property
  def size(self):
    return len(self.data)

  def reset(self):
    self.ind = 0

  def get(self):
    """iterate the dataset with random order."""
    # Shuffle before the iteration starts.
    # The iteration should better be separated with collection.
    if self.ind == 0:
      random.shuffle(self.data)
    ret = self.data[self.ind]
    self.ind += 1
    if self.ind == self.size:
      self.ind = 0
    return copy.deepcopy(ret)
