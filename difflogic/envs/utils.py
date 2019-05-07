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

from jaclearn.rl.env import ProxyRLEnvBase
from jaclearn.rl.space import DiscreteActionSpace

__all__ = ['MapActionProxy', 'get_action_mapping', 'get_action_mapping_graph',
           'get_action_mapping_sorting', 'get_action_mapping_blocksworld']


class MapActionProxy(ProxyRLEnvBase):
  """RL Env proxy to map actions using provided mapping function."""

  def __init__(self, other, mapping):
    super().__init__(other)
    self._mapping = mapping

  @property
  def mapping(self):
    return self._mapping

  def map_action(self, action):
    assert action < len(self._mapping)
    return self._mapping[action]

  def _get_action_space(self):
    return DiscreteActionSpace(len(self._mapping))

  def _action(self, action):
    return self.proxy.action(self.map_action(action))


def get_action_mapping(n, exclude_self=True):
  """In a matrix view, this a mapping from 1d-index to 2d-coordinate."""
  mapping = [
      (i, j) for i in range(n) for j in range(n) if (i != j or not exclude_self)
  ]
  return mapping

get_action_mapping_graph = get_action_mapping
get_action_mapping_sorting = get_action_mapping


def get_action_mapping_blocksworld(nr_blocks, exclude_self=True):
  return get_action_mapping_graph(nr_blocks + 1, exclude_self)
