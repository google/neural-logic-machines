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
"""Quick access for blocksworld environments."""

from jaclearn.rl.proxy import LimitLengthProxy

from .envs import FinalBlocksWorldEnv
from ..utils import get_action_mapping_blocksworld
from ..utils import MapActionProxy

__all__ = ['get_final_env', 'make']


def get_final_env(nr_blocks,
                  random_order=False,
                  exclude_self=True,
                  shape_only=False,
                  fix_ground=False,
                  limit_length=None):
  """Get the blocksworld environment for the final task."""
  p = FinalBlocksWorldEnv(
      nr_blocks,
      random_order=random_order,
      shape_only=shape_only,
      fix_ground=fix_ground)
  p = LimitLengthProxy(p, limit_length or nr_blocks * 4)
  mapping = get_action_mapping_blocksworld(nr_blocks, exclude_self=exclude_self)
  p = MapActionProxy(p, mapping)
  return p


def make(task, *args, **kwargs):
  if task == 'final':
    return get_final_env(*args, **kwargs)
  else:
    raise ValueError('Unknown task: {}.'.format(task))
