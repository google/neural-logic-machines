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
"""Quick access for algorithmic environments."""

from jaclearn.rl.proxy import LimitLengthProxy

from .sort_envs import ListSortingEnv
from ..utils import get_action_mapping_sorting
from ..utils import MapActionProxy

__all__ = ['get_sort_env', 'make']


def get_sort_env(n, exclude_self=True):
  env_cls = ListSortingEnv
  p = env_cls(n)
  p = LimitLengthProxy(p, n * 2)
  mapping = get_action_mapping_sorting(n, exclude_self=exclude_self)
  p = MapActionProxy(p, mapping)
  return p


def make(task, *args, **kwargs):
  if task == 'sort':
    return get_sort_env(*args, **kwargs)
  else:
    raise ValueError('Unknown task: {}.'.format(task))
