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
"""Quick access for graph environments."""

from .graph_env import PathGraphEnv

__all__ = ['get_path_env', 'make']


def get_path_env(n, dist_range, pmin, pmax, directed=False, gen_method='dnc'):
  env_cls = PathGraphEnv
  p = env_cls(
      n, dist_range, pmin, pmax, directed=directed, gen_method=gen_method)
  return p


def make(task, *args, **kwargs):
  if task == 'path':
    return get_path_env(*args, **kwargs)
  else:
    raise ValueError('Unknown task: {}.'.format(task))
