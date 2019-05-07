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
"""The environment class for graph tasks."""

import numpy as np

import jacinle.random as random
from jacinle.utils.meta import notnone_property
from jaclearn.rl.env import SimpleRLEnvBase

from .graph import get_random_graph_generator

__all__ = ['GraphEnvBase', 'PathGraphEnv']


class GraphEnvBase(SimpleRLEnvBase):
  """The base class for Graph Environment.

  Args:
    nr_nodes: The number of nodes in the graph.
    pmin: The lower bound of the parameter controlling the graph generation.
    pmax: The upper bound of the parameter controlling the graph generation,
        the same as $pmin in default.
    directed: Generator directed graph if directed=True.
    gen_method: Controlling the graph generation method.
        If gen_method='dnc', use the similar way as in DNC paper.
        Else using Erdos-Renyi algorithm (each edge exists with prob).
  """

  def __init__(self,
               nr_nodes,
               pmin,
               pmax=None,
               directed=False,
               gen_method='dnc'):
    super().__init__()
    self._nr_nodes = nr_nodes
    self._pmin = pmin
    self._pmax = pmin if pmax is None else pmax
    self._directed = directed
    self._gen_method = gen_method
    self._graph = None

  @notnone_property
  def graph(self):
    return self._graph

  @property
  def nr_nodes(self):
    return self._nr_nodes

  def _restart(self):
    """Restart the environment."""
    self._gen_graph()

  def _gen_graph(self):
    """generate the graph by specified method."""
    n = self._nr_nodes
    p = self._pmin + random.rand() * (self._pmax - self._pmin)
    assert self._gen_method in ['edge', 'dnc']
    gen = get_random_graph_generator(self._gen_method)
    self._graph = gen(n, p, self._directed)


class PathGraphEnv(GraphEnvBase):
  """Env for Finding a path from starting node to the destination."""

  def __init__(self,
               nr_nodes,
               dist_range,
               pmin,
               pmax=None,
               directed=False,
               gen_method='dnc'):
    super().__init__(nr_nodes, pmin, pmax, directed, gen_method)
    self._dist_range = dist_range

  @property
  def dist(self):
    return self._dist

  def _restart(self):
    super()._restart()
    self._dist = self._sample_dist()
    self._task = None
    while True:
      self._task = self._gen()
      if self._task is not None:
        break
      # Generate another graph if fail to find two nodes with desired distance.
      self._gen_graph()
    self._current = self._task[0]
    self._set_current_state(self._task)
    self._steps = 0

  def _sample_dist(self):
    """Sample the distance between the starting node and the destination."""
    lower, upper = self._dist_range
    upper = min(upper, self._nr_nodes - 1)
    return random.randint(upper - lower + 1) + lower

  def _gen(self):
    """Sample the starting node and the destination according to the distance."""
    dist_matrix = self._graph.get_shortest()
    st, ed = np.where(dist_matrix == self.dist)
    if len(st) == 0:
      return None
    ind = random.randint(len(st))
    return st[ind], ed[ind]

  def _action(self, target):
    """Move to the target node from current node if has_edge(current -> target)."""
    if self._current == self._task[1]:
      return 1, True
    if self._graph.has_edge(self._current, target):
      self._current = target
    self._set_current_state((self._current, self._task[1]))
    if self._current == self._task[1]:
      return 1, True
    self._steps += 1
    if self._steps >= self.dist:
      return 0, True
    return 0, False
