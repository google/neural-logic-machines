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
"""Implement queries for different representations of blocksworld."""

import numpy as np

__all__ = [
    'get_world_string', 'get_coordinates', 'get_is_ground', 'get_moveable',
    'get_placeable', 'decorate'
]


def get_world_string(world):
  """Format the blocks world instance as string to view."""
  index_mapping = {b.index: i for i, b in enumerate(world.blocks)}
  raw_blocks = world.blocks.raw

  result = ''

  def dfs(block, indent):
    nonlocal result

    result += '{}Block #{}: (IsGround={}, Moveable={}, Placeable={})\n'.format(
        ' ' * (indent * 2), index_mapping[block.index], block.is_ground,
        block.moveable, block.placeable)
    for c in block.children:
      dfs(c, indent + 1)

  dfs(raw_blocks[0], 0)
  return result


def get_coordinates(world, absolute=False):
  """Get the coordinates of each block in the blocks world."""
  coordinates = [None for _ in range(world.size)]
  raw_blocks = world.blocks.raw

  def dfs(block):
    """Use depth-first-search to get the coordinate of each block."""
    if block.is_ground:
      coordinates[block.index] = (0, 0)
      for j, c in enumerate(block.children):
        # When using absolute coordinate, the block x directly placed on the
        # ground gets coordinate (x, 1).
        x = world.blocks.inv_index(c.index) if absolute else j
        coordinates[c.index] = (x, 1)
        dfs(c)
    else:
      coor = coordinates[block.index]
      assert coor is not None
      x, y = coor
      for c in block.children:
        coordinates[c.index] = (x, y + 1)
        dfs(c)

  dfs(raw_blocks[0])
  coordinates = world.blocks.permute(coordinates)
  return np.array(coordinates)


def get_is_ground(world):
  return np.array([block.is_ground for block in world.blocks])


def get_moveable(world):
  return np.array([block.moveable for block in world.blocks])


def get_placeable(world):
  return np.array([block.placeable for block in world.blocks])


def decorate(state, nr_objects, world_id=None):
  """Append world index and object index information to state."""
  info = []
  if world_id is not None:
    info.append(np.ones((nr_objects, 1)) * world_id)
  info.extend([np.array(range(nr_objects))[:, np.newaxis], state])
  return np.hstack(info)
