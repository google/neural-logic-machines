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
"""The utility functions for tqdm."""

from jacinle.utils.tqdm import tqdm_pbar

__all__ = ['tqdm_for']


def tqdm_for(total, func):
  """wrapper of the for function with message showing on the progress bar."""
  # Not support break cases for now.
  with tqdm_pbar(total=total) as pbar:
    for i in range(total):
      message = func(i)
      pbar.set_description(message)
      pbar.update()
