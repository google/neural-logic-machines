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
"""Implement family tree generator and family tree class."""

import jacinle.random as random
import numpy as np

__all__ = ['Family', 'randomly_generate_family']


class Family(object):
  """Family tree class to support queries about relations between family members.

  Args:
    nr_people: The number of people in the family tree.
    relations: The relations between family members. The relations should be an
        matrix of shape [nr_people, nr_people, 6]. The relations in the order
        are: husband, wife, father, mother, son, daughter.
  """

  def __init__(self, nr_people, relations):
    self._n = nr_people
    self._relations = relations

  def mul(self, x, y):
    return np.clip(np.matmul(x, y), 0, 1)

  @property
  def nr_people(self):
    return self._n

  @property
  def relations(self):
    return self._relations

  @property
  def father(self):
    return self._relations[:, :, 2]

  @property
  def mother(self):
    return self._relations[:, :, 3]

  @property
  def son(self):
    return self._relations[:, :, 4]

  @property
  def daughter(self):
    return self._relations[:, :, 5]

  def has_father(self):
    return self.father.max(axis=1)

  def has_daughter(self):
    return self.daughter.max(axis=1)

  def has_sister(self):
    daughter_cnt = self.daughter.sum(axis=1)
    is_daughter = np.clip(self.daughter.sum(axis=0), 0, 1)
    return ((np.matmul(self.father, daughter_cnt) - is_daughter) >
            0).astype('float')
    # The wrong implementation: count herself as sister.
    # return self.mul(self.father, self.daughter).max(axis=1)

  def get_parents(self):
    return np.clip(self.father + self.mother, 0, 1)

  def get_grandfather(self):
    return self.mul(self.get_parents(), self.father)

  def get_grandmother(self):
    return self.mul(self.get_parents(), self.mother)

  def get_grandparents(self):
    parents = self.get_parents()
    return self.mul(parents, parents)

  def get_uncle(self):
    return np.clip(self.mul(self.get_grandparents(), self.son) - self.father, 0, 1)
    # The wrong Implementation: not exclude father.
    # return self.mul(self.get_grandparents(), self.son)

  def get_maternal_great_uncle(self):
    return self.mul(self.mul(self.get_grandmother(), self.mother), self.son)


def randomly_generate_family(n, p_marriage=0.8, verbose=False):
  """Randomly generate family trees.

  Mimic the process of families growing using a timeline. Each time a new person
  is created, randomly sample the gender and parents (could be none, indicating
  not included in the family tree) of the person. Also maintain lists of singles
  of each gender. With probability $p_marrige, randomly pick two from each list
  to be married. Finally randomly permute the order of people.

  Args:
    n: The number of people in the family tree.
    p_marriage: The probability of marriage happens each time.
    verbose: print the marriage and child born process if verbose=True.
  Returns:
    A family tree instance of $n people.
  """
  assert n > 0
  ids = list(random.permutation(n))

  single_m = []
  single_w = []
  couples = [None]
  # The relations are: husband, wife, father, mother, son, daughter
  rel = np.zeros((n, n, 6))
  fathers = [None for i in range(n)]
  mothers = [None for i in range(n)]

  def add_couple(man, woman):
    """Add a couple relation among (man, woman)."""
    couples.append((man, woman))
    rel[woman, man, 0] = 1  # husband
    rel[man, woman, 1] = 1  # wife
    if verbose:
      print('couple', man, woman)

  def add_child(parents, child, gender):
    """Add a child relation between parents and the child according to gender."""
    father, mother = parents
    fathers[child] = father
    mothers[child] = mother
    rel[child, father, 2] = 1  # father
    rel[child, mother, 3] = 1  # mother
    if gender == 0:  # son
      rel[father, child, 4] = 1
      rel[mother, child, 4] = 1
    else:  # daughter
      rel[father, child, 5] = 1
      rel[mother, child, 5] = 1
    if verbose:
      print('child', father, mother, child, gender)

  def check_relations(man, woman):
    """Disable marriage between cousins."""
    if fathers[man] is None or fathers[woman] is None:
      return True
    if fathers[man] == fathers[woman]:
      return False

    def same_parent(x, y):
      return fathers[x] is not None and fathers[y] is not None and fathers[
          x] == fathers[y]

    for x in [fathers[man], mothers[man]]:
      for y in [fathers[woman], mothers[woman]]:
        if same_parent(man, y) or same_parent(woman, x) or same_parent(x, y):
          return False
    return True

  while ids:
    x = ids.pop()
    gender = random.randint(2)
    parents = random.choice(couples)
    if gender == 0:
      single_m.append(x)
    else:
      single_w.append(x)
    if parents is not None:
      add_child(parents, x, gender)

    if random.rand() < p_marriage and len(single_m) > 0 and len(single_w) > 0:
      mi = random.randint(len(single_m))
      wi = random.randint(len(single_w))
      man = single_m[mi]
      woman = single_w[wi]
      if check_relations(man, woman):
        add_couple(man, woman)
        del single_m[mi]
        del single_w[wi]

  return Family(n, rel)

