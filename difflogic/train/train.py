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
"""Implement training process including curriculum and hard negative mining."""

import argparse
import collections
import copy

from ..tqdm_utils import tqdm_for

from difflogic.dataset.utils import RandomlyIterDataset
from difflogic.thutils import monitor_gradrms

import jacinle.random as random
from jacinle.logging import get_logger
from jacinle.utils.meter import GroupMeters
from jacinle.utils.tqdm import tqdm_pbar

from jactorch.train.env import TrainerEnv
from jactorch.utils.meta import as_cuda

__all__ = ['TrainerBase', 'CurriculumTrainerBase', 'MiningTrainerBase']

logger = get_logger(__file__)


class TrainerBase(TrainerEnv):
  """The base Trainer class supports basic training and testing interfaces.

  Implement the basic training and testing procedure. The training have multiple
  epochs, with multiple iterations in each epoch. A list defined by
  [begin:step:end] represents the argument (number of objects) for testing.

  Args:
    model: The model for both training and evaluation, the mode is turned by
      calling model.eval() or model.train().
    optimizer: The optimizer for the model when being optimized.
    epochs: The number of epochs for training.
    epoch_size: The number of iterations per epoch during training.
    test_epoch_size: The number of iterations per epoch during testing.
    test_number_begin: The begin number of the list.
    test_number_step: The step size of the list.
    test_number_end: The end number of the list.
  """

  def __init__(self, model, optimizer, epochs, epoch_size, test_epoch_size,
               test_number_begin, test_number_step, test_number_end):
    super().__init__(model, optimizer)

    self.epochs = epochs
    self.epoch_size = epoch_size
    self.test_epoch_size = test_epoch_size
    self.test_number_begin = test_number_begin
    self.test_number_step = test_number_step
    self.test_number_end = test_number_end

  __hyperparams__ = ('epochs', 'epoch_size', 'test_epoch_size',
                     'test_number_begin', 'test_number_step', 'test_number_end')

  __hyperparam_defaults__ = {'test_number_step': 0}

  @classmethod
  def _get_hyperparams(cls):
    return TrainerBase.__hyperparams__

  @classmethod
  def make_trainer_parser(cls, parser, defaults, prefix=None):
    for k, v in TrainerBase.__hyperparam_defaults__.items():
      defaults.setdefault(k, v)

    prefix = '--' if prefix is None else '--' + str(prefix) + '-'

    if not isinstance(parser, argparse._ArgumentGroup):
      parser = parser.add_argument_group('Trainer')
    parser.add_argument(
        prefix + 'epochs',
        type=int,
        default=defaults['epochs'],
        metavar='N',
        help='number of total epochs to run')
    parser.add_argument(
        prefix + 'epoch-size',
        type=int,
        default=defaults['epoch_size'],
        metavar='N',
        help='number of iterations per epoch')
    parser.add_argument(
        prefix + 'test-epoch-size',
        type=int,
        default=defaults['test_epoch_size'],
        metavar='N',
        help='number of iterations per test epoch')
    parser.add_argument(
        prefix + 'test-number-begin',
        type=int,
        default=defaults['test_number_begin'],
        metavar='N',
        help='begin number of nodes for test')
    parser.add_argument(
        prefix + 'test-number-step',
        type=int,
        default=defaults['test_number_step'],
        metavar='N',
        help='step number of nodes for test')
    parser.add_argument(
        prefix + 'test-number-end',
        type=int,
        default=defaults['test_number_end'],
        metavar='N',
        help='end number of nodes for test')

  @classmethod
  def from_args(cls, model, optimizer, args, prefix=None, **kwargs):
    prefix = '' if prefix is None else str(prefix) + '_'

    init_params = {k: getattr(args, prefix + k) for k in cls._get_hyperparams()}
    init_params.update(kwargs)

    return cls(model, optimizer, **init_params)

  def _dump_meters(self, meters, mode):
    """Provide ways to dump the statistics (stored in meters)

        for plotting or analysing.
    """
    pass

  def _prepare_dataset(self, epoch_size, mode):
    """Prepare dataset for getting training/testing data.

    Args:
      epoch_size: The number of iterations in each epoch.
      mode: 'train' or 'test' for training or testing.

    Returns:
      None, this is a hook function before getting training/testing data.
    """
    raise NotImplementedError()

  def _get_train_data(self, index, meters):
    """Get training data can be directly fed into the train_step function.

    Args:
      index: The current iteration index in current epoch.
      meters: a stats collector to collect information.

    Returns:
      feed_dict, A dict can be directly fed into the train_step function.
    """
    raise NotImplementedError()

  def _get_result(self, index, meters, mode):
    """Include two steps, get testing data from dataset & evaluate the model.

    Args:
      index: The current iteration index in current epoch.
      meters: a stats collector to collect information.
      mode: 'train' or 'test' or others.

    Returns:
      message: The message to be shown on tqdm progress bar.
      extra_info: An extra variable to give extra information.
    """
    raise NotImplementedError()

  def _train_step(self, feed_dict, meters):
    ret = self.step(feed_dict)
    loss, monitors, output_dict, extras = ret
    meters.update(monitor_gradrms(self.model))
    meters.update(monitors)
    meters.update(loss=loss)
    return 'Train: loss={loss:.4f}'.format(loss=loss), ret

  def _train_epoch(self, epoch_size):
    model = self.model
    meters = GroupMeters()
    self._prepare_dataset(epoch_size, mode='train')

    def train_func(index):
      model.eval()
      feed_dict = self._get_train_data(index, meters)
      model.train()
      message, _ = self._train_step(feed_dict, meters)
      return message

    # For $epoch_size times, do train_func with tqdm progress bar.
    tqdm_for(epoch_size, train_func)
    logger.info(
        meters.format_simple(
            '> Train Epoch {:5d}: '.format(self.current_epoch),
            compressed=False))
    self._dump_meters(meters, 'train')
    return meters

  def _test_epoch(self, epoch_size):
    meters = GroupMeters()
    self._prepare_dataset(epoch_size, mode='test')

    def test_func(index):
      message, _ = self._get_result(index, meters, mode='test')
      return message

    tqdm_for(epoch_size, test_func)
    logger.info(meters.format_simple('> Evaluation: ', compressed=False))
    self._dump_meters(meters, 'test')
    return meters

  def _early_stop(self, meters):
    """A hook function to enable early_stop checking."""
    return False

  def train(self):
    self.early_stopped = False
    for i in range(1, 1 + self.epochs):
      self.current_epoch = i
      meters = self._train_epoch(self.epoch_size)
      if self._early_stop(meters):
        self.early_stopped = True
        break
    return meters

  def test(self):
    self.model.eval()
    results = []

    self.test_number = self.test_number_begin
    while self.test_number <= self.test_number_end:
      meters = self._test_epoch(self.test_epoch_size)
      results.append(meters)
      if self.test_number_step <= 0:
        break
      self.test_number += self.test_number_step
    return results


class CurriculumTrainerBase(TrainerBase):
  """A base trainer class supports curriculum learning w.r.t an integer argument.

  The lessons in the curriculum are defined by an integer argument: The number
  of object. The lessons are defined by a list of the form [start:step:graduate].
  """

  def __init__(self, model, optimizer, epochs, epoch_size, test_epoch_size,
               test_number_begin, test_number_step, test_number_end,
               curriculum_start, curriculum_step, curriculum_graduate,
               enable_candidate, curriculum_thresh, curriculum_thresh_relax,
               curriculum_force_upgrade_epochs, sample_array_capacity,
               enhance_epochs):
    super().__init__(model, optimizer, epochs, epoch_size, test_epoch_size,
                     test_number_begin, test_number_step, test_number_end)

    self.curriculum_start = curriculum_start
    self.curriculum_step = curriculum_step
    self.curriculum_graduate = curriculum_graduate
    self.enable_candidate = enable_candidate
    self.curriculum_thresh = curriculum_thresh
    self.curriculum_thresh_relax = curriculum_thresh_relax
    self.curriculum_force_upgrade_epochs = curriculum_force_upgrade_epochs
    self.sample_array_capacity = sample_array_capacity
    self.enhance_epochs = enhance_epochs

  __hyperparams__ = ('curriculum_start', 'curriculum_step',
                     'curriculum_graduate', 'enable_candidate',
                     'curriculum_thresh', 'curriculum_thresh_relax',
                     'curriculum_force_upgrade_epochs', 'sample_array_capacity',
                     'enhance_epochs')

  __hyperparam_defaults__ = {
      'curriculum_step': 1,
      'enable_candidate': True,
      'curriculum_thresh': 1.0,
      'curriculum_thresh_relax': 0.0,
      'curriculum_force_upgrade_epochs': None,
      'sample_array_capacity': 1,
      'enhance_epochs': 0
  }

  @classmethod
  def _get_hyperparams(cls):
    return super()._get_hyperparams() + CurriculumTrainerBase.__hyperparams__

  @classmethod
  def make_trainer_parser(cls, parser, defaults, prefix=None):
    super().make_trainer_parser(parser, defaults, prefix)
    for k, v in CurriculumTrainerBase.__hyperparam_defaults__.items():
      defaults.setdefault(k, v)

    prefix = '--' if prefix is None else '--' + str(prefix) + '-'

    if not isinstance(parser, argparse._ArgumentGroup):
      parser = parser.add_argument_group('CurriculumTrainer')
    parser.add_argument(
        prefix + 'curriculum-start',
        type=int,
        default=defaults['curriculum_start'],
        metavar='N',
        help='starting number of nodes for curriculum')
    parser.add_argument(
        prefix + 'curriculum-step',
        type=int,
        default=defaults['curriculum_step'],
        metavar='N',
        help='number of nodes difference between lessons in curriculum')
    parser.add_argument(
        prefix + 'curriculum-graduate',
        type=int,
        default=defaults['curriculum_graduate'],
        metavar='N',
        help='graduate number of nodes for curriculum')
    parser.add_argument(
        prefix + 'enable-candidate',
        type='bool',
        default=defaults['enable_candidate'],
        metavar='B',
        help='enable candidate stage in curriculum')
    parser.add_argument(
        prefix + 'curriculum-thresh',
        type=float,
        default=defaults['curriculum_thresh'],
        metavar='F',
        help='threshold for curriculum lessons')
    parser.add_argument(
        prefix + 'curriculum-thresh-relax',
        type=float,
        default=defaults['curriculum_thresh_relax'],
        metavar='F',
        help='threshold = 1 - (graduate_number - current_number) * relax')
    parser.add_argument(
        prefix + 'curriculum-force-upgrade-epochs',
        type=int,
        default=defaults['curriculum_force_upgrade_epochs'],
        metavar='N',
        help='maximum number of epochs to force upgrade lesson')
    parser.add_argument(
        prefix + 'sample-array-capacity',
        type=int,
        default=defaults['sample_array_capacity'],
        metavar='N',
        help='the capacity of the sample array for numbers')
    parser.add_argument(
        prefix + 'enhance-epochs',
        type=int,
        default=defaults['enhance_epochs'],
        metavar='N',
        help='The number of training epochs even after graduation.')

  def _get_accuracy(self, meters):
    """return the statistics to be compared with the threshold."""
    raise NotImplementedError()

  def _get_threshold(self):
    return self.curriculum_thresh - self.curriculum_thresh_relax * \
        (self.curriculum_graduate - self.current_number)

  def _pass_lesson(self, meters):
    """Check whether the current performance is enough to pass current lesson."""
    acc = self._get_accuracy(meters)
    thresh = self._get_threshold()
    if acc >= thresh:
      return True
    # Force upgrade to next lesson if used too much epochs.
    t = self.curriculum_force_upgrade_epochs
    if t is not None and self.current_epoch - self.last_upgrade_epoch >= t:
      return True
    return False

  def _upgrade_lesson(self):
    """Upgrade to next lesson."""
    self.nr_upgrades += 1
    self.last_upgrade_epoch = self.current_epoch
    if self.enable_candidate:
      # When all lessons finished, it becomes candidate before graduated.
      if self.current_number < self.curriculum_graduate:
        self.current_number += self.curriculum_step
        # sample_array records the lessons the model recently studied.
        self.sample_array.append(self.current_number)
      elif self.is_candidate:
        self.is_graduated = True
      else:
        self.is_candidate = True
    else:
      if self.current_number < self.curriculum_graduate:
        self.current_number += self.curriculum_step
        self.sample_array.append(self.current_number)
      else:
        self.is_graduated = True

  def _take_exam(self, train_meters=None):
    """Use training results as exam result, upgrade to next lesson if pass."""
    if self._pass_lesson(train_meters):
      self._upgrade_lesson()

  def _train_epoch(self, epoch_size):
    """Add an exam session after each training epoch."""
    meters = super()._train_epoch(epoch_size)
    if not self.is_graduated:
      self._take_exam(train_meters=copy.copy(meters))
    return meters

  def _early_stop(self, meters):
    """Early stop the training when the model graduated from the curriculum."""
    return self.is_graduated and \
        self.current_epoch - self.last_upgrade_epoch >= self.enhance_epochs

  def _sample_number(self, mode):
    """Sample an integer argument from choices defined by an array."""
    if mode == 'test':
      return self.test_number
    # review (sample training data) from recently studied lessons.
    return random.choice(self.sample_array)

  def train(self):
    self.is_candidate = False
    self.is_graduated = False
    self.nr_upgrades = 0
    self.last_upgrade_epoch = 0
    self.current_number = self.curriculum_start
    self.sample_array = collections.deque(maxlen=self.sample_array_capacity)
    self.sample_array.append(self.current_number)
    super().train()
    return self.is_graduated


class MiningTrainerBase(CurriculumTrainerBase):
  """A trainer class supports both curriculum learning and hard negative mining.

  Targeted on RL cases (with environment provided). Maintain two list of data
  represents positive and negative ones. The environment instance is regarded as
  positive if the agent can successfully accomplish the task.

  Based on the curriculum schedule, there are periodically mining process (also
  used as exams to determine the upgrade to next lesson or not). During the
  mining process, random environment instances are being sampled, and collected
  into positive and negative ones according to the outcome. Dur training, the
  data are being balanced sampled from the positive and negative examples.
  """

  pos_data = None
  neg_data = None

  def __init__(self, model, optimizer, epochs, epoch_size, test_epoch_size,
               test_number_begin, test_number_step, test_number_end,
               curriculum_start, curriculum_step, curriculum_graduate,
               enable_candidate, curriculum_thresh, curriculum_thresh_relax,
               curriculum_force_upgrade_epochs, sample_array_capacity,
               enhance_epochs, enable_mining, repeat_mining, candidate_mul,
               mining_interval, mining_epoch_size, mining_dataset_size,
               inherit_neg_data, disable_balanced_sample, prob_pos_data):
    super().__init__(model, optimizer, epochs, epoch_size, test_epoch_size,
                     test_number_begin, test_number_step, test_number_end,
                     curriculum_start, curriculum_step, curriculum_graduate,
                     enable_candidate, curriculum_thresh,
                     curriculum_thresh_relax, curriculum_force_upgrade_epochs,
                     sample_array_capacity, enhance_epochs)

    self.enable_mining = enable_mining
    self.repeat_mining = repeat_mining
    self.candidate_mul = candidate_mul
    self.mining_interval = mining_interval
    self.mining_epoch_size = mining_epoch_size
    self.mining_dataset_size = mining_dataset_size
    self.inherit_neg_data = inherit_neg_data
    self.disable_balanced_sample = disable_balanced_sample
    self.prob_pos_data = prob_pos_data

  __hyperparams__ = ('enable_mining', 'repeat_mining', 'candidate_mul',
                     'mining_interval', 'mining_epoch_size',
                     'mining_dataset_size', 'inherit_neg_data',
                     'disable_balanced_sample', 'prob_pos_data')

  __hyperparam_defaults__ = {
      'repeat_mining': True,
      'candidate_mul': 2,
      'inherit_neg_data': False,
      'disable_balanced_sample': False,
      'prob_pos_data': 0.5
  }

  @classmethod
  def _get_hyperparams(cls):
    return super()._get_hyperparams() + MiningTrainerBase.__hyperparams__

  @classmethod
  def make_trainer_parser(cls, parser, defaults, prefix=None):
    super().make_trainer_parser(parser, defaults, prefix)
    for k, v in MiningTrainerBase.__hyperparam_defaults__.items():
      defaults.setdefault(k, v)

    prefix = '--' if prefix is None else '--' + str(prefix) + '-'

    if not isinstance(parser, argparse._ArgumentGroup):
      parser = parser.add_argument_group('MiningTrainer')
    parser.add_argument(
        prefix + 'enable-mining',
        type='bool',
        default=defaults['enable_mining'],
        metavar='B',
        help='enable hard-env mining')
    parser.add_argument(
        prefix + 'repeat-mining',
        type='bool',
        default=defaults['repeat_mining'],
        metavar='B',
        help='repeat mining until failing on a lesson')
    parser.add_argument(
        prefix + 'candidate-mul',
        type=int,
        default=defaults['candidate_mul'],
        metavar='N',
        help='x times more mining iters when being candidate')
    parser.add_argument(
        prefix + 'mining-interval',
        type=int,
        default=defaults['mining_interval'],
        metavar='N',
        help='the interval(number of epochs) of the mining')
    parser.add_argument(
        prefix + 'mining-epoch-size',
        type=int,
        default=defaults['mining_epoch_size'],
        metavar='N',
        help='number of iterations per epoch of mining')
    parser.add_argument(
        prefix + 'mining-dataset-size',
        type=int,
        default=defaults['mining_dataset_size'],
        metavar='N',
        help='size of the dataset collected during mining')
    parser.add_argument(
        prefix + 'inherit-neg-data',
        type='bool',
        default=defaults['inherit_neg_data'],
        metavar='B',
        help='recompute the negative data from last mining')
    parser.add_argument(
        prefix + 'disable-balanced-sample',
        type='bool',
        default=defaults['disable_balanced_sample'],
        metavar='B',
        help='use random samples instead of balanced samples when enable mining'
    )
    parser.add_argument(
        prefix + 'prob-pos-data',
        type=float,
        default=defaults['prob_pos_data'],
        metavar='F',
        help='the probability of use positive data during training')

  def _get_player(self, number, mode):
    """Get an environment to be interact with, with nr_obj & mode specified."""
    raise NotImplementedError()

  def _balanced_sample(self, meters):
    """Balanced sample positive and negative data with $prob_pos_data."""
    nr_pos, nr_neg = self.pos_data.size, self.neg_data.size
    assert nr_pos + nr_neg > 0
    if nr_neg == 0:
      use_pos_data = True
    elif nr_pos == 0:
      use_pos_data = False
    else:
      use_pos_data = random.rand() < self.prob_pos_data
    meters.update(pos_data_ratio=int(use_pos_data))
    pool = self.pos_data if use_pos_data else self.neg_data
    return pool.get()

  def _get_number_and_player(self, meters, mode):
    """Sample both the number of objects and the environment."""
    balanced_sample = mode == 'train' and self.enable_mining and (
        not self.disable_balanced_sample and self.pos_data is not None)
    if balanced_sample:
      number, player = self._balanced_sample(meters)
    else:
      number = self._sample_number(mode)
      player = self._get_player(number, mode)
    if mode == 'train':
      meters.update(train_number=number)
    return number, player

  def _get_result_given_player(self, index, meters, number, player, mode):
    """Compute the result given player, upon the mode.

    Args:
      index: Current episode id.
      meters: Used to collect stats.
      number: The number of objects/blocks.
      player: Environment for player to interact.
      mode: 'train'/'test'/'mining'/'inherit'

    Returns('train' mode):
      feed_dict: feed_dict for train_step
    Returns(other modes):
      message: The message shown on the progress bar.
      result: necessary extra information, see also _extract_info
    """
    raise NotImplementedError()

  def _get_result(self, index, meters, mode):
    number, player = self._get_number_and_player(meters, mode)
    return self._get_result_given_player(index, meters, number, player, mode)

  def _extract_info(self, extra):
    """Extract necessary information from extra variable.

    Args:
      extra: An extra variable returned by _get_result_given_player.

    Returns:
      succ: The result of the episode, success or not, to classify as pos/neg.
      number: The number of objects/blocks
      backup: The clone of the environment, for interacting multiple times.
    """
    raise NotImplementedError()

  def _get_train_data(self, index, meters):
    return self._get_result(index, meters, mode='train')

  def _inherit_neg_data(self, neg_data, old_neg_data, meters,
                        mining_dataset_size):
    """To avoid wasting already collect negative data, re-exam them."""
    if not self.inherit_neg_data or \
            (old_neg_data is None or old_neg_data.size == 0):
      return
    original_size = neg_data.size
    old_neg_data.reset()
    maximum_inherit_size = min(old_neg_data.size, mining_dataset_size)

    def inherit_func(index):
      number, player = old_neg_data.get()
      message, result = self._get_result_given_player(
          index, meters, number, player, mode='inherit')
      positive, number, backup = self._extract_info(result)
      if not positive:
        neg_data.append((number, backup))
      return message

    tqdm_for(maximum_inherit_size, inherit_func)
    logger.info(
        meters.format_simple(
            '> Inherit: new_size:{}, old_size:{}'.format(
                neg_data.size - original_size, old_neg_data.size),
            compressed=False))

  def _mining_epoch(self, mining_epoch_size, mining_dataset_size):
    """Take exam, collect and update positive dataset and negative dataset"""
    pos_data = RandomlyIterDataset()
    neg_data = RandomlyIterDataset()
    self.model.eval()
    meters = GroupMeters()
    with tqdm_pbar(total=mining_epoch_size) as pbar:
      for i in range(mining_epoch_size):
        message, result = self._get_result(i, meters, mode='mining')
        positive, number, backup = self._extract_info(result)
        dataset = pos_data if positive else neg_data
        if dataset.size < mining_dataset_size:
          dataset.append((number, backup))
        pbar.set_description(message)
        pbar.update()
        # When both positive and negative dataset are full, break.
        if pos_data.size >= mining_dataset_size and \
                neg_data.size >= mining_dataset_size:
          break
    logger.info(meters.format_simple('> Mining: ', compressed=False))
    self._inherit_neg_data(neg_data, self.neg_data, meters, mining_dataset_size)
    self.pos_data = pos_data
    self.neg_data = neg_data
    self._dump_meters(meters, 'mining')
    return meters

  def _upgrade_lesson(self):
    super()._upgrade_lesson()
    if self.is_graduated:
      self.pos_data, self.neg_data = None, None

  def _take_exam(self, train_meters=None):
    if not self.enable_mining:
      super()._take_exam(train_meters)
      return

    # The mining process, as well as the examing time,
    # only taken at a certain rate.
    if self.need_mining or (self.mining_interval <=
                            self.current_epoch - self.last_mining_epoch):
      self.last_mining_epoch = self.current_epoch
      mining_epoch_size = self.mining_epoch_size
      # The exam elapses longer when in candidate status.
      if self.is_candidate:
        mining_epoch_size *= self.candidate_mul
      meters = self._mining_epoch(mining_epoch_size, self.mining_dataset_size)

      # Use the performance during mining as the outcome for the exam.
      if self._pass_lesson(meters):
        self._upgrade_lesson()
        if self.is_graduated or (not self.repeat_mining and self.need_mining):
          self.need_mining = False
        else:
          # Can take exam consecutively if repeat_mining=True.
          self.need_mining = True
          self._take_exam()
      else:
        self.need_mining = False

  def train(self):
    self.need_mining = False
    self.last_mining_epoch = 0
    return super().train()
