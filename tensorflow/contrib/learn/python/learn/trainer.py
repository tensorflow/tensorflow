"""Generic trainer for TensorFlow models.

This module is deprecated, please use graph_actions.
"""
#  Copyright 2015-present The Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.platform import tf_logging as logging


def train(session,
          train_op,
          loss,
          global_step,
          feed_dict_fn,
          steps,
          monitor,
          summary_writer=None,
          summaries=None,
          feed_params_fn=None):
  """Trains a model for given number of steps, given feed_dict function.

  Args:
    session: Session object.
    train_op: Tensor, trains model.
    loss: Tensor, loss value.
    global_step: Tensor, global step of the model.
    feed_dict_fn: Function that will return a feed dictionary.
    steps: Number of steps to run.
    monitor: Monitor object to track training progress and induce early
      stopping
    summary_writer: SummaryWriter object to use for writing summaries.
    summaries: Joined object of all summaries that should be ran.
    feed_params_fn: Feed params function.
  """
  logging.warning("learn.trainer.train is deprecated. "
                  "Please use learn.graph_actions.train instead.")
  for step in xrange(steps):
    feed_dict = feed_dict_fn()
    if summaries is not None:
      global_step_value, loss_value, summ, _ = session.run(
          [global_step, loss, summaries, train_op],
          feed_dict=feed_dict)
    else:
      global_step_value, loss_value, _ = session.run(
          [global_step, loss, train_op],
          feed_dict=feed_dict)
    if summaries is not None and summary_writer and summ is not None:
      summary_writer.add_summary(summ, global_step_value)
