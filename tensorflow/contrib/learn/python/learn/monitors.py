"""Monitors to track model training, report on progress and request early stopping"""
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

import sys
import numpy as np

from tensorflow.contrib.learn.python.learn.io.data_feeder import setup_train_data_feeder


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=attribute-defined-outside-init


def default_monitor(verbose=1):
    """Returns very simple monitor object to summarize training progress.

    Args:
      verbose: Level of verbosity of output.

    Returns:
      Default monitor object.
    """
    return BaseMonitor(verbose=verbose)


class BaseMonitor(object):
    """Base class for all learning monitors. Stores and reports training loss throughout learning

    Parameters:
        print_steps: Number of steps in between printing cost.
        early_stopping_rounds:  Activates early stopping if this is not None.
                                Loss needs to decrease at least every every <early_stopping_rounds>
                                round(s) to continue training. (default: None)
        verbose: Level of verbosity of output.

    """
    def __init__(self, print_steps=100, early_stopping_rounds=None, verbose=1):
        self.print_steps = print_steps
        self.early_stopping_rounds = early_stopping_rounds

        self.converged = False
        self.min_loss = np.inf
        self.min_loss_i = 0
        self.last_loss_seen = np.inf
        self.steps = 0
        self.print_train_loss_buffer = []
        self.all_train_loss_buffer = []
        self.verbose = verbose
        self.epoch = None

    def update(self, global_step, step_number, training_loss,
               sess, feed_params_fn, loss_expression_tensor):
        """Adds training_loss to monitor. Triggers printed output if appropriate

            global_step:
            step_number: current step in training
            training_loss: float value of training loss
            sess: session for computation (used to calculate validation loss)
            feed_params_fn: function generating dict with information like epoch. Sometimes None.
            loss_expression_tensor: Tensor applied to validation data to calculate val loss

        """
        self.steps = step_number
        self.global_step = global_step
        self.print_train_loss_buffer.append(training_loss)
        self.all_train_loss_buffer.append(training_loss)
        self.sess = sess
        self.loss_expression_tensor = loss_expression_tensor
        self._set_last_loss_seen()
        if self.last_loss_seen < self.min_loss:
            self.min_loss = self.last_loss_seen
            self.min_loss_i = self.steps
        self._set_epoch(feed_params_fn)
        self.report()

    def _set_last_loss_seen(self):
        """Sets last_loss_seen attribute to most recent training error"""
        self.last_loss_seen = self.all_train_loss_buffer[-1]

    def report(self):
        """Checks whether to report, and prints loss information if appropriate"""
        if self.verbose and (self.steps % self.print_steps == 0):
            self._set_training_summary()
            print(self._summary_str)

    def monitor_inducing_stop(self):
        """Returns True if the monitor requests the model stop (e.g. for early stopping)"""
        if self.early_stopping_rounds is None:
            return False
        stop_now = (self.steps - self.min_loss_i >= self.early_stopping_rounds)
        if stop_now:
            sys.stderr.write("Stopping. Best step:\n step {} with loss {}\n"
                             .format(self.min_loss_i, self.min_loss))
        return stop_now

    def create_val_feed_dict(self, inp, out):
        """Validation requires access to TensorFlow placeholders. Not used in this Monitor"""
        pass

    def _set_epoch(self, feed_params_fn):
        """Sets self.epoch from a function that generates a dictionary including this info"""
        if feed_params_fn:
            feed_params = feed_params_fn()
            self.epoch = feed_params['epoch'] if 'epoch' in feed_params else None

    def _set_training_summary(self):
        """Returns the string to be written describing training progress"""
        avg_train_loss = np.mean(self.print_train_loss_buffer)
        self.print_train_loss_buffer = []
        if self.epoch:
            self._summary_str = ("Step #{step}, epoch #{epoch}, avg. train loss: {loss:.5f}"
                                 .format(step=self.steps, loss=avg_train_loss,
                                         epoch=self.epoch))
        else:
            self._summary_str = ("Step #{step}, avg. train loss: {loss:.5f}"
                                 .format(step=self.global_step,
                                         loss=avg_train_loss))
        self._modify_summary_string()

    def _modify_summary_string(self):
        """Makes monitor specific changes to printed summary. Nothing interesting in BaseMonitor"""
        pass


class ValidationMonitor(BaseMonitor):
    """Monitor that reports score for validation data and uses validation data for early stopping

        val_X: Validation features
        val_y: Validation labels
        n_classes: Number of labels in output. 0 for regression
        print_steps: Number of steps in between printing cost.
        early_stopping_rounds:  Activates early stopping if this is not None.
                                Loss needs to decrease at least every every <early_stopping_rounds>
                                round(s) to continue training. (default: None)

    """
    def __init__(self, val_X, val_y, n_classes=0, print_steps=100,
                 early_stopping_rounds=None):
        super(ValidationMonitor, self).__init__(print_steps=print_steps,
                                                early_stopping_rounds=early_stopping_rounds)
        self.val_feeder = setup_train_data_feeder(val_X, val_y, n_classes, -1)
        self.print_val_loss_buffer = []
        self.all_val_loss_buffer = []

    def create_val_feed_dict(self, inp, out):
        """Set tensorflow placeholders and create validation data feed"""
        self.val_feeder.set_placeholders(inp, out)
        self.val_dict = self.val_feeder.get_feed_dict_fn()()

    def _set_last_loss_seen(self):
        """Sets self.last_loss_seen to most recent validation loss

        Also stores this value to appropriate buffers
        """
        [val_loss] = self.sess.run([self.loss_expression_tensor], feed_dict=self.val_dict)
        self.last_loss_seen = val_loss
        self.all_val_loss_buffer.append(val_loss)
        self.print_val_loss_buffer.append(val_loss)

    def _modify_summary_string(self):
        """Adds validation data to string to print and resets validation printing buffer"""
        avg_val_loss = np.mean(self.print_val_loss_buffer)
        self.print_val_loss_buffer = []
        val_loss_string = "avg. val loss: {val_loss:.5f}".format(val_loss=avg_val_loss)
        self._summary_str = (", ".join([self._summary_str, val_loss_string]))
