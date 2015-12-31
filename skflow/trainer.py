"""Generic trainer for TensorFlow models."""
#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
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

from __future__ import division, print_function, absolute_import

import sys
import math
from six.moves import xrange   # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf


OPTIMIZER_CLS_NAMES = {
    "SGD": tf.train.GradientDescentOptimizer,
    "Adagrad": tf.train.AdagradOptimizer,
    "Adam": tf.train.AdamOptimizer,
}


def _print_report(print_loss_buffer, global_step):
    """Prints report for given losses and global step."""
    avg_loss = np.mean(print_loss_buffer)
    print("Step #{step}, avg. loss: {loss:.5f}".format(step=global_step,
                                                       loss=avg_loss))


class TensorFlowTrainer(object):
    """General trainer class.

    Attributes:
      model: Model object.
      gradients: Gradients tensor.
    """

    def __init__(self, loss, global_step, optimizer, learning_rate, clip_gradients=5.0):
        """Build a trainer part of graph.

        Args:
          loss: Tensor that evaluates to model's loss.
          global_step: Tensor with global step of the model.
          optimizer: Name of the optimizer class (SGD, Adam, Adagrad) or class.
        """
        self.loss = loss
        self.global_step = global_step
        self._learning_rate = tf.get_variable(
            "learning_rate",
            [],
            initializer=tf.constant_initializer(learning_rate))
        params = tf.trainable_variables()
        self.gradients = tf.gradients(loss, params)
        if clip_gradients > 0.0:
            self.gradients, self.gradients_norm = tf.clip_by_global_norm(
                self.gradients, clip_gradients)
        grads_and_vars = zip(self.gradients, params)
        if isinstance(optimizer, str):
            self._optimizer = OPTIMIZER_CLS_NAMES[
                optimizer](self._learning_rate)
        else:
            self._optimizer = optimizer(self.learning_rate)
        self.trainer = self._optimizer.apply_gradients(grads_and_vars,
                                                       global_step=global_step,
                                                       name="train")
        # Get all initializers for all trainable variables.
        self._initializers = tf.initialize_all_variables()

    def initialize(self, sess):
        """Initalizes all variables.

        Args:
            sess: Session object.

        Returns:
            Values of initializers.
        """
        return sess.run(self._initializers)

    def train(self, sess, feed_dict_fn, steps,
              summary_writer=None, summaries=None,
              print_steps=0, verbose=1, early_stopping_rounds=None):
        """Trains a model for given number of steps, given feed_dict function.

        Args:
            sess: Session object.
            feed_dict_fn: Function that will return a feed dictionary.
            steps: Number of steps to run.
            summary_writer: SummaryWriter object to use for writing summaries.
            summaries: Joined object of all summaries that should be ran.
            print_steps: Number of steps in between printing cost.
            verbose: Controls the verbosity. If set to 0, the algorithm is muted.
            early_stopping_rounds: Activates early stopping if this is not None.
                Loss needs to decrease at least every every <early_stopping_rounds>
                round(s) to continue training. (default: None)

        Returns:
            List of losses for each step.
        """
        losses, print_loss_buffer = [], []
        print_steps = (print_steps if print_steps else
                       math.ceil(float(steps) / 10))

        min_loss = float('inf')
        min_loss_i = 0
        if early_stopping_rounds is not None:
            sys.stderr.write("Performing early stopping. ")

        for step in xrange(steps):
            feed_dict = feed_dict_fn()
            if summaries:
                global_step, loss, summ, _ = sess.run(
                    [self.global_step, self.loss, summaries, self.trainer],
                    feed_dict=feed_dict)
            else:
                global_step, loss, _ = sess.run(
                    [self.global_step, self.loss, self.trainer],
                    feed_dict=feed_dict)

            if early_stopping_rounds is not None:
                if loss < min_loss:
                    min_loss = loss
                    min_loss_i = step
                elif step - min_loss_i >= early_stopping_rounds:
                    sys.stderr.write("Stopping. Best step:\n \
                                     {} with loss {}\n".format(min_loss_i,
                                                               min_loss))
                    break

            losses.append(loss)
            print_loss_buffer.append(loss)
            if summaries and summary_writer:
                summary_writer.add_summary(summ, global_step)
            if verbose > 0:
                if step % print_steps == 0:
                    _print_report(print_loss_buffer, global_step)
                    print_loss_buffer = []
        return losses
