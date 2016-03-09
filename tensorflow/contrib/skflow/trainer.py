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

from six.moves import xrange   # pylint: disable=redefined-builtin
import tensorflow as tf

OPTIMIZER_CLS_NAMES = {
    "SGD": tf.train.GradientDescentOptimizer,
    "Adagrad": tf.train.AdagradOptimizer,
    "Adam": tf.train.AdamOptimizer,
}


class TensorFlowTrainer(object):
    """General trainer class.

    Attributes:
      model: Model object.
      gradients: Gradients tensor.
    """

    def __init__(self, loss, global_step, optimizer,
                 learning_rate, clip_gradients=5.0):
        """Build a trainer part of graph.

        Args:
          loss: Tensor that evaluates to model's loss.
          global_step: Tensor with global step of the model.
          optimizer: Name of the optimizer class (SGD, Adam, Adagrad) or class.
          learning_rate: If this is constant float value, no decay function is used.
                         Instead, a customized decay function can be passed that accepts
                         global_step as parameter and returns a Tensor.
                         e.g. exponential decay function:
                         def exp_decay(global_step):
                            return tf.train.exponential_decay(
                                learning_rate=0.1, global_step=global_step,
                                decay_steps=2, decay_rate=0.001)
        Raises:
            ValueError: if learning_rate is not a float or a callable.
        """
        self.loss = loss
        self.global_step = global_step
        # pylint: disable=redefined-variable-type
        if isinstance(learning_rate, float):
            self._learning_rate = tf.get_variable(
                "learning_rate",
                [],
                initializer=tf.constant_initializer(learning_rate))
        elif callable(learning_rate):
            self._learning_rate = learning_rate(self.global_step)
        else:
            raise ValueError("learning_rate should be a float or a callable function.")
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
            self._optimizer = optimizer(self._learning_rate)
        self.trainer = self._optimizer.apply_gradients(grads_and_vars,
                                                       global_step=global_step,
                                                       name="train")
        # Update ops during training, e.g. batch_norm_ops
        self.trainer = tf.group(self.trainer, *tf.get_collection('update_ops'))
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

    def train(self, sess, feed_dict_fn, steps, monitor,
              summary_writer=None, summaries=None,
              feed_params_fn=None):
        """Trains a model for given number of steps, given feed_dict function.

        Args:
            sess: Session object.
            feed_dict_fn: Function that will return a feed dictionary.
            summary_writer: SummaryWriter object to use for writing summaries.
            steps: Number of steps to run.
            monitor: Monitor object to track training progress and induce early stopping
            summaries: Joined object of all summaries that should be ran.

        Returns:
            List of losses for each step.
        """
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
            monitor.update(step, global_step, loss, sess,
                           feed_params_fn, loss_expression_tensor=self.loss)
            if summaries and summary_writer and summ is not None:
                summary_writer.add_summary(summ, global_step)
            if monitor.monitor_inducing_stop():
                break
        return


class RestoredTrainer(TensorFlowTrainer):
    """Trainer class  that takes already existing graph."""

    # pylint: disable=super-init-not-called
    def __init__(self, loss, global_step, trainer):
        self.global_step = global_step
        self.loss = loss
        self.trainer = trainer
