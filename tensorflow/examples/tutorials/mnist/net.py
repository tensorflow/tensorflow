# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds a network.

Implements the inference/loss/training/evaluation pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.
4. evaluation() Adds to the eval model the Ops required to eval trained results.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

import tensorflow as tf

class Net(object):
  __metaclass__ = ABCMeta

  def __init__(self, num_classes):
    # The default MNIST dataset has 10 classes, representing the digits 0 through 9.
    self._NUM_CLASSES = num_classes

    # The MNIST images are always 28x28 pixels.
    self._IMAGE_SIZE = 28
    self._IMAGE_PIXELS = self._IMAGE_SIZE * self._IMAGE_SIZE

    # Create a session for running Ops on the Graph.
    self._sess = tf.Session()

  @property
  def NUM_CLASSES(self):
    return self._NUM_CLASSES

  @property
  def IMAGE_SIZE(self):
    return self._IMAGE_SIZE

  @property
  def IMAGE_PIXELS(self):
    return self._IMAGE_PIXELS

  @property
  def sess(self):
    return self._sess

  @abstractmethod
  def inference(self, images):
    """Build the MNIST model up to where it may be used for inference.

    Args:
      images: Images placeholder, from inputs().

    Returns:
      logits: Output tensor with the computed logits.
    """
    pass

  @abstractmethod
  def loss(self, logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
      logits: Logits tensor.
      labels: Labels tensor.

    Returns:
      loss: Loss tensor of type float.
    """
    pass

  @abstractmethod
  def training(self, loss, learning_rate):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """
    pass

  @abstractmethod
  def evaluation(self, logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor.
      labels: Labels tensor.

    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    pass

  def init(self, images_placeholder, labels_placeholder, learning_rate):

    # Build a Graph that computes predictions from the inference model.
    logits = self.inference(images_placeholder)

    # Add to the Graph the Ops for loss calculation.
    self._loss_op = self.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    self._train_op = self.training(self._loss_op, learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    self._eval_op = self.evaluation(logits, labels_placeholder)

    # Run the Op to initialize the variables.
    init_val = tf.initialize_all_variables()
    self._sess.run(init_val)

  def run(self, ops, feed_dict):
    return self._sess.run(ops, feed_dict=feed_dict)

  def train(self, feed_dict):
    _, loss_value = self._sess.run([self._train_op, self._loss_op],
                                   feed_dict=feed_dict)
    return loss_value

  def eval(self, feed_dict):
    true_count = self._sess.run(self._eval_op,
                                feed_dict=feed_dict)
    return true_count

