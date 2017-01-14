# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Classifier class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import metrics as metrics_lib
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn


def _get_classifier_metrics(unused_n_classes):
  return {
      ('accuracy', 'classes'): metrics_lib.streaming_accuracy
  }


class Classifier(estimator.Estimator):
  """Classifier single output Estimator.

  Given logits generating function, provides class / probabilities heads and
  functions to work with them.
  """

  CLASS_OUTPUT = 'classes'
  PROBABILITY_OUTPUT = 'probabilities'

  def __init__(self, model_fn, n_classes, model_dir=None, config=None):
    """Constructor for Classifier.

    Args:
      model_fn: (targets, predictions, mode) -> logits, loss, train_op
      n_classes: Number of classes
      model_dir: Directory to save model parameters, graph and etc. This can also
        be used to load checkpoints from the directory into a estimator to continue
        training a previously saved model.
      config: Configuration object (optional)
    """
    self._n_classes = n_classes
    self._logits_fn = model_fn
    super(Classifier, self).__init__(model_fn=self._classifier_model,
                                     model_dir=model_dir, config=config)

  def evaluate(self, x=None, y=None, input_fn=None, batch_size=None,
               steps=None, metrics=None):
    metrics = metrics or _get_classifier_metrics(self._n_classes)
    return super(Classifier, self).evaluate(x=x, y=y, input_fn=input_fn,
                                            batch_size=batch_size,
                                            steps=steps, metrics=metrics)

  def predict(self, x=None, input_fn=None, batch_size=None, as_iterable=False):
    """Returns predicted classes for given features.

    Args:
      x: Matrix of shape [n_samples, n_features...]. Can be iterator that
         returns arrays of features. The training input samples for fitting the
         model. If set, `input_fn` must be `None`.
      input_fn: Input function. If set, `x` and 'batch_size' must be `None`.
      batch_size: Override default batch size. If set, 'input_fn' must be
        'None'.
      as_iterable: If True, return an iterable which keeps yielding predictions
        for each example until inputs are exhausted. Note: The inputs must
        terminate if you want the iterable to terminate (e.g. be sure to pass
        num_epochs=1 if you are using something like read_batch_features).

    Returns:
      A numpy array of predicted classes (or an iterable of predicted classes if
      as_iterable is True).

    Raises:
      ValueError: If x and input_fn are both provided or both `None`.
    """
    predictions = super(Classifier, self).predict(
        x=x, input_fn=input_fn, batch_size=batch_size, as_iterable=as_iterable,
        outputs=[self.CLASS_OUTPUT])
    if as_iterable:
      return (p[self.CLASS_OUTPUT] for p in predictions)
    else:
      return predictions[self.CLASS_OUTPUT]

  def predict_proba(
      self, x=None, input_fn=None, batch_size=None, as_iterable=False):
    """Returns predicted probabilty distributions for given features.

    Args:
      x: Matrix of shape [n_samples, n_features...]. Can be iterator that
         returns arrays of features. The training input samples for fitting the
         model. If set, `input_fn` must be `None`.
      input_fn: Input function. If set, `x` and 'batch_size' must be `None`.
      batch_size: Override default batch size. If set, 'input_fn' must be
        'None'.
      as_iterable: If True, return an iterable which keeps yielding predictions
        for each example until inputs are exhausted. Note: The inputs must
        terminate if you want the iterable to terminate (e.g. be sure to pass
        num_epochs=1 if you are using something like read_batch_features).

    Returns:
      A numpy array of predicted probability distributions (or an iterable of
      predicted probability distributions if as_iterable is True).

    Raises:
      ValueError: If x and input_fn are both provided or both `None`.
    """
    predictions = super(Classifier, self).predict(
        x=x, input_fn=input_fn, batch_size=batch_size, as_iterable=as_iterable,
        outputs=[self.PROBABILITY_OUTPUT])
    if as_iterable:
      return (p[self.PROBABILITY_OUTPUT] for p in predictions)
    else:
      return predictions[self.PROBABILITY_OUTPUT]

  def _classifier_model(self, features, targets, mode):
    logits, loss, train_op = self._logits_fn(features, targets, mode)
    return {
        'classes': math_ops.argmax(logits, len(logits.get_shape()) - 1),
        'probabilities': nn.softmax(logits)
    }, loss, train_op

