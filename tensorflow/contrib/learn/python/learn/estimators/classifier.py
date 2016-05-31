# pylint: disable=g-bad-file-header
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

from tensorflow.contrib import metrics as metrics_lib
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn


def get_classifier_metrics(n_classes):
  return {('accuracy', 'classes'): metrics_lib.streaming_accuracy} 


class Classifier(estimator.Estimator):
  """Classifier single output Estimator.

  Given logits generating function, provides class / probabilities heads and
  functions to work with them.
  """

  CLASS_OUTPUT = 'classes'
  PROBABILITY_OUTPUT = 'probabilities'

  def __init__(self, model_fn, n_classes, model_dir=None, config=None):
    """
    
    Args:
      model_fn: (targets, predictions, mode) -> logits, loss, train_op
    """
    self._n_classes = n_classes
    self._logits_fn = model_fn
    super(Classifier, self).__init__(model_fn=self._classifier_model,
                                     model_dir=model_dir, config=None)

  def evaluate(self, x=None, y=None, input_fn=None, batch_size=None,
               steps=None, metrics=None):
    metrics = metrics or get_classifier_metrics(self._n_classes)
    return super(Classifier, self).evaluate(
        x=x, y=y, input_fn=input_fn, batch_size=batch_size, steps=steps, metrics=metrics)

  def predict(self, x=None, input_fn=None, batch_size=None):
    return super(Classifier, self).predict(
        x=x, input_fn=input_fn, batch_size=batch_size,
        outputs=[self.CLASS_OUTPUT])[self.CLASS_OUTPUT]

  def predict_proba(self, x=None, input_fn=None, batch_size=None):
    return super(Classifier, self).predict(
        x=x, input_fn=input_fn, batch_size=batch_size,
        outputs=[self.PROBABILITY_OUTPUT])[self.PROBABILITY_OUTPUT]

  def _classifier_model(self, features, targets, mode):
    logits, loss, train_op = self._logits_fn(features, targets, mode)
    return {
        'classes': math_ops.argmax(logits, len(logits.get_shape()) - 1), 
        'probabilities': nn.softmax(logits)
    }, loss, train_op

