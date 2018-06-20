# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Baseline estimators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.estimator import estimator
from tensorflow.python.estimator.canned import baseline


class BaselineEstimator(estimator.Estimator):
  """An estimator that can establish a simple baseline.

  The estimator uses a user-specified head.

  This estimator ignores feature values and will learn to predict the average
  value of each label. E.g. for single-label classification problems, this will
  predict the probability distribution of the classes as seen in the labels.
  For multi-label classification problems, it will predict the ratio of examples
  that contain each class.

  Example:

  ```python

  # Build baseline multi-label classifier.
  estimator = BaselineEstimator(
      head=tf.contrib.estimator.multi_label_head(n_classes=3))

  # Input builders
  def input_fn_train: # returns x, y (where y represents label's class index).
    pass

  def input_fn_eval: # returns x, y (where y represents label's class index).
    pass

  # Fit model.
  estimator.train(input_fn=input_fn_train)

  # Evaluates cross entropy between the test and train labels.
  loss = classifier.evaluate(input_fn=input_fn_eval)["loss"]

  # For each class, predicts the ratio of training examples that contain the
  # class.
  predictions = classifier.predict(new_samples)

  ```

  Input of `train` and `evaluate` should have following features,
    otherwise there will be a `KeyError`:

  * if `weight_column` passed to the `head` constructor is not `None`, a feature
    with `key=weight_column` whose value is a `Tensor`.
  """

  def __init__(self,
               head,
               model_dir=None,
               optimizer='Ftrl',
               config=None):
    """Initializes a BaselineEstimator instance.

    Args:
      head: A `_Head` instance constructed with a method such as
        `tf.contrib.estimator.multi_label_head`.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      optimizer: String, `tf.Optimizer` object, or callable that creates the
        optimizer to use for training. If not specified, will use
        `FtrlOptimizer` with a default learning rate of 0.3.
      config: `RunConfig` object to configure the runtime settings.
    """
    def _model_fn(features, labels, mode, config):
      return baseline._baseline_model_fn(  # pylint: disable=protected-access
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          optimizer=optimizer,
          config=config)
    super(BaselineEstimator, self).__init__(
        model_fn=_model_fn,
        model_dir=model_dir,
        config=config)
