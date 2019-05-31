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
"""Debug estimators (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.

Debug estimators are bias-only estimators that can be used for debugging
and as simple baselines.

Example:

```
# Build DebugClassifier
classifier = DebugClassifier()

# Input builders
def input_fn_train: # returns x, y (where y represents label's class index).
  pass

def input_fn_eval: # returns x, y (where y represents label's class index).
  pass

# Fit model.
classifier.fit(input_fn=input_fn_train)

# Evaluate cross entropy between the test and train labels.
loss = classifier.evaluate(input_fn=input_fn_eval)["loss"]

# predict_classes outputs the most commonly seen class in training.
predicted_label = classifier.predict_classes(new_samples)

# predict_proba outputs the class distribution from training.
label_distribution = classifier.predict_proba(new_samples)
```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.layers.python.layers import optimizers
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops


def _get_feature_dict(features):
  if isinstance(features, dict):
    return features
  return {"": features}


def debug_model_fn(features, labels, mode, params, config=None):
  """Model_fn for debug models.

  Args:
    features: `Tensor` or dict of `Tensor` (depends on data passed to `fit`).
    labels: Labels that are compatible with the `_Head` instance in `params`.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    params: A dict of hyperparameters containing:
      * head: A `_Head` instance.
    config: `RunConfig` object to configure the runtime settings.

  Raises:
    KeyError: If weight column is specified but not present.
    ValueError: If features is an empty dictionary.

  Returns:
    A `ModelFnOps` instance.
  """
  del config  # Unused.

  features = _get_feature_dict(features)
  if not features:
    raise ValueError("Features cannot be empty.")

  head = params["head"]
  size_checks = []
  batch_size = None

  # The first dimension is assumed to be a batch size and must be consistent
  # among all of the features.
  for feature in features.values():
    first_dim = array_ops.shape(feature)[0]
    if batch_size is None:
      batch_size = first_dim
    else:
      size_checks.append(check_ops.assert_equal(batch_size, first_dim))

  with ops.control_dependencies(size_checks):
    logits = array_ops.zeros([batch_size, head.logits_dimension])

  def train_op_fn(loss):
    return optimizers.optimize_loss(
        loss, global_step=None, learning_rate=0.3, optimizer="Adagrad")

  return head.create_model_fn_ops(
      features=features,
      labels=labels,
      mode=mode,
      train_op_fn=train_op_fn,
      logits=logits)


class DebugClassifier(estimator.Estimator):
  """A classifier for TensorFlow Debug models.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Example:

  ```python

  # Build DebugClassifier
  classifier = DebugClassifier()

  # Input builders
  def input_fn_train: # returns x, y (where y represents label's class index).
    pass

  def input_fn_eval: # returns x, y (where y represents label's class index).
    pass

  # Fit model.
  classifier.fit(input_fn=input_fn_train)

  # Evaluate cross entropy between the test and train labels.
  loss = classifier.evaluate(input_fn=input_fn_eval)["loss"]

  # predict_class outputs the most commonly seen class in training.
  predicted_label = classifier.predict_class(new_samples)

  # predict_proba outputs the class distribution from training.
  label_distribution = classifier.predict_proba(new_samples)
  ```

  Input of `fit` and `evaluate` should have following features,
    otherwise there will be a `KeyError`:

  * if `weight_column_name` is not `None`, a feature with
     `key=weight_column_name` whose value is a `Tensor`.
  """

  def __init__(self,
               model_dir=None,
               n_classes=2,
               weight_column_name=None,
               config=None,
               feature_engineering_fn=None,
               label_keys=None):
    """Initializes a DebugClassifier instance.

    Args:
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      n_classes: number of label classes. Default is binary classification.
        It must be greater than 1. Note: Class labels are integers representing
        the class index (i.e. values from 0 to n_classes-1). For arbitrary
        label values (e.g. string labels), convert to class indices first.
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      config: `RunConfig` object to configure the runtime settings.
      feature_engineering_fn: Feature engineering function. Takes features and
                        labels which are the output of `input_fn` and returns
                        features and labels which will be fed into the model.
      label_keys: Optional list of strings with size `[n_classes]` defining the
        label vocabulary. Only supported for `n_classes` > 2.
    Returns:
      A `DebugClassifier` estimator.

    Raises:
      ValueError: If `n_classes` < 2.
    """
    params = {"head":
              head_lib.multi_class_head(
                  n_classes=n_classes,
                  weight_column_name=weight_column_name,
                  enable_centered_bias=True,
                  label_keys=label_keys)}

    super(DebugClassifier, self).__init__(
        model_fn=debug_model_fn,
        model_dir=model_dir,
        config=config,
        params=params,
        feature_engineering_fn=feature_engineering_fn)

  def predict_classes(self, input_fn=None, batch_size=None):
    """Returns predicted classes for given features.

    Args:
      input_fn: Input function.
      batch_size: Override default batch size.

    Returns:
      An iterable of predicted classes. Each predicted class is represented by
      its class index (i.e. integer from 0 to n_classes-1).
    """
    key = prediction_key.PredictionKey.CLASSES
    preds = self.predict(
        input_fn=input_fn, batch_size=batch_size, outputs=[key])
    return (pred[key] for pred in preds)

  def predict_proba(self,
                    input_fn=None,
                    batch_size=None):
    """Returns prediction probabilities for given features.

    Args:
      input_fn: Input function.
      batch_size: Override default batch size.

    Returns:
      An iterable of predicted probabilities with shape [batch_size, n_classes].
    """
    key = prediction_key.PredictionKey.PROBABILITIES
    preds = self.predict(
        input_fn=input_fn,
        batch_size=batch_size,
        outputs=[key])
    return (pred[key] for pred in preds)


class DebugRegressor(estimator.Estimator):
  """A regressor for TensorFlow Debug models.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Example:

  ```python

  # Build DebugRegressor
  regressor = DebugRegressor()

  # Input builders
  def input_fn_train: # returns x, y (where y represents label's class index).
    pass

  def input_fn_eval: # returns x, y (where y represents label's class index).
    pass

  # Fit model.
  regressor.fit(input_fn=input_fn_train)

  # Evaluate squared-loss between the test and train targets.
  loss = regressor.evaluate(input_fn=input_fn_eval)["loss"]

  # predict_scores outputs mean value seen during training.
  predicted_targets = regressor.predict_scores(new_samples)
  ```

  Input of `fit` and `evaluate` should have following features,
    otherwise there will be a `KeyError`:

  * if `weight_column_name` is not `None`, a feature with
     `key=weight_column_name` whose value is a `Tensor`.
  """

  def __init__(self,
               model_dir=None,
               label_dimension=1,
               weight_column_name=None,
               config=None,
               feature_engineering_fn=None):
    """Initializes a DebugRegressor instance.

    Args:
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      label_dimension: Number of regression targets per example. This is the
        size of the last dimension of the labels and logits `Tensor` objects
        (typically, these have shape `[batch_size, label_dimension]`).
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      config: `RunConfig` object to configure the runtime settings.
      feature_engineering_fn: Feature engineering function. Takes features and
                        labels which are the output of `input_fn` and returns
                        features and labels which will be fed into the model.
    Returns:
      A `DebugRegressor` estimator.
    """

    params = {
        "head":
            head_lib.regression_head(
                weight_column_name=weight_column_name,
                label_dimension=label_dimension,
                enable_centered_bias=True)
    }

    super(DebugRegressor, self).__init__(
        model_fn=debug_model_fn,
        model_dir=model_dir,
        config=config,
        params=params,
        feature_engineering_fn=feature_engineering_fn)

  def predict_scores(self, input_fn=None, batch_size=None):
    """Returns predicted scores for given features.

    Args:
      input_fn: Input function.
      batch_size: Override default batch size.

    Returns:
      An iterable of predicted scores.
    """
    key = prediction_key.PredictionKey.SCORES
    preds = self.predict(
        input_fn=input_fn, batch_size=batch_size, outputs=[key])
    return (pred[key] for pred in preds)
