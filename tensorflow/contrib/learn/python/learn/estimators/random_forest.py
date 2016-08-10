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
"""A tf.learn implementation of tensor_forest (extremely random forests)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib.learn.python.learn import monitors as mon

from tensorflow.contrib.learn.python.learn.estimators import estimator

from tensorflow.contrib.tensor_forest.client import eval_metrics
from tensorflow.contrib.tensor_forest.data import data_ops
from tensorflow.contrib.tensor_forest.python import tensor_forest

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops


def _assert_float32(tensors):
  """Assert all tensors are float32.

  Args:
    tensors: `Tensor` or `dict` of `Tensor` objects.

  Raises:
    TypeError: if any tensor is not float32.
  """
  if not isinstance(tensors, dict):
    tensors = [tensors]
  else:
    tensors = tensors.values()
  for tensor in tensors:
    if tensor.dtype.base_dtype != dtypes.float32:
      raise TypeError('Expected dtype=float32, %s.' % tensor)


class LossMonitor(mon.EveryN):
  """Terminates training when training loss stops decreasing."""

  def __init__(self,
               early_stopping_rounds,
               every_n_steps):
    super(LossMonitor, self).__init__(every_n_steps=every_n_steps)
    self.early_stopping_rounds = early_stopping_rounds
    self.min_loss = None
    self.min_loss_step = 0

  def set_estimator(self, est):
    """This function gets called in the same graph as _get_train_ops."""
    super(LossMonitor, self).set_estimator(est)
    self._loss_op_name = est.training_loss.name

  def every_n_step_end(self, step, outputs):
    super(LossMonitor, self).every_n_step_end(step, outputs)
    current_loss = outputs[self._loss_op_name]
    if self.min_loss is None or current_loss < self.min_loss:
      self.min_loss = current_loss
      self.min_loss_step = step
    return step - self.min_loss_step >= self.early_stopping_rounds


class TensorForestEstimator(estimator.BaseEstimator):
  """An estimator that can train and evaluate a random forest."""

  def __init__(self, params, device_assigner=None, model_dir=None,
               graph_builder_class=tensor_forest.RandomForestGraphs,
               master='', accuracy_metric=None,
               tf_random_seed=None, config=None):
    self.params = params.fill()
    self.accuracy_metric = (accuracy_metric or
                            ('r2' if self.params.regression else 'accuracy'))
    self.data_feeder = None
    self.device_assigner = (
        device_assigner or tensor_forest.RandomForestDeviceAssigner())
    self.graph_builder_class = graph_builder_class
    self.training_args = {}
    self.construction_args = {}

    super(TensorForestEstimator, self).__init__(model_dir=model_dir,
                                                config=config)

  def predict_proba(
      self, x=None, input_fn=None, batch_size=None, as_iterable=False):
    """Returns prediction probabilities for given features (classification).

    Args:
      x: features.
      input_fn: Input function. If set, x and y must be None.
      batch_size: Override default batch size.
      as_iterable: If True, return an iterable which keeps yielding predictions
        for each example until inputs are exhausted. Note: The inputs must
        terminate if you want the iterable to terminate (e.g. be sure to pass
        num_epochs=1 if you are using something like read_batch_features).

    Returns:
      Numpy array of predicted probabilities (or an iterable of predicted
      probabilities if as_iterable is True).

    Raises:
      ValueError: If both or neither of x and input_fn were given.
    """
    return super(TensorForestEstimator, self).predict(
        x=x, input_fn=input_fn, batch_size=batch_size, as_iterable=as_iterable)

  def predict(
      self, x=None, input_fn=None, axis=None, batch_size=None,
      as_iterable=False):
    """Returns predictions for given features.

    Args:
      x: features.
      input_fn: Input function. If set, x must be None.
      axis: Axis on which to argmax (for classification).
            Last axis is used by default.
      batch_size: Override default batch size.
      as_iterable: If True, return an iterable which keeps yielding predictions
        for each example until inputs are exhausted. Note: The inputs must
        terminate if you want the iterable to terminate (e.g. be sure to pass
        num_epochs=1 if you are using something like read_batch_features).

    Returns:
      Numpy array of predicted classes or regression values (or an iterable of
      predictions if as_iterable is True).
    """
    probabilities = self.predict_proba(
        x=x, input_fn=input_fn, batch_size=batch_size, as_iterable=as_iterable)
    if self.params.regression:
      return probabilities
    else:
      if as_iterable:
        return (np.argmax(p, axis=0) for p in probabilities)
      else:
        return np.argmax(probabilities, axis=1)

  def _get_train_ops(self, features, targets):
    """Method that builds model graph and returns trainer ops.

    Args:
      features: `Tensor` or `dict` of `Tensor` objects.
      targets: `Tensor` or `dict` of `Tensor` objects.

    Returns:
      Tuple of train `Operation` and loss `Tensor`.
    """
    _assert_float32(features)
    _assert_float32(targets)
    features, spec = data_ops.ParseDataTensorOrDict(features)
    labels = data_ops.ParseLabelTensorOrDict(targets)

    graph_builder = self.graph_builder_class(
        self.params, device_assigner=self.device_assigner,
        **self.construction_args)

    epoch = None
    if self.data_feeder:
      epoch = self.data_feeder.make_epoch_variable()

    train = control_flow_ops.group(
        graph_builder.training_graph(
            features, labels, data_spec=spec, epoch=epoch,
            **self.training_args),
        state_ops.assign_add(contrib_framework.get_global_step(), 1))

    self.training_loss = graph_builder.training_loss(features, targets)

    return train, self.training_loss

  def _get_predict_ops(self, features):
    _assert_float32(features)
    graph_builder = self.graph_builder_class(
        self.params, device_assigner=self.device_assigner, training=False,
        **self.construction_args)
    features, spec = data_ops.ParseDataTensorOrDict(features)
    return graph_builder.inference_graph(features, data_spec=spec)

  def _get_eval_ops(self, features, targets, metrics):
    _assert_float32(features)
    _assert_float32(targets)
    features, spec = data_ops.ParseDataTensorOrDict(features)
    labels = data_ops.ParseLabelTensorOrDict(targets)

    graph_builder = self.graph_builder_class(
        self.params, device_assigner=self.device_assigner, training=False,
        **self.construction_args)

    probabilities = graph_builder.inference_graph(features, data_spec=spec)

    # One-hot the labels.
    if not self.params.regression:
      labels = math_ops.to_int64(array_ops.one_hot(math_ops.to_int64(
          array_ops.squeeze(labels)), self.params.num_classes, 1, 0))

    if metrics is None:
      metrics = {self.accuracy_metric:
                 eval_metrics.get_metric(self.accuracy_metric)}

    result = {}
    for name, metric in six.iteritems(metrics):
      result[name] = metric(probabilities, labels)

    return result
