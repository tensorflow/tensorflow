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
"""TensorFlow composable models used as building blocks for estimators (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import re

import six

from tensorflow.contrib import layers
from tensorflow.contrib.framework import list_variables
from tensorflow.contrib.framework import load_variable
from tensorflow.contrib.layers.python.layers import feature_column_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import nn
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.summary import summary
from tensorflow.python.util.deprecation import deprecated


class _ComposableModel(object):
  """ABC for building blocks that can be used to create estimators.

  Subclasses need to implement the following methods:
    - build_model
    - _get_optimizer
  See below for the required signatures.
  _ComposableModel and its subclasses are not part of the public tf.learn API.
  """

  @deprecated(None, "Please use model_fns in tf.estimator.")
  def __init__(self,
               num_label_columns,
               optimizer,
               gradient_clip_norm,
               num_ps_replicas,
               scope,
               trainable=True):
    """Common initialization for all _ComposableModel objects.

    Args:
      num_label_columns: The number of label columns.
      optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the model. If `None`, will use a FTRL optimizer.
      gradient_clip_norm: A float > 0. If provided, gradients are clipped
        to their global norm with this clipping ratio. See
        tf.clip_by_global_norm for more details.
      num_ps_replicas: The number of parameter server replicas.
      scope: Scope for variables created in this model.
      trainable: True if this model contains variables that can be trained.
        False otherwise (in cases where the variables are used strictly for
        transforming input labels for training).
    """
    self._num_label_columns = num_label_columns
    self._optimizer = optimizer
    self._gradient_clip_norm = gradient_clip_norm
    self._num_ps_replicas = num_ps_replicas
    self._scope = scope
    self._trainable = trainable
    self._feature_columns = None

  def get_scope_name(self):
    """Returns the scope name used by this model for variables."""
    return self._scope

  def build_model(self, features, feature_columns, is_training):
    """Builds the model that can calculate the logits.

    Args:
      features: A mapping from feature columns to tensors.
      feature_columns: An iterable containing all the feature columns used
        by the model. All items in the set should be instances of
        classes derived from `FeatureColumn`.
      is_training: Set to True when training, False otherwise.

    Returns:
      The logits for this model.
    """
    raise NotImplementedError

  def get_train_step(self, loss):
    """Returns the ops to run to perform a training step on this estimator.

    Args:
      loss: The loss to use when calculating gradients.

    Returns:
      The ops to run to perform a training step.
    """
    my_vars = self._get_vars()
    if not (self._get_feature_columns() or my_vars):
      return []

    grads = gradients.gradients(loss, my_vars)
    if self._gradient_clip_norm:
      grads, _ = clip_ops.clip_by_global_norm(grads, self._gradient_clip_norm)
    return [self._get_optimizer().apply_gradients(zip(grads, my_vars))]

  def _get_feature_columns(self):
    if not self._feature_columns:
      return None
    feature_column_ops.check_feature_columns(self._feature_columns)
    return sorted(set(self._feature_columns), key=lambda x: x.key)

  def _get_vars(self):
    if self._get_feature_columns():
      return ops.get_collection(self._scope)
    return []

  def _get_optimizer(self):
    if (self._optimizer is None or isinstance(self._optimizer,
                                              six.string_types)):
      optimizer = self._get_default_optimizer(self._optimizer)
    elif callable(self._optimizer):
      optimizer = self._optimizer()
    else:
      optimizer = self._optimizer
    return optimizer

  def _get_default_optimizer(self, optimizer_name=None):
    raise NotImplementedError


class LinearComposableModel(_ComposableModel):
  """A _ComposableModel that implements linear regression.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Instances of this class can be used to build estimators through the use
  of composition.
  """

  def __init__(self,
               num_label_columns,
               optimizer=None,
               _joint_weights=False,
               gradient_clip_norm=None,
               num_ps_replicas=0,
               scope=None,
               trainable=True):
    """Initializes LinearComposableModel objects.

    Args:
      num_label_columns: The number of label columns.
      optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the model. If `None`, will use a FTRL optimizer.
      _joint_weights: If True use a single (possibly partitioned) variable
        to store all weights in this model. Faster, but requires that all
        feature columns are sparse and have the 'sum' combiner.
      gradient_clip_norm: A float > 0. If provided, gradients are clipped
        to their global norm with this clipping ratio. See
        tf.clip_by_global_norm for more details.
      num_ps_replicas: The number of parameter server replicas.
      scope: Optional scope for variables created in this model. If scope
        is not supplied, it will default to 'linear'.
      trainable: True if this model contains variables that can be trained.
        False otherwise (in cases where the variables are used strictly for
        transforming input labels for training).
    """
    scope = "linear" if not scope else scope
    super(LinearComposableModel, self).__init__(
        num_label_columns=num_label_columns,
        optimizer=optimizer,
        gradient_clip_norm=gradient_clip_norm,
        num_ps_replicas=num_ps_replicas,
        scope=scope,
        trainable=trainable)
    self._joint_weights = _joint_weights

  def get_weights(self, model_dir):
    """Returns weights per feature of the linear part.

    Args:
      model_dir: Directory where model parameters, graph and etc. are saved.

    Returns:
      The weights created by this model (without the optimizer weights).
    """
    all_variables = [name for name, _ in list_variables(model_dir)]
    values = {}
    optimizer_regex = r".*/" + self._get_optimizer().get_name() + r"(_\d)?$"
    for name in all_variables:
      if (name.startswith(self._scope + "/") and
          name != self._scope + "/bias_weight" and
          not re.match(optimizer_regex, name)):
        values[name] = load_variable(model_dir, name)
    if len(values) == 1:
      return values[list(values.keys())[0]]
    return values

  def get_bias(self, model_dir):
    """Returns bias of the model.

    Args:
      model_dir: Directory where model parameters, graph and etc. are saved.

    Returns:
      The bias weights created by this model.
    """
    return load_variable(model_dir, name=(self._scope + "/bias_weight"))

  def build_model(self, features, feature_columns, is_training):
    """See base class."""
    self._feature_columns = feature_columns
    partitioner = partitioned_variables.min_max_variable_partitioner(
        max_partitions=self._num_ps_replicas, min_slice_size=64 << 20)
    with variable_scope.variable_scope(
        self._scope, values=features.values(),
        partitioner=partitioner) as scope:
      if self._joint_weights:
        logits, _, _ = layers.joint_weighted_sum_from_feature_columns(
            columns_to_tensors=features,
            feature_columns=self._get_feature_columns(),
            num_outputs=self._num_label_columns,
            weight_collections=[self._scope],
            trainable=self._trainable,
            scope=scope)
      else:
        logits, _, _ = layers.weighted_sum_from_feature_columns(
            columns_to_tensors=features,
            feature_columns=self._get_feature_columns(),
            num_outputs=self._num_label_columns,
            weight_collections=[self._scope],
            trainable=self._trainable,
            scope=scope)
    return logits

  def _get_default_optimizer(self, optimizer_name=None):
    if optimizer_name is None:
      optimizer_name = "Ftrl"
    default_learning_rate = 1. / math.sqrt(len(self._get_feature_columns()))
    default_learning_rate = min(0.2, default_learning_rate)
    return layers.OPTIMIZER_CLS_NAMES[optimizer_name](
        learning_rate=default_learning_rate)


class DNNComposableModel(_ComposableModel):
  """A _ComposableModel that implements a DNN.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Instances of this class can be used to build estimators through the use
  of composition.
  """

  def __init__(self,
               num_label_columns,
               hidden_units,
               optimizer=None,
               activation_fn=nn.relu,
               dropout=None,
               gradient_clip_norm=None,
               num_ps_replicas=0,
               scope=None,
               trainable=True):
    """Initializes DNNComposableModel objects.

    Args:
      num_label_columns: The number of label columns.
      hidden_units: List of hidden units per layer. All layers are fully
        connected.
      optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the model. If `None`, will use a FTRL optimizer.
      activation_fn: Activation function applied to each layer. If `None`,
        will use `tf.nn.relu`.
      dropout: When not None, the probability we will drop out
        a given coordinate.
      gradient_clip_norm: A float > 0. If provided, gradients are clipped
        to their global norm with this clipping ratio. See
        tf.clip_by_global_norm for more details.
      num_ps_replicas: The number of parameter server replicas.
      scope: Optional scope for variables created in this model. If not scope
        is supplied, one is generated.
      trainable: True if this model contains variables that can be trained.
        False otherwise (in cases where the variables are used strictly for
        transforming input labels for training).
    """
    scope = "dnn" if not scope else scope
    super(DNNComposableModel, self).__init__(
        num_label_columns=num_label_columns,
        optimizer=optimizer,
        gradient_clip_norm=gradient_clip_norm,
        num_ps_replicas=num_ps_replicas,
        scope=scope,
        trainable=trainable)
    self._hidden_units = hidden_units
    self._activation_fn = activation_fn
    self._dropout = dropout

  def get_weights(self, model_dir):
    """Returns the weights of the model.

    Args:
      model_dir: Directory where model parameters, graph and etc. are saved.

    Returns:
      The weights created by this model.
    """
    return [
        load_variable(
            model_dir, name=(self._scope + "/hiddenlayer_%d/weights" % i))
        for i, _ in enumerate(self._hidden_units)
    ] + [load_variable(
        model_dir, name=(self._scope + "/logits/weights"))]

  def get_bias(self, model_dir):
    """Returns the bias of the model.

    Args:
      model_dir: Directory where model parameters, graph and etc. are saved.

    Returns:
      The bias weights created by this model.
    """
    return [
        load_variable(
            model_dir, name=(self._scope + "/hiddenlayer_%d/biases" % i))
        for i, _ in enumerate(self._hidden_units)
    ] + [load_variable(
        model_dir, name=(self._scope + "/logits/biases"))]

  def _add_hidden_layer_summary(self, value, tag):
    # TODO(zakaria): Move this code to tf.learn and add test.
    summary.scalar("%s/fraction_of_zero_values" % tag, nn.zero_fraction(value))
    summary.histogram("%s/activation" % tag, value)

  def build_model(self, features, feature_columns, is_training):
    """See base class."""
    self._feature_columns = feature_columns

    input_layer_partitioner = (
        partitioned_variables.min_max_variable_partitioner(
            max_partitions=self._num_ps_replicas, min_slice_size=64 << 20))
    with variable_scope.variable_scope(
        self._scope + "/input_from_feature_columns",
        values=features.values(),
        partitioner=input_layer_partitioner) as scope:
      net = layers.input_from_feature_columns(
          features,
          self._get_feature_columns(),
          weight_collections=[self._scope],
          trainable=self._trainable,
          scope=scope)

    hidden_layer_partitioner = (
        partitioned_variables.min_max_variable_partitioner(
            max_partitions=self._num_ps_replicas))
    for layer_id, num_hidden_units in enumerate(self._hidden_units):
      with variable_scope.variable_scope(
          self._scope + "/hiddenlayer_%d" % layer_id,
          values=[net],
          partitioner=hidden_layer_partitioner) as scope:
        net = layers.fully_connected(
            net,
            num_hidden_units,
            activation_fn=self._activation_fn,
            variables_collections=[self._scope],
            trainable=self._trainable,
            scope=scope)
        if self._dropout is not None and is_training:
          net = layers.dropout(net, keep_prob=(1.0 - self._dropout))
      self._add_hidden_layer_summary(net, scope.name)

    with variable_scope.variable_scope(
        self._scope + "/logits",
        values=[net],
        partitioner=hidden_layer_partitioner) as scope:
      logits = layers.fully_connected(
          net,
          self._num_label_columns,
          activation_fn=None,
          variables_collections=[self._scope],
          trainable=self._trainable,
          scope=scope)
    self._add_hidden_layer_summary(logits, "logits")
    return logits

  def _get_default_optimizer(self, optimizer_name=None):
    if optimizer_name is None:
      optimizer_name = "Adagrad"
    return layers.OPTIMIZER_CLS_NAMES[optimizer_name](learning_rate=0.05)
