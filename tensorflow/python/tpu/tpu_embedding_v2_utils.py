# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Companion classes for mid level API for TPU Embeddings in TF2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import functools
import math
import six

from tensorflow.core.protobuf.tpu import optimization_parameters_pb2
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util.tf_export import tf_export


@six.add_metaclass(abc.ABCMeta)
class _Optimizer(object):
  """Base class for all optimizers, with common parameters."""

  def __init__(self, learning_rate, use_gradient_accumulation, clip_weight_min,
               clip_weight_max, weight_decay_factor,
               multiply_weight_decay_factor_by_learning_rate,
               slot_variable_creation_fn=None):
    self.learning_rate = learning_rate
    self.use_gradient_accumulation = use_gradient_accumulation
    self.clip_weight_min = clip_weight_min
    self.clip_weight_max = clip_weight_max
    self.weight_decay_factor = weight_decay_factor
    self.multiply_weight_decay_factor_by_learning_rate = (
        multiply_weight_decay_factor_by_learning_rate)

    if (slot_variable_creation_fn is not None and
        not callable(slot_variable_creation_fn)):
      raise ValueError("slot_variable_creation_fn must be either None or a "
                       "callable.")
    self.slot_variable_creation_fn = slot_variable_creation_fn

  @abc.abstractmethod
  def _slot_names(self):
    """Returns the name of all the slot variables.

    This does not include the 'parameters' variable and these names must match
    the names of the slots variables as used in the corresponding
    `tpu_ops.load_tpu_embedding_*` ops.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def _slot_initializers(self):
    """Returns initializers for slot variables.

    This returns a parallel list to self._slot_names().
    """
    raise NotImplementedError

  def _set_optimization_parameters(self, parameters):
    """Sets the optimizer fields in the OptimizationParameters."""
    if self.use_gradient_accumulation:
      parameters.gradient_accumulation_status = (
          optimization_parameters_pb2.GradientAccumulationStatus.ENABLED)
    else:
      parameters.gradient_accumulation_status = (
          optimization_parameters_pb2.GradientAccumulationStatus.DISABLED)

    if self.clip_weight_min is not None:
      parameters.clipping_limits.lower.value = self.clip_weight_min

    if self.clip_weight_max is not None:
      parameters.clipping_limits.upper.value = self.clip_weight_max

    if self.weight_decay_factor:
      parameters.weight_decay_factor = self.weight_decay_factor
      if self.multiply_weight_decay_factor_by_learning_rate:
        parameters.multiply_weight_decay_factor_by_learning_rate = True

  @abc.abstractmethod
  def _load(self):
    """Returns the load function for the optimizer."""
    raise NotImplementedError

  @abc.abstractmethod
  def _retrieve(self):
    """Returns the retrieve function for the optimizer."""
    raise NotImplementedError

  def _create_slots(self, table):
    """Creates slot variables for table.

    Uses shape of table to create parallel slot variables.

    Args:
      table: A Variable or equivalent.

    Returns:
      A dict of variables, keyed by self._slot_names().
    """
    if self.slot_variable_creation_fn is not None:
      return self.slot_variable_creation_fn(table, self._slot_names())
    else:
      slots = {}
      for slot, initializer in zip(self._slot_names(),
                                   self._slot_initializers()):
        slots[slot] = tf_variables.Variable(
            name=table.name + "/" + slot,
            initial_value=functools.partial(
                initializer, shape=table.shape, dtype=table.dtype),
            trainable=False)
      return slots


@tf_export("tpu.experimental.embedding.SGD")
class SGD(_Optimizer):
  """Optimization parameters for stochastic gradient descent for TPU embeddings.

  Pass this to `tf.tpu.experimental.embedding.TPUEmbedding` via the `optimizer`
  argument to set the global optimizer and its parameters:

  ```
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      ...
      optimizer=tf.tpu.experimental.embedding.SGD(0.1))
  ```

  This can also be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  optimizer parameter to set a table specific optimizer. This will override the
  optimizer and parameters for global embedding optimizer defined above:

  ```
  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      optimizer=tf.tpu.experimental.embedding.SGD(0.2))
  table_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)

  feature_config = (
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_one),
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_two))

  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.SGD(0.1))
  ```

  In the above example, the first feature will be looked up in a table that has
  a learning rate of 0.2 while the second feature will be looked up in a table
  that has a learning rate of 0.1.

  See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
  complete description of these parameters and their impacts on the optimizer
  algorithm.
  """

  def __init__(self,
               learning_rate=0.01,
               clip_weight_min=None,
               clip_weight_max=None,
               weight_decay_factor=None,
               multiply_weight_decay_factor_by_learning_rate=None):
    """Optimization parameters for stochastic gradient descent.

    Args:
      learning_rate: The learning rate. It should be a floating point value or a
        callable taking no arguments for a dynamic learning rate.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed. Weights are decayed by multiplying the weight
        by this factor each step.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
    """
    super(SGD, self).__init__(
        learning_rate, False, clip_weight_min, clip_weight_max,
        weight_decay_factor, multiply_weight_decay_factor_by_learning_rate)

  def _slot_names(self):
    return []

  def _slot_initializers(self):
    return []

  def _set_optimization_parameters(self, parameters):
    super(SGD, self)._set_optimization_parameters(parameters)
    parameters.stochastic_gradient_descent.SetInParent()

  def _load(self):
    return tpu_ops.load_tpu_embedding_stochastic_gradient_descent_parameters

  def _retrieve(self):
    return tpu_ops.retrieve_tpu_embedding_stochastic_gradient_descent_parameters


@tf_export("tpu.experimental.embedding.Adagrad")
class Adagrad(_Optimizer):
  """Optimization parameters for Adagrad with TPU embeddings.

  Pass this to `tf.tpu.experimental.embedding.TPUEmbedding` via the `optimizer`
  argument to set the global optimizer and its parameters:

  ```python
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      ...
      optimizer=tf.tpu.experimental.embedding.Adagrad(0.1))
  ```

  This can also be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  optimizer parameter to set a table specific optimizer. This will override the
  optimizer and parameters for global embedding optimizer defined above:

  ```python
  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      optimizer=tf.tpu.experimental.embedding.Adagrad(0.2))
  table_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)

  feature_config = (
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_one),
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_two))

  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.Adagrad(0.1))
  ```

  In the above example, the first feature will be looked up in a table that has
  a learning rate of 0.2 while the second feature will be looked up in a table
  that has a learning rate of 0.1.

  See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
  complete description of these parameters and their impacts on the optimizer
  algorithm.
  """

  def __init__(self,
               learning_rate=0.001,
               initial_accumulator_value=0.1,
               use_gradient_accumulation=True,
               clip_weight_min=None,
               clip_weight_max=None,
               weight_decay_factor=None,
               multiply_weight_decay_factor_by_learning_rate=None,
               slot_variable_creation_fn=None):
    """Optimization parameters for Adagrad.

    Args:
      learning_rate: The learning rate. It should be a floating point value or a
        callable taking no arguments for a dynamic learning rate.
      initial_accumulator_value: initial accumulator for Adagrad.
      use_gradient_accumulation: setting this to `False` makes embedding
        gradients calculation less accurate but faster.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      slot_variable_creation_fn: Defaults to `None`. If you wish do directly
        control the creation of the slot variables, set this to a callable
        taking two parameters, a variable and a list of slot names to create for
        it. This function should return a dict with the slot names as keys and
        the created variables as values. When set to None (the default), uses
        the built-in variable creation.
    """
    super(Adagrad, self).__init__(
        learning_rate, use_gradient_accumulation, clip_weight_min,
        clip_weight_max, weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate,
        slot_variable_creation_fn)
    if initial_accumulator_value <= 0:
      raise ValueError("Adagrad initial_accumulator_value must be positive")
    self.initial_accumulator_value = initial_accumulator_value

  def _slot_names(self):
    return ["accumulators"]

  def _slot_initializers(self):
    return [init_ops_v2.Constant(self.initial_accumulator_value)]

  def _set_optimization_parameters(self, parameters):
    super(Adagrad, self)._set_optimization_parameters(parameters)
    parameters.adagrad.SetInParent()

  def _load(self):
    return tpu_ops.load_tpu_embedding_adagrad_parameters

  def _retrieve(self):
    return tpu_ops.retrieve_tpu_embedding_adagrad_parameters


@tf_export("tpu.experimental.embedding.Adam")
class Adam(_Optimizer):
  """Optimization parameters for Adam with TPU embeddings.

  Pass this to `tf.tpu.experimental.embedding.TPUEmbedding` via the `optimizer`
  argument to set the global optimizer and its parameters:

  NOTE: By default this optimizer is lazy, i.e. it will not apply the gradient
  update of zero to rows that were not looked up. You can change this behavior
  by setting `lazy_adam` to `False`.

  ```python
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      ...
      optimizer=tf.tpu.experimental.embedding.Adam(0.1))
  ```

  This can also be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  optimizer parameter to set a table specific optimizer. This will override the
  optimizer and parameters for global embedding optimizer defined above:

  ```python
  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      optimizer=tf.tpu.experimental.embedding.Adam(0.2))
  table_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)

  feature_config = (
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_one),
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_two))

  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.Adam(0.1))
  ```

  In the above example, the first feature will be looked up in a table that has
  a learning rate of 0.2 while the second feature will be looked up in a table
  that has a learning rate of 0.1.

  See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
  complete description of these parameters and their impacts on the optimizer
  algorithm.
  """

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-07,
               lazy_adam=True,
               sum_inside_sqrt=True,
               use_gradient_accumulation=True,
               clip_weight_min=None,
               clip_weight_max=None,
               weight_decay_factor=None,
               multiply_weight_decay_factor_by_learning_rate=None,
               slot_variable_creation_fn=None):
    """Optimization parameters for Adam.

    See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
    complete description of these parameters and their impacts on the optimizer
    algorithm.

    Args:
      learning_rate: The learning rate. It should be a floating point value or a
        callable taking no arguments for a dynamic learning rate.
      beta_1: A float value.
        The exponential decay rate for the 1st moment estimates.
      beta_2: A float value.
        The exponential decay rate for the 2nd moment estimates.
      epsilon: A small constant for numerical stability.
      lazy_adam: Use lazy Adam instead of Adam. Lazy Adam trains faster.
      sum_inside_sqrt: When this is true, the Adam update formula is changed
        from `m / (sqrt(v) + epsilon)` to `m / sqrt(v + epsilon**2)`. This
        option improves the performance of TPU training and is not expected to
        harm model quality.
      use_gradient_accumulation: Setting this to `False` makes embedding
        gradients calculation less accurate but faster.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      slot_variable_creation_fn: a callable taking two parameters, a variable
        and a list of slot names to create for it. This function should return
        a dict with the slot names as keys and the created variables as values.
        When set to None (the default), uses the built-in variable creation.
    """
    super(Adam, self).__init__(
        learning_rate, use_gradient_accumulation, clip_weight_min,
        clip_weight_max, weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate,
        slot_variable_creation_fn)
    if beta_1 < 0. or beta_1 >= 1.:
      raise ValueError("beta1 must be in the range [0, 1), but received {}."
                       .format(beta_1))
    if beta_2 < 0. or beta_2 >= 1.:
      raise ValueError("beta2 must be in the range [0, 1), but received {}."
                       .format(beta_2))
    if epsilon <= 0.:
      raise ValueError("epsilon must be positive; got {}.".format(epsilon))
    if not use_gradient_accumulation and not lazy_adam:
      raise ValueError(
          "When disabling Lazy Adam, gradient accumulation must be used.")

    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.lazy_adam = lazy_adam
    self.sum_inside_sqrt = sum_inside_sqrt

  def _slot_names(self):
    return ["momenta", "velocities"]

  def _slot_initializers(self):
    return [init_ops_v2.Constant(), init_ops_v2.Constant()]

  def _set_optimization_parameters(self, parameters):
    super(Adam, self)._set_optimization_parameters(parameters)
    parameters.adam.beta1 = self.beta_1
    parameters.adam.beta2 = self.beta_2
    parameters.adam.epsilon = self.epsilon
    parameters.adam.use_non_lazy_adam = not self.lazy_adam
    parameters.adam.use_sum_inside_sqrt = self.sum_inside_sqrt

  def _load(self):
    return tpu_ops.load_tpu_embedding_adam_parameters

  def _retrieve(self):
    return tpu_ops.retrieve_tpu_embedding_adam_parameters


@tf_export("tpu.experimental.embedding.TableConfig")
class TableConfig(object):
  """Configuration data for one embedding table.

  This class holds the configuration data for a single embedding table. It is
  used as the `table` parameter of a
  `tf.tpu.experimental.embedding.FeatureConfig`. Multiple
  `tf.tpu.experimental.embedding.FeatureConfig` objects can use the same
  `tf.tpu.experimental.embedding.TableConfig` object. In this case a shared
  table will be created for those feature lookups.

  ```python
  table_config_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  table_config_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  feature_config = {
      'feature_one': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_two': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_three': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_two)}
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.Adam(0.1))
  ```

  The above configuration has 2 tables, and three features. The first two
  features will be looked up in the first table and the third feature will be
  looked up in the second table.

  """

  def __init__(self, vocabulary_size, dim, initializer, optimizer=None,
               combiner="mean", name=None):
    """Embedding table configuration.

    Args:
      vocabulary_size: Size of the table's vocabulary (number of rows).
      dim: The embedding dimension (width) of the table.
      initializer: A callable initializer taking one parameter, the shape of the
        variable that will be initialized. Will be called once per task, to
        initialize that task's shard of the embedding table. If not specified,
        defaults to `truncated_normal_initializer` with mean `0.0` and standard
        deviation `1/sqrt(dim)`.
      optimizer: An optional instance of an optimizer parameters class, instance
        of one of `tf.tpu.experimental.embedding.SGD`,
        `tf.tpu.experimental.embedding.Adagrad` or
        `tf.tpu.experimental.embedding.Adam`. It set will override the global
        optimizer passed to `tf.tpu.experimental.embedding.TPUEmbedding`.
      combiner: A string specifying how to reduce if there are multiple entries
        in a single row. Currently 'mean', 'sqrtn', 'sum' are
        supported, with 'mean' the default. 'sqrtn' often achieves good
        accuracy, in particular with bag-of-words columns. For more information,
        see `tf.nn.embedding_lookup_sparse`.
      name: An optional string used to name the table. Useful for debugging.

    Returns:
      `TableConfig`.

    Raises:
      ValueError: if `vocabulary_size` is not a positive integer.
      ValueError: if `dim` is not a positive integer.
      ValueError: if `initializer` is specified and is not callable.
      ValueError: if `combiner` is not supported.
    """
    if not isinstance(vocabulary_size, int) or vocabulary_size < 1:
      raise ValueError("Invalid vocabulary_size {}.".format(vocabulary_size))

    if not isinstance(dim, int) or dim < 1:
      raise ValueError("Invalid dim {}.".format(dim))

    if (initializer is not None) and (not callable(initializer)):
      raise ValueError("initializer must be callable if specified.")
    if initializer is None:
      initializer = init_ops_v2.TruncatedNormal(mean=0.0,
                                                stddev=1/math.sqrt(dim))

    if combiner not in ("mean", "sum", "sqrtn"):
      raise ValueError("Invalid combiner {}".format(combiner))

    self.vocabulary_size = vocabulary_size
    self.dim = dim
    self.initializer = initializer
    self.optimizer = optimizer
    self.combiner = combiner
    self.name = name


@tf_export("tpu.experimental.embedding.FeatureConfig")
class FeatureConfig(object):
  """Configuration data for one embedding feature.

  This class holds the configuration data for a single embedding feature. The
  main use is to assign features to `tf.tpu.experimental.embedding.TableConfig`s
  via the table parameter:

  ```python
  table_config_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  table_config_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  feature_config = {
      'feature_one': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_two': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_three': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_two)}
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.Adam(0.1))
  ```

  The above configuration has 2 tables, and three features. The first two
  features will be looked up in the first table and the third feature will be
  looked up in the second table.

  When feeding features into `embedding.enqueue` they can be `tf.Tensor`s,
  `tf.SparseTensor`s or `tf.RaggedTensor`s. When the argument
  `max_sequence_length` is 0, the default, you should expect a output of
  `embedding.dequeue` for this feature of shape `(batch_size, dim)`. If
  `max_sequence_length` is greater than 0, the feature is embedded as a sequence
  and padded up to the given length. The shape of the output for this feature
  will be `(batch_size, max_sequence_length, dim)`.
  """

  def __init__(self, table, max_sequence_length=0, name=None):
    """Feature configuration.

    Args:
      table: An instance of `tf.tpu.experimental.embedding.TableConfig`,
        describing the table in which this feature should be looked up.
      max_sequence_length: If positive, the feature is a sequence feature with
        the corresponding maximum sequence length. If the sequence is longer
        than this, it will be truncated. If 0, the feature is not a sequence
        feature.
      name: An optional name for the feature, useful for debugging.

    Returns:
      `FeatureConfig`.

    Raises:
      ValueError: if `table` is not an instance of
        `tf.tpu.experimental.embedding.TableConfig`.
      ValueError: if `max_sequence_length` not an integer or is negative.
    """
    if not isinstance(table, TableConfig):
      raise ValueError("table is type {}, expected "
                       "`tf.tpu.experimental.embedding.TableConfig`".format(
                           type(table)))

    if not isinstance(max_sequence_length, int) or max_sequence_length < 0:
      raise ValueError("Invalid max_sequence_length {}.".format(
          max_sequence_length))

    self.table = table
    self.max_sequence_length = max_sequence_length
    self.name = name
