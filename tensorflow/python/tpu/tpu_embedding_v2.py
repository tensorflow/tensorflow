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
"""Mid level API for TPU Embeddings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
from typing import Any, Dict, Callable, Iterable, List, Optional, Text, Tuple, Union

from absl import logging

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import save_context
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training.saving import saveable_hook
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import tracking
from tensorflow.python.types import core
from tensorflow.python.types import internal as internal_types
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


_HOOK_KEY = "TPUEmbedding_saveable"
_NAME_KEY = "_tpu_embedding_layer"


# TODO(bfontain): Cleanup and remove this once there is an implementation of
# sharded variables that can be used in the PSStrategy with optimizers.
# We implement just enough of the of a tf.Variable so that this could be passed
# to an optimizer.
class TPUShardedVariable(sharded_variable.ShardedVariableMixin):
  """A ShardedVariable class for TPU."""

  @property
  def _in_graph_mode(self):
    return self.variables[0]._in_graph_mode  # pylint: disable=protected-access

  @property
  def _unique_id(self):
    return self.variables[0]._unique_id  # pylint: disable=protected-access

  @property
  def _distribute_strategy(self):
    return self.variables[0]._distribute_strategy  # pylint: disable=protected-access

  @property
  def _shared_name(self):
    return self._name


def _add_key_attr(op, name):
  op._set_attr(_NAME_KEY, attr_value_pb2.AttrValue(s=compat.as_bytes(name)))  # pylint: disable=protected-access


@tf_export("tpu.experimental.embedding.TPUEmbedding")
class TPUEmbedding(tracking.AutoTrackable):
  """The TPUEmbedding mid level API.

  NOTE: When instantiated under a TPUStrategy, this class can only be created
  once per call to `tf.tpu.experimental.initialize_tpu_system`. If you wish to
  re-initialize the embedding engine you must re-initialize the tpu as well.
  Doing this will clear any variables from TPU, so ensure you have checkpointed
  before you do this. If a further instances of the class are needed,
  set the `initialize_tpu_embedding` argument to `False`.

  This class can be used to support training large embeddings on TPU. When
  creating an instance of this class, you must specify the complete set of
  tables and features you expect to lookup in those tables. See the
  documentation of `tf.tpu.experimental.embedding.TableConfig` and
  `tf.tpu.experimental.embedding.FeatureConfig` for more details on the complete
  set of options. We will cover the basic usage here.

  NOTE: multiple `FeatureConfig` objects can use the same `TableConfig` object,
  allowing different features to share the same table:

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
  ```

  There are two modes under which the `TPUEmbedding` class can used. This
  depends on if the class was created under a `TPUStrategy` scope or not.

  Under `TPUStrategy`, we allow access to the method `enqueue`, `dequeue` and
  `apply_gradients`. We will show examples below of how to use these to train
  and evaluate your model. Under CPU, we only access to the `embedding_tables`
  property which allow access to the embedding tables so that you can use them
  to run model evaluation/prediction on CPU.

  First lets look at the `TPUStrategy` mode. Initial setup looks like:

  ```python
  strategy = tf.distribute.TPUStrategy(...)
  with strategy.scope():
    embedding = tf.tpu.experimental.embedding.TPUEmbedding(
        feature_config=feature_config,
        optimizer=tf.tpu.experimental.embedding.SGD(0.1))
  ```

  When creating a distributed dataset that is to be passed to the enqueue
  operation a special input option must be specified:

  ```python
  distributed_dataset = (
      strategy.distribute_datasets_from_function(
          dataset_fn=...,
          options=tf.distribute.InputOptions(
              experimental_fetch_to_device=False))
  dataset_iterator = iter(distributed_dataset)
  ```

  NOTE: All batches passed to the layer must have the same batch size for each
  input, more over once you have called the layer with one batch size all
  subsequent calls must use the same batch_size. In the event that the batch
  size cannot be automatically determined by the enqueue method, you must call
  the build method with the batch size to initialize the layer.

  To use this API on TPU you should use a custom training loop. Below is an
  example of a training and evaluation step:

  ```python
  @tf.function
  def training_step(dataset_iterator, num_steps):
    def tpu_step(tpu_features):
      with tf.GradientTape() as tape:
        activations = embedding.dequeue()
        tape.watch(activations)
        model_output = model(activations)
        loss = ...  # some function of labels and model_output

      embedding_gradients = tape.gradient(loss, activations)
      embedding.apply_gradients(embedding_gradients)
      # Insert your model gradient and optimizer application here

    for _ in tf.range(num_steps):
      embedding_features, tpu_features = next(dataset_iterator)
      embedding.enqueue(embedding_features, training=True)
      strategy.run(tpu_step, args=(embedding_features, ))

  @tf.function
  def evalution_step(dataset_iterator, num_steps):
    def tpu_step(tpu_features):
      activations = embedding.dequeue()
      model_output = model(activations)
      # Insert your evaluation code here.

    for _ in tf.range(num_steps):
      embedding_features, tpu_features = next(dataset_iterator)
      embedding.enqueue(embedding_features, training=False)
      strategy.run(tpu_step, args=(embedding_features, ))
  ```

  NOTE: The calls to `enqueue` have `training` set to `True` when
  `embedding.apply_gradients` is used and set to `False` when
  `embedding.apply_gradients` is not present in the function. If you don't
  follow this pattern you may cause an error to be raised or the tpu may
  deadlock.

  In the above examples, we assume that the user has a dataset which returns
  a tuple where the first element of the tuple matches the structure of what
  was passed as the `feature_config` argument to the object initializer. Also we
  utilize `tf.range` to get a `tf.while_loop` in order to increase performance.

  When checkpointing your model, you should include your
  `tf.tpu.experimental.embedding.TPUEmbedding` object in the checkpoint. It is a
  trackable object and saving it will save the embedding tables and their
  optimizer slot variables:

  ```python
  checkpoint = tf.train.Checkpoint(model=model, embedding=embedding)
  checkpoint.save(...)
  ```

  On CPU, only the `embedding_table` property is usable. This will allow you to
  restore a checkpoint to the object and have access to the table variables:

  ```python
  model = model_fn(...)
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      optimizer=tf.tpu.experimental.embedding.SGD(0.1))
  checkpoint = tf.train.Checkpoint(model=model, embedding=embedding)
  checkpoint.restore(...)

  tables = embedding.embedding_tables
  ```

  You can now use table in functions like `tf.nn.embedding_lookup` to perform
  your embedding lookup and pass to your model.

  """

  def __init__(
      self,
      feature_config: Union[tpu_embedding_v2_utils.FeatureConfig, Iterable],  # pylint:disable=g-bare-generic
      optimizer: Optional[tpu_embedding_v2_utils._Optimizer],  # pylint:disable=protected-access
      pipeline_execution_with_tensor_core: bool = False):
    """Creates the TPUEmbedding mid level API object.

    ```python
    strategy = tf.distribute.TPUStrategy(...)
    with strategy.scope():
      embedding = tf.tpu.experimental.embedding.TPUEmbedding(
          feature_config=tf.tpu.experimental.embedding.FeatureConfig(
              table=tf.tpu.experimental.embedding.TableConfig(
                  dim=...,
                  vocabulary_size=...)))
    ```

    Args:
      feature_config: A nested structure of
        `tf.tpu.experimental.embedding.FeatureConfig` configs.
      optimizer: An instance of one of `tf.tpu.experimental.embedding.SGD`,
        `tf.tpu.experimental.embedding.Adagrad` or
        `tf.tpu.experimental.embedding.Adam`. When not created under
        TPUStrategy may be set to None to avoid the creation of the optimizer
        slot variables, useful for optimizing memory consumption when exporting
        the model for serving where slot variables aren't needed.
      pipeline_execution_with_tensor_core: If True, the TPU embedding
        computations will overlap with the TensorCore computations (and hence
        will be one step old). Set to True for improved performance.

    Raises:
      ValueError: If optimizer is not one of tf.tpu.experimental.embedding.(SGD,
      Adam or Adagrad) or None when created under a TPUStrategy.
    """
    self._strategy = distribution_strategy_context.get_strategy()
    self._using_tpu = isinstance(self._strategy, (tpu_strategy.TPUStrategy,
                                                  tpu_strategy.TPUStrategyV2))
    self._pipeline_execution_with_tensor_core = (
        pipeline_execution_with_tensor_core)

    self._feature_config = feature_config

    # The TPU embedding ops are slightly inconsistent with how they refer to
    # tables:
    # * The enqueue op takes a parallel list of tensors for input, one of those
    #   is the table id for the feature which matches the integer index of the
    #   table in the proto created by _create_config_proto().
    # * The recv_tpu_embedding_activations op emits lookups per table in the
    #   order from the config proto.
    # * The send_tpu_embedding_gradients expects input tensors to be per table
    #   in the same order as the config proto.
    # * Per optimizer load and retrieve ops are specified per table and take the
    #   table name rather than the table id.
    # Thus we must fix a common order to tables and ensure they have unique
    # names.

    # Set table order here to the order of the first occurence of the table in a
    # feature provided by the user. The order of this struct must be fixed
    # to provide the user with deterministic behavior over multiple
    # instantiations.
    self._table_config = []
    for feature in nest.flatten(feature_config):
      if feature.table not in self._table_config:
        self._table_config.append(feature.table)

    # Ensure tables have unique names. Also error check the optimizer as we
    # specifically don't do that in the TableConfig class to allow high level
    # APIs that are built on this to use strings/other classes to represent
    # optimizers (before they are passed to this class).
    table_names = []
    for i, table in enumerate(self._table_config):
      if table.optimizer is None:
        # TODO(bfontain) Should we allow some sort of optimizer merging here?
        table.optimizer = optimizer
      if ((table.optimizer is not None or self._using_tpu) and
          not isinstance(table.optimizer, tpu_embedding_v2_utils._Optimizer)):  # pylint: disable=protected-access
        raise ValueError("{} is an unsupported optimizer class. Please pass an "
                         "instance of one of the optimizer classes under "
                         "tf.tpu.experimental.embedding.".format(
                             type(table.optimizer)))
      if table.name is None:
        table.name = "table_{}".format(i)
      if table.name in table_names:
        raise ValueError("Multiple tables with name {} found.".format(
            table.name))
      table_names.append(table.name)

    if self._using_tpu:
      # Extract a list of callable learning rates also in fixed order. Each
      # table in the confix proto will get a index into this list and we will
      # pass this list in the same order after evaluation to the
      # send_tpu_embedding_gradients op.
      self._dynamic_learning_rates = list({
          table.optimizer.learning_rate for table in self._table_config if
          callable(table.optimizer.learning_rate)})

      # We need to list of host devices for the load/retrieve operations.
      self._hosts = get_list_of_hosts(self._strategy)

    self._built = False

  def build(self, per_replica_batch_size: Optional[int] = None):
    """Create the underlying variables and initializes the TPU for embeddings.

    This method creates the underlying variables (including slot variables). If
    created under a TPUStrategy, this will also initialize the TPU for
    embeddings.

    This function will automatically get called by enqueue, which will try to
    determine your batch size automatically. If this fails, you must manually
    call this method before you call enqueue.

    Args:
      per_replica_batch_size: The per replica batch size that you intend to use.
        Note that is fixed and the same batch size must be used for both
        training and evaluation. If you want to calculate this from the global
        batch size, you can use `num_replicas_in_sync` property of your strategy
        object. May be set to None if not created under a TPUStrategy.

    Raises:
      ValueError: If per_replica_batch_size is None and object was created in a
        TPUStrategy scope.
    """
    if self._built:
      return

    if self._using_tpu:
      if per_replica_batch_size is None:
        raise ValueError("You must specify a per_replica_batch_size when "
                         "calling build if object is created under a "
                         "TPUStrategy.")

      self._batch_size = per_replica_batch_size

      self._config_proto = self._create_config_proto()

      logging.info("Initializing TPU Embedding engine.")
      tpu_embedding_v2_utils.log_tpu_embedding_configuration(self._config_proto)

      @def_function.function
      def load_config():
        tpu.initialize_system_for_tpu_embedding(self._config_proto)

      load_config()
      logging.info("Done initializing TPU Embedding engine.")

    # Create and load variables and slot variables into the TPU.
    # Note that this is a dict of dicts. Keys to the first dict are table names.
    # We would prefer to use TableConfigs, but then these variables won't be
    # properly tracked by the tracking API.
    self._variables = self._create_variables_and_slots()

    self._built = True

    # This is internally conditioned self._built and self._using_tpu
    self._load_variables()

  def _maybe_build(self, batch_size: Optional[int]):
    if not self._built:
      # This can be called while tracing a function, so we wrap the
      # initialization code with init_scope so it runs eagerly, this means that
      # it will not be included the function graph generated by tracing so that
      # we can be sure that we only initialize the TPU for embeddings exactly
      # once.
      with ops.init_scope():
        self.build(batch_size)

  @property
  def embedding_tables(
      self
  ) -> Dict[tpu_embedding_v2_utils.TableConfig, tf_variables.Variable]:
    """Returns a dict of embedding tables, keyed by `TableConfig`.

    This property only works when the `TPUEmbedding` object is created under a
    non-TPU strategy. This is intended to be used to for CPU based lookup when
    creating a serving checkpoint.

    Returns:
      A dict of embedding tables, keyed by `TableConfig`.

    Raises:
      RuntimeError: If object was created under a `TPUStrategy`.
    """
    # We don't support returning tables on TPU due to their sharded nature and
    # the fact that when using a TPUStrategy:
    # 1. Variables are stale and are only updated when a checkpoint is made.
    # 2. Updating the variables won't affect the actual tables on the TPU.
    if self._using_tpu:
      if save_context.in_save_context():
        return {table: self._variables[table.name]["parameters"].variables[0]
                for table in self._table_config}
      raise RuntimeError("Unable to retrieve embedding tables when using a TPU "
                         "strategy. If you need access, save your model, "
                         "create this object under a CPU strategy and restore.")

    self._maybe_build(None)

    # Only return the tables and not the slot variables. On CPU this are honest
    # tf.Variables.
    return {table: self._variables[table.name]["parameters"]
            for table in self._table_config}

  def _create_config_proto(
      self
  ) -> tpu_embedding_configuration_pb2.TPUEmbeddingConfiguration:
    """Creates the TPUEmbeddingConfiguration proto.

    This proto is used to initialize the TPU embedding engine.

    Returns:
      A TPUEmbeddingConfiguration proto.
    """

    config_proto = tpu_embedding_configuration_pb2.TPUEmbeddingConfiguration()

    # There are several things that need to be computed here:
    # 1. Each table has a num_features, which corresponds to the number of
    #    output rows per example for this table. Sequence features count for
    #    their maximum sequence length.
    # 2. Learning rate index: the index of the dynamic learning rate for this
    #    table (if it exists) in the list we created at initialization.
    #    We don't simply create one learning rate index per table as this has
    #    extremely bad performance characteristics. The more separate
    #    optimization configurations we have, the worse the performance will be.
    num_features = {table: 0 for table in self._table_config}
    for feature in nest.flatten(self._feature_config):
      num_features[feature.table] += (1 if feature.max_sequence_length == 0
                                      else feature.max_sequence_length)

    # Map each callable dynamic learning rate to its in index in the list.
    learning_rate_index = {r: i for i, r in enumerate(
        self._dynamic_learning_rates)}

    for table in self._table_config:
      table_descriptor = config_proto.table_descriptor.add()
      table_descriptor.name = table.name

      # For small tables, we pad to the number of hosts so that at least one
      # id will be assigned to each host.
      table_descriptor.vocabulary_size = max(table.vocabulary_size,
                                             self._strategy.extended.num_hosts)
      table_descriptor.dimension = table.dim

      table_descriptor.num_features = num_features[table]

      parameters = table_descriptor.optimization_parameters

      # We handle the learning rate separately here and don't allow the
      # optimization class to handle this, as it doesn't know about dynamic
      # rates.
      if callable(table.optimizer.learning_rate):
        parameters.learning_rate.dynamic.tag = (
            learning_rate_index[table.optimizer.learning_rate])
      else:
        parameters.learning_rate.constant = table.optimizer.learning_rate

      # Use optimizer to handle the rest of the parameters.
      table.optimizer._set_optimization_parameters(parameters)  # pylint: disable=protected-access

    # Always set mode to training, we override the mode during enqueue.
    config_proto.mode = (
        tpu_embedding_configuration_pb2.TPUEmbeddingConfiguration.TRAINING)

    config_proto.batch_size_per_tensor_core = self._batch_size
    config_proto.num_hosts = self._strategy.extended.num_hosts
    config_proto.num_tensor_cores = self._strategy.num_replicas_in_sync

    # TODO(bfontain): Allow users to pick MOD for the host sharding.
    config_proto.sharding_strategy = (
        tpu_embedding_configuration_pb2.TPUEmbeddingConfiguration.DIV_DEFAULT)
    config_proto.pipeline_execution_with_tensor_core = (
        self._pipeline_execution_with_tensor_core)

    return config_proto

  def _compute_per_table_gradients(
      self,
      gradients
  ) -> Dict[Text, List[core.Tensor]]:
    """Computes a dict of lists of gradients, keyed by table name.

    Args:
      gradients: A nested structure of Tensors (and Nones) with the same
        structure as the feature config.

    Returns:
      A dict of lists of tensors, keyed by the table names, containing the
    gradients in the correct order with None gradients replaced by zeros.
    """

    nest.assert_same_structure(self._feature_config, gradients)

    per_table_gradients = {table: [] for table in self._table_config}
    for (path, gradient), feature in zip(
        nest.flatten_with_joined_string_paths(gradients),
        nest.flatten(self._feature_config)):
      if gradient is not None and not isinstance(gradient, ops.Tensor):
        raise ValueError(
            "Found {} at path {} in gradients. Expected Tensor.".format(
                type(gradient), path))

      # Expected tensor shape differs for sequence and non-sequence features.
      if feature.max_sequence_length > 0:
        shape = [self._batch_size, feature.max_sequence_length,
                 feature.table.dim]
      else:
        shape = [self._batch_size, feature.table.dim]

      if gradient is not None:
        if gradient.shape != shape:
          raise ValueError("Found gradient of shape {} at path {}. Expected "
                           "shape {}.".format(gradient.shape, path, shape))

        # We expand dims on non-sequence features so that all features are
        # of rank 3 and we can concat on axis=1.
        if len(shape) == 2:
          gradient = array_ops.expand_dims(gradient, axis=1)
      else:
        # No gradient for this feature, since we must give a gradient for all
        # features, pass in a zero tensor here. Note that this is not correct
        # for all optimizers.
        logging.warn("No gradient passed for feature %s, sending zero "
                     "gradient. This may not be correct behavior for certain "
                     "optimizers like Adam.", path)
        # Create a shape to mimic the expand_dims above for non-sequence
        # features.
        if len(shape) == 2:
          shape = [shape[0], 1, shape[1]]
        gradient = array_ops.zeros(shape, dtype=dtypes.float32)
      per_table_gradients[feature.table].append(gradient)

    return per_table_gradients

  def apply_gradients(self, gradients, name: Text = None):
    """Applies the gradient update to the embedding tables.

    If a gradient of `None` is passed in any position of the nested structure,
    then an gradient update with a zero gradient is applied for that feature.
    For optimizers like SGD or Adagrad, this is the same as applying no update
    at all. For lazy Adam and other sparsely applied optimizers with decay,
    ensure you understand the effect of applying a zero gradient.

    ```python
    strategy = tf.distribute.TPUStrategy(...)
    with strategy.scope():
      embedding = tf.tpu.experimental.embedding.TPUEmbedding(...)

    distributed_dataset = (
        strategy.distribute_datasets_from_function(
            dataset_fn=...,
            options=tf.distribute.InputOptions(
                experimental_fetch_to_device=False))
    dataset_iterator = iter(distributed_dataset)

    @tf.function
    def training_step():
      def tpu_step(tpu_features):
        with tf.GradientTape() as tape:
          activations = embedding.dequeue()
          tape.watch(activations)

          loss = ... #  some computation involving activations

        embedding_gradients = tape.gradient(loss, activations)
        embedding.apply_gradients(embedding_gradients)

      embedding_features, tpu_features = next(dataset_iterator)
      embedding.enqueue(embedding_features, training=True)
      strategy.run(tpu_step, args=(embedding_features, ))

    training_step()
    ```

    Args:
      gradients: A nested structure of gradients, with structure matching the
        `feature_config` passed to this object.
      name: A name for the underlying op.

    Raises:
      RuntimeError: If called when object wasn't created under a `TPUStrategy`
        or if not built (either by manually calling build or calling enqueue).
      ValueError: If a non-`tf.Tensor` non-`None` gradient is passed in, or a
        `tf.Tensor` of the incorrect shape is passed in. Also if
        the size of any sequence in `gradients` does not match corresponding
        sequence in `feature_config`.
      TypeError: If the type of any sequence in `gradients` does not match
        corresponding sequence in `feature_config`.
    """
    if not self._using_tpu:
      raise RuntimeError("apply_gradients is not valid when TPUEmbedding "
                         "object is not created under a TPUStrategy.")

    if not self._built:
      raise RuntimeError("apply_gradients called on unbuilt TPUEmbedding "
                         "object. Please either call enqueue first or manually "
                         "call the build method.")

    # send_tpu_embedding_gradients requires per table gradient, if we only have
    # one feature per table this isn't an issue. When multiple features share
    # the same table, the order of the features in per table tensor returned by
    # recv_tpu_embedding_activations matches the order in which they were passed
    # to enqueue.
    # In all three places, we use the fixed order given by nest.flatten to have
    # a consistent feature order.

    # First construct a dict of tensors one for each table.
    per_table_gradients = self._compute_per_table_gradients(gradients)

    # Now that we have a list of gradients we can compute a list of gradients
    # in the fixed order of self._table_config which interleave the gradients of
    # the individual features. We concat on axis 1 and then reshape into a 2d
    # tensor. The send gradients op expects a tensor of shape
    # [num_features*batch_size, dim] for each table.
    interleaved_gradients = []
    for table in self._table_config:
      interleaved_gradients.append(array_ops.reshape(
          array_ops.concat(per_table_gradients[table], axis=1),
          [-1, table.dim]))
    op = tpu_ops.send_tpu_embedding_gradients(
        inputs=interleaved_gradients,
        learning_rates=[math_ops.cast(fn(), dtype=dtypes.float32)
                        for fn in self._dynamic_learning_rates],
        config=self._config_proto.SerializeToString())

    # Apply the name tag to the op.
    if name is not None:
      _add_key_attr(op, name)

  def dequeue(self, name: Text = None):
    """Get the embedding results.

    Returns a nested structure of `tf.Tensor` objects, matching the structure of
    the `feature_config` argument to the `TPUEmbedding` class. The output shape
    of the tensors is `(batch_size, dim)`, where `batch_size` is the per core
    batch size, `dim` is the dimension of the corresponding `TableConfig`. If
    the feature's corresponding `FeatureConfig` has `max_sequence_length`
    greater than 0, the output will be a sequence of shape
    `(batch_size, max_sequence_length, dim)` instead.

    ```python
    strategy = tf.distribute.TPUStrategy(...)
    with strategy.scope():
      embedding = tf.tpu.experimental.embedding.TPUEmbedding(...)

    distributed_dataset = (
        strategy.distribute_datasets_from_function(
            dataset_fn=...,
            options=tf.distribute.InputOptions(
                experimental_fetch_to_device=False))
    dataset_iterator = iter(distributed_dataset)

    @tf.function
    def training_step():
      def tpu_step(tpu_features):
        with tf.GradientTape() as tape:
          activations = embedding.dequeue()
          tape.watch(activations)

          loss = ... #  some computation involving activations

        embedding_gradients = tape.gradient(loss, activations)
        embedding.apply_gradients(embedding_gradients)

      embedding_features, tpu_features = next(dataset_iterator)
      embedding.enqueue(embedding_features, training=True)
      strategy.run(tpu_step, args=(embedding_features, ))

    training_step()
    ```

    Args:
      name: A name for the underlying op.

    Returns:
      A nested structure of tensors, with the same structure as `feature_config`
    passed to this instance of the `TPUEmbedding` object.

    Raises:
      RuntimeError: If called when object wasn't created under a `TPUStrategy`
        or if not built (either by manually calling build or calling enqueue).
    """
    if not self._using_tpu:
      raise RuntimeError("dequeue is not valid when TPUEmbedding object is not "
                         "created under a TPUStrategy.")

    if not self._built:
      raise RuntimeError("dequeue called on unbuilt TPUEmbedding object. "
                         "Please either call enqueue first or manually call "
                         "the build method.")

    # The activations returned by this op are per table. So we must separate
    # them out into per feature activations. The activations are interleaved:
    # for each table, we expect a [num_features*batch_size, dim] tensor.
    # E.g. we expect the slice [:num_features, :] to contain the lookups for the
    # first example of all features using this table.
    activations = tpu_ops.recv_tpu_embedding_activations(
        num_outputs=len(self._table_config),
        config=self._config_proto.SerializeToString())

    # Apply the name tag to the op.
    if name is not None:
      _add_key_attr(activations[0].op, name)

    # Compute the number of features for this  table.
    num_features = {table: 0 for table in self._table_config}
    for feature in nest.flatten(self._feature_config):
      num_features[feature.table] += (1 if feature.max_sequence_length == 0
                                      else feature.max_sequence_length)

    # Activations are reshaped so that they are indexed by batch size and then
    # by the 'feature' index within the batch. The final dimension should equal
    # the dimension of the table.
    table_to_activation = {
        table: array_ops.reshape(activation,
                                 [self._batch_size, num_features[table], -1])
        for table, activation in zip(self._table_config, activations)}

    # We process the features in the same order we enqueued them.
    # For each feature we take the next slice of the activations, so need to
    # track the activations and the current position we are in.
    table_to_position = {table: 0 for table in self._table_config}

    per_feature_activations = []
    for feature in nest.flatten(self._feature_config):
      activation = table_to_activation[feature.table]
      feature_index = table_to_position[feature.table]
      # We treat non-sequence and sequence features differently here as sequence
      # features have rank 3 while non-sequence features have rank 2.
      if feature.max_sequence_length == 0:
        per_feature_activations.append(
            activation[:, feature_index, :])
        table_to_position[feature.table] += 1
      else:
        per_feature_activations.append(
            activation[:, feature_index:(
                feature_index+feature.max_sequence_length), :])
        table_to_position[feature.table] += feature.max_sequence_length

    # Pack the list back into the same nested structure as the features.
    return nest.pack_sequence_as(self._feature_config, per_feature_activations)

  def _create_variables_and_slots(
      self
  ) -> Dict[Text, Dict[Text, tf_variables.Variable]]:
    """Create variables for TPU embeddings.

    Note under TPUStrategy this will ensure that all creations happen within a
    variable creation scope of the sharded variable creator.

    Returns:
      A dict of dicts. The outer dict is keyed by the table names and the inner
      dicts are keyed by 'parameters' and the slot variable names.
    """

    def create_variables(table):
      """Create all variables."""
      variable_shape = (table.vocabulary_size, table.dim)

      def getter(name, shape, dtype, initializer, trainable):
        del shape
        # _add_variable_with_custom_getter clears the shape sometimes, so we
        # take the global shape from outside the getter.
        initial_value = functools.partial(initializer, variable_shape,
                                          dtype=dtype)
        return tf_variables.Variable(
            name=name,
            initial_value=initial_value,
            shape=variable_shape,
            dtype=dtype,
            trainable=trainable)

      def variable_creator(name, initializer, trainable=True):
        # use add_variable_with_custom_getter here so that we take advantage of
        # the checkpoint loading to allow restore before the variables get
        # created which avoids double initialization.
        return self._add_variable_with_custom_getter(
            name=name,
            initializer=initializer,
            shape=variable_shape,
            dtype=dtypes.float32,
            getter=getter,
            trainable=trainable)

      parameters = variable_creator(table.name, table.initializer,
                                    trainable=not self._using_tpu)

      def slot_creator(name, initializer):
        return variable_creator(table.name + "/" + name,
                                initializer,
                                False)

      if table.optimizer is not None:
        slot_vars = table.optimizer._create_slots(parameters, slot_creator)  # pylint: disable=protected-access
      else:
        slot_vars = {}
      slot_vars["parameters"] = parameters
      return slot_vars

    # Store tables based on name rather than TableConfig as we can't track
    # through dicts with non-string keys, i.e. we won't be able to save.
    variables = {}
    for table in self._table_config:
      if not self._using_tpu:
        variables[table.name] = create_variables(table)
      else:
        with variable_scope.variable_creator_scope(
            make_sharded_variable_creator(self._hosts)):
          variables[table.name] = create_variables(table)

    return variables

  def _load_variables(self):
    # Only load the variables if we are:
    # 1) Using TPU
    # 2) Variables are created
    # 3) Not in save context (except if running eagerly)
    if self._using_tpu and self._built and not (
        not context.executing_eagerly() and save_context.in_save_context()):
      _load_variables_impl(self._config_proto.SerializeToString(),
                           self._hosts,
                           self._variables,
                           self._table_config)

  def _retrieve_variables(self):
    # Only retrieve the variables if we are:
    # 1) Using TPU
    # 2) Variables are created
    # 3) Not in save context (except if running eagerly)
    if self._using_tpu and self._built and not (
        not context.executing_eagerly() and save_context.in_save_context()):
      _retrieve_variables_impl(self._config_proto.SerializeToString(),
                               self._hosts,
                               self._variables,
                               self._table_config)

  def _gather_saveables_for_checkpoint(
      self
  ) -> Dict[Text, Callable[[Text], "TPUEmbeddingSaveable"]]:
    """Overrides default Trackable implementation to add load/retrieve hook."""
    # This saveable should be here in both TPU and CPU checkpoints, so when on
    # CPU, we add the hook with no functions.
    # TODO(bfontain): Update restore logic in saver so that these hooks are
    # always executed. Once that is done, we can output an empty list when on
    # CPU.

    def factory(name=_HOOK_KEY):
      return TPUEmbeddingSaveable(name, self._load_variables,
                                  self._retrieve_variables)
    return {_HOOK_KEY: factory}

  # Some helper functions for the below enqueue function.
  def _add_data_for_tensor(self, tensor, weight, indices, values, weights,
                           int_zeros, float_zeros, path):
    if weight is not None:
      raise ValueError(
          "Weight specified for dense input {}, which is not allowed. "
          "Weight will always be 1 in this case.".format(path))
    # For tensors, there are no indices and no weights.
    indices.append(int_zeros)
    values.append(math_ops.cast(tensor, dtypes.int32))
    weights.append(float_zeros)

  def _add_data_for_sparse_tensor(self, tensor, weight, indices, values,
                                  weights, int_zeros, float_zeros, path):
    indices.append(math_ops.cast(tensor.indices, dtypes.int32))
    values.append(math_ops.cast(tensor.values, dtypes.int32))
    # If we have weights they must be a SparseTensor.
    if weight is not None:
      if not isinstance(weight, sparse_tensor.SparseTensor):
        raise ValueError("Weight for {} is type {} which does not match "
                         "type input which is SparseTensor.".format(
                             path, type(weight)))
      weights.append(math_ops.cast(weight.values, dtypes.float32))
    else:
      weights.append(float_zeros)

  def _add_data_for_ragged_tensor(self, tensor, weight, indices, values,
                                  weights, int_zeros, float_zeros, path):
    indices.append(math_ops.cast(tensor.row_splits, dtypes.int32))
    values.append(math_ops.cast(tensor.values, dtypes.int32))
    # If we have weights they must be a RaggedTensor.
    if weight is not None:
      if not isinstance(weight, ragged_tensor.RaggedTensor):
        raise ValueError("Weight for {} is type {} which does not match "
                         "type input which is RaggedTensor.".format(
                             path, type(weight)))
      weights.append(math_ops.cast(weight.values, dtypes.float32))
    else:
      weights.append(float_zeros)

  def _generate_enqueue_op(
      self,
      flat_inputs: List[internal_types.NativeObject],
      flat_weights: List[Optional[internal_types.NativeObject]],
      flat_features: List[tpu_embedding_v2_utils.FeatureConfig],
      device_ordinal: int,
      mode_override: Text
  ) -> ops.Operation:
    """Outputs a the enqueue op given the inputs and weights.

    Args:
      flat_inputs: A list of input tensors.
      flat_weights: A list of input weights (or None) of the same length as
        flat_inputs.
      flat_features: A list of FeatureConfigs of the same length as flat_inputs.
      device_ordinal: The device to create the enqueue op for.
      mode_override: A tensor containing the string "train" or "inference".

    Returns:
      The enqueue op.
    """

    # First we need to understand which op to use. This depends on if sparse
    # or ragged tensors are in the flat_inputs.
    sparse = False
    ragged = False
    for inp in flat_inputs:
      if isinstance(inp, sparse_tensor.SparseTensor):
        sparse = True
      elif isinstance(inp, ragged_tensor.RaggedTensor):
        ragged = True
    if sparse and ragged:
      raise ValueError(
          "Found both SparseTensors and RaggedTensors in the input to the "
          "enqueue operation. Please ensure that your data does not include "
          "both SparseTensors and RaggedTensors. It is ok to have Tensors in "
          "combination with one of the previous types.")

    # Combiners are per table, list in the same order as the table order.
    combiners = [table.combiner for table in self._table_config]

    # Reverse mapping of self._table_config, so that we can lookup the table
    # index.
    table_to_id = {table: i for i, table in enumerate(self._table_config)}

    # These parallel arrays will be the inputs to the enqueue op.
    indices = []  # sample_indices for sparse, sample_splits for ragged.
    values = []
    weights = []
    table_ids = []
    max_sequence_lengths = []

    # We have to supply a empty/zero tensor in a list position where we don't
    # have data (e.g. indices for standard Tensor input, weight when no weight
    # is specified). We create one op here per call, so that we reduce the
    # graph size.
    int_zeros = array_ops.zeros((0,), dtype=dtypes.int32)
    float_zeros = array_ops.zeros((0,), dtype=dtypes.float32)

    # In the following loop we insert casts so that everything is either int32
    # or float32. This is because op inputs which are lists of tensors must be
    # of the same type within the list. Moreover the CPU implementations of
    # these ops cast to these types anyway, so we don't lose any data by casting
    # early.
    for inp, weight, (path, feature) in zip(
        flat_inputs, flat_weights, flat_features):
      table_ids.append(table_to_id[feature.table])
      max_sequence_lengths.append(feature.max_sequence_length)
      if isinstance(inp, ops.Tensor):
        self._add_data_for_tensor(inp, weight, indices, values, weights,
                                  int_zeros, float_zeros, path)
      elif isinstance(inp, sparse_tensor.SparseTensor):
        self._add_data_for_sparse_tensor(inp, weight, indices, values, weights,
                                         int_zeros, float_zeros, path)
      elif isinstance(inp, ragged_tensor.RaggedTensor):
        self._add_data_for_ragged_tensor(inp, weight, indices, values, weights,
                                         int_zeros, float_zeros, path)
      else:
        raise ValueError("Input {} is of unknown type {}. Please only pass "
                         "Tensor, SparseTensor or RaggedTensor as input to "
                         "enqueue.".format(path, type(inp)))

    if ragged:
      return tpu_ops.enqueue_tpu_embedding_ragged_tensor_batch(
          sample_splits=indices,
          embedding_indices=values,
          aggregation_weights=weights,
          mode_override=mode_override,
          device_ordinal=device_ordinal,
          combiners=combiners,
          table_ids=table_ids,
          max_sequence_lengths=max_sequence_lengths)
    return tpu_ops.enqueue_tpu_embedding_sparse_tensor_batch(
        sample_indices=indices,
        embedding_indices=values,
        aggregation_weights=weights,
        mode_override=mode_override,
        device_ordinal=device_ordinal,
        combiners=combiners,
        table_ids=table_ids,
        max_sequence_lengths=max_sequence_lengths)

  def _raise_error_for_incorrect_control_flow_context(self):
    """Raises an error if we are not in the TPUReplicateContext."""
    # Do not allow any XLA control flow (i.e. control flow in between a
    # TPUStrategy's run call and the call to this function), as we can't
    # extract the enqueue from the head when in XLA control flow.
    graph = ops.get_default_graph()
    in_tpu_ctx = False
    while graph is not None:
      ctx = graph._get_control_flow_context()  # pylint: disable=protected-access
      while ctx is not None:
        if isinstance(ctx, tpu.TPUReplicateContext):
          in_tpu_ctx = True
          break
        ctx = ctx.outer_context
      if in_tpu_ctx:
        break
      graph = getattr(graph, "outer_graph", None)
    if graph != ops.get_default_graph() and in_tpu_ctx:
      raise RuntimeError(
          "Current graph {} does not match graph which contains "
          "TPUReplicateContext {}. This is most likely due to the fact that "
          "enqueueing embedding data is called inside control flow or a "
          "nested function inside `strategy.run`. This is not supported "
          "because outside compilation fails to extract the enqueue ops as "
          "head of computation.".format(ops.get_default_graph(), graph))
    return in_tpu_ctx

  def _raise_error_for_non_direct_inputs(self, features):
    """Checks all tensors in features to see if they are a direct input."""

    # expand_composites here is important: as composite tensors pass through
    # tpu.replicate, they get 'flattened' into their component tensors and then
    # repacked before being passed to the tpu function. In means that it is the
    # component tensors which are produced by an op with the
    # "_tpu_input_identity" attribute.
    for path, input_tensor in nest.flatten_with_joined_string_paths(
        features, expand_composites=True):
      if input_tensor.op.type == "Placeholder":
        continue
      try:
        is_input = input_tensor.op.get_attr("_tpu_input_identity")
      except ValueError:
        is_input = False
      if not is_input:
        raise ValueError(
            "Received input tensor {} which is the output of op {} (type {}) "
            "which does not have the `_tpu_input_identity` attr. Please "
            "ensure that the inputs to this layer are taken directly from "
            "the arguments of the function called by "
            "strategy.run. Two possible causes are: dynamic batch size "
            "support or you are using a keras layer and are not passing "
            "tensors which match the dtype of the `tf.keras.Input`s."
            "If you are triggering dynamic batch size support, you can "
            "disable it by passing tf.distribute.RunOptions("
            "experimental_enable_dynamic_batch_size=False) to the options "
            "argument of strategy.run().".format(path,
                                                 input_tensor.op.name,
                                                 input_tensor.op.type))

  def _raise_error_for_inputs_not_on_cpu(self, features):
    """Checks all tensors in features to see are placed on the CPU."""

    def check_device(path, device_string):
      spec = tf_device.DeviceSpec.from_string(device_string)
      if spec.device_type == "TPU":
        raise ValueError(
            "Received input tensor {} which is on a TPU input device {}. Input "
            "tensors for TPU embeddings must be placed on the CPU. Please "
            "ensure that your dataset is prefetching tensors to the host by "
            "setting the 'experimental_fetch_to_device' option of the "
            "dataset distribution function. See the documentation of the "
            "enqueue method for an example.".format(path, device_string))

    # expand_composites here is important, we need to check the device of each
    # underlying tensor.
    for path, input_tensor in nest.flatten_with_joined_string_paths(
        features, expand_composites=True):
      if (input_tensor.op.type == "Identity" and
          input_tensor.op.inputs[0].op.type == "TPUReplicatedInput"):
        for tensor in input_tensor.op.inputs[0].op.inputs:
          check_device(path, tensor.device)
      else:
        check_device(path, input_tensor.device)

  def enqueue(
      self,
      features,
      weights=None,
      training: bool = True,
      name: Optional[Text] = None,
      device: Optional[Text] = None):
    """Enqueues id tensors for embedding lookup.

    This function enqueues a structure of features to be looked up in the
    embedding tables. We expect that the batch size of each of the tensors in
    features matches the per core batch size. This will automatically happen if
    your input dataset is batched to the global batch size and you use
    `tf.distribute.TPUStrategy`'s `experimental_distribute_dataset`
    or if you use `distribute_datasets_from_function` and batch
    to the per core batch size computed by the context passed to your input
    function.

    ```python
    strategy = tf.distribute.TPUStrategy(...)
    with strategy.scope():
      embedding = tf.tpu.experimental.embedding.TPUEmbedding(...)

    distributed_dataset = (
        strategy.distribute_datasets_from_function(
            dataset_fn=...,
            options=tf.distribute.InputOptions(
                experimental_fetch_to_device=False))
    dataset_iterator = iter(distributed_dataset)

    @tf.function
    def training_step():
      def tpu_step(tpu_features):
        with tf.GradientTape() as tape:
          activations = embedding.dequeue()
          tape.watch(activations)

          loss = ... #  some computation involving activations

        embedding_gradients = tape.gradient(loss, activations)
        embedding.apply_gradients(embedding_gradients)

      embedding_features, tpu_features = next(dataset_iterator)
      embedding.enqueue(embedding_features, training=True)
      strategy.run(tpu_step, args=(embedding_features,))

    training_step()
    ```

    NOTE: You should specify `training=True` when using
    `embedding.apply_gradients` as above and `training=False` when not using
    `embedding.apply_gradients` (e.g. for frozen embeddings or when doing
    evaluation).

    For finer grained control, in the above example the line

    ```
      embedding.enqueue(embedding_features, training=True)
    ```

    may be replaced with

    ```
      per_core_embedding_features = self.strategy.experimental_local_results(
          embedding_features)

      def per_core_enqueue(ctx):
        core_id = ctx.replica_id_in_sync_group
        device = strategy.extended.worker_devices[core_id]
        embedding.enqueue(per_core_embedding_features[core_id],
                          device=device)

      strategy.experimental_distribute_values_from_function(
          per_core_queue_inputs)
    ```

    Args:
      features: A nested structure of `tf.Tensor`s, `tf.SparseTensor`s or
        `tf.RaggedTensor`s, with the same structure as `feature_config`. Inputs
        will be downcast to `tf.int32`. Only one type out of `tf.SparseTensor`
        or `tf.RaggedTensor` is supported per call.
      weights: If not `None`, a nested structure of `tf.Tensor`s,
        `tf.SparseTensor`s or `tf.RaggedTensor`s, matching the above, except
        that the tensors should be of float type (and they will be downcast to
        `tf.float32`). For `tf.SparseTensor`s we assume the `indices` are the
        same for the parallel entries from `features` and similarly for
        `tf.RaggedTensor`s we assume the row_splits are the same.
      training: Defaults to `True`. If `False`, enqueue the batch as inference
        batch (forward pass only). Do not call `apply_gradients` when this is
        `False` as this may lead to a deadlock.
       name: A name for the underlying op.
       device: The device name (e.g. '/task:0/device:TPU:2') where this batch
         should be enqueued. This should be set if and only if features is not a
         `tf.distribute.DistributedValues` and enqueue is not being called
         inside a TPU context (e.g. inside `TPUStrategy.run`).

    Raises:
      ValueError: When called inside a strategy.run call and input is not
        directly taken from the args of the `strategy.run` call. Also if
        the size of any sequence in `features` does not match corresponding
        sequence in `feature_config`. Similarly for `weights`, if not `None`.
        If batch size of features is unequal or different from a previous call.
      RuntimeError: When called inside a strategy.run call and inside XLA
        control flow. If batch_size is not able to be determined and build was
        not called.
      TypeError: If the type of any sequence in `features` does not match
        corresponding sequence in `feature_config`. Similarly for `weights`, if
        not `None`.
    """
    if not self._using_tpu:
      raise RuntimeError("enqueue is not valid when TPUEmbedding object is not "
                         "created under a TPUStrategy.")

    in_tpu_context = self._raise_error_for_incorrect_control_flow_context()

    # Should we also get batch_size from weights if they exist?
    # Since features is assumed to be batched at the per replica batch size
    # the returned batch size here is per replica an not global.
    batch_size = self._get_batch_size(features, in_tpu_context)
    if batch_size is None and not self._built:
      raise RuntimeError("Unable to determine batch size from input features."
                         "Please call build() with global batch size to "
                         "initialize the TPU for embeddings.")
    if batch_size is not None:
      self._maybe_build(batch_size)
      if self._batch_size != batch_size:
        raise ValueError("Multiple calls to enqueue with different batch sizes "
                         "{} and {}.".format(self._batch_size,
                                             batch_size))

    nest.assert_same_structure(self._feature_config, features)

    flat_inputs = nest.flatten(features)
    flat_weights = [None] * len(flat_inputs)
    if weights is not None:
      nest.assert_same_structure(self._feature_config, weights)
      flat_weights = nest.flatten(weights)
    flat_features = nest.flatten_with_joined_string_paths(self._feature_config)

    self._raise_error_for_inputs_not_on_cpu(features)
    # If we are in a tpu_context, automatically apply outside compilation.
    if in_tpu_context:
      self._raise_error_for_non_direct_inputs(features)

      def generate_enqueue_ops():
        """Generate enqueue ops for outside compilation."""
        # Note that we put array_ops.where_v2 rather than a python if so that
        # the op is explicitly create and the constant ops are both in the graph
        # even though we don't expect training to be a tensor (and thus generate
        # control flow automatically). This need to make it easier to re-write
        # the graph later if we need to fix which mode needs to be used.
        mode_override = array_ops.where_v2(training,
                                           constant_op.constant("train"),
                                           constant_op.constant("inference"))

        # Device ordinal is -1 here, a later rewrite will fix this once the op
        # is expanded by outside compilation.
        enqueue_op = self._generate_enqueue_op(
            flat_inputs, flat_weights, flat_features, device_ordinal=-1,
            mode_override=mode_override)

        # Apply the name tag to the op.
        if name is not None:
          _add_key_attr(enqueue_op, name)

        # Ensure that this op has outbound control flow, otherwise it won't be
        # executed.
        ops.get_default_graph().control_outputs.append(enqueue_op)

      tpu.outside_compilation(generate_enqueue_ops)

    elif device is None:
      mode_override = "train" if training else "inference"
      # We generate enqueue ops per device, so we need to gather the all
      # features for a single device in to a dict.
      # We rely here on the fact that the devices in the PerReplica value occur
      # in the same (standard) order as self._strategy.extended.worker_devices.
      enqueue_ops = []
      for replica_id in range(self._strategy.num_replicas_in_sync):
        replica_inputs = distribute_utils.select_replica(replica_id,
                                                         flat_inputs)
        replica_weights = distribute_utils.select_replica(replica_id,
                                                          flat_weights)
        tpu_device = self._strategy.extended.worker_devices[replica_id]
        # TPU devices string are like /job:worker/replica:0/task:0/device:TPU:0
        # the device ordinal is the last number
        device_ordinal = (
            tf_device.DeviceSpec.from_string(tpu_device).device_index)
        with ops.device(device_util.get_host_for_device(tpu_device)):
          enqueue_op = self._generate_enqueue_op(
              replica_inputs, replica_weights, flat_features,
              device_ordinal=device_ordinal, mode_override=mode_override)

          # Apply the name tag to the op.
          if name is not None:
            _add_key_attr(enqueue_op, name)
          enqueue_ops.append(enqueue_op)
      ops.get_default_graph().control_outputs.extend(enqueue_ops)
    else:
      mode_override = "train" if training else "inference"
      device_spec = tf_device.DeviceSpec.from_string(device)
      if device_spec.device_type != "TPU":
        raise ValueError(
            "Non-TPU device {} passed to enqueue.".format(device))
      with ops.device(device_util.get_host_for_device(device)):
        enqueue_op = self._generate_enqueue_op(
            flat_inputs, flat_weights, flat_features,
            device_ordinal=device_spec.device_index,
            mode_override=mode_override)

        # Apply the name tag to the op.
        if name is not None:
          _add_key_attr(enqueue_op, name)
        ops.get_default_graph().control_outputs.append(enqueue_op)

  def _get_batch_size(self, tensors, in_tpu_context: bool):
    """Gets the batch size from a nested structure of features."""
    batch_size = None
    for path, maybe_tensor in nest.flatten_with_joined_string_paths(tensors):
      tensor_list = []
      if not in_tpu_context:
        # if we are not in a context, then this is PerReplica and we need to
        # check each replica's batch size.
        for replica_id in range(self._strategy.num_replicas_in_sync):
          tensor_list.append(distribute_utils.select_replica(replica_id,
                                                             maybe_tensor))
      else:
        tensor_list = [maybe_tensor]

      for tensor in tensor_list:
        if tensor.shape.rank < 1:
          raise ValueError(
              "Input {} has rank 0, rank must be at least 1.".format(path))
        shape = tensor.shape.as_list()
        if shape[0] is not None:
          if batch_size is None:
            batch_size = shape[0]
          elif batch_size != shape[0]:
            raise ValueError("Found multiple batch sizes {} and {}. All inputs "
                             "must have the same batch dimensions size.".format(
                                 batch_size, shape[0]))
    return batch_size


@def_function.function
def _load_variables_impl(
    config: Text,
    hosts: List[Tuple[int, Text]],
    variables: Dict[Text, Dict[Text, tf_variables.Variable]],
    table_config: tpu_embedding_v2_utils.TableConfig):
  """Load embedding tables to onto TPU for each table and host.

  Args:
    config: A serialized TPUEmbeddingConfiguration proto.
    hosts: A list of CPU devices, on per host.
    variables: A dictionary of dictionaries of TPUShardedVariables. First key is
      the table name, second key is 'parameters' or the optimizer slot name.
    table_config: A list of tf.tpu.experimental.embedding.TableConfig objects.
  """
  def select_fn(host_id):

    def select_or_zeros(x):
      if host_id >= len(x.variables):
        # In the edge case where we have more hosts than variables, due to using
        # a small number of rows, we load zeros for the later hosts. We copy
        # the shape of the first host's variables, which we assume is defined
        # because TableConfig guarantees at least one row.
        return array_ops.zeros_like(x.variables[0])
      return x.variables[host_id]

    return select_or_zeros

  for host_id, host in enumerate(hosts):
    with ops.device(host):
      host_variables = nest.map_structure(select_fn(host_id), variables)
      for table in table_config:
        table.optimizer._load()(  # pylint: disable=protected-access
            table_name=table.name,
            num_shards=len(hosts),
            shard_id=host_id,
            config=config,
            **host_variables[table.name])
        # Ensure that only the first table/first host gets a config so that we
        # don't bloat graph by attaching this large string to each op.
        # We have num tables * num hosts of these so for models with a large
        # number of tables training on a large slice, this can be an issue.
        config = None


@def_function.function
def _retrieve_variables_impl(
    config: Text,
    hosts: List[Tuple[int, Text]],
    variables: Dict[Text, Dict[Text, tf_variables.Variable]],
    table_config: tpu_embedding_v2_utils.TableConfig):
  """Retrieve embedding tables from TPU to host memory.

  Args:
    config: A serialized TPUEmbeddingConfiguration proto.
    hosts: A list of all the host CPU devices.
    variables: A dictionary of dictionaries of TPUShardedVariables. First key is
      the table name, second key is 'parameters' or the optimizer slot name.
    table_config: A list of tf.tpu.experimental.embedding.TableConfig objects.
  """
  for host_id, host in enumerate(hosts):
    with ops.device(host):
      for table in table_config:
        retrieved = table.optimizer._retrieve()(  # pylint: disable=protected-access
            table_name=table.name,
            num_shards=len(hosts),
            shard_id=host_id,
            config=config)
        # When there are no slot variables (e.g with SGD) this returns a
        # single tensor rather than a tuple. In this case we put the tensor in
        # a list to make the following code easier to write.
        if not isinstance(retrieved, tuple):
          retrieved = (retrieved,)

        for i, slot in enumerate(["parameters"] +
                                 table.optimizer._slot_names()):  # pylint: disable=protected-access
          # We must assign the CPU variables the values of tensors that were
          # returned from the TPU.
          sharded_var = variables[table.name][slot]
          if host_id < len(sharded_var.variables):
            # In the edge case where we have more hosts than variables, due to
            # using a small number of rows, we skip the later hosts.
            sharded_var.variables[host_id].assign(retrieved[i])
        # Ensure that only the first table/first host gets a config so that we
        # don't bloat graph by attaching this large string to each op.
        # We have num tables * num hosts of these so for models with a large
        # number of tables training on a large slice, this can be an issue.
        config = None


class TPUEmbeddingSaveable(saveable_hook.SaveableHook):
  """Save/Restore hook to Retrieve/Load TPUEmbedding variables."""

  def __init__(
      self,
      name: Text,
      load: Callable[[], Any],
      retrieve: Callable[[], Any]):
    self._load = load
    self._retrieve = retrieve
    super(TPUEmbeddingSaveable, self).__init__(name=name)

  def before_save(self):
    if self._retrieve is not None:
      self._retrieve()

  def after_restore(self):
    if self._load is not None:
      self._load()


def _ragged_embedding_lookup_with_reduce(
    table: tf_variables.Variable,
    ragged: ragged_tensor.RaggedTensor,
    weights: ragged_tensor.RaggedTensor,
    combiner: Text) -> core.Tensor:
  """Compute a ragged lookup followed by a reduce on axis 1.

  Args:
    table: The embedding table.
    ragged: A RaggedTensor of ids to look up.
    weights: A RaggedTensor of weights (or None).
    combiner: One of "mean", "sum", "sqrtn".

  Returns:
    A Tensor.
  """
  if weights is None:
    weights = array_ops.ones_like(ragged, dtype=table.dtype)
  weights = array_ops.expand_dims(weights, axis=2)
  ragged_result = embedding_ops.embedding_lookup_ragged(table, ragged)
  ragged_result = math_ops.reduce_sum(ragged_result * weights, axis=1)
  if combiner == "mean":
    ragged_result = ragged_result / math_ops.reduce_sum(weights, axis=1)
  elif combiner == "sqrtn":
    ragged_result = ragged_result, math_ops.sqrt(math_ops.reduce_sum(
        weights*weights, axis=1))
  return ragged_result


@tf_export("tpu.experimental.embedding.serving_embedding_lookup")
def cpu_embedding_lookup(inputs, weights, tables, feature_config):
  """Apply standard lookup ops with `tf.tpu.experimental.embedding` configs.

  This function is a utility which allows using the
  `tf.tpu.experimental.embedding` config objects with standard lookup functions.
  This can be used when exporting a model which uses
  `tf.tpu.experimental.embedding.TPUEmbedding` for serving on CPU. In particular
  `tf.tpu.experimental.embedding.TPUEmbedding` only supports lookups on TPUs and
  should not be part of your serving graph.

  Note that TPU specific options (such as `max_sequence_length`) in the
  configuration objects will be ignored.

  In the following example we take a trained model (see the documentation for
  `tf.tpu.experimental.embedding.TPUEmbedding` for the context) and create a
  saved model with a serving function that will perform the embedding lookup and
  pass the results to your model:

  ```python
  model = model_fn(...)
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=1024,
      optimizer=tf.tpu.experimental.embedding.SGD(0.1))
  checkpoint = tf.train.Checkpoint(model=model, embedding=embedding)
  checkpoint.restore(...)

  @tf.function(input_signature=[{'feature_one': tf.TensorSpec(...),
                                 'feature_two': tf.TensorSpec(...),
                                 'feature_three': tf.TensorSpec(...)}])
  def serve_tensors(embedding_featurese):
    embedded_features = tf.tpu.experimental.embedding.serving_embedding_lookup(
        embedding_features, None, embedding.embedding_tables,
        feature_config)
    return model(embedded_features)

  model.embedding_api = embedding
  tf.saved_model.save(model,
                      export_dir=...,
                      signatures={'serving_default': serve_tensors})

  ```

  NOTE: Its important to assign the embedding api object to a member of your
  model as `tf.saved_model.save` only supports saving variables one `Trackable`
  object. Since the model's weights are in `model` and the embedding table are
  managed by `embedding`, we assign `embedding` to and attribute of `model` so
  that tf.saved_model.save can find the embedding variables.

  NOTE: The same `serve_tensors` function and `tf.saved_model.save` call will
  work directly from training.

  Args:
    inputs: a nested structure of Tensors, SparseTensors or RaggedTensors.
    weights: a nested structure of Tensors, SparseTensors or RaggedTensors or
      None for no weights. If not None, structure must match that of inputs, but
      entries are allowed to be None.
    tables: a dict of mapping TableConfig objects to Variables.
    feature_config: a nested structure of FeatureConfig objects with the same
      structure as inputs.

  Returns:
    A nested structure of Tensors with the same structure as inputs.
  """

  nest.assert_same_structure(inputs, feature_config)

  flat_inputs = nest.flatten(inputs)
  flat_weights = [None] * len(flat_inputs)
  if weights is not None:
    nest.assert_same_structure(inputs, weights)
    flat_weights = nest.flatten(weights)
  flat_features = nest.flatten_with_joined_string_paths(feature_config)

  outputs = []
  for inp, weight, (path, feature) in zip(
      flat_inputs, flat_weights, flat_features):
    table = tables[feature.table]

    if weight is not None:
      if isinstance(inp, ops.Tensor):
        raise ValueError(
            "Weight specified for {}, but input is dense.".format(path))
      elif type(weight) is not type(inp):
        raise ValueError(
            "Weight for {} is of type {} but it does not match type of the "
            "input which is {}.".format(path, type(weight), type(inp)))
      elif feature.max_sequence_length > 0:
        raise ValueError("Weight specified for {}, but this is a sequence "
                         "feature.".format(path))

    if isinstance(inp, ops.Tensor):
      if feature.max_sequence_length > 0:
        raise ValueError("Feature {} is a sequence feature but a dense tensor "
                         "was passed.".format(path))
      outputs.append(embedding_ops.embedding_lookup_v2(table, inp))

    elif isinstance(inp, sparse_tensor.SparseTensor):
      if feature.max_sequence_length > 0:
        batch_size = math_ops.cast(array_ops.shape(inp)[0], dtype=dtypes.int64)
        sparse_shape = array_ops.stack(
            [batch_size, feature.max_sequence_length], axis=0)
        # TPU Embedding truncates sequences to max_sequence_length, and if we
        # don't truncate, scatter_nd will error out if the index was out of
        # bounds.
        truncated_inp = sparse_ops.sparse_slice(inp, start=[0, 0],
                                                size=sparse_shape)

        dense_output_shape = array_ops.stack(
            [batch_size, feature.max_sequence_length, feature.table.dim],
            axis=0)
        outputs.append(
            array_ops.scatter_nd(
                inp.indices, array_ops.gather(table, truncated_inp.values),
                dense_output_shape))
      else:
        outputs.append(embedding_ops.safe_embedding_lookup_sparse_v2(
            table, inp, sparse_weights=weight, combiner=feature.table.combiner))

    elif isinstance(inp, ragged_tensor.RaggedTensor):
      if feature.max_sequence_length > 0:
        batch_size = inp.shape[0]
        dense_output_shape = [
            batch_size, feature.max_sequence_length, feature.table.dim]
        ragged_lookup = embedding_ops.embedding_lookup_v2(table, inp)
        # Unlike scatter_nd, RaggedTensor.to_tensor truncates to the given
        # shape.
        outputs.append(ragged_lookup.to_tensor(shape=dense_output_shape))
      else:
        outputs.append(_ragged_embedding_lookup_with_reduce(
            table, inp, weight, feature.table.combiner))

    else:
      raise ValueError("Input {} is type {}. Tensor, SparseTensor or "
                       "RaggedTensor expected.".format(path, type(inp)))
  return nest.pack_sequence_as(feature_config, outputs)


def get_list_of_hosts(strategy: tpu_strategy.TPUStrategy) -> List[Text]:
  """Returns a sorted list of CPU devices for the remote jobs.

  Args:
    strategy: A TPUStrategy object.

  Returns:
    A sort list of device strings.
  """
  list_of_hosts = []
  # Assume this is sorted by task
  for tpu_device in strategy.extended.worker_devices:
    host = device_util.get_host_for_device(tpu_device)
    if host not in list_of_hosts:
      list_of_hosts.append(host)
  assert len(list_of_hosts) == strategy.extended.num_hosts
  return list_of_hosts


def extract_variable_info(
    kwargs) -> Tuple[Text, Tuple[int, ...], dtypes.DType, Callable[[], Any]]:
  """Extracts the variable creation attributes from the kwargs.

  Args:
    kwargs: a dict of keyword arguments that were passed to a variable creator
      scope.

  Returns:
    A tuple of variable name, shape, dtype, initialization function.
  """
  if (isinstance(kwargs["initial_value"], functools.partial) and (
      "shape" in kwargs["initial_value"].keywords or
      kwargs["initial_value"].args)):
    # Sometimes shape is passed positionally, sometimes it's passed as a kwarg.
    if "shape" in kwargs["initial_value"].keywords:
      shape = kwargs["initial_value"].keywords["shape"]
    else:
      shape = kwargs["initial_value"].args[0]
    return (kwargs["name"], shape,
            kwargs["initial_value"].keywords.get("dtype", kwargs["dtype"]),
            kwargs["initial_value"].func)
  elif "shape" not in kwargs or kwargs["shape"] is None or not callable(
      kwargs["initial_value"]):
    raise ValueError(
        "Unable to extract initializer function and shape from {}. Please "
        "either pass a function that expects a shape and dtype as the "
        "initial value for your variable or functools.partial object with "
        "the shape and dtype kwargs set. This is needed so that we can "
        "initialize the shards of the ShardedVariable locally.".format(
            kwargs["initial_value"]))
  else:
    return (kwargs["name"], kwargs["shape"], kwargs["dtype"],
            kwargs["initial_value"])


def make_sharded_variable_creator(
    hosts: List[Text]) -> Callable[..., TPUShardedVariable]:
  """Makes a sharded variable creator given a list of hosts.

  Args:
    hosts: a list of tensorflow devices on which to shard the tensors.

  Returns:
    A variable creator function.
  """

  def sharded_variable_creator(
      next_creator: Callable[..., tf_variables.Variable], *args, **kwargs):
    """The sharded variable creator."""
    kwargs["skip_mirrored_creator"] = True

    num_hosts = len(hosts)
    name, shape, dtype, unwrapped_initial_value = extract_variable_info(kwargs)
    initial_value = kwargs["initial_value"]
    rows = shape[0]
    cols = shape[1]
    partial_partition = rows % num_hosts
    full_rows_per_host = rows // num_hosts
    # We partition as if we were using MOD sharding: at least
    # `full_rows_per_host` rows to `num_hosts` hosts, where the first
    # `partial_partition` hosts get an additional row when the number of rows
    # is not cleanly divisible. Note that `full_rows_per_host` may be zero.
    partitions = (
        [full_rows_per_host + 1] * partial_partition
        + [full_rows_per_host] * (num_hosts - partial_partition))
    variables = []
    sharding_aware = "shard_info" in tf_inspect.getargspec(initial_value).args

    # Keep track of offset for sharding aware initializers.
    offset = 0
    kwargs["dtype"] = dtype
    for i, p in enumerate(partitions):
      if p == 0:
        # Skip variable creation for empty partitions, resulting from the edge
        # case of 'rows < num_hosts'. This is safe because both load/restore
        # can handle the missing values.
        continue
      with ops.device(hosts[i]):
        kwargs["name"] = "{}_{}".format(name, i)
        kwargs["shape"] = (p, cols)
        if sharding_aware:
          shard_info = base.ShardInfo(kwargs["shape"], (offset, 0))
          kwargs["initial_value"] = functools.partial(
              initial_value, shard_info=shard_info)
          offset += p
        else:
          kwargs["initial_value"] = functools.partial(
              unwrapped_initial_value, kwargs["shape"], dtype=dtype)
        variables.append(next_creator(*args, **kwargs))
    return TPUShardedVariable(variables, name=name)
  return sharded_variable_creator
