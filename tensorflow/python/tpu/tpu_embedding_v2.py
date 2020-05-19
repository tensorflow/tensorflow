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
from absl import logging

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import values as tf_values
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training.saving import saveable_hook
from tensorflow.python.training.tracking import tracking
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


_HOOK_KEY = "TPUEmbedding_saveable"
_NAME_KEY = "_tpu_embedding_layer"


# TODO(bfontain): Cleanup and remove this once there is an implementation of
# sharded variables that can be used in the PSStrategy with optimizers.
# We implement just enough of the of a tf.Variable so that this could be passed
# to an optimizer.
class TPUShardedVariable(sharded_variable.ShardedVariable):
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
  strategy = tf.distribute.experimental.TPUStrategy(...)
  with strategy.scope():
    embedding = tf.tpu.experimental.embedding.TPUEmbedding(
        feature_config=feature_config,
        batch_size=1024,
        optimizer=tf.tpu.experimental.embedding.SGD(0.1))
  ```

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
      batch_size=1024,
      optimizer=tf.tpu.experimental.embedding.SGD(0.1))
  checkpoint = tf.train.Checkpoint(model=model, embedding=embedding)
  checkpoint.restore(...)

  tables = embedding.embedding_tables
  ```

  You can now use table in functions like `tf.nn.embedding_lookup` to perform
  your embedding lookup and pass to your model.

  """

  def __init__(self, feature_config, batch_size, optimizer,
               pipeline_execution_with_tensor_core=False,
               initialize_tpu_embedding=True):
    """Creates the TPUEmbedding mid level API object.

    ```python
    strategy = tf.distribute.experimental.TPUStrategy(...)
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
      batch_size: The global batch size that you indend to use. Note that is
        fixed and the same batch size must be used for both training and
        evaluation.
      optimizer: An instance of one of `tf.tpu.experimental.embedding.SGD`,
        `tf.tpu.experimental.embedding.Adagrad` or
        `tf.tpu.experimental.embedding.Adam`.
      pipeline_execution_with_tensor_core: If True, the TPU embedding
        computations will overlap with the TensorCore computations (and hence
        will be one step old). Set to True for improved performance.
      initialize_tpu_embedding: If False, will not initialize the TPU embedding
        engine. If this is set to False and another instance of this class has
        not initialized the tpu embedding engine, the creation of this object
        will fail.

    Raises:
      ValueError: If optimizer is not one of tf.tpu.experimental.embedding.(SGD,
      Adam or Adagrad).
    """
    self._strategy = distribution_strategy_context.get_strategy()
    self._using_tpu = isinstance(self._strategy, tpu_strategy.TPUStrategy)
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

    # Set table order here
    self._table_config = list(
        {feature.table for feature in nest.flatten(feature_config)})

    # Ensure tables have unique names. Also error check the optimizer as we
    # specifically don't do that in the TableConfig class to allow high level
    # APIs that are built on this to use strings/other classes to represent
    # optimizers (before they are passed to this class).
    table_names = []
    for i, table in enumerate(self._table_config):
      if table.optimizer is None:
        # TODO(bfontain) Should we allow some sort of optimizer merging here?
        table.optimizer = optimizer
      if not isinstance(table.optimizer, tpu_embedding_v2_utils._Optimizer):  # pylint: disable=protected-access
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

      # TODO(bfontain) Remove this once we have an official way of splitting
      # prefetch between host and device.
      self._strategy.extended._set_prefetch_on_host(True)  # pylint: disable=protected-access

      # We generally use the per core batch size, but will have the user pass
      # in a global batch size.
      self._batch_size = batch_size // self._strategy.num_replicas_in_sync

      self._config_proto = self._create_config_proto()
      if initialize_tpu_embedding:
        # This is mainly for testing purposes, sometimes we don't want to
        # initialize the embedding engine, but just want a copy of the API
        # which can interact with an already initialized engine.
        logging.info("Initializing TPU Embedding engine with config: %s",
                     self._config_proto)
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
    if self._using_tpu:
      self._load_variables()

  @property
  def embedding_tables(self):
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
      raise RuntimeError("Unable to retrieve embedding tables when using a TPU "
                         "strategy. If you need access, save your model, "
                         "create this object under a CPU strategy and restore.")

    # Only return the tables and not the slot variables. On CPU this are honest
    # tf.Variables.
    return {table: self._variables[table.name]["parameters"]
            for table in self._table_config}

  def _create_config_proto(self):
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

  def _compute_per_table_gradients(self, gradients):
    """Computes a dict of lists of gradients, keyed by table name.

    Args:
      gradients: A nested structure of Tensors (and Nones) with the same
        structure as the feature config.

    Returns:
      A dict of lists of tensors, keyed by the table names, containing the
    gradients in the correct order with None gradients repalaced by zeros.
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

  def apply_gradients(self, gradients, name=None):
    """Applies the gradient update to the embedding tables.

    If a gradient of `None` is passed in any position of the nested structure,
    then an gradient update with a zero gradient is applied for that feature.
    For optimizers like SGD or Adagrad, this is the same as applying no update
    at all. For lazy Adam and other sparsely applied optimizers with decay,
    ensure you understand the effect of applying a zero gradient.

    ```python
    strategy = tf.distribute.experimental.TPUStrategy(...)
    with strategy.scope():
      embedding = tf.tpu.experimental.embedding.TPUEmbedding(...)

    distributed_dataset = strategy.experimental_distribute_dataset(...)
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
      RuntimeError: If called when object wasn't created under a `TPUStrategy`.
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

  def dequeue(self, name=None):
    """Get the embedding results.

    Returns a nested structure of `tf.Tensor` objects, matching the structure of
    the `feature_config` argument to the `TPUEmbedding` class. The output shape
    of the tensors is `(batch_size, dim)`, where `batch_size` is the per core
    batch size, `dim` is the dimension of the corresponding `TableConfig`. If
    the feature's corresponding `FeatureConfig` has `max_sequence_length`
    greater than 0, the output will be a sequence of shape
    `(batch_size, max_sequence_length, dim)` instead.

    ```python
    strategy = tf.distribute.experimental.TPUStrategy(...)
    with strategy.scope():
      embedding = tf.tpu.experimental.embedding.TPUEmbedding(...)

    distributed_dataset = strategy.experimental_distribute_dataset(...)
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
      RuntimeError: If called when object wasn't created under a `TPUStrategy`.
    """
    if not self._using_tpu:
      raise RuntimeError("dequeue is not valid when TPUEmbedding object is not "
                         "created under a TPUStrategy.")

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

  def _create_variables_and_slots(self):
    """Create variables for TPU embeddings.

    Note under TPUStrategy this will ensure that all creations happen within a
    variable creation scope of the sharded variable creator.

    Returns:
      A dict of dicts. The outer dict is keyed by the table names and the inner
      dicts are keyed by 'parameters' and the slot variable names.
    """

    def create_variables(table):
      """Create all variables."""
      shape = (table.vocabulary_size, table.dim)

      # We use functools.partial here for the initial_value so that we have a
      # variable creation that is compatible with both the sharded variable
      # creator and the normal variable creator. The sharded variable creator
      # will extract the shape of the tensor from the functool.partial object to
      # decide on the sharding.
      parameters = tf_variables.Variable(
          name=table.name,
          initial_value=functools.partial(
              table.initializer, shape=shape, dtype=dtypes.float32),
          trainable=not self._using_tpu)
      slot_vars = table.optimizer._create_slots(parameters)  # pylint: disable=protected-access
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

  @def_function.function
  def _load_variables(self):
    """Load embedding tables to onto TPU for each table and host."""

    def select_fn(host_id):
      return lambda x: x.variables[host_id]

    num_hosts = self._strategy.extended.num_hosts
    config = self._config_proto.SerializeToString()
    for host_id, host in enumerate(self._hosts):
      variables = nest.map_structure(select_fn(host_id), self._variables)
      with ops.device(host):
        for table in self._table_config:
          table.optimizer._load()(  # pylint: disable=protected-access
              table_name=table.name,
              num_shards=num_hosts,
              shard_id=host_id,
              config=config,
              **variables[table.name])
          # Ensure that only the first table/first host gets a config so that we
          # don't bloat graph by attaching this large string to each op.
          # We have num tables * num hosts of these so for models with a large
          # number of tables training on a large slice, this can be an issue.
          config = None

  @def_function.function
  def _retrieve_variables(self):
    """Retrieve embedding tables from TPU to host memory."""
    num_hosts = self._strategy.extended.num_hosts
    config = self._config_proto.SerializeToString()
    for host_id, host in enumerate(self._hosts):
      with ops.device(host):
        for table in self._table_config:
          retrieved = table.optimizer._retrieve()(  # pylint: disable=protected-access
              table_name=table.name,
              num_shards=num_hosts,
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
            self._variables[table.name][slot].variables[host_id].assign(
                retrieved[i])
          # Ensure that only the first table/first host gets a config so that we
          # don't bloat graph by attaching this large string to each op.
          # We have num tables * num hosts of these so for models with a large
          # number of tables training on a large slice, this can be an issue.
          config = None

  def _gather_saveables_for_checkpoint(self):
    """Overrides default Trackable implementation to add load/retrieve hook."""
    # This saveable should be here in both TPU and CPU checkpoints, so when on
    # CPU, we add the hook with no functions.
    # TODO(bfontain): Update restore logic in saver so that these hooks are
    # always executed. Once that is done, we can output an empty list when on
    # CPU.
    def factory(name=_HOOK_KEY):
      return TPUEmbeddingSaveable(
          name,
          self._load_variables if self._using_tpu else None,
          self._retrieve_variables if self._using_tpu else None)
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

  def _generate_enqueue_op(self, flat_inputs, flat_weights, flat_features,
                           device_ordinal, mode_override):
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
    # of the same type within the list. Moreover the CPU implementions of these
    # ops cast to these types anyway, so we don't lose any data by casting
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

  def enqueue(self, features, weights=None, training=True, name=None):
    """Enqueues id tensors for embedding lookup.

    This function enqueues a structure of features to be looked up in the
    embedding tables. We expect that the batch size of each of the tensors in
    features matches the per core batch size. This will automatically happen if
    your input dataset is batched to the global batch size and you use
    `tf.distribute.experimental.TPUStrategy`'s `experimental_distribute_dataset`
    or if you use `experimental_distribute_datasets_from_function` and batch
    to the per core batch size computed by the context passed to your input
    function.

    ```python
    strategy = tf.distribute.experimental.TPUStrategy(...)
    with strategy.scope():
      embedding = tf.tpu.experimental.embedding.TPUEmbedding(...)

    distributed_dataset = strategy.experimental_distribute_dataset(...)
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

    Raises:
      ValueError: When called inside a strategy.run call and input is not
        directly taken from the args of the `strategy.run` call. Also if
        the size of any sequence in `features` does not match corresponding
        sequence in `feature_config`. Similarly for `weights`, if not `None`.
      RuntimeError: When called inside a strategy.run call and inside XLA
        control flow.
      TypeError: If the type of any sequence in `features` does not match
        corresponding sequence in `feature_config`. Similarly for `weights`, if
        not `None`.
    """
    if not self._using_tpu:
      raise RuntimeError("enqueue is not valid when TPUEmbedding object is not "
                         "created under a TPUStrategy.")

    nest.assert_same_structure(self._feature_config, features)

    # TODO(bfontain): Add a check that the input batch_size matches the per core
    # batch size that this instance of the API was initialized with.

    flat_inputs = nest.flatten(features)
    flat_weights = [None] * len(flat_inputs)
    if weights is not None:
      nest.assert_same_structure(self._feature_config, weights)
      flat_weights = nest.flatten(weights)
    flat_features = nest.flatten_with_joined_string_paths(self._feature_config)

    in_tpu_context = self._raise_error_for_incorrect_control_flow_context()
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

    else:
      mode_override = "train" if training else "inference"
      # We generate enqueue ops per device, so we need to gather the all
      # features for a single device in to a dict.
      # We rely here on the fact that the devices in the PerReplica value occur
      # in the same (standard) order as self._strategy.extended.worker_devices.
      enqueue_ops = []
      for replica_id in range(self._strategy.num_replicas_in_sync):
        replica_inputs = tf_values.select_replica(replica_id, flat_inputs)
        replica_weights = tf_values.select_replica(replica_id, flat_weights)
        tpu_device = self._strategy.extended.worker_devices[replica_id]
        # TPU devices string are like /job:worker/replica:0/task:0/device:TPU:0
        # the device ordinal is the last number
        device_ordinal = int(tpu_device.rsplit(":", 1)[1])
        with ops.device(device_util.get_host_for_device(tpu_device)):
          enqueue_op = self._generate_enqueue_op(
              replica_inputs, replica_weights, flat_features,
              device_ordinal=device_ordinal, mode_override=mode_override)

          # Apply the name tag to the op.
          if name is not None:
            _add_key_attr(enqueue_op, name)
          enqueue_ops.append(enqueue_op)
      ops.get_default_graph().control_outputs.extend(enqueue_ops)


class TPUEmbeddingSaveable(saveable_hook.SaveableHook):
  """Save/Restore hook to Retrieve/Load TPUEmbedding variables."""

  def __init__(self, name, load, retrieve):
    self._load = load
    self._retrieve = retrieve
    super(TPUEmbeddingSaveable, self).__init__(name=name)

  def before_save(self):
    if self._retrieve is not None:
      self._retrieve()

  def after_restore(self):
    if self._load is not None:
      self._load()


def _ragged_embedding_lookup_with_reduce(table, ragged, weights, combiner):
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
    weights = array_ops.ones_like(ragged)
  weights = array_ops.expand_dims(weights, axis=2)
  ragged_result = embedding_ops.embedding_lookup_ragged(table, ragged)
  ragged_result = math_ops.reduce_sum(ragged_result * weights, axis=1)
  if combiner == "mean":
    ragged_result = ragged_result / math_ops.reduce_sum(weights, axis=1)
  elif combiner == "sqrtn":
    ragged_result = ragged_result, math_ops.sqrt(math_ops.reduce_sum(
        weights*weights, axis=1))
  return ragged_result


def cpu_embedding_lookup(inputs, weights, tables, feature_config):
  """Uses CPU embedding lookup for embedding ids in features.

  Args:
    inputs: a nested structure of Tensors, SparseTensors or RaggedTensors.
    weights: a nested structure of Tensors, SparseTensors or RaggedTensors or
      None for no weights.
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
    if feature.max_sequence_length > 0:
      raise ValueError("Sequence features unsupported at this time.")

    if weight is not None:
      if isinstance(inp, ops.Tensor):
        raise ValueError(
            "Weight specified for {}, but input is dense.".format(path))
      elif type(weight) is not type(inp):
        raise ValueError(
            "Weight for {} is of type {} but it does not match type of the "
            "input which is {}.".format(path, type(weight), type(inp)))

    if isinstance(inp, ops.Tensor):
      outputs.append(embedding_ops.embedding_lookup_v2(table, inp))

    elif isinstance(inp, sparse_tensor.SparseTensor):
      outputs.append(embedding_ops.safe_embedding_lookup_sparse_v2(
          table, inp, sparse_weights=weight, combiner=feature.table.combiner))

    elif isinstance(inp, ragged_tensor.RaggedTensor):
      outputs.append(_ragged_embedding_lookup_with_reduce(
          table, inp, weight, feature.table.combiner))

    else:
      raise ValueError("Input {} is type {}. Tensor, SparseTensor or "
                       "RaggedTensor expected.".format(path, type(inp)))
  return nest.pack_sequence_as(feature_config, outputs)


def get_list_of_hosts(strategy):
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


def extract_variable_info(kwargs):
  """Extracts the variable creation attributes from the kwargs.

  Args:
    kwargs: a dict of keyword arguments that were passed to a variable creator
      scope.

  Returns:
    A tuple of variable name, initialization function, shape, and dtype.

  Raises:
    ValueError: if unable to extract this information from the given keyword
      args.
  """
  if "shape" not in kwargs or kwargs["shape"] is None:
    if not isinstance(kwargs["initial_value"], functools.partial):
      raise ValueError(
          "Unable to extract initializer function and shape from {}. Please "
          "either pass a function that expects a shape and dtype as the "
          "initial value for your variable or functools.partial object with "
          "the shape and dtype kwargs set. This is needed so that we can "
          "initialize the shards of the ShardedVariable locally.".format(
              kwargs["initial_value"]))
    return (kwargs["name"], kwargs["initial_value"].keywords["shape"],
            kwargs["initial_value"].keywords.get("dtype", kwargs["dtype"]),
            kwargs["initial_value"].func)
  else:
    return (kwargs["name"], kwargs["shape"], kwargs["dtype"],
            kwargs["initial_value"])


def make_sharded_variable_creator(hosts):
  """Makes a sharded variable creator given a list of hosts.

  Args:
    hosts: a list of tensorflow devices on which to shard the tensors.

  Returns:
    A variable creator function.
  """

  def sharded_variable_creator(next_creator, *args, **kwargs):
    """The sharded variable creator."""
    kwargs["skip_mirrored_creator"] = True

    num_hosts = len(hosts)
    name, shape, dtype, initial_value = extract_variable_info(kwargs)
    rows = shape[0]
    cols = shape[1]
    missing = rows % num_hosts
    # we partition as if we were using MOD sharding.
    partitions = ([rows // num_hosts + 1] * missing + [rows // num_hosts] *
                  (num_hosts - missing))
    variables = []
    newkwargs = kwargs
    newkwargs["dtype"] = dtype
    for i, p in enumerate(partitions):
      with ops.device(hosts[i]):
        newkwargs["shape"] = (p, cols)
        newkwargs["name"] = "{}_{}".format(name, i)
        newkwargs["initial_value"] = (
            lambda: initial_value(newkwargs["shape"], dtype=dtype))
        variables.append(next_creator(*args, **kwargs))
    return TPUShardedVariable(variables, name=name)
  return sharded_variable_creator
