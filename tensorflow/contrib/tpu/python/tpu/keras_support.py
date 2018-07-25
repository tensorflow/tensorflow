# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""*Experimental* support for running Keras models on the TPU.

To use, wrap your model with the `keras_support.tpu_model` function.

Example usage:

```
image = tf.keras.layers.Input(shape=(28, 28, 3), name='image')
c1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3))( image)
flattened = tf.keras.layers.Flatten()(c1)
logits = tf.keras.layers.Dense(10, activation='softmax')(flattened)
model = tf.keras.Model(inputs=[image], outputs=[logits])

strategy = keras_support.TPUDistributionStrategy(num_cores_per_host=8)
model = keras_support.tpu_model(model,
                                strategy=strategy,
                                tpu_name_or_address=tpu_name)

# Only TF optimizers are currently supported.
model.compile(optimizer=tf.train.AdamOptimizer(), ...)

# `images` and `labels` should be Numpy arrays.  Support for tensor input
# (e.g. datasets) is planned.
model.fit(images, labels)
```
"""

# pylint: disable=protected-access

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import contextlib
import re
import sys
import time

import numpy as np

from tensorflow.contrib.cluster_resolver.python.training import tpu_cluster_resolver
from tensorflow.contrib.distribute.python import tpu_strategy
from tensorflow.contrib.framework.python.framework import experimental
from tensorflow.contrib.tpu.proto import compilation_result_pb2 as tpu_compilation_result
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as tf_session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers as keras_optimizers
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.layers import embeddings
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging

TPUDistributionStrategy = tpu_strategy.TPUStrategy  # pylint: disable=invalid-name


class TPUEmbedding(embeddings.Embedding):
  """TPU compatible embedding layer.

  The default Keras layer is not TPU compatible.  This layer is a drop-in
  replacement: it has the same behavior and will work on CPU and GPU devices.
  """

  def build(self, input_shape):
    if input_shape[0] is None:
      raise ValueError(
          'TPUEmbeddings must have a fixed input_length or input shape.')
    return super(TPUEmbedding, self).build(input_shape)

  def call(self, inputs):
    if K.dtype(inputs) != 'int32':
      inputs = math_ops.cast(inputs, 'int32')

    inputs = array_ops.one_hot(inputs, self.input_dim)
    return math_ops.tensordot(inputs, self.embeddings, 1)


class KerasCrossShardOptimizer(keras_optimizers.Optimizer):
  """An optimizer that averages gradients across TPU shards."""

  def __init__(self, opt, name='KerasCrossShardOptimizer'):
    """Construct a new cross-shard optimizer.

    Args:
      opt: An existing `Optimizer` to encapsulate.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "KerasCrossShardOptimizer".

    Raises:
      ValueError: If reduction is not a valid cross-shard reduction.
    """
    super(KerasCrossShardOptimizer, self).__init__()
    self._name = name
    self._opt = opt

  def get_updates(self, loss, params):
    logging.info('Get updates: %s', loss)
    self._opt.get_gradients = self.get_gradients
    return self._opt.get_updates(loss, params)

  def get_gradients(self, loss, params):
    num_shards = tpu_function.get_tpu_context().number_of_shards
    grads = super(KerasCrossShardOptimizer, self).get_gradients(loss, params)
    return [tpu_ops.cross_replica_sum(grad) / num_shards for grad in grads]

  def set_weights(self, weights):
    self._opt.set_weights()

  def get_weights(self):
    return self._opt.get_weights()

  @property
  def lr(self):
    return self._opt.lr


class TPUModelOp(
    collections.namedtuple('TPUModelOp', [
        'compile_op', 'execute_op', 'infeed_tensors', 'infeed_op', 'outfeed_op'
    ])):
  pass


def _valid_name(tensor_name):
  """Return a valid tensor name (strips '/', ':', etc)."""
  return re.sub('[^a-zA-Z0-9_-]+', '', tensor_name)


def _replicated_optimizer(opt):
  """Wrap the optimizer `opt` with CrossShardOptimizer if applicable."""
  if tpu_function.get_tpu_context().number_of_shards == 1:
    return opt

  if isinstance(opt, keras_optimizers.TFOptimizer):
    return tpu_optimizer.CrossShardOptimizer(opt.optimizer)
  else:
    return KerasCrossShardOptimizer(opt)


class TPURewriteContext(object):
  """Prepare the environment for a Keras model during `tpu.rewrite`.

  This overrides the default placeholder behaviour to instead refer to a preset
  input mapping.  Placeholders are unsupported in TPU compiled code, and must
  be replaced with explicit inputs or values from the infeed queue.

  Instead of explicitly threading inputs all the way through the Keras codebase,
  we override the behavior of the placeholder while compiling and inject the
  Tensors from the infeed in place of the placeholder.

  Similarly, as we compile a new sub-graph for each unique shape and execution
  mode, we need to override the behavior of an embedded `name_scope` call in
  the base Keras layer code.  This allows us to re-use the same weights across
  many compiles and share a single session/graph.
  """

  def __init__(self, input_map):
    self._input_map = input_map
    self._default_placeholder = None
    self._default_name_scope = None

  def __enter__(self):

    def _placeholder(dtype, shape=None, name=None):  # pylint: disable=unused-argument
      logging.info('Remapping placeholder for %s', name)
      if name in self._input_map:
        return self._input_map[name]
      else:
        logging.info('Default: %s', name)
        return self._default_placeholder(dtype, shape, name)

    def _name_scope(name, default_name=None, values=None):
      caller_frame = sys._getframe().f_back
      caller_obj = caller_frame.f_locals.get('self')
      if (caller_obj is not None and
          isinstance(caller_obj, base_layer.Layer) and name is not None):
        return variable_scope.variable_scope(
            name, default_name, values, reuse=variable_scope.AUTO_REUSE)

      return self._default_name_scope(name, default_name, values)

    self._default_placeholder = array_ops.placeholder
    self._default_name_scope = ops.name_scope
    self._default_make_variable = base_layer.make_variable
    self._default_random_normal = random_ops.random_normal
    self._default_qr = gen_linalg_ops.qr

    array_ops.placeholder = _placeholder

    # Replace random_ops.random_normal with a dummy function because
    # `random_normal` isn't yet implemented on the TPU. Because these
    # initialized values are overwritten by the CPU values, this is okay.
    def random_normal(shape,
                      mean=0.0,
                      stddev=1.0,
                      dtype=dtypes.float32,
                      seed=None,
                      name=None):
      del mean
      del stddev
      del seed
      return array_ops.zeros(shape, dtype=dtype, name=name)

    random_ops.random_normal = random_normal

    # Replace gen_linalg_ops.qr because QR decomposition is not yet implemented.
    # TODO(saeta): Remove qr override once we confirm the qr implementation is
    # ok.
    # pylint: disable=redefined-builtin
    def qr(input, full_matrices=False, name=None):
      """Dummy implementation of qr decomposition."""
      del full_matrices  # TODO(saeta): Properly handle the full matrix case.
      input_shape = input.shape
      if len(input_shape) < 2:
        raise ValueError('Invalid shape passed to qr: %s' % input_shape)
      p = min(input_shape[-1], input_shape[-2])
      if len(input_shape) == 2:
        q = array_ops.zeros((p, p), name=name)
        r = array_ops.zeros(input_shape, name=name)
        return (r, q)
      elif len(input_shape) == 3:
        n = input_shape[0]
        q = array_ops.zeros((n, p, p), name=name)
        r = array_ops.zeros(input_shape, name=name)
        return (r, q)
      else:
        raise ValueError('Invalid shape passed to qr: %s' % input_shape)
    gen_linalg_ops.qr = qr

    ops.name_scope = _name_scope
    base_layer.make_variable = variable_scope.get_variable
    logging.info('Overriding default placeholder.')
    return

  def __exit__(self, exc_type, exc_val, exc_tb):
    array_ops.placeholder = self._default_placeholder
    ops.name_scope = self._default_name_scope
    base_layer.make_variable = self._default_make_variable
    random_ops.random_normal = self._default_random_normal
    gen_linalg_ops.qr = self._default_qr


class SizedInfeed(collections.namedtuple('SizedInfeed',
                                         ['sharded_infeed_tensors',
                                          'infeed_ops'])):
  """Represents an instantiation of the infeed ops for a concrete input shape.

  sharded_infeed_tensors: A data structure of Tensors used to represent the
    placeholder tensors that must be fed when using feed_dicts.

  infeed_ops: the set of ops that will be run to drive infeed for a single step.
  """
  pass


class TPUInfeedInstance(object):
  """TPUInfeedInstance represents the logic to manage feeding in a single step.

  See the comments on the `TPUInfeedManager` for a description for how infeed
  is managed.
  """

  @abc.abstractmethod
  def make_input_specs(self, input_tensors):
    """Constructs the infeed_specs for the given Infeed instance.

    Args:
      input_tensors: The inputs to the model.

    Returns:
      A list of
    """
    pass

  def make_feed_dict(self, tpu_model_op):
    """Constructs a feed_dict for this instance, given the tpu_model_op.

    Args:
      tpu_model_op: A `TPUModelOp` representing the TPU Model for this
        instance's input spec.

    Returns:
      A dictionary to use as the feed_dict of a `session.run` call.
    """
    pass


class TPUInfeedManager(object):
  """TPUInfeedManager manages the data infeeding of data to a TPU computation.

  Because there are multiple data sources (e.g. in-memory NumPy arrays,
  `tf.data.Dataset`s), we abstract the different logic behind a single
  interface: the `TPUInfeedManager`.

  (1) A `TPUFunction` is called with a set of inputs. Based on the inputs,
  `TPUFunction` retrieves the corresponding `TPUInfeedManager` (or constructs a
  new one if required).

  (2) The `TPUFunction` calls `make_infeed_instance` on the `TPUInfeedManager`
  which returns a `TPUInfeedInstance`.

  (3) The `TPUFunction` checks in the shape cache for a pre-compiled instance of
  the model based on the returned `input_specs` from `TPUInfeedInstance`.

  (4) [Optional.] If the model has not already been instantiated for the given
  input spec, the `TPUFunction` compiles the model for the input spec (using the
  `TPUInfeedManager`).

  (5) The `TPUInfeedInstance` constructs the session.run's feed_dict given the
  compiled model instance corresponding to its shape.
  """

  @abc.abstractmethod
  def make_infeed_instance(self, inputs):
    """Given a single step's input, construct a `TPUInfeedInstance`.

    Args:
      inputs: The inputs to a given step.

    Returns:
      A subclass of `TPUInfeedInstance`.
    """
    pass

  @abc.abstractmethod
  def build_infeed_from_input_specs(self, input_specs, execution_mode):
    """For a given input specification (size, type), construct the infeed ops.

    This is called only once for a given input specification and builds the
    graph ops. It does not have a pointer to the actual infeed data.

    Args:
      input_specs: TODO(saeta): Document me!
      execution_mode: TODO(saeta): Document me!

    Returns:
      A `SizedInfeed` instance.
    """
    pass


class TPUNumpyInfeedManager(TPUInfeedManager):
  """TPU Infeed manager for Numpy inputs."""

  class NumpyInfeedInstance(TPUInfeedInstance):
    """Infeed instance for Numpy inputs."""

    def __init__(self, sharded_inputs):
      self._sharded_inputs = sharded_inputs

    def make_input_specs(self, input_tensors):
      # Compute an input specification (used to generate infeed enqueue and
      # dequeue operations).  We use the shape from our input array and the
      # dtype from our model.  A user may pass in a float64 for a float32
      # input: for model compatibility we still must generate a float32 infeed.
      input_specs = []
      # We use the shape and dtype from the first shard to compute the input
      # metadata (`input_specs`); all replicas have the same type and shape.
      for tensor, ary in zip(input_tensors, self._sharded_inputs[0]):
        input_specs.append(
            tensor_spec.TensorSpec(ary.shape, tensor.dtype,
                                   _valid_name(tensor.name)))

      return input_specs

    def make_feed_dict(self, tpu_model_op):
      infeed_dict = {}
      for infeed_tensors, inputs in zip(tpu_model_op.infeed_tensors,
                                        self._sharded_inputs):
        for tensor, value in zip(infeed_tensors, inputs):
          infeed_dict[tensor] = value
      return infeed_dict

  def __init__(self, distribution_strategy):
    self._strategy = distribution_strategy

  def _split_tensors(self, inputs):
    """Split input data across shards.

    Each input is sliced along the batch axis.

    Args:
      inputs: List of Numpy arrays to run on the TPU.

    Returns:
      List of lists containing the input to feed to each TPU shard.
    """
    if self._strategy.num_towers == 1:
      return [inputs]

    batch_size = inputs[0].shape[0]
    assert batch_size % self._strategy.num_towers == 0, (
        'batch_size must be divisible by strategy.num_towers (%s vs %s)' %
        (batch_size, self._strategy.num_towers))
    shard_size = batch_size // self._strategy.num_towers
    input_list = []
    for index in range(self._strategy.num_towers):
      shard_inputs = [
          x[index * shard_size:(index + 1) * shard_size] for x in inputs
      ]
      input_list.append(shard_inputs)
    return input_list

  def make_infeed_instance(self, inputs):
    sharded_inputs = self._split_tensors(inputs)
    return self.NumpyInfeedInstance(sharded_inputs)

  def build_infeed_from_input_specs(self, input_specs, execution_mode):
    infeed_op = []
    shard_infeed_tensors = []

    for shard_id in range(self._strategy.num_towers):
      with ops.device('/device:CPU:0'):
        infeed_tensors = []
        with ops.device('/device:TPU:%d' % shard_id):
          for spec in input_specs:
            # Construct placeholders for each of the inputs.
            infeed_tensors.append(
                array_ops.placeholder(
                    dtype=spec.dtype,
                    shape=spec.shape,
                    name='infeed-enqueue-%s-%d' % (spec.name, shard_id)))
        shard_infeed_tensors.append(infeed_tensors)

        infeed_op.append(
            tpu_ops.infeed_enqueue_tuple(
                infeed_tensors, [spec.shape for spec in input_specs],
                name='infeed-enqueue-%s-%d' % (execution_mode, shard_id),
                device_ordinal=shard_id))
    return SizedInfeed(infeed_ops=infeed_op,
                       sharded_infeed_tensors=shard_infeed_tensors)


class TPUDatasetInfeedManager(TPUInfeedManager):
  """Manages infeed for a `tf.data.Dataset` into a TPU computation.
  """

  class DatasetInfeedInstance(TPUInfeedInstance):
    """An instance of the TPU infeed."""

    def __init__(self, input_specs):
      self._input_specs = input_specs

    def make_input_specs(self, input_tensors):
      # TODO(saeta): Do error checking here!
      return self._input_specs

    def make_feed_dict(self, tpu_model_op):
      # TODO(saeta): Verify tpu_model_op is as expected!
      return {}

  def __init__(self, dataset, distribution_strategy, tpu_session):
    """Constructs a TPUDatasetInfeedManager.

    Must be called within a `KerasTPUModel.tpu_session` context!

    Args:
      dataset: A `tf.data.Dataset` to infeed.
      distribution_strategy: The `TPUDistributionStrategy` used to configure the
        Keras TPU model.
      tpu_session: The `tf.Session` object used for running the TPU model.
    """
    self._verify_dataset_shape(dataset)
    self._dataset = dataset
    self._strategy = distribution_strategy
    dummy_x_shape = dataset.output_shapes[0].as_list()
    dummy_x_shape[0] *= distribution_strategy.num_towers
    dummy_y_shape = dataset.output_shapes[1].as_list()
    dummy_y_shape[0] *= distribution_strategy.num_towers
    self._iterator = dataset.make_initializable_iterator()
    tpu_session.run(self._iterator.initializer)

    self._get_next_ops = []
    ctrl_deps = []
    for i in range(distribution_strategy.num_towers):
      with ops.control_dependencies(ctrl_deps):  # Ensure deterministic
        # TODO(saeta): Ensure correct placement!
        get_next_op = self._iterator.get_next()
        self._get_next_ops.append(get_next_op)
        ctrl_deps.extend(get_next_op)

    # Use dummy numpy inputs for the rest of Keras' shape checking. We
    # intercept them when building the model.
    self._dummy_x = np.zeros(dummy_x_shape,
                             dtype=dataset.output_types[0].as_numpy_dtype)
    self._dummy_y = np.zeros(dummy_y_shape,
                             dtype=dataset.output_types[1].as_numpy_dtype)

    input_specs = []
    if isinstance(self._iterator.output_shapes, tuple):
      assert isinstance(self._iterator.output_types, tuple)
      assert len(self._iterator.output_shapes) == len(
          self._iterator.output_types)
      for i in range(len(self._iterator.output_shapes)):
        spec = tensor_spec.TensorSpec(self._iterator.output_shapes[i],
                                      self._iterator.output_types[i])
        input_specs.append(spec)
    elif isinstance(self._iterator.output_shapes, tensor_shape.TensorShape):
      spec = tensor_spec.TensorSpec(self._iterator.output_shapes,
                                    self._iterator.output_types)
      input_specs.append(spec)

    self._infeed_instance = self.DatasetInfeedInstance(input_specs)

  def _verify_dataset_shape(self, dataset):
    """Verifies a dataset is of an appropriate shape for TPUs."""
    if not isinstance(dataset, dataset_ops.Dataset):
      raise ValueError('The function passed as the `x` parameter did not '
                       'return a `tf.data.Dataset`.')
    if not isinstance(dataset.output_classes, tuple):
      raise ValueError('The dataset must return a tuple of tf.Tensors, '
                       'instead it returns: %s' % dataset.output_classes)
    if len(dataset.output_classes) != 2:
      raise ValueError(
          'The dataset must return a 2-element tuple, got '
          '%s output classes instead.' % (dataset.output_classes,))
    for i, cls in enumerate(dataset.output_classes):
      if cls != ops.Tensor:
        raise ValueError('The dataset returned a non-Tensor type (%s) at '
                         'index %d.' % (cls, i))
    for i, shape in enumerate(dataset.output_shapes):
      if not shape:
        raise ValueError('The dataset returns a scalar tensor in '
                         'tuple index %d. Did you forget to batch? '
                         '(Output shapes: %s).' % (i,
                                                   dataset.output_shapes))
      for j, dim in enumerate(shape):
        if dim.value is None:
          if j == 0:
            hint = (' Hint: did you use `ds.batch(BATCH_SIZE, '
                    'drop_remainder=True)`?')
          else:
            hint = ''
          raise ValueError(
              'The Keras-TPU integration for `tf.data` '
              'currently requires static shapes. The provided '
              'dataset only has a partially defined shape. '
              '(Dimension %d of output tensor %d is not statically known '
              'for output shapes: %s.%s)' % (i, j, dataset.output_shapes, hint))

  @property
  def dummy_x(self):
    return self._dummy_x

  @property
  def dummy_y(self):
    return self._dummy_y

  def make_infeed_instance(self, inputs):
    # TODO(saeta): Verify inputs is as expected.
    return self._infeed_instance

  def build_infeed_from_input_specs(self, input_specs, execution_mode):
    shard_infeed_tensors = self._get_next_ops
    assert len(shard_infeed_tensors) == self._strategy.num_towers
    infeed_ops = []
    for shard_id in range(self._strategy.num_towers):
      with ops.device('/device:CPU:0'):
        infeed_ops.append(
            tpu_ops.infeed_enqueue_tuple(
                shard_infeed_tensors[shard_id],
                [spec.shape for spec in input_specs],
                name='infeed-enqueue-%s-%d' % (execution_mode, shard_id),
                device_ordinal=shard_id))
    return SizedInfeed(infeed_ops=infeed_ops,
                       sharded_infeed_tensors=shard_infeed_tensors)


class TPUFunction(object):
  """K.function compatible interface for invoking a TPU compiled function.

  Recompilation is triggered on-demand for each set of new inputs shapes: the
  results are cached for future execution.  We expect most computations will
  be dominated by a standard batch-size, followed by a straggler batch for
  the end of training or evaluation.

  All `inputs` and `outputs` will be loaded via the infeed and outfeed queues
  instead of being injected as `feed_dict` items or fetches.
  """

  def __init__(self, model, execution_mode, strategy):
    self.model = model
    self.execution_mode = execution_mode
    self._strategy = strategy
    self._compilation_cache = {}
    self._cloned_model = None

    # Copy optimizer configuration.  This is done prior to `_specialize_model`
    # as the configuration may require evaluating variables in the CPU session.
    self._optimizer_config = None
    if not isinstance(self.model.optimizer, keras_optimizers.TFOptimizer):
      self._optimizer_config = self.model.optimizer.get_config()

  def _specialize_model(self, input_specs, infeed_manager):
    """Specialize `self.model` (a Keras model) for the given input shapes."""
    # Re-create our input and output layers inside our subgraph.  They will be
    # attached to the true computation when we clone our model in `tpu_fn`.
    K.set_learning_phase(self.execution_mode == model_fn_lib.ModeKeys.TRAIN)

    # functools.partial and callable objects are not supported by tpu.rewrite
    def _model_fn():
      """Compute fit/eval/predict for the TPU."""
      is_training = self.execution_mode == model_fn_lib.ModeKeys.TRAIN
      is_test = self.execution_mode == model_fn_lib.ModeKeys.EVAL
      is_predict = self.execution_mode == model_fn_lib.ModeKeys.PREDICT

      # During train/eval, we infeed our features as well as labels.
      if is_training or is_test:
        infeed_layers = self.model._input_layers + self.model._output_layers
      else:
        infeed_layers = self.model._input_layers

      # Generate our infeed operation to read features & labels.
      infeed_tensors = tpu_ops.infeed_dequeue_tuple(
          dtypes=[spec.dtype for spec in input_specs],
          shapes=[spec.shape for spec in input_specs],
          name='infeed-%s' % self.execution_mode)

      assert len(infeed_tensors) == len(infeed_layers), (
          'Infeed inputs did not match model: %s vs %s' % (infeed_layers,
                                                           infeed_tensors))

      tpu_targets = []
      tpu_input_map = {}

      # Sort infeed outputs into inputs and labels for calling our Keras model.
      for tensor, layer in zip(infeed_tensors, infeed_layers):
        if layer in self.model._input_layers:
          tpu_input_map[layer.name] = tensor
        if layer in self.model._output_layers:
          tpu_targets.append(tensor)

      # Clone our CPU model, running within the TPU device context.
      with TPURewriteContext(tpu_input_map):
        # TODO(power): Replicate variables.
        with ops.device('/device:TPU:0'):
          self._cloned_model = models.clone_model(self.model)

      # Create a copy of the optimizer for this graph.
      if isinstance(self.model.optimizer, keras_optimizers.TFOptimizer):
        cloned_optimizer = keras_optimizers.TFOptimizer(
            self.model.optimizer.optimizer)
      else:
        logging.info('Cloning %s %s', self.model.optimizer.__class__.__name__,
                     self._optimizer_config)
        cloned_optimizer = self.model.optimizer.__class__.from_config(
            self._optimizer_config)

      if is_training or is_test:
        self._cloned_model.compile(
            optimizer=_replicated_optimizer(cloned_optimizer),
            loss=self.model.loss,
            loss_weights=self.model.loss_weights,
            metrics=self.model.metrics,
            weighted_metrics=self.model.weighted_metrics,
            target_tensors=tpu_targets,
        )

      # Compute our outfeed depending on the execution mode
      if is_training:
        self._cloned_model._make_train_function()
        self._outfeed_spec = [
            tensor_spec.TensorSpec(tensor.shape, tensor.dtype, tensor.name)
            for tensor in self._cloned_model.train_function.outputs
        ]
        return [
            self._cloned_model.train_function.updates_op,
            tpu_ops.outfeed_enqueue_tuple(
                self._cloned_model.train_function.outputs,
                name='outfeed-enqueue-train')
        ]
      elif is_test:
        self._cloned_model._make_test_function()
        self._outfeed_spec = [
            tensor_spec.TensorSpec(tensor.shape, tensor.dtype, tensor.name)
            for tensor in self._cloned_model.test_function.outputs
        ]
        return [
            tpu_ops.outfeed_enqueue_tuple(
                self._cloned_model.test_function.outputs,
                name='outfeed-enqueue-test')
        ]
      elif is_predict:
        self._cloned_model._make_predict_function()
        self._outfeed_spec = [
            tensor_spec.TensorSpec(tensor.shape, tensor.dtype, tensor.name)
            for tensor in self._cloned_model.predict_function.outputs
        ]
        return [
            tpu_ops.outfeed_enqueue_tuple(
                self._cloned_model.predict_function.outputs,
                name='outfeed-enqueue-predict',
            )
        ]
      else:
        assert False, 'Unexpected execution mode: %s' % self.execution_mode

    # Capture outfeed metadata computed during the rewrite.
    self._outfeed_spec = None

    # Generate out TPU operations using `tpu.split_compile_and_replicate`.
    # `compile_op` can be used to test the TPU model compiles before execution.
    # `execute op` replicates `_model_fn` `num_replicas` times, with each shard
    # running on a different logical core.
    compile_op, execute_op = tpu.split_compile_and_replicate(
        _model_fn, inputs=[[]] * self._strategy.num_towers)

    # Generate CPU side operations to enqueue features/labels and dequeue
    # outputs from the model call.
    sized_infeed = infeed_manager.build_infeed_from_input_specs(
        input_specs, self.execution_mode)
    # Build output ops.
    outfeed_op = []
    for shard_id in range(self._strategy.num_towers):
      with ops.device('/device:CPU:0'):
        outfeed_op.extend(
            tpu_ops.outfeed_dequeue_tuple(
                dtypes=[spec.dtype for spec in self._outfeed_spec],
                shapes=[spec.shape for spec in self._outfeed_spec],
                name='outfeed-dequeue-%s-%d' % (self.execution_mode, shard_id),
                device_ordinal=shard_id))

    return TPUModelOp(
        compile_op,
        execute_op,
        infeed_tensors=sized_infeed.sharded_infeed_tensors,
        infeed_op=sized_infeed.infeed_ops,
        outfeed_op=outfeed_op)

  def _test_model_compiles(self, tpu_model_ops):
    """Verifies that the given TPUModelOp can be compiled via XLA."""
    logging.info('Started compiling')
    start_time = time.clock()

    result = K.get_session().run(tpu_model_ops.compile_op)
    proto = tpu_compilation_result.CompilationResultProto()
    proto.ParseFromString(result)
    if proto.status_error_message:
      raise RuntimeError('Compilation failed: {}'.format(
          proto.status_error_message))

    end_time = time.clock()
    logging.info('Finished compiling. Time elapsed: %s secs',
                 end_time - start_time)

  def __call__(self, inputs):
    assert isinstance(inputs, list)

    infeed_manager = None
    for x, mgr in self.model._numpy_to_infeed_manager_list:
      if inputs[0] is x:
        infeed_manager = mgr
        break
    if infeed_manager is None:
      infeed_manager = TPUNumpyInfeedManager(self.model._strategy)

    # Strip sample weight from inputs
    if (self.execution_mode == model_fn_lib.ModeKeys.TRAIN or
        self.execution_mode == model_fn_lib.ModeKeys.EVAL):
      input_tensors = self.model._feed_inputs + self.model._feed_targets
      inputs = inputs[:len(input_tensors)]
    else:
      input_tensors = self.model._feed_inputs

    infeed_instance = infeed_manager.make_infeed_instance(inputs)
    del inputs  # To avoid accident usage.
    input_specs = infeed_instance.make_input_specs(input_tensors)

    # XLA requires every operation in the graph has a fixed shape.  To
    # handle varying batch sizes we recompile a new sub-graph for each
    # unique input shape.
    shape_key = tuple([tuple(spec.shape.as_list()) for spec in input_specs])

    if shape_key not in self._compilation_cache:
      with self.model.tpu_session():
        logging.info('New input shapes; (re-)compiling: mode=%s, %s',
                     self.execution_mode, input_specs)
        new_tpu_model_ops = self._specialize_model(input_specs,
                                                   infeed_manager)
        self._compilation_cache[shape_key] = new_tpu_model_ops
        self._test_model_compiles(new_tpu_model_ops)

    # Initialize our TPU weights on the first compile.
    self.model._initialize_weights(self._cloned_model)
    tpu_model_ops = self._compilation_cache[shape_key]

    infeed_dict = infeed_instance.make_feed_dict(tpu_model_ops)

    with self.model.tpu_session() as session:
      _, _, outfeed_outputs = session.run([
          tpu_model_ops.infeed_op, tpu_model_ops.execute_op,
          tpu_model_ops.outfeed_op
      ], infeed_dict)

    # TODO(xiejw): Decide how to reduce outputs, or just discard all but first.
    if self.execution_mode == model_fn_lib.ModeKeys.PREDICT:
      outputs = [[]] * len(self._outfeed_spec)
      outputs_per_replica = len(self._outfeed_spec)

      for i in range(self._strategy.num_towers):
        output_group = outfeed_outputs[i * outputs_per_replica:(i + 1) *
                                       outputs_per_replica]
        for j in range(outputs_per_replica):
          outputs[j].append(output_group[j])

      return [np.concatenate(group) for group in outputs]
    else:
      return outfeed_outputs[:len(outfeed_outputs) // self._strategy.num_towers]


class KerasTPUModel(models.Model):
  """TPU compatible Keras model wrapper."""

  def __init__(self, cpu_model, tpu_name_or_address, strategy):
    super(models.Model, self).__init__(  # pylint: disable=bad-super-call
        inputs=cpu_model.inputs,
        outputs=cpu_model.outputs,
        name=cpu_model.name,
    )

    # Create a mapping from numpy arrays to infeed managers.
    # Note: uses a list of tuples instead of a map because numpy arrays are
    # not hashable.
    self._numpy_to_infeed_manager_list = []

    self.predict_function = None
    self.test_function = None
    self.train_function = None
    self._strategy = strategy

    self._tpu_name_or_address = tpu_name_or_address
    self._cpu_model = cpu_model
    self._tpu_model = None
    self._tpu_weights_initialized = False
    self._graph = ops.Graph()

    self._cluster_resolver = tpu_cluster_resolver.TPUClusterResolver(
        tpu_name_or_address)
    master = self._cluster_resolver.master()
    cluster_spec = self._cluster_resolver.cluster_spec()
    self._session = tf_session.Session(
        graph=self._graph,
        target=master,
        config=config_pb2.ConfigProto(isolate_session_state=True))

    # TODO(saeta): Confirm the lines below work in ClusterSpec propagation env.
    if cluster_spec:
      self._session.cluster_def.CopyFrom(cluster_spec.as_cluster_def())

    with self._graph.as_default():
      self._session.run(tpu.initialize_system())

    # If the input CPU model has already been compiled, compile our TPU model
    # immediately.
    if self._cpu_model.optimizer:
      self.compile(
          self._cpu_model.optimizer,
          self._cpu_model.loss,
          self._cpu_model.metrics,
          self._cpu_model.loss_weights,
          self._cpu_model.sample_weight_mode,
          self._cpu_model.weighted_metrics,
          self._cpu_model.target_tensors,
      )

  def get_config(self):
    return {
        'cpu_model': self._cpu_model,
        'tpu_name_or_address': self._tpu_name_or_address,
        'strategy': self._strategy,
    }

  def compile(self,
              optimizer,
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              target_tensors=None,
              **kwargs):
    if sample_weight_mode:
      raise ValueError('sample_weight_mode not supported for TPU execution.')
    if weighted_metrics:
      raise ValueError('weighted_metrics not supported for TPU execution.')
    if target_tensors:
      raise ValueError('target_tensors is not supported for TPU execution.')

    super(KerasTPUModel, self).compile(optimizer, loss, metrics, loss_weights,
                                       sample_weight_mode, weighted_metrics,
                                       target_tensors, **kwargs)

    if not self._cpu_model.optimizer:
      self._cpu_model.compile(optimizer, loss, metrics, loss_weights,
                              sample_weight_mode, weighted_metrics,
                              target_tensors, **kwargs)

  def fit(self,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          **kwargs):
    assert not self._numpy_to_infeed_manager_list  # Ensure empty.

    infeed_managers = []  # Managers to clean up at the end of the fit call.
    if isinstance(x, dataset_ops.Dataset):
      # TODO(b/111413240): Support taking a tf.data.Dataset directly.
      raise ValueError(
          'Taking a Dataset directly is not yet supported. Please '
          'wrap your dataset construction code in a function and '
          'pass that to fit instead. For examples, see: '
          'https://github.com/tensorflow/tpu/tree/master/models/experimental'
          '/keras')
    if callable(x):
      with self.tpu_session() as sess:
        dataset = x()
        if steps_per_epoch is None:
          raise ValueError('When using tf.data as input to a model, you '
                           'should specify the steps_per_epoch argument.')
        if y is not None:
          raise ValueError('When using tf.data as input to a model, y must be '
                           'None')
        infeed_manager = TPUDatasetInfeedManager(dataset, self._strategy, sess)
        # Use dummy numpy inputs for the rest of Keras' shape checking. We
        # intercept them when building the model.
        x = infeed_manager.dummy_x
        y = infeed_manager.dummy_y
        infeed_managers.append((x, infeed_manager))

    if isinstance(validation_data, dataset_ops.Dataset):
      # TODO(b/111413240): Support taking a tf.data.Dataset directly.
      raise ValueError(
          'Taking a Dataset directly is not yet supported. Please '
          'wrap your dataset construction code in a function and '
          'pass that to fit instead. For examples, see: '
          'https://github.com/tensorflow/tpu/tree/master/models/experimental'
          '/keras')
    if callable(validation_data):
      with self.tpu_session() as sess:
        dataset = validation_data()
        if validation_steps is None:
          raise ValueError('When using tf.data as validation for a model, you '
                           'should specify the validation_steps argument.')
        infeed_manager = TPUDatasetInfeedManager(dataset, self._strategy, sess)
        # Use dummy numpy inputs for the rest of Keras' shape checking. We
        # intercept them when building the model.
        val_x = infeed_manager.dummy_x
        val_y = infeed_manager.dummy_y
        infeed_managers.append((val_x, infeed_manager))
        validation_data = (val_x, val_y)

    self._numpy_to_infeed_manager_list = infeed_managers
    try:
      return super(KerasTPUModel, self).fit(
          x,
          y,
          batch_size,
          epochs,
          verbose,
          callbacks,
          validation_split,
          validation_data,
          shuffle,
          class_weight,
          sample_weight,
          initial_epoch,
          steps_per_epoch,
          validation_steps,
          **kwargs)
    finally:
      self._numpy_to_infeed_manager_list = []

  def _make_train_function(self):
    if not self.train_function:
      self.train_function = TPUFunction(
          self, model_fn_lib.ModeKeys.TRAIN, strategy=self._strategy)

    return self.train_function

  def _make_test_function(self):
    if not self.test_function:
      self.test_function = TPUFunction(
          self, model_fn_lib.ModeKeys.EVAL, strategy=self._strategy)
    return self.test_function

  def _make_predict_function(self):
    if not self.predict_function:
      self.predict_function = TPUFunction(
          self, model_fn_lib.ModeKeys.PREDICT, strategy=self._strategy)
    return self.predict_function

  def _initialize_weights(self, cloned_model):
    """Initialize TPU weights.

    This is called on the first compile of the TPU model (first call to
    fit/predict/evaluate).

    Args:
      cloned_model: `keras.Model`, TPU model to initialize.
    """
    if self._tpu_weights_initialized:
      return

    self._tpu_model = cloned_model
    self._tpu_weights_initialized = True

    weights = self._cpu_model.get_weights()
    with self.tpu_session():
      logging.info('Setting weights on TPU model.')
      cloned_model.set_weights(weights)

  def sync_to_cpu(self):
    """Copy weights from the CPU, returning a synchronized CPU model."""
    if self._tpu_weights_initialized:
      with self.tpu_session():
        logging.info('Copying TPU weights to the CPU')
        tpu_weights = self._tpu_model.get_weights()

      self._cpu_model.set_weights(tpu_weights)

    return self._cpu_model

  def get_weights(self):
    return self.sync_to_cpu().get_weights()

  def save_weights(self, *args, **kw):
    return self.sync_to_cpu().save_weights(*args, **kw)

  def save(self, *args, **kw):
    return self.sync_to_cpu().save(*args, **kw)

  def set_weights(self, weights):
    # We may not have a TPU model available if we haven't run fit/predict, so
    # we can't directly set the TPU weights here.
    # Instead, reset CPU model weights and force TPU re-initialization at the
    # next call.
    self._cpu_model.set_weights(weights)
    self._tpu_weights_initialized = False

  @contextlib.contextmanager
  def tpu_session(self):
    """Yields a TPU session and sets it as the default Keras session."""
    with self._graph.as_default():
      default_session = K.get_session()
      # N.B. We have to call `K.set_session()` AND set our session as the
      # TF default. `K.get_session()` surprisingly does not return the value
      # supplied by K.set_session otherwise.
      K.set_session(self._session)
      with self._session.as_default():
        yield self._session
      K.set_session(default_session)

  def shutdown(self):
    # TODO(b/111364423): Actually shut down the system.
    logging.info('Skipping shutting down TPU system.')
    # with self.tpu_session() as session:
    #   session.run(tpu.shutdown_system())
    self._session.close()


def _validate_shapes(model):
  """Validate that all layers in `model` have constant shape."""
  for layer in model.layers:
    if isinstance(layer.input_shape, tuple):
      input_shapes = [layer.input_shape]
    else:
      input_shapes = layer.input_shape

    if isinstance(layer.output_shape, tuple):
      output_shapes = [layer.output_shape]
    else:
      output_shapes = layer.output_shape

    for shape in input_shapes + output_shapes:
      for dim in shape[1:]:
        if dim is None:
          raise ValueError(
              """
Layer %(layer)s has a variable shape in a non-batch dimension.  TPU models must
have constant shapes for all operations.

You may have to specify `input_length` for RNN/TimeDistributed layers.

Layer: %(layer)s
Input shape: %(input_shape)s
Output shape: %(output_shape)s
  """ % {
      'layer': layer,
      'input_shape': layer.input_shape,
      'output_shape': layer.output_shape
  })


@experimental
def tpu_model(model, tpu_name_or_address=None, strategy=None):
  """Copy `model` along with weights to the TPU.  Returns a TPU model.

  Usage:
  ```
  a = Input(shape=(32,))
  b = Dense(32)(a)
  model = Model(inputs=a, outputs=b)

  # If `num_cores_per_host` is greater than one, batch parallelism will be used
  # to run on multiple TPU cores.
  strategy = keras_support.TPUDistributionStrategy(num_cores_per_host=8)
  model = keras_support.tpu_model(model, strategy)
  model.compile(
      optimizer=tf.train.GradientDescentOptimizer(learning_rate=1.0),
      ...)
  model.shutdown()
  ```

  Args:
    model: A `KerasTPUModel`.
    tpu_name_or_address: A string that is either the name of the Cloud TPU,
      the grpc address of the Cloud TPU, or (Googlers only) the BNS name of the
      Cloud TPU. If tpu_name_or_address is None, the TPUClusterResolver will
      examine the environment to determine a potential Cloud TPU to use.
    strategy: `TPUDistributionStrategy`.  The strategy to use for replicating
              model across multiple TPU cores.

  Returns:
    A new `KerasTPUModel` instance.
  """
  # Force initialization of the CPU model.
  model.get_weights()
  model.reset_states()

  _validate_shapes(model)
  # TODO(xiejw): Validate TPU model. TPUModel only?
  # TODO(xiejw): Validate replicas. Full or 1. Shall we allow subset?
  # TODO(xiejw): Adds reduction option.
  if strategy is None:
    strategy = TPUDistributionStrategy(num_cores_per_host=1)
  return KerasTPUModel(
      cpu_model=model,
      tpu_name_or_address=tpu_name_or_address,
      strategy=strategy)
