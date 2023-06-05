# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=protected-access
"""Input layer code (`Input` and `InputLayer`)."""

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export


def _assert_other_arg_none(arg_name, arg):
  if arg is not None:
    raise ValueError('When `type_spec` is not None, all other args '
                     'except `name` must be None, '
                     'but %s is not None.' % arg_name)


@keras_export('keras.layers.InputLayer')
class InputLayer(base_layer.Layer):
  """Layer to be used as an entry point into a Network (a graph of layers).

  It can either wrap an existing tensor (pass an `input_tensor` argument)
  or create a placeholder tensor (pass arguments `input_shape`, and
  optionally, `dtype`).

  It is generally recommend to use the functional layer API via `Input`,
  (which creates an `InputLayer`) without directly using `InputLayer`.

  When using InputLayer with Keras Sequential model, it can be skipped by
  moving the input_shape parameter to the first layer after the InputLayer.

  This class can create placeholders for tf.Tensors, tf.SparseTensors, and
  tf.RaggedTensors by choosing 'sparse=True' or 'ragged=True'. Note that
  'sparse' and 'ragged' can't be configured to True at same time.
  Usage:

  ```python
  # With explicit InputLayer.
  model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(4,)),
    tf.keras.layers.Dense(8)])
  model.compile(tf.optimizers.RMSprop(0.001), loss='mse')
  model.fit(np.zeros((10, 4)),
            np.ones((10, 8)))

  # Without InputLayer and let the first layer to have the input_shape.
  # Keras will add a input for the model behind the scene.
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_shape=(4,))])
  model.compile(tf.optimizers.RMSprop(0.001), loss='mse')
  model.fit(np.zeros((10, 4)),
            np.ones((10, 8)))
  ```

  Args:
      input_shape: Shape tuple (not including the batch axis), or `TensorShape`
        instance (not including the batch axis).
      batch_size: Optional input batch size (integer or None).
      dtype: Optional datatype of the input. When not provided, the Keras
          default float type will be used.
      input_tensor: Optional tensor to use as layer input. If set, the layer
          will use the `tf.TypeSpec` of this tensor rather
          than creating a new placeholder tensor.
      sparse: Boolean, whether the placeholder created is meant to be sparse.
          Default to False.
      ragged: Boolean, whether the placeholder created is meant to be ragged.
          In this case, values of 'None' in the 'shape' argument represent
          ragged dimensions. For more information about RaggedTensors, see
          [this guide](https://www.tensorflow.org/guide/ragged_tensors).
          Default to False.
      type_spec: A `tf.TypeSpec` object to create Input from. This `tf.TypeSpec`
          represents the entire batch. When provided, all other args except
          name must be None.
      name: Optional name of the layer (string).
  """

  def __init__(self,
               input_shape=None,
               batch_size=None,
               dtype=None,
               input_tensor=None,
               sparse=None,
               name=None,
               ragged=None,
               type_spec=None,
               **kwargs):
    self._init_input_shape = input_shape
    self._init_batch_size = batch_size
    self._init_dtype = dtype
    self._init_sparse = sparse
    self._init_ragged = ragged
    self._init_type_spec = type_spec

    strategy = distribute_lib.get_strategy()
    if strategy and batch_size is not None and \
        distributed_training_utils.global_batch_size_supported(strategy):
      if batch_size % strategy.num_replicas_in_sync != 0:
        raise ValueError('The `batch_size` argument ({}) must be divisible by '
                         'the number of replicas ({})'.format(
                             batch_size, strategy.num_replicas_in_sync))
      batch_size = batch_size // strategy.num_replicas_in_sync

    if 'batch_input_shape' in kwargs:
      batch_input_shape = kwargs.pop('batch_input_shape')
      if input_shape and batch_input_shape:
        raise ValueError('Only provide the input_shape OR '
                         'batch_input_shape argument to '
                         'InputLayer, not both at the same time.')
      # Set the input shape and batch size from the batch_input_shape.
      # Note that batch_input_shape can be None (unknown rank) or [] (scalar),
      # in which case the batch size must be None.
      if batch_input_shape:
        batch_size = batch_input_shape[0]
        input_shape = batch_input_shape[1:]
    if kwargs:
      raise ValueError('Unrecognized keyword arguments:', kwargs.keys())

    if sparse and ragged:
      raise ValueError(
          'Cannot set both sparse and ragged to True in a Keras input.')

    if not name:
      prefix = 'input'
      name = prefix + '_' + str(backend.get_uid(prefix))

    if not dtype:
      if input_tensor is None:
        dtype = backend.floatx()
      else:
        dtype = backend.dtype(input_tensor)
    elif input_tensor is not None and input_tensor.dtype != dtype:
      raise ValueError('`input_tensor.dtype` differs from `dtype`: %s vs. %s' %
                       (input_tensor.dtype, dtype))
    super(InputLayer, self).__init__(dtype=dtype, name=name)
    self.built = True
    self.sparse = True if sparse else False
    self.ragged = True if ragged else False
    self.batch_size = batch_size
    self.supports_masking = True

    if isinstance(input_shape, tensor_shape.TensorShape):
      input_shape = tuple(input_shape.as_list())
    elif isinstance(input_shape, int):
      input_shape = (input_shape,)

    if type_spec is not None:
      args_that_must_be_none = [
          ('(input_)shape', self._init_input_shape),
          ('batch_size', self._init_batch_size),
          ('dtype', self._init_dtype),
          ('input_tensor', input_tensor),
          ('sparse', self._init_sparse),
          ('ragged', self._init_ragged),
      ]
      for arg_name, arg in args_that_must_be_none:
        _assert_other_arg_none(arg_name, arg)
      if not ops.executing_eagerly_outside_functions():
        raise ValueError('Creating Keras inputs from a type_spec is only '
                         'supported when eager execution is enabled.')
      input_tensor = keras_tensor.keras_tensor_from_type_spec(type_spec)
      if isinstance(input_tensor, keras_tensor.SparseKerasTensor):
        self.sparse = True
      if isinstance(input_tensor, keras_tensor.RaggedKerasTensor):
        self.ragged = True
      self.is_placeholder = True
      try:
        self._batch_input_shape = tuple(input_tensor.shape.as_list())
      except ValueError:
        # If the shape cannot be represented as a tuple (e.g. unknown rank)
        self._batch_input_shape = None
    elif input_tensor is None:
      if input_shape is not None:
        batch_input_shape = (batch_size,) + tuple(input_shape)
      else:
        batch_input_shape = None
      graph = backend.get_graph()
      with graph.as_default():
        input_tensor = backend.placeholder(
            shape=batch_input_shape,
            dtype=dtype,
            name=self.name,
            sparse=sparse,
            ragged=ragged)

      self.is_placeholder = True
      self._batch_input_shape = batch_input_shape
    else:
      if ops.executing_eagerly_outside_functions():
        if not isinstance(input_tensor, keras_tensor.KerasTensor):
          input_tensor = keras_tensor.keras_tensor_from_tensor(input_tensor)
      else:
        if not tf_utils.is_symbolic_tensor(input_tensor):
          raise ValueError('You should not pass an EagerTensor to `Input`. '
                           'For example, instead of creating an '
                           'InputLayer, you should instantiate your model and '
                           'directly call it on your input.')
      self.is_placeholder = False
      try:
        self._batch_input_shape = tuple(input_tensor.shape.as_list())
      except ValueError:
        # If the shape cannot be represented as a tuple (e.g. unknown rank)
        self._batch_input_shape = None
    # Create an input node.
    input_tensor._keras_mask = None
    node_module.Node(layer=self, outputs=input_tensor)

    # Store type spec
    if isinstance(input_tensor, keras_tensor.KerasTensor) or (
        tf_utils.is_extension_type(input_tensor)):
      self._type_spec = input_tensor._type_spec  # pylint: disable=protected-access
    else:
      self._type_spec = tensor_spec.TensorSpec(
          shape=input_tensor.shape, dtype=input_tensor.dtype, name=self.name)

  def get_config(self):
    if self._init_type_spec is not None:
      config = {
          'name': self.name,
          'type_spec': self._init_type_spec
      }
    else:
      config = {
          'batch_input_shape': self._batch_input_shape,
          'dtype': self.dtype,
          'sparse': self.sparse,
          'ragged': self.ragged,
          'name': self.name,
      }
    return config

  @property
  def _trackable_saved_model_saver(self):
    return layer_serialization.InputLayerSavedModelSaver(self)


@keras_export('keras.Input', 'keras.layers.Input')
def Input(  # pylint: disable=invalid-name
    shape=None,
    batch_size=None,
    name=None,
    dtype=None,
    sparse=None,
    tensor=None,
    ragged=None,
    type_spec=None,
    **kwargs):
  """`Input()` is used to instantiate a Keras tensor.

  A Keras tensor is a symbolic tensor-like object,
  which we augment with certain attributes that allow us to build a Keras model
  just by knowing the inputs and outputs of the model.

  For instance, if `a`, `b` and `c` are Keras tensors,
  it becomes possible to do:
  `model = Model(input=[a, b], output=c)`

  Args:
      shape: A shape tuple (integers), not including the batch size.
          For instance, `shape=(32,)` indicates that the expected input
          will be batches of 32-dimensional vectors. Elements of this tuple
          can be None; 'None' elements represent dimensions where the shape is
          not known.
      batch_size: optional static batch size (integer).
      name: An optional name string for the layer.
          Should be unique in a model (do not reuse the same name twice).
          It will be autogenerated if it isn't provided.
      dtype: The data type expected by the input, as a string
          (`float32`, `float64`, `int32`...)
      sparse: A boolean specifying whether the placeholder to be created is
          sparse. Only one of 'ragged' and 'sparse' can be True. Note that,
          if `sparse` is False, sparse tensors can still be passed into the
          input - they will be densified with a default value of 0.
      tensor: Optional existing tensor to wrap into the `Input` layer.
          If set, the layer will use the `tf.TypeSpec` of this tensor rather
          than creating a new placeholder tensor.
      ragged: A boolean specifying whether the placeholder to be created is
          ragged. Only one of 'ragged' and 'sparse' can be True. In this case,
          values of 'None' in the 'shape' argument represent ragged dimensions.
          For more information about RaggedTensors, see
          [this guide](https://www.tensorflow.org/guide/ragged_tensors).
      type_spec: A `tf.TypeSpec` object to create the input placeholder from.
          When provided, all other args except name must be None.
      **kwargs: deprecated arguments support. Supports `batch_shape` and
          `batch_input_shape`.

  Returns:
    A `tensor`.

  Example:

  ```python
  # this is a logistic regression in Keras
  x = Input(shape=(32,))
  y = Dense(16, activation='softmax')(x)
  model = Model(x, y)
  ```

  Note that even if eager execution is enabled,
  `Input` produces a symbolic tensor-like object (i.e. a placeholder).
  This symbolic tensor-like object can be used with lower-level
  TensorFlow ops that take tensors as inputs, as such:

  ```python
  x = Input(shape=(32,))
  y = tf.square(x)  # This op will be treated like a layer
  model = Model(x, y)
  ```

  (This behavior does not work for higher-order TensorFlow APIs such as
  control flow and being directly watched by a `tf.GradientTape`).

  However, the resulting model will not track any variables that were
  used as inputs to TensorFlow ops. All variable usages must happen within
  Keras layers to make sure they will be tracked by the model's weights.

  The Keras Input can also create a placeholder from an arbitrary `tf.TypeSpec`,
  e.g:

  ```python
  x = Input(type_spec=tf.RaggedTensorSpec(shape=[None, None],
                                          dtype=tf.float32, ragged_rank=1))
  y = x.values
  model = Model(x, y)
  ```
  When passing an arbitrary `tf.TypeSpec`, it must represent the signature of an
  entire batch instead of just one example.

  Raises:
    ValueError: If both `sparse` and `ragged` are provided.
    ValueError: If both `shape` and (`batch_input_shape` or `batch_shape`) are
      provided.
    ValueError: If `shape`, `tensor` and `type_spec` are None.
    ValueError: If arguments besides `type_spec` are non-None while `type_spec`
                is passed.
    ValueError: if any unrecognized parameters are provided.
  """
  if sparse and ragged:
    raise ValueError(
        'Cannot set both sparse and ragged to True in a Keras input.')

  input_layer_config = {'name': name, 'dtype': dtype, 'sparse': sparse,
                        'ragged': ragged, 'input_tensor': tensor,
                        'type_spec': type_spec}

  batch_input_shape = kwargs.pop('batch_input_shape',
                                 kwargs.pop('batch_shape', None))
  if shape is not None and batch_input_shape is not None:
    raise ValueError('Only provide the `shape` OR `batch_input_shape` argument '
                     'to Input, not both at the same time.')
  if (batch_input_shape is None and shape is None and tensor is None
      and type_spec is None):
    raise ValueError('Please provide to Input a `shape`'
                     ' or a `tensor` or a `type_spec` argument. Note that '
                     '`shape` does not include the batch '
                     'dimension.')
  if kwargs:
    raise ValueError('Unrecognized keyword arguments:', kwargs.keys())

  if batch_input_shape:
    shape = batch_input_shape[1:]
    input_layer_config.update({'batch_input_shape': batch_input_shape})
  else:
    input_layer_config.update(
        {'batch_size': batch_size, 'input_shape': shape})
  input_layer = InputLayer(**input_layer_config)

  # Return tensor including `_keras_history`.
  # Note that in this case train_output and test_output are the same pointer.
  outputs = input_layer._inbound_nodes[0].outputs
  if isinstance(outputs, list) and len(outputs) == 1:
    return outputs[0]
  else:
    return outputs
