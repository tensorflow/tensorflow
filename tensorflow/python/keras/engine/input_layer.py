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
"""Input layer code (`Input` and `InputLayer`).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export('keras.layers.InputLayer')
class InputLayer(base_layer.Layer):
  """Layer to be used as an entry point into a Network (a graph of layers).

  It can either wrap an existing tensor (pass an `input_tensor` argument)
  or create its a placeholder tensor (pass arguments `input_shape`, and
  optionally, `dtype`).

  It is generally recommend to use the functional layer API via `Input`,
  (which creates an `InputLayer`) without directly using `InputLayer`.

  Arguments:
      input_shape: Shape tuple (not including the batch axis), or `TensorShape`
        instance (not including the batch axis).
      batch_size: Optional input batch size (integer or None).
      dtype: Datatype of the input.
      input_tensor: Optional tensor to use as layer input
          instead of creating a placeholder.
      sparse: Boolean, whether the placeholder created
          is meant to be sparse.
      name: Name of the layer (string).
  """

  def __init__(self,
               input_shape=None,
               batch_size=None,
               dtype=None,
               input_tensor=None,
               sparse=False,
               name=None,
               **kwargs):
    if 'batch_input_shape' in kwargs:
      batch_input_shape = kwargs.pop('batch_input_shape')
      if input_shape and batch_input_shape:
        raise ValueError('Only provide the input_shape OR '
                         'batch_input_shape argument to '
                         'InputLayer, not both at the same time.')
      batch_size = batch_input_shape[0]
      input_shape = batch_input_shape[1:]
    if kwargs:
      raise ValueError('Unrecognized keyword arguments:', kwargs.keys())

    if not name:
      prefix = 'input'
      name = prefix + '_' + str(backend.get_uid(prefix))

    if not dtype:
      if input_tensor is None:
        dtype = backend.floatx()
      else:
        dtype = backend.dtype(input_tensor)
    super(InputLayer, self).__init__(dtype=dtype, name=name)
    self.built = True
    self.sparse = sparse
    self.batch_size = batch_size
    self.supports_masking = True

    if isinstance(input_shape, tensor_shape.TensorShape):
      input_shape = tuple(input_shape.as_list())

    if input_tensor is None:
      if input_shape is not None:
        batch_input_shape = (batch_size,) + tuple(input_shape)
      else:
        batch_input_shape = None
      graph = backend.get_graph()
      with context.graph_mode():
        with graph.as_default():
          # In graph mode, create a graph placeholder to call the layer on.
          if sparse:
            input_tensor = array_ops.sparse_placeholder(
                shape=batch_input_shape,
                dtype=dtype,
                name=self.name)
          else:
            input_tensor = array_ops.placeholder(
                shape=batch_input_shape,
                dtype=dtype,
                name=self.name)

      self.is_placeholder = True
      self._batch_input_shape = batch_input_shape
    else:
      if not tf_utils.is_symbolic_tensor(input_tensor):
        raise ValueError('You should not pass an EagerTensor to `Input`. '
                         'For example, instead of creating an '
                         'InputLayer, you should instantiate your model and '
                         'directly call it on your input.')
      self.is_placeholder = False
      self._batch_input_shape = tuple(input_tensor.get_shape().as_list())

    # Create an input node to add to self.outbound_node
    # and set output_tensors' _keras_history.
    input_tensor._keras_history = (self, 0, 0)  # pylint: disable=protected-access
    base_layer.Node(
        self,
        inbound_layers=[],
        node_indices=[],
        tensor_indices=[],
        input_tensors=[input_tensor],
        output_tensors=[input_tensor])

  def get_config(self):
    config = {
        'batch_input_shape': self._batch_input_shape,
        'dtype': self.dtype,
        'sparse': self.sparse,
        'name': self.name
    }
    return config


@tf_export('keras.layers.Input', 'keras.Input')
def Input(  # pylint: disable=invalid-name
    shape=None,
    batch_size=None,
    name=None,
    dtype=None,
    sparse=False,
    tensor=None,
    **kwargs):
  """`Input()` is used to instantiate a Keras tensor.

  A Keras tensor is a tensor object from the underlying backend
  (Theano or TensorFlow), which we augment with certain
  attributes that allow us to build a Keras model
  just by knowing the inputs and outputs of the model.

  For instance, if a, b and c are Keras tensors,
  it becomes possible to do:
  `model = Model(input=[a, b], output=c)`

  The added Keras attribute is:
      `_keras_history`: Last layer applied to the tensor.
          the entire layer graph is retrievable from that layer,
          recursively.

  Arguments:
      shape: A shape tuple (integers), not including the batch size.
          For instance, `shape=(32,)` indicates that the expected input
          will be batches of 32-dimensional vectors.
      batch_size: optional static batch size (integer).
      name: An optional name string for the layer.
          Should be unique in a model (do not reuse the same name twice).
          It will be autogenerated if it isn't provided.
      dtype: The data type expected by the input, as a string
          (`float32`, `float64`, `int32`...)
      sparse: A boolean specifying whether the placeholder
          to be created is sparse.
      tensor: Optional existing tensor to wrap into the `Input` layer.
          If set, the layer will not create a placeholder tensor.
      **kwargs: deprecated arguments support.

  Returns:
      A tensor.

  Example:

      ```python
      # this is a logistic regression in Keras
      x = Input(shape=(32,))
      y = Dense(16, activation='softmax')(x)
      model = Model(x, y)
      ```

      Note that even if eager execution is enabled,
      `Input` produces a symbolic tensor (i.e. a placeholder).
      This symbolic tensor can be used with other
      TensorFlow ops, as such:

      ```python
      x = Input(shape=(32,))
      y = tf.square(x)
      ```

  Raises:
    ValueError: in case of invalid arguments.
  """
  if 'batch_shape' in kwargs:
    batch_shape = kwargs.pop('batch_shape')
    if shape and batch_shape:
      raise ValueError('Only provide the shape OR '
                       'batch_shape argument to '
                       'Input, not both at the same time.')
    batch_size = batch_shape[0]
    shape = batch_shape[1:]
  if kwargs:
    raise ValueError('Unrecognized keyword arguments:', kwargs.keys())

  if dtype is None:
    dtype = backend.floatx()
  if shape is None and tensor is None:
    raise ValueError('Please provide to Input either a `shape`'
                     ' or a `tensor` argument. Note that '
                     '`shape` does not include the batch '
                     'dimension.')
  input_layer = InputLayer(
      input_shape=shape,
      batch_size=batch_size,
      name=name,
      dtype=dtype,
      sparse=sparse,
      input_tensor=tensor)
  # Return tensor including `_keras_history`.
  # Note that in this case train_output and test_output are the same pointer.
  outputs = input_layer._inbound_nodes[0].output_tensors
  if len(outputs) == 1:
    return outputs[0]
  else:
    return outputs
