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
# pylint: disable=redefined-outer-name
# pylint: disable=redefined-builtin
"""Keras backend API.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_module
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.layers import base as tf_base_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients as gradients_module
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.training import moving_averages
from tensorflow.python.util import tf_inspect


py_all = all
py_sum = sum

# INTERNAL UTILS

# This is the default internal TF session used by Keras.
# It can be set manually via `set_session(sess)`.
_SESSION = None

# This dictionary holds a mapping {graph: learning_phase}.
# A learning phase is a bool tensor used to run Keras models in
# either train mode (learning_phase == 1) or test mode (learning_phase == 0).
_GRAPH_LEARNING_PHASES = {}

# This dictionary holds a mapping {graph: UID_DICT}.
# each UID_DICT is a dictionary mapping name prefixes to a current index,
# used for generatic graph-specific string UIDs
# for various names (e.g. layer names).
_GRAPH_UID_DICTS = {}

# This boolean flag can be set to True to leave variable initialization
# up to the user.
# Change its value via `manual_variable_initialization(value)`.
_MANUAL_VAR_INIT = False

# The type of float to use throughout a session.
_FLOATX = 'float32'

# Epsilon fuzz factor used throughout the codebase.
_EPSILON = 10e-8

# Default image data format, one of "channels_last", "channels_first".
_IMAGE_DATA_FORMAT = 'channels_last'


def backend():
  """Publicly accessible method for determining the current backend.

  Only exists for API compatibility with multi-backend Keras.

  Returns:
      The string "tensorflow".
  """
  return 'tensorflow'


def epsilon():
  """Returns the value of the fuzz factor used in numeric expressions.

  Returns:
      A float.

  Example:
  ```python
      >>> keras.backend.epsilon()
      1e-08
  ```
  """
  return _EPSILON


def set_epsilon(value):
  """Sets the value of the fuzz factor used in numeric expressions.

  Arguments:
      value: float. New value of epsilon.

  Example:
  ```python
      >>> from keras import backend as K
      >>> K.epsilon()
      1e-08
      >>> K.set_epsilon(1e-05)
      >>> K.epsilon()
      1e-05
  ```
  """
  global _EPSILON
  _EPSILON = value


def floatx():
  """Returns the default float type, as a string.

  E.g. 'float16', 'float32', 'float64'.

  Returns:
      String, the current default float type.

  Example:
  ```python
      >>> keras.backend.floatx()
      'float32'
  ```
  """
  return _FLOATX


def set_floatx(value):
  """Sets the default float type.

  Arguments:
      value: String; 'float16', 'float32', or 'float64'.

  Example:
  ```python
      >>> from keras import backend as K
      >>> K.floatx()
      'float32'
      >>> K.set_floatx('float16')
      >>> K.floatx()
      'float16'
  ```

  Raises:
      ValueError: In case of invalid value.
  """
  global _FLOATX
  if value not in {'float16', 'float32', 'float64'}:
    raise ValueError('Unknown floatx type: ' + str(value))
  _FLOATX = str(value)


def cast_to_floatx(x):
  """Cast a Numpy array to the default Keras float type.

  Arguments:
      x: Numpy array.

  Returns:
      The same Numpy array, cast to its new type.

  Example:
  ```python
      >>> from keras import backend as K
      >>> K.floatx()
      'float32'
      >>> arr = numpy.array([1.0, 2.0], dtype='float64')
      >>> arr.dtype
      dtype('float64')
      >>> new_arr = K.cast_to_floatx(arr)
      >>> new_arr
      array([ 1.,  2.], dtype=float32)
      >>> new_arr.dtype
      dtype('float32')
  ```
  """
  return np.asarray(x, dtype=_FLOATX)


def image_data_format():
  """Returns the default image data format convention.

  Returns:
      A string, either `'channels_first'` or `'channels_last'`

  Example:
  ```python
      >>> keras.backend.image_data_format()
      'channels_first'
  ```
  """
  return _IMAGE_DATA_FORMAT


def set_image_data_format(data_format):
  """Sets the value of the image data format convention.

  Arguments:
      data_format: string. `'channels_first'` or `'channels_last'`.

  Example:
  ```python
      >>> from keras import backend as K
      >>> K.image_data_format()
      'channels_first'
      >>> K.set_image_data_format('channels_last')
      >>> K.image_data_format()
      'channels_last'
  ```

  Raises:
      ValueError: In case of invalid `data_format` value.
  """
  global _IMAGE_DATA_FORMAT
  if data_format not in {'channels_last', 'channels_first'}:
    raise ValueError('Unknown data_format:', data_format)
  _IMAGE_DATA_FORMAT = str(data_format)


def get_uid(prefix=''):
  """Associates a string prefix with an integer counter in a TensorFlow graph.

  Arguments:
    prefix: String prefix to index.

  Returns:
    Unique integer ID.

  Example:

  ```
    >>> get_uid('dense')
    1
    >>> get_uid('dense')
    2
  ```
  """
  graph = ops.get_default_graph()
  layer_name_uids = tf_base_layers.PER_GRAPH_LAYER_NAME_UIDS[graph]
  layer_name_uids[prefix] += 1
  return layer_name_uids[prefix]


def reset_uids():
  layer_name_uids_collection = ops.get_collection_ref('LAYER_NAME_UIDS')
  if layer_name_uids_collection:
    layer_name_uids_collection.pop()


def clear_session():
  """Destroys the current TF graph and creates a new one.

  Useful to avoid clutter from old models / layers.
  """
  global _SESSION
  global _GRAPH_LEARNING_PHASES  # pylint: disable=global-variable-not-assigned
  ops.reset_default_graph()
  reset_uids()
  _SESSION = None
  phase = array_ops.placeholder(dtype='bool', name='keras_learning_phase')
  _GRAPH_LEARNING_PHASES = {}
  _GRAPH_LEARNING_PHASES[ops.get_default_graph()] = phase


def manual_variable_initialization(value):
  """Sets the manual variable initialization flag.

  This boolean flag determines whether
  variables should be initialized
  as they are instantiated (default), or if
  the user should handle the initialization
  (e.g. via `tf.initialize_all_variables()`).

  Arguments:
      value: Python boolean.
  """
  global _MANUAL_VAR_INIT
  _MANUAL_VAR_INIT = value


def learning_phase():
  """Returns the learning phase flag.

  The learning phase flag is a bool tensor (0 = test, 1 = train)
  to be passed as input to any Keras function
  that uses a different behavior at train time and test time.

  Returns:
      Learning phase (scalar integer tensor or Python integer).
  """
  graph = ops.get_default_graph()
  if graph not in _GRAPH_LEARNING_PHASES:
    phase = array_ops.placeholder(dtype='bool', name='keras_learning_phase')
    _GRAPH_LEARNING_PHASES[graph] = phase
  return _GRAPH_LEARNING_PHASES[graph]


def set_learning_phase(value):
  """Sets the learning phase to a fixed value.

  Arguments:
      value: Learning phase value, either 0 or 1 (integers).

  Raises:
      ValueError: if `value` is neither `0` nor `1`.
  """
  global _GRAPH_LEARNING_PHASES  # pylint: disable=global-variable-not-assigned
  if value not in {0, 1}:
    raise ValueError('Expected learning phase to be ' '0 or 1.')
  _GRAPH_LEARNING_PHASES[ops.get_default_graph()] = value


def get_session():
  """Returns the TF session to be used by the backend.

  If a default TensorFlow session is available, we will return it.

  Else, we will return the global Keras session.

  If no global Keras session exists at this point:
  we will create a new global session.

  Note that you can manually set the global session
  via `K.set_session(sess)`.

  Returns:
      A TensorFlow session.
  """
  global _SESSION
  if ops.get_default_session() is not None:
    session = ops.get_default_session()
  else:
    if _SESSION is None:
      if not os.environ.get('OMP_NUM_THREADS'):
        config = config_pb2.ConfigProto(allow_soft_placement=True)
      else:
        num_thread = int(os.environ.get('OMP_NUM_THREADS'))
        config = config_pb2.ConfigProto(
            intra_op_parallelism_threads=num_thread, allow_soft_placement=True)
      _SESSION = session_module.Session(config=config)
    session = _SESSION
  if not _MANUAL_VAR_INIT:
    with session.graph.as_default():
      _initialize_variables()
  return session


def set_session(session):
  """Sets the global TensorFlow session.

  Arguments:
      session: A TF Session.
  """
  global _SESSION
  _SESSION = session


# VARIABLE MANIPULATION


def _convert_string_dtype(dtype):
  """Get the type from a string.

  Arguments:
      dtype: A string representation of a type.

  Returns:
      The type requested.

  Raises:
      ValueError: if `dtype` is not supported.
  """
  if dtype == 'float16':
    return dtypes_module.float16
  if dtype == 'float32':
    return dtypes_module.float32
  elif dtype == 'float64':
    return dtypes_module.float64
  elif dtype == 'int16':
    return dtypes_module.int16
  elif dtype == 'int32':
    return dtypes_module.int32
  elif dtype == 'int64':
    return dtypes_module.int64
  elif dtype == 'uint8':
    return dtypes_module.int8
  elif dtype == 'uint16':
    return dtypes_module.uint16
  else:
    raise ValueError('Unsupported dtype:', dtype)


def _to_tensor(x, dtype):
  """Convert the input `x` to a tensor of type `dtype`.

  Arguments:
      x: An object to be converted (numpy array, list, tensors).
      dtype: The destination type.

  Returns:
      A tensor.
  """
  x = ops.convert_to_tensor(x)
  if x.dtype != dtype:
    x = math_ops.cast(x, dtype)
  return x


def is_sparse(tensor):
  """Returns whether a tensor is a sparse tensor.

  Arguments:
      tensor: A tensor instance.

  Returns:
      A boolean.

  Example:
  ```python
      >>> from keras import backend as K
      >>> a = K.placeholder((2, 2), sparse=False)
      >>> print(K.is_sparse(a))
      False
      >>> b = K.placeholder((2, 2), sparse=True)
      >>> print(K.is_sparse(b))
      True
  ```
  """
  return isinstance(tensor, sparse_tensor.SparseTensor)


def to_dense(tensor):
  """Converts a sparse tensor into a dense tensor and returns it.

  Arguments:
      tensor: A tensor instance (potentially sparse).

  Returns:
      A dense tensor.

  Examples:
  ```python
      >>> from keras import backend as K
      >>> b = K.placeholder((2, 2), sparse=True)
      >>> print(K.is_sparse(b))
      True
      >>> c = K.to_dense(b)
      >>> print(K.is_sparse(c))
      False
  ```
  """
  if is_sparse(tensor):
    return sparse_ops.sparse_tensor_to_dense(tensor)
  else:
    return tensor


name_scope = ops.name_scope


def variable(value, dtype=None, name=None):
  """Instantiates a variable and returns it.

  Arguments:
      value: Numpy array, initial value of the tensor.
      dtype: Tensor type.
      name: Optional name string for the tensor.

  Returns:
      A variable instance (with Keras metadata included).

  Examples:
  ```python
      >>> from keras import backend as K
      >>> val = np.array([[1, 2], [3, 4]])
      >>> kvar = K.variable(value=val, dtype='float64', name='example_var')
      >>> K.dtype(kvar)
      'float64'
      >>> print(kvar)
      example_var
      >>> kvar.eval()
      array([[ 1.,  2.],
             [ 3.,  4.]])
  ```
  """
  if dtype is None:
    dtype = floatx()
  if hasattr(value, 'tocoo'):
    sparse_coo = value.tocoo()
    indices = np.concatenate((np.expand_dims(sparse_coo.row, 1), np.expand_dims(
        sparse_coo.col, 1)), 1)
    v = sparse_tensor.SparseTensor(
        indices=indices, values=sparse_coo.data, dense_shape=sparse_coo.shape)
    v._uses_learning_phase = False
    return v
  v = variables_module.Variable(
      value, dtype=_convert_string_dtype(dtype), name=name)
  v._uses_learning_phase = False
  return v


def _initialize_variables():
  """Utility to initialize uninitialized variables on the fly.
  """
  variables = variables_module.global_variables()
  uninitialized_variables = []
  for v in variables:
    if not hasattr(v, '_keras_initialized') or not v._keras_initialized:
      uninitialized_variables.append(v)
      v._keras_initialized = True
  if uninitialized_variables:
    sess = get_session()
    sess.run(variables_module.variables_initializer(uninitialized_variables))


def constant(value, dtype=None, shape=None, name=None):
  """Creates a constant tensor.

  Arguments:
      value: A constant value (or list)
      dtype: The type of the elements of the resulting tensor.
      shape: Optional dimensions of resulting tensor.
      name: Optional name for the tensor.

  Returns:
      A Constant Tensor.
  """
  if dtype is None:
    dtype = floatx()
  return constant_op.constant(value, dtype=dtype, shape=shape, name=name)


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
  """Instantiates a placeholder tensor and returns it.

  Arguments:
      shape: Shape of the placeholder
          (integer tuple, may include `None` entries).
      ndim: Number of axes of the tensor.
          At least one of {`shape`, `ndim`} must be specified.
          If both are specified, `shape` is used.
      dtype: Placeholder type.
      sparse: Boolean, whether the placeholder should have a sparse type.
      name: Optional name string for the placeholder.

  Returns:
      Tensor instance (with Keras metadata included).

  Examples:
  ```python
      >>> from keras import backend as K
      >>> input_ph = K.placeholder(shape=(2, 4, 5))
      >>> input_ph
      <tf.Tensor 'Placeholder_4:0' shape=(2, 4, 5) dtype=float32>
  ```
  """
  if dtype is None:
    dtype = floatx()
  if not shape:
    if ndim:
      shape = tuple([None for _ in range(ndim)])
  if sparse:
    x = array_ops.sparse_placeholder(dtype, shape=shape, name=name)
  else:
    x = array_ops.placeholder(dtype, shape=shape, name=name)
  x._uses_learning_phase = False
  return x


def shape(x):
  """Returns the symbolic shape of a tensor or variable.

  Arguments:
      x: A tensor or variable.

  Returns:
      A symbolic shape (which is itself a tensor).

  Examples:
  ```
      # TensorFlow example
      >>> from keras import backend as K
      >>> tf_session = K.get_session()
      >>> val = np.array([[1, 2], [3, 4]])
      >>> kvar = K.variable(value=val)
      >>> input = keras.backend.placeholder(shape=(2, 4, 5))
      >>> K.shape(kvar)
      <tf.Tensor 'Shape_8:0' shape=(2,) dtype=int32>
      >>> K.shape(input)
      <tf.Tensor 'Shape_9:0' shape=(3,) dtype=int32>
      # To get integer shape (Instead, you can use K.int_shape(x))
      >>> K.shape(kvar).eval(session=tf_session)
      array([2, 2], dtype=int32)
      >>> K.shape(input).eval(session=tf_session)
      array([2, 4, 5], dtype=int32)
  ```
  """
  return array_ops.shape(x)


def int_shape(x):
  """Returns the shape tensor or variable as a tuple of int or None entries.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tuple of integers (or None entries).

  Examples:
  ```python
      >>> from keras import backend as K
      >>> input = K.placeholder(shape=(2, 4, 5))
      >>> K.int_shape(input)
      (2, 4, 5)
      >>> val = np.array([[1, 2], [3, 4]])
      >>> kvar = K.variable(value=val)
      >>> K.int_shape(kvar)
      (2, 2)
  ```
  """
  shape = x.get_shape()
  try:
    return tuple([i.__int__() for i in shape])
  except ValueError:
    return None


def ndim(x):
  """Returns the number of axes in a tensor, as an integer.

  Arguments:
      x: Tensor or variable.

  Returns:
      Integer (scalar), number of axes.

  Examples:
  ```python
      >>> from keras import backend as K
      >>> input = K.placeholder(shape=(2, 4, 5))
      >>> val = np.array([[1, 2], [3, 4]])
      >>> kvar = K.variable(value=val)
      >>> K.ndim(input)
      3
      >>> K.ndim(kvar)
      2
  ```
  """
  dims = x.get_shape()._dims
  if dims is not None:
    return len(dims)
  return None


def dtype(x):
  """Returns the dtype of a Keras tensor or variable, as a string.

  Arguments:
      x: Tensor or variable.

  Returns:
      String, dtype of `x`.

  Examples:
  ```python
      >>> from keras import backend as K
      >>> K.dtype(K.placeholder(shape=(2,4,5)))
      'float32'
      >>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float32'))
      'float32'
      >>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float64'))
      'float64'
      # Keras variable
      >>> kvar = K.variable(np.array([[1, 2], [3, 4]]))
      >>> K.dtype(kvar)
      'float32_ref'
      >>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
      >>> K.dtype(kvar)
      'float32_ref'
  ```
  """
  return x.dtype.name


def eval(x):
  """Evaluates the value of a variable.

  Arguments:
      x: A variable.

  Returns:
      A Numpy array.

  Examples:
  ```python
      >>> from keras import backend as K
      >>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
      >>> K.eval(kvar)
      array([[ 1.,  2.],
             [ 3.,  4.]], dtype=float32)
  ```
  """
  return to_dense(x).eval(session=get_session())


def zeros(shape, dtype=None, name=None):
  """Instantiates an all-zeros variable and returns it.

  Arguments:
      shape: Tuple of integers, shape of returned Keras variable
      dtype: String, data type of returned Keras variable
      name: String, name of returned Keras variable

  Returns:
      A variable (including Keras metadata), filled with `0.0`.

  Example:
  ```python
      >>> from keras import backend as K
      >>> kvar = K.zeros((3,4))
      >>> K.eval(kvar)
      array([[ 0.,  0.,  0.,  0.],
             [ 0.,  0.,  0.,  0.],
             [ 0.,  0.,  0.,  0.]], dtype=float32)
  ```
  """
  if dtype is None:
    dtype = floatx()
  shape = tuple(map(int, shape))
  tf_dtype = _convert_string_dtype(dtype)
  return variable(
      init_ops.constant_initializer(0., dtype=tf_dtype)(shape), dtype, name)


def ones(shape, dtype=None, name=None):
  """Instantiates an all-ones tensor variable and returns it.

  Arguments:
      shape: Tuple of integers, shape of returned Keras variable.
      dtype: String, data type of returned Keras variable.
      name: String, name of returned Keras variable.

  Returns:
      A Keras variable, filled with `1.0`.

  Example:
  ```python
      >>> from keras import backend as K
      >>> kvar = K.ones((3,4))
      >>> K.eval(kvar)
      array([[ 1.,  1.,  1.,  1.],
             [ 1.,  1.,  1.,  1.],
             [ 1.,  1.,  1.,  1.]], dtype=float32)
  ```
  """
  if dtype is None:
    dtype = floatx()
  shape = tuple(map(int, shape))
  tf_dtype = _convert_string_dtype(dtype)
  return variable(
      init_ops.constant_initializer(1., dtype=tf_dtype)(shape), dtype, name)


def eye(size, dtype=None, name=None):
  """Instantiate an identity matrix and returns it.

  Arguments:
      size: Integer, number of rows/columns.
      dtype: String, data type of returned Keras variable.
      name: String, name of returned Keras variable.

  Returns:
      A Keras variable, an identity matrix.

  Example:
  ```python
      >>> from keras import backend as K
      >>> kvar = K.eye(3)
      >>> K.eval(kvar)
      array([[ 1.,  0.,  0.],
             [ 0.,  1.,  0.],
             [ 0.,  0.,  1.]], dtype=float32)
  ```

  """
  return variable(np.eye(size), dtype, name)


def zeros_like(x, dtype=None, name=None):
  """Instantiates an all-zeros variable of the same shape as another tensor.

  Arguments:
      x: Keras variable or Keras tensor.
      dtype: String, dtype of returned Keras variable.
           None uses the dtype of x.
      name: String, name for the variable to create.

  Returns:
      A Keras variable with the shape of x filled with zeros.

  Example:
  ```python
      >>> from keras import backend as K
      >>> kvar = K.variable(np.random.random((2,3)))
      >>> kvar_zeros = K.zeros_like(kvar)
      >>> K.eval(kvar_zeros)
      array([[ 0.,  0.,  0.],
             [ 0.,  0.,  0.]], dtype=float32)
  ```
  """
  return array_ops.zeros_like(x, dtype=dtype, name=name)


def ones_like(x, dtype=None, name=None):
  """Instantiates an all-ones variable of the same shape as another tensor.

  Arguments:
      x: Keras variable or tensor.
      dtype: String, dtype of returned Keras variable.
           None uses the dtype of x.
      name: String, name for the variable to create.

  Returns:
      A Keras variable with the shape of x filled with ones.

  Example:
  ```python
      >>> from keras import backend as K
      >>> kvar = K.variable(np.random.random((2,3)))
      >>> kvar_ones = K.ones_like(kvar)
      >>> K.eval(kvar_ones)
      array([[ 1.,  1.,  1.],
             [ 1.,  1.,  1.]], dtype=float32)
  ```
  """
  return array_ops.ones_like(x, dtype=dtype, name=name)


def identity(x):
  """Returns a tensor with the same content as the input tensor.

  Arguments:
      x: The input tensor.

  Returns:
      A tensor of the same shape, type and content.
  """
  return array_ops.identity(x)


def random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None):
  """Instantiates a variable with values drawn from a uniform distribution.

  Arguments:
      shape: Tuple of integers, shape of returned Keras variable.
      low: Float, lower boundary of the output interval.
      high: Float, upper boundary of the output interval.
      dtype: String, dtype of returned Keras variable.
      name: String, name of returned Keras variable.
      seed: Integer, random seed.

  Returns:
      A Keras variable, filled with drawn samples.

  Example:
  ```python
      # TensorFlow example
      >>> kvar = K.random_uniform_variable((2,3), 0, 1)
      >>> kvar
      <tensorflow.python.ops.variables.Variable object at 0x10ab40b10>
      >>> K.eval(kvar)
      array([[ 0.10940075,  0.10047495,  0.476143  ],
             [ 0.66137183,  0.00869417,  0.89220798]], dtype=float32)
  ```
  """
  if dtype is None:
    dtype = floatx()
  shape = tuple(map(int, shape))
  tf_dtype = _convert_string_dtype(dtype)
  if seed is None:
    # ensure that randomness is conditioned by the Numpy RNG
    seed = np.random.randint(10e8)
  value = init_ops.random_uniform_initializer(
      low, high, dtype=tf_dtype, seed=seed)(shape)
  return variable(value, dtype=dtype, name=name)


def random_normal_variable(shape, mean, scale, dtype=None, name=None,
                           seed=None):
  """Instantiates a variable with values drawn from a normal distribution.

  Arguments:
      shape: Tuple of integers, shape of returned Keras variable.
      mean: Float, mean of the normal distribution.
      scale: Float, standard deviation of the normal distribution.
      dtype: String, dtype of returned Keras variable.
      name: String, name of returned Keras variable.
      seed: Integer, random seed.

  Returns:
      A Keras variable, filled with drawn samples.

  Example:
  ```python
      # TensorFlow example
      >>> kvar = K.random_normal_variable((2,3), 0, 1)
      >>> kvar
      <tensorflow.python.ops.variables.Variable object at 0x10ab12dd0>
      >>> K.eval(kvar)
      array([[ 1.19591331,  0.68685907, -0.63814116],
             [ 0.92629528,  0.28055015,  1.70484698]], dtype=float32)
  ```
  """
  if dtype is None:
    dtype = floatx()
  shape = tuple(map(int, shape))
  tf_dtype = _convert_string_dtype(dtype)
  if seed is None:
    # ensure that randomness is conditioned by the Numpy RNG
    seed = np.random.randint(10e8)
  value = init_ops.random_normal_initializer(
      mean, scale, dtype=tf_dtype, seed=seed)(shape)
  return variable(value, dtype=dtype, name=name)


def count_params(x):
  """Returns the number of scalars in a Keras variable.

  Arguments:
      x: Keras variable.

  Returns:
      Integer, the number of scalars in `x`.

  Example:
  ```python
      >>> kvar = K.zeros((2,3))
      >>> K.count_params(kvar)
      6
      >>> K.eval(kvar)
      array([[ 0.,  0.,  0.],
             [ 0.,  0.,  0.]], dtype=float32)
  ```
  """
  shape = x.get_shape()
  return np.prod([shape[i]._value for i in range(len(shape))])


def cast(x, dtype):
  """Casts a tensor to a different dtype and returns it.

  You can cast a Keras variable but it still returns a Keras tensor.

  Arguments:
      x: Keras tensor (or variable).
      dtype: String, either (`'float16'`, `'float32'`, or `'float64'`).

  Returns:
      Keras tensor with dtype `dtype`.

  Example:
  ```python
      >>> from keras import backend as K
      >>> input = K.placeholder((2, 3), dtype='float32')
      >>> input
      <tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
      # It doesn't work in-place as below.
      >>> K.cast(input, dtype='float16')
      <tf.Tensor 'Cast_1:0' shape=(2, 3) dtype=float16>
      >>> input
      <tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
      # you need to assign it.
      >>> input = K.cast(input, dtype='float16')
      >>> input
      <tf.Tensor 'Cast_2:0' shape=(2, 3) dtype=float16>
  ```
  """
  return math_ops.cast(x, dtype)


# UPDATES OPS


def update(x, new_x):
  return state_ops.assign(x, new_x)


def update_add(x, increment):
  """Update the value of `x` by adding `increment`.

  Arguments:
      x: A Variable.
      increment: A tensor of same shape as `x`.

  Returns:
      The variable `x` updated.
  """
  return state_ops.assign_add(x, increment)


def update_sub(x, decrement):
  """Update the value of `x` by subtracting `decrement`.

  Arguments:
      x: A Variable.
      decrement: A tensor of same shape as `x`.

  Returns:
      The variable `x` updated.
  """
  return state_ops.assign_sub(x, decrement)


def moving_average_update(x, value, momentum):
  """Compute the moving average of a variable.

  Arguments:
      x: A Variable.
      value: A tensor with the same shape as `variable`.
      momentum: The moving average momentum.

  Returns:
      An Operation to update the variable.
  """
  return moving_averages.assign_moving_average(
      x, value, momentum, zero_debias=False)


# LINEAR ALGEBRA


def dot(x, y):
  """Multiplies 2 tensors (and/or variables) and returns a *tensor*.

  When attempting to multiply a nD tensor
  with a nD tensor, it reproduces the Theano behavior.
  (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A tensor, dot product of `x` and `y`.

  Examples:
  ```python
      # dot product between tensors
      >>> x = K.placeholder(shape=(2, 3))
      >>> y = K.placeholder(shape=(3, 4))
      >>> xy = K.dot(x, y)
      >>> xy
      <tf.Tensor 'MatMul_9:0' shape=(2, 4) dtype=float32>
  ```

  ```python
      # dot product between tensors
      >>> x = K.placeholder(shape=(32, 28, 3))
      >>> y = K.placeholder(shape=(3, 4))
      >>> xy = K.dot(x, y)
      >>> xy
      <tf.Tensor 'MatMul_9:0' shape=(32, 28, 4) dtype=float32>
  ```

  ```python
      # Theano-like behavior example
      >>> x = K.random_uniform_variable(shape=(2, 3), low=0, high=1)
      >>> y = K.ones((4, 3, 5))
      >>> xy = K.dot(x, y)
      >>> K.int_shape(xy)
      (2, 4, 5)
  ```
  """
  if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
    x_shape = []
    for i, s in zip(int_shape(x), array_ops.unstack(array_ops.shape(x))):
      if i is not None:
        x_shape.append(i)
      else:
        x_shape.append(s)
    x_shape = tuple(x_shape)
    y_shape = []
    for i, s in zip(int_shape(y), array_ops.unstack(array_ops.shape(y))):
      if i is not None:
        y_shape.append(i)
      else:
        y_shape.append(s)
    y_shape = tuple(y_shape)
    y_permute_dim = list(range(ndim(y)))
    y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
    xt = array_ops.reshape(x, [-1, x_shape[-1]])
    yt = array_ops.reshape(
        array_ops.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
    return array_ops.reshape(
        math_ops.matmul(xt, yt), x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
  if is_sparse(x):
    out = sparse_ops.sparse_tensor_dense_matmul(x, y)
  else:
    out = math_ops.matmul(x, y)
  return out


def batch_dot(x, y, axes=None):
  """Batchwise dot product.

  `batch_dot` is used to compute dot product of `x` and `y` when
  `x` and `y` are data in batch, i.e. in a shape of
  `(batch_size, :)`.
  `batch_dot` results in a tensor or variable with less dimensions
  than the input. If the number of dimensions is reduced to 1,
  we use `expand_dims` to make sure that ndim is at least 2.

  Arguments:
      x: Keras tensor or variable with `ndim >= 2`.
      y: Keras tensor or variable with `ndim >= 2`.
      axes: list of (or single) int with target dimensions.
          The lengths of `axes[0]` and `axes[1]` should be the same.

  Returns:
      A tensor with shape equal to the concatenation of `x`'s shape
      (less the dimension that was summed over) and `y`'s shape
      (less the batch dimension and the dimension that was summed over).
      If the final rank is 1, we reshape it to `(batch_size, 1)`.

  Examples:
      Assume `x = [[1, 2], [3, 4]]` and `y = [[5, 6], [7, 8]]`
      `batch_dot(x, y, axes=1) = [[17, 53]]` which is the main diagonal
      of `x.dot(y.T)`, although we never have to calculate the off-diagonal
      elements.

      Shape inference:
      Let `x`'s shape be `(100, 20)` and `y`'s shape be `(100, 30, 20)`.
      If `axes` is (1, 2), to find the output shape of resultant tensor,
          loop through each dimension in `x`'s shape and `y`'s shape:

      * `x.shape[0]` : 100 : append to output shape
      * `x.shape[1]` : 20 : do not append to output shape,
          dimension 1 of `x` has been summed over. (`dot_axes[0]` = 1)
      * `y.shape[0]` : 100 : do not append to output shape,
          always ignore first dimension of `y`
      * `y.shape[1]` : 30 : append to output shape
      * `y.shape[2]` : 20 : do not append to output shape,
          dimension 2 of `y` has been summed over. (`dot_axes[1]` = 2)
      `output_shape` = `(100, 30)`

  ```python
      >>> x_batch = K.ones(shape=(32, 20, 1))
      >>> y_batch = K.ones(shape=(32, 30, 20))
      >>> xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[1, 2])
      >>> K.int_shape(xy_batch_dot)
      (32, 1, 30)
  ```
  """
  if isinstance(axes, int):
    axes = (axes, axes)
  x_ndim = ndim(x)
  y_ndim = ndim(y)
  if x_ndim > y_ndim:
    diff = x_ndim - y_ndim
    y = array_ops.reshape(y,
                          array_ops.concat(
                              [array_ops.shape(y), [1] * (diff)], axis=0))
  elif y_ndim > x_ndim:
    diff = y_ndim - x_ndim
    x = array_ops.reshape(x,
                          array_ops.concat(
                              [array_ops.shape(x), [1] * (diff)], axis=0))
  else:
    diff = 0
  if ndim(x) == 2 and ndim(y) == 2:
    if axes[0] == axes[1]:
      out = math_ops.reduce_sum(math_ops.multiply(x, y), axes[0])
    else:
      out = math_ops.reduce_sum(
          math_ops.multiply(array_ops.transpose(x, [1, 0]), y), axes[1])
  else:
    if axes is not None:
      adj_x = None if axes[0] == ndim(x) - 1 else True
      adj_y = True if axes[1] == ndim(y) - 1 else None
    else:
      adj_x = None
      adj_y = None
    out = math_ops.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
  if diff:
    if x_ndim > y_ndim:
      idx = x_ndim + y_ndim - 3
    else:
      idx = x_ndim - 1
    out = array_ops.squeeze(out, list(range(idx, idx + diff)))
  if ndim(out) == 1:
    out = expand_dims(out, 1)
  return out


def transpose(x):
  """Transposes a tensor and returns it.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.

  Examples:
  ```python
      >>> var = K.variable([[1, 2, 3], [4, 5, 6]])
      >>> K.eval(var)
      array([[ 1.,  2.,  3.],
             [ 4.,  5.,  6.]], dtype=float32)
      >>> var_transposed = K.transpose(var)
      >>> K.eval(var_transposed)
      array([[ 1.,  4.],
             [ 2.,  5.],
             [ 3.,  6.]], dtype=float32)
  ```

  ```python
      >>> input = K.placeholder((2, 3))
      >>> input
      <tf.Tensor 'Placeholder_11:0' shape=(2, 3) dtype=float32>
      >>> input_transposed = K.transpose(input)
      >>> input_transposed
      <tf.Tensor 'transpose_4:0' shape=(3, 2) dtype=float32>

  ```
  """
  return array_ops.transpose(x)


def gather(reference, indices):
  """Retrieves the elements of indices `indices` in the tensor `reference`.

  Arguments:
      reference: A tensor.
      indices: An integer tensor of indices.

  Returns:
      A tensor of same type as `reference`.
  """
  return array_ops.gather(reference, indices)


# ELEMENT-WISE OPERATIONS


def _normalize_axis(axis, ndim):
  """Converts negative axes to positive values.

  Arguments:
      axis: Integer axis (possibly negative).
      ndim: Rank of the tensor considered.

  Returns:
      Positive integer axis.
  """
  if isinstance(axis, tuple):
    axis = list(axis)
  if isinstance(axis, list):
    for i, a in enumerate(axis):
      if a is not None and a < 0:
        axis[i] = a % ndim
  else:
    if axis is not None and axis < 0:
      axis %= ndim
  return axis


def max(x, axis=None, keepdims=False):
  """Maximum value in a tensor.

  Arguments:
      x: A tensor or variable.
      axis: An integer, the axis to find maximum values.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`,
          the reduced dimension is retained with length 1.

  Returns:
      A tensor with maximum values of `x`.
  """
  axis = _normalize_axis(axis, ndim(x))
  return math_ops.reduce_max(x, reduction_indices=axis, keep_dims=keepdims)


def min(x, axis=None, keepdims=False):
  """Minimum value in a tensor.

  Arguments:
      x: A tensor or variable.
      axis: An integer, the axis to find minimum values.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`,
          the reduced dimension is retained with length 1.

  Returns:
      A tensor with miminum values of `x`.
  """
  axis = _normalize_axis(axis, ndim(x))
  return math_ops.reduce_min(x, reduction_indices=axis, keep_dims=keepdims)


def sum(x, axis=None, keepdims=False):
  """Sum of the values in a tensor, alongside the specified axis.

  Arguments:
      x: A tensor or variable.
      axis: An integer, the axis to sum over.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`,
          the reduced dimension is retained with length 1.

  Returns:
      A tensor with sum of `x`.
  """
  axis = _normalize_axis(axis, ndim(x))
  return math_ops.reduce_sum(x, reduction_indices=axis, keep_dims=keepdims)


def prod(x, axis=None, keepdims=False):
  """Multiplies the values in a tensor, alongside the specified axis.

  Arguments:
      x: A tensor or variable.
      axis: An integer, the axis to compute the product.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`,
          the reduced dimension is retained with length 1.

  Returns:
      A tensor with the product of elements of `x`.
  """
  axis = _normalize_axis(axis, ndim(x))
  return math_ops.reduce_prod(x, reduction_indices=axis, keep_dims=keepdims)


def cumsum(x, axis=0):
  """Cumulative sum of the values in a tensor, alongside the specified axis.

  Arguments:
      x: A tensor or variable.
      axis: An integer, the axis to compute the sum.

  Returns:
      A tensor of the cumulative sum of values of `x` along `axis`.
  """
  axis = _normalize_axis(axis, ndim(x))
  return math_ops.cumsum(x, axis=axis)


def cumprod(x, axis=0):
  """Cumulative product of the values in a tensor, alongside the specified axis.

  Arguments:
      x: A tensor or variable.
      axis: An integer, the axis to compute the product.

  Returns:
      A tensor of the cumulative product of values of `x` along `axis`.
  """
  axis = _normalize_axis(axis, ndim(x))
  return math_ops.cumprod(x, axis=axis)


def var(x, axis=None, keepdims=False):
  """Variance of a tensor, alongside the specified axis.

  Arguments:
      x: A tensor or variable.
      axis: An integer, the axis to compute the variance.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`,
          the reduced dimension is retained with length 1.

  Returns:
      A tensor with the variance of elements of `x`.
  """
  axis = _normalize_axis(axis, ndim(x))
  if x.dtype.base_dtype == dtypes_module.bool:
    x = math_ops.cast(x, floatx())
  m = math_ops.reduce_mean(x, reduction_indices=axis, keep_dims=True)
  devs_squared = math_ops.square(x - m)
  return math_ops.reduce_mean(
      devs_squared, reduction_indices=axis, keep_dims=keepdims)


def std(x, axis=None, keepdims=False):
  """Standard deviation of a tensor, alongside the specified axis.

  Arguments:
      x: A tensor or variable.
      axis: An integer, the axis to compute the standard deviation.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`,
          the reduced dimension is retained with length 1.

  Returns:
      A tensor with the standard deviation of elements of `x`.
  """
  return math_ops.sqrt(var(x, axis=axis, keepdims=keepdims))


def mean(x, axis=None, keepdims=False):
  """Mean of a tensor, alongside the specified axis.

  Arguments:
      x: A tensor or variable.
      axis: A list of integer. Axes to compute the mean.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1 for each entry in `axis`. If `keep_dims` is `True`,
          the reduced dimensions are retained with length 1.

  Returns:
      A tensor with the mean of elements of `x`.
  """
  axis = _normalize_axis(axis, ndim(x))
  if x.dtype.base_dtype == dtypes_module.bool:
    x = math_ops.cast(x, floatx())
  return math_ops.reduce_mean(x, reduction_indices=axis, keep_dims=keepdims)


def any(x, axis=None, keepdims=False):
  """Bitwise reduction (logical OR).

  Arguments:
      x: Tensor or variable.
      axis: axis along which to perform the reduction.
      keepdims: whether the drop or broadcast the reduction axes.

  Returns:
      A uint8 tensor (0s and 1s).
  """
  axis = _normalize_axis(axis, ndim(x))
  x = math_ops.cast(x, dtypes_module.bool)
  return math_ops.reduce_any(x, reduction_indices=axis, keep_dims=keepdims)


def all(x, axis=None, keepdims=False):
  """Bitwise reduction (logical AND).

  Arguments:
      x: Tensor or variable.
      axis: axis along which to perform the reduction.
      keepdims: whether the drop or broadcast the reduction axes.

  Returns:
      A uint8 tensor (0s and 1s).
  """
  axis = _normalize_axis(axis, ndim(x))
  x = math_ops.cast(x, dtypes_module.bool)
  return math_ops.reduce_all(x, reduction_indices=axis, keep_dims=keepdims)


def argmax(x, axis=-1):
  """Returns the index of the maximum value along an axis.

  Arguments:
      x: Tensor or variable.
      axis: axis along which to perform the reduction.

  Returns:
      A tensor.
  """
  axis = _normalize_axis(axis, ndim(x))
  return math_ops.argmax(x, axis)


def argmin(x, axis=-1):
  """Returns the index of the minimum value along an axis.

  Arguments:
      x: Tensor or variable.
      axis: axis along which to perform the reduction.

  Returns:
      A tensor.
  """
  axis = _normalize_axis(axis, ndim(x))
  return math_ops.argmin(x, axis)


def square(x):
  """Element-wise square.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.square(x)


def abs(x):
  """Element-wise absolute value.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.abs(x)


def sqrt(x):
  """Element-wise square root.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  zero = _to_tensor(0., x.dtype.base_dtype)
  inf = _to_tensor(np.inf, x.dtype.base_dtype)
  x = clip_ops.clip_by_value(x, zero, inf)
  return math_ops.sqrt(x)


def exp(x):
  """Element-wise exponential.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.exp(x)


def log(x):
  """Element-wise log.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.log(x)


def logsumexp(x, axis=None, keepdims=False):
  """Computes log(sum(exp(elements across dimensions of a tensor))).

  This function is more numerically stable than log(sum(exp(x))).
  It avoids overflows caused by taking the exp of large inputs and
  underflows caused by taking the log of small inputs.

  Arguments:
      x: A tensor or variable.
      axis: An integer, the axis to reduce over.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`, the reduced dimension is
          retained with length 1.

  Returns:
      The reduced tensor.
  """
  axis = _normalize_axis(axis, ndim(x))
  return math_ops.reduce_logsumexp(x, axis=axis, keep_dims=keepdims)


def round(x):
  """Element-wise rounding to the closest integer.

  In case of tie, the rounding mode used is "half to even".

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.round(x)


def sign(x):
  """Element-wise sign.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.sign(x)


def pow(x, a):
  """Element-wise exponentiation.

  Arguments:
      x: Tensor or variable.
      a: Python integer.

  Returns:
      A tensor.
  """
  return math_ops.pow(x, a)


def clip(x, min_value, max_value):
  """Element-wise value clipping.

  Arguments:
      x: Tensor or variable.
      min_value: Python float or integer.
      max_value: Python float or integer.

  Returns:
      A tensor.
  """
  if max_value is not None and max_value < min_value:
    max_value = min_value
  if max_value is None:
    max_value = np.inf
  min_value = _to_tensor(min_value, x.dtype.base_dtype)
  max_value = _to_tensor(max_value, x.dtype.base_dtype)
  return clip_ops.clip_by_value(x, min_value, max_value)


def equal(x, y):
  """Element-wise equality between two tensors.

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  """
  return math_ops.equal(x, y)


def not_equal(x, y):
  """Element-wise inequality between two tensors.

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  """
  return math_ops.not_equal(x, y)


def greater(x, y):
  """Element-wise truth value of (x > y).

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  """
  return math_ops.greater(x, y)


def greater_equal(x, y):
  """Element-wise truth value of (x >= y).

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  """
  return math_ops.greater_equal(x, y)


def less(x, y):
  """Element-wise truth value of (x < y).

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  """
  return math_ops.less(x, y)


def less_equal(x, y):
  """Element-wise truth value of (x <= y).

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  """
  return math_ops.less_equal(x, y)


def maximum(x, y):
  """Element-wise maximum of two tensors.

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.maximum(x, y)


def minimum(x, y):
  """Element-wise minimum of two tensors.

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.minimum(x, y)


def sin(x):
  """Computes sin of x element-wise.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.sin(x)


def cos(x):
  """Computes cos of x element-wise.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.cos(x)


def normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=1e-3):
  """Computes mean and std for batch then apply batch_normalization on batch.

  Arguments:
      x: Input tensor or variable.
      gamma: Tensor by which to scale the input.
      beta: Tensor with which to center the input.
      reduction_axes: iterable of integers,
          axes over which to normalize.
      epsilon: Fuzz factor.

  Returns:
      A tuple length of 3, `(normalized_tensor, mean, variance)`.
  """
  mean, var = nn.moments(
      x, reduction_axes, shift=None, name=None, keep_dims=False)
  if sorted(reduction_axes) == list(range(ndim(x)))[:-1]:
    normed = nn.batch_normalization(x, mean, var, beta, gamma, epsilon)
  else:
    # need broadcasting
    target_shape = []
    for axis in range(ndim(x)):
      if axis in reduction_axes:
        target_shape.append(1)
      else:
        target_shape.append(array_ops.shape(x)[axis])
    target_shape = array_ops.stack(target_shape)

    broadcast_mean = array_ops.reshape(mean, target_shape)
    broadcast_var = array_ops.reshape(var, target_shape)
    if gamma is None:
      broadcast_gamma = None
    else:
      broadcast_gamma = array_ops.reshape(gamma, target_shape)
    if beta is None:
      broadcast_beta = None
    else:
      broadcast_beta = array_ops.reshape(beta, target_shape)
    normed = nn.batch_normalization(x, broadcast_mean, broadcast_var,
                                    broadcast_beta, broadcast_gamma, epsilon)
  return normed, mean, var


def batch_normalization(x, mean, var, beta, gamma, epsilon=1e-3):
  """Applies batch normalization on x given mean, var, beta and gamma.

  I.e. returns:
  `output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta`

  Arguments:
      x: Input tensor or variable.
      mean: Mean of batch.
      var: Variance of batch.
      beta: Tensor with which to center the input.
      gamma: Tensor by which to scale the input.
      epsilon: Fuzz factor.

  Returns:
      A tensor.
  """
  return nn.batch_normalization(x, mean, var, beta, gamma, epsilon)


# SHAPE OPERATIONS


def concatenate(tensors, axis=-1):
  """Concatenates a list of tensors alongside the specified axis.

  Arguments:
      tensors: list of tensors to concatenate.
      axis: concatenation axis.

  Returns:
      A tensor.
  """
  if axis < 0:
    rank = ndim(tensors[0])
    if rank:
      axis %= rank
    else:
      axis = 0

  if py_all([is_sparse(x) for x in tensors]):
    return sparse_ops.sparse_concat(axis, tensors)
  else:
    return array_ops.concat([to_dense(x) for x in tensors], axis)


def reshape(x, shape):
  """Reshapes a tensor to the specified shape.

  Arguments:
      x: Tensor or variable.
      shape: Target shape tuple.

  Returns:
      A tensor.
  """
  return array_ops.reshape(x, shape)


def permute_dimensions(x, pattern):
  """Permutes axes in a tensor.

  Arguments:
      x: Tensor or variable.
      pattern: A tuple of
          dimension indices, e.g. `(0, 2, 1)`.

  Returns:
      A tensor.
  """
  return array_ops.transpose(x, perm=pattern)


def resize_images(x, height_factor, width_factor, data_format):
  """Resizes the images contained in a 4D tensor.

  Arguments:
      x: Tensor or variable to resize.
      height_factor: Positive integer.
      width_factor: Positive integer.
      data_format: One of `"channels_first"`, `"channels_last"`.

  Returns:
      A tensor.

  Raises:
      ValueError: if `data_format` is neither
          `channels_last` or `channels_first`.
  """
  if data_format == 'channels_first':
    original_shape = int_shape(x)
    new_shape = array_ops.shape(x)[2:]
    new_shape *= constant_op.constant(
        np.array([height_factor, width_factor]).astype('int32'))
    x = permute_dimensions(x, [0, 2, 3, 1])
    x = image_ops.resize_nearest_neighbor(x, new_shape)
    x = permute_dimensions(x, [0, 3, 1, 2])
    x.set_shape((None, None, original_shape[2] * height_factor
                 if original_shape[2] is not None else None,
                 original_shape[3] * width_factor
                 if original_shape[3] is not None else None))
    return x
  elif data_format == 'channels_last':
    original_shape = int_shape(x)
    new_shape = array_ops.shape(x)[1:3]
    new_shape *= constant_op.constant(
        np.array([height_factor, width_factor]).astype('int32'))
    x = image_ops.resize_nearest_neighbor(x, new_shape)
    x.set_shape((None, original_shape[1] * height_factor
                 if original_shape[1] is not None else None,
                 original_shape[2] * width_factor
                 if original_shape[2] is not None else None, None))
    return x
  else:
    raise ValueError('Invalid data_format:', data_format)


def resize_volumes(x, depth_factor, height_factor, width_factor, data_format):
  """Resizes the volume contained in a 5D tensor.

  Arguments:
      x: Tensor or variable to resize.
      depth_factor: Positive integer.
      height_factor: Positive integer.
      width_factor: Positive integer.
      data_format: One of `"channels_first"`, `"channels_last"`.

  Returns:
      A tensor.

  Raises:
      ValueError: if `data_format` is neither
          `channels_last` or `channels_first`.
  """
  if data_format == 'channels_first':
    output = repeat_elements(x, depth_factor, axis=2)
    output = repeat_elements(output, height_factor, axis=3)
    output = repeat_elements(output, width_factor, axis=4)
    return output
  elif data_format == 'channels_last':
    output = repeat_elements(x, depth_factor, axis=1)
    output = repeat_elements(output, height_factor, axis=2)
    output = repeat_elements(output, width_factor, axis=3)
    return output
  else:
    raise ValueError('Invalid data_format:', data_format)


def repeat_elements(x, rep, axis):
  """Repeats the elements of a tensor along an axis, like `np.repeat`.

  If `x` has shape `(s1, s2, s3)` and `axis` is `1`, the output
  will have shape `(s1, s2 * rep, s3)`.

  Arguments:
      x: Tensor or variable.
      rep: Python integer, number of times to repeat.
      axis: Axis along which to repeat.

  Raises:
      ValueError: In case `x.shape[axis]` is undefined.

  Returns:
      A tensor.
  """
  x_shape = x.get_shape().as_list()
  if x_shape[axis] is None:
    raise ValueError('Axis ' + str(axis) + ' of input tensor '
                     'should have a defined dimension, but is None. '
                     'Full tensor shape: ' + str(tuple(x_shape)) + '. '
                     'Typically you need to pass a fully-defined '
                     '`input_shape` argument to your first layer.')
  # slices along the repeat axis
  splits = array_ops.split(value=x, num_or_size_splits=x_shape[axis], axis=axis)
  # repeat each slice the given number of reps
  x_rep = [s for s in splits for _ in range(rep)]
  return concatenate(x_rep, axis)


def repeat(x, n):
  """Repeats a 2D tensor.

  if `x` has shape (samples, dim) and `n` is `2`,
  the output will have shape `(samples, 2, dim)`.

  Arguments:
      x: Tensor or variable.
      n: Python integer, number of times to repeat.

  Returns:
      A tensor.
  """
  assert ndim(x) == 2
  x = array_ops.expand_dims(x, 1)
  pattern = array_ops.stack([1, n, 1])
  return array_ops.tile(x, pattern)


def arange(start, stop=None, step=1, dtype='int32'):
  """Creates a 1D tensor containing a sequence of integers.

  The function arguments use the same convention as
  Theano's arange: if only one argument is provided,
  it is in fact the "stop" argument.

  The default type of the returned tensor is `'int32'` to
  match TensorFlow's default.

  Arguments:
      start: Start value.
      stop: Stop value.
      step: Difference between two successive values.
      dtype: Integer dtype to use.

  Returns:
      An integer tensor.

  """
  # Match the behavior of numpy and Theano by returning an empty seqence.
  if stop is None and start < 0:
    start = 0
  result = math_ops.range(start, limit=stop, delta=step, name='arange')
  if dtype != 'int32':
    result = cast(result, dtype)
  return result


def tile(x, n):
  """Creates a tensor by tiling `x` by `n`.

  Arguments:
      x: A tensor or variable
      n: A list of integer. The length must be the same as the number of
          dimensions in `x`.

  Returns:
      A tiled tensor.
  """
  if isinstance(n, int):
    n = [n]
  return array_ops.tile(x, n)


def flatten(x):
  """Flatten a tensor.

  Arguments:
      x: A tensor or variable.

  Returns:
      A tensor, reshaped into 1-D
  """
  return array_ops.reshape(x, [-1])


def batch_flatten(x):
  """Turn a nD tensor into a 2D tensor with same 0th dimension.

  In other words, it flattens each data samples of a batch.

  Arguments:
      x: A tensor or variable.

  Returns:
      A tensor.
  """
  x = array_ops.reshape(x, array_ops.stack([-1, prod(shape(x)[1:])]))
  return x


def expand_dims(x, axis=-1):
  """Adds a 1-sized dimension at index "axis".

  Arguments:
      x: A tensor or variable.
      axis: Position where to add a new axis.

  Returns:
      A tensor with expanded dimensions.
  """
  return array_ops.expand_dims(x, axis)


def squeeze(x, axis):
  """Removes a 1-dimension from the tensor at index "axis".

  Arguments:
      x: A tensor or variable.
      axis: Axis to drop.

  Returns:
      A tensor with the same data as `x` but reduced dimensions.
  """
  return array_ops.squeeze(x, [axis])


def temporal_padding(x, padding=(1, 1)):
  """Pads the middle dimension of a 3D tensor.

  Arguments:
      x: Tensor or variable.
      padding: Tuple of 2 integers, how many zeros to
          add at the start and end of dim 1.

  Returns:
      A padded 3D tensor.
  """
  assert len(padding) == 2
  pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
  return array_ops.pad(x, pattern)


def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
  """Pads the 2nd and 3rd dimensions of a 4D tensor.

  Arguments:
      x: Tensor or variable.
      padding: Tuple of 2 tuples, padding pattern.
      data_format: One of `channels_last` or `channels_first`.

  Returns:
      A padded 4D tensor.

  Raises:
      ValueError: if `data_format` is neither
          `channels_last` or `channels_first`.
  """
  assert len(padding) == 2
  assert len(padding[0]) == 2
  assert len(padding[1]) == 2
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format ' + str(data_format))

  if data_format == 'channels_first':
    pattern = [[0, 0], [0, 0], list(padding[0]), list(padding[1])]
  else:
    pattern = [[0, 0], list(padding[0]), list(padding[1]), [0, 0]]
  return array_ops.pad(x, pattern)


def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
  """Pads 5D tensor with zeros along the depth, height, width dimensions.

  Pads these dimensions with respectively
  "padding[0]", "padding[1]" and "padding[2]" zeros left and right.

  For 'channels_last' data_format,
  the 2nd, 3rd and 4th dimension will be padded.
  For 'channels_first' data_format,
  the 3rd, 4th and 5th dimension will be padded.

  Arguments:
      x: Tensor or variable.
      padding: Tuple of 3 tuples, padding pattern.
      data_format: One of `channels_last` or `channels_first`.

  Returns:
      A padded 5D tensor.

  Raises:
      ValueError: if `data_format` is neither
          `channels_last` or `channels_first`.

  """
  assert len(padding) == 3
  assert len(padding[0]) == 2
  assert len(padding[1]) == 2
  assert len(padding[2]) == 2
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format ' + str(data_format))

  if data_format == 'channels_first':
    pattern = [[0, 0], [0, 0], [padding[0][0], padding[0][1]],
               [padding[1][0], padding[1][1]], [padding[2][0], padding[2][1]]]
  else:
    pattern = [[0, 0], [padding[0][0], padding[0][1]],
               [padding[1][0], padding[1][1]], [padding[2][0],
                                                padding[2][1]], [0, 0]]
  return array_ops.pad(x, pattern)


def stack(x, axis=0):
  """Stacks a list of rank `R` tensors into a rank `R+1` tensor.

  Arguments:
      x: List of tensors.
      axis: Axis along which to perform stacking.

  Returns:
      A tensor.
  """
  return array_ops.stack(x, axis=axis)


def one_hot(indices, num_classes):
  """Computes the one-hot representation of an integer tensor.

  Arguments:
      indices: nD integer tensor of shape
          `(batch_size, dim1, dim2, ... dim(n-1))`
      num_classes: Integer, number of classes to consider.

  Returns:
      (n + 1)D one hot representation of the input
      with shape `(batch_size, dim1, dim2, ... dim(n-1), num_classes)`

  Returns:
      The one-hot tensor.
  """
  return array_ops.one_hot(indices, depth=num_classes, axis=-1)


def reverse(x, axes):
  """Reverse a tensor along the specified axes.

  Arguments:
      x: Tensor to reverse.
      axes: Integer or iterable of integers.
          Axes to reverse.

  Returns:
      A tensor.
  """
  if isinstance(axes, int):
    axes = [axes]
  return array_ops.reverse(x, axes)


# VALUE MANIPULATION


def get_value(x):
  """Returns the value of a variable.

  Arguments:
      x: input variable.

  Returns:
      A Numpy array.
  """
  return x.eval(session=get_session())


def batch_get_value(tensors):
  """Returns the value of more than one tensor variable.

  Arguments:
      tensors: list of ops to run.

  Returns:
      A list of Numpy arrays.
  """
  if tensors:
    return get_session().run(tensors)
  else:
    return []


def set_value(x, value):
  """Sets the value of a variable, from a Numpy array.

  Arguments:
      x: Tensor to set to a new value.
      value: Value to set the tensor to, as a Numpy array
          (of the same shape).
  """
  value = np.asarray(value)
  tf_dtype = _convert_string_dtype(x.dtype.name.split('_')[0])
  if hasattr(x, '_assign_placeholder'):
    assign_placeholder = x._assign_placeholder
    assign_op = x._assign_op
  else:
    assign_placeholder = array_ops.placeholder(tf_dtype, shape=value.shape)
    assign_op = x.assign(assign_placeholder)
    x._assign_placeholder = assign_placeholder
    x._assign_op = assign_op
  get_session().run(assign_op, feed_dict={assign_placeholder: value})


def batch_set_value(tuples):
  """Sets the values of many tensor variables at once.

  Arguments:
      tuples: a list of tuples `(tensor, value)`.
          `value` should be a Numpy array.
  """
  if tuples:
    assign_ops = []
    feed_dict = {}
    for x, value in tuples:
      value = np.asarray(value)
      tf_dtype = _convert_string_dtype(x.dtype.name.split('_')[0])
      if hasattr(x, '_assign_placeholder'):
        assign_placeholder = x._assign_placeholder
        assign_op = x._assign_op
      else:
        assign_placeholder = array_ops.placeholder(tf_dtype, shape=value.shape)
        assign_op = x.assign(assign_placeholder)
        x._assign_placeholder = assign_placeholder
        x._assign_op = assign_op
      assign_ops.append(assign_op)
      feed_dict[assign_placeholder] = value
    get_session().run(assign_ops, feed_dict=feed_dict)


def print_tensor(x, message=''):
  """Prints `message` and the tensor value when evaluated.

  Arguments:
      x: Tensor to print.
      message: Message to print jointly with the tensor.

  Returns:
      The same tensor `x`, unchanged.
  """
  return logging_ops.Print(x, [x], message)


# GRAPH MANIPULATION


class Function(object):
  """Runs a computation graph.

  Arguments:
      inputs: Feed placeholders to the computation graph.
      outputs: Output tensors to fetch.
      updates: Additional update ops to be run at function call.
      name: a name to help users identify what this function does.
  """

  def __init__(self, inputs, outputs, updates=None, name=None,
               **session_kwargs):
    updates = updates or []
    if not isinstance(inputs, (list, tuple)):
      raise TypeError('`inputs` to a TensorFlow backend function '
                      'should be a list or tuple.')
    if not isinstance(outputs, (list, tuple)):
      raise TypeError('`outputs` of a TensorFlow backend function '
                      'should be a list or tuple.')
    if not isinstance(updates, (list, tuple)):
      raise TypeError('`updates` in a TensorFlow backend function '
                      'should be a list or tuple.')
    self.inputs = list(inputs)
    self.outputs = list(outputs)
    with ops.control_dependencies(self.outputs):
      updates_ops = []
      for update in updates:
        if isinstance(update, tuple):
          p, new_p = update
          updates_ops.append(state_ops.assign(p, new_p))
        else:
          # assumed already an op
          updates_ops.append(update)
      self.updates_op = control_flow_ops.group(*updates_ops)
    self.name = name
    self.session_kwargs = session_kwargs

  def __call__(self, inputs):
    if not isinstance(inputs, (list, tuple)):
      raise TypeError('`inputs` should be a list or tuple.')
    feed_dict = {}
    for tensor, value in zip(self.inputs, inputs):
      if is_sparse(tensor):
        sparse_coo = value.tocoo()
        indices = np.concatenate((np.expand_dims(sparse_coo.row, 1),
                                  np.expand_dims(sparse_coo.col, 1)), 1)
        value = (indices, sparse_coo.data, sparse_coo.shape)
      feed_dict[tensor] = value
    session = get_session()
    updated = session.run(
        self.outputs + [self.updates_op],
        feed_dict=feed_dict,
        **self.session_kwargs)
    return updated[:len(self.outputs)]


def function(inputs, outputs, updates=None, **kwargs):
  """Instantiates a Keras function.

  Arguments:
      inputs: List of placeholder tensors.
      outputs: List of output tensors.
      updates: List of update ops.
      **kwargs: Passed to `tf.Session.run`.

  Returns:
      Output values as Numpy arrays.

  Raises:
      ValueError: if invalid kwargs are passed in.
  """
  if kwargs:
    for key in kwargs:
      if (key not in tf_inspect.getargspec(session_module.Session.run)[0] and
          key not in tf_inspect.getargspec(Function.__init__)[0]):
        msg = ('Invalid argument "%s" passed to K.function with Tensorflow '
               'backend') % key
        raise ValueError(msg)
  return Function(inputs, outputs, updates=updates, **kwargs)


def gradients(loss, variables):
  """Returns the gradients of `variables` w.r.t. `loss`.

  Arguments:
      loss: Scalar tensor to minimize.
      variables: List of variables.

  Returns:
      A gradients tensor.
  """
  return gradients_module.gradients(
      loss, variables, colocate_gradients_with_ops=True)


def stop_gradient(variables):
  """Returns `variables` but with zero gradient w.r.t. every other variable.

  Arguments:
      variables: List of variables.

  Returns:
      The same list of variables.
  """
  return array_ops.stop_gradient(variables)


# CONTROL FLOW


def rnn(step_function,
        inputs,
        initial_states,
        go_backwards=False,
        mask=None,
        constants=None,
        unroll=False):
  """Iterates over the time dimension of a tensor.

  Arguments:
      step_function: RNN step function.
          Parameters;
              input; tensor with shape `(samples, ...)` (no time dimension),
                  representing input for the batch of samples at a certain
                  time step.
              states; list of tensors.
          Returns;
              output; tensor with shape `(samples, output_dim)`
                  (no time dimension).
              new_states; list of tensors, same length and shapes
                  as 'states'. The first state in the list must be the
                  output tensor at the previous timestep.
      inputs: tensor of temporal data of shape `(samples, time, ...)`
          (at least 3D).
      initial_states: tensor with shape (samples, output_dim)
          (no time dimension),
          containing the initial values for the states used in
          the step function.
      go_backwards: boolean. If True, do the iteration over the time
          dimension in reverse order and return the reversed sequence.
      mask: binary tensor with shape `(samples, time, 1)`,
          with a zero for every element that is masked.
      constants: a list of constant values passed at each step.
      unroll: whether to unroll the RNN or to use a symbolic loop
          (`while_loop` or `scan` depending on backend).

  Returns:
      A tuple, `(last_output, outputs, new_states)`.
          last_output: the latest output of the rnn, of shape `(samples, ...)`
          outputs: tensor with shape `(samples, time, ...)` where each
              entry `outputs[s, t]` is the output of the step function
              at time `t` for sample `s`.
          new_states: list of tensors, latest states returned by
              the step function, of shape `(samples, ...)`.

  Raises:
      ValueError: if input dimension is less than 3.
      ValueError: if `unroll` is `True` but input timestep is not a fixed
      number.
      ValueError: if `mask` is provided (not `None`) but states is not provided
          (`len(states)` == 0).
  """
  ndim = len(inputs.get_shape())
  if ndim < 3:
    raise ValueError('Input should be at least 3D.')
  axes = [1, 0] + list(range(2, ndim))
  inputs = array_ops.transpose(inputs, (axes))

  if mask is not None:
    if mask.dtype != dtypes_module.bool:
      mask = math_ops.cast(mask, dtypes_module.bool)
    if len(mask.get_shape()) == ndim - 1:
      mask = expand_dims(mask)
    mask = array_ops.transpose(mask, axes)

  if constants is None:
    constants = []

  if unroll:
    if not inputs.get_shape()[0]:
      raise ValueError('Unrolling requires a ' 'fixed number of timesteps.')
    states = initial_states
    successive_states = []
    successive_outputs = []

    input_list = array_ops.unstack(inputs)
    if go_backwards:
      input_list.reverse()

    if mask is not None:
      mask_list = array_ops.unstack(mask)
      if go_backwards:
        mask_list.reverse()

      for inp, mask_t in zip(input_list, mask_list):
        output, new_states = step_function(inp, states + constants)

        # tf.where needs its condition tensor
        # to be the same shape as its two
        # result tensors, but in our case
        # the condition (mask) tensor is
        # (nsamples, 1), and A and B are (nsamples, ndimensions).
        # So we need to
        # broadcast the mask to match the shape of A and B.
        # That's what the tile call does,
        # it just repeats the mask along its second dimension
        # n times.
        tiled_mask_t = array_ops.tile(mask_t,
                                      array_ops.stack(
                                          [1, array_ops.shape(output)[1]]))

        if not successive_outputs:
          prev_output = zeros_like(output)
        else:
          prev_output = successive_outputs[-1]

        output = array_ops.where(tiled_mask_t, output, prev_output)

        return_states = []
        for state, new_state in zip(states, new_states):
          # (see earlier comment for tile explanation)
          tiled_mask_t = array_ops.tile(mask_t,
                                        array_ops.stack(
                                            [1,
                                             array_ops.shape(new_state)[1]]))
          return_states.append(array_ops.where(tiled_mask_t, new_state, state))
        states = return_states
        successive_outputs.append(output)
        successive_states.append(states)
      last_output = successive_outputs[-1]
      new_states = successive_states[-1]
      outputs = array_ops.stack(successive_outputs)
    else:
      for inp in input_list:
        output, states = step_function(inp, states + constants)
        successive_outputs.append(output)
        successive_states.append(states)
      last_output = successive_outputs[-1]
      new_states = successive_states[-1]
      outputs = array_ops.stack(successive_outputs)

  else:
    if go_backwards:
      inputs = reverse(inputs, 0)

    states = tuple(initial_states)

    time_steps = array_ops.shape(inputs)[0]
    outputs, _ = step_function(inputs[0], initial_states + constants)
    output_ta = tensor_array_ops.TensorArray(
        dtype=outputs.dtype, size=time_steps, tensor_array_name='output_ta')
    input_ta = tensor_array_ops.TensorArray(
        dtype=inputs.dtype, size=time_steps, tensor_array_name='input_ta')
    input_ta = input_ta.unstack(inputs)
    time = constant_op.constant(0, dtype='int32', name='time')

    if mask is not None:
      if not states:
        raise ValueError('No initial states provided! '
                         'When using masking in an RNN, you should '
                         'provide initial states '
                         '(and your step function should return '
                         'as its first state at time `t` '
                         'the output at time `t-1`).')
      if go_backwards:
        mask = reverse(mask, 0)

      mask_ta = tensor_array_ops.TensorArray(
          dtype=dtypes_module.bool,
          size=time_steps,
          tensor_array_name='mask_ta')
      mask_ta = mask_ta.unstack(mask)

      def _step(time, output_ta_t, *states):
        """RNN step function.

        Arguments:
            time: Current timestep value.
            output_ta_t: TensorArray.
            *states: List of states.

        Returns:
            Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
        """
        current_input = input_ta.read(time)
        mask_t = mask_ta.read(time)
        output, new_states = step_function(current_input,
                                           tuple(states) + tuple(constants))
        for state, new_state in zip(states, new_states):
          new_state.set_shape(state.get_shape())
        tiled_mask_t = array_ops.tile(mask_t,
                                      array_ops.stack(
                                          [1, array_ops.shape(output)[1]]))
        output = array_ops.where(tiled_mask_t, output, states[0])
        new_states = [
            array_ops.where(tiled_mask_t, new_states[i], states[i])
            for i in range(len(states))
        ]
        output_ta_t = output_ta_t.write(time, output)
        return (time + 1, output_ta_t) + tuple(new_states)
    else:

      def _step(time, output_ta_t, *states):
        """RNN step function.

        Arguments:
            time: Current timestep value.
            output_ta_t: TensorArray.
            *states: List of states.

        Returns:
            Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
        """
        current_input = input_ta.read(time)
        output, new_states = step_function(current_input,
                                           tuple(states) + tuple(constants))
        for state, new_state in zip(states, new_states):
          new_state.set_shape(state.get_shape())
        output_ta_t = output_ta_t.write(time, output)
        return (time + 1, output_ta_t) + tuple(new_states)

    final_outputs = control_flow_ops.while_loop(
        cond=lambda time, *_: time < time_steps,
        body=_step,
        loop_vars=(time, output_ta) + states,
        parallel_iterations=32,
        swap_memory=True)
    last_time = final_outputs[0]
    output_ta = final_outputs[1]
    new_states = final_outputs[2:]

    outputs = output_ta.stack()
    last_output = output_ta.read(last_time - 1)

  axes = [1, 0] + list(range(2, len(outputs.get_shape())))
  outputs = array_ops.transpose(outputs, axes)
  return last_output, outputs, new_states


def switch(condition, then_expression, else_expression):
  """Switches between two operations depending on a scalar value.

  Note that both `then_expression` and `else_expression`
  should be symbolic tensors of the *same shape*.

  Arguments:
      condition: scalar tensor (`int` or `bool`).
      then_expression: either a tensor, or a callable that returns a tensor.
      else_expression: either a tensor, or a callable that returns a tensor.

  Returns:
      The selected tensor.
  """
  if condition.dtype != dtypes_module.bool:
    condition = math_ops.cast(condition, 'bool')
  if not callable(then_expression):

    def then_expression_fn():
      return then_expression
  else:
    then_expression_fn = then_expression
  if not callable(else_expression):

    def else_expression_fn():
      return else_expression
  else:
    else_expression_fn = else_expression
  x = control_flow_ops.cond(condition, then_expression_fn, else_expression_fn)
  return x


def in_train_phase(x, alt, training=None):
  """Selects `x` in train phase, and `alt` otherwise.

  Note that `alt` should have the *same shape* as `x`.

  Arguments:
      x: What to return in train phase
          (tensor or callable that returns a tensor).
      alt: What to return otherwise
          (tensor or callable that returns a tensor).
      training: Optional scalar tensor
          (or Python boolean, or Python integer)
          specifying the learning phase.

  Returns:
      Either `x` or `alt` based on the `training` flag.
      the `training` flag defaults to `K.learning_phase()`.
  """
  if training is None:
    training = learning_phase()
    uses_learning_phase = True
  else:
    uses_learning_phase = False

  if training is 1 or training is True:
    if callable(x):
      return x()
    else:
      return x

  elif training is 0 or training is False:
    if callable(alt):
      return alt()
    else:
      return alt

  # else: assume learning phase is a placeholder tensor.
  x = switch(training, x, alt)
  if uses_learning_phase:
    x._uses_learning_phase = True
  return x


def in_test_phase(x, alt, training=None):
  """Selects `x` in test phase, and `alt` otherwise.

  Note that `alt` should have the *same shape* as `x`.

  Arguments:
      x: What to return in test phase
          (tensor or callable that returns a tensor).
      alt: What to return otherwise
          (tensor or callable that returns a tensor).
      training: Optional scalar tensor
          (or Python boolean, or Python integer)
          specifying the learning phase.

  Returns:
      Either `x` or `alt` based on `K.learning_phase`.
  """
  return in_train_phase(alt, x, training=training)


# NN OPERATIONS


def relu(x, alpha=0., max_value=None):
  """Rectified linear unit.

  With default values, it returns element-wise `max(x, 0)`.

  Arguments:
      x: A tensor or variable.
      alpha: A scalar, slope of negative section (default=`0.`).
      max_value: Saturation threshold.

  Returns:
      A tensor.
  """
  if alpha != 0.:
    negative_part = nn.relu(-x)
  x = nn.relu(x)
  if max_value is not None:
    max_value = _to_tensor(max_value, x.dtype.base_dtype)
    zero = _to_tensor(0., x.dtype.base_dtype)
    x = clip_ops.clip_by_value(x, zero, max_value)
  if alpha != 0.:
    alpha = _to_tensor(alpha, x.dtype.base_dtype)
    x -= alpha * negative_part
  return x


def elu(x, alpha=1.):
  """Exponential linear unit.

  Arguments:
      x: A tenor or variable to compute the activation function for.
      alpha: A scalar, slope of positive section.

  Returns:
      A tensor.
  """
  res = nn.elu(x)
  if alpha == 1:
    return res
  else:
    return array_ops.where(x > 0, res, alpha * res)


def softmax(x):
  """Softmax of a tensor.

  Arguments:
      x: A tensor or variable.

  Returns:
      A tensor.
  """
  return nn.softmax(x)


def softplus(x):
  """Softplus of a tensor.

  Arguments:
      x: A tensor or variable.

  Returns:
      A tensor.
  """
  return nn.softplus(x)


def softsign(x):
  """Softsign of a tensor.

  Arguments:
      x: A tensor or variable.

  Returns:
      A tensor.
  """
  return nn.softsign(x)


def categorical_crossentropy(output, target, from_logits=False):
  """Categorical crossentropy between an output tensor and a target tensor.

  Arguments:
      output: A tensor resulting from a softmax
          (unless `from_logits` is True, in which
          case `output` is expected to be the logits).
      target: A tensor of the same shape as `output`.
      from_logits: Boolean, whether `output` is the
          result of a softmax, or is a tensor of logits.

  Returns:
      Output tensor.
  """
  # Note: nn.softmax_cross_entropy_with_logits
  # expects logits, Keras expects probabilities.
  if not from_logits:
    # scale preds so that the class probas of each sample sum to 1
    output /= math_ops.reduce_sum(
        output, reduction_indices=len(output.get_shape()) - 1, keep_dims=True)
    # manual computation of crossentropy
    epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
    output = clip_ops.clip_by_value(output, epsilon, 1. - epsilon)
    return -math_ops.reduce_sum(
        target * math_ops.log(output),
        reduction_indices=len(output.get_shape()) - 1)
  else:
    return nn.softmax_cross_entropy_with_logits(labels=target, logits=output)


def sparse_categorical_crossentropy(output, target, from_logits=False):
  """Categorical crossentropy with integer targets.

  Arguments:
      output: A tensor resulting from a softmax
          (unless `from_logits` is True, in which
          case `output` is expected to be the logits).
      target: An integer tensor.
      from_logits: Boolean, whether `output` is the
          result of a softmax, or is a tensor of logits.

  Returns:
      Output tensor.
  """
  # Note: nn.softmax_cross_entropy_with_logits
  # expects logits, Keras expects probabilities.
  if not from_logits:
    epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
    output = clip_ops.clip_by_value(output, epsilon, 1 - epsilon)
    output = math_ops.log(output)

  output_shape = output.get_shape()
  targets = cast(flatten(target), 'int64')
  logits = array_ops.reshape(output, [-1, int(output_shape[-1])])
  res = nn.sparse_softmax_cross_entropy_with_logits(
      labels=targets, logits=logits)
  if len(output_shape) == 3:
    # if our output includes timesteps we need to reshape
    return array_ops.reshape(res, array_ops.shape(output)[:-1])
  else:
    return res


def binary_crossentropy(output, target, from_logits=False):
  """Binary crossentropy between an output tensor and a target tensor.

  Arguments:
      output: A tensor.
      target: A tensor with the same shape as `output`.
      from_logits: Whether `output` is expected to be a logits tensor.
          By default, we consider that `output`
          encodes a probability distribution.

  Returns:
      A tensor.
  """
  # Note: nn.softmax_cross_entropy_with_logits
  # expects logits, Keras expects probabilities.
  if not from_logits:
    # transform back to logits
    epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
    output = clip_ops.clip_by_value(output, epsilon, 1 - epsilon)
    output = math_ops.log(output / (1 - output))
  return nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)


def sigmoid(x):
  """Element-wise sigmoid.

  Arguments:
      x: A tensor or variable.

  Returns:
      A tensor.
  """
  return nn.sigmoid(x)


def hard_sigmoid(x):
  """Segment-wise linear approximation of sigmoid.

  Faster than sigmoid.
  Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
  In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.

  Arguments:
      x: A tensor or variable.

  Returns:
      A tensor.
  """
  x = (0.2 * x) + 0.5
  zero = _to_tensor(0., x.dtype.base_dtype)
  one = _to_tensor(1., x.dtype.base_dtype)
  x = clip_ops.clip_by_value(x, zero, one)
  return x


def tanh(x):
  """Element-wise tanh.

  Arguments:
      x: A tensor or variable.

  Returns:
      A tensor.
  """
  return nn.tanh(x)


def dropout(x, level, noise_shape=None, seed=None):
  """Sets entries in `x` to zero at random, while scaling the entire tensor.

  Arguments:
      x: tensor
      level: fraction of the entries in the tensor
          that will be set to 0.
      noise_shape: shape for randomly generated keep/drop flags,
          must be broadcastable to the shape of `x`
      seed: random seed to ensure determinism.

  Returns:
      A tensor.
  """
  retain_prob = 1. - level
  if seed is None:
    seed = np.random.randint(10e6)
  # the dummy 1. works around a TF bug
  # (float32_ref vs. float32 incomptability)
  return nn.dropout(x * 1., retain_prob, noise_shape, seed=seed)


def l2_normalize(x, axis):
  """Normalizes a tensor wrt the L2 norm alongside the specified axis.

  Arguments:
      x: Tensor or variable.
      axis: axis along which to perform normalization.

  Returns:
      A tensor.
  """
  if axis < 0:
    axis %= len(x.get_shape())
  return nn.l2_normalize(x, dim=axis)


def in_top_k(predictions, targets, k):
  """Returns whether the `targets` are in the top `k` `predictions`.

  Arguments:
      predictions: A tensor of shape `(batch_size, classes)` and type `float32`.
      targets: A 1D tensor of length `batch_size` and type `int32` or `int64`.
      k: An `int`, number of top elements to consider.

  Returns:
      A 1D tensor of length `batch_size` and type `bool`.
      `output[i]` is `True` if `predictions[i, targets[i]]` is within top-`k`
      values of `predictions[i]`.
  """
  return nn.in_top_k(predictions, targets, k)


# CONVOLUTIONS


def _preprocess_deconv_output_shape(x, shape, data_format):
  """Get the output_shape for the deconvolution.

  Arguments:
      x: input tensor.
      shape: output shape.
      data_format: string, one of 'channels_last', 'channels_first'.

  Returns:
      The output shape.
  """
  if data_format == 'channels_first':
    shape = (shape[0], shape[2], shape[3], shape[1])

  if shape[0] is None:
    shape = (array_ops.shape(x)[0],) + tuple(shape[1:])
    shape = array_ops.stack(list(shape))
  return shape


def _preprocess_conv2d_input(x, data_format):
  """Transpose and cast the input before the conv2d.

  Arguments:
      x: input tensor.
      data_format: string, one of 'channels_last', 'channels_first'.

  Returns:
      A tensor.
  """
  if dtype(x) == 'float64':
    x = math_ops.cast(x, 'float32')
  if data_format == 'channels_first':
    # TF uses the last dimension as channel dimension,
    # instead of the 2nd one.
    # TH input shape: (samples, input_depth, rows, cols)
    # TF input shape: (samples, rows, cols, input_depth)
    x = array_ops.transpose(x, (0, 2, 3, 1))
  return x


def _preprocess_conv3d_input(x, data_format):
  """Transpose and cast the input before the conv3d.

  Arguments:
      x: input tensor.
      data_format: string, one of 'channels_last', 'channels_first'.

  Returns:
      A tensor.
  """
  if dtype(x) == 'float64':
    x = math_ops.cast(x, 'float32')
  if data_format == 'channels_first':
    x = array_ops.transpose(x, (0, 2, 3, 4, 1))
  return x


def _preprocess_conv2d_kernel(kernel, data_format):
  """Transpose and cast the kernel before the conv2d.

  Arguments:
      kernel: kernel tensor.
      data_format: string, one of 'channels_last', 'channels_first'.

  Returns:
      A tensor.
  """
  if dtype(kernel) == 'float64':
    kernel = math_ops.cast(kernel, 'float32')
  if data_format == 'channels_first':
    kernel = array_ops.transpose(kernel, (2, 3, 1, 0))
  return kernel


def _preprocess_conv3d_kernel(kernel, data_format):
  """Transpose and cast the kernel before the conv3d.

  Arguments:
      kernel: kernel tensor.
      data_format: string, one of 'channels_last', 'channels_first'.

  Returns:
      A tensor.
  """
  if dtype(kernel) == 'float64':
    kernel = math_ops.cast(kernel, 'float32')
  if data_format == 'channels_first':
    kernel = array_ops.transpose(kernel, (2, 3, 4, 1, 0))
  return kernel


def _preprocess_padding(padding):
  """Convert keras' padding to tensorflow's padding.

  Arguments:
      padding: string, one of 'same' , 'valid'

  Returns:
      a string, one of 'SAME', 'VALID'.

  Raises:
      ValueError: if invalid `padding'`
  """
  if padding == 'same':
    padding = 'SAME'
  elif padding == 'valid':
    padding = 'VALID'
  else:
    raise ValueError('Invalid padding:', padding)
  return padding


def _postprocess_conv2d_output(x, data_format):
  """Transpose and cast the output from conv2d if needed.

  Arguments:
      x: A tensor.
      data_format: string, one of "channels_last", "channels_first".

  Returns:
      A tensor.
  """

  if data_format == 'channels_first':
    x = array_ops.transpose(x, (0, 3, 1, 2))

  if floatx() == 'float64':
    x = math_ops.cast(x, 'float64')
  return x


def _postprocess_conv3d_output(x, data_format):
  """Transpose and cast the output from conv3d if needed.

  Arguments:
      x: A tensor.
      data_format: string, one of "channels_last", "channels_first".

  Returns:
      A tensor.
  """
  if data_format == 'channels_first':
    x = array_ops.transpose(x, (0, 4, 1, 2, 3))

  if floatx() == 'float64':
    x = math_ops.cast(x, 'float64')
  return x


def conv1d(x,
           kernel,
           strides=1,
           padding='valid',
           data_format=None,
           dilation_rate=1):
  """1D convolution.

  Arguments:
      x: Tensor or variable.
      kernel: kernel tensor.
      strides: stride integer.
      padding: string, `"same"`, `"causal"` or `"valid"`.
      data_format: string, one of "channels_last", "channels_first".
      dilation_rate: integer dilate rate.

  Returns:
      A tensor, result of 1D convolution.
  """
  kernel_shape = kernel.get_shape().as_list()
  if padding == 'causal':
    # causal (dilated) convolution:
    left_pad = dilation_rate * (kernel_shape[0] - 1)
    x = temporal_padding(x, (left_pad, 0))
    padding = 'valid'
  padding = _preprocess_padding(padding)
  if data_format == 'channels_last':
    tf_data_format = 'NWC'
  else:
    tf_data_format = 'NCW'
  x = nn.convolution(
      input=x,
      filter=kernel,
      dilation_rate=(dilation_rate,),
      strides=(strides,),
      padding=padding,
      data_format=tf_data_format)
  return x


def conv2d(x,
           kernel,
           strides=(1, 1),
           padding='valid',
           data_format=None,
           dilation_rate=(1, 1)):
  """2D convolution.

  Arguments:
      x: Tensor or variable.
      kernel: kernel tensor.
      strides: strides tuple.
      padding: string, `"same"` or `"valid"`.
      data_format: `"channels_last"` or `"channels_first"`.
          Whether to use Theano or TensorFlow data format
          for inputs/kernels/outputs.
      dilation_rate: tuple of 2 integers.

  Returns:
      A tensor, result of 2D convolution.

  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format ' + str(data_format))

  # With 4d inputs, nn.convolution only supports
  # data_format NHWC, so we transpose the inputs
  # in case we are in data_format channels_first.
  x = _preprocess_conv2d_input(x, data_format)
  padding = _preprocess_padding(padding)
  x = nn.convolution(
      input=x,
      filter=kernel,
      dilation_rate=dilation_rate,
      strides=strides,
      padding=padding,
      data_format='NHWC')
  return _postprocess_conv2d_output(x, data_format)


def conv2d_transpose(x,
                     kernel,
                     output_shape,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None):
  """2D deconvolution (i.e.

  transposed convolution).

  Arguments:
      x: Tensor or variable.
      kernel: kernel tensor.
      output_shape: 1D int tensor for the output shape.
      strides: strides tuple.
      padding: string, `"same"` or `"valid"`.
      data_format: `"channels_last"` or `"channels_first"`.
          Whether to use Theano or TensorFlow data format
          for inputs/kernels/outputs.

  Returns:
      A tensor, result of transposed 2D convolution.

  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format ' + str(data_format))
  if isinstance(output_shape, (tuple, list)):
    output_shape = array_ops.stack(output_shape)

  x = _preprocess_conv2d_input(x, data_format)
  output_shape = _preprocess_deconv_output_shape(x, output_shape, data_format)
  padding = _preprocess_padding(padding)
  strides = (1,) + strides + (1,)

  x = nn.conv2d_transpose(x, kernel, output_shape, strides, padding=padding)
  x = _postprocess_conv2d_output(x, data_format)
  return x


def separable_conv2d(x,
                     depthwise_kernel,
                     pointwise_kernel,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1)):
  """2D convolution with separable filters.

  Arguments:
      x: input tensor
      depthwise_kernel: convolution kernel for the depthwise convolution.
      pointwise_kernel: kernel for the 1x1 convolution.
      strides: strides tuple (length 2).
      padding: padding mode, "valid" or "same".
      data_format: data format, "channels_first" or "channels_last".
      dilation_rate: tuple of integers,
          dilation rates for the separable convolution.

  Returns:
      Output tensor.

  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format ' + str(data_format))

  x = _preprocess_conv2d_input(x, data_format)
  padding = _preprocess_padding(padding)
  strides = (1,) + strides + (1,)

  x = nn.separable_conv2d(
      x,
      depthwise_kernel,
      pointwise_kernel,
      strides=strides,
      padding=padding,
      rate=dilation_rate)
  return _postprocess_conv2d_output(x, data_format)


def conv3d(x,
           kernel,
           strides=(1, 1, 1),
           padding='valid',
           data_format=None,
           dilation_rate=(1, 1, 1)):
  """3D convolution.

  Arguments:
      x: Tensor or variable.
      kernel: kernel tensor.
      strides: strides tuple.
      padding: string, `"same"` or `"valid"`.
      data_format: `"channels_last"` or `"channels_first"`.
          Whether to use Theano or TensorFlow data format
          for inputs/kernels/outputs.
      dilation_rate: tuple of 3 integers.

  Returns:
      A tensor, result of 3D convolution.

  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format ' + str(data_format))

  # With 5d inputs, nn.convolution only supports
  # data_format NDHWC, so we transpose the inputs
  # in case we are in data_format channels_first.
  x = _preprocess_conv3d_input(x, data_format)
  padding = _preprocess_padding(padding)
  x = nn.convolution(
      input=x,
      filter=kernel,
      dilation_rate=dilation_rate,
      strides=strides,
      padding=padding,
      data_format='NDHWC')
  return _postprocess_conv3d_output(x, data_format)


def pool2d(x,
           pool_size,
           strides=(1, 1),
           padding='valid',
           data_format=None,
           pool_mode='max'):
  """2D Pooling.

  Arguments:
      x: Tensor or variable.
      pool_size: tuple of 2 integers.
      strides: tuple of 2 integers.
      padding: one of `"valid"`, `"same"`.
      data_format: one of `"channels_first"`, `"channels_last"`.
      pool_mode: one of `"max"`, `"avg"`.

  Returns:
      A tensor, result of 2D pooling.

  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
      ValueError: if `pool_mode` is neither `max` or `avg`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format ' + str(data_format))

  padding = _preprocess_padding(padding)
  strides = (1,) + strides + (1,)
  pool_size = (1,) + pool_size + (1,)

  x = _preprocess_conv2d_input(x, data_format)

  if pool_mode == 'max':
    x = nn.max_pool(x, pool_size, strides, padding=padding)
  elif pool_mode == 'avg':
    x = nn.avg_pool(x, pool_size, strides, padding=padding)
  else:
    raise ValueError('Invalid pooling mode:', pool_mode)

  return _postprocess_conv2d_output(x, data_format)


def pool3d(x,
           pool_size,
           strides=(1, 1, 1),
           padding='valid',
           data_format=None,
           pool_mode='max'):
  """3D Pooling.

  Arguments:
      x: Tensor or variable.
      pool_size: tuple of 3 integers.
      strides: tuple of 3 integers.
      padding: one of `"valid"`, `"same"`.
      data_format: one of `"channels_first"`, `"channels_last"`.
      pool_mode: one of `"max"`, `"avg"`.

  Returns:
      A tensor, result of 3D pooling.

  Raises:
      ValueError: if `data_format` is neither
          `channels_last` or `channels_first`.
      ValueError: if `pool_mode` is neither `max` or `avg`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format ' + str(data_format))

  padding = _preprocess_padding(padding)
  strides = (1,) + strides + (1,)
  pool_size = (1,) + pool_size + (1,)

  x = _preprocess_conv3d_input(x, data_format)

  if pool_mode == 'max':
    x = nn.max_pool3d(x, pool_size, strides, padding=padding)
  elif pool_mode == 'avg':
    x = nn.avg_pool3d(x, pool_size, strides, padding=padding)
  else:
    raise ValueError('Invalid pooling mode:', pool_mode)

  return _postprocess_conv3d_output(x, data_format)


def bias_add(x, bias, data_format=None):
  """Adds a bias vector to a tensor.

  Arguments:
      x: Tensor or variable.
      bias: Bias tensor to add.
      data_format: Data format for 3D, 4D or 5D tensors:
          one of "channels_first", "channels_last".

  Returns:
      Output tensor.

  Raises:
      ValueError: In case of invalid `data_format` argument.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format ' + str(data_format))
  if ndim(x) == 5:
    if data_format == 'channels_first':
      x += reshape(bias, (1, int_shape(bias)[0], 1, 1, 1))
    elif data_format == 'channels_last':
      x += reshape(bias, (1, 1, 1, 1, int_shape(bias)[0]))
  elif ndim(x) == 4:
    if data_format == 'channels_first':
      # No support yet for NCHW in bias_add.
      x += reshape(bias, (1, int_shape(bias)[0], 1, 1))
    elif data_format == 'channels_last':
      x = nn.bias_add(x, bias, data_format='NHWC')
  elif ndim(x) == 3:
    if data_format == 'channels_first':
      x += reshape(bias, (1, int_shape(bias)[0], 1))
    elif data_format == 'channels_last':
      x += reshape(bias, (1, 1, int_shape(bias)[0]))
  else:
    x = nn.bias_add(x, bias)
  return x


# RANDOMNESS


def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
  """Returns a tensor with normal distribution of values.

  Arguments:
      shape: A tuple of integers, the shape of tensor to create.
      mean: A float, mean of the normal distribution to draw samples.
      stddev: A float, standard deviation of the normal distribution
          to draw samples.
      dtype: String, dtype of returned tensor.
      seed: Integer, random seed.

  Returns:
      A tensor.
  """
  if dtype is None:
    dtype = floatx()
  if seed is None:
    seed = np.random.randint(10e6)
  return random_ops.random_normal(
      shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)


def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
  """Returns a tensor with uniform distribution of values.

  Arguments:
      shape: A tuple of integers, the shape of tensor to create.
      minval: A float, lower boundary of the uniform distribution
          to draw samples.
      maxval: A float, upper boundary of the uniform distribution
          to draw samples.
      dtype: String, dtype of returned tensor.
      seed: Integer, random seed.

  Returns:
      A tensor.
  """
  if dtype is None:
    dtype = floatx()
  if seed is None:
    seed = np.random.randint(10e6)
  return random_ops.random_uniform(
      shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed)


def random_binomial(shape, p=0.0, dtype=None, seed=None):
  """Returns a tensor with random binomial distribution of values.

  Arguments:
      shape: A tuple of integers, the shape of tensor to create.
      p: A float, `0. <= p <= 1`, probability of binomial distribution.
      dtype: String, dtype of returned tensor.
      seed: Integer, random seed.

  Returns:
      A tensor.
  """
  if dtype is None:
    dtype = floatx()
  if seed is None:
    seed = np.random.randint(10e6)
  return array_ops.where(
      random_ops.random_uniform(shape, dtype=dtype, seed=seed) <= p,
      array_ops.ones(shape, dtype=dtype), array_ops.zeros(shape, dtype=dtype))


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
  """Returns a tensor with truncated random normal distribution of values.

  The generated values follow a normal distribution
  with specified mean and standard deviation,
  except that values whose magnitude is more than
  two standard deviations from the mean are dropped and re-picked.

  Arguments:
      shape: A tuple of integers, the shape of tensor to create.
      mean: Mean of the values.
      stddev: Standard deviation of the values.
      dtype: String, dtype of returned tensor.
      seed: Integer, random seed.

  Returns:
      A tensor.
  """
  if dtype is None:
    dtype = floatx()
  if seed is None:
    seed = np.random.randint(10e6)
  return random_ops.truncated_normal(
      shape, mean, stddev, dtype=dtype, seed=seed)


# CTC
# tensorflow has a native implemenation, but it uses sparse tensors
# and therefore requires a wrapper for Keras. The functions below convert
# dense to sparse tensors and also wraps up the beam search code that is
# in tensorflow's CTC implementation


def ctc_label_dense_to_sparse(labels, label_lengths):
  """Converts CTC labels from dense to sparse.

  Arguments:
      labels: dense CTC labels.
      label_lengths: length of the labels.

  Returns:
      A sparse tensor representation of the lablels.
  """
  label_shape = array_ops.shape(labels)
  num_batches_tns = array_ops.stack([label_shape[0]])
  max_num_labels_tns = array_ops.stack([label_shape[1]])

  def range_less_than(_, current_input):
    return array_ops.expand_dims(
        math_ops.range(label_shape[1]), 0) < array_ops.fill(
            max_num_labels_tns, current_input)

  init = math_ops.cast(
      array_ops.fill([1, label_shape[1]], 0), dtypes_module.bool)
  dense_mask = functional_ops.scan(
      range_less_than, label_lengths, initializer=init, parallel_iterations=1)
  dense_mask = dense_mask[:, 0, :]

  label_array = array_ops.reshape(
      array_ops.tile(math_ops.range(0, label_shape[1]), num_batches_tns),
      label_shape)
  label_ind = array_ops.boolean_mask(label_array, dense_mask)

  batch_array = array_ops.transpose(
      array_ops.reshape(
          array_ops.tile(math_ops.range(0, label_shape[0]), max_num_labels_tns),
          reverse(label_shape, 0)))
  batch_ind = array_ops.boolean_mask(batch_array, dense_mask)
  indices = array_ops.transpose(
      array_ops.reshape(concatenate([batch_ind, label_ind], axis=0), [2, -1]))

  vals_sparse = array_ops.gather_nd(labels, indices)

  return sparse_tensor.SparseTensor(
      math_ops.to_int64(indices), vals_sparse, math_ops.to_int64(label_shape))


def ctc_batch_cost(y_true, y_pred, input_length, label_length):
  """Runs CTC loss algorithm on each batch element.

  Arguments:
      y_true: tensor `(samples, max_string_length)`
          containing the truth labels.
      y_pred: tensor `(samples, time_steps, num_categories)`
          containing the prediction, or output of the softmax.
      input_length: tensor `(samples, 1)` containing the sequence length for
          each batch item in `y_pred`.
      label_length: tensor `(samples, 1)` containing the sequence length for
          each batch item in `y_true`.

  Returns:
      Tensor with shape (samples,1) containing the
          CTC loss of each element.
  """
  label_length = math_ops.to_int32(array_ops.squeeze(label_length))
  input_length = math_ops.to_int32(array_ops.squeeze(input_length))
  sparse_labels = math_ops.to_int32(
      ctc_label_dense_to_sparse(y_true, label_length))

  y_pred = math_ops.log(array_ops.transpose(y_pred, perm=[1, 0, 2]) + 1e-8)

  return array_ops.expand_dims(
      ctc.ctc_loss(
          inputs=y_pred, labels=sparse_labels, sequence_length=input_length), 1)


def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
  """Decodes the output of a softmax.

  Can use either greedy search (also known as best path)
  or a constrained dictionary search.

  Arguments:
      y_pred: tensor `(samples, time_steps, num_categories)`
          containing the prediction, or output of the softmax.
      input_length: tensor `(samples, )` containing the sequence length for
          each batch item in `y_pred`.
      greedy: perform much faster best-path search if `true`.
          This does not use a dictionary.
      beam_width: if `greedy` is `false`: a beam search decoder will be used
          with a beam of this width.
      top_paths: if `greedy` is `false`,
          how many of the most probable paths will be returned.

  Returns:
      Tuple:
          List: if `greedy` is `true`, returns a list of one element that
              contains the decoded sequence.
              If `false`, returns the `top_paths` most probable
              decoded sequences.
              Important: blank labels are returned as `-1`.
          Tensor `(top_paths, )` that contains
              the log probability of each decoded sequence.
  """
  y_pred = math_ops.log(array_ops.transpose(y_pred, perm=[1, 0, 2]) + 1e-8)
  input_length = math_ops.to_int32(input_length)

  if greedy:
    (decoded, log_prob) = ctc.ctc_greedy_decoder(
        inputs=y_pred, sequence_length=input_length)
  else:
    (decoded, log_prob) = ctc.ctc_beam_search_decoder(
        inputs=y_pred,
        sequence_length=input_length,
        beam_width=beam_width,
        top_paths=top_paths)
  decoded_dense = [
      sparse_ops.sparse_to_dense(
          st.indices, st.dense_shape, st.values, default_value=-1)
      for st in decoded
  ]
  return (decoded_dense, log_prob)


# HIGH ORDER FUNCTIONS


def map_fn(fn, elems, name=None, dtype=None):
  """Map the function fn over the elements elems and return the outputs.

  Arguments:
      fn: Callable that will be called upon each element in elems
      elems: tensor
      name: A string name for the map node in the graph
      dtype: Output data type.

  Returns:
      Tensor with dtype `dtype`.
  """
  return functional_ops.map_fn(fn, elems, name=name, dtype=dtype)


def foldl(fn, elems, initializer=None, name=None):
  """Reduce elems using fn to combine them from left to right.

  Arguments:
      fn: Callable that will be called upon each element in elems and an
          accumulator, for instance `lambda acc, x: acc + x`
      elems: tensor
      initializer: The first value used (`elems[0]` in case of None)
      name: A string name for the foldl node in the graph

  Returns:
      Tensor with same type and shape as `initializer`.
  """
  return functional_ops.foldl(fn, elems, initializer=initializer, name=name)


def foldr(fn, elems, initializer=None, name=None):
  """Reduce elems using fn to combine them from right to left.

  Arguments:
      fn: Callable that will be called upon each element in elems and an
          accumulator, for instance `lambda acc, x: acc + x`
      elems: tensor
      initializer: The first value used (`elems[-1]` in case of None)
      name: A string name for the foldr node in the graph

  Returns:
      Same type and shape as initializer
  """
  return functional_ops.foldr(fn, elems, initializer=initializer, name=name)


# Load Keras default configuration from config file if present.
_keras_base_dir = os.path.expanduser('~')
_keras_dir = os.path.join(_keras_base_dir, '.keras')
_config_path = os.path.expanduser(os.path.join(_keras_dir, 'keras.json'))
if os.path.exists(_config_path):
  try:
    _config = json.load(open(_config_path))
  except ValueError:
    _config = {}
  _floatx = _config.get('floatx', floatx())
  assert _floatx in {'float16', 'float32', 'float64'}
  _epsilon = _config.get('epsilon', epsilon())
  assert isinstance(_epsilon, float)
  _image_data_format = _config.get('image_data_format', image_data_format())
  assert _image_data_format in {'channels_last', 'channels_first'}
  set_floatx(_floatx)
  set_epsilon(_epsilon)
  set_image_data_format(_image_data_format)

# Save config file.
if not os.path.exists(_keras_dir):
  try:
    os.makedirs(_keras_dir)
  except OSError:
    # Except permission denied and potential race conditions
    # in multi-threaded environments.
    pass

if not os.path.exists(_config_path):
  _config = {
      'floatx': floatx(),
      'epsilon': epsilon(),
      'backend': 'tensorflow',
      'image_data_format': image_data_format()
  }
  try:
    with open(_config_path, 'w') as f:
      f.write(json.dumps(_config, indent=4))
  except IOError:
    # Except permission denied.
    pass
