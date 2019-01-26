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

import collections
import itertools
import json
import os
import threading
import weakref

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_module
from tensorflow.python.eager import context
from tensorflow.python.eager import function as eager_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend_config
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients as gradients_module
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables as variables_module

from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import keras_export

py_all = all
py_sum = sum

# INTERNAL UTILS

# The internal graph maintained by Keras and used by the symbolic Keras APIs
# while executing eagerly (such as the functional API for model-building).
_GRAPH = None

# This is a thread local object that will hold the default internal TF session
# used by Keras. It can be set manually via `set_session(sess)`.
_SESSION = threading.local()

# This dictionary holds a mapping {graph: learning_phase}.
# A learning phase is a bool tensor used to run Keras models in
# either train mode (learning_phase == 1) or test mode (learning_phase == 0).
_GRAPH_LEARNING_PHASES = weakref.WeakKeyDictionary()


# _DUMMY_EAGER_GRAPH is used as a key in _GRAPH_LEARNING_PHASES.
# We keep a separate reference to it to make sure it does not get removed from
# _GRAPH_LEARNING_PHASES.
_DUMMY_EAGER_GRAPH = threading.local()

# This boolean flag can be set to True to leave variable initialization
# up to the user.
# Change its value via `manual_variable_initialization(value)`.
_MANUAL_VAR_INIT = False

# This list holds the available devices.
# It is populated when `_get_available_gpus()` is called for the first time.
# We assume our devices don't change henceforth.
_LOCAL_DEVICES = None

# This dictionary holds a mapping between a graph and variables to initialize
# in the graph.
_GRAPH_VARIABLES = weakref.WeakKeyDictionary()

# This dictionary holds a mapping between a graph and TF optimizers created in
# the graph.
_GRAPH_TF_OPTIMIZERS = weakref.WeakKeyDictionary()

# The below functions are kept accessible from backend for compatibility.
epsilon = backend_config.epsilon
floatx = backend_config.floatx
image_data_format = backend_config.image_data_format
set_epsilon = backend_config.set_epsilon
set_floatx = backend_config.set_floatx
set_image_data_format = backend_config.set_image_data_format


@keras_export('keras.backend.backend')
def backend():
  """Publicly accessible method for determining the current backend.

  Only exists for API compatibility with multi-backend Keras.

  Returns:
      The string "tensorflow".
  """
  return 'tensorflow'


@keras_export('keras.backend.cast_to_floatx')
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
  return np.asarray(x, dtype=floatx())


# A global dictionary mapping graph objects to an index of counters used
# for various layer names in each graph.
# Allows to give unique autogenerated names to layers, in a graph-specific way.
PER_GRAPH_LAYER_NAME_UIDS = weakref.WeakKeyDictionary()


@keras_export('keras.backend.get_uid')
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
  graph = get_graph()
  if graph not in PER_GRAPH_LAYER_NAME_UIDS:
    PER_GRAPH_LAYER_NAME_UIDS[graph] = collections.defaultdict(int)
  layer_name_uids = PER_GRAPH_LAYER_NAME_UIDS[graph]
  layer_name_uids[prefix] += 1
  return layer_name_uids[prefix]


@keras_export('keras.backend.reset_uids')
def reset_uids():
  """Resets graph identifiers.
  """
  per_graph_layer_name_uids = PER_GRAPH_LAYER_NAME_UIDS
  keys = list(per_graph_layer_name_uids.keys())
  for key in keys:
    del per_graph_layer_name_uids[key]


@keras_export('keras.backend.clear_session')
def clear_session():
  """Destroys the current TF graph and creates a new one.

  Useful to avoid clutter from old models / layers.
  """
  global _SESSION
  global _GRAPH_LEARNING_PHASES  # pylint: disable=global-variable-not-assigned
  global _GRAPH_VARIABLES  # pylint: disable=global-variable-not-assigned
  global _GRAPH_TF_OPTIMIZERS  # pylint: disable=global-variable-not-assigned
  ops.reset_default_graph()
  reset_uids()
  _SESSION.session = None
  graph = get_graph()
  with graph.as_default():
    phase = array_ops.placeholder_with_default(
        False, shape=(), name='keras_learning_phase')
    _GRAPH_LEARNING_PHASES = {}
    _GRAPH_LEARNING_PHASES[graph] = phase
    _GRAPH_VARIABLES.pop(graph, None)
    _GRAPH_TF_OPTIMIZERS.pop(graph, None)


@keras_export('keras.backend.manual_variable_initialization')
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


@keras_export('keras.backend.learning_phase')
def learning_phase():
  """Returns the learning phase flag.

  The learning phase flag is a bool tensor (0 = test, 1 = train)
  to be passed as input to any Keras function
  that uses a different behavior at train time and test time.

  Returns:
      Learning phase (scalar integer tensor or Python integer).
  """
  if ops.get_default_graph() is _GRAPH:
    # Don't enter an init_scope for the learning phase if eager execution
    # is enabled but we're inside the Keras workspace graph.
    return symbolic_learning_phase()
  with ops.init_scope():
    # We always check & set the learning phase inside the init_scope,
    # otherwise the wrong default_graph will be used to look up the learning
    # phase inside of functions & defuns.
    #
    # This is because functions & defuns (both in graph & in eager mode)
    # will always execute non-eagerly using a function-specific default
    # subgraph.
    if context.executing_eagerly():
      if _DUMMY_EAGER_GRAPH not in _GRAPH_LEARNING_PHASES:
        # Fallback to inference mode as default.
        return 0
      return _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH]
    return symbolic_learning_phase()


def symbolic_learning_phase():
  graph = get_graph()
  with graph.as_default():
    if graph not in _GRAPH_LEARNING_PHASES:
      phase = array_ops.placeholder_with_default(
          False, shape=(), name='keras_learning_phase')
      _GRAPH_LEARNING_PHASES[graph] = phase
    return _GRAPH_LEARNING_PHASES[graph]


@keras_export('keras.backend.set_learning_phase')
def set_learning_phase(value):
  """Sets the learning phase to a fixed value.

  Arguments:
      value: Learning phase value, either 0 or 1 (integers).

  Raises:
      ValueError: if `value` is neither `0` nor `1`.
  """
  global _GRAPH_LEARNING_PHASES  # pylint: disable=global-variable-not-assigned
  if value not in {0, 1}:
    raise ValueError('Expected learning phase to be 0 or 1.')
  with ops.init_scope():
    if context.executing_eagerly():
      # In an eager context, the learning phase values applies to both the eager
      # context and the internal Keras graph.
      _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH] = value
    _GRAPH_LEARNING_PHASES[get_graph()] = value


def set_eager_learning_phase(value):
  """Internal utility that sets the learning phase in eager execution only.

  Arguments:
      value: Learning phase value, either 0 or 1 (integers).
  """
  global _GRAPH_LEARNING_PHASES  # pylint: disable=global-variable-not-assigned
  assert value in {0, 1}
  assert context.executing_eagerly()
  _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH] = value


@keras_export('keras.backend.learning_phase_scope')
@tf_contextlib.contextmanager
def learning_phase_scope(value):
  """Provides a scope within which the learning phase is equal to `value`.

  The learning phase gets restored to its original value upon exiting the scope.

  Arguments:
     value: Learning phase value, either 0 or 1 (integers).

  Yields:
    None.

  Raises:
     ValueError: if `value` is neither `0` nor `1`.
  """
  global _GRAPH_LEARNING_PHASES  # pylint: disable=global-variable-not-assigned
  if value not in {0, 1}:
    raise ValueError('Expected learning phase to be 0 or 1.')

  with ops.init_scope():
    if context.executing_eagerly():
      previous_eager_value = _GRAPH_LEARNING_PHASES.get(
          _DUMMY_EAGER_GRAPH, None)
    previous_graph_value = _GRAPH_LEARNING_PHASES.get(get_graph(), None)

  try:
    set_learning_phase(value)
    yield
  finally:
    # Restore learning phase to initial value.
    with ops.init_scope():
      if context.executing_eagerly():
        if previous_eager_value is not None:
          _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH] = previous_eager_value
        elif _DUMMY_EAGER_GRAPH in _GRAPH_LEARNING_PHASES:
          del _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH]

      graph = get_graph()
      if previous_graph_value is not None:
        _GRAPH_LEARNING_PHASES[graph] = previous_graph_value
      elif graph in _GRAPH_LEARNING_PHASES:
        del _GRAPH_LEARNING_PHASES[graph]

@tf_contextlib.contextmanager
def eager_learning_phase_scope(value):
  """Internal scope that sets the learning phase in eager execution only.

  Arguments:
      value: Learning phase value, either 0 or 1 (integers).

  Yields:
    None.

  Raises:
     ValueError: if `value` is neither `0` nor `1`.
  """
  global _GRAPH_LEARNING_PHASES  # pylint: disable=global-variable-not-assigned
  assert value in {0, 1}
  assert context.executing_eagerly()
  previous_value = learning_phase()
  try:
    _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH] = value
    yield
  finally:
    # Restore learning phase to initial value.
    _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH] = previous_value


def _get_session():
  """Returns the session object for the current thread."""
  global _SESSION
  default_session = ops.get_default_session()
  if default_session is not None:
    session = default_session
  else:
    if getattr(_SESSION, 'session', None) is None:
      _SESSION.session = session_module.Session(
          config=get_default_session_config())
    session = _SESSION.session
  return session


@keras_export(v1=['keras.backend.get_session'])
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
  session = _get_session()
  if not _MANUAL_VAR_INIT:
    with session.graph.as_default():
      _initialize_variables(session)
  return session


def get_graph():
  if context.executing_eagerly():
    global _GRAPH
    if _GRAPH is None:
      _GRAPH = func_graph.FuncGraph('keras_graph')
    return _GRAPH
  else:
    return ops.get_default_graph()


@keras_export('keras.backend.set_session')
def set_session(session):
  """Sets the global TensorFlow session.

  Arguments:
      session: A TF Session.
  """
  global _SESSION
  _SESSION.session = session


def get_default_session_config():
  if not os.environ.get('OMP_NUM_THREADS'):
    config = config_pb2.ConfigProto(allow_soft_placement=True)
  else:
    num_thread = int(os.environ.get('OMP_NUM_THREADS'))
    config = config_pb2.ConfigProto(
        intra_op_parallelism_threads=num_thread, allow_soft_placement=True)
  return config


# DEVICE MANIPULATION


class _TfDeviceCaptureOp(object):
  """Class for capturing the TF device scope."""

  def __init__(self):
    self.device = None

  def _set_device(self, device):
    """This method captures TF's explicit device scope setting."""
    self.device = device


def _get_current_tf_device():
  """Return explicit device of current context, otherwise returns `None`.

  Returns:
      If the current device scope is explicitly set, it returns a string with
      the device (`CPU` or `GPU`). If the scope is not explicitly set, it will
      return `None`.
  """
  graph = get_graph()
  op = _TfDeviceCaptureOp()
  graph._apply_device_functions(op)
  return op.device


def _is_current_explicit_device(device_type):
  """Check if the current device is explicitly set on the device type specified.

  Arguments:
      device_type: A string containing `GPU` or `CPU` (case-insensitive).

  Returns:
      A boolean indicating if the current device scope is explicitly set on the
      device type.

  Raises:
      ValueError: If the `device_type` string indicates an unsupported device.
  """
  device_type = device_type.upper()
  if device_type not in ['CPU', 'GPU']:
    raise ValueError('`device_type` should be either "CPU" or "GPU".')
  device = _get_current_tf_device()
  return device is not None and device.device_type == device_type.upper()


def _get_available_gpus():
  """Get a list of available gpu devices (formatted as strings).

  Returns:
      A list of available GPU devices.
  """
  if ops.executing_eagerly_outside_functions():
    # Returns names of devices directly.
    return [name for name in context.list_devices() if 'GPU' in name]

  global _LOCAL_DEVICES
  if _LOCAL_DEVICES is None:
    _LOCAL_DEVICES = get_session().list_devices()
  return [x.name for x in _LOCAL_DEVICES if x.device_type == 'GPU']


def _has_nchw_support():
  """Check whether the current scope supports NCHW ops.

  TensorFlow does not support NCHW on CPU. Therefore we check if we are not
  explicitly put on
  CPU, and have GPUs available. In this case there will be soft-placing on the
  GPU device.

  Returns:
      bool: if the current scope device placement would support nchw
  """
  explicitly_on_cpu = _is_current_explicit_device('CPU')
  gpus_available = bool(_get_available_gpus())
  return not explicitly_on_cpu and gpus_available


# VARIABLE MANIPULATION


def _to_tensor(x, dtype):
  """Convert the input `x` to a tensor of type `dtype`.

  Arguments:
      x: An object to be converted (numpy array, list, tensors).
      dtype: The destination type.

  Returns:
      A tensor.
  """
  return ops.convert_to_tensor(x, dtype=dtype)


@keras_export('keras.backend.is_sparse')
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


@keras_export('keras.backend.to_dense')
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


@keras_export('keras.backend.variable')
def variable(value, dtype=None, name=None, constraint=None):
  """Instantiates a variable and returns it.

  Arguments:
      value: Numpy array, initial value of the tensor.
      dtype: Tensor type.
      name: Optional name string for the tensor.
      constraint: Optional projection function to be
          applied to the variable after an optimizer update.

  Returns:
      A variable instance (with Keras metadata included).

  Examples:
  ```python
      >>> import numpy as np
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
    v._keras_shape = sparse_coo.shape
    return v
  v = resource_variable_ops.ResourceVariable(
      value,
      dtype=dtypes_module.as_dtype(dtype),
      name=name,
      constraint=constraint)
  if isinstance(value, np.ndarray):
    v._keras_shape = value.shape
  elif hasattr(value, 'shape'):
    v._keras_shape = int_shape(value)
  track_variable(v)
  return v


def track_tf_optimizer(tf_optimizer):
  """Tracks the given TF optimizer for initialization of its variables."""
  if context.executing_eagerly():
    return
  graph = get_graph()
  optimizers = _GRAPH_TF_OPTIMIZERS.setdefault(graph, weakref.WeakSet())
  optimizers.add(tf_optimizer)


def track_variable(v):
  """Tracks the given variable for initialization."""
  if context.executing_eagerly():
    return
  graph = v.graph if hasattr(v, 'graph') else get_graph()
  if graph not in _GRAPH_VARIABLES:
    _GRAPH_VARIABLES[graph] = weakref.WeakSet()
  _GRAPH_VARIABLES[graph].add(v)


def _get_variables(graph=None):
  """Returns variables corresponding to the given graph for initialization."""
  assert not context.executing_eagerly()
  variables = _GRAPH_VARIABLES.setdefault(graph, weakref.WeakSet())
  for opt in _GRAPH_TF_OPTIMIZERS.get(graph, set()):
    variables.update(opt.optimizer.variables())
  return variables


def _initialize_variables(session):
  """Utility to initialize uninitialized variables on the fly."""
  variables = _get_variables(get_graph())
  candidate_vars = []
  for v in variables:
    if not getattr(v, '_keras_initialized', False):
      candidate_vars.append(v)
  if candidate_vars:
    # This step is expensive, so we only run it on variables not already
    # marked as initialized.
    is_initialized = session.run(
        [variables_module.is_variable_initialized(v) for v in candidate_vars])
    uninitialized_vars = []
    for flag, v in zip(is_initialized, candidate_vars):
      if not flag:
        uninitialized_vars.append(v)
      v._keras_initialized = True
    if uninitialized_vars:
      session.run(variables_module.variables_initializer(uninitialized_vars))


@keras_export('keras.backend.constant')
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


def is_keras_tensor(x):
  """Returns whether `x` is a Keras tensor.

  A "Keras tensor" is a tensor that was returned by a Keras layer,
  (`Layer` class) or by `Input`.

  Arguments:
      x: A candidate tensor.

  Returns:
      A boolean: Whether the argument is a Keras tensor.

  Raises:
      ValueError: In case `x` is not a symbolic tensor.

  Examples:
  ```python
      >>> import tensorflow as tf
      >>> import numpy
      >>> from keras import backend as K
      >>> from keras.layers import Input, Dense
      >>> np_var = numpy.array([1, 2])
      >>> K.is_keras_tensor(np_var) # A numpy array is not a symbolic tensor.
      ValueError
      >>> k_var = tf.placeholder('float32', shape=(1,1))
      >>> K.is_keras_tensor(k_var) # A variable indirectly created outside of
      keras is not a Keras tensor.
      False
      >>> keras_var = K.variable(np_var)
      >>> K.is_keras_tensor(keras_var)  # A variable created with the keras
      backend is not a Keras tensor.
      False
      >>> keras_placeholder = K.placeholder(shape=(2, 4, 5))
      >>> K.is_keras_tensor(keras_placeholder)  # A placeholder is not a Keras
      tensor.
      False
      >>> keras_input = Input([10])
      >>> K.is_keras_tensor(keras_input) # An Input is a Keras tensor.
      True
      >>> keras_layer_output = Dense(10)(keras_input)
      >>> K.is_keras_tensor(keras_layer_output) # Any Keras layer output is a
      Keras tensor.
      True
  ```
  """
  if not isinstance(x, (ops.Tensor,
                        variables_module.Variable,
                        sparse_tensor.SparseTensor)):
    raise ValueError('Unexpectedly found an instance of type `' + str(type(x)) +
                     '`. Expected a symbolic tensor instance.')
  return hasattr(x, '_keras_history')


@keras_export('keras.backend.placeholder')
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

  Raises:
      ValueError: If called with eager execution.

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
  with get_graph().as_default():
    if sparse:
      x = array_ops.sparse_placeholder(dtype, shape=shape, name=name)
    else:
      x = array_ops.placeholder(dtype, shape=shape, name=name)
  return x


def is_placeholder(x):
  """Returns whether `x` is a placeholder.

  Arguments:
      x: A candidate placeholder.

  Returns:
      Boolean.
  """
  try:
    return x.op.type == 'Placeholder'
  except AttributeError:
    return False


@keras_export('keras.backend.shape')
def shape(x):
  """Returns the symbolic shape of a tensor or variable.

  Arguments:
      x: A tensor or variable.

  Returns:
      A symbolic shape (which is itself a tensor).

  Examples:

  ```python
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


@keras_export('keras.backend.int_shape')
def int_shape(x):
  """Returns the shape of tensor or variable as a tuple of int or None entries.

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
  try:
    shape = x.shape
    if not isinstance(shape, tuple):
      shape = tuple(shape.as_list())
    return shape
  except ValueError:
    return None


@keras_export('keras.backend.ndim')
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
  dims = x.shape._dims
  if dims is not None:
    return len(dims)
  return None


@keras_export('keras.backend.dtype')
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
  return x.dtype.base_dtype.name


@keras_export('keras.backend.eval')
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
  return get_value(to_dense(x))


@keras_export('keras.backend.zeros')
def zeros(shape, dtype=None, name=None):
  """Instantiates an all-zeros variable and returns it.

  Arguments:
      shape: Tuple of integers, shape of returned Keras variable
      dtype: String, data type of returned Keras variable
      name: String, name of returned Keras variable

  Returns:
      A variable (including Keras metadata), filled with `0.0`.
      Note that if `shape` was symbolic, we cannot return a variable,
      and will return a dynamically-shaped tensor instead.

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
  with ops.init_scope():
    if dtype is None:
      dtype = floatx()
    tf_dtype = dtypes_module.as_dtype(dtype)
    v = array_ops.zeros(shape=shape, dtype=tf_dtype, name=name)
    if py_all(v.shape.as_list()):
      return variable(v, dtype=dtype, name=name)
    track_variable(v)
    return v


@keras_export('keras.backend.ones')
def ones(shape, dtype=None, name=None):
  """Instantiates an all-ones variable and returns it.

  Arguments:
      shape: Tuple of integers, shape of returned Keras variable.
      dtype: String, data type of returned Keras variable.
      name: String, name of returned Keras variable.

  Returns:
      A Keras variable, filled with `1.0`.
      Note that if `shape` was symbolic, we cannot return a variable,
      and will return a dynamically-shaped tensor instead.

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
  with ops.init_scope():
    if dtype is None:
      dtype = floatx()
    tf_dtype = dtypes_module.as_dtype(dtype)
    v = array_ops.ones(shape=shape, dtype=tf_dtype, name=name)
    if py_all(v.shape.as_list()):
      return variable(v, dtype=dtype, name=name)
    track_variable(v)
    return v


@keras_export('keras.backend.eye')
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
  if dtype is None:
    dtype = floatx()
  tf_dtype = dtypes_module.as_dtype(dtype)
  return variable(linalg_ops.eye(size, dtype=tf_dtype), dtype, name)


@keras_export('keras.backend.zeros_like')
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


@keras_export('keras.backend.ones_like')
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


def identity(x, name=None):
  """Returns a tensor with the same content as the input tensor.

  Arguments:
      x: The input tensor.
      name: String, name for the variable to create.

  Returns:
      A tensor of the same shape, type and content.
  """
  return array_ops.identity(x, name=name)


@keras_export('keras.backend.random_uniform_variable')
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
  tf_dtype = dtypes_module.as_dtype(dtype)
  if seed is None:
    # ensure that randomness is conditioned by the Numpy RNG
    seed = np.random.randint(10e8)
  value = init_ops.random_uniform_initializer(
      low, high, dtype=tf_dtype, seed=seed)(shape)
  return variable(value, dtype=dtype, name=name)


@keras_export('keras.backend.random_normal_variable')
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
  tf_dtype = dtypes_module.as_dtype(dtype)
  if seed is None:
    # ensure that randomness is conditioned by the Numpy RNG
    seed = np.random.randint(10e8)
  value = init_ops.random_normal_initializer(
      mean, scale, dtype=tf_dtype, seed=seed)(shape)
  return variable(value, dtype=dtype, name=name)


@keras_export('keras.backend.count_params')
def count_params(x):
  """Returns the static number of elements in a variable or tensor.

  Arguments:
      x: Variable or tensor.

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
  return np.prod(x.shape.as_list())


@keras_export('keras.backend.cast')
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


@keras_export('keras.backend.update')
def update(x, new_x):
  return state_ops.assign(x, new_x)


@keras_export('keras.backend.update_add')
def update_add(x, increment):
  """Update the value of `x` by adding `increment`.

  Arguments:
      x: A Variable.
      increment: A tensor of same shape as `x`.

  Returns:
      The variable `x` updated.
  """
  return state_ops.assign_add(x, increment)


@keras_export('keras.backend.update_sub')
def update_sub(x, decrement):
  """Update the value of `x` by subtracting `decrement`.

  Arguments:
      x: A Variable.
      decrement: A tensor of same shape as `x`.

  Returns:
      The variable `x` updated.
  """
  return state_ops.assign_sub(x, decrement)


@keras_export('keras.backend.moving_average_update')
def moving_average_update(x, value, momentum):
  """Compute the moving average of a variable.

  Arguments:
      x: A Variable.
      value: A tensor with the same shape as `variable`.
      momentum: The moving average momentum.

  Returns:
      An Operation to update the variable.
  """
  # `training` is higher-up than the Keras backend in the abstraction hierarchy.
  # In particular, `training` depends on layers, and thus on Keras.
  # moving_averages, being low-level ops, should not be part of the training
  # module.
  from tensorflow.python.training import moving_averages  # pylint: disable=g-import-not-at-top
  return moving_averages.assign_moving_average(
      x, value, momentum, zero_debias=True)


# LINEAR ALGEBRA


@keras_export('keras.backend.dot')
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


@keras_export('keras.backend.batch_dot')
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
  if axes is None:
    # behaves like tf.batch_matmul as default
    axes = [x_ndim - 1, y_ndim - 2]
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
    adj_x = None if axes[0] == ndim(x) - 1 else True
    adj_y = True if axes[1] == ndim(y) - 1 else None
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


@keras_export('keras.backend.transpose')
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


@keras_export('keras.backend.gather')
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


@keras_export('keras.backend.max')
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
  return math_ops.reduce_max(x, axis, keepdims)


@keras_export('keras.backend.min')
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
  return math_ops.reduce_min(x, axis, keepdims)


@keras_export('keras.backend.sum')
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
  return math_ops.reduce_sum(x, axis, keepdims)


@keras_export('keras.backend.prod')
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
  return math_ops.reduce_prod(x, axis, keepdims)


@keras_export('keras.backend.cumsum')
def cumsum(x, axis=0):
  """Cumulative sum of the values in a tensor, alongside the specified axis.

  Arguments:
      x: A tensor or variable.
      axis: An integer, the axis to compute the sum.

  Returns:
      A tensor of the cumulative sum of values of `x` along `axis`.
  """
  return math_ops.cumsum(x, axis=axis)


@keras_export('keras.backend.cumprod')
def cumprod(x, axis=0):
  """Cumulative product of the values in a tensor, alongside the specified axis.

  Arguments:
      x: A tensor or variable.
      axis: An integer, the axis to compute the product.

  Returns:
      A tensor of the cumulative product of values of `x` along `axis`.
  """
  return math_ops.cumprod(x, axis=axis)


@keras_export('keras.backend.var')
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
  if x.dtype.base_dtype == dtypes_module.bool:
    x = math_ops.cast(x, floatx())
  return math_ops.reduce_variance(x, axis=axis, keepdims=keepdims)


@keras_export('keras.backend.std')
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
  if x.dtype.base_dtype == dtypes_module.bool:
    x = math_ops.cast(x, floatx())
  return math_ops.reduce_std(x, axis=axis, keepdims=keepdims)


@keras_export('keras.backend.mean')
def mean(x, axis=None, keepdims=False):
  """Mean of a tensor, alongside the specified axis.

  Arguments:
      x: A tensor or variable.
      axis: A list of integer. Axes to compute the mean.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1 for each entry in `axis`. If `keepdims` is `True`,
          the reduced dimensions are retained with length 1.

  Returns:
      A tensor with the mean of elements of `x`.
  """
  if x.dtype.base_dtype == dtypes_module.bool:
    x = math_ops.cast(x, floatx())
  return math_ops.reduce_mean(x, axis, keepdims)


@keras_export('keras.backend.any')
def any(x, axis=None, keepdims=False):
  """Bitwise reduction (logical OR).

  Arguments:
      x: Tensor or variable.
      axis: axis along which to perform the reduction.
      keepdims: whether the drop or broadcast the reduction axes.

  Returns:
      A uint8 tensor (0s and 1s).
  """
  x = math_ops.cast(x, dtypes_module.bool)
  return math_ops.reduce_any(x, axis, keepdims)


@keras_export('keras.backend.all')
def all(x, axis=None, keepdims=False):
  """Bitwise reduction (logical AND).

  Arguments:
      x: Tensor or variable.
      axis: axis along which to perform the reduction.
      keepdims: whether the drop or broadcast the reduction axes.

  Returns:
      A uint8 tensor (0s and 1s).
  """
  x = math_ops.cast(x, dtypes_module.bool)
  return math_ops.reduce_all(x, axis, keepdims)


@keras_export('keras.backend.argmax')
def argmax(x, axis=-1):
  """Returns the index of the maximum value along an axis.

  Arguments:
      x: Tensor or variable.
      axis: axis along which to perform the reduction.

  Returns:
      A tensor.
  """
  return math_ops.argmax(x, axis)


@keras_export('keras.backend.argmin')
def argmin(x, axis=-1):
  """Returns the index of the minimum value along an axis.

  Arguments:
      x: Tensor or variable.
      axis: axis along which to perform the reduction.

  Returns:
      A tensor.
  """
  return math_ops.argmin(x, axis)


@keras_export('keras.backend.square')
def square(x):
  """Element-wise square.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.square(x)


@keras_export('keras.backend.abs')
def abs(x):
  """Element-wise absolute value.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.abs(x)


@keras_export('keras.backend.sqrt')
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


@keras_export('keras.backend.exp')
def exp(x):
  """Element-wise exponential.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.exp(x)


@keras_export('keras.backend.log')
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
  return math_ops.reduce_logsumexp(x, axis, keepdims)


@keras_export('keras.backend.round')
def round(x):
  """Element-wise rounding to the closest integer.

  In case of tie, the rounding mode used is "half to even".

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.round(x)


@keras_export('keras.backend.sign')
def sign(x):
  """Element-wise sign.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.sign(x)


@keras_export('keras.backend.pow')
def pow(x, a):
  """Element-wise exponentiation.

  Arguments:
      x: Tensor or variable.
      a: Python integer.

  Returns:
      A tensor.
  """
  return math_ops.pow(x, a)


@keras_export('keras.backend.clip')
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


@keras_export('keras.backend.equal')
def equal(x, y):
  """Element-wise equality between two tensors.

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  """
  return math_ops.equal(x, y)


@keras_export('keras.backend.not_equal')
def not_equal(x, y):
  """Element-wise inequality between two tensors.

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  """
  return math_ops.not_equal(x, y)


@keras_export('keras.backend.greater')
def greater(x, y):
  """Element-wise truth value of (x > y).

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  """
  return math_ops.greater(x, y)


@keras_export('keras.backend.greater_equal')
def greater_equal(x, y):
  """Element-wise truth value of (x >= y).

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  """
  return math_ops.greater_equal(x, y)


@keras_export('keras.backend.less')
def less(x, y):
  """Element-wise truth value of (x < y).

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  """
  return math_ops.less(x, y)


@keras_export('keras.backend.less_equal')
def less_equal(x, y):
  """Element-wise truth value of (x <= y).

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  """
  return math_ops.less_equal(x, y)


@keras_export('keras.backend.maximum')
def maximum(x, y):
  """Element-wise maximum of two tensors.

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.maximum(x, y)


@keras_export('keras.backend.minimum')
def minimum(x, y):
  """Element-wise minimum of two tensors.

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.minimum(x, y)


@keras_export('keras.backend.sin')
def sin(x):
  """Computes sin of x element-wise.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.sin(x)


@keras_export('keras.backend.cos')
def cos(x):
  """Computes cos of x element-wise.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.cos(x)


def _regular_normalize_batch_in_training(x,
                                         gamma,
                                         beta,
                                         reduction_axes,
                                         epsilon=1e-3):
  """Non-fused version of `normalize_batch_in_training`.

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
  mean, var = nn.moments(x, reduction_axes, None, None, False)
  normed = nn.batch_normalization(x, mean, var, beta, gamma, epsilon)
  return normed, mean, var


def _broadcast_normalize_batch_in_training(x,
                                           gamma,
                                           beta,
                                           reduction_axes,
                                           epsilon=1e-3):
  """Non-fused, broadcast version of `normalize_batch_in_training`.

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
  mean, var = nn.moments(x, reduction_axes, None, None, False)
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


def _fused_normalize_batch_in_training(x,
                                       gamma,
                                       beta,
                                       reduction_axes,
                                       epsilon=1e-3):
  """Fused version of `normalize_batch_in_training`.

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
  if list(reduction_axes) == [0, 1, 2]:
    normalization_axis = 3
    tf_data_format = 'NHWC'
  else:
    normalization_axis = 1
    tf_data_format = 'NCHW'

  if gamma is None:
    gamma = constant_op.constant(
        1.0, dtype=x.dtype, shape=[x.shape[normalization_axis]])
  if beta is None:
    beta = constant_op.constant(
        0.0, dtype=x.dtype, shape=[x.shape[normalization_axis]])

  return nn.fused_batch_norm(
      x, gamma, beta, epsilon=epsilon, data_format=tf_data_format)


@keras_export('keras.backend.normalize_batch_in_training')
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
  if ndim(x) == 4 and list(reduction_axes) in [[0, 1, 2], [0, 2, 3]]:
    if not _has_nchw_support() and list(reduction_axes) == [0, 2, 3]:
      return _broadcast_normalize_batch_in_training(
          x, gamma, beta, reduction_axes, epsilon=epsilon)
    return _fused_normalize_batch_in_training(
        x, gamma, beta, reduction_axes, epsilon=epsilon)
  else:
    if sorted(reduction_axes) == list(range(ndim(x)))[:-1]:
      return _regular_normalize_batch_in_training(
          x, gamma, beta, reduction_axes, epsilon=epsilon)
    else:
      return _broadcast_normalize_batch_in_training(
          x, gamma, beta, reduction_axes, epsilon=epsilon)


@keras_export('keras.backend.batch_normalization')
def batch_normalization(x, mean, var, beta, gamma, axis=-1, epsilon=1e-3):
  """Applies batch normalization on x given mean, var, beta and gamma.

  I.e. returns:
  `output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta`

  Arguments:
      x: Input tensor or variable.
      mean: Mean of batch.
      var: Variance of batch.
      beta: Tensor with which to center the input.
      gamma: Tensor by which to scale the input.
      axis: Integer, the axis that should be normalized.
          (typically the features axis).
      epsilon: Fuzz factor.

  Returns:
      A tensor.
  """
  if ndim(x) == 4:
    # The CPU implementation of `fused_batch_norm` only supports NHWC
    if axis == 1 or axis == -3:
      tf_data_format = 'NCHW'
    elif axis == 3 or axis == -1:
      tf_data_format = 'NHWC'
    else:
      tf_data_format = None

    if (tf_data_format == 'NHWC' or
        tf_data_format == 'NCHW' and _has_nchw_support()):
      # The mean / var / beta / gamma tensors may be broadcasted
      # so they may have extra axes of size 1, which should be squeezed.
      if ndim(mean) > 1:
        mean = array_ops.reshape(mean, [-1])
      if ndim(var) > 1:
        var = array_ops.reshape(var, [-1])
      if beta is None:
        beta = zeros_like(mean)
      elif ndim(beta) > 1:
        beta = array_ops.reshape(beta, [-1])
      if gamma is None:
        gamma = ones_like(mean)
      elif ndim(gamma) > 1:
        gamma = array_ops.reshape(gamma, [-1])
    y, _, _ = nn.fused_batch_norm(
        x,
        gamma,
        beta,
        epsilon=epsilon,
        mean=mean,
        variance=var,
        data_format=tf_data_format,
        is_training=False
    )
    return y
  return nn.batch_normalization(x, mean, var, beta, gamma, epsilon)


# SHAPE OPERATIONS


@keras_export('keras.backend.concatenate')
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

  if py_all(is_sparse(x) for x in tensors):
    return sparse_ops.sparse_concat(axis, tensors)
  else:
    return array_ops.concat([to_dense(x) for x in tensors], axis)


@keras_export('keras.backend.reshape')
def reshape(x, shape):
  """Reshapes a tensor to the specified shape.

  Arguments:
      x: Tensor or variable.
      shape: Target shape tuple.

  Returns:
      A tensor.
  """
  return array_ops.reshape(x, shape)


@keras_export('keras.backend.permute_dimensions')
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


@keras_export('keras.backend.resize_images')
def resize_images(x, height_factor, width_factor, data_format,
                  interpolation='nearest'):
  """Resizes the images contained in a 4D tensor.

  Arguments:
      x: Tensor or variable to resize.
      height_factor: Positive integer.
      width_factor: Positive integer.
      data_format: One of `"channels_first"`, `"channels_last"`.
      interpolation: A string, one of `nearest` or `bilinear`.

  Returns:
      A tensor.

  Raises:
      ValueError: in case of incorrect value for
        `data_format` or `interpolation`.
  """
  if data_format == 'channels_first':
    rows, cols = 2, 3
  elif data_format == 'channels_last':
    rows, cols = 1, 2
  else:
    raise ValueError('Invalid `data_format` argument: %s' % (data_format,))

  original_shape = int_shape(x)
  new_shape = array_ops.shape(x)[rows:cols + 1]
  new_shape *= constant_op.constant(
      np.array([height_factor, width_factor], dtype='int32'))

  if data_format == 'channels_first':
    x = permute_dimensions(x, [0, 2, 3, 1])
  if interpolation == 'nearest':
    x = image_ops.resize_nearest_neighbor(x, new_shape)
  elif interpolation == 'bilinear':
    x = image_ops.resize_bilinear(x, new_shape)
  else:
    raise ValueError('interpolation should be one '
                     'of "nearest" or "bilinear".')
  if data_format == 'channels_first':
    x = permute_dimensions(x, [0, 3, 1, 2])

  if original_shape[rows] is None:
    new_height = None
  else:
    new_height = original_shape[rows] * height_factor

  if original_shape[cols] is None:
    new_width = None
  else:
    new_width = original_shape[cols] * width_factor

  if data_format == 'channels_first':
    output_shape = (None, None, new_height, new_width)
  else:
    output_shape = (None, new_height, new_width, None)
  x.set_shape(output_shape)
  return x


@keras_export('keras.backend.resize_volumes')
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
    raise ValueError('Invalid data_format: ' + str(data_format))


@keras_export('keras.backend.repeat_elements')
def repeat_elements(x, rep, axis):
  """Repeats the elements of a tensor along an axis, like `np.repeat`.

  If `x` has shape `(s1, s2, s3)` and `axis` is `1`, the output
  will have shape `(s1, s2 * rep, s3)`.

  Arguments:
      x: Tensor or variable.
      rep: Python integer, number of times to repeat.
      axis: Axis along which to repeat.

  Returns:
      A tensor.
  """
  x_shape = x.shape.as_list()
  # For static axis
  if x_shape[axis] is not None:
    # slices along the repeat axis
    splits = array_ops.split(value=x,
                             num_or_size_splits=x_shape[axis],
                             axis=axis)
    # repeat each slice the given number of reps
    x_rep = [s for s in splits for _ in range(rep)]
    return concatenate(x_rep, axis)

  # Here we use tf.tile to mimic behavior of np.repeat so that
  # we can handle dynamic shapes (that include None).
  # To do that, we need an auxiliary axis to repeat elements along
  # it and then merge them along the desired axis.

  # Repeating
  auxiliary_axis = axis + 1
  x_shape = array_ops.shape(x)
  x_rep = array_ops.expand_dims(x, axis=auxiliary_axis)
  reps = np.ones(len(x.shape) + 1)
  reps[auxiliary_axis] = rep
  x_rep = array_ops.tile(x_rep, reps)

  # Merging
  reps = np.delete(reps, auxiliary_axis)
  reps[axis] = rep
  reps = array_ops.constant(reps, dtype='int32')
  x_shape *= reps
  x_rep = array_ops.reshape(x_rep, x_shape)

  # Fix shape representation
  x_shape = x.shape.as_list()
  x_rep.set_shape(x_shape)
  x_rep._keras_shape = tuple(x_shape)
  return x_rep


@keras_export('keras.backend.repeat')
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


@keras_export('keras.backend.arange')
def arange(start, stop=None, step=1, dtype='int32'):
  """Creates a 1D tensor containing a sequence of integers.

  The function arguments use the same convention as
  Theano's arange: if only one argument is provided,
  it is in fact the "stop" argument and "start" is 0.

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
  # Match the behavior of numpy and Theano by returning an empty sequence.
  if stop is None and start < 0:
    start = 0
  result = math_ops.range(start, limit=stop, delta=step, name='arange')
  if dtype != 'int32':
    result = cast(result, dtype)
  return result


@keras_export('keras.backend.tile')
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


@keras_export('keras.backend.flatten')
def flatten(x):
  """Flatten a tensor.

  Arguments:
      x: A tensor or variable.

  Returns:
      A tensor, reshaped into 1-D
  """
  return array_ops.reshape(x, [-1])


@keras_export('keras.backend.batch_flatten')
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


@keras_export('keras.backend.expand_dims')
def expand_dims(x, axis=-1):
  """Adds a 1-sized dimension at index "axis".

  Arguments:
      x: A tensor or variable.
      axis: Position where to add a new axis.

  Returns:
      A tensor with expanded dimensions.
  """
  return array_ops.expand_dims(x, axis)


@keras_export('keras.backend.squeeze')
def squeeze(x, axis):
  """Removes a 1-dimension from the tensor at index "axis".

  Arguments:
      x: A tensor or variable.
      axis: Axis to drop.

  Returns:
      A tensor with the same data as `x` but reduced dimensions.
  """
  return array_ops.squeeze(x, [axis])


@keras_export('keras.backend.temporal_padding')
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


@keras_export('keras.backend.spatial_2d_padding')
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
    raise ValueError('Unknown data_format: ' + str(data_format))

  if data_format == 'channels_first':
    pattern = [[0, 0], [0, 0], list(padding[0]), list(padding[1])]
  else:
    pattern = [[0, 0], list(padding[0]), list(padding[1]), [0, 0]]
  return array_ops.pad(x, pattern)


@keras_export('keras.backend.spatial_3d_padding')
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
    raise ValueError('Unknown data_format: ' + str(data_format))

  if data_format == 'channels_first':
    pattern = [[0, 0], [0, 0], [padding[0][0], padding[0][1]],
               [padding[1][0], padding[1][1]], [padding[2][0], padding[2][1]]]
  else:
    pattern = [[0, 0], [padding[0][0], padding[0][1]],
               [padding[1][0], padding[1][1]], [padding[2][0],
                                                padding[2][1]], [0, 0]]
  return array_ops.pad(x, pattern)


@keras_export('keras.backend.stack')
def stack(x, axis=0):
  """Stacks a list of rank `R` tensors into a rank `R+1` tensor.

  Arguments:
      x: List of tensors.
      axis: Axis along which to perform stacking.

  Returns:
      A tensor.
  """
  return array_ops.stack(x, axis=axis)


@keras_export('keras.backend.one_hot')
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


@keras_export('keras.backend.reverse')
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


@keras_export('keras.backend.get_value')
def get_value(x):
  """Returns the value of a variable.

  Arguments:
      x: input variable.

  Returns:
      A Numpy array.

  Raises:
      RuntimeError: If this method is called inside defun.
  """
  if context.executing_eagerly():
    return x.numpy()
  elif not getattr(x, '_in_graph_mode', True):
    # This is a variable which was created in an eager context, but is being
    # evaluated from a Graph.
    with context.eager_mode():
      return x.numpy()
  elif ops.inside_function():
    raise RuntimeError('Cannot get value inside Tensorflow graph function.')
  return x.eval(session=get_session())


@keras_export('keras.backend.batch_get_value')
def batch_get_value(tensors):
  """Returns the value of more than one tensor variable.

  Arguments:
      tensors: list of ops to run.

  Returns:
      A list of Numpy arrays.

  Raises:
      RuntimeError: If this method is called inside defun.
  """
  if context.executing_eagerly():
    return [x.numpy() for x in tensors]
  elif ops.inside_function():  # pylint: disable=protected-access
    raise RuntimeError('Cannot get value inside Tensorflow graph function.')
  if tensors:
    return get_session().run(tensors)
  else:
    return []


@keras_export('keras.backend.set_value')
def set_value(x, value):
  """Sets the value of a variable, from a Numpy array.

  Arguments:
      x: Tensor to set to a new value.
      value: Value to set the tensor to, as a Numpy array
          (of the same shape).
  """
  value = np.asarray(value, dtype=dtype(x))
  if ops.executing_eagerly_outside_functions():
    x.assign(value)
  else:
    with get_graph().as_default():
      tf_dtype = dtypes_module.as_dtype(x.dtype.name.split('_')[0])
      if hasattr(x, '_assign_placeholder'):
        assign_placeholder = x._assign_placeholder
        assign_op = x._assign_op
      else:
        assign_placeholder = array_ops.placeholder(tf_dtype, shape=value.shape)
        assign_op = x.assign(assign_placeholder)
        x._assign_placeholder = assign_placeholder
        x._assign_op = assign_op
      get_session().run(assign_op, feed_dict={assign_placeholder: value})


@keras_export('keras.backend.batch_set_value')
def batch_set_value(tuples):
  """Sets the values of many tensor variables at once.

  Arguments:
      tuples: a list of tuples `(tensor, value)`.
          `value` should be a Numpy array.
  """
  if ops.executing_eagerly_outside_functions():
    for x, value in tuples:
      x.assign(np.asarray(value, dtype=dtype(x)))
  else:
    with get_graph().as_default():
      if tuples:
        assign_ops = []
        feed_dict = {}
        for x, value in tuples:
          value = np.asarray(value, dtype=dtype(x))
          tf_dtype = dtypes_module.as_dtype(x.dtype.name.split('_')[0])
          if hasattr(x, '_assign_placeholder'):
            assign_placeholder = x._assign_placeholder
            assign_op = x._assign_op
          else:
            assign_placeholder = array_ops.placeholder(tf_dtype,
                                                       shape=value.shape)
            assign_op = x.assign(assign_placeholder)
            x._assign_placeholder = assign_placeholder
            x._assign_op = assign_op
          assign_ops.append(assign_op)
          feed_dict[assign_placeholder] = value
        get_session().run(assign_ops, feed_dict=feed_dict)


@keras_export('keras.backend.print_tensor')
def print_tensor(x, message=''):
  """Prints `message` and the tensor value when evaluated.

  Note that `print_tensor` returns a new tensor identical to `x`
  which should be used in the following code. Otherwise the
  print operation is not taken into account during evaluation.

  Example:

  ```python
     >>> x = K.print_tensor(x, message="x is: ")
  ```

  Arguments:
      x: Tensor to print.
      message: Message to print jointly with the tensor.

  Returns:
      The same tensor `x`, unchanged.
  """
  return logging_ops.Print(x, [x], message)


# GRAPH MANIPULATION


class GraphExecutionFunction(object):
  """Runs a computation graph.

  It's possible to pass arguments to `tf.Session.run()` via `session_kwargs`.
  In particular additional operations via `fetches` argument and additional
  tensor substitutions via `feed_dict` arguments. Note that given
  substitutions are merged with substitutions from `inputs`. Even though
  `feed_dict` is passed once in the constructor (called in `model.compile()`)
  we can modify the values in the dictionary. Through this feed_dict we can
  provide additional substitutions besides Keras inputs.

  Arguments:
      inputs: Feed placeholders to the computation graph.
      outputs: Output tensors to fetch.
      updates: Additional update ops to be run at function call.
      name: A name to help users identify what this function does.
      session_kwargs: Arguments to `tf.Session.run()`:
                      `fetches`, `feed_dict`, `options`, `run_metadata`.
  """

  def __init__(self, inputs, outputs, updates=None, name=None,
               **session_kwargs):
    updates = updates or []
    if not isinstance(updates, (list, tuple)):
      raise TypeError('`updates` in a Keras backend function '
                      'should be a list or tuple.')
    self.inputs = nest.flatten(inputs)
    self._outputs_structure = outputs
    self.outputs = nest.flatten(outputs)
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
    # additional tensor substitutions
    self.feed_dict = session_kwargs.pop('feed_dict', None)
    # additional operations
    self.fetches = session_kwargs.pop('fetches', [])
    if not isinstance(self.fetches, list):
      self.fetches = [self.fetches]
    self.run_options = session_kwargs.pop('options', None)
    self.run_metadata = session_kwargs.pop('run_metadata', None)
    # The main use case of `fetches` being passed to a model is the ability
    # to run custom updates
    # This requires us to wrap fetches in `identity` ops.
    self.fetches = [array_ops.identity(x) for x in self.fetches]
    self.session_kwargs = session_kwargs
    # This mapping keeps track of the function that should receive the
    # output from a fetch in `fetches`: { fetch: function(fetch_output) }
    # A Callback can use this to register a function with access to the
    # output values for a fetch it added.
    self.fetch_callbacks = dict()

    if session_kwargs:
      raise ValueError('Some keys in session_kwargs are not supported at this '
                       'time: %s' % (session_kwargs.keys(),))

    self._callable_fn = None
    self._feed_arrays = None
    self._feed_symbols = None
    self._symbol_vals = None
    self._fetches = None
    self._session = None

  def _make_callable(self, feed_arrays, feed_symbols, symbol_vals, session):
    """Generates a callable that runs the graph.

    Arguments:
      feed_arrays: List of input tensors to be fed Numpy arrays at runtime.
      feed_symbols: List of input tensors to be fed symbolic tensors at runtime.
      symbol_vals: List of symbolic tensors to be fed to `feed_symbols`.
      session: Session to use to generate the callable.

    Returns:
      Function that runs the graph according to the above options.
    """
    # Prepare callable options.
    callable_opts = config_pb2.CallableOptions()
    # Handle external-data feed.
    for x in feed_arrays:
      callable_opts.feed.append(x.name)
    if self.feed_dict:
      for key in sorted(self.feed_dict.keys()):
        callable_opts.feed.append(key.name)
    # Handle symbolic feed.
    for x, y in zip(feed_symbols, symbol_vals):
      connection = callable_opts.tensor_connection.add()
      if x.dtype != y.dtype:
        y = math_ops.cast(y, dtype=x.dtype)
      from_tensor = ops._as_graph_element(y)
      if from_tensor is None:
        from_tensor = y
      connection.from_tensor = from_tensor.name  # Data tensor
      connection.to_tensor = x.name  # Placeholder
    # Handle fetches.
    for x in self.outputs + self.fetches:
      callable_opts.fetch.append(x.name)
    # Handle updates.
    callable_opts.target.append(self.updates_op.name)
    # Handle run_options.
    if self.run_options:
      callable_opts.run_options.CopyFrom(self.run_options)
    # Create callable.
    callable_fn = session._make_callable_from_options(callable_opts)
    # Cache parameters corresponding to the generated callable, so that
    # we can detect future mismatches and refresh the callable.
    self._callable_fn = callable_fn
    self._feed_arrays = feed_arrays
    self._feed_symbols = feed_symbols
    self._symbol_vals = symbol_vals
    self._fetches = list(self.fetches)
    self._session = session

  def _call_fetch_callbacks(self, fetches_output):
    for fetch, output in zip(self._fetches, fetches_output):
      if fetch in self.fetch_callbacks:
        self.fetch_callbacks[fetch](output)

  def __call__(self, inputs):
    inputs = nest.flatten(inputs)

    session = get_session()
    feed_arrays = []
    array_vals = []
    feed_symbols = []
    symbol_vals = []
    for tensor, value in zip(self.inputs, inputs):
      if value is None:
        continue
      if is_sparse(tensor):
        sparse_coo = value.tocoo()
        indices = np.concatenate((np.expand_dims(sparse_coo.row, 1),
                                  np.expand_dims(sparse_coo.col, 1)), 1)
        value = (indices, sparse_coo.data, sparse_coo.shape)
      if tensor_util.is_tensor(value):
        # Case: feeding symbolic tensor.
        feed_symbols.append(tensor)
        symbol_vals.append(value)
      else:
        # Case: feeding Numpy array.
        feed_arrays.append(tensor)
        # We need to do array conversion and type casting at this level, since
        # `callable_fn` only supports exact matches.
        tensor_type = dtypes_module.as_dtype(tensor.dtype)
        array_vals.append(np.asarray(value,
                                     dtype=tensor_type.as_numpy_dtype))

    if self.feed_dict:
      for key in sorted(self.feed_dict.keys()):
        array_vals.append(
            np.asarray(self.feed_dict[key], dtype=key.dtype.base_dtype.name))

    # Refresh callable if anything has changed.
    if (self._callable_fn is None or feed_arrays != self._feed_arrays or
        symbol_vals != self._symbol_vals or
        feed_symbols != self._feed_symbols or self.fetches != self._fetches or
        session != self._session):
      self._make_callable(feed_arrays, feed_symbols, symbol_vals, session)

    fetched = self._callable_fn(*array_vals,
                                run_metadata=self.run_metadata)
    self._call_fetch_callbacks(fetched[-len(self._fetches):])
    return nest.pack_sequence_as(self._outputs_structure,
                                 fetched[:len(self.outputs)])


class EagerExecutionFunction(object):
  """Helper class for constructing a TF graph function from the Keras graph.

  Arguments:
    inputs: Feed placeholders to the computation graph.
    outputs: Output tensors to fetch.
    updates: Additional update ops to be run at function call.
    name: A name to help users identify what this function does.
    session_kwargs: Unsupported.
  """

  def __init__(self, inputs, outputs, updates=None, name=None):
    updates = updates or []
    if not isinstance(updates, (list, tuple)):
      raise TypeError('`updates` in a Keras backend function '
                      'should be a list or tuple.')
    self.inputs = nest.flatten(inputs)
    self._outputs_structure = outputs
    self.outputs = nest.flatten(outputs)
    self.name = name

    graph = get_graph()
    # Consolidate updates
    with graph.as_default():
      with ops.control_dependencies(self.outputs):
        # In general, updates should be run after the outputs have been
        # computed. However, we can only ensure this when we create
        # the updates here (i.e. when updates are passed as tuples).
        # We cannot modify the control dependencies of preexisting update ops.
        updates_ops = []
        for update in updates:
          # For legacy reasons it is allowed to pass an update as a tuple
          # `(variable, new_value)` (this maps to an assign op).
          if isinstance(update, tuple):
            p, new_p = update
            updates_ops.append(state_ops.assign(p, new_p))
          else:
            # Assumed already an op -- we cannot control its execution order.
            updates_ops.append(update)

      # We set the update ops to run at the end by conditioning it on output[0]
      if updates and not self.outputs:
        # Edge case; never happens in practice
        raise ValueError('Cannot create a Keras backend function with updates'
                         ' but no outputs during eager execution.')
      with ops.control_dependencies(updates_ops):
        self.outputs[0] = array_ops.identity(self.outputs[0])

    # Prepare graph function
    # TODO(fchollet): can we restrict `captures` to variables actually used in
    # the relevant subgraph?
    graph.inputs = self.inputs + list(graph.captures.values())
    graph.outputs = self.outputs
    graph_fn = eager_function.ConcreteFunction(graph)
    graph_fn._num_positional_args = len(self.inputs)
    graph_fn._arg_keywords = []
    self._graph_fn = graph_fn

    # Handle placeholders with default
    # (treated as required placeholder by graph functions)
    self._placeholder_default_values = {}
    with graph.as_default():
      for x in self.inputs:
        if x.op.type == 'PlaceholderWithDefault':
          self._placeholder_default_values[x] = tensor_util.constant_value(
              x.op.inputs[0])

  def __call__(self, inputs):
    inputs = nest.flatten(inputs)
    converted_inputs = []
    for tensor, value in zip(self.inputs, inputs):
      if value is None:
        # Assume `value` is a placeholder with default
        value = self._placeholder_default_values.get(tensor, None)
        if value is None:
          raise ValueError(
              'You must feed a value for placeholder %s' % (tensor,))
      if not isinstance(value, ops.Tensor):
        value = ops.convert_to_tensor(value, dtype=tensor.dtype)
      if value.dtype != tensor.dtype:
        # Temporary workaround due to `convert_to_tensor` not casting floats.
        # See b/119637405
        value = math_ops.cast(value, tensor.dtype)
      converted_inputs.append(value)
    outputs = self._graph_fn(*converted_inputs)
    return nest.pack_sequence_as(self._outputs_structure,
                                 [x.numpy() for x in outputs])


@keras_export('keras.backend.function')
def function(inputs, outputs, updates=None, name=None, **kwargs):
  """Instantiates a Keras function.

  Arguments:
      inputs: List of placeholder tensors.
      outputs: List of output tensors.
      updates: List of update ops.
      name: String, name of function.
      **kwargs: Passed to `tf.Session.run`.

  Returns:
      Output values as Numpy arrays.

  Raises:
      ValueError: if invalid kwargs are passed in or if in eager execution.
  """
  if ops.executing_eagerly_outside_functions():
    if kwargs:
      raise ValueError('Session keyword arguments are not support during '
                       'eager execution. You passed: %s' % (kwargs,))
    return EagerExecutionFunction(inputs, outputs, updates=updates, name=name)

  if kwargs:
    for key in kwargs:
      if (key not in tf_inspect.getfullargspec(session_module.Session.run)[0]
          and key not in ['inputs', 'outputs', 'updates', 'name']):
        msg = ('Invalid argument "%s" passed to K.function with TensorFlow '
               'backend') % key
        raise ValueError(msg)
  return GraphExecutionFunction(inputs, outputs, updates=updates, **kwargs)


@keras_export('keras.backend.gradients')
def gradients(loss, variables):
  """Returns the gradients of `loss` w.r.t. `variables`.

  Arguments:
      loss: Scalar tensor to minimize.
      variables: List of variables.

  Returns:
      A gradients tensor.
  """
  return gradients_module.gradients(
      loss, variables, colocate_gradients_with_ops=True)


@keras_export('keras.backend.stop_gradient')
def stop_gradient(variables):
  """Returns `variables` but with zero gradient w.r.t. every other variable.

  Arguments:
      variables: Tensor or list of tensors to consider constant with respect
        to any other variable.


  Returns:
      A single tensor or a list of tensors (depending on the passed argument)
      that has no gradient with respect to any other variable.
  """
  if isinstance(variables, (list, tuple)):
    return map(array_ops.stop_gradient, variables)
  return array_ops.stop_gradient(variables)


# CONTROL FLOW


@keras_export('keras.backend.rnn')
def rnn(step_function,
        inputs,
        initial_states,
        go_backwards=False,
        mask=None,
        constants=None,
        unroll=False,
        input_length=None,
        time_major=False,
        zero_output_for_mask=False):
  """Iterates over the time dimension of a tensor.

  Arguments:
      step_function: RNN step function.
          Args;
              input; Tensor with shape `(samples, ...)` (no time dimension),
                  representing input for the batch of samples at a certain
                  time step.
              states; List of tensors.
          Returns;
              output; Tensor with shape `(samples, output_dim)`
                  (no time dimension).
              new_states; List of tensors, same length and shapes
                  as 'states'. The first state in the list must be the
                  output tensor at the previous timestep.
      inputs: Tensor of temporal data of shape `(samples, time, ...)`
          (at least 3D), or nested tensors, and each of which has shape
          `(samples, time, ...)`.
      initial_states: Tensor with shape `(samples, state_size)`
          (no time dimension), containing the initial values for the states used
          in the step function. In the case that state_size is in a nested
          shape, the shape of initial_states will also follow the nested
          structure.
      go_backwards: Boolean. If True, do the iteration over the time
          dimension in reverse order and return the reversed sequence.
      mask: Binary tensor with shape `(samples, time, 1)`,
          with a zero for every element that is masked.
      constants: List of constant values passed at each step.
      unroll: Whether to unroll the RNN or to use a symbolic `while_loop`.
      input_length: If specified, assume time dimension is of this length.
      time_major: Boolean. If true, the inputs and outputs will be in shape
          `(timesteps, batch, ...)`, whereas in the False case, it will be
          `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
          efficient because it avoids transposes at the beginning and end of the
          RNN calculation. However, most TensorFlow data is batch-major, so by
          default this function accepts input and emits output in batch-major
          form.
      zero_output_for_mask: Boolean. If True, the output for masked timestep
          will be zeros, whereas in the False case, output from previous
          timestep is returned.
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

  def swap_batch_timestep(input_t):
    # Swap the batch and timestep dim for the incoming tensor.
    axes = list(range(len(input_t.shape)))
    axes[0], axes[1] = 1, 0
    return array_ops.transpose(input_t, axes)

  if not time_major:
    inputs = nest.map_structure(swap_batch_timestep, inputs)

  flatted_inputs = nest.flatten(inputs)
  time_steps = flatted_inputs[0].shape[0]
  batch = flatted_inputs[0].shape[1]
  time_steps_t = array_ops.shape(flatted_inputs[0])[0]

  for input_ in flatted_inputs:
    input_.get_shape().with_rank_at_least(3)

  if mask is not None:
    if mask.dtype != dtypes_module.bool:
      mask = math_ops.cast(mask, dtypes_module.bool)
    if len(mask.shape) == 2:
      mask = expand_dims(mask)
    if not time_major:
      mask = swap_batch_timestep(mask)

  if constants is None:
    constants = []

  # tf.where needs its condition tensor to be the same shape as its two
  # result tensors, but in our case the condition (mask) tensor is
  # (nsamples, 1), and inputs are (nsamples, ndimensions) or even more.
  # So we need to broadcast the mask to match the shape of inputs.
  # That's what the tile call does, it just repeats the mask along its
  # second dimension n times.
  def _expand_mask(mask_t, input_t, fixed_dim=1):
    assert not nest.is_sequence(mask_t)
    assert not nest.is_sequence(input_t)
    rank_diff = len(input_t.shape) - len(mask_t.shape)
    for _ in range(rank_diff):
      mask_t = array_ops.expand_dims(mask_t, -1)
    multiples = [1] * fixed_dim + input_t.shape.as_list()[fixed_dim:]
    return array_ops.tile(mask_t, multiples)

  if unroll:
    if not time_steps:
      raise ValueError('Unrolling requires a fixed number of timesteps.')
    states = initial_states
    successive_states = []
    successive_outputs = []

    # Process the input tensors. The input tensor need to be split on the
    # time_step dim, and reverse if go_backwards is True. In the case of nested
    # input, the input is flattened and then transformed individually.
    # The result of this will be a tuple of lists, each of the item in tuple is
    # list of the tensor with shape (batch, feature)
    def _process_single_input_t(input_t):
      input_t = array_ops.unstack(input_t)  # unstack for time_step dim
      if go_backwards:
        input_t.reverse()
      return input_t

    if nest.is_sequence(inputs):
      processed_input = nest.map_structure(_process_single_input_t, inputs)
    else:
      processed_input = (_process_single_input_t(inputs),)

    def _get_input_tensor(time):
      inp = [t_[time] for t_ in processed_input]
      return nest.pack_sequence_as(inputs, inp)

    if mask is not None:
      mask_list = array_ops.unstack(mask)
      if go_backwards:
        mask_list.reverse()

      for i in range(time_steps):
        inp = _get_input_tensor(i)
        mask_t = mask_list[i]
        output, new_states = step_function(inp, states + constants)
        tiled_mask_t = _expand_mask(mask_t, output)

        if not successive_outputs:
          prev_output = zeros_like(output)
        else:
          prev_output = successive_outputs[-1]

        output = array_ops.where(tiled_mask_t, output, prev_output)

        return_states = []
        for state, new_state in zip(states, new_states):
          # (see earlier comment for tile explanation)
          tiled_mask_t = _expand_mask(mask_t, new_state)
          return_states.append(array_ops.where(tiled_mask_t, new_state, state))
        states = return_states
        successive_outputs.append(output)
        successive_states.append(states)
      last_output = successive_outputs[-1]
      new_states = successive_states[-1]
      outputs = array_ops.stack(successive_outputs)

      if zero_output_for_mask:
        last_output = array_ops.where(
            _expand_mask(mask_list[-1], last_output),
            last_output,
            zeros_like(last_output))
        outputs = array_ops.where(
            _expand_mask(mask, outputs, fixed_dim=2),
            outputs,
            zeros_like(outputs))

    else:
      for i in range(time_steps):
        inp = _get_input_tensor(i)
        output, states = step_function(inp, states + constants)
        successive_outputs.append(output)
        successive_states.append(states)
      last_output = successive_outputs[-1]
      new_states = successive_states[-1]
      outputs = array_ops.stack(successive_outputs)

  else:
    states = tuple(initial_states)

    # Create input tensor array, if the inputs is nested tensors, then it will
    # be flattened first, and tensor array will be created one per flattened
    # tensor.
    input_ta = tuple(
        tensor_array_ops.TensorArray(
            dtype=inp.dtype,
            size=time_steps_t,
            tensor_array_name='input_ta_%s' % i)
        for i, inp in enumerate(flatted_inputs))
    input_ta = tuple(
        ta.unstack(input_) if not go_backwards else ta
        .unstack(reverse(input_, 0))
        for ta, input_ in zip(input_ta, flatted_inputs))

    # Get the time(0) input and compute the output for that, the output will be
    # used to determine the dtype of output tensor array. Don't read from
    # input_ta due to TensorArray clear_after_read default to True.
    input_time_zero = nest.pack_sequence_as(inputs,
                                            [inp[0] for inp in flatted_inputs])
    # output_time_zero is used to determine the cell output shape and its dtype.
    # the value is discarded.
    output_time_zero, _ = step_function(input_time_zero,
                                        initial_states + constants)
    output_ta = tuple(
        tensor_array_ops.TensorArray(
            dtype=out.dtype,
            size=time_steps_t,
            tensor_array_name='output_ta_%s' % i)
        for i, out in enumerate(nest.flatten(output_time_zero)))

    time = constant_op.constant(0, dtype='int32', name='time')

    while_loop_kwargs = {
        'cond': lambda time, *_: time < time_steps_t,
        'maximum_iterations': input_length,
        'parallel_iterations': 32,
        'swap_memory': True,
    }

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
          size=time_steps_t,
          tensor_array_name='mask_ta')
      mask_ta = mask_ta.unstack(mask)

      # Mask for the T output will be base on the output of T - 1. In the case
      # T = 0, a zero filled tensor will be used.
      flat_zero_output = tuple(array_ops.zeros_like(o)
                               for o in nest.flatten(output_time_zero))
      def _step(time, output_ta_t, prev_output, *states):
        """RNN step function.

        Arguments:
            time: Current timestep value.
            output_ta_t: TensorArray.
            prev_output: tuple of outputs from time - 1.
            *states: List of states.

        Returns:
            Tuple: `(time + 1, output_ta_t, output) + tuple(new_states)`
        """
        current_input = tuple(ta.read(time) for ta in input_ta)
        # maybe set shape.
        current_input = nest.pack_sequence_as(inputs, current_input)
        mask_t = mask_ta.read(time)
        output, new_states = step_function(current_input,
                                           tuple(states) + tuple(constants))
        # mask output
        flat_output = nest.flatten(output)
        flat_mask_output = (flat_zero_output if zero_output_for_mask
                            else nest.flatten(prev_output))
        tiled_mask_t = tuple(_expand_mask(mask_t, o) for o in flat_output)
        flat_new_output = tuple(
            array_ops.where(m, o, zo) for m, o, zo in zip(
                tiled_mask_t, flat_output, flat_mask_output))

        # mask states
        flat_state = nest.flatten(states)
        flat_new_state = nest.flatten(new_states)
        for state, new_state in zip(flat_state, flat_new_state):
          new_state.set_shape(state.shape)
        tiled_mask_t = tuple(_expand_mask(mask_t, s) for s in flat_state)
        flat_final_state = tuple(
            array_ops.where(m, s, ps)
            for m, s, ps in zip(tiled_mask_t, flat_new_state, flat_state))
        new_states = nest.pack_sequence_as(new_states, flat_final_state)

        output_ta_t = tuple(
            ta.write(time, out)
            for ta, out in zip(output_ta_t, flat_new_output))
        return (time + 1, output_ta_t,
                tuple(flat_new_output)) + tuple(new_states)

      final_outputs = control_flow_ops.while_loop(
          body=_step,
          loop_vars=(time, output_ta, flat_zero_output) + states,
          **while_loop_kwargs)
      # Skip final_outputs[2] which is the output for final timestep.
      new_states = final_outputs[3:]
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
        current_input = tuple(ta.read(time) for ta in input_ta)
        current_input = nest.pack_sequence_as(inputs, current_input)
        output, new_states = step_function(current_input,
                                           tuple(states) + tuple(constants))
        flat_state = nest.flatten(states)
        flat_new_state = nest.flatten(new_states)
        for state, new_state in zip(flat_state, flat_new_state):
          new_state.set_shape(state.shape)

        flat_output = nest.flatten(output)
        output_ta_t = tuple(
            ta.write(time, out) for ta, out in zip(output_ta_t, flat_output))
        new_states = nest.pack_sequence_as(initial_states, flat_new_state)
        return (time + 1, output_ta_t) + tuple(new_states)

      final_outputs = control_flow_ops.while_loop(
          body=_step,
          loop_vars=(time, output_ta) + states,
          **while_loop_kwargs)
      new_states = final_outputs[2:]

    output_ta = final_outputs[1]

    outputs = tuple(o.stack() for o in output_ta)
    last_output = tuple(o[-1] for o in outputs)

    outputs = nest.pack_sequence_as(output_time_zero, outputs)
    last_output = nest.pack_sequence_as(output_time_zero, last_output)

  # static shape inference
  def set_shape(output_):
    shape = output_.shape.as_list()
    shape[0] = time_steps
    shape[1] = batch
    output_.set_shape(shape)
    return output_

  outputs = nest.map_structure(set_shape, outputs)

  if not time_major:
    outputs = nest.map_structure(swap_batch_timestep, outputs)

  return last_output, outputs, new_states


@keras_export('keras.backend.switch')
def switch(condition, then_expression, else_expression):
  """Switches between two operations depending on a scalar value.

  Note that both `then_expression` and `else_expression`
  should be symbolic tensors of the *same shape*.

  Arguments:
      condition: tensor (`int` or `bool`).
      then_expression: either a tensor, or a callable that returns a tensor.
      else_expression: either a tensor, or a callable that returns a tensor.

  Returns:
      The selected tensor.

  Raises:
      ValueError: If rank of `condition` is greater than rank of expressions.
  """
  if condition.dtype != dtypes_module.bool:
    condition = math_ops.cast(condition, 'bool')
  cond_ndim = ndim(condition)
  if not cond_ndim:
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
  else:
    # tf.where needs its condition tensor
    # to be the same shape as its two
    # result tensors
    if callable(then_expression):
      then_expression = then_expression()
    if callable(else_expression):
      else_expression = else_expression()
    expr_ndim = ndim(then_expression)
    if cond_ndim > expr_ndim:
      raise ValueError('Rank of `condition` should be less than or'
                       ' equal to rank of `then_expression` and '
                       '`else_expression`. ndim(condition)=' + str(cond_ndim) +
                       ', ndim(then_expression)'
                       '=' + str(expr_ndim))
    if cond_ndim > 1:
      ndim_diff = expr_ndim - cond_ndim
      cond_shape = array_ops.concat(
          [array_ops.shape(condition), [1] * ndim_diff], axis=0)
      condition = array_ops.reshape(condition, cond_shape)
      expr_shape = array_ops.shape(then_expression)
      shape_diff = expr_shape - cond_shape
      tile_shape = array_ops.where(shape_diff > 0, expr_shape,
                                   array_ops.ones_like(expr_shape))
      condition = array_ops.tile(condition, tile_shape)
    x = array_ops.where(condition, then_expression, else_expression)
  return x


@keras_export('keras.backend.in_train_phase')
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

  if training == 1 or training is True:
    if callable(x):
      return x()
    else:
      return x

  elif training == 0 or training is False:
    if callable(alt):
      return alt()
    else:
      return alt

  # else: assume learning phase is a placeholder tensor.
  x = switch(training, x, alt)
  return x


@keras_export('keras.backend.in_test_phase')
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


@keras_export('keras.backend.relu')
def relu(x, alpha=0., max_value=None, threshold=0):
  """Rectified linear unit.

  With default values, it returns element-wise `max(x, 0)`.

  Otherwise, it follows:
  `f(x) = max_value` for `x >= max_value`,
  `f(x) = x` for `threshold <= x < max_value`,
  `f(x) = alpha * (x - threshold)` otherwise.

  Arguments:
      x: A tensor or variable.
      alpha: A scalar, slope of negative section (default=`0.`).
      max_value: float. Saturation threshold.
      threshold: float. Threshold value for thresholded activation.

  Returns:
      A tensor.
  """

  if alpha != 0.:
    if max_value is None and threshold == 0:
      return nn.leaky_relu(x, alpha=alpha)

    if threshold != 0:
      negative_part = nn.relu(-x + threshold)
    else:
      negative_part = nn.relu(-x)

  clip_max = max_value is not None

  if threshold != 0:
    # computes x for x > threshold else 0
    x = x * math_ops.cast(math_ops.greater(x, threshold), floatx())
  elif max_value == 6:
    # if no threshold, then can use nn.relu6 native TF op for performance
    x = nn.relu6(x)
    clip_max = False
  else:
    x = nn.relu(x)

  if clip_max:
    max_value = _to_tensor(max_value, x.dtype.base_dtype)
    zero = _to_tensor(0., x.dtype.base_dtype)
    x = clip_ops.clip_by_value(x, zero, max_value)

  if alpha != 0.:
    alpha = _to_tensor(alpha, x.dtype.base_dtype)
    x -= alpha * negative_part
  return x


@keras_export('keras.backend.elu')
def elu(x, alpha=1.):
  """Exponential linear unit.

  Arguments:
      x: A tensor or variable to compute the activation function for.
      alpha: A scalar, slope of negative section.

  Returns:
      A tensor.
  """
  res = nn.elu(x)
  if alpha == 1:
    return res
  else:
    return array_ops.where(x > 0, res, alpha * res)


@keras_export('keras.backend.softmax')
def softmax(x, axis=-1):
  """Softmax of a tensor.

  Arguments:
      x: A tensor or variable.
      axis: The dimension softmax would be performed on.
          The default is -1 which indicates the last dimension.

  Returns:
      A tensor.
  """
  return nn.softmax(x, axis=axis)


@keras_export('keras.backend.softplus')
def softplus(x):
  """Softplus of a tensor.

  Arguments:
      x: A tensor or variable.

  Returns:
      A tensor.
  """
  return nn.softplus(x)


@keras_export('keras.backend.softsign')
def softsign(x):
  """Softsign of a tensor.

  Arguments:
      x: A tensor or variable.

  Returns:
      A tensor.
  """
  return nn.softsign(x)


@keras_export('keras.backend.categorical_crossentropy')
def categorical_crossentropy(target, output, from_logits=False, axis=-1):
  """Categorical crossentropy between an output tensor and a target tensor.

  Arguments:
      target: A tensor of the same shape as `output`.
      output: A tensor resulting from a softmax
          (unless `from_logits` is True, in which
          case `output` is expected to be the logits).
      from_logits: Boolean, whether `output` is the
          result of a softmax, or is a tensor of logits.
      axis: Int specifying the channels axis. `axis=-1` corresponds to data
          format `channels_last', and `axis=1` corresponds to data format
          `channels_first`.

  Returns:
      Output tensor.

  Raises:
      ValueError: if `axis` is neither -1 nor one of the axes of `output`.
  """
  if not from_logits:
    if context.executing_eagerly() or output.op.type != 'Softmax':
      axis = axis % len(output.shape)
      # scale preds so that the class probas of each sample sum to 1
      output = output / math_ops.reduce_sum(output, axis, True)

      # Compute cross entropy from probabilities.
      epsilon_ = _to_tensor(epsilon(), output.dtype.base_dtype)
      output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)
      return -math_ops.reduce_sum(target * math_ops.log(output), axis)
    else:
      # When softmax activation function is used for output operation, we
      # use logits from the softmax function directly to compute loss in order
      # to prevent collapsing zero when training.
      # See b/117284466
      assert len(output.op.inputs) == 1
      output = output.op.inputs[0]
  return nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=output)


@keras_export('keras.backend.sparse_categorical_crossentropy')
def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
  """Categorical crossentropy with integer targets.

  Arguments:
      target: An integer tensor.
      output: A tensor resulting from a softmax
          (unless `from_logits` is True, in which
          case `output` is expected to be the logits).
      from_logits: Boolean, whether `output` is the
          result of a softmax, or is a tensor of logits.
      axis: Int specifying the channels axis. `axis=-1` corresponds to data
          format `channels_last', and `axis=1` corresponds to data format
          `channels_first`.

  Returns:
      Output tensor.

  Raises:
      ValueError: if `axis` is neither -1 nor one of the axes of `output`.
  """
  if not from_logits:
    if context.executing_eagerly() or output.op.type != 'Softmax':
      epsilon_ = _to_tensor(epsilon(), output.dtype.base_dtype)
      output = clip_ops.clip_by_value(output, epsilon_, 1 - epsilon_)
      output = math_ops.log(output)
    else:
      # When softmax activation function is used for output operation, we
      # use logits from the softmax function directly to compute loss in order
      # to prevent collapsing zero when training.
      # See b/117284466
      assert len(output.op.inputs) == 1
      output = output.op.inputs[0]

  rank = len(output.shape)
  axis = axis % rank
  if axis != rank - 1:
    permutation = list(range(axis)) + list(range(axis + 1, rank)) + [axis]
    output = array_ops.transpose(output, perm=permutation)

  output_shape = output.shape
  targets = cast(flatten(target), 'int64')
  logits = array_ops.reshape(output, [-1, int(output_shape[-1])])
  res = nn.sparse_softmax_cross_entropy_with_logits(
      labels=targets, logits=logits)
  if len(output_shape) >= 3:
    # If our output includes timesteps or spatial dimensions we need to reshape
    return array_ops.reshape(res, array_ops.shape(output)[:-1])
  else:
    return res


@keras_export('keras.backend.binary_crossentropy')
def binary_crossentropy(target, output, from_logits=False):
  """Binary crossentropy between an output tensor and a target tensor.

  Arguments:
      target: A tensor with the same shape as `output`.
      output: A tensor.
      from_logits: Whether `output` is expected to be a logits tensor.
          By default, we consider that `output`
          encodes a probability distribution.

  Returns:
      A tensor.
  """
  if not from_logits:
    if context.executing_eagerly() or output.op.type != 'Sigmoid':
      epsilon_ = _to_tensor(epsilon(), output.dtype.base_dtype)
      output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)

      # Compute cross entropy from probabilities.
      bce = target * math_ops.log(output + epsilon())
      bce += (1 - target) * math_ops.log(1 - output + epsilon())
      return -bce
    else:
      # When sigmoid activation function is used for output operation, we
      # use logits from the sigmoid function directly to compute loss in order
      # to prevent collapsing zero when training.
      assert len(output.op.inputs) == 1
      output = output.op.inputs[0]
  return nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)


@keras_export('keras.backend.sigmoid')
def sigmoid(x):
  """Element-wise sigmoid.

  Arguments:
      x: A tensor or variable.

  Returns:
      A tensor.
  """
  return nn.sigmoid(x)


@keras_export('keras.backend.hard_sigmoid')
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


@keras_export('keras.backend.tanh')
def tanh(x):
  """Element-wise tanh.

  Arguments:
      x: A tensor or variable.

  Returns:
      A tensor.
  """
  return nn.tanh(x)


@keras_export('keras.backend.dropout')
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
  # (float32_ref vs. float32 incompatibility)
  return nn.dropout(x * 1., retain_prob, noise_shape, seed=seed)


@keras_export('keras.backend.l2_normalize')
def l2_normalize(x, axis=None):
  """Normalizes a tensor wrt the L2 norm alongside the specified axis.

  Arguments:
      x: Tensor or variable.
      axis: axis along which to perform normalization.

  Returns:
      A tensor.
  """
  return nn.l2_normalize(x, axis=axis)


@keras_export('keras.backend.in_top_k')
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


def _preprocess_conv1d_input(x, data_format):
  """Transpose and cast the input before the conv1d.

  Arguments:
      x: input tensor.
      data_format: string, `"channels_last"` or `"channels_first"`.

  Returns:
      A tensor.
  """
  tf_data_format = 'NWC'  # to pass TF Conv2dNative operations
  if data_format == 'channels_first':
    if not _has_nchw_support():
      x = array_ops.transpose(x, (0, 2, 1))  # NCW -> NWC
    else:
      tf_data_format = 'NCW'
  return x, tf_data_format


def _preprocess_conv2d_input(x, data_format, force_transpose=False):
  """Transpose and cast the input before the conv2d.

  Arguments:
      x: input tensor.
      data_format: string, `"channels_last"` or `"channels_first"`.
      force_transpose: Boolean. If True, the input will always be transposed
          from NCHW to NHWC if `data_format` is `"channels_first"`.
          If False, the transposition only occurs on CPU (GPU ops are
          assumed to support NCHW).

  Returns:
      A tensor.
  """
  tf_data_format = 'NHWC'
  if data_format == 'channels_first':
    if not _has_nchw_support() or force_transpose:
      x = array_ops.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
    else:
      tf_data_format = 'NCHW'
  return x, tf_data_format


def _preprocess_conv3d_input(x, data_format):
  """Transpose and cast the input before the conv3d.

  Arguments:
      x: input tensor.
      data_format: string, `"channels_last"` or `"channels_first"`.

  Returns:
      A tensor.
  """
  tf_data_format = 'NDHWC'
  if data_format == 'channels_first':
    if not _has_nchw_support():
      x = array_ops.transpose(x, (0, 2, 3, 4, 1))
    else:
      tf_data_format = 'NCDHW'
  return x, tf_data_format


def _preprocess_padding(padding):
  """Convert keras' padding to TensorFlow's padding.

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
    raise ValueError('Invalid padding: ' + str(padding))
  return padding


@keras_export('keras.backend.conv1d')
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

  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  kernel_shape = kernel.shape.as_list()
  if padding == 'causal':
    # causal (dilated) convolution:
    left_pad = dilation_rate * (kernel_shape[0] - 1)
    x = temporal_padding(x, (left_pad, 0))
    padding = 'valid'
  padding = _preprocess_padding(padding)

  x, tf_data_format = _preprocess_conv1d_input(x, data_format)
  x = nn.convolution(
      input=x,
      filter=kernel,
      dilation_rate=(dilation_rate,),
      strides=(strides,),
      padding=padding,
      data_format=tf_data_format)
  if data_format == 'channels_first' and tf_data_format == 'NWC':
    x = array_ops.transpose(x, (0, 2, 1))  # NWC -> NCW
  return x


@keras_export('keras.backend.conv2d')
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
    raise ValueError('Unknown data_format: ' + str(data_format))

  x, tf_data_format = _preprocess_conv2d_input(x, data_format)
  padding = _preprocess_padding(padding)
  x = nn.convolution(
      input=x,
      filter=kernel,
      dilation_rate=dilation_rate,
      strides=strides,
      padding=padding,
      data_format=tf_data_format)
  if data_format == 'channels_first' and tf_data_format == 'NHWC':
    x = array_ops.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
  return x


@keras_export('keras.backend.conv2d_transpose')
def conv2d_transpose(x,
                     kernel,
                     output_shape,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1)):
  """2D deconvolution (i.e.

  transposed convolution).

  Arguments:
      x: Tensor or variable.
      kernel: kernel tensor.
      output_shape: 1D int tensor for the output shape.
      strides: strides tuple.
      padding: string, `"same"` or `"valid"`.
      data_format: string, `"channels_last"` or `"channels_first"`.
          Whether to use Theano or TensorFlow/CNTK data format
          for inputs/kernels/outputs.
      dilation_rate: Tuple of 2 integers.

  Returns:
      A tensor, result of transposed 2D convolution.

  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))
  if isinstance(output_shape, (tuple, list)):
    output_shape = array_ops.stack(output_shape)

  # `atrous_conv2d_transpose` only supports NHWC format, even on GPU.
  if data_format == 'channels_first' and dilation_rate != (1, 1):
    force_transpose = True
  else:
    force_transpose = False

  x, tf_data_format = _preprocess_conv2d_input(x, data_format, force_transpose)

  if data_format == 'channels_first' and tf_data_format == 'NHWC':
    output_shape = (output_shape[0], output_shape[2], output_shape[3],
                    output_shape[1])
  if output_shape[0] is None:
    output_shape = (array_ops.shape(x)[0],) + tuple(output_shape[1:])
    output_shape = array_ops.stack(list(output_shape))

  padding = _preprocess_padding(padding)
  if tf_data_format == 'NHWC':
    strides = (1,) + strides + (1,)
  else:
    strides = (1, 1) + strides

  if dilation_rate == (1, 1):
    x = nn.conv2d_transpose(x, kernel, output_shape, strides,
                            padding=padding,
                            data_format=tf_data_format)
  else:
    assert dilation_rate[0] == dilation_rate[1]
    x = nn.atrous_conv2d_transpose(
        x,
        kernel,
        output_shape,
        rate=dilation_rate[0],
        padding=padding)
  if data_format == 'channels_first' and tf_data_format == 'NHWC':
    x = array_ops.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
  return x


def separable_conv1d(x,
                     depthwise_kernel,
                     pointwise_kernel,
                     strides=1,
                     padding='valid',
                     data_format=None,
                     dilation_rate=1):
  """1D convolution with separable filters.

  Arguments:
      x: input tensor
      depthwise_kernel: convolution kernel for the depthwise convolution.
      pointwise_kernel: kernel for the 1x1 convolution.
      strides: stride integer.
      padding: string, `"same"` or `"valid"`.
      data_format: string, `"channels_last"` or `"channels_first"`.
      dilation_rate: integer dilation rate.

  Returns:
      Output tensor.

  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  if isinstance(strides, int):
    strides = (strides,)
  if isinstance(dilation_rate, int):
    dilation_rate = (dilation_rate,)

  x, tf_data_format = _preprocess_conv1d_input(x, data_format)
  padding = _preprocess_padding(padding)
  if not isinstance(strides, tuple):
    strides = tuple(strides)
  if tf_data_format == 'NWC':
    spatial_start_dim = 1
    strides = (1,) + strides * 2 + (1,)
  else:
    spatial_start_dim = 2
    strides = (1, 1) + strides * 2
  x = array_ops.expand_dims(x, spatial_start_dim)
  depthwise_kernel = array_ops.expand_dims(depthwise_kernel, 0)
  pointwise_kernel = array_ops.expand_dims(pointwise_kernel, 0)
  dilation_rate = (1,) + dilation_rate

  x = nn.separable_conv2d(
      x,
      depthwise_kernel,
      pointwise_kernel,
      strides=strides,
      padding=padding,
      rate=dilation_rate,
      data_format=tf_data_format)

  x = array_ops.squeeze(x, [spatial_start_dim])

  if data_format == 'channels_first' and tf_data_format == 'NWC':
    x = array_ops.transpose(x, (0, 2, 1))  # NWC -> NCW

  return x


@keras_export('keras.backend.separable_conv2d')
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
      padding: string, `"same"` or `"valid"`.
      data_format: string, `"channels_last"` or `"channels_first"`.
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
    raise ValueError('Unknown data_format: ' + str(data_format))
  if len(strides) != 2:
    raise ValueError('`strides` must be a tuple of 2 integers.')

  x, tf_data_format = _preprocess_conv2d_input(x, data_format)
  padding = _preprocess_padding(padding)
  if not isinstance(strides, tuple):
    strides = tuple(strides)
  if tf_data_format == 'NHWC':
    strides = (1,) + strides + (1,)
  else:
    strides = (1, 1) + strides

  x = nn.separable_conv2d(
      x,
      depthwise_kernel,
      pointwise_kernel,
      strides=strides,
      padding=padding,
      rate=dilation_rate,
      data_format=tf_data_format)
  if data_format == 'channels_first' and tf_data_format == 'NHWC':
    x = array_ops.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
  return x


def depthwise_conv2d(x,
                     depthwise_kernel,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1)):
  """2D convolution with separable filters.

  Arguments:
      x: input tensor
      depthwise_kernel: convolution kernel for the depthwise convolution.
      strides: strides tuple (length 2).
      padding: string, `"same"` or `"valid"`.
      data_format: string, `"channels_last"` or `"channels_first"`.
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
    raise ValueError('Unknown data_format: ' + str(data_format))

  x, tf_data_format = _preprocess_conv2d_input(x, data_format)
  padding = _preprocess_padding(padding)
  if tf_data_format == 'NHWC':
    strides = (1,) + strides + (1,)
  else:
    strides = (1, 1) + strides

  x = nn.depthwise_conv2d(
      x,
      depthwise_kernel,
      strides=strides,
      padding=padding,
      rate=dilation_rate,
      data_format=tf_data_format)
  if data_format == 'channels_first' and tf_data_format == 'NHWC':
    x = array_ops.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
  return x


@keras_export('keras.backend.conv3d')
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
      data_format: string, `"channels_last"` or `"channels_first"`.
          Whether to use Theano or TensorFlow/CNTK data format
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
    raise ValueError('Unknown data_format: ' + str(data_format))

  x, tf_data_format = _preprocess_conv3d_input(x, data_format)
  padding = _preprocess_padding(padding)
  x = nn.convolution(
      input=x,
      filter=kernel,
      dilation_rate=dilation_rate,
      strides=strides,
      padding=padding,
      data_format=tf_data_format)
  if data_format == 'channels_first' and tf_data_format == 'NDHWC':
    x = array_ops.transpose(x, (0, 4, 1, 2, 3))
  return x


def conv3d_transpose(x,
                     kernel,
                     output_shape,
                     strides=(1, 1, 1),
                     padding='valid',
                     data_format=None):
  """3D deconvolution (i.e.

  transposed convolution).

  Arguments:
      x: input tensor.
      kernel: kernel tensor.
      output_shape: 1D int tensor for the output shape.
      strides: strides tuple.
      padding: string, "same" or "valid".
      data_format: string, `"channels_last"` or `"channels_first"`.
          Whether to use Theano or TensorFlow/CNTK data format
          for inputs/kernels/outputs.

  Returns:
      A tensor, result of transposed 3D convolution.

  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))
  if isinstance(output_shape, (tuple, list)):
    output_shape = array_ops.stack(output_shape)

  x, tf_data_format = _preprocess_conv3d_input(x, data_format)

  if data_format == 'channels_first' and tf_data_format == 'NDHWC':
    output_shape = (output_shape[0], output_shape[2], output_shape[3],
                    output_shape[4], output_shape[1])
  if output_shape[0] is None:
    output_shape = (array_ops.shape(x)[0],) + tuple(output_shape[1:])
    output_shape = array_ops.stack(list(output_shape))

  padding = _preprocess_padding(padding)
  if tf_data_format == 'NDHWC':
    strides = (1,) + strides + (1,)
  else:
    strides = (1, 1) + strides

  x = nn.conv3d_transpose(
      x,
      kernel,
      output_shape,
      strides,
      padding=padding,
      data_format=tf_data_format)
  if data_format == 'channels_first' and tf_data_format == 'NDHWC':
    x = array_ops.transpose(x, (0, 4, 1, 2, 3))
  return x


@keras_export('keras.backend.pool2d')
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
      padding: string, `"same"` or `"valid"`.
      data_format: string, `"channels_last"` or `"channels_first"`.
      pool_mode: string, `"max"` or `"avg"`.

  Returns:
      A tensor, result of 2D pooling.

  Raises:
      ValueError: if `data_format` is neither `"channels_last"` or
      `"channels_first"`.
      ValueError: if `pool_mode` is neither `"max"` or `"avg"`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))
  if len(pool_size) != 2:
    raise ValueError('`pool_size` must be a tuple of 2 integers.')
  if len(strides) != 2:
    raise ValueError('`strides` must be a tuple of 2 integers.')

  x, tf_data_format = _preprocess_conv2d_input(x, data_format)
  padding = _preprocess_padding(padding)
  if tf_data_format == 'NHWC':
    strides = (1,) + strides + (1,)
    pool_size = (1,) + pool_size + (1,)
  else:
    strides = (1, 1) + strides
    pool_size = (1, 1) + pool_size

  if pool_mode == 'max':
    x = nn.max_pool(
        x, pool_size, strides, padding=padding, data_format=tf_data_format)
  elif pool_mode == 'avg':
    x = nn.avg_pool(
        x, pool_size, strides, padding=padding, data_format=tf_data_format)
  else:
    raise ValueError('Invalid pooling mode: ' + str(pool_mode))

  if data_format == 'channels_first' and tf_data_format == 'NHWC':
    x = array_ops.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
  return x


@keras_export('keras.backend.pool3d')
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
      padding: string, `"same"` or `"valid"`.
      data_format: string, `"channels_last"` or `"channels_first"`.
      pool_mode: string, `"max"` or `"avg"`.

  Returns:
      A tensor, result of 3D pooling.

  Raises:
      ValueError: if `data_format` is neither `"channels_last"` or
      `"channels_first"`.
      ValueError: if `pool_mode` is neither `"max"` or `"avg"`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  x, tf_data_format = _preprocess_conv3d_input(x, data_format)
  padding = _preprocess_padding(padding)
  if tf_data_format == 'NDHWC':
    strides = (1,) + strides + (1,)
    pool_size = (1,) + pool_size + (1,)
  else:
    strides = (1, 1) + strides
    pool_size = (1, 1) + pool_size

  if pool_mode == 'max':
    x = nn.max_pool3d(
        x, pool_size, strides, padding=padding, data_format=tf_data_format)
  elif pool_mode == 'avg':
    x = nn.avg_pool3d(
        x, pool_size, strides, padding=padding, data_format=tf_data_format)
  else:
    raise ValueError('Invalid pooling mode: ' + str(pool_mode))

  if data_format == 'channels_first' and tf_data_format == 'NDHWC':
    x = array_ops.transpose(x, (0, 4, 1, 2, 3))
  return x


def local_conv(inputs,
               kernel,
               kernel_size,
               strides,
               output_shape,
               data_format=None):
  """Apply N-D convolution with un-shared weights.

  Arguments:
      inputs: (N+2)-D tensor with shape
          (batch_size, channels_in, d_in1, ..., d_inN)
          if data_format='channels_first', or
          (batch_size, d_in1, ..., d_inN, channels_in)
          if data_format='channels_last'.
      kernel: the unshared weight for N-D convolution,
          with shape (output_items, feature_dim, channels_out), where
          feature_dim = np.prod(kernel_size) * channels_in,
          output_items = np.prod(output_shape).
      kernel_size: a tuple of N integers, specifying the
          spatial dimensions of the N-D convolution window.
      strides: a tuple of N integers, specifying the strides
          of the convolution along the spatial dimensions.
      output_shape: a tuple of (d_out1, ..., d_outN) specifying the spatial
          dimensionality of the output.
      data_format: string, "channels_first" or "channels_last".

  Returns:
      An (N+2)-D tensor with shape:
      (batch_size, channels_out) + output_shape
      if data_format='channels_first', or:
      (batch_size,) + output_shape + (channels_out,)
      if data_format='channels_last'.

  Raises:
      ValueError: if `data_format` is neither
      `channels_last` nor `channels_first`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  kernel_shape = int_shape(kernel)
  feature_dim = kernel_shape[1]
  channels_out = kernel_shape[-1]
  ndims = len(output_shape)
  spatial_dimensions = list(range(ndims))

  xs = []
  output_axes_ticks = [range(axis_max) for axis_max in output_shape]
  for position in itertools.product(*output_axes_ticks):
    slices = [slice(None)]

    if data_format == 'channels_first':
      slices.append(slice(None))

    slices.extend([slice(position[d] * strides[d],
                         position[d] * strides[d] + kernel_size[d])
                   for d in spatial_dimensions])

    if data_format == 'channels_last':
      slices.append(slice(None))

    xs.append(reshape(inputs[slices], (1, -1, feature_dim)))

  x_aggregate = concatenate(xs, axis=0)
  output = batch_dot(x_aggregate, kernel)
  output = reshape(output, output_shape + (-1, channels_out))

  if data_format == 'channels_first':
    permutation = [ndims, ndims + 1] + spatial_dimensions
  else:
    permutation = [ndims] + spatial_dimensions + [ndims + 1]

  return permute_dimensions(output, permutation)


def local_conv1d(inputs, kernel, kernel_size, strides, data_format=None):
  """Apply 1D conv with un-shared weights.

  Arguments:
      inputs: 3D tensor with shape:
          (batch_size, steps, input_dim)
          if data_format is "channels_last" or
          (batch_size, input_dim, steps)
          if data_format is "channels_first".
      kernel: the unshared weight for convolution,
          with shape (output_length, feature_dim, filters).
      kernel_size: a tuple of a single integer,
          specifying the length of the 1D convolution window.
      strides: a tuple of a single integer,
          specifying the stride length of the convolution.
      data_format: the data format, channels_first or channels_last.

  Returns:
      A 3d tensor with shape:
      (batch_size, output_length, filters)
      if data_format='channels_first'
      or 3D tensor with shape:
      (batch_size, filters, output_length)
      if data_format='channels_last'.
  """
  output_shape = (kernel.shape[0],)
  return local_conv(inputs,
                    kernel,
                    kernel_size,
                    strides,
                    output_shape,
                    data_format)


def local_conv2d(inputs,
                 kernel,
                 kernel_size,
                 strides,
                 output_shape,
                 data_format=None):
  """Apply 2D conv with un-shared weights.

  Arguments:
      inputs: 4D tensor with shape:
          (batch_size, filters, new_rows, new_cols)
          if data_format='channels_first'
          or 4D tensor with shape:
          (batch_size, new_rows, new_cols, filters)
          if data_format='channels_last'.
      kernel: the unshared weight for convolution,
          with shape (output_items, feature_dim, filters).
      kernel_size: a tuple of 2 integers, specifying the
          width and height of the 2D convolution window.
      strides: a tuple of 2 integers, specifying the strides
          of the convolution along the width and height.
      output_shape: a tuple with (output_row, output_col).
      data_format: the data format, channels_first or channels_last.

  Returns:
      A 4D tensor with shape:
      (batch_size, filters, new_rows, new_cols)
      if data_format='channels_first'
      or 4D tensor with shape:
      (batch_size, new_rows, new_cols, filters)
      if data_format='channels_last'.
  """
  return local_conv(inputs,
                    kernel,
                    kernel_size,
                    strides,
                    output_shape,
                    data_format)


@keras_export('keras.backend.bias_add')
def bias_add(x, bias, data_format=None):
  """Adds a bias vector to a tensor.

  Arguments:
      x: Tensor or variable.
      bias: Bias tensor to add.
      data_format: string, `"channels_last"` or `"channels_first"`.

  Returns:
      Output tensor.

  Raises:
      ValueError: In one of the two cases below:
                  1. invalid `data_format` argument.
                  2. invalid bias shape.
                     the bias should be either a vector or
                     a tensor with ndim(x) - 1 dimension
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))
  bias_shape = int_shape(bias)
  if len(bias_shape) != 1 and len(bias_shape) != ndim(x) - 1:
    raise ValueError(
        'Unexpected bias dimensions %d, expect to be 1 or %d dimensions' %
        (len(bias_shape), ndim(x)))
  # pylint: disable=g-no-augmented-assignment
  if ndim(x) == 5:
    if data_format == 'channels_first':
      if len(bias_shape) == 1:
        x = x + reshape(bias, (1, bias_shape[0], 1, 1, 1))
      else:
        x = x + reshape(bias, (1, bias_shape[3]) + bias_shape[:3])
    elif data_format == 'channels_last':
      if len(bias_shape) == 1:
        x = x + reshape(bias, (1, 1, 1, bias_shape[0]))
      else:
        x = x + reshape(bias, (1,) + bias_shape)
  elif ndim(x) == 4:
    if data_format == 'channels_first':
      if len(bias_shape) == 1:
        if _has_nchw_support():
          x = nn.bias_add(x, bias, data_format='NCHW')
        else:
          x = x + reshape(bias, (1, bias_shape[0], 1, 1))
      else:
        x = x + reshape(bias, (1, bias_shape[2]) + bias_shape[:2])
    elif data_format == 'channels_last':
      if len(bias_shape) == 1:
        x = nn.bias_add(x, bias, data_format='NHWC')
      else:
        x = x + reshape(bias, (1,) + bias_shape)
  elif ndim(x) == 3:
    if data_format == 'channels_first':
      if len(bias_shape) == 1:
        x = x + reshape(bias, (1, bias_shape[0], 1))
      else:
        x = x + reshape(bias, (1, bias_shape[1], bias_shape[0]))
    elif data_format == 'channels_last':
      if len(bias_shape) == 1:
        x = x + reshape(bias, (1, 1, bias_shape[0]))
      else:
        x = x + reshape(bias, (1,) + bias_shape)
  else:
    x = nn.bias_add(x, bias)
  # pylint: enable=g-no-augmented-assignment
  return x


# RANDOMNESS


@keras_export('keras.backend.random_normal')
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


@keras_export('keras.backend.random_uniform')
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


@keras_export('keras.backend.random_binomial')
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


@keras_export('keras.backend.truncated_normal')
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
# TensorFlow has a native implementation, but it uses sparse tensors
# and therefore requires a wrapper for Keras. The functions below convert
# dense to sparse tensors and also wraps up the beam search code that is
# in TensorFlow's CTC implementation


@keras_export('keras.backend.ctc_label_dense_to_sparse')
def ctc_label_dense_to_sparse(labels, label_lengths):
  """Converts CTC labels from dense to sparse.

  Arguments:
      labels: dense CTC labels.
      label_lengths: length of the labels.

  Returns:
      A sparse tensor representation of the labels.
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


@keras_export('keras.backend.ctc_batch_cost')
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
  label_length = math_ops.to_int32(array_ops.squeeze(label_length, axis=-1))
  input_length = math_ops.to_int32(array_ops.squeeze(input_length, axis=-1))
  sparse_labels = math_ops.to_int32(
      ctc_label_dense_to_sparse(y_true, label_length))

  y_pred = math_ops.log(array_ops.transpose(y_pred, perm=[1, 0, 2]) + epsilon())

  return array_ops.expand_dims(
      ctc.ctc_loss(
          inputs=y_pred, labels=sparse_labels, sequence_length=input_length), 1)


@keras_export('keras.backend.ctc_decode')
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
  y_pred = math_ops.log(array_ops.transpose(y_pred, perm=[1, 0, 2]) + epsilon())
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


@keras_export('keras.backend.map_fn')
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


@keras_export('keras.backend.foldl')
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


@keras_export('keras.backend.foldr')
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
# Set Keras base dir path given KERAS_HOME env variable, if applicable.
# Otherwise either ~/.keras or /tmp.
if 'KERAS_HOME' in os.environ:
  _keras_dir = os.environ.get('KERAS_HOME')
else:
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
