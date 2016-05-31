"""Implementation of Immediate execution environment."""

# Implementation of Immediate Env
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers
import numpy as np

from .tensor import Tensor
from .tensor import _ENABLE_DEBUG_LOGGING
from . import module_rewriter

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import session_ops

__all__ = ["Env"]



# np.float32 not found
# pylint: disable=no-member
# pyasdflint: disable=invalid-name

# Global env to be reused between threads. Since graph is not thread-safe
# for concurrent modifications, user must take care to not get cache misses
# in separate threads as that triggers graph modifications
_global_default_env = None

class Env(object):
  """Env is an object that manages current graph and session and translates
  user commands into appropriate session.run calls

  It contains implementation of methods that translate between Python and
  immediate Tensor representations.
  """


  # TODO(yaroslavvb): use dtypes.as_dtype(dtype) is not None instead to check
  # if type is supported, add support/tests for strings beyond |S3
  supported_numpy_types = {np.dtype('int32'), np.dtype('int64'),
                           np.dtype('float32'), np.dtype('float64'),
                           np.dtype('bool'), np.dtype("|S3")}

  def __init__(self, tf_namespace, config=None):
    global _global_default_env
    if _ENABLE_DEBUG_LOGGING:
      print("Creating Env")
    global _global_default_env
    self.g = ops.Graph()

    # Override user-setting for soft_placement
    # TODO(yaroslavvb): remove after #2587 is fixed
    if not config:
      config = config_pb2.ConfigProto()
    if not config.allow_soft_placement:
      config.allow_soft_placement = True

    self.sess = session.Session(config=config, graph=self.g)

    symbol_rewriter = module_rewriter.ImmediateRewriter(self)
    rewriter = module_rewriter.ModuleRewriter(symbol_rewriter, "immediate.")

    # tf_namespace is like <{"tf": tf, "gen_math_ops": gen_math_ops}>
    if isinstance(tf_namespace, dict):
      for name, namespace in tf_namespace.items():
        self.__dict__[name] = rewriter(namespace)
    else: # tf_namespace is like <tf>
      self.tf = rewriter(tf_namespace)

    _global_default_env = self

  @staticmethod
  def _get_global_default_env():
    """Get global env for reuse (ie, in tests)."""
    return _global_default_env

  def close(self):
    """Closes Env and frees its resources."""
    self.sess.close()


  @property
  def _graph_version(self):
    """Gives version of the graph. This can be used for checking if graph
    modifications took place"""
    return self.g.version


  # TODO(yaroslavvb): implement graph caching logic for these ops
  def handle_to_numpy(self, tensor_handle):
    """Downloads contents of TensorHandle and returns corresponding numpy array.

    Args:
      tensor_handle: session_ops.TensorHandle object

    Returns:
      numpy array with a copy of data from tensor_handle
    """

    with self.g.as_default():
      holder, tensor = session_ops.get_session_tensor(tensor_handle._dtype)

    # TODO(yaroslavvb): use Env's session settings for run call
    return self.sess.run(tensor, feed_dict={holder: tensor_handle.handle})

  def numpy_to_handle(self, array):
    """Uploads numpy array to TensorFlow runtime.

    Args:
      array: numpy array to convert to TensorHandle

    Returns:
      TensorHandle corresponding to given numpy array.
    """

    with self.g.as_default():
      holder = array_ops.placeholder(dtype=array.dtype)
      tensor_handle_op = session_ops.get_session_handle(holder)

    tensor_handle = self.sess.run(tensor_handle_op, feed_dict={holder: array})
    return tensor_handle

  # TODO(yaroslavvb): test bad conversions
  def numpy_to_tensor(self, array, dtype=None, shape=None):
    """Converts numpy.ndarray or compatible type to immediate.Tensor."""

    # convert to numpy dtype if necessary
    if dtype:
      dtype = dtypes.as_dtype(dtype)
      dtype = dtype.as_numpy_dtype

    if isinstance(array, Tensor):
      raise ValueError("Passed immediate.Tensor instead of numpy into "
                       "numpy_to_tensor.")

    # try to convert Python lists to numpy array
    if not isinstance(array, np.ndarray):
      array = np.array(array, dtype=dtype)
      if not array.dtype in self.supported_numpy_types:
        raise ValueError("Unsupported type %s, only support types %s" % (
            repr(array.dtype), [repr(s) for s in self.supported_numpy_types]))

    # Follow downcasting convention as in python/framework/tensor_util.py#L357
    # python/numpy default float type is float64. We prefer float32 instead.
    if (array.dtype == np.float64) and dtype is None:
      array = array.astype(np.float32)
    # python/numpy default int type is int64. We prefer int32 instead.
    elif (array.dtype == np.int64) and dtype is None:
      downcasted_array = array.astype(np.int32)
      # Do not down cast if it leads to precision loss.
      if np.array_equal(downcasted_array, array):
        array = downcasted_array

    if shape and array.shape != shape:
      array = array.reshape(shape)

    handle = self.numpy_to_handle(array)
    return Tensor(self, handle)

  def constant(self, values, dtype=None, shape=None, name='Const'):
    """Immediate specific implementation of constant-op."""

    np_dtype = None

    # Convert numpy dtype to TensorFlow dtype if needed
    if dtype:
      try:
        dtype = dtypes.as_dtype(dtype)
        np_dtype = dtype.as_numpy_dtype
      except TypeError as exc:
        raise TypeError("Trying to create constant with dtype=%s, "
                        "got TypeError(%s)" % (dtype, exc.message))

    # Native TensorFlow has special handling for TensorProto initialized with
    # a scalar and non-empty shape. For feature parity in immedate.Tensor we
    # handle this case by tiling the constant explicitly.
    if isinstance(values, numbers.Number) and shape:
      return self.numpy_to_tensor(values*np.ones(shape=shape, dtype=np_dtype),
                                  dtype=dtype, shape=shape)

    return self.numpy_to_tensor(values, dtype, shape)

  # TODO(yaroslavvb): make these ops use graph-caching
  def get_session_tensor(self, dtype):
    """Graph-caching enabled version of get_session_tensor"""

    with self.g.as_default():
      holder, tensor = session_ops.get_session_tensor(dtype)
      return holder, tensor

  def get_session_handle(self, tf_tensor):
    """Graph-caching enabled version of get_session_handle"""

    with self.g.as_default():
      handle_op = session_ops.get_session_handle(tf_tensor)
      return handle_op

  # TODO(yaroslavvb): add support for Env-specific run options
  def run(self, *args, **kwargs):
    """Execute session.run in the current Env."""

    return self.sess.run(*args, **kwargs)
