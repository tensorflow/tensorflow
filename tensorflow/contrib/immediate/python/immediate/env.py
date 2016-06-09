"""Implementation of Immediate execution environment. All user-facing elements
of immediate execution framework should go here.

Env: immediate execution environment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers
import numpy as np

from . import itensor as itensor_lib
from . import module_rewriter as module_rewriter_lib
from . import util as util

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import session_ops

__all__ = ["Env"]

# Global env to be reused between threads.
_global_default_env = None

class Env(object):
  """Env is an object that manages current graph and session and translates
  user commands into appropriate session.run calls.

  The translation is done by means of wrapping all of tf functions that accept
  Tensor objects into a version that accepts ITensor objects that represent
  tensors with a concrete value.

  It keeps track of operations added to the current graph and tried to reuse
  parts of existing graph when possible.

  import tensorflow as tf
  env = immediate.Env(tf)
  c = env.tf.add(1, 2)

  """

  def __init__(self, tf_namespace, config=None):
    """Creates new immediate environment.

    Args:
      tf_namespace: tensorflow namespace to wrap, or a dictionary of namespace
          name-namespace pairs to wrap multiple namespaces
      config: ConfigProto to use for configuring session
    """

    global _global_default_env

    self.CACHE_ENABLED = True
    self.PRINT_CACHE_MISSES = False
    self.PRINT_CACHE_HITS = False

    self.op_cache = {}  # cache used for reusing parts of graph

    # Override user-setting for soft_placement
    # TODO(yaroslavvb): remove after #2587 is fixed
    if not config:
      config = config_pb2.ConfigProto()
    if not config.allow_soft_placement:
      config.allow_soft_placement = True

    self.g = ops_lib.Graph()
    self.sess = session.Session(config=config, graph=self.g)
    self._gc_default = self.session._DEAD_HANDLES_THRESHOLD

    # wrap provided namespace for immediate execution
    symbol_rewriter = module_rewriter_lib.ImmediateRewriter(self)
    rewriter = module_rewriter_lib.ModuleRewriter(symbol_rewriter, "immediate.")

    # tf_namespace is like {"tf": tf, "gen_math_ops": gen_math_ops}
    if isinstance(tf_namespace, dict):
      assert "tf" in tf_namespace, "Must include main tf namespace."
      self.original_tf = tf_namespace["tf"]
      for name, namespace in tf_namespace.items():
        self.__dict__[name] = rewriter(namespace)
    else: # tf_namespace is like "tf"
      self.original_tf = tf_namespace
      self.tf = rewriter(tf_namespace)
      
    _global_default_env = self


  def disable_gc(self):
    """Turn off garbage collection for persistent Tensors."""
    # it's saved in abstract base class of Session
    self.session.__class__.__base__._DEAD_HANDLES_THRESHOLD = 2**62


  def enable_gc(self):
    """Turn on garbage collection for persistent Tensors."""
    self.session.__class__.__base__._DEAD_HANDLES_THRESHOLD = self._gc_default



  @staticmethod
  def get_global_default_env():
    """Get global env for reuse (ie, in tests)."""
    return _global_default_env


  def close(self):
    """Close Env and free its resources."""
    self.sess.close()


  @property
  def session(self):
    """Session associated with current immediate env."""
    return self.sess


  @property
  def graph(self):
    """Graph associated with current immediate env."""
    return self.g


  @property
  def _graph_version(self):
    """Gives version of the graph. This can be used for checking if graph
    modifications took place"""
    return self.g.version


  @property
  def device(self):
    """Context manager for the default device to use in this env."""
    return self.g.device


  def cache_lookup(self, key):
    """Retrieve Op object from the cache."""
    if self.CACHE_ENABLED:
      if self.PRINT_CACHE_HITS:
        print("Immediate cache hit for %s" %(str(key)))
      return self.op_cache.get(key, None)


  def cache_add(self, key, op):
    """Add given Op object to the cache."""
    self.op_cache[key] = op


  def handle_to_numpy(self, tensor_handle):
    """Download contents of TensorHandle and return corresponding numpy array.

    Args:
      tensor_handle: session_ops.TensorHandle object

    Returns:
      numpy array with a copy of data from tensor_handle
    """

    tf_dtype = tensor_handle._dtype
    current_device = util.get_current_device_string(self.g)
    current_device_sanitized = current_device.replace(":", "")

    device_func = session_ops.TensorHandle._get_device_name
    handle_device = device_func(tensor_handle.handle)
    handle_device = util.shorten_device_string(handle_device)
    handle_device_sanitized = handle_device.replace(":", "")

    key = ("handle_to_numpy", tf_dtype.name, handle_device, current_device)

    if key in self.op_cache:
      holder, tensor = self.op_cache[key]
    else:
      if self.PRINT_CACHE_MISSES:
        print("Immediate cache miss for %s"%(str(key)))

      op_prefix = "handle_to_numpy.%s.%s.%s" % (tf_dtype.name,
                                                handle_device_sanitized,
                                                current_device_sanitized)
      with self.g.as_default():
        holder, tensor = session_ops.get_session_tensor(tensor_handle.handle,
                                                        tensor_handle._dtype,
                                                        name=op_prefix)
      self.op_cache[key] = (holder, tensor)

    return self.run(tensor, feed_dict={holder: tensor_handle.handle})


  def numpy_to_handle(self, array):
    """Upload numpy array into TensorFlow runtime.

    Args:
      array: numpy array to convert to TensorHandle

    Returns:
      TensorHandle corresponding to given numpy array.
    """

    tf_dtype = dtypes.as_dtype(array.dtype)
    current_device = util.get_current_device_string(self.g)
    current_device_sanitized = current_device.replace(":", "")
    key = ("numpy_to_handle", tf_dtype.name, current_device)

    if key in self.op_cache:
      holder, handle_op = self.op_cache[key]
    else:
      if self.PRINT_CACHE_MISSES:
        print("Immediate cache miss for %s"%(str(key)))

      op_prefix = "numpy_to_handle.%s.%s" % (tf_dtype.name,
                                             current_device_sanitized)
      with self.g.as_default():
        holder = array_ops.placeholder(dtype=array.dtype,
                                       name=op_prefix+".holder")
        handle_op = session_ops.get_session_handle(holder,
                                                   name=op_prefix+".handle")
      self.op_cache[key] = (holder, handle_op)

    handle = self.run(handle_op, feed_dict={holder: array})
    return handle


  def itensor_to_numpy(self, itensor):
    """Convert itensor to numpy array."""

    if itensor.env != self:
      raise ValueError("ITensor has incompatible env")
    return itensor.as_numpy()


  def numpy_to_itensor(self, array, dtype=None, shape=None):
    """Convert numpy.ndarray or compatible type to immediate.Tensor."""

    # convert to numpy dtype if necessary
    if dtype:
      tf_dtype = dtypes.as_dtype(dtype)
      np_dtype = tf_dtype.as_numpy_dtype
    else:
      np_dtype = None

    if isinstance(array, itensor_lib.ITensor):
      raise ValueError("Passed ITensor instead of numpy into "
                       "numpy_to_itensor.")

    # try to convert Python lists to numpy array
    if not isinstance(array, np.ndarray):
      array = np.array(array, dtype=np_dtype)
      tf_dtype = dtypes.as_dtype(array.dtype)

      if not tf_dtype or array.dtype == np.dtype("O"):
        raise ValueError("Unsupported type %s" %(type(array)))

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

    # if dtype is not None, and doesn't match given ndarray, convert to that
    # type
    if np_dtype and array.dtype != np_dtype:
      array = array.astype(np_dtype)

    if shape and array.shape != shape:
      array = array.reshape(shape)

    handle = self.numpy_to_handle(array)
    return itensor_lib.ITensor(self, handle)


  def constant(self, values, dtype=None, shape=None, name="Const"):
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
    # a scalar and non-empty shape. For feature parity with TensorFlow we
    # handle this case by tiling the constant explicitly.
    if isinstance(values, numbers.Number) and shape:
      return self.numpy_to_itensor(values*np.ones(shape=shape, dtype=np_dtype),
                                   dtype=dtype, shape=shape)

    return self.numpy_to_itensor(values, dtype, shape)

  
  def run(self, *args, **kwargs):
    """Execute session.run in the current Env."""
    return self.sess.run(*args, **kwargs)
