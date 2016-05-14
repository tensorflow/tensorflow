# Implementation of Immediate Env
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["Env"]

from tensorflow.python.framework import ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import graph_util
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.ops import session_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging

import inspect
import re
import os
import types

from .tensor import Tensor
from .op import OpFactory
from .op import OpWrapper
from .op import Op

import numpy as np


class Env(object):

  supported_numpy_types = {np.dtype('int32'), np.dtype('int64'),
                           np.dtype('float32'), np.dtype('float64'),
                           np.dtype('bool')}

  def __init__(self):
    self.sess = session.Session()
    self.op_factory = OpFactory(self)
    #    self.run_options = tf.RunOptions()

    # Walk the tree of symbols in "tf." namespace and save mappings
    # <generaldep> tfnamespace must be passed as an argument to __init__
    # if we want Env to be part of TensorFlow wherein can't use import tensorflow
    import tensorflow as tfnamespace

    self.tf_gen_ops = {}  # native Python op wrappers
    self.tf_python_ops = {}  # custom Python op functions
    self.tf_other = {}  # everything else

    for (name,symbol) in tfnamespace.__dict__.items():
      # only include functions
      if type(symbol) == types.FunctionType:
        basename = os.path.basename(inspect.getsourcefile(symbol))
        if re.match("^gen.*_ops.py$", basename):
          self.tf_gen_ops[name] = symbol
        elif re.match("^.*_ops.py$", basename):
          self.tf_python_ops[name] = symbol
        else:
          self.tf_other[name] = symbol
      else:
          self.tf_other[name] = symbol

      # add tf.nn functions
      # TODO(yaroslavvb): refactor this to include both tf.image and tf.nn
    for (name,symbol) in tfnamespace.nn.__dict__.items():
      # only include functions
      name = 'nn.'+name
      if type(symbol) == types.FunctionType:
        basename = os.path.basename(inspect.getsourcefile(symbol))
        if re.match("^gen.*_ops.py$", basename):
          self.tf_gen_ops[name] = symbol
        elif re.match("^.*_ops.py$", basename):
          self.tf_python_ops[name] = symbol
        else:
          self.tf_other[name] = symbol
      else:
          self.tf_other[name] = symbol

    # wrapper to handle 'env.nn' calls
    self.nn_submodule = EnvSubmoduleWrapper(self, 'nn.')


  def __getattr__(self, tf_op_name):
    if tf_op_name == 'nn':
      return self.nn_submodule

    if tf_op_name in self.tf_gen_ops:
      return OpWrapper(self, tf_op_name)
    else:
      raise ValueError("Do not have implementation of op "+tf_op_name)    

  @property
  def g(self):
    return self.sess.graph

  # Ops below are used by internal implementation of the immediate execution
  # system, so we get infinite recursion if we dispatch them through
  # immediate execution, so instead we implement them manually below
  # TODO(yaroslavvb): implement graph caching logic for these ops

  def handle_to_numpy(self, tensor_handle):
    """Downloads contents of TensorHandle and returns corresponding numpy array.

    Args:
      tensor_handle: session_ops.TensorHandle object

    Returns:
      numpy array with a copy of data from tensor_handle
    """

    holder, tensor = session_ops.get_session_tensor(tensor_handle._dtype)

    # TODO(yaroslavvb): use session settings for .run call
    array = self.sess.run(tensor, feed_dict={holder: tensor_handle.handle})
    return array


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

  def numpy_to_tensor(self, array):
    # try to convert Python lists to numpy array
    if not isinstance(array, np.ndarray):
      array = np.array(array)
      if not array.dtype in self.supported_numpy_types:
        raise ValueError("Unsupported type %s, only support types %s" % (
            repr(array.dtype), [repr(s) for s in self.supported_numpy_types]))

    handle = self.numpy_to_handle(array)
    return Tensor(self, handle)
  
  def tensor_to_numpy(self, tensor):
    return self.handle_to_numpy(tensor.handle)
    handle = self.numpy_to_handle(array)
    return Tensor(env, handle)
  
  def get_session_tensor(self, dtype):
    holder, tensor = session_ops.get_session_tensor(dtype)
    return holder, tensor

  def get_session_handle(self, tf_tensor):
    handle_op = session_ops.get_session_handle(tf_tensor)
    return handle_op
  
  def run(self, *args, **kwargs):
    return self.sess.run(*args, **kwargs)


class EnvSubmoduleWrapper(object):
  """Object serving as a wrapper of submodules for tf. namespace. IE, if "tf"
  namespace is represented by env, "tf.nn" will be an EnvSubmoduleWrapper that
  executes operations in the parent Environment."""
  
  def __init__(self, env, prefix):
    """Constructor for EnvSubmoduleWrapper.
    Args:
      env: Environment used for running operations
      prefix: Prefix corresponding to current wrapper, ie, ".nn" or ".image"
    """

    self.env = env
    self.prefix = prefix

  def __getattr__(self, tf_op_name):
    return self.env.__getattr__(self.prefix+tf_op_name)
