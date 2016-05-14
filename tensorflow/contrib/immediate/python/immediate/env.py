# Implementation of Immediate Env
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["Env", "Namespace"]

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
from tensorflow.python.ops import gen_math_ops
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
from .op import PythonOpWrapper
from .op import Op

import numpy as np


class Env(object):

  supported_numpy_types = {np.dtype('int32'), np.dtype('int64'),
                           np.dtype('float32'), np.dtype('float64'),
                           np.dtype('bool')}

  def __init__(self, tf_namespace):
    self.sess = session.Session()
    self.op_factory = OpFactory(self)
    self.tf =  Namespace(self, "tf", tf_namespace)

    # these will hold namespaces like gen_math_ops
    # that will hold wrapped versions of those namespaces
    # they will used when substituting immediate execution logic
    # for non-native Python TensorFlow ops
    self.gen_namespaces = {}
    self.gen_namespaces['gen_math_ops'] = Namespace(self, 'gen_math_ops',
                                                    gen_math_ops,
                                                    tf_root=False)

    # TODO(yaroslavvb): add run options
    #    self.run_options = tf.RunOptions()

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


class Namespace(object):
  """Object that is capable of mirroring namespace like "tf" but with immediate
  execution semantics."""

  def __init__(self, env, name, namespace, tf_root=True):
    # Walk the tree of symbols in namespace and save mappings

    self.env = env
    self.name = name

    self.gen_ops = {}  # native Python op wrappers
    self.python_ops = {}  # custom Python op functions
    self.other = {}  # everything else
    self.nested_modules = {}  # nested modules like tf.nn go here

    for (name,symbol) in namespace.__dict__.items():
      # only include functions
      if type(symbol) == types.FunctionType:
        basename = os.path.basename(inspect.getsourcefile(symbol))
        if re.match("^gen.*_ops.py$", basename):
          self.gen_ops[name] = symbol
        elif re.match("^.*_ops.py$", basename):
          self.python_ops[name] = symbol
        else:
          self.other[name] = symbol
      else:  # non-functions
          self.other[name] = symbol

    # if we are wrapping root of "tf." also wrap submodules like tf.nn
    if tf_root:
      self.nested_modules['nn'] = Namespace(self.env, self.name+'.nn',
                                            namespace.nn, tf_root=False)

  def __getattr__(self, symbol_name):
    if symbol_name in self.nested_modules:
      return self.nested_modules[symbol_name]

      # TODO(yaroslavvb): remove duplication
    derived_symbol_name = self.name+"."+symbol_name
    if symbol_name in self.gen_ops:
      symbol = self.gen_ops[symbol_name]
      return OpWrapper(self, self.env, derived_symbol_name,
                       symbol)
    elif symbol_name in self.python_ops:
      symbol = self.python_ops[symbol_name]
      return PythonOpWrapper(self, self.env, derived_symbol_name,
                             symbol)
    else:
      raise ValueError("Do not have implementation of op "+symbol_name)    
