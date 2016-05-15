# Implementation of Immediate Env
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["Env", "Namespace"]

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

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

from tensorflow.python.platform import tf_logging as logging

# Native TF ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_candidate_sampling_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_ctc_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import gen_script_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import gen_user_ops

# Python-only ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import session_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_ops

from tensorflow.python.framework import ops

import inspect
import re
import os
import types

from .tensor import Tensor
from .op import OpFactory
from .op import OpWrapper
from .op import PythonOpWrapper
from .op import ConstantOpWrapper
from .op import ConvertToTensorWrapper
from .op import Op

import numpy as np
import wrapping_util

class Env(object):

  # TODO(yaroslavvb): use dtypes.as_dtype(dtype) is not None instead
  # note: np.int32 has different hash from np.dtype('int32'), so must use later
  supported_numpy_types = {np.dtype('int32'), np.dtype('int64'),
                           np.dtype('float32'), np.dtype('float64'),
                           np.dtype('bool')}

  def __init__(self, tf_namespace):
    self.sess = session.Session()
    self.op_factory = OpFactory(self)


    # wrap all tensorflow op modules. Keep track of already wrapped modules
    # for __globals__ substitution dictionary "sub" 
    sub = {}
    self.wrapped_namespaces = {}


    # wrap ops namespace (to override convert_to_tensor method)
    wrapped_namespace = Namespace(self, "ops", ops, tf_root=False)
    sub["ops"] = wrapped_namespace
    
    for op_module in wrapping_util.gen_op_module_list:
      wrapped_namespace = Namespace(self, op_module, eval(op_module),
                                    tf_root=False)
      self.wrapped_namespaces[op_module] = wrapped_namespace
      sub[op_module] = wrapped_namespace

    for op_module in wrapping_util.python_op_module_list_sorted():
      wrapped_namespace = Namespace(self, op_module,
                                    eval(op_module),
                                    tf_root=False, global_sub=sub)

      self.wrapped_namespaces[op_module] = wrapped_namespace
      sub[op_module] = wrapped_namespace

    self.tf =  Namespace(self, "tf", tf_namespace, tf_root=True,
                         global_sub=sub)


    # TODO(yaroslavvb): add run options
    #    self.run_options = tf.RunOptions()

  @property
  def g(self):
    return self.sess.graph

  @property
  def python_op_whitelist(self):
    """Python-only ops whitelisted for wrapping."""
    return {"tf.pow", "tf.reduce_sum", "tf.range", "tf.random_uniform"}

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

  # TODO(yaroslavvb): test bad conversions

  def numpy_to_tensor(self, array, dtype=None):
    # convert to numpy dtype if necessary
    if dtype:
      dtype = dtypes.as_dtype(dtype)
      dtype = dtype.as_numpy_dtype

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

    handle = self.numpy_to_handle(array)
    return Tensor(self, handle)

  def constant(self, values, dtype=None, shape=None, name='Const'):
    if shape:
      assert (isinstance(values, types.FloatType) or
              isinstance(values, types.IntType) or
              isinstance(values, types.LongType))
      return self.numpy_to_tensor(values*np.ones(shape=shape), dtype=dtype)
    return self.numpy_to_tensor(values, dtype)

  
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

# TODO(yaroslavvb): better default for tf_root
class Namespace(object):
  """Object that is capable of mirroring namespace like "tf" but with immediate
  execution semantics."""

  def __init__(self, env, name, namespace, global_sub = {}, tf_root=True):
    """Initializes TF namespace wrapper
    Args:
      global_sub: stores a dictionary used to substituted __globals__ symbols
      """
    # Walk the tree of symbols in namespace and save mappings

    self.env = env
    self.name = name
    self.original_namespace = namespace
    self.global_sub = global_sub

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

    derived_symbol_name = self.name+"."+symbol_name

    # constant is #1 most often used op, treat it specially
    if symbol_name == 'constant':
      return ConstantOpWrapper(self, self.env, derived_symbol_name)

    if symbol_name == 'convert_to_tensor':
      return ConvertToTensorWrapper(self, self.env, derived_symbol_name)

    if symbol_name in self.nested_modules:
      return self.nested_modules[symbol_name]

      # TODO(yaroslavvb): remove duplication
    if symbol_name in self.gen_ops:
      symbol = self.gen_ops[symbol_name]
      return OpWrapper(self, self.env, derived_symbol_name,
                       symbol)
    elif symbol_name in self.python_ops:
      if not derived_symbol_name in self.env.python_op_whitelist:
        raise ValueError("Python-only op %s is not whitelisted." %
                         (derived_symbol_name))
      symbol = self.python_ops[symbol_name]
      return PythonOpWrapper(self, self.env, derived_symbol_name,
                             symbol, self.global_sub)
    # pass through the symbol to original implementation
    elif symbol_name in self.other:
      symbol = self.other[symbol_name]
      return symbol
    else:
      raise ValueError("Do not have implementation of op "+derived_symbol_name)    


  def __str__(self):
    return "Namespace(name=%s, original_namespace=%s)" %(self.name,
                                                         self.original_namespace.__name__)

  def __repr__(self):
    return "Namespace(name=%s, original_namespace=%s, global_sub=%s)" % (
      self.name, self.original_namespace, self.global_sub)

