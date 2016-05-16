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
from .tensor import _ENABLE_DEBUG_LOGGING
from .op import OpFactory
from .op import OpWrapper 
from .op import PythonOpWrapper  # deprecated
from .op import ConstantOpWrapper
from .op import ConvertToTensorWrapper
from .op import Op

import numpy as np
#import wrapping_util

# TODO(yaroslavvb): organize this better
from .wrapping_util import gen_op_module_list
from .wrapping_util import python_op_module_list_sorted

# TODO(yaroslavvb): find a good place for get_canonical_name
canonical_name_re = re.compile(".*/(tensorflow/python/.*py)[c]?")
def get_canonical_name(fname):
  """Gets canonical name used to refer to TensorFlow modules.
  The reflects location in tf directory hierarchy, starting with
  tensorflow/python/...

  Ie, tensorflow/_python_build/tensorflow/python/ops/gen_math_ops.py becomes
  tensorflow/python/ops/gen_math_ops.py after canonicalizing."""

  groups = canonical_name_re.finall(fname)
  if groups and len(groups)==1:
    return groups[0]
  else:
    raise ValueError("Couldn't extract canonical name from %s, match groups "
                     "were %s" % (fname, groups))


class Env(object):
  """Env is an object that manages current graph and session and translates user facing commands into appropriate session.run calls.

  It contains implementations of manually ported methods like
"convert_to_tensor" (to mirror ops.convert_to_tensor), as well as handling
automatically wrapped methods (like _slice in gen_array_ops)

In addition, it runs the logic to remap a set of whitelisted Python-only operations to their immediate-only versions.

init_namespace assumes session and whitelisting has been initialized, and
launches namespace wrapping logic
"""


  # TODO(yaroslavvb): use dtypes.as_dtype(dtype) is not None instead
  # note: np.int32 has different hash from np.dtype('int32'), so must use later
  supported_numpy_types = {np.dtype('int32'), np.dtype('int64'),
                           np.dtype('float32'), np.dtype('float64'),
                           np.dtype('bool')}

  def __init__(self, tf_namespace):
    self.tf_namespace = tf_namespace
    self.sess = session.Session()
    self.op_factory = OpFactory(self)

    self.symbol_replacements = {}
    self.function_whitelist = set()
    self.module_whitelist = set()

    # TODO(yaroslavvb): remove last name arg in special wrappers
    self.register_replacement("tensorflow/python/framework/ops.py",
                              "convert_to_tensor",
                              ConvertToTensorWrapper(None, self,
                                                     "convert_to_tensor"))

    self.register_replacement('tensorflow/python/ops/constant_op.py',
                              "constant",
                              ConstantOpWrapper(None, self, "constant"))
    
    self.whitelist_module('tensorflow/python/ops/array_ops.py')
    self.whitelist_function('tensorflow/python/ops/array_ops.py',
                            'reduce_sum')
    

    # wrap all tensorflow op modules. Keep track of already wrapped modules
    # for __globals__ substitution dictionary "sub" 
    sub = {}
    self.wrapped_namespaces = {}

    # Wrap ops twice
    # Wrap op_def_library with ops replacement

    # Wrap gen.*ops modules
    for op_module in gen_op_module_list:
      wrapped_namespace = Namespace(self, op_module, eval(op_module),
                                    tf_root=False)
      self.wrapped_namespaces[op_module] = wrapped_namespace
      sub[op_module] = wrapped_namespace

    # convert_to_tensor is called on ops.convert_n_to_tensor
    sub["convert_to_tensor"] = ConvertToTensorWrapper(None, self,
                                                      "convert_to_tensor")
    wrapped_namespace = Namespace(self, "ops", ops, tf_root=False,
                                  global_sub=sub)
    sub["ops"] = wrapped_namespace

    # Because of array_ops: tensorflow.python.ops.constant_op import constant
    sub["constant"] =  ConstantOpWrapper(None, self, "constant")

    # Because of array_ops using "identity"
    #    sub["identity"] =  OpWrapper(None, self, "identity")
    sub["identity"] = sub["gen_array_ops"].identity

    for op_module in python_op_module_list_sorted():
      wrapped_namespace = Namespace(self, op_module,
                                    eval(op_module),
                                    tf_root=False, global_sub=sub)

      self.wrapped_namespaces[op_module] = wrapped_namespace
      sub[op_module] = wrapped_namespace

    self.tf =  Namespace(self, "tf", tf_namespace, tf_root=True,
                         global_sub=sub)


  def init_namespace(self):
    pass

    # TODO(yaroslavvb): add run options
    #    self.run_options = tf.RunOptions()

  def register_replacement(self, module_name, symbol_name, replacement):
    """Register module specific replacement."""

    sym_name = module_name + ":" + symbol_name
    self.symbol_replacements[sym_name] = replacement

  def lookup_replacement(self, module_name, symbol_name, replacement):
    sym_name = module_name + ":" + symbol_name
    return self.symbol_replacements.get(sym_name, None)

  def whitelist_module(self, module_name):
    self.module_whitelist.add(module_name)

  def is_module_whitelisted(self, module_name):
    return module_name in self.module_whitelist

  def whitelist_function(self, module_name, symbol_name):
    sym_name = module_name + ":" + symbol_name
    self.function_whitelist.add(sym_name)

  def is_function_whitelisted(self, module_name, symbol_name):
    sym_name = module_name + ":" + symbol_name
    return sym_name in self.function_whitelist

  @property
  def g(self):
    return self.sess.graph

  @property
  def python_op_whitelist(self):
    """Python-only ops whitelisted for wrapping."""
    return {"tf.pow", "tf.reduce_sum", "tf.range", "tf.random_uniform", "tf.ones",
            "tf.concat", "tf.split"}

  @property
  def gen_op_blacklist(self):
    """Ops blacklisted for wrapping, because they are used by the immediate
    execution environment itself (like placeholder for feeding tensorhandle)"""
    return {"_placeholder", "placeholder"}

  @property
  def graph_version(self):
    """Gives version of the graph. This can be used for checking if graph
    modifications took place"""
    return self.g.version

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

    handle = self.numpy_to_handle(array)
    return Tensor(self, handle)

  def constant(self, values, dtype=None, shape=None, name='Const'):
    np_dtype = None

    if dtype:
      try:
        dtype = dtypes.as_dtype(dtype)
        np_dtype = dtype.as_numpy_dtype
      except TypeError as e:
        raise TypeError("Trying to create constant with dtype=%s, "
                        "got TypeError(%s)" % (dtype, e.message))

    if shape:
      assert (isinstance(values, types.FloatType) or
              isinstance(values, types.IntType) or
              isinstance(values, types.LongType))
      return self.numpy_to_tensor(values*np.ones(shape=shape, dtype=np_dtype),
                                  dtype=dtype)
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

  # This method is called from wrapping manager when deciding how to wrap
  # each function.
  def wrap_function(self, symbol, new_globals):
    # a function can be gen_.*_ops derived function in which case
    # we use OpWrapper
    # it can be a "special" function like convert_to_tensor for which we
    # substitute our manually crafted implementation
    # or it can be some other function, in which case we make a copy of it
    # TODO(yaroslavvb): get rid of "derived_symbol_name" in OpWrapper

    # TODO(yaroslavvb): precompile regex for efficiency
    # TODO(yaroslavvb): decide if we need to substitute globals for
    # gen_.*_ops functions, document this choice
    module_name = get_canonical_name(inspect.getsourcefile(symbol))
    basename = os.path.basename(module_name)
    if re.match("^gen.*_ops.py$", basename):
      return OpWrapper(self, self.env, "", symbol)
    
    manual_replacement = self.lookup_replacement(module_name, symbol.__name__)
    if manual_replacement:
      return manual_replacement

    else: # neither genereated op, nor manually written replacement, return
          # a copy with new globals dictionary
      new_symbol = self.copy_python_function(symbol, new_globals)



  def copy_python_function(f, new_globals):
    """Utility function to create a copy of Python function."""

    g = types.FunctionType(f.__code__, new_globals, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g.__dict__.update(f.__dict__)
    return g



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
    # TODO(yaroslavvb): rename python_ops to wrapped_symbols
    self.python_ops = {}  # custom Python op functions
    self.other = {}  # everything else
    self.nested_modules = {}  # nested modules like tf.nn go here

    for (name,symbol) in namespace.__dict__.items():

      # only include functions
      if (type(symbol) == types.FunctionType and 
          not name in self.env.gen_op_blacklist):
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

    # convert to Tensor is used as op.convert_to_tensor, override it with version that
    # can returns immediate.Tensors
    if symbol_name == 'convert_to_tensor':
      return ConvertToTensorWrapper(self, self.env, derived_symbol_name)

    # Concat infers numeric attribute for number of tensors, since we don't
    # have attribute inference, override it with custom version
    # for
    # gen_array_ops._concat(concat_dim=concat_dim, values=values, name=name)
    #    if derived_symbol_name == "gen_array_ops._concat":
    #      return ConcatOpWrapper(self, self.env, derived_symbol_name)

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

