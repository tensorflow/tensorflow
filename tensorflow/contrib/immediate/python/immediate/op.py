from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from .tensor import Tensor
from .tensor import _ENABLE_DEBUG_LOGGING

from tensorflow.python.framework import ops as tf_ops

# Implementation of Immediate Op with keyword call arguments
class Op(object):

  """Op represents an object which accepts immediate tensors and returns
  immediate tensors. It turns incoming tensors into TensorHandle objects, runs
  underlying op in env's session and wraps resulting TensorHandles in immediate
  Tensors."""

  def __init__(self, env, input_holders, output_handle, name="op"):
    """Initialize Op.

    Args:
      converted_tensors: dictionary of argument position -> converted immedate
        Tensor (only for argument positions where this conversion was made)
    """

    self.env = env   # used to issue .run calls
    self.input_holders = input_holders
    self.output_handle = output_handle   # tensorflow.TensorHandle
    self.name = name

  def __call__(self, **kwargs):

    feed_dict = {}
    for (argname, itensor) in kwargs.items():
      if isinstance(itensor, list):
        holder_list = self.input_holders[argname]
        tensor_list = itensor
        for holder, subtensor in zip(holder_list, tensor_list):
          feed_dict[holder] = subtensor.tf_handle
      else:
        feed_dict[self.input_holders[argname]] = itensor.tf_handle

    tensor_handle = self.env.run(self.output_handle, feed_dict=feed_dict)
    if isinstance(tensor_handle, list):
      return [Tensor(self.env, t) for t in tensor_handle]
    else:
      return Tensor(self.env, tensor_handle)

  def __str__(self):
    return "Op%s" % (str(self.name))

  def __repr__(self):
    return self.__str__()


class OpDefLibraryWrapper(object):
  """Wrapper class that replaces OpDefLibrary instances in all gen.*ops
  modules."""

  def __init__(self, env, original_op_def_library):
    self.env = env
    self.original_op_def_library = original_op_def_library

  def get_op_info(self, name):
    """Get input argnames/types from op_def_lib object."""

    op = self.original_op_def_library._ops[name].op_def
    argnames0 = [arg.name for arg in op.input_arg]
    argtypes0 = {}
    for arg in op.input_arg:
      if arg.number_attr or arg.type_list_attr:
        argtypes0[arg.name] = "list"
      else:
        argtypes0[arg.name] = "single"

    return argnames0, argtypes0

  def apply_op(self, op_type_name, name=None, **keywords):
    """
    stuff
    Retrieves op from the cache.
    op = env.get_op(op_type_name, keywords)
    return op(keywords)

    get_op(op_type_name, keywords):

    key = self.get_key(op_type_name, keywords)
    if key in cache:
      return cache[key]
    else:
      op_def = self.get_op_def(op_type_name, keywords)
      ...
    """

    # converted_args stores args converted to Tensors, ie, Python list [1]
    # becomes immediate.Tensor([1])), immediate.Tensor objects are unchanged
    itensor_args = {} 
    converted_tensors = {}
    #    input_names = op_input_argnames[op_type_name]
    #    input_types = op_input_argtypes[op_type_name]

    input_names, input_types = self.get_op_info(op_type_name)

    if _ENABLE_DEBUG_LOGGING:
      print("OpFactory __call__: %s(%s)" % (op_type_name, keywords))
      print("OpFactory inputs: %s" % (input_names))

    key = [op_type_name]

    # NOTE(yaroslavvb): this doesn't do attribute inference so for Python
    # types it can potentially create different dtype than native TF
    def try_convert_to_itensor(itensor, dtype=None):
      if isinstance(itensor, Tensor):
        return itensor

      if isinstance(itensor, tf_ops.Tensor):
        raise ValueError("Trying to feed a non-immediate Tensor %s to "
                         "immediate op %s" %
                         (itensor, op_type_name))
      try:
        result = self.env.numpy_to_tensor(itensor, dtype)
        if _ENABLE_DEBUG_LOGGING:
          print("Converting %s to %s, result is %s" %(itensor, dtype,
                                                      result.dtype))
        return result

      except ValueError:
        raise ValueError("Couldn't convert input argument %s=%s to immediate "
                         "tensor (%s)" % (input_name, itensor,
                                          sys.exc_info()))

    # TODO(yaroslavvb): replace with more generic logic to support other
    # ops that accept lists of arguments
    list_dtype = None
    if op_type_name == "Concat":
      for maybe_itensor in keywords["values"]:
        if isinstance(maybe_itensor, Tensor):
          list_dtype = maybe_itensor.dtype
          break

    for input_name in input_names:
      itensor = keywords[input_name]
      if input_types[input_name] == "list":
        for i in range(len(itensor)):
          if op_type_name == "Concat":
            itensor[i] = try_convert_to_itensor(itensor[i], list_dtype)
          else:
            itensor[i] = try_convert_to_itensor(itensor[i])
      else:
        itensor = try_convert_to_itensor(itensor)

      itensor_args[input_name] = itensor

    with self.env.g.as_default():
      input_holders = {}
      for input_name in input_names:
        if isinstance(itensor_args[input_name], list):
          holder_list = []
          tensor_list = []
          for subtensor in itensor_args[input_name]:
            holder, tensor = self.env.get_session_tensor(subtensor.dtype)
            holder_list.append(holder)
            tensor_list.append(tensor)
          keywords[input_name] = tensor_list
          input_holders[input_name] = holder_list
        else:
          input_dtype = itensor_args[input_name].dtype
          holder, tensor = self.env.get_session_tensor(input_dtype)
          input_holders[input_name] = holder
          keywords[input_name] = tensor

      output = self.original_op_def_library.apply_op(op_type_name,
                                                     **keywords)

      if isinstance(output, list) or isinstance(output, tuple):
        output_handle = [self.env.get_session_handle(o) for o in output]
      elif isinstance(output, tf_ops.Tensor):
        output_handle = self.env.get_session_handle(output)
      else:
        raise ValueError("Op %s gave output (%s) of unexpected type (%s)"
                         % (op_type_name, output, type(output)))

    op = Op(self.env, input_holders, output_handle)
    return op(**itensor_args)
    #    self.cache[key] = op


class ConstantOpWrapper(object):
  """A callable object that mirrors tf.constant."""

  def __init__(self, env, old_symbol):
    self.env = env
    self.old_symbol = old_symbol

  def __call__(self, *args, **kwargs):
    return self.env.constant(*args, **kwargs)

class ConvertToTensorWrapper(object):
  """A callable object that mirrors tf.convert_to_tensor in Immediate
  environment."""

#  def __init__(self, namespace, env, symbol_name):
  def __init__(self, env, old_symbol):
    self.env = env
    self.old_symbol = old_symbol

  def __call__(self, value, dtype=None, name=None, as_ref=False):
    if isinstance(value, Tensor):
      return value
    return self.env.numpy_to_tensor(value, dtype)


