from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["Op", "OpFactory", "OpWrapper", "PythonOpWrapper"]

from .tensor import Tensor

import sys

# Implementation of Immediate Op
class Op(object):

  def __init__(self, env, input_holders, output_handle, key,
               converted_tensors={}):
    """Initialize Op.

    Args:
      converted_tensors: dictionary of argument position -> converted immedate
        Tensor (only for argument positions where this conversion was made)
    """

    self.env = env
    self.input_holders = input_holders
    self.output_handle = output_handle
    self.key = key
    self.converted_tensors = converted_tensors

  def __call__(self, *args):
    if not len(args) == len(self.input_holders):
      raise ValueError("Too many arguments provided (%d), %s can only accept "
                       "%d" % (len(args), self.__str__(),
                               len(self.input_holders)))

    feed_dict = {}
    for (i, (itensor, holder)) in enumerate(zip(args, self.input_holders)):
      if i in self.converted_tensors:
        itensor = self.converted_tensors[i]
      if not isinstance(itensor, Tensor):
        raise ValueError("All positional arguments must be immediate "
                         "Tensors, instead we see "+str(itensor))
      feed_dict[holder] = itensor.tf_handle

    tensor_handle = self.env.run(self.output_handle, feed_dict=feed_dict)
    return Tensor(self.env, tensor_handle)

  def __str__(self):
    return "Op%s" % (str(self.key))

  def __repr__(self):
    return self.__str__()


def _fixup_args(symbol_name, *args, **kwargs):
  #  print("calling fixup_args for %s, %s, %s" % (symbol_name, args, kwargs))
  
  if symbol_name == 'gen_math_ops._sum':
    # handle gen_math_ops._sum(tensor,tensor,keep_dims)
    if len(args)==3:
      kwargs['keep_dims'] = args[2]
      return args[:-1], kwargs

  # handle gen_random_ops._random_uniform(shape, dtype)
  if symbol_name == 'gen_random_ops._random_uniform':
    if len(args)==2:
      kwargs['dtype'] = args[1]
      return args[:-1], kwargs
    
  return args, kwargs

# Implementating of OpFactory with graph caching
class OpFactory(object):
  def __init__(self, env):
    self.env = env
    self.cache = {}

  def __call__(self, symbol_name, symbol, *args, **kwargs):
    
    # create the key to see if the op has been created before
    key = [symbol_name]

    # our heuristic that Tensor arguments are positional args and attributes
    # are keyword args is failing in some popular cases, like reduce_sum
    # which calls "sum(tensor1, tensor2, keep_dims)"
    # This can be handled by following OpDef parsing logic in op_def_library.py
    # to figure out which args are attributes. Until them, have a procedure
    # that rearranges args to conform to convention for several important
    # cases.
    args,kwargs = _fixup_args(symbol_name, *args, **kwargs)

    # converted_args stores args converted to Tensors, ie, Python list [1]
    # becomes immediate.Tensor([1])), immediate.Tensor objects are unchanged
    converted_args = []  
    converted_tensors = {}
    for i,itensor in enumerate(args):
      if not isinstance(itensor, Tensor):
        try:
          itensor = self.env.numpy_to_tensor(itensor)
          converted_tensors[i] = itensor
        except ValueError as e:
          raise ValueError("All positional arguments must be immediate "
                           "Tensors, or convertible to immediate Tensors "
                           "instead we see %s (numpy error: %s)" % (
                            str(itensor), sys.exc_info()[0]))
      converted_args.append(itensor)
      key.append(itensor.dtype)

    # TODO(yaroslavvb): use signature binding to fill out default kwargs
    # otherwise may get cache miss
    for kwarg_key in sorted(kwargs.keys()):
      key.append(kwarg_key)
      key.append(str(kwargs[kwarg_key]))
    
    # convert to tuple to make it hashable
    key = tuple(key)

    if key in self.cache:
      return self.cache[key]
    
    # create the op
    with self.env.g.as_default():
      # convert args to TensorHandles
      # connect up things
      input_tensors = []
      input_holders = []
      for itensor in converted_args:
        holder, tensor = self.env.get_session_tensor(itensor.dtype)
        input_holders.append(holder)
        input_tensors.append(tensor)

      # extra check, make sure the user didn't use TF-style
      # kwargs approach to specify inputs
      for name,val in kwargs.items():
        if isinstance(val, Tensor):
          raise ValueError("Found Tensor in a keyword argument, use "
                           "positional arguments instead.")

      output = symbol(*input_tensors, **kwargs)

      # TODO(yaroslavvb): allow for multiple return values like tf.split
      if isinstance(output, list):
        raise ValueError("Only support TF ops that return a single Tensor.")
      
      # Convert result to TensorHandle
      output_handle = self.env.get_session_handle(output)

    op = Op(self.env, input_holders, output_handle, key, converted_tensors)
    self.cache[key] = op

    return op


  def _create_key(self, opname, *args, **kwargs):
    return opname

class OpWrapper(object):
  """A callable object that mirrors TF generated wrapper, but with immediate
  execution semantics."""

  def __init__(self, namespace, env, symbol_name, symbol):
    self.namespace = namespace
    self.env = env
    self.symbol_name = symbol_name  # name of function, ie "tf.nn.relu"
    self.symbol = symbol  # function object
  
  def __call__(self, *args, **kwargs):
    op = self.env.op_factory(self.symbol_name, self.symbol, *args, **kwargs)
    args, kwargs = _fixup_args(self.symbol_name, *args, **kwargs)
    return op(*args)

class PythonOpWrapper(object):
  """A callable object that mirrors Python tensorflow function."""

  def __init__(self, namespace, env, symbol_name, symbol, global_sub):
    self.namespace = namespace
    self.env = env
    self.symbol_name = symbol_name
    self.symbol = symbol
    self.global_sub = global_sub
    
    for global_name in global_sub:
      symbol.__globals__[global_name] = global_sub[global_name]

  def __call__(self, *args, **kwargs):
    return self.symbol(*args, **kwargs)

class ConstantOpWrapper(object):
  """A callable object that mirrors tf.constant."""

  def __init__(self, namespace, env, symbol_name):
    self.namespace = namespace
    self.env = env
    self.symbol_name = symbol_name
    

  def __call__(self, *args, **kwargs):
    return self.env.constant(*args, **kwargs)

class ConvertToTensorWrapper(object):
  """A callable object that mirrors tf.convert_to_tensor"""

  def __init__(self, namespace, env, symbol_name):
    self.namespace = namespace
    self.env = env
    self.symbol_name = symbol_name
    

  def __call__(self, value, dtype=None, name=None, as_ref=False):
    return self.env.numpy_to_tensor(value, dtype)
