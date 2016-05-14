from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["Op", "OpFactory", "OpWrapper"]

from .tensor import Tensor

# Implementation of Immediate Op
class Op(object):

  def __init__(self, env, input_holders, output_handle, key):
    self.env = env
    self.input_holders = input_holders
    self.output_handle = output_handle
    self.key = key

  def __call__(self, *args):
    if not len(args) == len(self.input_holders):
      raise ValueError("Too many arguments provided (%d), %s can only accept "
                       "%d" % (len(args), self.__str__(),
                               len(self.input_holders)))

    feed_dict = {}
    for (itensor, holder) in zip(args, self.input_holders):
      if not isinstance(itensor, Tensor):
        raise ValueError("All positional arguments of %s must be immediate "
                           "Tensors" % (self.__str__()))
      feed_dict[holder] = itensor.tf_handle

    tensor_handle = self.env.run(self.output_handle, feed_dict=feed_dict)
    return Tensor(self.env, tensor_handle)

  def __str__(self):
    return "Op%s" % (str(self.key))

  def __repr__(self):
    return self.__str__()

# Implementating of OpFactory with graph caching
class OpFactory(object):
  def __init__(self, env):
    self.env = env
    self.cache = {}

  def __call__(self, symbol_name, symbol, *args, **kwargs):
    
    # create the key to see if the op has been created before
    key = [symbol_name]
    for itensor in args:
      if not isinstance(itensor, Tensor):
        raise ValueError("All positional arguments must be immediate "
                         "Tensors")
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
      for itensor in args:
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

    op = Op(self.env, input_holders, output_handle, key)
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
    return op(*args)
