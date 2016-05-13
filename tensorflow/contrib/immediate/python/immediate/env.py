# Implementation of Immediate Env
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["Env"]

#from tensorflow.python.client.session import Session
#from tensorflow.python.framework.ops import Graph


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

def _create_namespace_map():
  namespace_map = {}
  import tensorflow as tf
  for (name,symbol) in tf.__dict__.items():
    # only include functions
    if not type(symbol) == types.FunctionType:
      continue

    basename = os.path.basename(inspect.getsourcefile(symbol))
    # only include native TensorFlow ops
    if not re.match("^gen.*.ops.py$", basename):
      continue

    namespace_map[name] = symbol

    # needed for type-checking in OpFactory
    namespace_map['Tensor'] = tf.Tensor

    # TODO(yaroslavvb): loop over "tf.nn" namespace here as well

  return namespace_map


class Env(object):
  
  def __init__(self):
    self.sess = session.Session()
    self.op_factory = OpFactory(self)
    #    self.run_options = tf.RunOptions()

    # <generaldep> functionality below needs to be called from
    # outside of this package if we want Env to be part of TensorFlow
    self.namespace_map = _create_namespace_map()


  def __getattr__(self, tf_op_name):
    if tf_op_name in self.namespace_map:
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

    # TODO(yaroslavvb): use session settings
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

  def get_session_tensor(self, dtype):
    holder, tensor = session_ops.get_session_tensor(dtype)
    return holder, tensor

  def get_session_handle(self, tf_tensor):
    handle_op = session_ops.get_session_handle(tf_tensor)
    return handle_op

  
  def run(self, *args, **kwargs):
    return self.sess.run(*args, **kwargs)
