# Implementation of Tensor for the immediate API

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["Tensor"]

_ENABLE_DEBUG_LOGGING = False

#import tensorflow as tf

from tensorflow.python.framework import tensor_shape
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import constant_op

class Tensor(object):

  def __init__(self, env, handle):
    self.env = env
    self.handle = handle  # TensorHandle object

  @property
  def tf_handle(self):
    """Give string handle representing this tensor in TF runtime.
    This string handle is suitable for feeding get_session_tensor op."""
    return self.handle.handle

  @property
  def dtype(self):
    return self.handle._dtype

  def as_numpy(self):
    """Convert current Tensor into numpy array."""

    return self.env.handle_to_numpy(self.handle)



  # for compatibility with array_ops transpose
  # that has following temporary hack for shape inference
  # input_shape = ret.op.inputs[0].get_shape().dims
  # since we don't care about shape inference in immediate mode, return dummy
  # node with no shape
  @property
  def op(self):
    node_def = graph_pb2.NodeDef()
    node_def.name = "immediate-dummy-node"
    node_def.input.extend(["dummy1", "dummy2", "dummy3"])

    dummy_input1 = constant_op.constant(1)
    dummy_input2 = constant_op.constant(2)
    dummy_input3 = constant_op.constant(3)
    dummy_op = tf_ops.Operation(node_def, tf_ops.Graph(), inputs=[dummy_input1,
                                                                  dummy_input2,
                                                                  dummy_input3])
    return dummy_op

  @property
  def name(self):
    return "ITensor"
   

  # tf.Tensor compatibility
  def eval(self):
    return self.as_numpy()

  @property
  def shape(self):
    return self.get_shape()

  def get_shape(self):
    shape_tensor = self.env.tf.shape(self)
    shape_tuple = tuple(shape_tensor.as_numpy())
    return tensor_shape.TensorShape(shape_tuple)

  def set_shape(self, _unused_shape):
    """Immediate tensors don't have static shape, but keep this method
    for compatibility with array_ops.py"""
    pass

  def __str__(self):
    return str(self.as_numpy())

  def __repr__(self):
    return "iTensor(%s, dtype=%s)" % (self.__str__(), self.dtype)


  # Methods to emulate Python numeric type
  # https://docs.python.org/2/reference/datamodel.html#emulating-numeric-types


  def __add__(self, other):
    # TODO(yaroslavvb): complain if downcasting
    if not isinstance(other, Tensor):
      other = self.env.numpy_to_tensor(other, dtype=self.dtype)
    return self.env.tf.add(self, other)

  def __radd__(self, other):
    if not isinstance(other, Tensor):
      other = self.env.numpy_to_tensor(other, dtype=self.dtype)
    return self.env.tf.add(other, self)

  def __neg__(self):
    return self.env.tf.neg(self)


  def __sub__(self, other):
    if not isinstance(other, Tensor):
      other = self.env.numpy_to_tensor(other, dtype=self.dtype)
    return self.env.tf.sub(self, other)

  def __rsub__(self, other):
    if not isinstance(other, Tensor):
      other = self.env.numpy_to_tensor(other, dtype=self.dtype)
    return self.env.tf.sub(other, self)

  def __mul__(self, other):
    if not isinstance(other, Tensor):
      other = self.env.numpy_to_tensor(other, dtype=self.dtype)
    return self.env.tf.mul(self, other)

  def __rmul__(self, other):
    if not isinstance(other, Tensor):
      other = self.env.numpy_to_tensor(other, dtype=self.dtype)
    return self.env.tf.mul(other, self)

  def __bool__(self):
    # TODO(yaroslavvb): add in cast after Python-only ops are supported
    #    bool_tensor = self.env.cast(self, dtype=tf.bool)
    return bool(self.as_numpy())

  def __nonzero__(self):
    return self.__bool__()

  def __lt__(self, other):
    return self.env.tf.less(self, other)
  
  def __le__(self, other):
    return self.env.tf.less_equal(self, other)

  def __eq__(self, other):
    return self.env.tf.equal(self, other)

  def __ne__(self, other):
    return self.env.tf.not_equal(self, other)

  def __gt__(self, other):
    return self.env.tf.greater(self, other)

  def __ge__(self, other):
    return self.env.tf.greater_equal(self, other)
