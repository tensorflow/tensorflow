"""Operations for implementing attention.
"""
import tensorflow.python.platform

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_attention_ops
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_attention_ops import *


# TODO(bsteiner): Implement the gradient function for extract_glimpse
ops.NoGradient("ExtractGlimpse")


@ops.RegisterShape("ExtractGlimpse")
def _ExtractGlimpseShape(op):
  """Shape function for ExtractGlimpse op."""
  input_shape = op.inputs[0].get_shape().with_rank(4)
  unused_size_shape = op.inputs[1].get_shape().merge_with(
      tensor_shape.vector(2))
  offsets_shape = op.inputs[2].get_shape().merge_with(
      input_shape[:1].concatenate([2]))
  offsets_shape = offsets_shape
  size_value = tensor_util.ConstantValue(op.inputs[1])
  if size_value is not None:
    height = size_value[0]
    width = size_value[1]
  else:
    height = None
    width = None
  return [tensor_shape.TensorShape(
      [input_shape[0], height, width, input_shape[3]])]
