import sys
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
# pylint: disable=wildcard-import
# 'Constant' gets imported in the module 'array_ops'.
from tensorflow.python.ops import gen_sampling_ops


@ops.RegisterShape("BernoulliSample")
def _BernoulliSampleShape(op):
  a_shape = op.inputs[1].get_shape().with_rank(1)
  b_shape = op.inputs[2].get_shape().with_rank(1)

  assert a_shape == b_shape

  return [a_shape]

ops.NoGradient("BernoulliSample")

@ops.RegisterShape("SampleDistributionIndex")
def _SampleDistributionIndexShape(op):
  batch_size = op.inputs[0].get_shape().with_rank(2)[0].value
  return [tensor_shape.TensorShape([batch_size])]

ops.NoGradient("SampleDistributionIndex")
