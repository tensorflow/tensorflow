# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Labels for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.labeled_tensor.python.ops import core as _core
from tensorflow.contrib.labeled_tensor.python.ops import io_ops as _io_ops
from tensorflow.contrib.labeled_tensor.python.ops import nn
from tensorflow.contrib.labeled_tensor.python.ops import ops as _ops
from tensorflow.contrib.labeled_tensor.python.ops import sugar as _sugar

# pylint: disable=invalid-name

# Core types.
Axis = _core.Axis
Axes = _core.Axes
LabeledTensor = _core.LabeledTensor

as_axis = _core.as_axis
convert_to_labeled_tensor = _core.convert_to_labeled_tensor

identity = _core.identity
slice = _core.slice_function  # pylint: disable=redefined-builtin
transpose = _core.transpose
expand_dims = _core.expand_dims
align = _core.align

axis_order_scope = _core.axis_order_scope
check_axis_order = _core.check_axis_order
impose_axis_order = _core.impose_axis_order
AxisOrderError = _core.AxisOrderError

define_unary_op = _core.define_unary_op
define_binary_op = _core.define_binary_op
define_reduce_op = _ops.define_reduce_op

abs = _core.abs_function  # pylint: disable=redefined-builtin
neg = _core.neg
sign = _core.sign
reciprocal = _core.reciprocal
square = _core.square
round = _core.round_function  # pylint: disable=redefined-builtin
sqrt = _core.sqrt
rsqrt = _core.rsqrt
exp = _core.exp
log = _core.log
ceil = _core.ceil
floor = _core.floor
cos = _core.cos
sin = _core.sin
tan = _core.tan
acos = _core.acos
asin = _core.asin
atan = _core.atan
lgamma = _core.lgamma
digamma = _core.digamma
erf = _core.erf
erfc = _core.erfc
logical_not = _core.logical_not
tanh = _core.tanh
sigmoid = _core.sigmoid

add = _core.add
sub = _core.sub
mul = _core.mul
div = _core.div
mod = _core.mod
pow = _core.pow_function  # pylint: disable=redefined-builtin

equal = _core.equal
greater = _core.greater
greater_equal = _core.greater_equal
not_equal = _core.not_equal
less = _core.less
less_equal = _core.less_equal
logical_and = _core.logical_and
logical_or = _core.logical_or
logical_xor = _core.logical_xor

maximum = _core.maximum
minimum = _core.minimum
squared_difference = _core.squared_difference
igamma = _core.igamma
igammac = _core.igammac
zeta = _core.zeta
polygamma = _core.polygamma

select = _ops.select
concat = _ops.concat
pack = _ops.pack
unpack = _ops.unpack
reshape = _ops.reshape
rename_axis = _ops.rename_axis
random_crop = _ops.random_crop
map_fn = _ops.map_fn
foldl = _ops.foldl
squeeze = _ops.squeeze
matmul = _ops.matmul
tile = _ops.tile
pad = _ops.pad
constant = _ops.constant
zeros_like = _ops.zeros_like
ones_like = _ops.ones_like
cast = _ops.cast
verify_tensor_all_finite = _ops.verify_tensor_all_finite
boolean_mask = _ops.boolean_mask
where = _ops.where

reduce_all = _ops.reduce_all
reduce_any = _ops.reduce_any
reduce_logsumexp = _ops.reduce_logsumexp
reduce_max = _ops.reduce_max
reduce_mean = _ops.reduce_mean
reduce_min = _ops.reduce_min
reduce_prod = _ops.reduce_prod
reduce_sum = _ops.reduce_sum

batch = _ops.batch
shuffle_batch = _ops.shuffle_batch

FixedLenFeature = _io_ops.FixedLenFeature
parse_example = _io_ops.parse_example
parse_single_example = _io_ops.parse_single_example
placeholder = _io_ops.placeholder

ReshapeCoder = _sugar.ReshapeCoder
