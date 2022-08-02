# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""It lists ops of RaggedTensor for the interest of test."""

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import string_ops


# Constants listing various op types to test.  Each operation
# should be included in at least one list below, or tested separately if
# necessary (e.g., because it expects additional arguments).
UNARY_FLOAT_OPS = [
    math_ops.abs,
    math_ops.acos,
    math_ops.acosh,
    math_ops.angle,
    math_ops.asin,
    math_ops.asinh,
    math_ops.atan,
    math_ops.atanh,
    math_ops.ceil,
    math_ops.conj,
    math_ops.cos,
    math_ops.cosh,
    math_ops.digamma,
    math_ops.erf,
    math_ops.erfc,
    math_ops.erfcinv,
    math_ops.erfinv,
    math_ops.exp,
    math_ops.expm1,
    math_ops.floor,
    math_ops.imag,
    math_ops.is_finite,
    math_ops.is_inf,
    math_ops.is_nan,
    math_ops.lgamma,
    math_ops.log,
    math_ops.log1p,
    math_ops.log_sigmoid,
    math_ops.ndtri,
    math_ops.negative,
    math_ops.real,
    math_ops.reciprocal,
    math_ops.reciprocal_no_nan,
    math_ops.rint,
    math_ops.round,
    math_ops.rsqrt,
    math_ops.sign,
    math_ops.sigmoid,
    math_ops.sin,
    math_ops.sinh,
    math_ops.softplus,
    math_ops.sqrt,
    math_ops.square,
    math_ops.tan,
    math_ops.tanh,
    nn_ops.elu,
    nn_ops.gelu,
    nn_ops.leaky_relu,
    nn_ops.log_softmax,
    nn_ops.relu,
    nn_ops.relu6,
    nn_ops.selu,
    nn_ops.softsign,
    nn_impl.swish,
    array_ops.ones_like,
    array_ops.ones_like_v2,
    array_ops.zeros_like,
    array_ops.zeros_like_v2,
    special_math_ops.bessel_i0,
    special_math_ops.bessel_i0e,
    special_math_ops.bessel_i1,
    special_math_ops.bessel_j0,
    special_math_ops.bessel_j1,
    special_math_ops.bessel_i1e,
    special_math_ops.bessel_k0,
    special_math_ops.bessel_k0e,
    special_math_ops.bessel_k1,
    special_math_ops.bessel_k1e,
    special_math_ops.bessel_y0,
    special_math_ops.bessel_y1,
    special_math_ops.dawsn,
    special_math_ops.expint,
    special_math_ops.fresnel_cos,
    special_math_ops.fresnel_sin,
    special_math_ops.spence,
    string_ops.as_string,
]
UNARY_BOOL_OPS = [
    math_ops.logical_not,
]
UNARY_STRING_OPS = [
    string_ops.decode_base64,
    string_ops.encode_base64,
    string_ops.string_strip,
    string_ops.string_lower,
    string_ops.string_upper,
    string_ops.string_length,
    string_ops.string_length_v2,
    parsing_ops.decode_compressed,
]
BINARY_FLOAT_OPS = [
    math_ops.add,
    math_ops.atan2,
    math_ops.complex,
    math_ops.div,
    math_ops.div_no_nan,
    math_ops.divide,
    math_ops.equal,
    math_ops.floor_div,
    math_ops.floordiv,
    math_ops.floormod,
    math_ops.greater,
    math_ops.greater_equal,
    math_ops.less,
    math_ops.less_equal,
    math_ops.maximum,
    math_ops.minimum,
    math_ops.multiply,
    math_ops.multiply_no_nan,
    math_ops.not_equal,
    math_ops.pow,
    math_ops.realdiv,
    math_ops.squared_difference,
    math_ops.subtract,
    math_ops.truediv,
    math_ops.xdivy,
    math_ops.xlog1py,
    math_ops.xlogy,
    math_ops.zeta,
]
BINARY_BOOL_OPS = [
    math_ops.logical_and,
    math_ops.logical_or,
    math_ops.logical_xor,
]
UNARY_INT_OPS = [
    gen_bitwise_ops.invert,
    string_ops.unicode_script,
]
BINARY_INT_OPS = [
    gen_bitwise_ops.bitwise_and,
    gen_bitwise_ops.bitwise_or,
    gen_bitwise_ops.bitwise_xor,
    gen_bitwise_ops.left_shift,
    gen_bitwise_ops.right_shift,
    math_ops.truncatediv,
    math_ops.truncatemod,
]
BINARY_ASSERT_OPS = [
    check_ops.assert_equal,
    check_ops.assert_equal_v2,
    check_ops.assert_near,
    check_ops.assert_near_v2,
    check_ops.assert_none_equal,
    check_ops.assert_none_equal_v2,
    check_ops.assert_greater,
    check_ops.assert_greater_v2,
    check_ops.assert_greater_equal,
    check_ops.assert_greater_equal_v2,
    check_ops.assert_less,
    check_ops.assert_less_v2,
    check_ops.assert_less_equal,
    check_ops.assert_less_equal_v2,
]
