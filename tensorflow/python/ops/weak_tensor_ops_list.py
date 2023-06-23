# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Lists of ops that support WeakTensor."""

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import image_ops_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import special_math_ops


# Below are lists of unary ops that return a WeakTensor when given a WeakTensor
# input. These are some of the reasons why ops may not support WeakTensor.
# (1) The return dtype is specified. (e.g. tofloat(), cast(), is_finite())
# (2) The list is prioritized to unary elementwise ops, TF-NumPy ops, math_ops,
#     and array_ops.
# (3) There is no "weak" string type so any string ops are not supported.
# If you wish to add support to a specific unary op, add the unary op to a
# corresponding list.

_ELEMENTWISE_UNARY_OPS = [
    math_ops.abs,
    math_ops.softplus,
    math_ops.sign,
    math_ops.real,
    math_ops.imag,
    math_ops.angle,
    math_ops.round,
    math_ops.sigmoid,
    math_ops.log_sigmoid,
    math_ops.conj,
    math_ops.reciprocal_no_nan,
    math_ops.erfinv,
    math_ops.ndtri,
    math_ops.erfcinv,
    math_ops.ceil,
    math_ops.sqrt,
    math_ops.exp,
    math_ops.rsqrt,
    math_ops.acos,
    math_ops.floor,
    gen_bitwise_ops.invert,
    gen_math_ops.acosh,
    gen_math_ops.asin,
    gen_math_ops.asinh,
    gen_math_ops.atan,
    gen_math_ops.atanh,
    gen_math_ops.cos,
    gen_math_ops.cosh,
    gen_math_ops.digamma,
    gen_math_ops.erf,
    gen_math_ops.erfc,
    gen_math_ops.expm1,
    gen_math_ops.lgamma,
    gen_math_ops.log,
    gen_math_ops.log1p,
    gen_math_ops.neg,
    gen_math_ops.reciprocal,
    gen_math_ops.rint,
    gen_math_ops.sin,
    gen_math_ops.sinh,
    gen_math_ops.square,
    gen_math_ops.tan,
    gen_math_ops.tanh,
    array_ops.zeros_like,
    array_ops.ones_like,
    gen_array_ops.check_numerics,
    nn_ops.relu6,
    nn_ops.leaky_relu,
    nn_ops.gelu,
    nn_ops.log_softmax,
    gen_nn_ops.elu,
    gen_nn_ops.relu,
    gen_nn_ops.selu,
    gen_nn_ops.softsign,
    image_ops_impl.random_brightness,
    image_ops_impl.stateless_random_brightness,
    image_ops_impl.adjust_brightness,
    image_ops_impl.adjust_gamma,
    nn_impl.swish,
    clip_ops.clip_by_value,
    special_math_ops.dawsn,
    special_math_ops.expint,
    special_math_ops.fresnel_cos,
    special_math_ops.fresnel_sin,
    special_math_ops.spence,
    special_math_ops.bessel_i0,
    special_math_ops.bessel_i0e,
    special_math_ops.bessel_i1,
    special_math_ops.bessel_i1e,
    special_math_ops.bessel_k0,
    special_math_ops.bessel_k0e,
    special_math_ops.bessel_k1,
    special_math_ops.bessel_k1e,
    special_math_ops.bessel_j0,
    special_math_ops.bessel_j1,
    special_math_ops.bessel_y0,
    special_math_ops.bessel_y1,
]
_TF_OPS = []
_TF_NUMPY_OPS = []
ALL_UNARY_OPS = _ELEMENTWISE_UNARY_OPS + _TF_OPS + _TF_NUMPY_OPS
