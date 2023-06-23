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
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_math_ops


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
    array_ops.zeros_like_v2,
    array_ops.ones_like,
    array_ops.ones_like_v2,
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
_TF_UNARY_OPS = [
    math_ops.reduce_euclidean_norm,
    math_ops.reduce_logsumexp,
    math_ops.reduce_max,
    math_ops.reduce_max_v1,
    math_ops.reduce_mean,
    math_ops.reduce_mean_v1,
    math_ops.reduce_min,
    math_ops.reduce_min_v1,
    math_ops.reduce_prod,
    math_ops.reduce_prod_v1,
    math_ops.reduce_std,
    math_ops.reduce_sum,
    math_ops.reduce_sum_v1,
    math_ops.reduce_variance,
    math_ops.trace,
    array_ops.depth_to_space,
    array_ops.depth_to_space_v2,
    array_ops.expand_dims,
    array_ops.expand_dims_v2,
    array_ops.extract_image_patches,
    array_ops.extract_image_patches_v2,
    array_ops.identity,
    array_ops.matrix_diag,
    array_ops.matrix_diag_part,
    array_ops.matrix_transpose,
    array_ops.shape,
    array_ops.shape_v2,
    array_ops.size,
    array_ops.size_v2,
    array_ops.space_to_depth,
    array_ops.space_to_depth_v2,
    array_ops.squeeze,
    array_ops.squeeze_v2,
    array_ops.stop_gradient,
    array_ops.tensor_diag_part,
    array_ops.transpose,
    array_ops.transpose_v2,
]
_TF_NUMPY_UNARY_OPS = [
    np_math_ops.abs,
    np_math_ops.absolute,
    np_math_ops.angle,
    np_math_ops.arccos,
    np_math_ops.arcsin,
    np_math_ops.arcsinh,
    np_math_ops.arctan,
    np_math_ops.arctanh,
    np_math_ops.bitwise_not,
    np_math_ops.cbrt,
    np_math_ops.ceil,
    np_math_ops.conj,
    np_math_ops.conjugate,
    np_math_ops.cos,
    np_math_ops.cosh,
    np_math_ops.deg2rad,
    np_math_ops.exp,
    np_math_ops.exp2,
    np_math_ops.expm1,
    np_math_ops.fabs,
    np_math_ops.fix,
    np_math_ops.floor,
    np_math_ops.log,
    np_math_ops.negative,
    np_math_ops.rad2deg,
    np_math_ops.reciprocal,
    np_math_ops.sin,
    np_math_ops.sinh,
    np_math_ops.sqrt,
    np_math_ops.tan,
    np_math_ops.tanh,
    np_math_ops.nanmean,
    np_math_ops.log2,
    np_math_ops.log10,
    np_math_ops.log1p,
    np_math_ops.positive,
    np_math_ops.sinc,
    np_math_ops.square,
    np_math_ops.diff,
    np_math_ops.sort,
    np_math_ops.average,
    np_math_ops.trace,
    np_array_ops.amax,
    np_array_ops.amin,
    np_array_ops.around,
    np_array_ops.arange,
    np_array_ops.array,
    np_array_ops.asanyarray,
    np_array_ops.asarray,
    np_array_ops.ascontiguousarray,
    np_array_ops.copy,
    np_array_ops.cumprod,
    np_array_ops.cumsum,
    np_array_ops.diag,
    np_array_ops.diagflat,
    np_array_ops.diagonal,
    np_array_ops.empty_like,
    np_array_ops.expand_dims,
    np_array_ops.flatten,
    np_array_ops.flip,
    np_array_ops.fliplr,
    np_array_ops.flipud,
    np_array_ops.imag,
    np_array_ops.max,
    np_array_ops.mean,
    np_array_ops.min,
    np_array_ops.moveaxis,
    np_array_ops.ones_like,
    np_array_ops.prod,
    np_array_ops.ravel,
    np_array_ops.real,
    np_array_ops.reshape,
    np_array_ops.rot90,
    np_array_ops.round,
    np_array_ops.squeeze,
    np_array_ops.std,
    np_array_ops.sum,
    np_array_ops.swapaxes,
    np_array_ops.transpose,
    np_array_ops.triu,
    np_array_ops.vander,
    np_array_ops.var,
    np_array_ops.zeros_like,
]

# Below are lists of binary ops that have support for WeakTensor input(s).
_ELEMENTWISE_BINARY_OPS = []

ALL_UNARY_OPS = _ELEMENTWISE_UNARY_OPS + _TF_UNARY_OPS + _TF_NUMPY_UNARY_OPS
ALL_BINARY_OPS = _ELEMENTWISE_BINARY_OPS
