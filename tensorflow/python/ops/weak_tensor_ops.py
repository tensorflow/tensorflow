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
"""Support for WeakTensor in TF ops."""

import inspect

from tensorflow.python.framework import flexible_dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import weak_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import dispatch
from tensorflow.python.util import tf_decorator


# List of unary ops that have support for WeakTensor.
_TF_UNARY_APIS = []
_TF_BINARY_APIS = []


# ==============================================================================
# Utils to handle WeakTensor inputs and outputs.
# ==============================================================================
# pylint: disable=g-doc-args,g-doc-return-or-yield
def _convert_or_cast(x, dtype, name):
  """Converts/casts the input x to dtype."""
  # TODO(b/290216343): remove this branch once we fix the precision loss bug in
  # tf.cast.
  if isinstance(x, (int, float, complex)):
    return ops.convert_to_tensor(x, dtype=dtype, name=name)
  else:
    return math_ops.cast(x, dtype=dtype, name=name)


def weak_tensor_unary_op_wrapper(op, x_arg_name=None):
  """Infers input type and adds WeakTensor support to unary ops.

  This wrapper infers input type according to the auto dtype conversion
  semantics - Tensor and NumPy inputs as Tensor of corresponding dtype and
  WeakTensor and python inputs as WeakTensor of corresponding dtype. If the
  inferred input dtype is "weak" and the op doesn't specify a return dtype,
  returns WeakTensor.
  """
  signature = inspect.signature(op)
  if x_arg_name is None:
    arg_names = iter(signature.parameters.keys())
    x_arg_name = next(arg_names)

  def wrapper(*args, **kwargs):
    if not ops.is_auto_dtype_conversion_enabled():
      return op(*args, **kwargs)
    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    bound_kwargs = bound_arguments.arguments
    x = bound_kwargs[x_arg_name]
    # No input/output handling needed when input is a Tensor because Tensor
    # input in unary op always outputs a Tensor.
    if isinstance(x, tensor.Tensor):
      return op(**bound_kwargs)
    # Infer input type and determine the result promotion type.
    try:
      target_type, is_weak = flexible_dtypes.result_type(x)
    # NotImplementedError is thrown from result_type when x is an
    # unsupported input type (e.g. CompositeTensor).
    except NotImplementedError:
      logging.warning(
          "The new dtype semantics do not support"
          f" {op.__module__}.{op.__name__}({type(x)}). Falling back to old"
          " semantics."
      )
      return op(**bound_kwargs)
    bound_kwargs[x_arg_name] = _convert_or_cast(x, target_type, "x")
    # Only return WeakTensor when dtype is NOT specified.
    if bound_kwargs.get("dtype", None) is not None:
      is_weak = False
    return weak_tensor.convert_to_weak_tensor_or_tensor(
        op(**bound_kwargs), is_weak
    )

  wrapper = tf_decorator.make_decorator(op, wrapper)

  # Update dispatch dictionary to store monkey-patched op references.
  _update_weak_tensor_patched_ops_in_dispatch_dict(wrapper)

  # Add the updated function to list of unary ops with WeakTensor support.
  _TF_UNARY_APIS.append(wrapper)
  return wrapper


def weak_tensor_binary_op_wrapper(op):
  """Determines result promotion type and adds WeakTensor support to binary ops.

  This wrapper first infers dtype of any Tensor, WeakTensor, python/numpy
  inputs. Then, both inputs are promoted to the correct promotion result dtype.
  If the result promotion dtype is "weak", returns WeakTensor.
  """
  signature = inspect.signature(op)
  arg_names = iter(signature.parameters.keys())
  x_arg_name = next(arg_names)
  y_arg_name = next(arg_names)

  def wrapper(*args, **kwargs):
    if not ops.is_auto_dtype_conversion_enabled():
      return op(*args, **kwargs)
    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    bound_kwargs = bound_arguments.arguments
    x = bound_kwargs[x_arg_name]
    y = bound_kwargs[y_arg_name]
    # Infer input type and determine the result promotion type.
    try:
      target_type, is_weak = flexible_dtypes.result_type(x, y)
    # NotImplementedError is thrown from result_type when x or y is an
    # unsupported input type (e.g. CompositeTensor).
    except NotImplementedError:
      logging.warning(
          "The new dtype semantics do not support"
          f" {op.__module__}.{op.__name__}({type(x)}, {type(y)}). Falling back"
          " to old semantics."
      )
      return op(**bound_kwargs)

    bound_kwargs[x_arg_name] = _convert_or_cast(x, target_type, "x")
    bound_kwargs[y_arg_name] = _convert_or_cast(y, target_type, "y")
    return weak_tensor.convert_to_weak_tensor_or_tensor(
        op(**bound_kwargs), is_weak
    )

  wrapper = tf_decorator.make_decorator(op, wrapper)

  # Update dispatch dictionary to store monkey-patched op references.
  _update_weak_tensor_patched_ops_in_dispatch_dict(wrapper)

  # Add the updated function to list of binary ops with WeakTensor support.
  _TF_BINARY_APIS.append(wrapper)
  return wrapper


# TODO(b/290672237): Investigate if there is a more elegant solution.
def _update_weak_tensor_patched_ops_in_dispatch_dict(patched_op):
  """Update dispatch dictionary to store WeakTensor patched op references.

  _TYPE_BASED_DISPATCH_SIGNATURES in dispatch.py stores mappings from op
  reference to all the dispatchers it's registered with. We need to update
  this dictionary to add a mapping from the patched-op reference to the
  signature dictionary the unpatched-op reference is mapped to. This ensures
  that dispatch can be reigstered and unregistered with monkey-patched ops.
  """
  dispatch_dict = dispatch._TYPE_BASED_DISPATCH_SIGNATURES  # pylint: disable=protected-access
  unpatched_api = patched_op.__wrapped__
  if unpatched_api in dispatch_dict:
    dispatch_dict[patched_op] = dispatch_dict[unpatched_api]


# ==============================================================================
# Monkey patching to add WeakTensor Support.
# ==============================================================================
# Elementwise unary ops
math_ops.abs = weak_tensor_unary_op_wrapper(math_ops.abs)
math_ops.softplus = weak_tensor_unary_op_wrapper(math_ops.softplus)
math_ops.sign = weak_tensor_unary_op_wrapper(math_ops.sign)
math_ops.real = weak_tensor_unary_op_wrapper(math_ops.real)
math_ops.imag = weak_tensor_unary_op_wrapper(math_ops.imag)
math_ops.angle = weak_tensor_unary_op_wrapper(math_ops.angle)
math_ops.round = weak_tensor_unary_op_wrapper(math_ops.round)
math_ops.sigmoid = weak_tensor_unary_op_wrapper(math_ops.sigmoid)
math_ops.log_sigmoid = weak_tensor_unary_op_wrapper(math_ops.log_sigmoid)
math_ops.conj = weak_tensor_unary_op_wrapper(math_ops.conj)
math_ops.reciprocal_no_nan = weak_tensor_unary_op_wrapper(
    math_ops.reciprocal_no_nan
)
math_ops.erfinv = weak_tensor_unary_op_wrapper(math_ops.erfinv)
math_ops.ndtri = weak_tensor_unary_op_wrapper(math_ops.ndtri)
math_ops.erfcinv = weak_tensor_unary_op_wrapper(math_ops.erfcinv)
math_ops.ceil = weak_tensor_unary_op_wrapper(math_ops.ceil)
math_ops.sqrt = weak_tensor_unary_op_wrapper(math_ops.sqrt)
math_ops.exp = weak_tensor_unary_op_wrapper(math_ops.exp)
math_ops.rsqrt = weak_tensor_unary_op_wrapper(math_ops.rsqrt)
math_ops.acos = weak_tensor_unary_op_wrapper(math_ops.acos)
math_ops.floor = weak_tensor_unary_op_wrapper(math_ops.floor)
gen_bitwise_ops.invert = weak_tensor_unary_op_wrapper(gen_bitwise_ops.invert)
gen_math_ops.acosh = weak_tensor_unary_op_wrapper(gen_math_ops.acosh)
gen_math_ops.asin = weak_tensor_unary_op_wrapper(gen_math_ops.asin)
gen_math_ops.asinh = weak_tensor_unary_op_wrapper(gen_math_ops.asinh)
gen_math_ops.atan = weak_tensor_unary_op_wrapper(gen_math_ops.atan)
gen_math_ops.atanh = weak_tensor_unary_op_wrapper(gen_math_ops.atanh)
gen_math_ops.cos = weak_tensor_unary_op_wrapper(gen_math_ops.cos)
gen_math_ops.cosh = weak_tensor_unary_op_wrapper(gen_math_ops.cosh)
gen_math_ops.digamma = weak_tensor_unary_op_wrapper(gen_math_ops.digamma)
gen_math_ops.erf = weak_tensor_unary_op_wrapper(gen_math_ops.erf)
gen_math_ops.erfc = weak_tensor_unary_op_wrapper(gen_math_ops.erfc)
gen_math_ops.expm1 = weak_tensor_unary_op_wrapper(gen_math_ops.expm1)
gen_math_ops.lgamma = weak_tensor_unary_op_wrapper(gen_math_ops.lgamma)
gen_math_ops.log = weak_tensor_unary_op_wrapper(gen_math_ops.log)
gen_math_ops.log1p = weak_tensor_unary_op_wrapper(gen_math_ops.log1p)
gen_math_ops.neg = weak_tensor_unary_op_wrapper(gen_math_ops.neg)
gen_math_ops.reciprocal = weak_tensor_unary_op_wrapper(gen_math_ops.reciprocal)
gen_math_ops.rint = weak_tensor_unary_op_wrapper(gen_math_ops.rint)
gen_math_ops.sin = weak_tensor_unary_op_wrapper(gen_math_ops.sin)
gen_math_ops.sinh = weak_tensor_unary_op_wrapper(gen_math_ops.sinh)
gen_math_ops.square = weak_tensor_unary_op_wrapper(gen_math_ops.square)
gen_math_ops.tan = weak_tensor_unary_op_wrapper(gen_math_ops.tan)
gen_math_ops.tanh = weak_tensor_unary_op_wrapper(gen_math_ops.tanh)
array_ops.zeros_like = weak_tensor_unary_op_wrapper(array_ops.zeros_like)
array_ops.zeros_like_v2 = weak_tensor_unary_op_wrapper(array_ops.zeros_like_v2)
array_ops.ones_like = weak_tensor_unary_op_wrapper(array_ops.ones_like)
array_ops.ones_like_v2 = weak_tensor_unary_op_wrapper(array_ops.ones_like_v2)
gen_array_ops.check_numerics = weak_tensor_unary_op_wrapper(
    gen_array_ops.check_numerics
)
nn_ops.relu6 = weak_tensor_unary_op_wrapper(nn_ops.relu6)
nn_ops.leaky_relu = weak_tensor_unary_op_wrapper(nn_ops.leaky_relu)
nn_ops.gelu = weak_tensor_unary_op_wrapper(nn_ops.gelu)
nn_ops.log_softmax = weak_tensor_unary_op_wrapper(nn_ops.log_softmax)
nn_ops.log_softmax_v2 = weak_tensor_unary_op_wrapper(nn_ops.log_softmax_v2)
nn_impl.swish = weak_tensor_unary_op_wrapper(nn_impl.swish)
nn_ops.elu = weak_tensor_unary_op_wrapper(nn_ops.elu)
nn_ops.relu = weak_tensor_unary_op_wrapper(nn_ops.relu)
nn_ops.selu = weak_tensor_unary_op_wrapper(nn_ops.selu)
nn_ops.softsign = weak_tensor_unary_op_wrapper(nn_ops.softsign)
image_ops.random_brightness = weak_tensor_unary_op_wrapper(
    image_ops.random_brightness
)
image_ops.stateless_random_brightness = weak_tensor_unary_op_wrapper(
    image_ops.stateless_random_brightness
)
image_ops.adjust_brightness = weak_tensor_unary_op_wrapper(
    image_ops.adjust_brightness
)
image_ops.adjust_gamma = weak_tensor_unary_op_wrapper(image_ops.adjust_gamma)
clip_ops.clip_by_value = weak_tensor_unary_op_wrapper(clip_ops.clip_by_value)
special_math_ops.dawsn = weak_tensor_unary_op_wrapper(special_math_ops.dawsn)
special_math_ops.expint = weak_tensor_unary_op_wrapper(special_math_ops.expint)
special_math_ops.fresnel_cos = weak_tensor_unary_op_wrapper(
    special_math_ops.fresnel_cos
)
special_math_ops.fresnel_sin = weak_tensor_unary_op_wrapper(
    special_math_ops.fresnel_sin
)
special_math_ops.spence = weak_tensor_unary_op_wrapper(special_math_ops.spence)
special_math_ops.bessel_i0 = weak_tensor_unary_op_wrapper(
    special_math_ops.bessel_i0
)
special_math_ops.bessel_i0e = weak_tensor_unary_op_wrapper(
    special_math_ops.bessel_i0e
)
special_math_ops.bessel_i1 = weak_tensor_unary_op_wrapper(
    special_math_ops.bessel_i1
)
special_math_ops.bessel_i1e = weak_tensor_unary_op_wrapper(
    special_math_ops.bessel_i1e
)
special_math_ops.bessel_k0 = weak_tensor_unary_op_wrapper(
    special_math_ops.bessel_k0
)
special_math_ops.bessel_k0e = weak_tensor_unary_op_wrapper(
    special_math_ops.bessel_k0e
)
special_math_ops.bessel_k1 = weak_tensor_unary_op_wrapper(
    special_math_ops.bessel_k1
)
special_math_ops.bessel_k1e = weak_tensor_unary_op_wrapper(
    special_math_ops.bessel_k1e
)
special_math_ops.bessel_j0 = weak_tensor_unary_op_wrapper(
    special_math_ops.bessel_j0
)
special_math_ops.bessel_j1 = weak_tensor_unary_op_wrapper(
    special_math_ops.bessel_j1
)
special_math_ops.bessel_y0 = weak_tensor_unary_op_wrapper(
    special_math_ops.bessel_y0
)
special_math_ops.bessel_y1 = weak_tensor_unary_op_wrapper(
    special_math_ops.bessel_y1
)

# TF Non-Elementwise Unary Ops
math_ops.reduce_euclidean_norm = weak_tensor_unary_op_wrapper(
    math_ops.reduce_euclidean_norm
)
math_ops.reduce_logsumexp = weak_tensor_unary_op_wrapper(
    math_ops.reduce_logsumexp
)
math_ops.reduce_max = weak_tensor_unary_op_wrapper(math_ops.reduce_max)
math_ops.reduce_max_v1 = weak_tensor_unary_op_wrapper(math_ops.reduce_max_v1)
math_ops.reduce_mean = weak_tensor_unary_op_wrapper(math_ops.reduce_mean)
math_ops.reduce_mean_v1 = weak_tensor_unary_op_wrapper(math_ops.reduce_mean_v1)
math_ops.reduce_min = weak_tensor_unary_op_wrapper(math_ops.reduce_min)
math_ops.reduce_min_v1 = weak_tensor_unary_op_wrapper(math_ops.reduce_min_v1)
math_ops.reduce_prod = weak_tensor_unary_op_wrapper(math_ops.reduce_prod)
math_ops.reduce_prod_v1 = weak_tensor_unary_op_wrapper(math_ops.reduce_prod_v1)
math_ops.reduce_std = weak_tensor_unary_op_wrapper(math_ops.reduce_std)
math_ops.reduce_sum = weak_tensor_unary_op_wrapper(math_ops.reduce_sum)
math_ops.reduce_sum_v1 = weak_tensor_unary_op_wrapper(math_ops.reduce_sum_v1)
math_ops.reduce_variance = weak_tensor_unary_op_wrapper(
    math_ops.reduce_variance
)
math_ops.trace = weak_tensor_unary_op_wrapper(math_ops.trace)
array_ops.reshape = weak_tensor_unary_op_wrapper(array_ops.reshape)
array_ops.depth_to_space = weak_tensor_unary_op_wrapper(
    array_ops.depth_to_space
)
array_ops.depth_to_space_v2 = weak_tensor_unary_op_wrapper(
    array_ops.depth_to_space_v2
)
array_ops.expand_dims = weak_tensor_unary_op_wrapper(array_ops.expand_dims)
array_ops.expand_dims_v2 = weak_tensor_unary_op_wrapper(
    array_ops.expand_dims_v2
)
array_ops.extract_image_patches = weak_tensor_unary_op_wrapper(
    array_ops.extract_image_patches
)
array_ops.extract_image_patches_v2 = weak_tensor_unary_op_wrapper(
    array_ops.extract_image_patches_v2
)
array_ops.identity = weak_tensor_unary_op_wrapper(array_ops.identity)
array_ops.matrix_diag = weak_tensor_unary_op_wrapper(array_ops.matrix_diag)
array_ops.matrix_diag_part = weak_tensor_unary_op_wrapper(
    array_ops.matrix_diag_part
)
array_ops.matrix_transpose = weak_tensor_unary_op_wrapper(
    array_ops.matrix_transpose
)
array_ops.space_to_depth = weak_tensor_unary_op_wrapper(
    array_ops.space_to_depth
)
array_ops.space_to_depth_v2 = weak_tensor_unary_op_wrapper(
    array_ops.space_to_depth_v2
)
array_ops.squeeze = weak_tensor_unary_op_wrapper(array_ops.squeeze)
array_ops.squeeze_v2 = weak_tensor_unary_op_wrapper(array_ops.squeeze_v2)
array_ops.stop_gradient = weak_tensor_unary_op_wrapper(array_ops.stop_gradient)
array_ops.tensor_diag_part = weak_tensor_unary_op_wrapper(
    array_ops.tensor_diag_part
)
array_ops.transpose = weak_tensor_unary_op_wrapper(array_ops.transpose)
array_ops.transpose_v2 = weak_tensor_unary_op_wrapper(array_ops.transpose_v2)

# TF NumPy Unary Ops
np_math_ops.abs = weak_tensor_unary_op_wrapper(np_math_ops.abs)
np_math_ops.absolute = weak_tensor_unary_op_wrapper(np_math_ops.absolute)
np_math_ops.angle = weak_tensor_unary_op_wrapper(np_math_ops.angle)
np_math_ops.arccos = weak_tensor_unary_op_wrapper(np_math_ops.arccos)
np_math_ops.arcsin = weak_tensor_unary_op_wrapper(np_math_ops.arcsin)
np_math_ops.arcsinh = weak_tensor_unary_op_wrapper(np_math_ops.arcsinh)
np_math_ops.arctan = weak_tensor_unary_op_wrapper(np_math_ops.arctan)
np_math_ops.arctanh = weak_tensor_unary_op_wrapper(np_math_ops.arctanh)
np_math_ops.bitwise_not = weak_tensor_unary_op_wrapper(np_math_ops.bitwise_not)
np_math_ops.cbrt = weak_tensor_unary_op_wrapper(np_math_ops.cbrt)
np_math_ops.ceil = weak_tensor_unary_op_wrapper(np_math_ops.ceil)
np_math_ops.conj = weak_tensor_unary_op_wrapper(np_math_ops.conj)
np_math_ops.conjugate = weak_tensor_unary_op_wrapper(np_math_ops.conjugate)
np_math_ops.cos = weak_tensor_unary_op_wrapper(np_math_ops.cos)
np_math_ops.cosh = weak_tensor_unary_op_wrapper(np_math_ops.cosh)
np_math_ops.deg2rad = weak_tensor_unary_op_wrapper(np_math_ops.deg2rad)
np_math_ops.exp = weak_tensor_unary_op_wrapper(np_math_ops.exp)
np_math_ops.exp2 = weak_tensor_unary_op_wrapper(np_math_ops.exp2)
np_math_ops.expm1 = weak_tensor_unary_op_wrapper(np_math_ops.expm1)
np_math_ops.fabs = weak_tensor_unary_op_wrapper(np_math_ops.fabs)
np_math_ops.fix = weak_tensor_unary_op_wrapper(np_math_ops.fix)
np_math_ops.floor = weak_tensor_unary_op_wrapper(np_math_ops.floor)
np_math_ops.log = weak_tensor_unary_op_wrapper(np_math_ops.log)
np_math_ops.negative = weak_tensor_unary_op_wrapper(np_math_ops.negative)
np_math_ops.rad2deg = weak_tensor_unary_op_wrapper(np_math_ops.rad2deg)
np_math_ops.reciprocal = weak_tensor_unary_op_wrapper(np_math_ops.reciprocal)
np_math_ops.sin = weak_tensor_unary_op_wrapper(np_math_ops.sin)
np_math_ops.sinh = weak_tensor_unary_op_wrapper(np_math_ops.sinh)
np_math_ops.sqrt = weak_tensor_unary_op_wrapper(np_math_ops.sqrt)
np_math_ops.tan = weak_tensor_unary_op_wrapper(np_math_ops.tan)
np_math_ops.tanh = weak_tensor_unary_op_wrapper(np_math_ops.tanh)
np_math_ops.nanmean = weak_tensor_unary_op_wrapper(np_math_ops.nanmean)
np_math_ops.log2 = weak_tensor_unary_op_wrapper(np_math_ops.log2)
np_math_ops.log10 = weak_tensor_unary_op_wrapper(np_math_ops.log10)
np_math_ops.log1p = weak_tensor_unary_op_wrapper(np_math_ops.log1p)
np_math_ops.positive = weak_tensor_unary_op_wrapper(np_math_ops.positive)
np_math_ops.sinc = weak_tensor_unary_op_wrapper(np_math_ops.sinc)
np_math_ops.square = weak_tensor_unary_op_wrapper(np_math_ops.square)
np_math_ops.diff = weak_tensor_unary_op_wrapper(np_math_ops.diff)
np_math_ops.sort = weak_tensor_unary_op_wrapper(np_math_ops.sort)
np_math_ops.average = weak_tensor_unary_op_wrapper(np_math_ops.average)
np_math_ops.trace = weak_tensor_unary_op_wrapper(np_math_ops.trace)
np_array_ops.amax = weak_tensor_unary_op_wrapper(np_array_ops.amax)
np_array_ops.amin = weak_tensor_unary_op_wrapper(np_array_ops.amin)
np_array_ops.around = weak_tensor_unary_op_wrapper(np_array_ops.around)
np_array_ops.arange = weak_tensor_unary_op_wrapper(np_array_ops.arange)
np_array_ops.array = weak_tensor_unary_op_wrapper(np_array_ops.array)
np_array_ops.asanyarray = weak_tensor_unary_op_wrapper(np_array_ops.asanyarray)
np_array_ops.asarray = weak_tensor_unary_op_wrapper(np_array_ops.asarray)
np_array_ops.ascontiguousarray = weak_tensor_unary_op_wrapper(
    np_array_ops.ascontiguousarray
)
np_array_ops.copy = weak_tensor_unary_op_wrapper(np_array_ops.copy)
np_array_ops.cumprod = weak_tensor_unary_op_wrapper(np_array_ops.cumprod)
np_array_ops.cumsum = weak_tensor_unary_op_wrapper(np_array_ops.cumsum)
np_array_ops.diag = weak_tensor_unary_op_wrapper(np_array_ops.diag)
np_array_ops.diagflat = weak_tensor_unary_op_wrapper(np_array_ops.diagflat)
np_array_ops.diagonal = weak_tensor_unary_op_wrapper(np_array_ops.diagonal)
np_array_ops.empty_like = weak_tensor_unary_op_wrapper(np_array_ops.empty_like)
np_array_ops.expand_dims = weak_tensor_unary_op_wrapper(
    np_array_ops.expand_dims
)
np_array_ops.flatten = weak_tensor_unary_op_wrapper(np_array_ops.flatten)
np_array_ops.flip = weak_tensor_unary_op_wrapper(np_array_ops.flip)
np_array_ops.fliplr = weak_tensor_unary_op_wrapper(np_array_ops.fliplr)
np_array_ops.flipud = weak_tensor_unary_op_wrapper(np_array_ops.flipud)
np_array_ops.full_like = weak_tensor_unary_op_wrapper(np_array_ops.full_like)
np_array_ops.imag = weak_tensor_unary_op_wrapper(np_array_ops.imag)
np_array_ops.max = weak_tensor_unary_op_wrapper(np_array_ops.max)
np_array_ops.mean = weak_tensor_unary_op_wrapper(np_array_ops.mean)
np_array_ops.min = weak_tensor_unary_op_wrapper(np_array_ops.min)
np_array_ops.moveaxis = weak_tensor_unary_op_wrapper(np_array_ops.moveaxis)
np_array_ops.ones_like = weak_tensor_unary_op_wrapper(np_array_ops.ones_like)
np_array_ops.prod = weak_tensor_unary_op_wrapper(np_array_ops.prod)
np_array_ops.ravel = weak_tensor_unary_op_wrapper(np_array_ops.ravel)
np_array_ops.real = weak_tensor_unary_op_wrapper(np_array_ops.real)
np_array_ops.reshape = weak_tensor_unary_op_wrapper(np_array_ops.reshape)
np_array_ops.repeat = weak_tensor_unary_op_wrapper(np_array_ops.repeat)
np_array_ops.rot90 = weak_tensor_unary_op_wrapper(np_array_ops.rot90)
np_array_ops.round = weak_tensor_unary_op_wrapper(np_array_ops.round)
np_array_ops.squeeze = weak_tensor_unary_op_wrapper(np_array_ops.squeeze)
np_array_ops.std = weak_tensor_unary_op_wrapper(np_array_ops.std)
np_array_ops.sum = weak_tensor_unary_op_wrapper(np_array_ops.sum)
np_array_ops.swapaxes = weak_tensor_unary_op_wrapper(np_array_ops.swapaxes)
np_array_ops.transpose = weak_tensor_unary_op_wrapper(np_array_ops.transpose)
np_array_ops.triu = weak_tensor_unary_op_wrapper(np_array_ops.triu)
np_array_ops.vander = weak_tensor_unary_op_wrapper(np_array_ops.vander)
np_array_ops.var = weak_tensor_unary_op_wrapper(np_array_ops.var)
np_array_ops.zeros_like = weak_tensor_unary_op_wrapper(np_array_ops.zeros_like)

# Binary ops
math_ops.add = weak_tensor_binary_op_wrapper(math_ops.add)
gen_math_ops.sub = weak_tensor_binary_op_wrapper(gen_math_ops.sub)
math_ops.multiply = weak_tensor_binary_op_wrapper(math_ops.multiply)
math_ops.multiply_no_nan = weak_tensor_binary_op_wrapper(
    math_ops.multiply_no_nan
)
math_ops.matmul = weak_tensor_binary_op_wrapper(math_ops.matmul)
# In scalar_mul(scalar, x), dtype should be solely inferred from the dtype of x.
math_ops.scalar_mul = weak_tensor_unary_op_wrapper(math_ops.scalar_mul, "x")
math_ops.divide = weak_tensor_binary_op_wrapper(math_ops.divide)
math_ops.div_no_nan = weak_tensor_binary_op_wrapper(math_ops.div_no_nan)
# pylint: disable=protected-access
math_ops._truediv_python3 = weak_tensor_binary_op_wrapper(
    math_ops._truediv_python3
)
gen_math_ops.real_div = weak_tensor_binary_op_wrapper(gen_math_ops.real_div)
gen_math_ops.truncate_div = weak_tensor_binary_op_wrapper(
    gen_math_ops.truncate_div
)
gen_math_ops.floor_div = weak_tensor_binary_op_wrapper(gen_math_ops.floor_div)
gen_math_ops.truncate_mod = weak_tensor_binary_op_wrapper(
    gen_math_ops.truncate_mod
)
gen_math_ops.floor_mod = weak_tensor_binary_op_wrapper(gen_math_ops.floor_mod)
gen_math_ops._pow = weak_tensor_binary_op_wrapper(gen_math_ops._pow)


# ==============================================================================
# Update old op references.
# ==============================================================================
math_ops.realdiv = gen_math_ops.real_div
math_ops.truncatediv = gen_math_ops.truncate_div
math_ops.floor_div = gen_math_ops.floor_div
math_ops.truncatemod = gen_math_ops.truncate_mod
math_ops.floormod = gen_math_ops.floor_mod

# Set WeakTensor dunder methods.
# Tensor unary ops do not need WeakTensor support.
weak_tensor.WeakTensor.__invert__ = math_ops.invert_
weak_tensor.WeakTensor.__neg__ = gen_math_ops.neg
weak_tensor.WeakTensor.__abs__ = math_ops.abs

# Inherit rest of the dunder methods from Tensor.
unary_dunder_methods = ["__invert__", "__neg__", "__abs__"]
for operator in tensor.Tensor.OVERLOADABLE_OPERATORS:
  if operator in unary_dunder_methods:
    continue
  tensor_oper = getattr(tensor.Tensor, operator)
  setattr(weak_tensor.WeakTensor, operator, tensor_oper)

# Add/Update NumPy methods in Tensor and WeakTensor.
np_math_ops.enable_numpy_methods_on_tensor()
np_math_ops._enable_numpy_methods(weak_tensor.WeakTensor)
# pylint: enable=protected-access
