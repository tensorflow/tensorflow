# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Utility methods for DTensor testing."""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import gen_stateless_random_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import special_math_ops


def expand_test_config(op_list, test_configs):
  """Returns a list of test case args that covers ops and test_configs.

  The list is a Cartesian product between op_list and test_configs.

  Args:
    op_list: A list of dicts, with items keyed by 'testcase_name' and 'op'.
      Available lists are defined later in this module.
    test_configs: A list of dicts, additional kwargs to be appended for each
      test parameters.

  Returns:
    test_configurations: a list of test parameters that covers all
      provided ops in op_list and args in test_configs.
  """
  test_configurations = []
  for op_info in op_list:
    test_index = 0
    for added_test_config in test_configs:
      test_config = op_info.copy()
      test_config.update(added_test_config)
      test_config['testcase_name'] = op_info['testcase_name'] + '_' + str(
          test_index)
      test_index += 1
      test_configurations.append(test_config)
  return test_configurations


# Disable pyformat for this block to force compact style.
# pyformat: disable
#
# Disable g-long-lambda to make the unit test suits compact (avoid def new func)
# pylint: disable=g-long-lambda
UNARY_OPS = [
    {
        'testcase_name': 'Identity',
        'op': array_ops.identity
    },
    {
        'testcase_name': 'ZerosLike',
        'op': array_ops.zeros_like_v2
    },
    {
        'testcase_name': 'Abs',
        'op': math_ops.abs
    },
    {
        'testcase_name': 'Negative',
        'op': gen_math_ops.neg
    },
    {
        'testcase_name': 'Cast',
        'op': lambda x: math_ops.cast(x, dtypes.int32)
    },
    {
        'testcase_name': 'ErfOp',
        'op': gen_math_ops.erf
    },
    {
        'testcase_name': 'Softmax',
        'op': nn_ops.softmax_v2
    },
    {
        'testcase_name': 'LogSoftmax',
        'op': nn_ops.log_softmax_v2
    },
    {
        'testcase_name': 'StopGradient',
        'op': array_ops.stop_gradient
    },
    {
        'testcase_name': 'Exp',
        'op': math_ops.exp
    },
    {
        'testcase_name': 'Sqrt',
        'op': math_ops.sqrt
    },
    {
        'testcase_name': 'Rsqrt',
        'op': math_ops.rsqrt
    },
    {
        'testcase_name': 'Reciprocal',
        'op': gen_math_ops.reciprocal
    },
    {
        'testcase_name': 'Relu',
        'op': gen_nn_ops.relu
    },
    {
        'testcase_name': 'Square',
        'op': gen_math_ops.square
    },
    {
        'testcase_name': 'Tanh',
        'op': gen_math_ops.tanh
    },
    {
        'testcase_name': 'Cos',
        'op': gen_math_ops.cos
    },
    {
        'testcase_name': 'Sigmoid',
        'op': math_ops.sigmoid
    },
    {
        'testcase_name': 'Acos',
        'op': math_ops.acos
    },
    {
        'testcase_name': 'Acosh',
        'op': gen_math_ops.acosh
    },
    {
        'testcase_name': 'Angle',
        'op': math_ops.angle
    },
    {
        'testcase_name': 'Asin',
        'op': gen_math_ops.asin
    },
    {
        'testcase_name': 'Asinh',
        'op': gen_math_ops.asinh
    },
    {
        'testcase_name': 'Atan',
        'op': gen_math_ops.atan
    },
    {
        'testcase_name': 'Bessel0e',
        'op': special_math_ops.bessel_i0e
    },
    {
        'testcase_name': 'Bessel1e',
        'op': special_math_ops.bessel_i1e
    },
    {
        'testcase_name': 'Bitcast',
        'op': lambda x: gen_array_ops.bitcast(x, type=dtypes.int32)
    },
    {
        'testcase_name': 'Ceil',
        'op': math_ops.ceil
    },
    {
        'testcase_name': 'CheckNumbers',
        'op': (lambda x: gen_array_ops.check_numerics(x, message='bug'))
    },
    {
        'testcase_name': 'ClipByValue',
        'op': (lambda x: clip_ops.clip_by_value(x, 1.5, 2.5))
    },
    {
        'testcase_name': 'Conj',
        'op': math_ops.conj
    },
    {
        'testcase_name': 'Cosh',
        'op': gen_math_ops.cosh
    },
    {
        'testcase_name': 'Digamma',
        'op': gen_math_ops.digamma
    },
    {
        'testcase_name':
            'ComplexAbs',
        'op':
            lambda x: gen_math_ops.complex_abs(
                x=math_ops.cast(x, dtypes.complex64), Tout=float, name='raw')
    },
    {
        'testcase_name': 'Sign',
        'op': math_ops.sign
    },
    {
        'testcase_name': 'Elu',
        'op': gen_nn_ops.elu
    },
    {
        'testcase_name': 'Erfc',
        'op': gen_math_ops.erfc
    },
    {
        'testcase_name': 'Expm1',
        'op': gen_math_ops.expm1
    },
    {
        'testcase_name': 'Floor',
        'op': math_ops.floor
    },
    {
        'testcase_name': 'Imag',
        'op': math_ops.imag
    },
    {
        'testcase_name': 'Inv',
        'op': (lambda x: gen_math_ops.inv(x=x, name='Inv'))
    },
    {
        'testcase_name': 'IsInf',
        'op': gen_math_ops.is_inf
    },
    {
        'testcase_name': 'IsNan',
        'op': gen_math_ops.is_nan
    },
    {
        'testcase_name': 'LeakyRelu',
        'op': (lambda x: nn_ops.leaky_relu((x - 2), alpha=0.3)),
    },
    {
        'testcase_name': 'Lgamma',
        'op': gen_math_ops.lgamma,
    },
    {
        'testcase_name': 'Log1p',
        'op': gen_math_ops.log1p,
    },
    {
        'testcase_name': 'Ndtri',
        'op': (lambda x: math_ops.ndtri(x / 100)),
    },
    {
        'testcase_name': 'Selu',
        'op': gen_nn_ops.selu,
    },
    {
        'testcase_name': 'Sin',
        'op': gen_math_ops.sin,
    },
    {
        'testcase_name': 'Sinh',
        'op': gen_math_ops.sinh,
    },
    {
        'testcase_name': 'Softplus',
        'op': math_ops.softplus,
    },
    {
        'testcase_name': 'Softsign',
        'op': gen_nn_ops.softsign,
    },
    {
        'testcase_name': 'Tan',
        'op': gen_math_ops.tan,
    },
    {
        'testcase_name': 'Round',
        'op': math_ops.round,
    },
    {
        'testcase_name': 'Rint',
        'op': gen_math_ops.rint,
    },
    {
        'testcase_name': 'Relu6',
        'op': nn_ops.relu6,
    },
    {
        'testcase_name': 'Real',
        'op': math_ops.real,
    },
    {
        'testcase_name': 'PreventGradient',
        'op': lambda x: gen_array_ops.prevent_gradient(input=x),
    },
]

BINARY_ANY_TYPE_OPS_WITH_BROADCASTING_SUPPORT = [
    {
        'testcase_name': 'Add',
        'op': math_ops.add
    },
    {
        'testcase_name': 'Subtract',
        'op': math_ops.subtract
    },
    {
        'testcase_name': 'Multiply',
        'op': math_ops.multiply
    },
    {
        'testcase_name': 'Maximum',
        'op': gen_math_ops.maximum
    },
    {
        'testcase_name': 'Minimum',
        'op': gen_math_ops.minimum
    },
    {
        'testcase_name': 'Squared_Difference',
        'op': gen_math_ops.squared_difference
    },
    {
        'testcase_name': 'GreaterEqual',
        'op': gen_math_ops.greater_equal
    },
    {
        'testcase_name': 'Equal',
        'op': math_ops.equal
    },
    {
        'testcase_name': 'NotEqual',
        'op': math_ops.not_equal
    },
    {
        'testcase_name': 'LessEqual',
        'op': gen_math_ops.less_equal
    },
    {
        'testcase_name': 'Less',
        'op': gen_math_ops.less
    },
    {
        'testcase_name': 'Pow',
        'op': math_ops.pow
    },
]

BINARY_FLOAT_OPS_WITH_BROADCASTING_SUPPORT = [
    {
        'testcase_name': 'Real_Divide',
        'op': math_ops.divide
    },
    {
        'testcase_name': 'DivNoNan',
        'op': math_ops.div_no_nan
    },
] + BINARY_ANY_TYPE_OPS_WITH_BROADCASTING_SUPPORT

BINARY_INT_OPS_WITH_BROADCASTING_SUPPORT = [
    {
        'testcase_name': 'LeftShift',
        'op': gen_bitwise_ops.left_shift
    },
    {
        'testcase_name': 'RightShift',
        'op': gen_bitwise_ops.right_shift
    },
    {
        'testcase_name': 'BitwiseOr',
        'op': gen_bitwise_ops.bitwise_or
    },
    {
        'testcase_name': 'BitwiseAnd',
        'op': gen_bitwise_ops.bitwise_and
    },
    {
        'testcase_name': 'BitwiseXor',
        'op': gen_bitwise_ops.bitwise_xor
    },
    {
        'testcase_name': 'TruncateDiv',
        'op': gen_math_ops.truncate_div
    },
    {
        'testcase_name': 'TruncateMod',
        'op': gen_math_ops.truncate_mod
    },
] + BINARY_ANY_TYPE_OPS_WITH_BROADCASTING_SUPPORT

BINARY_BOOL_OPS = [{
    'testcase_name': 'LogicalOr',
    'op': gen_math_ops.logical_or
}]

BINARY_FLOAT_OPS = [
    {
        'testcase_name': 'RsqrtGrad',
        'op': lambda y, dy: gen_math_ops.rsqrt_grad(y=y, dy=dy)
    },
    {
        'testcase_name': 'SqrtGrad',
        'op': lambda y, dy: gen_math_ops.sqrt_grad(y=y, dy=dy)
    },
    {
        'testcase_name': 'Atan2',
        'op': gen_math_ops.atan2
    },
    {
        'testcase_name': 'Betainc',
        'op': lambda a, b: gen_math_ops.betainc(a, b, 1.0)
    },
    {
        'testcase_name': 'Complex',
        'op': math_ops.complex
    },
    {
        'testcase_name':
            'EluGrad',
        'op': (lambda x, y: gen_nn_ops.elu_grad(
            gradients=x, outputs=y, name='op_elugrad'))
    },
    {
        'testcase_name': 'Igamma',
        'op': gen_math_ops.igamma
    },
    {
        'testcase_name':
            'IgammaGradA',
        'op': (lambda a, x: gen_math_ops.igamma_grad_a(
            a=a, x=x, name='IgammaGradA'))
    },
    {
        'testcase_name':
            'LeakyReluGrad',
        'op':
            (lambda x, y: gen_nn_ops.leaky_relu_grad(gradients=x, features=y)),
    },
    {
        'testcase_name': 'MulNoNan',
        'op': (lambda x, y: gen_math_ops.mul_no_nan(x=x, y=y)),
    },
    {
        'testcase_name': 'NextAfter',
        'op': gen_math_ops.next_after,
    },
    {
        'testcase_name': 'PolyGamma',
        'op': gen_math_ops.polygamma,
    },
    {
        'testcase_name': 'SeluGrad',
        'op': (lambda x, y: gen_nn_ops.selu_grad(gradients=x, outputs=y)),
    },
    {
        'testcase_name': 'Relu6Grad',
        'op': (lambda x, y: gen_nn_ops.relu6_grad(gradients=x, features=y)),
    },
    {
        'testcase_name': 'ReciprocalGrad',
        'op': (lambda x, y: gen_math_ops.reciprocal_grad(y=x, dy=y)),
    },
    {
        'testcase_name': 'Xdivy',
        'op': math_ops.xdivy,
    },
    {
        'testcase_name': 'Xlog1py',
        'op': math_ops.xlog1py,
    },
    {
        'testcase_name': 'Xlogy',
        'op': gen_math_ops.xlogy,
    },
    {
        'testcase_name': 'Zeta',
        'op': gen_math_ops.zeta,
    },
] + BINARY_FLOAT_OPS_WITH_BROADCASTING_SUPPORT

BINARY_INT_OPS = [] + BINARY_INT_OPS_WITH_BROADCASTING_SUPPORT

REDUCTION_OPS = [
    {
        'testcase_name': 'Sum',
        'op': math_ops.reduce_sum
    },
    {
        'testcase_name': 'Mean',
        'op': math_ops.reduce_mean
    },
    {
        'testcase_name': 'Prod',
        'op': math_ops.reduce_prod
    },
    {
        'testcase_name': 'Max',
        'op': math_ops.reduce_max
    },
    {
        'testcase_name': 'Min',
        'op': math_ops.reduce_min
    },
]

# TODO(b/171746536): added v2 rng ops here once supported.
RANDOM_OPS = [{
    'testcase_name': 'StatelessNorm',
    'op': gen_stateless_random_ops.stateless_random_normal,
    'dtype': dtypes.float32,
    'op_version': 'V1'
}, {
    'testcase_name': 'StatelessTruncatedNorm',
    'op': gen_stateless_random_ops.stateless_truncated_normal,
    'dtype': dtypes.float32,
    'op_version': 'V1'
}, {
    'testcase_name': 'StatelessUniform',
    'op': gen_stateless_random_ops.stateless_random_uniform,
    'dtype': dtypes.float32,
    'op_version': 'V1'
}, {
    'testcase_name': 'StatelessUniformFullInt',
    'op': gen_stateless_random_ops.stateless_random_uniform_full_int,
    'dtype': dtypes.int32,
    'op_version': 'V1'
}, {
    'testcase_name': 'StatelessRandomUniformFullIntV2',
    'op': gen_stateless_random_ops_v2.stateless_random_uniform_full_int_v2,
    'dtype': dtypes.int32,
    'op_version': 'V2'
}, {
    'testcase_name': 'StatelessRandomNormalV2',
    'op': gen_stateless_random_ops_v2.stateless_random_normal_v2,
    'dtype': dtypes.float32,
    'op_version': 'V2'
}, {
    'testcase_name': 'StatelessTruncatedNormalV2',
    'op': gen_stateless_random_ops_v2.stateless_truncated_normal_v2,
    'dtype': dtypes.float32,
    'op_version': 'V2'
}, {
    'testcase_name': 'StatelessRandomUniformV2',
    'op': gen_stateless_random_ops_v2.stateless_random_uniform_v2,
    'dtype': dtypes.float32,
    'op_version': 'V2'
}, {
    'testcase_name': 'StatelessRandomUniformIntV2',
    'op': gen_stateless_random_ops_v2.stateless_random_uniform_int_v2,
    'dtype': dtypes.int32,
    'op_version': 'V2_RANGE'
}]

# op(inputs()) is expected to return an NxM tensor (N, M both even) with a
# flexible output sharding, depending on the context `op` runs in.
EXPANSION_OPS = [
    dict(
        testcase_name='TileFrom1x1Array',
        inputs=lambda: (constant_op.constant([[1.]]), [4, 4]),
        op=gen_array_ops.tile),
    dict(
        testcase_name='TileFrom2x2Array',
        inputs=lambda: (constant_op.constant([[1., 2.], [3., 4.]]), [2, 4]),
        op=gen_array_ops.tile),
    dict(
        testcase_name='Fill',
        inputs=lambda: ([2, 4], constant_op.constant(1.)),
        op=array_ops.fill),
]

BATCH_PARALLEL_2D_WINDOW_OPS = [(
    'AvgPool',
    nn_ops.avg_pool_v2,
)]

BATCH_PARALLEL_3D_WINDOW_OPS = [(
    'MaxPool3D',
    nn_ops.max_pool3d,
), (
    'AvgPool3D',
    nn_ops.avg_pool3d,
)]

FFT_OPS = [(
    'FFT',
    gen_spectral_ops.fft,
    1,
), (
    'FFT2D',
    gen_spectral_ops.fft2d,
    2,
), (
    'FFT3D',
    gen_spectral_ops.fft3d,
    3,
), (
    'IFFT',
    gen_spectral_ops.ifft,
    1,
), (
    'IFFT2D',
    gen_spectral_ops.ifft2d,
    2,
), (
    'IFFT3D',
    gen_spectral_ops.ifft3d,
    3,
)]

RFFT_OPS = [(
    'IRFFT',
    gen_spectral_ops.irfft,
    1,
    dtypes.complex64,
), (
    'IRFFT2D',
    gen_spectral_ops.irfft2d,
    2,
    dtypes.complex64,
), (
    'IRFFT3D',
    gen_spectral_ops.irfft3d,
    3,
    dtypes.complex64,
), (
    'RFFT',
    gen_spectral_ops.rfft,
    1,
    dtypes.float32,
), (
    'RFFT2D',
    gen_spectral_ops.rfft2d,
    2,
    dtypes.float32,
), (
    'RFFT3D',
    gen_spectral_ops.rfft3d,
    3,
    dtypes.float32,
)]

PADDINGS = [
    {
        'testcase_name': 'SamePadding',
        'padding': 'SAME'
    },
    {
        'testcase_name': 'ValidPadding',
        'padding': 'VALID'
    },
]
# pylint: enable=g-long-lambda
# pyformat: enable
