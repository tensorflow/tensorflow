# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Grappler Remapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import sysconfig
from tensorflow.python.platform import test
from tensorflow.python.util import _pywrap_utils


def _input(shape):
  """Generates an input of a given shape."""
  return variables.Variable(random_ops.truncated_normal(shape, seed=0))


def _weight(shape):
  """Generates a weight of a given shape."""
  # Note that the lambda is needed to allow construction inside loops.
  return variables.Variable(lambda: init_ops.glorot_uniform_initializer(seed=0)
                            (shape))


def _bias(shape):
  """Generates a bias of a given shape."""
  return constant_op.constant(0.1, shape=shape)


def _get_config(remapping_on=False):
  """Returns a CongfigProto with remapper optimizer on/off."""
  rewrite_config = rewriter_config_pb2.RewriterConfig(
      remapping=rewriter_config_pb2.RewriterConfig
      .ON if remapping_on else rewriter_config_pb2.RewriterConfig.OFF)
  rewrite_config.min_graph_nodes = -1
  graph_options = config_pb2.GraphOptions(rewrite_options=rewrite_config)
  config = config_pb2.ConfigProto(graph_options=graph_options)
  return config


class RemapperTest(test.TestCase, parameterized.TestCase):
  """Tests the Grappler remapper optimizer."""
  def setUp(self):
    # Gelu fusion on GPU requires cublasLt
    os.environ['TF_USE_CUBLASLT'] = '1'

  def _maybe_skip(self, mode):
    if mode == 'cuda':
      # It seems the windows os cannot correctly query the cuda_version.
      # TODO(kaixih@nvidia): Remove this when it works.
      if os.name == "nt":
        self.skipTest("This test doesn't support Windows")

      if not test.is_gpu_available(cuda_only=True):
        self.skipTest('This test requires GPU.')
      cuda_version_str = sysconfig.get_build_info().get('cuda_version', '0.0')
      cuda_version = tuple([int(x) for x in cuda_version_str.split('.')])
      if cuda_version < (11, 4):
        self.skipTest('This test requires CUDA >= 11.4.')

    if mode == 'mkl' and not test_util.IsMklEnabled():
      self.skipTest('MKL is not enabled.')

  def _VerifyValues(self, model_fn, use_low_precision, fused_op, epilog_ops):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()
    # Compute reference value.
    config = _get_config(remapping_on=False)
    with session.Session(config=config) as sess:
      sess.run(variables.global_variables_initializer())
      output_ref = sess.run(
          model_fn, options=run_options, run_metadata=metadata)
    # Compute output with fusion.
    config = _get_config(remapping_on=True)
    with session.Session(config=config) as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(
          model_fn, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]

    # Graph should contain fused op.
    found_fused_op = False
    for node in graph.node:
      if node.op in fused_op:
        fused_ops = node.attr['fused_ops'].list.s
        ops_matched = len(fused_ops) >= 1 and len(fused_ops) == len(epilog_ops)
        for op_a, op_b in zip(fused_ops, epilog_ops):
          if op_a != op_b:
            ops_matched = False
            break
        found_fused_op = ops_matched
        break
    self.assertTrue(found_fused_op)

    # Computed output value should be close to reference value.
    tol = 1e-2 if use_low_precision else 1e-5
    self.assertAllClose(output_ref, output_val, atol=tol, rtol=tol)

    return graph

  @parameterized.parameters(['cuda', 'mkl'])
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_matmul_biasadd_gelu_fusion(self, mode):
    """Test MatMul+BiasAdd+Gelu fusion."""
    self._maybe_skip(mode)
    data_types = [dtypes.float32]
    if mode == 'cuda':
      data_types.append(dtypes.float16)
    elif mode == 'mkl':
      data_types.append(dtypes.bfloat16)

    is_bf16_supported = _pywrap_utils.IsBF16SupportedByOneDNNOnThisCPU()

    m, n, k = (3, 3, 4)  # Matrix dimensions
    for precision in data_types:
      for approximate in (False, True):
        # Gelu exact (approximate=False) is not supported with bfloat16
        # precision since no support for Erf with bfloat16 data type.
        # TODO(intel-tf): Enable gelu exact with bfloat16, when Erf op is
        # supported with bfloat16.
        if precision == dtypes.bfloat16:
          if not (approximate and is_bf16_supported):
            continue

        # TODO(kaixih@nvidia): Enable gelu exact when Erf op is supported with
        # cublaslt.
        if mode == 'cuda' and not approximate:
          continue

        device = '/device:GPU:0' if mode == 'cuda' else '/device:CPU:0'
        # Create MatMul + BiasAdd + Gelu graph
        ops.reset_default_graph()
        with ops.device(device):
          x = _input([m, k])
          w = _weight([k, n])
          b = _bias([n])
          x = math_ops.cast(x, precision)
          w = math_ops.cast(w, precision)
          b = math_ops.cast(b, precision)
          y = math_ops.matmul(x, w)
          z = nn.bias_add(y, b)
          out = nn.gelu(z, approximate=approximate)

        gelu_type = b'GeluApproximate' if approximate else b'GeluExact'
        epilog_ops = [b'BiasAdd', gelu_type]
        fused_op = ['_MklNativeFusedMatMul', '_MklFusedMatMul']
        graph = self._VerifyValues(out, precision == 'bfloat16', fused_op,
                                   epilog_ops)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_conv2d_biasadd_relu_fusion(self):
    """Test Conv2D+BiasAdd+Relu fusion."""
    if not test_util.is_gpu_available():
      self.skipTest('No GPU available')

    N, H, W, C = (5, 3, 3, 4)

    for precision in ('float16', 'float32'):
      ops.reset_default_graph()
      x_shape = [N, C, H, W]
      x_format = 'NCHW'
      b_format = 'NC..'
      use_fp16 = precision == 'float16'
      if use_fp16:
        x_shape = [N, H, W, C]
        x_format = 'NHWC'
        b_format = 'N..C'

      x = _input(x_shape)
      w = _weight([2, 2, C, C])
      b = _bias([C])

      if use_fp16:
        x = math_ops.cast(x, dtypes.float16)
        w = math_ops.cast(w, dtypes.float16)
        b = math_ops.cast(b, dtypes.float16)

      y = nn_ops.conv2d(
          x, w, strides=(1, 1), padding='SAME', data_format=x_format)
      z = nn.bias_add(y, b, data_format=b_format)
      out = nn.relu(z)
      out = array_ops.identity(out)

      epilog_ops = [b'BiasAdd', b'Relu']
      fused_op = ['_FusedConv2D']
      graph = self._VerifyValues(out, use_fp16, fused_op, epilog_ops)

if __name__ == '__main__':
  test.main()
