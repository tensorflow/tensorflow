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
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import sysconfig as sysconfig_lib
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
    super(RemapperTest, self).setUp()
    # GeluApproximate fusion on GPU requires cublasLt.
    os.environ['TF_USE_CUBLASLT'] = '1'
    # GeluExact fusion and conv runtime fusion on GPU requires cuDNN frontend.
    os.environ['TF_CUDNN_USE_FRONTEND'] = '1'
    os.environ['TF_CUDNN_USE_RUNTIME_FUSION'] = '1'

  def maybe_skip_test(self, mode):
    if mode == 'cuda':
      # It seems the windows os cannot correctly query the cuda_version.
      # TODO(kaixih@nvidia): Remove this when it works.
      if os.name == 'nt':
        self.skipTest("This test doesn't support Windows")

      # The cublaslt matmul with gelu epilog is only supported since cuda 11.4.
      if not test.is_gpu_available(cuda_only=True):
        self.skipTest('This test requires GPU.')
      cuda_version_str = sysconfig_lib.get_build_info().get(
          'cuda_version', '0.0')
      cuda_version = tuple([int(x) for x in cuda_version_str.split('.')])
      if cuda_version < (11, 4):
        self.skipTest('This test requires CUDA >= 11.4.')

    if mode == 'mkl' and not test_util.IsMklEnabled():
      self.skipTest('MKL is not enabled.')

  def _VerifyNoFusion(self, model_fn):
    ops.add_to_collection('train_op', model_fn)
    mg = meta_graph.create_meta_graph_def(graph=model_fn.graph)

    # Compute referene
    config = _get_config(remapping_on=False)
    gdef_ref = tf_optimizer.OptimizeGraph(config, mg)

    # Compute with remapping ON
    config = _get_config(remapping_on=True)
    gdef = tf_optimizer.OptimizeGraph(config, mg)

    self.assertEqual(len(gdef_ref.node), len(gdef.node))
    self.assertAllEqual([n.op for n in gdef_ref.node],
                        [n.op for n in gdef.node])

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
  def test_matmul_biasadd_activation_fusion(self, mode):
    """Test MatMul+BiasAdd+Gelu fusion."""
    self.maybe_skip_test(mode)

    def gelu_approximate(x):
      return nn.gelu(x, approximate=True)

    def gelu_exact(x):
      return nn.gelu(x, approximate=False)

    device = '/device:GPU:0' if mode == 'cuda' else '/device:CPU:0'
    config = []
    if mode == 'mkl':
      config.append((dtypes.float32, gelu_exact, b'GeluExact'))
      config.append((dtypes.float32, gelu_approximate, b'GeluApproximate'))
      if _pywrap_utils.IsBF16SupportedByOneDNNOnThisCPU():
        config.append((dtypes.bfloat16, gelu_approximate, b'GeluApproximate'))
        config.append((dtypes.bfloat16, gelu_exact, b'GeluExact'))
    elif mode == 'cuda':
      config.append((dtypes.float32, gelu_approximate, b'GeluApproximate'))
      config.append((dtypes.float16, gelu_approximate, b'GeluApproximate'))
      # Gelu exact fusion is supported by cuDNN frontend APIs and performant
      # with fp16 and on Ampere GPUs and later.
      if (test_util.is_gpu_available(
          cuda_only=True, min_cuda_compute_capability=(8, 0))):
        config.append((dtypes.float16, gelu_exact, b'GeluExact'))
        config.append((dtypes.float16, math_ops.tanh, b'Tanh'))
        config.append((dtypes.float16, math_ops.sigmoid, b'Sigmoid'))

    m, n, k = (2, 4, 6)  # Matrix dimensions
    fused_op = ['_MklNativeFusedMatMul', '_MklFusedMatMul', '_FusedMatMul']

    for precision, act_fn, act_name in config:
      for transpose in (False, True):
        # Create MatMul + BiasAdd + Activation graph
        ops.reset_default_graph()
        with ops.device(device):
          x = _input([k, m] if transpose else [m, k])
          w = _weight([n, k] if transpose else [k, n])
          b = _bias([n])
          x = math_ops.cast(x, precision)
          w = math_ops.cast(w, precision)
          b = math_ops.cast(b, precision)
          y = math_ops.matmul(
              x, w, transpose_a=transpose, transpose_b=transpose)
          z = nn.bias_add(y, b)
          out = act_fn(z)

        if transpose and (device == '/device:CPU:0') and \
            act_name in (b'GeluApproximate', b'GeluExact'):
          if precision == dtypes.bfloat16:
            # No fusion should happen on CPU.
            self._VerifyNoFusion(out)
            continue
          else:
            # Gelu should not get fused, only BiasAdd.
            epilog_ops = [b'BiasAdd']
        else:
          epilog_ops = [b'BiasAdd', act_name]
        graph = self._VerifyValues(out, precision != dtypes.float32, fused_op,
                                   epilog_ops)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_conv2d_biasadd_act_fusion(self):
    """Test Conv2D+BiasAdd+Relu fusion."""
    if not test_util.is_gpu_available():
      self.skipTest('No GPU available')

    N, H, W, C = (5, 3, 3, 8)  # pylint: disable=invalid-name
    # The runtime fusion requires the output dims to be 32-bit aligned.
    self.assertEqual(C % 2, 0)

    act_fns = [nn.relu]
    act_names = [b'Relu']

    if test_util.is_gpu_available(
        cuda_only=True, min_cuda_compute_capability=(8, 0)):
      act_fns += [nn.elu, nn.relu6, nn.leaky_relu]
      act_names += [b'Elu', b'Relu6', b'LeakyRelu']

    for precision in ('float16', 'float32'):
      for act_fn, act_name in zip(act_fns, act_names):
        use_fp16 = precision == 'float16'
        # The runtime fusion (when the activation is not relu) only supports
        # fp16 at this moment.
        if not use_fp16 and act_name != b'Relu':
          continue

        ops.reset_default_graph()
        x_shape = [N, C, H, W]
        x_format, b_format = ('NCHW', 'NC..')
        if use_fp16:
          x_shape = [N, H, W, C]
          x_format, b_format = ('NHWC', 'N..C')

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
        out = act_fn(z)
        out = array_ops.identity(out)

        epilog_ops = [b'BiasAdd', act_name]
        fused_op = ['_FusedConv2D']
        graph = self._VerifyValues(out, use_fp16, fused_op, epilog_ops)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_two_conv2d_fusions(self):
    """Test two Conv2D patterns and only the second is fusable."""
    if not test_util.is_gpu_available(
        cuda_only=True, min_cuda_compute_capability=(8, 0)):
      self.skipTest('No GPU with compute compatibility >= 8.0 available')

    N, H, W, C = (5, 3, 3, 8)  # pylint: disable=invalid-name

    ops.reset_default_graph()
    x_shape = [N, C, H, W]
    x_format, b_format = ('NCHW', 'NC..')

    x = _input(x_shape)
    w = _weight([2, 2, C, C])
    b = _bias([C])

    y = nn_ops.conv2d(
        x, w, strides=(1, 1), padding='SAME', data_format=x_format)
    y = nn.bias_add(y, b, data_format=b_format)
    y = nn.leaky_relu(y)
    y = nn_ops.conv2d(
        y, w, strides=(1, 1), padding='SAME', data_format=x_format)
    y = nn.bias_add(y, b, data_format=b_format)
    y = nn.relu(y)
    out = array_ops.identity(y)

    # The first Conv-BiasAdd-LeakyRelu is not fusable because cuDNN requires
    # fp16 for this pattern. The second Conv-BiasAdd-Relu is fusable.
    epilog_ops = [b'BiasAdd', b'Relu']
    fused_op = ['_FusedConv2D']
    self._VerifyValues(out, False, fused_op, epilog_ops)


if __name__ == '__main__':
  test.main()
