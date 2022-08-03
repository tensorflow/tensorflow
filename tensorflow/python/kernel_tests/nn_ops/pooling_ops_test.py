# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for pooling operations."""

import collections
import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
import tensorflow.python.framework.config as config_exec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def GetDeviceScope(self, use_gpu=False):
  if context.executing_eagerly():
    if use_gpu and test.is_gpu_available():
      return ops.device("GPU:0")
    return ops.device("CPU:0")
  else:
    return self.session(use_gpu=use_gpu)


# TODO(jlebar): Convert the rest of this file to parameters.parameterized().
# Then remove GetTestConfigs() and rename GetTestConfigsDicts().
def GetTestConfigsDicts(v1_fn,
                        v2_fn=None,
                        one_dimensional=False,
                        allow_gpu=True):
  # (data_format, use_gpu) tuple
  if one_dimensional:
    configs0 = [
        ("NWC", False),
        ("NWC", True),
        ("NCW", True),
    ]
  else:
    configs0 = [
        ("NHWC", False),
        ("NHWC", True),
        ("NCHW", True),
    ]
    # NCHW_VECT_C only supported for max_pool.
    if (v1_fn == nn_ops.max_pool or v1_fn == nn_ops.max_pool1d or
        v2_fn == nn_ops.max_pool_v2 or v2_fn == gen_nn_ops.max_pool_v2):
      configs0.append(("NCHW_VECT_C", True))

  # (data_format, use_gpu, data_type) tuple
  configs1 = []
  for data_format, use_gpu in configs0:
    configs1.append((data_format, use_gpu, dtypes.float32))

    # In our test, VECT_C always uses float32.  (It gets converted to int8 in
    # the test runner.)
    if data_format == "NCHW_VECT_C":
      continue

    configs1 += [(data_format, use_gpu, dtypes.float16),
                 (data_format, use_gpu, dtypes.float64)]

  # Convert from tuple to dict and add v1/v2 versions.
  ret = []
  for data_format, use_gpu, data_type in configs1:
    ret.append({
        "pool_func": v1_fn,
        "data_format": data_format,
        "data_type": data_type,
        "use_gpu": use_gpu,
        "v2": False
    })
    if v2_fn:
      ret.append({
          "pool_func": v2_fn,
          "data_format": data_format,
          "data_type": data_type,
          "use_gpu": use_gpu,
          "v2": False
      })
      ret.append({
          "pool_func": v2_fn,
          "data_format": data_format,
          "data_type": data_type,
          "use_gpu": use_gpu,
          "v2": True
      })

  # Filter out GPU configs if necessary.
  if not allow_gpu:
    ret = [c for c in ret if not c["use_gpu"]]

  return ret


def GetTestConfigs(include_nchw_vect_c=False, one_dimensional=False):
  """Get all the valid tests configs to run.

  Args:
    include_nchw_vect_c: Whether to include NCHW_VECT_C in the test configs.
    one_dimensional: If it's a 1D test

  Returns:
    all the valid test configs as tuples of data_format and use_gpu.
  """
  if one_dimensional:
    test_configs = [("NWC", False), ("NWC", True)]
    if test.is_gpu_available(cuda_only=True):
      test_configs += [("NCW", True)]
    return test_configs
  test_configs = [("NHWC", False), ("NHWC", True)]
  if not test.is_gpu_available(cuda_only=True):
    tf_logging.info("NCHW and NCHW_VECT_C tests skipped because not run with "
                    "--config=cuda or no GPUs available.")
    return test_configs
  # "NCHW" format is currently supported exclusively on CUDA GPUs.
  test_configs += [("NCHW", True)]
  if include_nchw_vect_c:
    if test.is_gpu_available(
        cuda_only=True, min_cuda_compute_capability=(6, 1)):
      test_configs += [("NCHW_VECT_C", True)]
    else:
      tf_logging.info("NCHW_VECT_C test skipped because no GPUs with "
                      "compute capability >= 6.1 are available.")

  return test_configs


def GetShrunkInceptionMaxPoolShapes(shrink=30):
  """Iterator for some of the max pool ops in the Inception 2015 model.

  Args:
    shrink: Factor to shrink depth relative to Inception.

  Yields:
    Tuple (name, input_size, filter_size, out_size, strides, padding)
  """
  names = ["maxpool2", "maxpool3", "maxpool4", "maxpool5"]
  input_sizes = [[32, 71, 71, 192], [32, 35, 35, 288], [32, 17, 17, 1248],
                 [32, 8, 8, 2048]]
  filter_sizes = [[1, 3, 3, 1], [1, 3, 3, 1], [1, 3, 3, 1], [1, 3, 3, 1]]
  output_sizes = [[32, 35, 35, 192], [32, 17, 17, 288], [32, 8, 8, 1248],
                  [32, 8, 8, 2048]]
  strides = [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1]]
  # Shrink each depth value
  for i in input_sizes:
    i[3] //= shrink
  for o in output_sizes:
    o[3] //= shrink
  paddings = ["VALID", "VALID", "VALID", "SAME"]
  for n, i, f, o, s, p in zip(names, input_sizes, filter_sizes, output_sizes,
                              strides, paddings):
    yield n, i, f, o, s, p


@test_util.with_eager_op_as_function
class PoolingTest(test.TestCase, parameterized.TestCase):

  def _isMaxPool(self, func):
    return func in (nn_ops.max_pool, nn_ops.max_pool_v2)

  def _VerifyOneType(self, pool_func, input_sizes, ksize, strides, padding,
                     data_format, data_type, expected, use_gpu, v2,
                     use_negative_input=False):
    """Verifies the output values of the pooling function.

    Args:
      pool_func: Function to be called, co.MaxPool, co.AvgPool,
        or the Lua version.
      input_sizes: Input tensor dimensions.
      ksize: The kernel size dimensions
      strides: The stride dimensions
      padding: Padding type.
      data_format: The data format we use to run the pooling operation.
      data_type: The data type to use to run the pooling operation.
      expected: An array containing the expected operation outputs.
      use_gpu: Whether we are running on GPU.
      v2: Whether to use v2 version.
      use_negative_input: If the input values should be negative.
    """
    # Check that this test is compatible with the hardware we have.  (Really
    # this should be done in GetTestConfigsDicts(), but when that runs, we
    # haven't initialized enough of TF to know what our hardware is!)
    if use_gpu and not test.is_gpu_available():
      self.skipTest("No GPU is available.")
    if use_gpu and data_type == dtypes.float64 and test.is_built_with_rocm():
      self.skipTest("ROCm pooling ops don't support float64.")
    if use_gpu and data_format == "NCHW_VECT_C" and not test.is_gpu_available(
        cuda_only=True, min_cuda_compute_capability=(6, 1)):
      self.skipTest("NCHW_VECT_C requires sm61+.")

    if v2 and data_format != "NHWC":
      self.skipTest("v2 not supported for %s" % data_format)
    if v2 and not isinstance(padding, str):
      self.skipTest("non-constant ksize/strides requires nonexplicit padding")
    if data_format == "NCHW_VECT_C":
      if data_type != dtypes.float32:
        self.skipTest("quantization to qint8 not implemented for %r" %
                      data_type)
      if input_sizes[-1] % 4 != 0:
        self.skipTest("Skipping test for depth %d" % input_sizes[-1])

    total_size = 1
    for s in input_sizes:
      total_size *= s
    tf_logging.info("Running %s test. %r %r %d %r %r %r %s", data_format, v2,
                    input_sizes, total_size, pool_func, ksize, strides,
                    data_type)
    # Initializes the input tensor with array containing incrementing
    # numbers from 1, wrapping round to -127 after 127 to support int8.
    y = -1 if use_negative_input else 1
    x = [(((f + 128) % 255) - 127)*y for f in range(total_size)]
    with self.cached_session(use_gpu=use_gpu):
      t = constant_op.constant(x, shape=input_sizes, dtype=data_type)
      if data_format in ("NCHW", "NCHW_VECT_C", "NCW"):
        if data_format == "NCHW_VECT_C":
          t = test_util.NHWCToNCHW_VECT_C(t)
          t, _, _ = gen_array_ops.quantize_v2(t, -128.0, 127.0, dtypes.qint8)
        else:
          t = test_util.NHWCToNCHW(t)
        ksize = test_util.NHWCToNCHW(ksize)
        strides = test_util.NHWCToNCHW(strides)
        if isinstance(padding, list):
          padding = test_util.NHWCToNCHW(padding)
      ksize_placeholder = array_ops.placeholder(dtypes.int32, shape=[4])
      strides_placeholder = array_ops.placeholder(dtypes.int32, shape=[4])
      if v2:
        t = pool_func(
            t,
            ksize=ksize_placeholder,
            strides=strides_placeholder,
            padding=padding,
            data_format=data_format)
      else:
        t = pool_func(
            t,
            ksize=ksize,
            strides=strides,
            padding=padding,
            data_format=data_format)
      if data_format == "NCHW_VECT_C":
        t = gen_array_ops.dequantize(t, -128, 127)
        t = test_util.NCHW_VECT_CToNHWC(t)
      elif data_format == "NCHW":
        t = test_util.NCHWToNHWC(t)
      if v2:
        actual = t.eval(feed_dict={
            ksize_placeholder: ksize,
            strides_placeholder: strides
        })
      else:
        actual = self.evaluate(t)
        self.assertShapeEqual(actual, t)
      self.assertAllCloseAccordingToType(expected, actual.flatten())

  def _VerifyOneTest(self, pool_func, input_sizes, ksize, strides, padding,
                     data_format, expected, use_gpu, v2,
                     use_negative_input=False):
    """Verifies the output values of the pooling function.

    Args:
      pool_func: Function to be called, co.MaxPool, co.AvgPool,
        or the Lua version.
      input_sizes: Input tensor dimensions.
      ksize: The kernel size dimensions
      strides: The stride dimensions
      padding: Padding type.
      data_format: The data format we use to run the pooling operation.
      expected: An array containing the expected operation outputs.
      use_gpu: Whether we are running on GPU.
      v2: Whether to use v2 version.
      use_negative_input: If the input values should be negative."
    """
    if data_format == "NCHW_VECT_C":
      avg_pool_func = nn_ops.avg_pool
      tf_logging.info("pool_func=%s", pool_func)
      if pool_func == avg_pool_func:
        tf_logging.info("NCHW_VECT_C not yet implemented for avg_pool")
        return
      if (self._isMaxPool(pool_func) and isinstance(padding, list)):
        tf_logging.info("NCHW_VECT_C not yet implemented for max pool" +
                        " with explicit padding")
        return

    self._VerifyOneType(pool_func, input_sizes, ksize, strides, padding,
                        data_format, dtypes.float32, expected, use_gpu, v2,
                        use_negative_input)
    if not test.is_built_with_rocm():
      # double datatype is not supported for pooling ops on the ROCm platform
      self._VerifyOneType(pool_func, input_sizes, ksize, strides, padding,
                          data_format, dtypes.float64, expected, use_gpu, v2,
                          use_negative_input)

    if not use_gpu or test_util.GpuSupportsHalfMatMulAndConv():
      self._VerifyOneType(pool_func, input_sizes, ksize, strides, padding,
                          data_format, dtypes.float16, expected, use_gpu, v2,
                          use_negative_input)

  def _VerifyValues(self,
                    pool_func,
                    input_sizes,
                    ksize,
                    strides,
                    padding,
                    expected,
                    use_gpu,
                    v2=False,
                    one_dim=False,
                    use_negative_input=False):
    """Verifies the output values of the pooling function.

    Args:
      pool_func: Function to be called, co.MaxPool, co.AvgPool,
        or the Lua version.
      input_sizes: Input tensor dimensions.
      ksize: The kernel size dimensions
      strides: The stride dimensions
      padding: Padding type.
      expected: An array containing the expected operation outputs.
      use_gpu: Whether we are running on GPU.
      v2: Whether to use v2 version.
      one_dim: If one dimensional pools should be done instead of two
        dimensional pools.
      use_negative_input: If the input values should be negative.
    """
    for (data_format, use_gpu_2) in GetTestConfigs(
        include_nchw_vect_c=True, one_dimensional=one_dim):
      if use_gpu_2 == use_gpu:
        self._VerifyOneTest(pool_func, input_sizes, ksize, strides, padding,
                            data_format, expected, use_gpu, v2,
                            use_negative_input)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolValidPadding(self, **kwargs):
    self._VerifyOneType(
        input_sizes=[1, 3, 3, 3],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="VALID",
        expected=[7.0, 8.0, 9.0],
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolEmpty(self, **kwargs):
    self._VerifyOneType(
        input_sizes=[1, 3, 3, 0],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="VALID",
        expected=[],
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolSamePadding(self, **kwargs):
    self._VerifyOneType(
        input_sizes=[1, 2, 4, 3],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=[8.5, 9.5, 10.5, 14.5, 15.5, 16.5],
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolSamePaddingNonSquareWindow(self, **kwargs):
    # input is:
    # [1.0, 2.0
    #  3.0  4.0]
    #
    # Window of [x, x] should do:
    #  [avg(1.0, 2.0), avg(2.0, padded0),
    #   avg(3.0, 4.0), avg(4.0, padded0)]
    self._VerifyOneType(
        input_sizes=[1, 2, 2, 1],
        ksize=[1, 1, 2, 1],
        strides=[1, 1, 1, 1],
        padding="SAME",
        expected=[1.5, 2.0, 3.5, 4.0],
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolSamePaddingNonSquareWindow_2(self, **kwargs):
    # Window of [x,
    #            x] should do:
    #  [avg(1.0, 3.0), avg(2.0, 4.0)
    #   avg(3.0, padded0), avg(4.0, padded0)]
    self._VerifyOneType(
        input_sizes=[1, 2, 2, 1],
        ksize=[1, 2, 1, 1],
        strides=[1, 1, 1, 1],
        padding="SAME",
        expected=[2.0, 3.0, 3.0, 4.0],
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolSamePaddingNonSquareWindowMultiBatch(self, **kwargs):
    self._VerifyOneType(
        input_sizes=[2, 2, 2, 2],
        ksize=[1, 1, 2, 1],
        strides=[1, 1, 1, 1],
        padding="SAME",
        expected=[
            2.0, 3.0, 3.0, 4.0, 6.0, 7.0, 7.0, 8.0, 10.0, 11.0, 11.0, 12.0,
            14.0, 15.0, 15.0, 16.0
        ],
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolSamePaddingNonSquareWindowMultiBatch_2(self, **kwargs):
    self._VerifyOneType(
        input_sizes=[2, 2, 2, 2],
        ksize=[1, 2, 1, 1],
        strides=[1, 1, 1, 1],
        padding="SAME",
        expected=[
            3.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 14.0,
            13.0, 14.0, 15.0, 16.0
        ],
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolValidPaddingUnevenStride(self, **kwargs):
    self._VerifyOneType(
        input_sizes=[1, 3, 3, 3],
        ksize=[1, 2, 2, 1],
        strides=[1, 1, 2, 1],
        padding="VALID",
        expected=[7.0, 8.0, 9.0, 16.0, 17.0, 18.0],
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolValidPaddingUnevenStride_2(self, **kwargs):
    self._VerifyOneType(
        input_sizes=[1, 3, 3, 3],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 1, 1],
        padding="VALID",
        expected=[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolSamePadding_2(self, **kwargs):
    expected_output = [
        11.0, 12.0, 13.0, 14.0, 19.0, 20.0, 21.0, 22.0, 43.0, 44.0, 45.0, 46.0,
        51.0, 52.0, 53.0, 54.0
    ]
    self._VerifyOneType(
        input_sizes=[1, 4, 4, 4],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=expected_output,
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolSamePaddingPacket_4(self, **kwargs):
    expected_output = [
        21.0, 22.0, 23.0, 24.0, 27.0, 28.0, 29.0, 30.0, 45.0, 46.0, 47.0, 48.0,
        51.0, 52.0, 53.0, 54.0
    ]
    self._VerifyOneType(
        input_sizes=[1, 4, 4, 4],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=expected_output,
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolSamePaddingPacket_8(self, **kwargs):
    expected_output = [
        -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, 4.0, 5.0, 6.0, 7.0,
        8.0, 9.0, 10.0, 11.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
        32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, -3.5, -54.0, -53.0, -52.0,
        -51.0, -50.0, -49.0, -48.0, -47.0, -38.0, -37.0, -36.0, -35.0, -34.0,
        -33.0, -32.0, -31.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0, -16.0,
        -15.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -11.0, -10.0,
        -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
        12.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 33.0, 34.0, 35.0,
        36.0, 37.0, 38.0, -3.5, -2.5, -85.0, -84.0, -83.0, -82.0, -81.0, -80.0,
        -79.0, -78.0, -69.0, -68.0, -67.0, -66.0, -65.0, -64.0, -63.0, -62.0,
        -53.0, -52.0, -51.0, -50.0, -49.0, -48.0, -47.0, -46.0, -41.0, -40.0,
        -39.0, -38.0, -37.0, -36.0, -35.0, -34.0
    ]
    self._VerifyOneType(
        input_sizes=[1, 8, 8, 8],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=expected_output,
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolEmptyInput(self, **kwargs):
    self._VerifyOneType(
        input_sizes=[0, 8, 8, 8],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=[],
        **kwargs)

  @test_util.run_in_graph_and_eager_modes
  def testRawAvgPoolLargeKsizeRaiseError(self):
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      with self.cached_session():
        t = gen_nn_ops.avg_pool(
            value=np.ones([1, 1, 1, 1]),
            ksize=[1, 1e20, 1, 1],
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC")
        self.evaluate(t)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, gen_nn_ops.max_pool_v2))
  @test_util.run_deprecated_v1
  def testMaxPoolValidPadding(self, **kwargs):
    self._VerifyOneType(
        input_sizes=[1, 3, 3, 3],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="VALID",
        expected=[13.0, 14.0, 15.0],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, gen_nn_ops.max_pool_v2))
  @test_util.run_deprecated_v1
  def testMaxPoolSamePadding(self, **kwargs):
    self._VerifyOneType(
        input_sizes=[1, 2, 3, 3],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=[13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, nn_ops.max_pool_v2))
  @test_util.xla_allow_fallback("XLA doesn't support explicit padding")
  @test_util.run_deprecated_v1
  def testMaxPoolZeroExplicitPadding(self, **kwargs):
    self._VerifyOneType(
        input_sizes=[1, 3, 3, 1],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding=[[0, 0], [0, 0], [0, 0], [0, 0]],
        expected=[9.0],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, nn_ops.max_pool_v2))
  @test_util.xla_allow_fallback("XLA doesn't support explicit padding")
  @test_util.run_deprecated_v1
  def testMaxPoolNegativeInputExpPadding(self, **kwargs):
    self._VerifyOneType(
        input_sizes=[1, 3, 3, 1],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding=[[0, 0], [2, 1], [2, 1], [0, 0]],
        expected=[-1, -1, -1, -1],
        use_negative_input=True,
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, nn_ops.max_pool_v2))
  @test_util.xla_allow_fallback("XLA doesn't support explicit padding")
  @test_util.run_deprecated_v1
  def testMaxPoolExplicitPadding(self, **kwargs):
    expected_output = [9.0, 9.0]
    self._VerifyOneType(
        input_sizes=[1, 3, 3, 1],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding=[[0, 0], [0, 2], [0, 1], [0, 0]],
        expected=expected_output,
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, nn_ops.max_pool_v2))
  @test_util.xla_allow_fallback("XLA doesn't support explicit padding")
  @test_util.run_deprecated_v1
  def testMaxPoolExplicitPaddingAdvanced(self, **kwargs):
    expected_output = [7, 9, 11, 12, 19, 21, 23, 24, 31, 33, 35, 36, 31, 33,
                       35, 36]
    self._VerifyOneType(
        input_sizes=[1, 6, 6, 1],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding=[[0, 0], [1, 2], [2, 1], [0, 0]],
        expected=expected_output,
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, nn_ops.max_pool_v2))
  @test_util.xla_allow_fallback("XLA doesn't support explicit padding")
  @test_util.run_deprecated_v1
  def testMaxPoolNegativeInputExpPaddingAdv(self, **kwargs):
    expected_output = [-1, -1, -3, -5, -7, -7, -9, -11, -19, -19, -21, -23, -31,
                       -31, -33, -35]

    self._VerifyOneType(
        input_sizes=[1, 6, 6, 1],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding=[[0, 0], [1, 2], [2, 1], [0, 0]],
        expected=expected_output,
        use_negative_input=True,
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, nn_ops.max_pool_v2))
  @test_util.xla_allow_fallback("XLA doesn't support explicit padding")
  @test_util.run_deprecated_v1
  def testMaxPoolExplicitPadding2_(self, **kwargs):
    expected_output = [9.0, 9.0]
    self._VerifyOneType(
        input_sizes=[1, 3, 3, 1],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding=[[0, 0], [0, 2], [0, 1], [0, 0]],
        expected=expected_output,
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(
          nn_ops.max_pool1d, nn_ops.max_pool_v2, one_dimensional=True))
  @test_util.xla_allow_fallback("XLA doesn't support explicit padding")
  @test_util.run_deprecated_v1
  def testMaxPoolExplicitPadding_1D(self, **kwargs):
    self._VerifyOneType(
        input_sizes=[1, 3, 1],
        ksize=[1, 2, 1],
        strides=[1, 2, 1],
        padding=[[0, 0], [0, 1], [0, 0]],
        expected=[2.0, 3.0],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, gen_nn_ops.max_pool_v2))
  @test_util.run_deprecated_v1
  def testMaxPoolSamePaddingNonSquareWindow(self, **kwargs):
    # input is:
    # [1.0, 2.0
    #  3.0  4.0]
    #
    # Window of [x, x] should do:
    #
    #  [max(1.0, 2.0), max(2.0, padded0),
    #   max(3.0, 4.0), max(4.0, padded0)]
    self._VerifyOneType(
        input_sizes=[1, 2, 2, 1],
        ksize=[1, 1, 2, 1],
        strides=[1, 1, 1, 1],
        padding="SAME",
        expected=[2.0, 2.0, 4.0, 4.0],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, gen_nn_ops.max_pool_v2))
  @test_util.run_deprecated_v1
  def testMaxPoolValidPaddingUnevenStride(self, **kwargs):
    self._VerifyOneType(
        input_sizes=[1, 4, 4, 1],
        ksize=[1, 2, 2, 1],
        strides=[1, 1, 2, 1],
        padding="VALID",
        expected=[6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, gen_nn_ops.max_pool_v2))
  @test_util.run_deprecated_v1
  def testMaxPoolValidPaddingUnevenStride2_(self, **kwargs):
    self._VerifyOneType(
        input_sizes=[1, 4, 4, 1],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 1, 1],
        padding="VALID",
        expected=[6.0, 7.0, 8.0, 14.0, 15.0, 16.0],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, gen_nn_ops.max_pool_v2))
  @test_util.run_deprecated_v1
  def testMaxPoolSamePaddingPacket4_(self, **kwargs):
    expected_output = [
        21.0, 22.0, 23.0, 24.0, 29.0, 30.0, 31.0, 32.0, 53.0, 54.0, 55.0, 56.0,
        61.0, 62.0, 63.0, 64.0
    ]
    self._VerifyOneType(
        input_sizes=[1, 4, 4, 4],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=expected_output,
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, gen_nn_ops.max_pool_v2))
  @test_util.run_deprecated_v1
  def testMaxPoolSamePaddingPacket8_(self, **kwargs):
    expected_output = [
        81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 97.0, 98.0, 99.0, 100.0,
        101.0, 102.0, 103.0, 104.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0,
        119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 120.0,
        18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 34.0, 35.0, 36.0, 37.0,
        38.0, 39.0, 40.0, 41.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0,
        58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 82.0, 83.0, 84.0, 85.0,
        86.0, 87.0, 88.0, 89.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0,
        105.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0,
        123.0, 124.0, 125.0, 126.0, 127.0, 120.0, 121.0, -45.0, -44.0, -43.0,
        -42.0, -41.0, -40.0, -39.0, -38.0, -29.0, -28.0, -27.0, -26.0, -25.0,
        -24.0, -23.0, -22.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0,
        -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0
    ]
    self._VerifyOneType(
        input_sizes=[1, 8, 8, 8],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=expected_output,
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, gen_nn_ops.max_pool_v2))
  @test_util.run_deprecated_v1
  def testMaxPoolEmptyInput(self, **kwargs):
    self._VerifyOneType(
        input_sizes=[0, 8, 8, 8],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=[],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, gen_nn_ops.max_pool_v2))
  @test_util.run_deprecated_v1
  def testMaxPoolInvalidFilterSize(self, **kwargs):
    with self.cached_session(use_gpu=test.is_gpu_available()):
      t = constant_op.constant(1.0, shape=[1, 1, 1, 1])
      with self.assertRaisesRegex(
          (errors_impl.InvalidArgumentError, ValueError),
          "Negative dimension size"):
        t = self.evaluate(
            nn_ops.max_pool(t, ksize=[1, 1, 2, 1], strides=1, padding="VALID"))

  # Tests for DepthwiseMaxPooling on CPU only.
  @parameterized.parameters(
      GetTestConfigsDicts(
          nn_ops.max_pool, gen_nn_ops.max_pool_v2, allow_gpu=False))
  @test_util.run_deprecated_v1
  def testDepthwiseMaxPool1x1DepthWindow(self, **kwargs):
    # input is:
    # [1.0, ..., 10.0] along depth,
    #
    # We maxpool by depth in patches of 2.
    self._VerifyOneType(
        input_sizes=[1, 1, 1, 10],
        ksize=[1, 1, 1, 2],
        strides=[1, 1, 1, 2],
        padding="SAME",
        expected=[2.0, 4.0, 6.0, 8.0, 10.0],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(
          nn_ops.max_pool, gen_nn_ops.max_pool_v2, allow_gpu=False))
  @test_util.run_deprecated_v1
  def testDepthwiseMaxPool2x2DepthWindow(self, **kwargs):
    # input is:
    #
    # a 2x2x6 cube, and we depthwise max across 3 to produce a 2x2x2
    # output.  Each node has contiguous values, so the depthwise max
    # should be multiples of 3.0.
    self._VerifyOneType(
        input_sizes=[1, 2, 2, 6],
        ksize=[1, 1, 1, 3],
        strides=[1, 1, 1, 3],
        padding="SAME",
        expected=[3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(
          nn_ops.max_pool, gen_nn_ops.max_pool_v2, allow_gpu=False))
  @test_util.run_deprecated_v1
  def testMaxPoolKernelSmallerThanStrideValid(self, **kwargs):
    self._VerifyOneType(
        input_sizes=[1, 7, 7, 1],
        ksize=[1, 2, 2, 1],
        strides=[1, 3, 3, 1],
        padding="VALID",
        expected=[9, 12, 30, 33],
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolKernelSmallerThanStride(self, **kwargs):
    self._VerifyOneType(
        input_sizes=[1, 7, 7, 1],
        ksize=[1, 2, 2, 1],
        strides=[1, 3, 3, 1],
        padding="VALID",
        expected=[5, 8, 26, 29],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, gen_nn_ops.max_pool_v2) +
      GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testKernelSmallerThanStrideSame1_(self, **kwargs):
    self._VerifyOneType(
        input_sizes=[1, 3, 3, 1],
        ksize=[1, 1, 1, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=[1, 3, 7, 9],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, gen_nn_ops.max_pool_v2) +
      GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testKernelSmallerThanStrideSame2_(self, **kwargs):
    self._VerifyOneType(
        input_sizes=[1, 4, 4, 1],
        ksize=[1, 1, 1, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=[1, 3, 9, 11],
        **kwargs)

  def _testDepthwiseMaxPoolInvalidConfig(self,
                                         in_size,
                                         ksize,
                                         strides,
                                         error_msg,
                                         use_gpu=False):
    with self.cached_session(use_gpu=use_gpu):
      t = constant_op.constant(1.0, shape=in_size)
      with self.assertRaisesRegex(errors_impl.UnimplementedError, error_msg):
        t = nn_ops.max_pool(
            t, ksize=ksize, strides=strides, padding="SAME").eval()

  @test_util.disable_xla("b/123338077")  # Passes with XLA
  def testDepthwiseMaxPoolInvalidConfigs(self):
    self._testDepthwiseMaxPoolInvalidConfig(
        [1, 2, 2, 4], [1, 2, 2, 2], [1, 1, 1, 2],
        "exactly one of pooling across depth")
    self._testDepthwiseMaxPoolInvalidConfig(
        [1, 2, 2, 4], [1, 1, 1, 2], [1, 1, 1, 1],
        "depth window to equal the depth stride")
    self._testDepthwiseMaxPoolInvalidConfig([1, 2, 2, 4], [1, 1, 1, 3],
                                            [1, 1, 1, 3], "evenly divide")
    if test.is_gpu_available():
      with self.session():
        t = variables.Variable(np.ones([1, 2, 2, 4]))
        self.evaluate(variables.global_variables_initializer())
        with self.assertRaisesOpError("for CPU devices"):
          nn_ops.max_pool(
              t, ksize=[1, 1, 1, 2], strides=[1, 1, 1, 2],
              padding="SAME").eval()

  # The following are tests that verify that the CPU and GPU implementations
  # produce the same results.
  def _CompareMaxPoolingFwd(self, input_shape, ksize, strides, padding):
    # double datatype is currently not supported for pooling ops
    # on the ROCm platform
    for dtype in [np.float32, np.float16] \
        + [np.float64] if not test.is_built_with_rocm() else []:
      tensor_input = np.random.rand(*input_shape).astype(dtype)
      with self.cached_session():
        t = constant_op.constant(tensor_input, shape=input_shape)
        out_op, _ = nn_ops.max_pool_with_argmax(t, ksize, strides, padding)
        gpu_val = self.evaluate(out_op)
      with self.cached_session(use_gpu=False):
        t = constant_op.constant(tensor_input, shape=input_shape)
        out_op = nn_ops.max_pool(t, ksize, strides, padding)
        cpu_val = self.evaluate(out_op)
      self.assertAllCloseAccordingToType(cpu_val, gpu_val)

  def _CompareMaxPoolingBk(self, input_shape, output_shape, ksize, strides,
                           padding):
    # double datatype is currently not supported for pooling ops
    # on the ROCm platform
    for dtype in [np.float32, np.float16] \
        + [np.float64] if not test.is_built_with_rocm() else []:
      # Generate numbers in a narrow range, so that there are many duplicates
      # in the input.
      tensor_input = np.random.random_integers(0, 3, input_shape).astype(dtype)
      tensor_output = np.random.rand(*output_shape).astype(dtype)
      with self.cached_session():
        t = constant_op.constant(tensor_input, shape=input_shape)
        _, argmax_op = nn_ops.max_pool_with_argmax(t, ksize, strides, padding)
        argmax = self.evaluate(argmax_op)
        grad_in = constant_op.constant(tensor_output, shape=output_shape)
        out_op = gen_nn_ops.max_pool_grad_with_argmax(t, grad_in, argmax, ksize,
                                                      strides, padding)
        gpu_val = self.evaluate(out_op)
        self.assertShapeEqual(gpu_val, out_op)
      with self.cached_session(use_gpu=False):
        t = constant_op.constant(tensor_input, shape=input_shape)
        out_op = nn_ops.max_pool(t, ksize, strides, padding)
        orig_out = self.evaluate(out_op)
        grad_in = constant_op.constant(tensor_output, shape=output_shape)
        out_op = gen_nn_ops.max_pool_grad(t, orig_out, grad_in, ksize, strides,
                                          padding)
        cpu_val = self.evaluate(out_op)
        self.assertShapeEqual(cpu_val, out_op)
      # The CPU version accumulates its gradient on fp16, so it's less
      # accurate than the GPU version that does the accumulation on fp32
      self.assertAllCloseAccordingToType(
          cpu_val, gpu_val, half_rtol=0.01, half_atol=0.01)

  def _CompareMaxPoolingGradBk(self, input_shape, output_shape, ksize, strides,
                               padding):
    # double datatype is currently not supported for pooling ops
    # on the ROCm platform
    for dtype in [np.float32, np.float16] \
        + [np.float64] if not test.is_built_with_rocm() else []:
      # Generate numbers in a narrow range, so that there are many duplicates
      # in the input.
      tensor_input = np.random.random_integers(0, 3, input_shape).astype(dtype)
      with self.cached_session(use_gpu=False):
        t = constant_op.constant(tensor_input, shape=input_shape)
        _, argmax_op = nn_ops.max_pool_with_argmax(t, ksize, strides, padding)
        argmax = self.evaluate(argmax_op)
        grad_in = constant_op.constant(tensor_input, shape=input_shape)
        out_op = gen_nn_ops.max_pool_grad_grad_with_argmax(
            t, grad_in, argmax, ksize, strides, padding)
        gpu_val = self.evaluate(out_op)
        self.assertShapeEqual(gpu_val, out_op)
      with self.cached_session(use_gpu=False):
        t = constant_op.constant(tensor_input, shape=input_shape)
        out_op = nn_ops.max_pool(t, ksize, strides, padding)
        orig_out = self.evaluate(out_op)
        grad_in = constant_op.constant(tensor_input, shape=input_shape)
        out_op = gen_nn_ops.max_pool_grad_grad(t, orig_out, grad_in, ksize,
                                               strides, padding)
        cpu_val = self.evaluate(out_op)
        self.assertShapeEqual(cpu_val, out_op)
      # The CPU version accumulates its gradient on fp16, so it's less
      # accurate than the GPU version that does the accumulation on fp32
      self.assertAllCloseAccordingToType(
          cpu_val, gpu_val, half_rtol=0.01, half_atol=0.01)

  def testMaxPoolingWithArgmax(self):
    tensor_input = [
        1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0
    ]

    Config = collections.namedtuple(
        "Config", ["use_gpu", "include_batch_in_index", "argmax", "Targmax"])
    configs = [
        Config(False, False, [0, 1, 3, 5, 0, 2, 6, 8], dtypes.int64),
        Config(False, True, [0, 1, 3, 5, 9, 11, 15, 17], dtypes.int64),
        Config(False, False, [0, 1, 3, 5, 0, 2, 6, 8], dtypes.int32),
        Config(False, True, [0, 1, 3, 5, 9, 11, 15, 17], dtypes.int32),
        Config(True, False, [0, 1, 3, 5, 0, 2, 6, 8], dtypes.int64),
        Config(True, True, [0, 1, 3, 5, 9, 11, 15, 17], dtypes.int64),
    ]

    for config in configs:
      with GetDeviceScope(self, use_gpu=config.use_gpu):
        t = constant_op.constant(tensor_input, shape=[2, 3, 3, 1])
        out_op, argmax_op = nn_ops.max_pool_with_argmax(
            t,
            ksize=[1, 2, 2, 1],
            strides=[1, 1, 1, 1],
            Targmax=config.Targmax,
            padding="VALID",
            include_batch_in_index=config.include_batch_in_index)
        out, argmax = self.evaluate([out_op, argmax_op])
        self.assertShapeEqual(out, out_op)
        self.assertShapeEqual(argmax, argmax_op)
        self.assertAllClose(out.ravel(),
                            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertAllEqual(argmax.ravel(), config.argmax)

  def testMaxPoolingGradWithArgmax(self):
    orig_input = [
        1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0
    ]
    tensor_input = [11.0, 12.0, 13.0, 14.0, 21.0, 22.0, 23.0, 24.0]

    Config = collections.namedtuple(
        "Config", ["use_gpu", "include_batch_in_index", "argmax"])
    configs = [
        Config(False, False, [0, 1, 3, 5, 0, 2, 6, 8]),
        Config(False, True, [0, 1, 3, 5, 9, 11, 15, 17]),
        Config(True, False, [0, 1, 3, 5, 0, 2, 6, 8]),
        Config(True, True, [0, 1, 3, 5, 9, 11, 15, 17])
    ]

    for config in configs:
      with GetDeviceScope(self, config.use_gpu):
        orig_in = constant_op.constant(orig_input, shape=[2, 3, 3, 1])
        t = constant_op.constant(tensor_input, shape=[2, 2, 2, 1])
        argmax_t = constant_op.constant(
            config.argmax, shape=[2, 2, 2, 1], dtype=dtypes.int64)
        out_op = gen_nn_ops.max_pool_grad_with_argmax(
            orig_in,
            t,
            argmax_t,
            ksize=[1, 2, 2, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
            include_batch_in_index=config.include_batch_in_index)
        out = self.evaluate(out_op).flatten()
        self.assertAllClose(out, [
            11.0, 12.0, 0.0, 13.0, 0.0, 14.0, 0.0, 0.0, 0.0, 21.0, 0.0, 22.0,
            0.0, 0.0, 0.0, 23.0, 0.0, 24.0
        ])

  def testMaxPoolingGradThrowDeterminismError(self):
    if test.is_gpu_available(cuda_only=True):
      try:
        config_exec.enable_op_determinism()
        orig_input = [
            1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 1.0
        ]
        tensor_input = [11.0, 12.0, 13.0, 14.0, 21.0, 22.0, 23.0, 24.0]

        with GetDeviceScope(self, True):
          orig_in = constant_op.constant(orig_input, shape=[2, 3, 3, 1])
          t = constant_op.constant(tensor_input, shape=[2, 2, 2, 1])
          argmax_t = constant_op.constant(
              [0, 1, 3, 5, 0, 2, 6, 8], shape=[2, 2, 2, 1], dtype=dtypes.int64)
          with self.assertRaisesRegexp(
              errors_impl.UnimplementedError, "Determinism is not yet supported "
              "for MaxPoolGradWithArgmax."):
            out_op = gen_nn_ops.max_pool_grad_with_argmax(
                orig_in,
                t,
                argmax_t,
                ksize=[1, 2, 2, 1],
                strides=[1, 1, 1, 1],
                padding="VALID",
                include_batch_in_index=False)
            self.evaluate(out_op)
      finally:
        config_exec.disable_op_determinism()
    else:
      try:
        config_exec.enable_op_determinism()
        orig_input = [
            1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 1.0
        ]
        tensor_input = [11.0, 12.0, 13.0, 14.0, 21.0, 22.0, 23.0, 24.0]

        with GetDeviceScope(self, False):
          orig_in = constant_op.constant(orig_input, shape=[2, 3, 3, 1])
          t = constant_op.constant(tensor_input, shape=[2, 2, 2, 1])
          argmax_t = constant_op.constant(
              [0, 1, 3, 5, 0, 2, 6, 8], shape=[2, 2, 2, 1], dtype=dtypes.int64)
          out_op = gen_nn_ops.max_pool_grad_with_argmax(
              orig_in,
              t,
              argmax_t,
              ksize=[1, 2, 2, 1],
              strides=[1, 1, 1, 1],
              padding="VALID",
              include_batch_in_index=False)
          self.evaluate(out_op)
      finally:
        config_exec.disable_op_determinism()

  def testMaxPoolingGradGradWithArgmax(self):
    # MaxPoolWithArgMax is implemented only on CUDA.
    if not test.is_gpu_available(cuda_only=True):
      return
    orig_input = [
        1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0
    ]
    tensor_input = [
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 21.0, 22.0, 23.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0
    ]

    Config = collections.namedtuple(
        "Config", ["use_gpu", "include_batch_in_index", "argmax"])
    configs = [
        Config(True, False, [0, 1, 3, 5, 0, 2, 6, 8]),
        Config(True, True, [0, 1, 3, 5, 9, 11, 15, 17])
    ]

    for config in configs:
      with GetDeviceScope(self, config.use_gpu):
        orig_in = constant_op.constant(orig_input, shape=[2, 3, 3, 1])
        t = constant_op.constant(tensor_input, shape=[2, 3, 3, 1])
        argmax_t = constant_op.constant(
            config.argmax, shape=[2, 2, 2, 1], dtype=dtypes.int64)
        out_op = gen_nn_ops.max_pool_grad_grad_with_argmax(
            orig_in,
            t,
            argmax_t,
            ksize=[1, 2, 2, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
            include_batch_in_index=config.include_batch_in_index)
        out = self.evaluate(out_op).flatten()
        self.assertAllClose(out,
                            [11.0, 12.0, 14.0, 16.0, 21.0, 23.0, 27.0, 29.0])

  def _ConstructAndTestGradient(self,
                                pool_func,
                                input_sizes,
                                output_sizes,
                                window_rows,
                                window_cols,
                                row_stride,
                                col_stride,
                                padding,
                                data_format,
                                use_gpu,
                                x_init_value=None):
    """Verifies the gradients of the max or avg pooling function.

    Args:
      pool_func: Function to be called, co.MaxPool, co.AvgPool,
        or the Lua version.
      input_sizes: Input tensor dimensions.
      output_sizes: Output tensor dimensions.
      window_rows: kernel size in row dim
      window_cols: kernel size in col dim
      row_stride: Row Stride.
      col_stride: Col Stride.
      padding: Padding type.
      data_format: Data format.
      use_gpu: whether we are running on GPU
      x_init_value: Values to be passed to the gradient checker.
    """
    assert input_sizes[0] == output_sizes[0]
    assert input_sizes[3] == output_sizes[3]
    total_size = 1
    for s in input_sizes:
      total_size *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x = [f * 1.0 for f in range(1, total_size + 1)]
    with self.cached_session(use_gpu=use_gpu):
      input_tensor = constant_op.constant(x, shape=input_sizes, name="input")
      if pool_func == nn_ops.avg_pool:
        func_name = "avg_pool"
        err_tolerance = 1e-4
      else:
        if x_init_value is None:
          x_init_value = np.asfarray(
              np.arange(1, total_size + 1),
              dtype=np.float32).reshape(input_sizes)
        func_name = "max_pool"
        err_tolerance = 1e-3
      if data_format == "NCHW":
        ksize = [1, 1, window_rows, window_cols]
        strides = [1, 1, row_stride, col_stride]
        if isinstance(padding, list):
          padding = test_util.NHWCToNCHW(padding)
        t = test_util.NHWCToNCHW(input_tensor)
      else:
        ksize = [1, window_rows, window_cols, 1]
        strides = [1, row_stride, col_stride, 1]
        t = input_tensor
      t = pool_func(
          t,
          ksize=ksize,
          strides=strides,
          padding=padding,
          data_format=data_format,
          name=func_name)
      if data_format == "NCHW":
        t = test_util.NCHWToNHWC(t)

      err = gradient_checker.compute_gradient_error(
          input_tensor,
          input_sizes,
          t,
          output_sizes,
          x_init_value=x_init_value,
          delta=1e-2)
    tf_logging.info("%s gradient error = %.4f" % (func_name, err))
    self.assertLess(err, err_tolerance)

  def _ConstructAndTestSecondGradient(self,
                                      pool_func,
                                      input_sizes,
                                      output_sizes,
                                      window_rows,
                                      window_cols,
                                      row_stride,
                                      col_stride,
                                      padding,
                                      data_format,
                                      use_gpu,
                                      x_init_value=None):
    """Verifies the second-order gradients of the pooling function.

    Args:
      pool_func: Function to be called, co.MaxPool, co.AvgPool,
        or the Lua version.
      input_sizes: Input tensor dimensions.
      output_sizes: Output tensor dimensions.
      window_rows: kernel size in row dim
      window_cols: kernel size in col dim
      row_stride: Row Stride.
      col_stride: Col Stride.
      padding: Padding type.
      data_format: Data format.
      use_gpu: whether we are running on GPU
      x_init_value: Values to be passed to the gradient checker.
    """
    assert input_sizes[0] == output_sizes[0]
    assert input_sizes[3] == output_sizes[3]
    total_size = 1
    for s in input_sizes:
      total_size *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x = [f * 1.0 for f in range(1, total_size + 1)]
    with self.cached_session(use_gpu=use_gpu):
      input_tensor = constant_op.constant(x, shape=input_sizes, name="input")
      if pool_func == nn_ops.avg_pool:
        func_name = "avg_pool"
        err_tolerance = 1e-3
      else:
        if x_init_value is None:
          x_init_value = np.asfarray(
              np.arange(1, total_size + 1),
              dtype=np.float32).reshape(input_sizes)
        func_name = "max_pool"
        err_tolerance = 1e-2
      if data_format == "NCHW":
        ksize = [1, 1, window_rows, window_rows]
        strides = [1, 1, row_stride, col_stride]
        t = test_util.NHWCToNCHW(input_tensor)
      else:
        ksize = [1, window_rows, window_rows, 1]
        strides = [1, row_stride, col_stride, 1]
        t = input_tensor
      t = pool_func(
          t,
          ksize=ksize,
          strides=strides,
          padding=padding,
          data_format=data_format,
          name=func_name)
      if data_format == "NCHW":
        t = test_util.NHWCToNCHW(t)

      t_g = gradients_impl.gradients(t**2, input_tensor)[0]
      err = gradient_checker.compute_gradient_error(
          input_tensor,
          input_sizes,
          t_g,
          input_sizes,
          x_init_value=x_init_value,
          delta=1e-2)
    tf_logging.info("%s second-order gradient error = %.4f" % (func_name, err))
    self.assertLess(err, err_tolerance)

  def _testMaxPoolGradValidPadding1_1(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[1, 3, 3, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=1,
          window_cols=1,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradValidPadding2_1_6(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[2, 6, 6, 3],
          output_sizes=[2, 5, 5, 3],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradValidPadding2_1_7(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[2, 7, 7, 3],
          output_sizes=[2, 6, 6, 3],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradValidPadding1_2(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[1, 3, 3, 1],
          output_sizes=[1, 2, 2, 1],
          window_rows=1,
          window_cols=1,
          row_stride=2,
          col_stride=2,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradValidPadding2_2(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[2, 2, 2, 3],
          output_sizes=[2, 1, 1, 3],
          window_rows=2,
          window_cols=2,
          row_stride=2,
          col_stride=2,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradSamePadding1_1(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[2, 2, 4, 3],
          output_sizes=[2, 2, 4, 3],
          window_rows=1,
          window_cols=1,
          row_stride=1,
          col_stride=1,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradSamePadding1_2(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[2, 2, 4, 3],
          output_sizes=[2, 1, 2, 3],
          window_rows=1,
          window_cols=1,
          row_stride=2,
          col_stride=2,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradSamePadding2_1(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[2, 2, 4, 3],
          output_sizes=[2, 2, 4, 3],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradSamePadding2_2(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[2, 2, 4, 3],
          output_sizes=[2, 1, 2, 3],
          window_rows=2,
          window_cols=2,
          row_stride=2,
          col_stride=2,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradSamePadding3_1(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[1, 7, 7, 1],
          output_sizes=[1, 7, 7, 1],
          window_rows=3,
          window_cols=3,
          row_stride=1,
          col_stride=1,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolExplicitPadding_1(self, data_format, use_gpu):
    for pool_func in [nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[1, 7, 7, 1],
          output_sizes=[1, 7, 7, 1],
          window_rows=3,
          window_cols=3,
          row_stride=1,
          col_stride=1,
          padding=[[0, 0], [1, 1], [1, 1], [0, 0]],
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolExplicitPadding_2(self, data_format, use_gpu):
    for pool_func in [nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[1, 7, 7, 1],
          output_sizes=[1, 6, 8, 1],
          window_rows=3,
          window_cols=5,
          row_stride=1,
          col_stride=1,
          padding=[[0, 0], [0, 1], [2, 3], [0, 0]],
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolExplicitPaddingLeftGreater(self, data_format, use_gpu):
    for pool_func in [nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[1, 7, 7, 1],
          output_sizes=[1, 6, 8, 1],
          window_rows=3,
          window_cols=5,
          row_stride=1,
          col_stride=1,
          padding=[[0, 0], [0, 1], [3, 2], [0, 0]],
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolExplicitPaddingBatchChannel(self, data_format, use_gpu):
    for pool_func in [nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[4, 7, 7, 3],
          output_sizes=[4, 6, 8, 3],
          window_rows=3,
          window_cols=5,
          row_stride=1,
          col_stride=1,
          padding=[[0, 0], [0, 1], [3, 2], [0, 0]],
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolExplicitPaddingStrides(self, data_format, use_gpu):
    for pool_func in [nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[1, 7, 7, 1],
          output_sizes=[1, 4, 3, 1],
          window_rows=3,
          window_cols=3,
          row_stride=2,
          col_stride=3,
          padding=[[0, 0], [1, 1], [1, 1], [0, 0]],
          data_format=data_format,
          use_gpu=use_gpu)

  @test_util.run_deprecated_v1
  def testMaxPoolGrad(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self._testMaxPoolGradValidPadding1_1(data_format, use_gpu)
      self._testMaxPoolGradValidPadding1_2(data_format, use_gpu)
      self._testMaxPoolGradValidPadding2_1_6(data_format, use_gpu)
      self._testMaxPoolGradValidPadding2_1_7(data_format, use_gpu)
      self._testMaxPoolGradValidPadding2_2(data_format, use_gpu)
      self._testMaxPoolGradSamePadding1_1(data_format, use_gpu)
      self._testMaxPoolGradSamePadding1_2(data_format, use_gpu)
      self._testMaxPoolGradSamePadding2_1(data_format, use_gpu)
      self._testMaxPoolGradSamePadding2_2(data_format, use_gpu)
      self._testMaxPoolGradSamePadding3_1(data_format, use_gpu)
      self._testMaxPoolExplicitPadding_1(data_format, use_gpu)
      self._testMaxPoolExplicitPadding_2(data_format, use_gpu)
      self._testMaxPoolExplicitPaddingStrides(data_format, use_gpu)
      self._testMaxPoolExplicitPaddingLeftGreater(data_format, use_gpu)
      self._testMaxPoolExplicitPaddingBatchChannel(data_format, use_gpu)

  def _MaxPoolGrad(self, orig_input, orig_output, grad, window_rows,
                   window_cols, row_stride, col_stride, padding, v2):
    """Max Pooling Gradient.

    Args:
      orig_input: A float Tensor. The original input tensor.
      orig_output: A float Tensor. The original output tensor.
      grad: A float Tensor.
        The 4D (batch x rows x cols x depth) output backprop.
      window_rows: integer. Kernel size along rows dimension.
      window_cols: integer. Kernel size along cols dimension.
      row_stride: integer. Stride along rows dimension
      col_stride: integer. Stride along cols dimension
      padding: PoolingOpDef.Padding.  Padding type.

    Returns:
      A Tensor.
    """
    pool_func = gen_nn_ops.max_pool_grad_v2 if v2 else gen_nn_ops.max_pool_grad
    if v2:
      return pool_func(orig_input, orig_output, grad,
                       [1, window_rows, window_cols, 1],
                       [1, row_stride, col_stride, 1], padding)
    else:
      padding, explicit_paddings = nn_ops.convert_padding(padding)
      return pool_func(orig_input, orig_output, grad,
                       [1, window_rows, window_cols, 1],
                       [1, row_stride, col_stride, 1], padding,
                       explicit_paddings)

  def _testMaxPoolGradDirect(self, input_data, output_backprop,
                             expected_input_backprop, input_sizes, output_sizes,
                             window_rows, window_cols, row_stride, col_stride,
                             padding, use_gpu, v2):
    pool_func = gen_nn_ops.max_pool_v2 if v2 else nn_ops.max_pool
    with self.cached_session(use_gpu=use_gpu):
      input_tensor = variables.Variable(
          np.array(input_data, dtype=np.float32).reshape(input_sizes))
      self.evaluate(variables.global_variables_initializer())
      output_tensor = pool_func(input_tensor, [1, window_rows, window_cols, 1],
                                [1, row_stride, col_stride, 1], padding)
      output_backprop_tensor = constant_op.constant(
          output_backprop, shape=output_sizes)

      input_backprop_tensor = self._MaxPoolGrad(
          input_tensor, output_tensor, output_backprop_tensor, window_rows,
          window_cols, row_stride, col_stride, padding, v2)

      actual_input_backprop = self.evaluate(input_backprop_tensor)
      self.assertShapeEqual(actual_input_backprop, input_backprop_tensor)
      actual_input_backprop = actual_input_backprop.flatten()
      actual_input_backprop = self._GetNdArray(actual_input_backprop)

      actual_output = self.evaluate(output_tensor).flatten()
      actual_output = self._GetNdArray(actual_output)

      self.assertAllClose(
          expected_input_backprop, actual_input_backprop, rtol=1e-6, atol=1e-6)

  def _testMaxPoolGradDirect1_1(self):
    input_data = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0
    ]
    output_backprop = [11.0, 12.0, 13.0, 15.0, 16.0, 17.0, 19.0, 20.0, 21.0]
    expected_input_backprop = [
        11.0, 12.0, 13.0, 0.0, 15.0, 16.0, 17.0, 0.0, 19.0, 20.0, 21.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    ]

    for use_gpu in True, False:
      for v2 in [True, False]:
        self._testMaxPoolGradDirect(
            input_data,
            output_backprop,
            expected_input_backprop,
            input_sizes=[1, 4, 4, 1],
            output_sizes=[1, 3, 3, 1],
            window_rows=2,
            window_cols=2,
            row_stride=1,
            col_stride=1,
            padding="VALID",
            use_gpu=use_gpu,
            v2=v2)

  def _testMaxPoolGradDirect1_2(self):
    input_data = [
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        0.0, 1.0
    ]
    output_backprop = [11.0, 12.0, 13.0, 15.0, 16.0, 17.0, 19.0, 20.0, 21.0]
    expected_input_backprop = [
        11.0, 0.0, 25.0, 0.0, 0.0, 31.0, 0.0, 17.0, 19.0, 0.0, 41.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ]

    for use_gpu in True, False:
      for v2 in [True, False]:
        self._testMaxPoolGradDirect(
            input_data,
            output_backprop,
            expected_input_backprop,
            input_sizes=[1, 4, 4, 1],
            output_sizes=[1, 3, 3, 1],
            window_rows=2,
            window_cols=2,
            row_stride=1,
            col_stride=1,
            padding="VALID",
            use_gpu=use_gpu,
            v2=v2)

  def _testMaxPoolGradDirect1_3(self):
    input_data = [
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
    ]
    output_backprop = [
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
        23.0, 24.0, 25.0, 26.0
    ]
    expected_input_backprop = [
        54,
        0.0,
        62,
        0.0,
        0.0,
        60,
        0.0,
        22.0,
        47,
        0.0,
        51,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    for use_gpu in True, False:
      for v2 in [True, False]:
        self._testMaxPoolGradDirect(
            input_data,
            output_backprop,
            expected_input_backprop,
            input_sizes=[1, 4, 4, 1],
            output_sizes=[1, 4, 4, 1],
            window_rows=3,
            window_cols=3,
            row_stride=1,
            col_stride=1,
            padding="SAME",
            use_gpu=use_gpu,
            v2=v2)

  def _testMaxPoolGradZeroExplicitPadding(self):
    input_data = [
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        0.0, 1.0
    ]
    output_backprop = [11.0, 12.0, 13.0, 15.0, 16.0, 17.0, 19.0, 20.0, 21.0]
    expected_input_backprop = [
        11.0, 0.0, 25.0, 0.0, 0.0, 31.0, 0.0, 17.0, 19.0, 0.0, 41.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ]

    for use_gpu in True, False:
      for v2 in [False]:
        self._testMaxPoolGradDirect(
            input_data,
            output_backprop,
            expected_input_backprop,
            input_sizes=[1, 4, 4, 1],
            output_sizes=[1, 3, 3, 1],
            window_rows=2,
            window_cols=2,
            row_stride=1,
            col_stride=1,
            padding=[[0, 0], [0, 0], [0, 0], [0, 0]],
            use_gpu=use_gpu,
            v2=v2)

  def _testMaxPoolGradExplicitPadding_1(self):
    input_data = [
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        0.0, 1.0
    ]
    output_backprop = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
                       20.0, 21.0, 22.0]
    expected_input_backprop = [
        11.0, 0.0, 25.0, 0.0, 0.0, 31.0, 0.0, 49.0, 19.0, 0.0, 41.0, 0.0, 0.0,
        0.0, 0.0, 22.0
    ]

    for use_gpu in True, False:
      for v2 in [False]:
        self._testMaxPoolGradDirect(
            input_data,
            output_backprop,
            expected_input_backprop,
            input_sizes=[1, 4, 4, 1],
            output_sizes=[1, 3, 4, 1],
            window_rows=2,
            window_cols=2,
            row_stride=1,
            col_stride=1,
            padding=[[0, 0], [0, 0], [0, 1], [0, 0]],
            use_gpu=use_gpu,
            v2=v2)

  def _testMaxPoolGradExplicitPadding_2(self):
    input_data = [
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        0.0, 1.0
    ]
    output_backprop = [11.0, 12.0, 13.0, 15.0, 16.0, 17.0, 19.0, 20.0, 21.0]
    expected_input_backprop = [
        54.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 39.0, 0.0, 21.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ]

    for use_gpu in True, False:
      for v2 in [False]:
        self._testMaxPoolGradDirect(
            input_data,
            output_backprop,
            expected_input_backprop,
            input_sizes=[1, 4, 4, 1],
            output_sizes=[1, 3, 3, 1],
            window_rows=3,
            window_cols=3,
            row_stride=2,
            col_stride=2,
            padding=[[0, 0], [2, 1], [2, 1], [0, 0]],
            use_gpu=use_gpu,
            v2=v2)

  def _testMaxPoolGradExplicitPadding_3(self):
    input_data = [
        -1.0, -5.0, -1.0, -5.0, -5.0, -1.0, -5.0, -1.0, -1.0, -5.0, -1.0, -5.0,
        -5.0, -1.0, -5.0, -1.0
    ]
    output_backprop = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
                       20.0, 21.0, 22.0]
    expected_input_backprop = [
        11.0, 0.0, 25.0, 0.0, 0.0, 31.0, 0.0, 49.0, 19.0, 0.0, 41.0, 0.0, 0.0,
        0.0, 0.0, 22.0
    ]

    for use_gpu in True, False:
      for v2 in [False]:
        self._testMaxPoolGradDirect(
            input_data,
            output_backprop,
            expected_input_backprop,
            input_sizes=[1, 4, 4, 1],
            output_sizes=[1, 3, 4, 1],
            window_rows=2,
            window_cols=2,
            row_stride=1,
            col_stride=1,
            padding=[[0, 0], [0, 0], [0, 1], [0, 0]],
            use_gpu=use_gpu,
            v2=v2)

  @test_util.no_xla_auto_jit("b/123923733")  # NaNs handled differently
  def _testMaxPoolGradDirectWithNans2_1(self):
    input_data = [float("nan")] * 16
    output_backprop = [11.0, 12.0, 13.0, 15.0, 16.0, 17.0, 19.0, 20.0, 21.0]
    # Test the CPU implementation, which propagates diffs in case of NaN
    expected_input_backprop_tf_cpu = [
        11.0, 12.0, 13.0, 0.0, 15.0, 16.0, 17.0, 0.0, 19.0, 20.0, 21.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    ]
    for v2 in [True, False]:
      self._testMaxPoolGradDirect(
          input_data,
          output_backprop,
          expected_input_backprop_tf_cpu,
          input_sizes=[1, 4, 4, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          use_gpu=False,
          v2=v2)

    if not test.is_gpu_available():
      return

    # The functionality associated with TF_ENABLE_NANPROP is currently
    # not supported on the ROCm platform, so skip this part of the test
    # NANs in input lead to non-deterministic results, and hence skipping
    # the remaining tests altogether on the ROCm platform
    if test.is_built_with_rocm():
      return

    # Test the GPU implementation that uses cudnn for now.
    saved_nanprop = os.environ.get("TF_ENABLE_MAXPOOL_NANPROP")
    # Do not propagate the diff in cases of NaNs
    os.environ["TF_ENABLE_MAXPOOL_NANPROP"] = "0"
    expected_input_backprop_cudnn = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0
    ]

    for v2 in [True, False]:
      self._testMaxPoolGradDirect(
          input_data,
          output_backprop,
          expected_input_backprop_cudnn,
          input_sizes=[1, 4, 4, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          use_gpu=True,
          v2=v2)

    # Propagate the diff in cases of NaNs
    os.environ["TF_ENABLE_MAXPOOL_NANPROP"] = "1"
    expected_input_backprop_cudnn = expected_input_backprop_tf_cpu

    for v2 in [True, False]:
      self._testMaxPoolGradDirect(
          input_data,
          output_backprop,
          expected_input_backprop_cudnn,
          input_sizes=[1, 4, 4, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          use_gpu=True,
          v2=v2)

    if saved_nanprop:
      os.environ["TF_ENABLE_MAXPOOL_NANPROP"] = saved_nanprop
    else:
      del os.environ["TF_ENABLE_MAXPOOL_NANPROP"]

  @test_util.no_xla_auto_jit("b/123923733")  # NaNs handled differently
  def _testMaxPoolGradDirectWithNans2_2(self):
    input_data = [float("nan")] * 16
    output_backprop = [
        float("nan"), 12.0, 13.0, 15.0,
        float("nan"), 17.0, 19.0, 20.0,
        float("nan")
    ]
    # Test the CPU implementation, which propagates diffs in case of NaN
    expected_input_backprop_tf_cpu = [
        float("nan"), 12.0, 13.0, 0.0, 15.0,
        float("nan"), 17.0, 0.0, 19.0, 20.0,
        float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    for v2 in [True, False]:
      self._testMaxPoolGradDirect(
          input_data,
          output_backprop,
          expected_input_backprop_tf_cpu,
          input_sizes=[1, 4, 4, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          use_gpu=False,
          v2=v2)

    if not test.is_gpu_available():
      return

    # The functionality associated with TF_ENABLE_NANPROP is currently
    # not supported on the ROCm platform, so skip this part of the test
    # NANs in input lead to non-deterministic results, and hence skipping
    # the remaining tests altogether on the ROCm platform
    if test.is_built_with_rocm():
      return

    # Test the GPU implementation that uses cudnn for now.
    saved_nanprop = os.environ.get("TF_ENABLE_MAXPOOL_NANPROP")
    # Do not propagate the diff in cases of NaNs
    os.environ["TF_ENABLE_MAXPOOL_NANPROP"] = "0"
    expected_input_backprop_cudnn = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0
    ]

    for v2 in [True, False]:
      self._testMaxPoolGradDirect(
          input_data,
          output_backprop,
          expected_input_backprop_cudnn,
          input_sizes=[1, 4, 4, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          use_gpu=True,
          v2=v2)

    # Propagate the diff in cases of NaNs
    os.environ["TF_ENABLE_MAXPOOL_NANPROP"] = "1"
    expected_input_backprop_cudnn = expected_input_backprop_tf_cpu

    for v2 in [True, False]:
      self._testMaxPoolGradDirect(
          input_data,
          output_backprop,
          expected_input_backprop_cudnn,
          input_sizes=[1, 4, 4, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          use_gpu=True,
          v2=v2)

    if saved_nanprop:
      os.environ["TF_ENABLE_MAXPOOL_NANPROP"] = saved_nanprop
    else:
      del os.environ["TF_ENABLE_MAXPOOL_NANPROP"]

  @test_util.run_deprecated_v1
  def testMaxPoolGradDirect(self):
    self._testMaxPoolGradDirect1_1()
    self._testMaxPoolGradDirect1_2()
    self._testMaxPoolGradDirect1_3()
    self._testMaxPoolGradDirectWithNans2_1()
    self._testMaxPoolGradDirectWithNans2_2()
    self._testMaxPoolGradZeroExplicitPadding()
    self._testMaxPoolGradExplicitPadding_1()
    self._testMaxPoolGradExplicitPadding_2()
    self._testMaxPoolGradExplicitPadding_3()

  def _testMaxPoolGradGradValidPadding1_1(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[1, 3, 3, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=1,
          window_cols=1,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradGradValidPadding2_1_6(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[2, 6, 6, 3],
          output_sizes=[2, 5, 5, 3],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradGradValidPadding2_1_7(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[2, 7, 7, 3],
          output_sizes=[2, 6, 6, 3],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradGradValidPadding2_2(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[2, 2, 2, 3],
          output_sizes=[2, 1, 1, 3],
          window_rows=2,
          window_cols=2,
          row_stride=2,
          col_stride=2,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradGradSamePadding1_1(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[2, 2, 4, 3],
          output_sizes=[2, 2, 4, 3],
          window_rows=1,
          window_cols=1,
          row_stride=1,
          col_stride=1,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradGradSamePadding2_1(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[2, 2, 4, 3],
          output_sizes=[2, 2, 4, 3],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradGradSamePadding2_2(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[2, 2, 4, 3],
          output_sizes=[2, 1, 2, 3],
          window_rows=2,
          window_cols=2,
          row_stride=2,
          col_stride=2,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradGradSamePadding3_1(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[1, 7, 7, 1],
          output_sizes=[1, 7, 7, 1],
          window_rows=3,
          window_cols=3,
          row_stride=1,
          col_stride=1,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  @test_util.run_deprecated_v1
  def testMaxPoolGradGrad(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self._testMaxPoolGradGradValidPadding1_1(data_format, use_gpu)
      self._testMaxPoolGradGradValidPadding2_1_6(data_format, use_gpu)
      self._testMaxPoolGradGradValidPadding2_1_7(data_format, use_gpu)
      self._testMaxPoolGradGradValidPadding2_2(data_format, use_gpu)
      self._testMaxPoolGradGradSamePadding1_1(data_format, use_gpu)
      self._testMaxPoolGradGradSamePadding2_1(data_format, use_gpu)
      self._testMaxPoolGradGradSamePadding2_2(data_format, use_gpu)
      self._testMaxPoolGradGradSamePadding3_1(data_format, use_gpu)

  def _MaxPoolGradGrad(self, orig_input, orig_output, grad, window_rows,
                       window_cols, row_stride, col_stride, padding):
    """Max Pooling Second-Order Gradient.

    Args:
      orig_input: A float Tensor. The original input tensor.
      orig_output: A float Tensor. The original output tensor.
      grad: A float Tensor.
        The 4D (batch x out_rows x out_cols x depth) output backprop.
      window_rows: integer. Kernel size along rows dimension.
      window_cols: integer. Kernel size along cols dimension.
      row_stride: integer. Stride along rows dimension
      col_stride: integer. Stride along cols dimension
      padding: PoolingOpDef.Padding.  Padding type.

    Returns:
      A Tensor.
    """
    return gen_nn_ops.max_pool_grad_grad(
        orig_input, orig_output, grad, [1, window_rows, window_cols, 1],
        [1, row_stride, col_stride, 1], padding)

  @test_util.run_deprecated_v1
  def testAvgPoolGrad(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self._testAvgPoolGradValidPadding1_1(data_format, use_gpu)
      self._testAvgPoolGradValidPadding1_2(data_format, use_gpu)
      self._testAvgPoolGradValidPadding2_1(data_format, use_gpu)
      self._testAvgPoolGradValidPadding2_2(data_format, use_gpu)
      self._testAvgPoolGradSamePadding1_1(data_format, use_gpu)
      self._testAvgPoolGradSamePadding1_2(data_format, use_gpu)
      self._testAvgPoolGradSamePadding2_1(data_format, use_gpu)
      self._testAvgPoolGradSamePadding2_2(data_format, use_gpu)
      self._testAvgPoolGradSamePadding3_1(data_format, use_gpu)

  def _testAvgPoolGradValidPadding1_1(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 3, 3, 3],
        output_sizes=[2, 3, 3, 3],
        window_rows=1,
        window_cols=1,
        row_stride=1,
        col_stride=1,
        padding="VALID",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradValidPadding1_2(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 3, 3, 3],
        output_sizes=[2, 2, 2, 3],
        window_rows=1,
        window_cols=1,
        row_stride=2,
        col_stride=2,
        padding="VALID",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradValidPadding2_1(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 3, 3, 3],
        output_sizes=[2, 2, 2, 3],
        window_rows=2,
        window_cols=2,
        row_stride=1,
        col_stride=1,
        padding="VALID",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradValidPadding2_2(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 2, 2, 3],
        output_sizes=[2, 1, 1, 3],
        window_rows=2,
        window_cols=2,
        row_stride=2,
        col_stride=2,
        padding="VALID",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding1_1(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 2, 4, 3],
        output_sizes=[2, 2, 4, 3],
        window_rows=1,
        window_cols=1,
        row_stride=1,
        col_stride=1,
        padding="SAME",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding1_2(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 2, 4, 3],
        output_sizes=[2, 1, 2, 3],
        window_rows=1,
        window_cols=1,
        row_stride=2,
        col_stride=2,
        padding="SAME",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding2_1(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 2, 4, 3],
        output_sizes=[2, 2, 4, 3],
        window_rows=2,
        window_cols=2,
        row_stride=1,
        col_stride=1,
        padding="SAME",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding2_2(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 2, 4, 3],
        output_sizes=[2, 1, 2, 3],
        window_rows=2,
        window_cols=2,
        row_stride=2,
        col_stride=2,
        padding="SAME",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding3_1(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[1, 7, 7, 1],
        output_sizes=[1, 7, 7, 1],
        window_rows=3,
        window_cols=3,
        row_stride=1,
        col_stride=1,
        padding="SAME",
        data_format=data_format,
        use_gpu=use_gpu)

  @test_util.run_deprecated_v1
  def testShapeFunctionEdgeCases(self):
    # All shapes unknown.
    for pool_func in [nn_ops.max_pool, nn_ops.avg_pool]:
      p = pool_func(
          array_ops.placeholder(dtypes.float32),
          ksize=[1, 1, 1, 1],
          strides=[1, 1, 1, 1],
          padding="SAME")
      self.assertEqual([None, None, None, None], p.get_shape().as_list())
    p, am = nn_ops.max_pool_with_argmax(
        array_ops.placeholder(dtypes.float32),
        ksize=[1, 1, 1, 1],
        strides=[1, 1, 1, 1],
        padding="SAME")
    self.assertEqual([None, None, None, None], p.get_shape().as_list())
    self.assertEqual([None, None, None, None], am.get_shape().as_list())

    # Incorrect input shape.
    for pool_func in [
        nn_ops.max_pool, nn_ops.avg_pool, nn_ops.max_pool_with_argmax
    ]:
      with self.assertRaises(ValueError):
        pool_func(
            array_ops.placeholder(dtypes.float32, shape=[1, 3]),
            ksize=[1, 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding="SAME")

  @test_util.run_deprecated_v1
  @test_util.disable_xla("b/123337890")  # Error messages differ
  def testOpEdgeCases(self):
    with self.session(use_gpu=test.is_gpu_available()) as sess:
      pool_funcs = [nn_ops.max_pool, nn_ops.avg_pool]
      if test.is_gpu_available():
        pool_funcs.append(nn_ops.max_pool_with_argmax)
      for pool_func in pool_funcs:
        if pool_func != nn_ops.max_pool:
          # Illegal strides.
          with self.assertRaisesRegex(
              errors_impl.UnimplementedError,
              "Pooling is not yet supported on the batch"):
            sess.run(
                pool_func(
                    array_ops.placeholder(dtypes.float32),
                    ksize=[1, 1, 1, 1],
                    strides=[2, 1, 1, 1],
                    padding="SAME"))

        # Filter too large.
        with self.assertRaisesRegex(ValueError, "Negative dimension size"):
          sess.run(
              pool_func(
                  array_ops.placeholder(dtypes.float32, shape=[32, 20, 20, 3]),
                  ksize=[1, 20, 21, 1],
                  strides=[1, 1, 1, 1],
                  padding="VALID"))
        with self.assertRaisesRegex(ValueError, "Negative dimension size"):
          pool_func(
              array_ops.placeholder(dtypes.float32, shape=[32, 20, 20, 3]),
              ksize=[1, 21, 20, 1],
              strides=[1, 1, 1, 1],
              padding="VALID")

  @test_util.run_deprecated_v1
  def testEdgeCasesRaiseErrors(self):
    with self.assertRaisesRegexp(
        ValueError, "NCHW_VECT_C.*is not supported with "
        "explicit padding|XLA does not support pooling ops with explicit "
        "padding"):
      nn_ops.max_pool(
          array_ops.placeholder(dtypes.float32, shape=[1, 3, 3, 1]),
          ksize=[1, 2, 2, 1],
          strides=[1, 2, 2, 1],
          padding=[[0, 0], [0, 1], [0, 1], [0, 0]],
          data_format="NCHW_VECT_C")
    with self.assertRaisesRegexp(
        ValueError, "Explicit padding is not supported with an input "
                    "tensor of rank 5"):
      nn_ops.max_pool_v2(
          array_ops.placeholder(dtypes.float32, shape=[1, 3, 3, 1, 1]),
          ksize=[1, 2, 2, 1, 1],
          strides=[1, 2, 2, 1, 1],
          padding=[[0, 0], [0, 1], [0, 1], [0, 0]],
          data_format="NCHW")
    with self.assertRaisesRegexp(
        ValueError, "Attr 'padding' of 'MaxPoolV2' Op passed "
                    "string 'EXPLICIT'"):
      gen_nn_ops.max_pool_v2(
          array_ops.placeholder(dtypes.float32, shape=[1, 3, 3, 1, 1]),
          ksize=[1, 2, 2, 1, 1],
          strides=[1, 2, 2, 1, 1],
          padding="EXPLICIT",
          data_format="NHWC")

  @test_util.run_deprecated_v1
  def testEdgeCasesExcessPadding(self):
    with self.session(use_gpu=test.is_gpu_available()) as sess:
      with self.assertRaisesRegexp(
          (errors_impl.UnimplementedError, errors_impl.InvalidArgumentError),
          "Right padding 2 needs to be smaller than the window size 2|"
          "XLA does not support pooling ops with explicit padding"):
        input_sizes = [1, 3, 3, 1]
        x = [(((f + 128) % 255) - 127) for f in range(9)]
        t = constant_op.constant(x, shape=input_sizes, dtype=dtypes.float32)
        sess.run(gen_nn_ops.max_pool(
            t,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="EXPLICIT",
            explicit_paddings=[0, 0, 0, 1, 0, 2, 0, 0],
            data_format="NHWC"))

  @test_util.run_deprecated_v1
  def testNegativePadding(self):
    with self.session(use_gpu=test.is_gpu_available()) as sess:
      with self.assertRaisesRegexp(
          ValueError, "All elements of explicit_paddings must be "
                      "nonnegative for"):
        input_sizes = [1, 3, 3, 1]
        x = [(((f + 128) % 255) - 127) for f in range(9)]
        t = constant_op.constant(x, shape=input_sizes, dtype=dtypes.float32)
        sess.run(gen_nn_ops.max_pool(
            t,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="EXPLICIT",
            explicit_paddings=[0, 0, -1, -1, -1, -1, 0, 0],
            data_format="NHWC"))

  @test_util.run_deprecated_v1
  def testExplicitPaddingBatch(self):
    with self.session(use_gpu=test.is_gpu_available()) as sess:
      with self.assertRaisesRegexp(
          ValueError, "Nonzero explicit padding in the batch or depth "
                      "dimensions is not supported"):
        input_sizes = [1, 3, 3, 1]
        x = [(((f + 128) % 255) - 127) for f in range(9)]
        t = constant_op.constant(x, shape=input_sizes, dtype=dtypes.float32)
        sess.run(gen_nn_ops.max_pool(
            t,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="EXPLICIT",
            explicit_paddings=[1, 1, 1, 1, 1, 1, 0, 0],
            data_format="NHWC"))

  @test_util.disable_xla(
      "b/205634417")  # XLA is not throwing shape errors for multiple *Grad ops.
  def testMaxPoolGradEagerShapeErrors(self):
    with context.eager_mode():
      orig_in = array_ops.ones((1, 1, 1, 1))

      # Test invalid orig_out shape
      orig_out = array_ops.ones((1, 1, 1, 2))
      grad = array_ops.ones((1, 1, 1, 1))
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          r"Expected orig_output shape to be \[1,1,1,1\], but got \[1,1,1,2\]"):
        gen_nn_ops.max_pool_grad(
            orig_in, orig_out, grad, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],
            padding="VALID")
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          r"Expected orig_output shape to be \[1,1,1,1\], but got \[1,1,1,2\]"):
        gen_nn_ops.max_pool_grad_grad(
            orig_in, orig_out, grad, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],
            padding="VALID")

      # Test invalid grad shape
      orig_out = array_ops.ones((1, 1, 1, 1))
      grad = array_ops.ones((1, 1, 1, 2))
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          r"Expected grad shape to be \[1,1,1,1\], but got \[1,1,1,2\]"):
        gen_nn_ops.max_pool_grad(
            orig_in, orig_out, grad, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],
            padding="VALID")
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          r"Expected grad shape to be \[1,1,1,1\], but got \[1,1,1,2\]"):
        gen_nn_ops.max_pool_grad_grad(
            orig_in, orig_out, grad, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],
            padding="VALID")

  def testMaxPoolGradWithArgmaxEagerShapeErrors(self):
    with context.eager_mode():
      inp = array_ops.ones((1, 1, 1, 1))

      # Test invalid grad shape
      grad = array_ops.ones((1, 1, 1, 2))
      argmax = array_ops.zeros((1, 1, 1, 1), dtype=dtypes.int64)
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          r"Expected grad shape to be \[1,1,1,1\], but got \[1,1,1,2\]"):
        gen_nn_ops.max_pool_grad_with_argmax(
            inp, grad, argmax, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],
            padding="VALID")
      # max_pool_grad_grad_with_argmax is only implemented for GPUs
      if test.is_gpu_available():
        with self.assertRaisesRegex(
            errors_impl.InvalidArgumentError,
            r"Expected grad shape to be \[1,1,1,1\], but got \[1,1,1,2\]"):
          gen_nn_ops.max_pool_grad_grad_with_argmax(
              inp, grad, argmax, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],
              padding="VALID")

      # Test invalid argmax shape
      grad = array_ops.ones((1, 1, 1, 1))
      argmax = array_ops.ones((1, 1, 1, 2), dtype=dtypes.int64)
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          r"Expected argmax shape to be \[1,1,1,1\], but got \[1,1,1,2\]"):
        gen_nn_ops.max_pool_grad_with_argmax(
            inp, grad, argmax, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],
            padding="VALID")
      # max_pool_grad_grad_with_argmax is only implemented for GPUs
      if test.is_gpu_available():
        with self.assertRaisesRegex(
            errors_impl.InvalidArgumentError,
            r"Expected argmax shape to be \[1,1,1,1\], but got \[1,1,1,2\]"):
          gen_nn_ops.max_pool_grad_grad_with_argmax(
              inp, grad, argmax, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],
              padding="VALID")

  def testAvgPoolGradInvalidInputShapeRaiseError(self):
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      with self.cached_session():
        orig_input_shape = constant_op.constant(
            -536870912, shape=[4], dtype=dtypes.int32)
        grad = constant_op.constant(
            .0890338004362538, shape=[1, 5, 7, 1], dtype=dtypes.float64)
        t = gen_nn_ops.AvgPoolGrad(
            orig_input_shape=orig_input_shape,
            grad=grad,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="VALID",
            data_format="NHWC")
        self.evaluate(t)


def GetMaxPoolFwdTest(input_size, filter_size, strides, padding):

  def Test(self):
    # MaxPoolWithArgMax is implemented only on CUDA.
    if not test.is_gpu_available(cuda_only=True):
      return
    self._CompareMaxPoolingFwd(input_size, filter_size, strides, padding)

  return Test


def GetMaxPoolGradTest(input_size, filter_size, output_size, strides, padding):

  def Test(self):
    # MaxPoolWithArgMax is implemented only on CUDA.
    if not test.is_gpu_available(cuda_only=True):
      return
    self._CompareMaxPoolingBk(input_size, output_size, filter_size, strides,
                              padding)

  return Test


def GetMaxPoolGradGradTest(input_size, filter_size, output_size, strides,
                           padding):

  def Test(self):
    # MaxPoolWithArgMax is implemented only on CUDA.
    if not test.is_gpu_available(cuda_only=True):
      return
    self._CompareMaxPoolingGradBk(input_size, output_size, filter_size, strides,
                                  padding)

  return Test


if __name__ == "__main__":
  for (name_, input_size_, filter_size_, output_size_, stride_,
       padding_) in GetShrunkInceptionMaxPoolShapes():
    setattr(PoolingTest, "testMaxPoolFwd_" + name_,
            GetMaxPoolFwdTest(input_size_, filter_size_, stride_, padding_))
    setattr(PoolingTest, "testMaxPoolGrad_" + name_,
            GetMaxPoolGradTest(input_size_, filter_size_, output_size_, stride_,
                               padding_))
    setattr(PoolingTest, "testMaxPoolGradGrad_" + name_,
            GetMaxPoolGradGradTest(input_size_, filter_size_, output_size_,
                                   stride_, padding_))
  test.main()
