# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for 3D convolutions using the XLA JIT."""

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import test_utils
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import googletest


CONV_CONFIGS = (
    ("_Conv3D_data_format_NDHWC", "NDHWC", "Conv3D"),
    ("_Conv3D_data_format_NCDHW", "NCDHW", "Conv3D"),
    ("_Conv_data_format_NDHWC", "NDHWC", "Conv"),
    ("_Conv_data_format_NCDHW", "NCDHW", "Conv"),
)


# Test outputs computed in prod (colab) by running nn.conv3d on a GPU device
# with its GPU (non-xla) kernel.
class Conv3DTest(xla_test.XLATestCase, parameterized.TestCase):

  def _VerifyValues(
      self,
      input_sizes=None,
      filter_sizes=None,
      strides=None,
      dilations=None,
      padding=None,
      data_format_src="NDHWC",
      data_format_dst="NDHWC",
      expected=None,
      op_name="Conv3D",
  ):
    """Tests that tf.nn.conv3d produces the expected value.

    Args:
      input_sizes: Input tensor dimensions in [batch, input_rows, input_cols,
        input_depth].
      filter_sizes: Filter tensor dimensions in [kernel_rows, kernel_cols,
        input_depth, output_depth].
      strides: Strides.
      dilations: RHS dilations.
      padding: Padding type.
      data_format_src: Data format input is in.
      data_format_dst: Data format verification will run and input is converted
        to.
      expected: Expected output.
      op_name: Name of operation to test (Conv/Conv2D)
    """

    total_size_1 = np.prod(input_sizes)
    total_size_2 = np.prod(filter_sizes)
    x1 = np.reshape(
        [f * 1.0 / total_size_1 for f in range(1, total_size_1 + 1)],
        input_sizes,
    )
    x2 = np.reshape(
        [f * 1.0 / total_size_2 for f in range(1, total_size_2 + 1)],
        filter_sizes,
    )
    strides = [1] + strides + [1]
    if dilations is None:
      dilations = [1, 1, 1]
    dilations = [1] + dilations + [1]

    # Convert between data formats.
    expected = test_utils.ConvertBetweenDataFormats(
        expected, data_format_src, data_format_dst
    )
    x1 = test_utils.ConvertBetweenDataFormats(
        x1, data_format_src, data_format_dst
    )
    input_sizes = test_utils.PermuteDimsBetweenDataFormats(
        input_sizes, data_format_src, data_format_dst
    )
    strides = test_utils.PermuteDimsBetweenDataFormats(
        strides, data_format_src, data_format_dst
    )
    dilations = test_utils.PermuteDimsBetweenDataFormats(
        dilations, data_format_src, data_format_dst
    )

    with self.session() as sess:
      t1 = array_ops.placeholder(dtypes.bfloat16, shape=input_sizes)
      t2 = array_ops.placeholder(dtypes.bfloat16, shape=filter_sizes)
      with self.test_scope():
        if op_name == "Conv":
          conv_format = (
              "CHANNELS_LAST"
              if data_format_dst == "NDHWC"
              else "CHANNELS_FIRST"
          )
          out = gen_nn_ops.conv(
              t1,
              t2,
              strides=strides,
              padding=padding,
              data_format=conv_format,
              dilations=dilations,
          )
        elif op_name == "Conv3D":
          out = nn_ops.conv3d(
              t1,
              t2,
              strides=strides,
              padding=padding,
              data_format=data_format_dst,
              dilations=dilations,
          )
        else:
          raise ValueError("Invalid op name: %s" % op_name)

      value = sess.run(out, {t1: x1, t2: x2})
      self.assertAllCloseAccordingToType(expected, value)

  @parameterized.named_parameters(*CONV_CONFIGS)
  def testConv3D1x1x1Filter(self, data_format, op_name):
    expected_output = np.reshape(
        [
            0.18518518518518517,
            0.2222222222222222,
            0.25925925925925924,
            0.4074074074074074,
            0.5,
            0.5925925925925926,
            0.6296296296296297,
            0.7777777777777777,
            0.9259259259259259,
            0.8518518518518519,
            1.0555555555555556,
            1.259259259259259,
            1.074074074074074,
            1.3333333333333333,
            1.5925925925925926,
            1.2962962962962963,
            1.6111111111111112,
            1.9259259259259258,
        ],
        [1, 2, 3, 1, 3],
    )

    # These are equivalent to the Conv2D1x1 case.
    self._VerifyValues(
        input_sizes=[1, 2, 3, 1, 3],
        filter_sizes=[1, 1, 1, 3, 3],
        strides=[1, 1, 1],
        padding="VALID",
        expected=expected_output,
        data_format_src="NDHWC",
        data_format_dst=data_format,
        op_name=op_name,
    )
    self._VerifyValues(
        input_sizes=[1, 2, 1, 3, 3],
        filter_sizes=[1, 1, 1, 3, 3],
        strides=[1, 1, 1],
        padding="VALID",
        expected=np.reshape(expected_output, [1, 2, 1, 3, 3]),
        data_format_src="NDHWC",
        data_format_dst=data_format,
        op_name=op_name,
    )
    self._VerifyValues(
        input_sizes=[1, 1, 2, 3, 3],
        filter_sizes=[1, 1, 1, 3, 3],
        strides=[1, 1, 1],
        padding="VALID",
        expected=np.reshape(expected_output, [1, 1, 2, 3, 3]),
        data_format_src="NDHWC",
        data_format_dst=data_format,
        op_name=op_name,
    )

  @parameterized.named_parameters(*CONV_CONFIGS)
  def testConv3D1x1x1Filter2x1x1Dilation(self, data_format, op_name):
    expected_output = np.reshape(
        [
            0.05555555555555555,
            0.1111111111111111,
            0.16666666666666666,
            0.2222222222222222,
            0.2777777777777778,
            0.3333333333333333,
            0.3888888888888889,
            0.4444444444444444,
            0.5,
            0.5555555555555556,
            0.6111111111111112,
            0.6666666666666666,
            0.7222222222222222,
            0.7777777777777778,
            0.8333333333333334,
            0.8888888888888888,
            0.9444444444444444,
            1.0,
        ],
        [1, 3, 6, 1, 1],
    )

    self._VerifyValues(
        input_sizes=[1, 3, 6, 1, 1],
        filter_sizes=[1, 1, 1, 1, 1],
        strides=[1, 1, 1],
        padding="VALID",
        dilations=[2, 1, 1],
        expected=expected_output,
        data_format_src="NDHWC",
        data_format_dst=data_format,
        op_name=op_name,
    )

  # Expected values computed using scipy's correlate function.
  @parameterized.named_parameters(*CONV_CONFIGS)
  def testConv3D2x2x2Filter(self, data_format, op_name):
    expected_output = np.reshape(
        [
            3.7719907407407405,
            3.850694444444445,
            3.929398148148149,
            4.265046296296295,
            4.357638888888888,
            4.450231481481481,
            6.730324074074074,
            6.892361111111109,
            7.054398148148148,
            7.223379629629629,
            7.399305555555557,
            7.575231481481481,
            9.688657407407408,
            9.934027777777779,
            10.17939814814815,
            10.181712962962962,
            10.440972222222221,
            10.700231481481481,
        ],
        [1, 3, 1, 2, 3],
    )
    # expected_shape = [1, 3, 1, 2, 5]
    self._VerifyValues(
        input_sizes=[1, 4, 2, 3, 3],  # b, z, y, x, fin
        filter_sizes=[2, 2, 2, 3, 3],  # z, y, x, fin, fout
        strides=[1, 1, 1],
        padding="VALID",
        expected=expected_output,
        data_format_src="NDHWC",
        data_format_dst=data_format,
        op_name=op_name,
    )

  @parameterized.named_parameters(*CONV_CONFIGS)
  def testConv3D2x2x2Filter1x2x1Dilation(self, data_format, op_name):
    expected_output = np.reshape(
        [
            1.1388888888888888,
            1.2013888888888888,
            1.3263888888888888,
            1.3888888888888888,
            1.5138888888888888,
            1.5763888888888888,
            1.701388888888889,
            1.763888888888889,
            2.263888888888889,
            2.3263888888888893,
            2.451388888888889,
            2.513888888888889,
            2.6388888888888893,
            2.701388888888889,
            2.826388888888889,
            2.888888888888889,
            3.388888888888889,
            3.451388888888889,
            3.576388888888889,
            3.6388888888888884,
            3.7638888888888893,
            3.8263888888888893,
            3.9513888888888893,
            4.013888888888889,
        ],
        [1, 3, 4, 2, 1],
    )

    self._VerifyValues(
        input_sizes=[1, 4, 6, 3, 1],
        filter_sizes=[2, 2, 2, 1, 1],
        strides=[1, 1, 1],
        padding="VALID",
        dilations=[1, 2, 1],
        expected=expected_output,
        data_format_src="NDHWC",
        data_format_dst=data_format,
        op_name=op_name,
    )

  @parameterized.named_parameters(*CONV_CONFIGS)
  def testConv3DStrides(self, data_format, op_name):
    expected_output = np.reshape(
        [
            0.06071428571428571,
            0.08988095238095238,
            0.10238095238095238,
            0.11488095238095238,
            0.12738095238095237,
            0.13988095238095238,
            0.08452380952380953,
            0.26071428571428573,
            0.35238095238095235,
            0.36488095238095236,
            0.3773809523809524,
            0.3898809523809524,
            0.4023809523809524,
            0.23452380952380952,
            0.46071428571428574,
            0.6148809523809524,
            0.6273809523809524,
            0.6398809523809523,
            0.6523809523809524,
            0.6648809523809525,
            0.3845238095238095,
            1.1273809523809524,
            1.4898809523809524,
            1.5023809523809524,
            1.5148809523809523,
            1.5273809523809523,
            1.5398809523809525,
            0.8845238095238095,
            1.3273809523809526,
            1.7523809523809522,
            1.764880952380952,
            1.7773809523809523,
            1.7898809523809525,
            1.8023809523809526,
            1.0345238095238096,
            1.5273809523809525,
            2.0148809523809526,
            2.0273809523809523,
            2.0398809523809525,
            2.052380952380952,
            2.0648809523809524,
            1.1845238095238095,
            2.1940476190476192,
            2.8898809523809526,
            2.9023809523809527,
            2.9148809523809525,
            2.9273809523809526,
            2.9398809523809524,
            1.6845238095238095,
            2.394047619047619,
            3.1523809523809523,
            3.1648809523809525,
            3.177380952380952,
            3.1898809523809524,
            3.2023809523809526,
            1.8345238095238097,
            2.594047619047619,
            3.4148809523809525,
            3.427380952380952,
            3.4398809523809524,
            3.4523809523809526,
            3.4648809523809523,
            1.9845238095238096,
        ],
        [1, 3, 3, 7, 1],
    )
    self._VerifyValues(
        input_sizes=[1, 5, 8, 7, 1],
        filter_sizes=[1, 2, 3, 1, 1],
        strides=[2, 3, 1],  # different stride for each spatial dimension
        padding="SAME",
        expected=expected_output,
        data_format_src="NDHWC",
        data_format_dst=data_format,
        op_name=op_name,
    )

  @parameterized.named_parameters(*CONV_CONFIGS)
  def testConv3D2x2x2FilterStride2(self, data_format, op_name):
    expected_output = np.reshape(
        [
            3.7719907407407405,
            3.850694444444445,
            3.929398148148149,
            9.688657407407408,
            9.934027777777779,
            10.17939814814815,
        ],
        [1, 2, 1, 1, 3],
    )
    self._VerifyValues(
        input_sizes=[1, 4, 2, 3, 3],
        filter_sizes=[2, 2, 2, 3, 3],
        strides=[2, 2, 2],
        padding="VALID",
        expected=expected_output,
        data_format_src="NDHWC",
        data_format_dst=data_format,
        op_name=op_name,
    )

  @parameterized.named_parameters(*CONV_CONFIGS)
  def testConv3DStride3(self, data_format, op_name):
    expected_output = np.reshape(
        [
            1.5114087301587302,
            1.5716765873015872,
            1.6319444444444446,
            1.5634920634920635,
            1.6267361111111112,
            1.6899801587301588,
            1.6155753968253967,
            1.681795634920635,
            1.748015873015873,
            1.9280753968253967,
            2.012152777777778,
            2.096230158730159,
            1.9801587301587302,
            2.067212301587302,
            2.154265873015873,
            2.0322420634920637,
            2.122271825396825,
            2.2123015873015874,
            4.428075396825396,
            4.65500992063492,
            4.881944444444444,
            4.480158730158729,
            4.710069444444444,
            4.939980158730158,
            4.532242063492063,
            4.7651289682539675,
            4.9980158730158735,
            4.844742063492064,
            5.095486111111112,
            5.346230158730158,
            4.896825396825397,
            5.150545634920635,
            5.4042658730158735,
            4.94890873015873,
            5.205605158730158,
            5.462301587301588,
        ],
        [1, 2, 2, 3, 3],
    )
    self._VerifyValues(
        input_sizes=[1, 6, 7, 8, 2],
        filter_sizes=[3, 2, 1, 2, 3],
        strides=[3, 3, 3],
        padding="VALID",
        expected=expected_output,
        data_format_src="NDHWC",
        data_format_dst=data_format,
        op_name=op_name,
    )

  @parameterized.named_parameters(*CONV_CONFIGS)
  def testConv3D2x2x2FilterStride2Same(self, data_format, op_name):
    expected_output = np.reshape(
        [
            3.7719907407407405,
            3.850694444444445,
            3.929398148148149,
            2.0162037037037037,
            2.0659722222222223,
            2.1157407407407405,
            9.688657407407408,
            9.934027777777779,
            10.17939814814815,
            4.599537037037037,
            4.732638888888889,
            4.8657407407407405,
        ],
        [1, 2, 1, 2, 3],
    )
    self._VerifyValues(
        input_sizes=[1, 4, 2, 3, 3],
        filter_sizes=[2, 2, 2, 3, 3],
        strides=[2, 2, 2],
        padding="SAME",
        expected=expected_output,
        data_format_src="NDHWC",
        data_format_dst=data_format,
        op_name=op_name,
    )

  @parameterized.named_parameters(*CONV_CONFIGS)
  def testKernelSmallerThanStride(self, data_format, op_name):
    expected_output = np.reshape(
        [
            0.037037037037037035,
            0.1111111111111111,
            0.25925925925925924,
            0.3333333333333333,
            0.7037037037037037,
            0.7777777777777778,
            0.9259259259259259,
            1.0,
        ],
        [1, 2, 2, 2, 1],
    )

    self._VerifyValues(
        input_sizes=[1, 3, 3, 3, 1],
        filter_sizes=[1, 1, 1, 1, 1],
        strides=[2, 2, 2],
        padding="SAME",
        expected=expected_output,
        data_format_src="NDHWC",
        data_format_dst=data_format,
        op_name=op_name,
    )
    self._VerifyValues(
        input_sizes=[1, 3, 3, 3, 1],
        filter_sizes=[1, 1, 1, 1, 1],
        strides=[2, 2, 2],
        padding="VALID",
        expected=expected_output,
        data_format_src="NDHWC",
        data_format_dst=data_format,
        op_name=op_name,
    )

    expected_output = np.reshape(
        [
            0.5408163265306123,
            0.5801749271137027,
            0.28061224489795916,
            0.8163265306122448,
            0.8556851311953353,
            0.4030612244897959,
            0.41873177842565595,
            0.43403790087463556,
            0.19642857142857142,
            2.4693877551020407,
            2.5087463556851315,
            1.1377551020408163,
            2.7448979591836733,
            2.7842565597667637,
            1.260204081632653,
            1.168731778425656,
            1.1840379008746356,
            0.5178571428571429,
            1.0951166180758019,
            1.1060495626822158,
            0.4464285714285714,
            1.1716472303206997,
            1.1825801749271136,
            0.4770408163265306,
            0.3691690962099125,
            0.37244897959183676,
            0.125,
        ],
        [1, 3, 3, 3, 1],
    )
    self._VerifyValues(
        input_sizes=[1, 7, 7, 7, 1],
        filter_sizes=[2, 2, 2, 1, 1],
        strides=[3, 3, 3],
        padding="SAME",
        expected=expected_output,
        data_format_src="NDHWC",
        data_format_dst=data_format,
        op_name=op_name,
    )

    expected_output = np.reshape(
        [
            0.5408163265306123,
            0.5801749271137027,
            0.8163265306122448,
            0.8556851311953353,
            2.4693877551020407,
            2.5087463556851315,
            2.7448979591836733,
            2.7842565597667637,
        ],
        [1, 2, 2, 2, 1],
    )
    self._VerifyValues(
        input_sizes=[1, 7, 7, 7, 1],
        filter_sizes=[2, 2, 2, 1, 1],
        strides=[3, 3, 3],
        padding="VALID",
        expected=expected_output,
        data_format_src="NDHWC",
        data_format_dst=data_format,
        op_name=op_name,
    )

  @parameterized.named_parameters(*CONV_CONFIGS)
  def testKernelSizeMatchesInputSize(self, data_format, op_name):
    expected_output = np.reshape([1.5625, 1.875], [1, 1, 1, 1, 2])
    self._VerifyValues(
        input_sizes=[1, 2, 1, 2, 1],
        filter_sizes=[2, 1, 2, 1, 2],
        strides=[1, 1, 1],
        padding="VALID",
        expected=expected_output,
        data_format_src="NDHWC",
        data_format_dst=data_format,
        op_name=op_name,
    )

  def testConvExpandedBatch(self):
    tensor_in_sizes_batch = [10, 2, 3, 1, 3]
    tensor_in_sizes_expanded_batch = [2, 5, 2, 3, 1, 3]
    batch_dims = 2
    filter_in_sizes = [1, 1, 1, 3, 3]
    filter_in = np.arange(
        1, np.prod(filter_in_sizes) + 1, dtype=np.float32
    ).reshape(filter_in_sizes)
    x1 = np.arange(
        1, np.prod(tensor_in_sizes_batch) + 1, dtype=np.float32
    ).reshape(tensor_in_sizes_batch)
    x2 = x1.reshape(tensor_in_sizes_expanded_batch)

    with self.session() as sess:
      t1 = array_ops.placeholder(dtypes.bfloat16, shape=tensor_in_sizes_batch)
      t2 = array_ops.placeholder(
          dtypes.bfloat16, shape=tensor_in_sizes_expanded_batch
      )
      filter_t = array_ops.placeholder(dtypes.bfloat16, shape=filter_in_sizes)

      out1 = gen_nn_ops.conv(
          t1, filter_t, strides=[1, 1, 1, 1, 1], padding="VALID"
      )
      out2 = gen_nn_ops.conv(
          t2,
          filter_t,
          strides=[1, 1, 1, 1, 1],
          padding="VALID",
          batch_dims=batch_dims,
      )
      value1 = sess.run(out1, {t1: x1, filter_t: filter_in})
      value2 = sess.run(out2, {t2: x2, filter_t: filter_in})

      self.assertEqual(list(value1.shape), tensor_in_sizes_batch)
      self.assertEqual(list(value2.shape), tensor_in_sizes_expanded_batch)
      self.assertAllCloseAccordingToType(value1, value2.reshape(value1.shape))


# Test cloned from
# tensorflow/python/kernel_tests/conv3d_backprop_filter_v2_grad_test.py
class Conv3DBackpropFilterV2GradTest(xla_test.XLATestCase):

  def testGradient(self):
    with self.session(), self.test_scope():
      for padding in ["SAME", "VALID"]:
        for stride in [1, 2]:
          np.random.seed(1)
          in_shape = [2, 4, 3, 3, 2]
          in_val = constant_op.constant(
              2 * np.random.random_sample(in_shape) - 1, dtype=dtypes.float32)
          filter_shape = [3, 3, 3, 2, 3]
          strides = [1, stride, stride, stride, 1]
          # Make a convolution op with the current settings, just to easily get
          # the shape of the output.
          conv_out = nn_ops.conv3d(in_val,
                                   array_ops.zeros(filter_shape), strides,
                                   padding)
          out_backprop_shape = conv_out.get_shape().as_list()
          out_backprop_val = constant_op.constant(
              2 * np.random.random_sample(out_backprop_shape) - 1,
              dtype=dtypes.float32)
          output = nn_ops.conv3d_backprop_filter_v2(in_val, filter_shape,
                                                    out_backprop_val, strides,
                                                    padding)
          err = gradient_checker.compute_gradient_error(
              [in_val, out_backprop_val], [in_shape, out_backprop_shape],
              output, filter_shape)
          print("conv3d_backprop_filter gradient err = %g " % err)
          err_tolerance = 1e-3
          self.assertLess(err, err_tolerance)


# Test cloned from tensorflow/python/kernel_tests/conv3d_transpose_test.py
class Conv3DTransposeTest(xla_test.XLATestCase):

  def testConv3DTransposeSingleStride(self):
    with self.session(), self.test_scope():
      strides = [1, 1, 1, 1, 1]

      # Input, output: [batch, depth, height, width, channel]
      x_shape = [2, 5, 6, 4, 3]
      y_shape = [2, 5, 6, 4, 2]

      # Filter: [kernel_depth, kernel_height, kernel_width, out_depth, in_depth]
      f_shape = [3, 3, 3, 2, 3]

      x = constant_op.constant(
          1.0, shape=x_shape, name="x", dtype=dtypes.float32)
      f = constant_op.constant(
          1.0, shape=f_shape, name="filter", dtype=dtypes.float32)
      output = nn_ops.conv3d_transpose(
          x, f, y_shape, strides=strides, padding="SAME")
      value = self.evaluate(output)

      # We count the number of cells being added at the locations in the output.
      # At the center, #cells = kernel_depth * kernel_height * kernel_width
      # At the corners, #cells = ceil(kernel_depth/2) * ceil(kernel_height/2)
      #                          * ceil(kernel_width/2)
      # At the edges, #cells =
      #   kernel_depth * ceil(kernel_height/2) * ceil(kernel_width/2) or
      #   ceil(kernel_depth/2) * kernel_height * ceil(kernel_width/2) or
      #   ceil(kernel_depth/2) * ceil(kernel_height/2) * kernel_width
      # At the borders, #cells =
      #   ceil(kernel_depth/2) * kernel_height * kernel_width or
      #   kernel_depth * ceil(kernel_height/2) * kernel_width or
      #   kernel_depth * kernel_height * ceil(kernel_width/2)

      for n in range(x_shape[0]):
        for k in range(f_shape[3]):
          for w in range(y_shape[3]):
            for h in range(y_shape[2]):
              for d in range(y_shape[1]):
                d_in = d > 0 and d < y_shape[1] - 1
                h_in = h > 0 and h < y_shape[2] - 1
                w_in = w > 0 and w < y_shape[3] - 1
                if d_in + h_in + w_in == 3:
                  target = 27 * 3.0
                elif d_in + h_in + w_in == 2:
                  target = 18 * 3.0
                elif d_in or h_in or w_in:
                  target = 12 * 3.0
                else:
                  target = 8 * 3.0
                self.assertAllClose(target, value[n, d, h, w, k])

  def testConv3DTransposeSame(self):
    with self.session(), self.test_scope():
      strides = [1, 2, 2, 2, 1]

      # Input, output: [batch, depth, height, width, depth]
      x_shape = [2, 5, 6, 4, 3]
      y_shape = [2, 10, 12, 8, 2]

      # Filter: [kernel_depth, kernel_height, kernel_width, out_depth, in_depth]
      f_shape = [3, 3, 3, 2, 3]

      x = constant_op.constant(
          1.0, shape=x_shape, name="x", dtype=dtypes.float32)
      f = constant_op.constant(
          1.0, shape=f_shape, name="filter", dtype=dtypes.float32)
      output = nn_ops.conv3d_transpose(
          x, f, y_shape, strides=strides, padding="SAME")
      value = self.evaluate(output)

      for n in range(x_shape[0]):
        for k in range(f_shape[3]):
          for w in range(y_shape[3]):
            for h in range(y_shape[2]):
              for d in range(y_shape[1]):
                # We add a case for locations divisible by the stride.
                d_in = d % strides[1] == 0 and 0 < d < y_shape[1] - 1
                h_in = h % strides[2] == 0 and 0 < h < y_shape[2] - 1
                w_in = w % strides[3] == 0 and 0 < w < y_shape[3] - 1
                if d_in + h_in + w_in == 3:
                  target = 8 * 3.0
                elif d_in + h_in + w_in == 2:
                  target = 4 * 3.0
                elif d_in or h_in or w_in:
                  target = 2 * 3.0
                else:
                  target = 3.0
                self.assertAllClose(target, value[n, d, h, w, k])

  def testConv3DTransposeValid(self):
    with self.session(), self.test_scope():
      strides = [1, 2, 2, 2, 1]

      # Input, output: [batch, depth, height, width, depth]
      x_shape = [2, 5, 6, 4, 3]
      y_shape = [2, 11, 13, 9, 2]

      # Filter: [kernel_depth, kernel_height, kernel_width, out_depth, in_depth]
      f_shape = [3, 3, 3, 2, 3]

      x = constant_op.constant(
          1.0, shape=x_shape, name="x", dtype=dtypes.float32)
      f = constant_op.constant(
          1.0, shape=f_shape, name="filter", dtype=dtypes.float32)
      output = nn_ops.conv3d_transpose(
          x, f, y_shape, strides=strides, padding="VALID")
      value = self.evaluate(output)

      cache_values = np.zeros(y_shape, dtype=np.float32)

      # The amount of padding added
      pad = 1

      for n in range(x_shape[0]):
        for k in range(f_shape[3]):
          for w in range(y_shape[3]):
            for h in range(y_shape[2]):
              for d in range(y_shape[1]):
                # We add a case for locations divisible by the stride.
                d_in = d % strides[1] == 0 and pad < d < y_shape[1] - 1 - pad
                h_in = h % strides[2] == 0 and pad < h < y_shape[2] - 1 - pad
                w_in = w % strides[3] == 0 and pad < w < y_shape[3] - 1 - pad
                if d_in + h_in + w_in == 3:
                  target = 8 * 3.0
                elif d_in + h_in + w_in == 2:
                  target = 4 * 3.0
                elif d_in or h_in or w_in:
                  target = 2 * 3.0
                else:
                  target = 3.0
                cache_values[n, d, h, w, k] = target

          # copy values in the border
          cache_values[n, :, :, 0, k] = cache_values[n, :, :, 1, k]
          cache_values[n, :, :, -1, k] = cache_values[n, :, :, -2, k]
          cache_values[n, :, 0, :, k] = cache_values[n, :, 1, :, k]
          cache_values[n, :, -1, :, k] = cache_values[n, :, -2, :, k]
          cache_values[n, 0, :, :, k] = cache_values[n, 1, :, :, k]
          cache_values[n, -1, :, :, k] = cache_values[n, -2, :, :, k]

    self.assertAllClose(cache_values, value)

  def testGradient(self):
    x_shape = [2, 3, 4, 3, 2]
    f_shape = [3, 3, 3, 2, 2]
    y_shape = [2, 6, 8, 6, 2]
    strides = [1, 2, 2, 2, 1]
    np.random.seed(1)  # Make it reproducible.
    x_val = np.random.random_sample(x_shape).astype(np.float64)
    f_val = np.random.random_sample(f_shape).astype(np.float64)
    with self.session(), self.test_scope():
      x = constant_op.constant(x_val, name="x", dtype=dtypes.float32)
      f = constant_op.constant(f_val, name="f", dtype=dtypes.float32)
      output = nn_ops.conv3d_transpose(
          x, f, y_shape, strides=strides, padding="SAME")
      err = gradient_checker.compute_gradient_error([x, f], [x_shape, f_shape],
                                                    output, y_shape)
    print("conv3d_transpose gradient err = %g " % err)
    err_tolerance = 0.001
    self.assertLess(err, err_tolerance)


if __name__ == "__main__":
  googletest.main()
