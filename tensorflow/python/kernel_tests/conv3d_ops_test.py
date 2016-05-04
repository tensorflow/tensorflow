# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Functional tests for 3d convolutional operations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util


def GetTestConfigs():
    """Get all the valid tests configs to run.

    Returns:
    all the valid test configs as tuples of data_format and use_gpu.
    """
    test_configs = [("NDHWC", False)]
    # TODO: Implement GPU version
    # if test_util.IsGoogleCudaEnabled():
    #     test_configs += [("NDHWC", True)]
    return test_configs


class Conv3DTest(tf.test.TestCase):

    def _SetupValuesForDevice(self, tensor_in_sizes, filter_in_sizes, strides,
                              padding, data_format, use_gpu):
        """Verifies the output values of the convolution function.

        Args:
          tensor_in_sizes: Input tensor dimensions in
            [batch, input_depth, input_height, input_width, input_channels].
          filter_in_sizes: Filter tensor dimensions in
                [kernel_depth, kernel_width, kernel_height,
                in_channels, out_channels].
          strides: Stride: [depth_stride, height_stride, width_stride].
            (Batch and channel strides are not supported at the moment)
          padding: Padding type.
          data_format: Format of the data tensors.
          use_gpu: True if the operations should be run on GPU
        Returns:
          Symbolic tensor value that can be used to execute the computation
        """
        total_size_1 = 1
        total_size_2 = 1
        for s in tensor_in_sizes:
            total_size_1 *= s
        for s in filter_in_sizes:
            total_size_2 *= s
        # Initializes the input tensor with array containing incrementing
        # numbers from 1.
        x1 = [f * 1.0 for f in range(1, total_size_1 + 1)]
        x2 = [f * 1.0 for f in range(1, total_size_2 + 1)]
        with self.test_session(use_gpu=use_gpu):
            t1 = tf.constant(x1, shape=tensor_in_sizes)
            t2 = tf.constant(x2, shape=filter_in_sizes)
            strides = [1] + strides + [1]
            conv = tf.nn.conv3d(t1, t2, strides=strides,
                                padding=padding,
                                data_format=data_format)
            return conv

    def _VerifyValues(self, tensor_in_sizes, filter_in_sizes,
                      strides, padding, expected):
        tensors = []
        for (data_format, use_gpu) in GetTestConfigs():
            result = self._SetupValuesForDevice(
                tensor_in_sizes, filter_in_sizes, strides,
                padding, data_format, use_gpu=use_gpu)
            tensors.append(result)
        with self.test_session() as sess:
            values = sess.run(tensors)
            for i in range(len(tensors)):
                conv = tensors[i]
                value = values[i]
                flat_value = np.ravel(value)
                print("expected = ", expected)
                print("actual = ", flat_value)
                self.assertArrayNear(expected, flat_value, 1e-5)
                self.assertShapeEqual(value, conv)

    def testConv3D1x1Filter(self):
        # Outputs are computed through 3rd party implementations
        expected_output = [1, 2, 2, 4, 3, 6, 4, 8, 5, 10, 6, 12,
                           7, 14, 8, 16, 9, 18, 10, 20, 11, 22,
                           12, 24, 13, 26, 14, 28, 15, 30, 16, 32]
        self._VerifyValues(tensor_in_sizes=[2, 2, 2, 2, 1],
                           filter_in_sizes=[1, 1, 1, 1, 2],
                           strides=[1, 1, 1], padding="VALID",
                           expected=expected_output)

    def testConv3DEmpty(self):
        # Outputs are computed through 3rd party implementations
        expected_output = []
        self._VerifyValues(tensor_in_sizes=[0, 2, 2, 3, 3],
                           filter_in_sizes=[1, 1, 1, 3, 3],
                           strides=[1, 1, 1], padding="VALID",
                           expected=expected_output)

    def testConv3D2x2Filter(self):
        # Outputs are computed through 3rd party implementations
        expected_output = [540, 576, 612]
        self._VerifyValues(tensor_in_sizes=[1, 2, 2, 2, 1],
                           filter_in_sizes=[2, 2, 2, 1, 3],
                           strides=[1, 1, 1], padding="VALID",
                           expected=expected_output)

    def testConv3D1x2x3FilterA(self):
        # Outputs are computed through 3rd party implementations
        expected_output = [
            2748, 2862, 2976, 3168, 3306, 3444, 3588, 3750,
            3912, 4008, 4194, 4380, 5268, 5526, 5784, 5688,
            5970, 6252, 6108, 6414, 6720, 6528, 6858, 7188,
            7788, 8190, 8592, 8208, 8634, 9060, 8628, 9078,
            9528, 9048, 9522, 9996, 12828, 13518, 14208, 13248,
            13962, 14676, 13668, 14406, 15144, 14088, 14850, 15612,
            15348, 16182, 17016, 15768, 16626, 17484, 16188, 17070,
            17952, 16608, 17514, 18420, 17868, 18846, 19824, 18288,
            19290, 20292, 18708, 19734, 20760, 19128, 20178, 21228]
        self._VerifyValues(tensor_in_sizes=[1, 2, 4, 6, 2],
                           filter_in_sizes=[1, 2, 3, 2, 3],
                           strides=[1, 1, 1], padding="VALID",
                           expected=expected_output)

    def testConv3D1x2x3FilterB(self):
        # Outputs are computed through 3rd party implementations
        expected_output = [
            161, 182, 269, 308, 485, 560, 593, 686, 809, 938,
            917, 1064]
        self._VerifyValues(tensor_in_sizes=[1, 3, 3, 3, 1],
                           filter_in_sizes=[1, 2, 3, 1, 2],
                           strides=[1, 1, 1], padding="VALID",
                           expected=expected_output)

    def testConv3D2x2FilterStride2(self):
        # Outputs are computed through 3rd party implementations
        expected_output = [
            2184, 2316, 2448, 2368, 2516, 2664, 3104,
            3316, 3528, 3288, 3516, 3744, 6784, 7316,
            7848, 6968, 7516, 8064, 7704, 8316, 8928,
            7888, 8516, 9144]
        self._VerifyValues(tensor_in_sizes=[1, 5, 5, 5, 1],
                           filter_in_sizes=[2, 2, 2, 1, 3],
                           strides=[2, 2, 2], padding="VALID",
                           expected=expected_output)

    def testConv3D2x2FilterStride1x2x1(self):
        # Outputs are computed through 3rd party implementations
        expected_output = [652, 712, 716, 784, 1228, 1360, 1292, 1432]
        self._VerifyValues(tensor_in_sizes=[1, 3, 3, 3, 1],
                           filter_in_sizes=[2, 2, 2, 1, 2],
                           strides=[1, 2, 1], padding="VALID",
                           expected=expected_output)

    def testConv3D2x2FilterStride2x1x1(self):
        # Outputs are computed through 3rd party implementations
        expected_output = [652, 712, 716, 784, 844, 928, 908, 1000]
        self._VerifyValues(tensor_in_sizes=[1, 3, 3, 3, 1],
                           filter_in_sizes=[2, 2, 2, 1, 2],
                           strides=[2, 1, 1], padding="VALID",
                           expected=expected_output)

    def testConv3D2x2FilterStride2Same(self):
        # Outputs are computed through 3rd party implementations
        expected_output = [560, 632, 848, 920, 1712, 1784, 2000, 2072]
        self._VerifyValues(tensor_in_sizes=[1, 4, 4, 4, 1],
                           filter_in_sizes=[2, 2, 2, 1, 1],
                           strides=[2, 2, 2], padding="SAME",
                           expected=expected_output)

    def testConv3D3x3x3FilterSame(self):
        # Outputs are computed through 3rd party implementations
        expected_output = [
            1412, 2198, 1508, 2370, 3648, 2478,
            1652, 2522, 1700, 2982, 4512, 3018, 4608,
            6930, 4608, 3018, 4512, 2982, 1700, 2522,
            1652, 2478, 3648, 2370, 1508, 2198, 1412]
        self._VerifyValues(tensor_in_sizes=[1, 3, 3, 3, 1],
                           filter_in_sizes=[3, 3, 3, 1, 1],
                           strides=[1, 1, 1], padding="SAME",
                           expected=expected_output)

    def testConv3DKernelSmallerThanStrideSame(self):
        # Outputs are computed through 3rd party implementations
        expected_output = [
            3188, 3524, 4892, 5036, 8900, 8420, 7724, 7052]
        self._VerifyValues(tensor_in_sizes=[1, 5, 5, 5, 1],
                           filter_in_sizes=[3, 3, 3, 1, 1],
                           strides=[4, 4, 4], padding="SAME",
                           expected=expected_output)


class Conv3DBackpropTest(tf.test.TestCase):

    def _RunAndVerifyBackpropInput(self, input_sizes, filter_sizes,
                                   output_sizes, strides, padding,
                                   expected, data_format, use_gpu):
        total_output_size = 1
        total_filter_size = 1
        for s in output_sizes:
            total_output_size *= s
        for s in filter_sizes:
            total_filter_size *= s
        # Initializes the input tensor with array containing incrementing
        # numbers from 1.
        x1 = [f * 1.0 for f in range(1, total_filter_size + 1)]
        x2 = [f * 1.0 for f in range(1, total_output_size + 1)]
        with self.test_session(use_gpu=use_gpu) as sess:
            t0 = tf.constant(input_sizes, shape=[len(input_sizes)])
            t1 = tf.constant(x1, shape=filter_sizes)
            t2 = tf.constant(x2, shape=output_sizes)
            strides = [1] + strides + [1]
            conv = tf.nn.conv3d_backprop_input(t0, t1, t2,
                                               strides=strides,
                                               padding=padding,
                                               data_format=data_format)
            # "values" consists of two tensors for two backprops
            value = sess.run(conv)
            self.assertShapeEqual(value, conv)
        print("expected = ", expected)
        flat = value.flatten()
        print("actual = ", flat)
        self.assertArrayNear(expected, flat, 1e-5)

    def testConv3DValidBackpropInput(self):
        # Gradients are computed through 3rd party implementations
        expected_output = [1]
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropInput(input_sizes=[1, 1, 1, 1, 1],
                                            filter_sizes=[1, 1, 1, 1, 1],
                                            output_sizes=[1, 1, 1, 1, 1],
                                            strides=[1, 1, 1],
                                            padding="VALID",
                                            expected=expected_output,
                                            data_format=data_format,
                                            use_gpu=use_gpu)

    def testConv3D2x2x2x3ValidBackpropInput(self):
        # Gradients are computed through 3rd party implementations
        expected_output = [
            14, 32, 50, 68, 86, 104, 122, 140]
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropInput(input_sizes=[1, 2, 2, 2, 1],
                                            filter_sizes=[2, 2, 2, 1, 3],
                                            output_sizes=[1, 1, 1, 1, 3],
                                            strides=[1, 1, 1],
                                            padding="VALID",
                                            expected=expected_output,
                                            data_format=data_format,
                                            use_gpu=use_gpu)

    def testConv3DStride2ValidBackpropInput(self):
        # Gradients are computed through 3rd party implementations
        expected_output = [
            14, 32, 32, 77, 50, 68, 122, 167, 50, 122, 68, 167,
            194, 266, 266, 365, 86, 104, 212, 257, 122, 140, 302,
            347, 338, 410, 464, 563, 482, 554, 662, 761, 86, 212,
            104, 257, 338, 464, 410, 563, 122, 302, 140, 347, 482, 662,
            554, 761, 590, 716, 716, 869, 842, 968, 1022, 1175, 842,
            1022, 968, 1175, 1202, 1382, 1382, 1589]
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropInput(input_sizes=[1, 4, 4, 4, 1],
                                            filter_sizes=[2, 2, 2, 1, 3],
                                            output_sizes=[1, 2, 2, 2, 3],
                                            strides=[2, 2, 2],
                                            padding="VALID",
                                            expected=expected_output,
                                            data_format=data_format,
                                            use_gpu=use_gpu)

    def testConv3DStride123ValidBackpropInput(self):
        expected_output = [
            5, 11, 0, 0, 17, 23, 0, 0, 11, 25, 0, 0, 39,
            53, 0, 0, 46, 74, 0, 0, 102, 130, 0, 0, 90, 134, 0, 0, 178,
            222, 0, 0, 134, 194, 0, 0, 254, 314, 0, 0, 178, 254,
            0, 0, 330, 406, 0, 0, 181, 219, 0, 0, 257, 295, 0,
            0, 219, 265, 0, 0, 311, 357, 0, 0]
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropInput(input_sizes=[1, 4, 4, 4, 1],
                                            filter_sizes=[2, 2, 2, 1, 2],
                                            output_sizes=[1, 3, 2, 1, 2],
                                            strides=[1, 2, 3],
                                            padding="VALID",
                                            expected=expected_output,
                                            data_format=data_format,
                                            use_gpu=use_gpu)

    def testConv3DFilter123ValidBackpropInput(self):
        # Gradients are computed through 3rd party implementations
        expected_output = [
            5, 11, 17, 0, 23, 29, 35, 0, 11, 25, 39, 0, 53, 67, 81, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 39, 61,
            0, 83, 105, 127, 0, 23, 53, 83, 0, 113, 143, 173, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropInput(input_sizes=[1, 4, 4, 4, 1],
                                            filter_sizes=[1, 2, 3, 1, 2],
                                            output_sizes=[1, 2, 2, 1, 2],
                                            strides=[2, 2, 2],
                                            padding="VALID",
                                            expected=expected_output,
                                            data_format=data_format,
                                            use_gpu=use_gpu)

    def _RunAndVerifyBackpropFilter(self, input_sizes, filter_sizes,
                                    output_sizes, strides, padding,
                                    expected, data_format, use_gpu):
        total_input_size = 1
        total_output_size = 1
        for s in input_sizes:
            total_input_size *= s
        for s in output_sizes:
            total_output_size *= s
        # Initializes the input tensor with array containing incrementing
        # numbers from 1.
        x0 = [f * 1.0 for f in range(1, total_input_size + 1)]
        x2 = [f * 1.0 for f in range(1, total_output_size + 1)]
        with self.test_session(use_gpu=use_gpu) as sess:
            t0 = tf.constant(x0, shape=input_sizes)
            t1 = tf.constant(filter_sizes, shape=[len(filter_sizes)])
            t2 = tf.constant(x2, shape=output_sizes)
            strides = [1] + strides + [1]
            conv = tf.nn.conv3d_backprop_filter(t0, t1, t2,
                                                strides=strides,
                                                padding=padding,
                                                data_format=data_format)
            value = sess.run(conv)
            self.assertShapeEqual(value, conv)
        print("expected = ", expected)
        flat = value.flatten()
        print("actual = ", flat)
        self.assertArrayNear(expected, flat, 1e-5)

    def testConv3D1x1x1ValidBackpropFilter(self):
        # Gradients are computed through 3rd party implementations
        expected = [204]
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropFilter(input_sizes=[1, 2, 2, 2, 1],
                                             filter_sizes=[1, 1, 1, 1, 1],
                                             output_sizes=[1, 2, 2, 2, 1],
                                             strides=[1, 1, 1],
                                             padding="VALID",
                                             expected=expected,
                                             data_format=data_format,
                                             use_gpu=use_gpu)

    def testConv3D2x2x2x2ValidBackpropFilter(self):
        # Gradients are computed through 3rd party implementations
        expected = [
            652, 712, 716, 784, 844, 928, 908, 1000, 1228,
            1360, 1292, 1432, 1420, 1576, 1484, 1648]
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropFilter(input_sizes=[1, 3, 3, 3, 1],
                                             filter_sizes=[2, 2, 2, 1, 2],
                                             output_sizes=[1, 2, 2, 2, 2],
                                             strides=[1, 1, 1],
                                             padding="VALID",
                                             expected=expected,
                                             data_format=data_format,
                                             use_gpu=use_gpu)

    def testConv3DFilter123ValidBackpropFilter(self):
        expected = [567, 636, 603, 678, 639, 720, 675,
                    762, 711, 804, 747, 846]
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropFilter(input_sizes=[1, 3, 3, 3, 1],
                                             filter_sizes=[1, 2, 3, 1, 2],
                                             output_sizes=[1, 3, 2, 1, 2],
                                             strides=[1, 1, 1],
                                             padding="VALID",
                                             expected=expected,
                                             data_format=data_format,
                                             use_gpu=use_gpu)

    def testConv3DStride123ValidBackpropFilter(self):
        expected = [
            1036, 1162, 1072, 1204, 1180, 1330, 1216,
            1372, 1612, 1834, 1648, 1876, 1756, 2002, 1792, 2044]
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropFilter(input_sizes=[1, 4, 4, 4, 1],
                                             filter_sizes=[2, 2, 2, 1, 2],
                                             output_sizes=[1, 3, 2, 1, 2],
                                             strides=[1, 2, 3],
                                             padding="VALID",
                                             expected=expected,
                                             data_format=data_format,
                                             use_gpu=use_gpu)

    def testConv3DFilter123Stride123ValidBackpropFilter(self):
        expected = [1804, 1968, 1868, 2040, 1932, 2112, 2188,
                    2400, 2252, 2472, 2316, 2544]
        for (data_format, use_gpu) in GetTestConfigs():
            self._RunAndVerifyBackpropFilter(input_sizes=[1, 2, 4, 6, 1],
                                             filter_sizes=[1, 2, 3, 1, 2],
                                             output_sizes=[1, 2, 2, 2, 2],
                                             strides=[1, 2, 3],
                                             padding="VALID",
                                             expected=expected,
                                             data_format=data_format,
                                             use_gpu=use_gpu)

    def ConstructAndTestGradient(self, batch, input_depth, input_rows,
                                 input_cols, filter_depth, filter_rows,
                                 filter_cols, in_channels, out_channels,
                                 stride_depth, stride_rows, stride_cols,
                                 padding, test_input, data_format, use_gpu):
        input_shape = [batch, input_depth, input_rows, input_cols, in_channels]
        filter_shape = [filter_depth, filter_rows, filter_cols,
                        in_channels, out_channels]

        if padding == "VALID":
            output_depth = (input_depth - filter_depth + stride_depth)
            output_depth //= stride_depth
            output_rows = (input_rows - filter_rows + stride_rows)
            output_rows //= stride_rows
            output_cols = (input_cols - filter_cols + stride_cols)
            output_cols //= stride_cols
        else:
            output_depth = (input_depth + stride_depth - 1) // stride_depth
            output_rows = (input_rows + stride_rows - 1) // stride_rows
            output_cols = (input_cols + stride_cols - 1) // stride_cols
        output_shape = [batch, output_depth, output_rows,
                        output_cols, out_channels]
        input_size = 1
        for x in input_shape:
            input_size *= x
        filter_size = 1
        for x in filter_shape:
            filter_size *= x
        input_data = [x * 1.0 / input_size for x in range(0, input_size)]
        filter_data = [x * 1.0 / filter_size for x in range(0, filter_size)]
        with self.test_session(use_gpu=use_gpu):
            # Conv3DGrad functions are not compiled for double due to
            # a problem in the way Eigen's Conv3DGrad works for double.
            # So we disable the DOUBLE path.  We should re-enable this
            # when double support returns for CPU and/or GPU.

            # data_type = tf.float64
            # tolerance = 1e-8

            data_type = tf.float32
            tolerance = 0.002
            input_tensor = tf.constant(input_data, shape=input_shape,
                                       dtype=data_type, name="input")
            filter_tensor = tf.constant(filter_data, shape=filter_shape,
                                        dtype=data_type, name="filter")
            strides = [1, stride_depth, stride_rows, stride_cols, 1]
            conv = tf.nn.conv3d(input_tensor, filter_tensor, strides,
                                padding, data_format=data_format, name="conv")
            self.assertEqual(output_shape, conv.get_shape())
            if test_input:
                err = tf.test.compute_gradient_error(input_tensor, input_shape,
                                                     conv, output_shape)
            else:
                err = tf.test.compute_gradient_error(filter_tensor,
                                                     filter_shape, conv,
                                                     output_shape)
            print("conv3d gradient error = ", err)
            self.assertLess(err, tolerance)

    def testInputGradientValidPaddingStrideOne(self):
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(
                batch=2, input_depth=4, input_rows=4, input_cols=4,
                filter_depth=2, filter_rows=2, filter_cols=2,
                in_channels=2, out_channels=3, stride_depth=1,
                stride_rows=1, stride_cols=1, padding="VALID",
                test_input=True, data_format=data_format, use_gpu=use_gpu)

    def testFilterGradientValidPaddingStrideOne(self):
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(
                batch=2, input_depth=4, input_rows=4, input_cols=4,
                filter_depth=2, filter_rows=2, filter_cols=2,
                in_channels=2, out_channels=3, stride_depth=1,
                stride_rows=1, stride_cols=1, padding="VALID",
                test_input=False, data_format=data_format, use_gpu=use_gpu)

    def testInputGradientValidPaddingStrideTwo(self):
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(
                batch=2, input_depth=4, input_rows=4, input_cols=4,
                filter_depth=2, filter_rows=2, filter_cols=2,
                in_channels=2, out_channels=3, stride_depth=2,
                stride_rows=2, stride_cols=2, padding="VALID",
                test_input=True, data_format=data_format, use_gpu=use_gpu)

    def testFilterGradientValidPaddingStrideTwo(self):
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(
                batch=2, input_depth=4, input_rows=4, input_cols=4,
                filter_depth=2, filter_rows=2, filter_cols=2,
                in_channels=2, out_channels=3, stride_depth=2,
                stride_rows=2, stride_cols=2, padding="VALID",
                test_input=False, data_format=data_format, use_gpu=use_gpu)

    def testInputGradientValidPaddingStrideThree(self):
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(
                batch=2, input_depth=5, input_rows=6, input_cols=7,
                filter_depth=3, filter_rows=3, filter_cols=3,
                in_channels=2, out_channels=3, stride_depth=3,
                stride_rows=3, stride_cols=3, padding="VALID",
                test_input=True, data_format=data_format, use_gpu=use_gpu)

    def testFilterGradientValidPaddingStrideThree(self):
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(
                batch=2, input_depth=5, input_rows=6, input_cols=7,
                filter_depth=3, filter_rows=3, filter_cols=3,
                in_channels=2, out_channels=3, stride_depth=3,
                stride_rows=3, stride_cols=3, padding="VALID",
                test_input=False, data_format=data_format, use_gpu=use_gpu)

    def testInputGradientValidPaddingFilter123(self):
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(
                batch=2, input_depth=5, input_rows=6, input_cols=7,
                filter_depth=1, filter_rows=2, filter_cols=3,
                in_channels=2, out_channels=3, stride_depth=2,
                stride_rows=2, stride_cols=32, padding="VALID",
                test_input=True, data_format=data_format, use_gpu=use_gpu)

    def testFilterGradientValidPaddingFilter123(self):
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(
                batch=2, input_depth=5, input_rows=6, input_cols=7,
                filter_depth=1, filter_rows=2, filter_cols=3,
                in_channels=2, out_channels=3, stride_depth=2,
                stride_rows=2, stride_cols=2, padding="VALID",
                test_input=False, data_format=data_format, use_gpu=use_gpu)

    def testInputGradientSamePaddingStrideOne(self):
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(
                batch=2, input_depth=4, input_rows=5, input_cols=6,
                filter_depth=2, filter_rows=3, filter_cols=3,
                in_channels=2, out_channels=3, stride_depth=1,
                stride_rows=1, stride_cols=1, padding="SAME",
                test_input=True, data_format=data_format, use_gpu=use_gpu)

    def testFilterGradientSamePaddingStrideOne(self):
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(
                batch=4, input_depth=4, input_rows=5, input_cols=6,
                filter_depth=2, filter_rows=2, filter_cols=2,
                in_channels=2, out_channels=3, stride_depth=1,
                stride_rows=1, stride_cols=1, padding="SAME",
                test_input=False, data_format=data_format, use_gpu=use_gpu)

    def testInputGradientSamePaddingStrideTwo(self):
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(
                batch=2, input_depth=4, input_rows=5, input_cols=6,
                filter_depth=2, filter_rows=3, filter_cols=3,
                in_channels=2, out_channels=3, stride_depth=2,
                stride_rows=2, stride_cols=2, padding="SAME",
                test_input=True, data_format=data_format, use_gpu=use_gpu)

    def testFilterGradientSamePaddingStrideTwo(self):
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(
                batch=4, input_depth=4, input_rows=5, input_cols=6,
                filter_depth=2, filter_rows=2, filter_cols=2,
                in_channels=2, out_channels=3, stride_depth=2,
                stride_rows=2, stride_cols=2, padding="SAME",
                test_input=False, data_format=data_format, use_gpu=use_gpu)

    def testInputGradientSamePaddingStrideThree(self):
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(
                batch=4, input_depth=5, input_rows=6, input_cols=8,
                filter_depth=2, filter_rows=3, filter_cols=2,
                in_channels=1, out_channels=2, stride_depth=3,
                stride_rows=3, stride_cols=3, padding="SAME",
                test_input=True, data_format=data_format, use_gpu=use_gpu)

    def testFilterGradientSamePaddingStrideThree(self):
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(
                batch=4, input_depth=5, input_rows=6, input_cols=8,
                filter_depth=2, filter_rows=3, filter_cols=2,
                in_channels=1, out_channels=2, stride_depth=3,
                stride_rows=3, stride_cols=3, padding="SAME",
                test_input=False, data_format=data_format, use_gpu=use_gpu)

    def testInputGradientSamePaddingStride123(self):
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(
                batch=2, input_depth=6, input_rows=8, input_cols=7,
                filter_depth=2, filter_rows=4, filter_cols=4,
                in_channels=2, out_channels=3, stride_depth=1,
                stride_rows=2, stride_cols=3, padding="SAME",
                test_input=True, data_format=data_format, use_gpu=use_gpu)

    def testFilterGradientSamePaddingStride123(self):
        for (data_format, use_gpu) in GetTestConfigs():
            self.ConstructAndTestGradient(
                batch=2, input_depth=6, input_rows=8, input_cols=7,
                filter_depth=2, filter_rows=4, filter_cols=4,
                in_channels=2, out_channels=3, stride_depth=1,
                stride_rows=2, stride_cols=3, padding="SAME",
                test_input=False, data_format=data_format, use_gpu=use_gpu)

    def testShapeFunctionEdgeCases(self):
        # All shapes unknown.
        c1 = tf.nn.conv3d(tf.placeholder(tf.float32),
                          tf.placeholder(tf.float32),
                          strides=[1, 1, 1, 1, 1], padding="SAME")
        self.assertEqual([None, None, None, None, None],
                         c1.get_shape().as_list())

        # Incorrect input shape.
        with self.assertRaises(ValueError):
            tf.nn.conv3d(tf.placeholder(tf.float32, shape=[1, 3, 2]),
                         tf.placeholder(tf.float32),
                         strides=[1, 1, 1, 1, 1], padding="SAME")

        # Incorrect filter shape.
        with self.assertRaises(ValueError):
            tf.nn.conv3d(tf.placeholder(tf.float32),
                         tf.placeholder(tf.float32, shape=[2, 1, 3]),
                         strides=[1, 1, 1, 1, 1], padding="SAME")

        # Channel mismatch.
        with self.assertRaises(ValueError):
            tf.nn.conv3d(tf.placeholder(tf.float32, shape=[32, 20, 20, 10, 3]),
                         tf.placeholder(tf.float32, shape=[5, 4, 4, 2, 2]),
                         strides=[1, 1, 1, 1, 1], padding="SAME")

        # Illegal strides.
        with self.assertRaisesRegexp(ValueError,
                                     "strides in the batch and depth"):
            tf.nn.conv3d(tf.placeholder(tf.float32),
                         tf.placeholder(tf.float32),
                         strides=[2, 1, 1, 1, 1], padding="SAME")

        with self.assertRaisesRegexp(ValueError,
                                     "strides in the batch and depth"):
            tf.nn.conv3d(tf.placeholder(tf.float32),
                         tf.placeholder(tf.float32),
                         strides=[1, 1, 1, 1, 2], padding="SAME")

        # Filter larger than input.
        with self.assertRaisesRegexp(
                ValueError, "filter must not be larger than the input"):
            tf.nn.conv3d(
                tf.placeholder(tf.float32, shape=[32, 20, 20, 20, 3]),
                tf.placeholder(tf.float32, shape=[20, 20, 21, 3, 2]),
                strides=[1, 1, 1, 1, 1], padding="SAME")

        with self.assertRaisesRegexp(
                ValueError, "filter must not be larger than the input"):
            tf.nn.conv3d(
                tf.placeholder(tf.float32, shape=[32, 15, 20, 20, 3]),
                tf.placeholder(tf.float32, shape=[16, 20, 20, 3, 2]),
                strides=[1, 1, 1, 1, 1], padding="SAME")


if __name__ == "__main__":
    tf.test.main()
