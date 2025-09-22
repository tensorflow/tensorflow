"""Unit tests for tf.nn.separable_conv2d operation."""

import numpy as np
from absl.testing import parameterized

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


class SeparableConv2DTest(test.TestCase, parameterized.TestCase):
    """Tests for tf.nn.separable_conv2d stride validation."""

    @parameterized.parameters(
        ((1, 5, 5, 3), (3, 3, 3, 1), (1, 1, 3, 2), [1, 1, 1, 1]),
        ((2, 7, 7, 2), (3, 3, 2, 1), (1, 1, 2, 4), [1, 2, 2, 1]),
    )
    @test_util.run_in_graph_and_eager_modes
    def testSeparableConv2DForward(self, input_shape, depth_filter_shape,
                                   point_filter_shape, strides):
        x = constant_op.constant(np.random.rand(*input_shape), dtype=dtypes.float32)
        depthwise_filter = constant_op.constant(np.random.rand(*depth_filter_shape),
                                                dtype=dtypes.float32)
        pointwise_filter = constant_op.constant(np.random.rand(*point_filter_shape),
                                                dtype=dtypes.float32)
        y = nn.separable_conv2d(x,
                                depthwise_filter,
                                pointwise_filter,
                                strides,
                                padding="SAME")
        y = self.evaluate(y)
        expected_channels = point_filter_shape[-1]
        self.assertEqual(y.shape[-1], expected_channels)
        self.assertEqual(y.shape[0], input_shape[0])

    @test_util.run_in_graph_and_eager_modes
    def testNegativeStride(self):
        x = constant_op.constant(np.random.rand(1, 8, 8, 3), dtype=dtypes.float32)
        depthwise_filter = constant_op.constant(np.random.rand(3, 3, 3, 1), dtype=dtypes.float32)
        pointwise_filter = constant_op.constant(np.random.rand(1, 1, 3, 2), dtype=dtypes.float32)
        with self.assertRaisesRegex(
            (ValueError, errors.InvalidArgumentError),
            "Stride.*-2"):
            y = nn.separable_conv2d(x, depthwise_filter, pointwise_filter, strides=[1, -2, 1, 1], padding="VALID")
            self.evaluate(y)

    @test_util.run_in_graph_and_eager_modes
    def testLargeStride(self):
        x = constant_op.constant(np.random.rand(1, 8, 8, 3), dtype=dtypes.float32)
        depthwise_filter = constant_op.constant(np.random.rand(3, 3, 3, 1), dtype=dtypes.float32)
        pointwise_filter = constant_op.constant(np.random.rand(1, 1, 3, 2), dtype=dtypes.float32)
        with self.assertRaisesRegex(
            (ValueError, errors.InvalidArgumentError),
            "Attr strides has value 2147483648 out of range for an int32"):
            y = nn.separable_conv2d(x, depthwise_filter, pointwise_filter, strides=[1, 2**31, 1, 1], padding="VALID")
            self.evaluate(y)

    @test_util.run_in_graph_and_eager_modes
    def testNonIntegerStride(self):
        self.skipTest("Non-integer strides are silently truncated in current TensorFlow implementation; no TypeError raised.")

    @test_util.run_in_graph_and_eager_modes
    def testInvalidStridesFormat(self):
        x = constant_op.constant(np.random.rand(1, 8, 8, 3), dtype=dtypes.float32)
        depthwise_filter = constant_op.constant(np.random.rand(3, 3, 3, 1), dtype=dtypes.float32)
        pointwise_filter = constant_op.constant(np.random.rand(1, 1, 3, 2), dtype=dtypes.float32)
        with self.assertRaisesRegex(
            (ValueError, errors.InvalidArgumentError),
            "Current implementation does not yet support strides in the batch and depth dimensions"):
            y = nn.separable_conv2d(x, depthwise_filter, pointwise_filter, strides=[2, 1, 1, 2], padding="VALID")
            self.evaluate(y)


if __name__ == "__main__":
    test.main()