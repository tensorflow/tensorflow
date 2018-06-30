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
"""Tests for shape_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.signal.python.kernel_tests import test_util
from tensorflow.contrib.signal.python.ops import shape_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class FrameTest(test.TestCase):

  def test_mapping_of_indices_without_padding(self):
    with self.test_session(use_gpu=True):
      tensor = constant_op.constant(np.arange(9152), dtypes.int32)
      tensor = array_ops.expand_dims(tensor, 0)

      result = shape_ops.frame(tensor, 512, 180, pad_end=False).eval()

      expected = np.tile(np.arange(512), (49, 1))
      expected += np.tile(np.arange(49) * 180, (512, 1)).T

      expected = np.expand_dims(expected, axis=0)
      expected = np.array(expected, dtype=np.int32)

      self.assertAllEqual(expected, result)

  def test_mapping_of_indices_with_padding(self):
    with self.test_session(use_gpu=True):
      tensor = constant_op.constant(np.arange(10000), dtypes.int32)
      tensor = array_ops.expand_dims(tensor, 0)

      result = shape_ops.frame(tensor, 512, 192, pad_end=True).eval()

      expected = np.tile(np.arange(512), (53, 1))
      expected += np.tile(np.arange(53) * 192, (512, 1)).T

      expected[expected >= 10000] = 0

      expected = np.expand_dims(expected, axis=0)
      expected = np.array(expected, dtype=np.int32)

      self.assertAllEqual(expected, result)

  def test_invalid_inputs(self):
    # Rank 0 input signal.
    with self.assertRaises(ValueError):
      shape_ops.frame(1, 1, 1)

    # If the rank is unknown, do not raise an exception.
    shape_ops.frame(array_ops.placeholder(dtypes.float32), 1, 1)

    # Non-scalar frame_length.
    with self.assertRaises(ValueError):
      shape_ops.frame([1], [1], 1)

    # Non-scalar frame_step.
    with self.assertRaises(ValueError):
      shape_ops.frame([1], 1, [1])

    # Non-scalar pad_value.
    with self.assertRaises(ValueError):
      shape_ops.frame([1], 1, 1, pad_end=True, pad_value=[1])

  def test_length_zero(self):
    signal = constant_op.constant([], dtype=dtypes.float32)
    frame_length = 2
    frame_step = 1

    with self.test_session(use_gpu=True):
      result = shape_ops.frame(signal, frame_length, frame_step,
                               pad_end=True, pad_value=99).eval()
      self.assertEqual((0, 2), result.shape)

      result = shape_ops.frame(signal, frame_length, frame_step,
                               pad_end=False).eval()
      self.assertEqual((0, 2), result.shape)

  def test_shape_inference(self):
    signal = array_ops.placeholder(dtypes.int32, shape=[1, 1])
    frame_length = 2
    frame_step = 1
    # Shape inference is able to detect the rank and inner-most dimension
    # if frame_length is known at graph definition time.
    result = shape_ops.frame(signal, frame_length, frame_step,
                             pad_end=True, pad_value=99)
    self.assertEqual([1, 1, 2], result.shape.as_list())

    result = shape_ops.frame(signal, frame_length, frame_step,
                             pad_end=False)
    self.assertEqual([1, 0, 2], result.shape.as_list())

    # If frame_length is not known, rank and (known) outer and inner dimensions
    # are inferred.
    signal = array_ops.placeholder(dtypes.int32, shape=[1, 2, 3, 4])
    frame_length = array_ops.placeholder(dtypes.int32, shape=[])
    frame_step = 1
    result = shape_ops.frame(signal, frame_length, frame_step,
                             pad_end=True, pad_value=99, axis=1)
    self.assertEqual([1, 2, None, 3, 4], result.shape.as_list())

    result = shape_ops.frame(signal, frame_length, frame_step,
                             pad_end=False, axis=1)
    self.assertEqual([1, None, None, 3, 4], result.shape.as_list())

    # If frame_length and inner-most dimension is known, rank, inner dimensions,
    # and known outer dimensions are inferred.
    signal = array_ops.placeholder(dtypes.int32,
                                   shape=[None, 5, None, 20, 5, 3])
    frame_length = 4
    frame_step = 3
    result = shape_ops.frame(signal, frame_length, frame_step,
                             pad_end=True, pad_value=99, axis=3)
    self.assertEqual([None, 5, None, 7, 4, 5, 3], result.shape.as_list())

    result = shape_ops.frame(signal, frame_length, frame_step,
                             pad_end=False, axis=3)
    self.assertEqual([None, 5, None, 6, 4, 5, 3], result.shape.as_list())

    # Test that shape inference is consistent with actual returned shapes for
    # small values of signal_length, frame_length, frame_step, and pad_end in
    # [True, False].
    frame_step = 1
    for signal_length in range(2):
      signal = [0] * signal_length
      for frame_length in range(2):
        for pad_end in [False, True]:
          op = shape_ops.frame(signal, frame_length, frame_step,
                               pad_end=pad_end, pad_value=99)
          with self.test_session(use_gpu=True):
            result = op.eval()
          self.assertEqual(op.shape.as_list(), list(result.shape))

  def test_basic_mono(self):
    signal = np.arange(6)
    frame_length = 3
    frame_step = 2

    with self.test_session(use_gpu=True):
      for rank in range(5):
        nd_signal = np.reshape(signal, (1,) * rank + signal.shape)

        # With padding, we pad the last frame with pad_value.
        result = shape_ops.frame(nd_signal, frame_length, frame_step,
                                 pad_end=True, pad_value=99).eval()
        expected_inner_frames = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 99]])
        expected = np.reshape(
            expected_inner_frames, (1,) * rank + expected_inner_frames.shape)
        self.assertAllEqual(expected, result)

        # Without padding, we drop the last frame.
        expected_inner_frames = np.array([[0, 1, 2], [2, 3, 4]])
        expected = np.reshape(
            expected_inner_frames, (1,) * rank + expected_inner_frames.shape)
        result = shape_ops.frame(nd_signal, frame_length, frame_step,
                                 pad_end=False).eval()
        self.assertAllEqual(expected, result)

  def test_basic_stereo(self):
    signal = np.vstack([np.arange(6),
                        np.arange(6) + 10])
    frame_length = 3
    frame_step = 2

    with self.test_session(use_gpu=True):
      for rank in range(5):
        nd_signal = np.reshape(signal, (1,) * rank + signal.shape)

        # With padding, we pad the last frame with pad_value.
        result = shape_ops.frame(nd_signal, frame_length, frame_step,
                                 pad_end=True, pad_value=99).eval()
        expected_inner_frames = np.array([
            [[0, 1, 2], [2, 3, 4], [4, 5, 99]],
            [[10, 11, 12], [12, 13, 14], [14, 15, 99]]])
        expected = np.reshape(
            expected_inner_frames, (1,) * rank + expected_inner_frames.shape)
        self.assertAllEqual(expected, result)

        # Without padding, we drop the last frame.
        expected_inner_frames = np.array([[[0, 1, 2], [2, 3, 4]],
                                          [[10, 11, 12], [12, 13, 14]]])
        expected = np.reshape(
            expected_inner_frames, (1,) * rank + expected_inner_frames.shape)
        result = shape_ops.frame(nd_signal, frame_length, frame_step,
                                 pad_end=False).eval()
        self.assertAllEqual(expected, result)

  def test_complex_shape(self):
    signal = np.vstack([np.arange(6),
                        np.arange(6) + 10,
                        np.arange(6) + 20,
                        np.arange(6) + 30,
                        np.arange(6) + 40,
                        np.arange(6) + 50])
    signal = np.reshape(signal, (2, 1, 3, 1, 6))
    frame_length = 3
    frame_step = 2

    with self.test_session(use_gpu=True):
      # With padding, we pad the last frame with pad_value.
      result = shape_ops.frame(signal, frame_length, frame_step,
                               pad_end=True, pad_value=99).eval()
      # Resulting shape is (2, 1, 3, 1, 3, 3).
      expected = [[[[[[0, 1, 2], [2, 3, 4], [4, 5, 99]]],
                    [[[10, 11, 12], [12, 13, 14], [14, 15, 99]]],
                    [[[20, 21, 22], [22, 23, 24], [24, 25, 99]]]]],
                  [[[[[30, 31, 32], [32, 33, 34], [34, 35, 99]]],
                    [[[40, 41, 42], [42, 43, 44], [44, 45, 99]]],
                    [[[50, 51, 52], [52, 53, 54], [54, 55, 99]]]]]]
      self.assertAllEqual(expected, result)

      result = shape_ops.frame(signal, frame_length, frame_step,
                               pad_end=False).eval()
      # Resulting shape is (2, 1, 3, 1, 3, 2).
      expected = [[[[[[0, 1, 2], [2, 3, 4]]],
                    [[[10, 11, 12], [12, 13, 14]]],
                    [[[20, 21, 22], [22, 23, 24]]]]],
                  [[[[[30, 31, 32], [32, 33, 34]]],
                    [[[40, 41, 42], [42, 43, 44]]],
                    [[[50, 51, 52], [52, 53, 54]]]]]]
      self.assertAllEqual(expected, result)

  def test_axis(self):
    signal = np.reshape(np.arange(16), (2, 4, 2))
    with self.test_session(use_gpu=True):
      result = shape_ops.frame(signal, frame_length=2, frame_step=2,
                               pad_end=True, axis=1)
      expected = np.reshape(np.arange(16), (2, 2, 2, 2))
      self.assertAllEqual(expected, result.eval())

      result = shape_ops.frame(signal, frame_length=2, frame_step=1,
                               pad_end=True, axis=1)
      expected = [[[[0, 1], [2, 3]],
                   [[2, 3], [4, 5]],
                   [[4, 5], [6, 7]],
                   [[6, 7], [0, 0]]],
                  [[[8, 9], [10, 11]],
                   [[10, 11], [12, 13]],
                   [[12, 13], [14, 15]],
                   [[14, 15], [0, 0]]]]
      self.assertAllEqual(expected, result.eval())

      result = shape_ops.frame(signal, frame_length=3, frame_step=1,
                               pad_end=True, axis=1)
      expected = [[[[0, 1], [2, 3], [4, 5]],
                   [[2, 3], [4, 5], [6, 7]],
                   [[4, 5], [6, 7], [0, 0]],
                   [[6, 7], [0, 0], [0, 0]]],
                  [[[8, 9], [10, 11], [12, 13]],
                   [[10, 11], [12, 13], [14, 15]],
                   [[12, 13], [14, 15], [0, 0]],
                   [[14, 15], [0, 0], [0, 0]]]]
      self.assertAllEqual(expected, result.eval())

  def test_window_larger_than_signal(self):
    signal = constant_op.constant([[1, 2], [11, 12]], dtype=dtypes.float32)
    frame_length = 4
    frame_step = 1

    with self.test_session(use_gpu=True):
      result = shape_ops.frame(signal, frame_length, frame_step,
                               pad_end=True, pad_value=99).eval()
      self.assertAllClose([[[1, 2, 99, 99], [2, 99, 99, 99]],
                           [[11, 12, 99, 99], [12, 99, 99, 99]]], result)

      result = shape_ops.frame(signal, frame_length, frame_step,
                               pad_end=False).eval()
      self.assertEqual((2, 0, 4), result.shape)

      frame_step = 2
      result = shape_ops.frame(signal, frame_length, frame_step,
                               pad_end=True, pad_value=99).eval()
      self.assertAllClose([[[1, 2, 99, 99]], [[11, 12, 99, 99]]], result)

      result = shape_ops.frame(signal, frame_length, frame_step,
                               pad_end=False).eval()
      self.assertEqual((2, 0, 4), result.shape)

  def test_preserves_type(self):
    signal = math_ops.range(10, dtype=dtypes.float64)
    frame_length = 2
    frame_step = 3

    with self.test_session(use_gpu=True):
      result = shape_ops.frame(signal, frame_length, frame_step)
      self.assertEqual(result.dtype, signal.dtype)

  def test_dynamic_tensor(self):
    # Show that frame works even when the dimensions of its input are
    # not known at graph creation time.
    input_signal = np.vstack([np.arange(4), np.arange(4) + 10,
                              np.arange(4) + 20])
    frame_length = 2
    frame_step = 2

    with self.test_session(use_gpu=True) as sess:
      signal_placeholder = array_ops.placeholder(shape=(None, None),
                                                 dtype=dtypes.float32)
      result = sess.run(shape_ops.frame(
          signal_placeholder, frame_length, frame_step),
                        feed_dict={signal_placeholder: input_signal})
      self.assertAllEqual([[[0, 1], [2, 3]],
                           [[10, 11], [12, 13]],
                           [[20, 21], [22, 23]]], result)

  def test_gradient_numerical(self):
    with self.test_session(use_gpu=True):
      signal_shape = (2, 128)
      signal = array_ops.ones(signal_shape)
      frame_length = 33
      frame_step = 9
      frames = shape_ops.frame(signal, frame_length, frame_step)
      error = test.compute_gradient_error(
          signal, signal_shape, frames, frames.shape.as_list())
      self.assertLess(error, 2e-5)

  def test_constant_folding(self):
    """frame should be constant foldable for constant inputs."""
    for pad_end in [True, False]:
      g = ops.Graph()
      with g.as_default():
        frame_length, frame_step = 32, 16
        signal_shape = (2, 128)
        signal = array_ops.ones(signal_shape)
        frames = shape_ops.frame(signal, frame_length, frame_step,
                                 pad_end=pad_end)
        rewritten_graph = test_util.grappler_optimize(g, [frames])
        self.assertEqual(1, len(rewritten_graph.node))


if __name__ == "__main__":
  test.main()
