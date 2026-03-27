# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for compile_utils."""

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine import compile_utils
from tensorflow.python.platform import test as test_lib


class MetricsContainerTest(test_lib.TestCase):

  def test_unknown_shape_raises_clear_error(self):
    """Metric auto-selection raises a clear error for unknown shapes.

    When dataset elements come from tf.numpy_function without set_shape(),
    shapes are unknown. The error message should guide the user to set
    shapes or use an explicit metric object.
    """
    container = compile_utils.MetricsContainer(metrics=['accuracy'])

    y_t_spec = tf.TensorSpec(shape=None, dtype=tf.float32)
    y_p_spec = tf.TensorSpec(shape=None, dtype=tf.float32)

    @tf.function(input_signature=[y_t_spec, y_p_spec])
    def call_get_metric(y_true, y_pred):
      container._get_metric_object('accuracy', y_true, y_pred)
      return tf.constant(0.0)

    with self.assertRaisesRegex(ValueError, 'unknown shapes'):
      call_get_metric(
          tf.constant([1.0, 0.0]),
          tf.constant([0.9, 0.1]))


class DisplayShapeTest(test_lib.TestCase):

  def test_display_shape_unknown_rank(self):
    """display_shape should handle unknown-rank shapes gracefully."""
    from tensorflow.python.keras.engine import input_spec  # pylint: disable=g-import-not-at-top
    unknown_shape = tensor_shape.TensorShape(None)
    result = input_spec.display_shape(unknown_shape)
    self.assertIsInstance(result, str)

  def test_display_shape_known_rank(self):
    """display_shape works normally for known-rank shapes."""
    from tensorflow.python.keras.engine import input_spec  # pylint: disable=g-import-not-at-top
    known_shape = tensor_shape.TensorShape([2, 3])
    result = input_spec.display_shape(known_shape)
    self.assertEqual(result, '(2, 3)')


if __name__ == '__main__':
  test_lib.main()
