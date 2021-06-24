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
"""Unit tests for authoring package."""
# pylint: disable=g-direct-tensorflow-import

import tensorflow as tf

from tensorflow.lite.python.authoring import authoring
from tensorflow.python.eager import test
from tensorflow.python.platform import tf_logging as logging


class OpCallbacksTest(tf.test.TestCase):

  def test_simple_cosh(self):
    @authoring.compatible
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float32)
    ])
    def f(x):
      return tf.cosh(x)

    result = f(tf.constant([0.0]))
    log_messages = f.get_compatibility_log()
    self.assertEqual(result, tf.constant([1.0]))
    self.assertIn(
        "CompatibilityWarning: op 'Cosh' require(s) \"Select TF Ops\" for "
        "model conversion for TensorFlow Lite. "
        "https://www.tensorflow.org/lite/guide/ops_select", log_messages)

  def test_simple_cosh_raises_CompatibilityError(self):
    @authoring.compatible(raise_exception=True)
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float32)
    ])
    def f(x):
      return tf.cosh(x)

    # Check if the CompatibilityError exception is raised.
    with self.assertRaises(authoring.CompatibilityError):
      result = f(tf.constant([0.0]))
      del result
    log_messages = f.get_compatibility_log()
    self.assertIn(
        "CompatibilityWarning: op 'Cosh' require(s) \"Select TF Ops\" for "
        "model conversion for TensorFlow Lite. "
        "https://www.tensorflow.org/lite/guide/ops_select", log_messages)

  def test_flex_compatibility(self):
    @authoring.compatible
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[3, 3, 3, 3, 3], dtype=tf.float32)
    ])
    def f(inp):
      tanh = tf.math.tanh(inp)
      conv3d = tf.nn.conv3d(
          tanh,
          tf.ones([3, 3, 3, 3, 3]),
          strides=[1, 1, 1, 1, 1],
          padding="SAME")
      erf = tf.math.erf(conv3d)
      output = tf.math.tanh(erf)
      return output

    f(tf.ones(shape=(3, 3, 3, 3, 3), dtype=tf.float32))
    log_messages = f.get_compatibility_log()
    self.assertIn(
        "CompatibilityWarning: op 'Erf' require(s) \"Select TF Ops\" for model "
        "conversion for TensorFlow Lite. "
        "https://www.tensorflow.org/lite/guide/ops_select", log_messages)

  def test_compatibility_error(self):
    @authoring.compatible
    @tf.function
    def f():
      dataset = tf.data.Dataset.range(3)
      dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
      return dataset

    f()
    log_messages = f.get_compatibility_log()
    self.assertIn(
        "CompatibilityError: op 'DummySeedGenerator, RangeDataset, "
        "ShuffleDatasetV3' is(are) not natively supported by "
        "TensorFlow Lite. You need to provide a custom operator. "
        "https://www.tensorflow.org/lite/guide/ops_custom", log_messages)

  def test_simple_variable(self):
    external_var = tf.Variable(1.0)
    @authoring.compatible
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float32)
    ])
    def f(x):
      return x * external_var

    result = f(tf.constant(2.0, shape=(1)))
    log_messages = f.get_compatibility_log()

    self.assertEqual(result, tf.constant([2.0]))
    self.assertEmpty(log_messages)

  def test_class_method(self):
    class Model(tf.Module):

      @authoring.compatible
      @tf.function(input_signature=[
          tf.TensorSpec(shape=[None], dtype=tf.float32)
      ])
      def eval(self, x):
        return tf.cosh(x)

    m = Model()
    result = m.eval(tf.constant([0.0]))
    log_messages = m.eval.get_compatibility_log()

    self.assertEqual(result, tf.constant([1.0]))
    self.assertIn(
        "CompatibilityWarning: op 'Cosh' require(s) \"Select TF Ops\" for "
        "model conversion for TensorFlow Lite. "
        "https://www.tensorflow.org/lite/guide/ops_select", log_messages)

  def test_decorated_function_type(self):
    @authoring.compatible
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float32)
    ])
    def func(x):
      return tf.cos(x)

    result = func(tf.constant([0.0]))
    self.assertEqual(result, tf.constant([1.0]))

    # Check if the decorator keeps __name__ attribute.
    self.assertEqual(func.__name__, "func")

    # Check if the decorator works with get_concrete_function method.
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [func.get_concrete_function()])
    converter.convert()

  def test_decorated_class_method_type(self):
    class Model(tf.Module):

      @authoring.compatible
      @tf.function(input_signature=[
          tf.TensorSpec(shape=[None], dtype=tf.float32)
      ])
      def eval(self, x):
        return tf.cos(x)

    m = Model()
    result = m.eval(tf.constant([0.0]))
    self.assertEqual(result, tf.constant([1.0]))

    # Check if the decorator keeps __name__ attribute.
    self.assertEqual(m.eval.__name__, "eval")

    # Check if the decorator works with get_concrete_function method.
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [m.eval.get_concrete_function()])
    converter.convert()

  def test_simple_cosh_multiple(self):
    @authoring.compatible
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float32)
    ])
    def f(x):
      return tf.cosh(x)

    warning_messages = []

    def mock_warning(msg):
      warning_messages.append(msg)

    with test.mock.patch.object(logging, "warning", mock_warning):
      f(tf.constant([1.0]))
      f(tf.constant([2.0]))
      f(tf.constant([3.0]))

    # Test if compatiblility checks happens only once.
    self.assertEqual(1, len(warning_messages))

  def test_user_tf_ops_all_filtered(self):
    target_spec = tf.lite.TargetSpec()
    target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    target_spec.experimental_select_user_tf_ops = [
        "RangeDataset", "DummySeedGenerator", "ShuffleDatasetV3"
    ]
    @authoring.compatible(converter_target_spec=target_spec)
    @tf.function
    def f():
      dataset = tf.data.Dataset.range(3)
      dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
      return dataset

    f()
    log_messages = f.get_compatibility_log()
    self.assertEmpty(log_messages)

  def test_user_tf_ops_partial_filtered(self):
    target_spec = tf.lite.TargetSpec()
    target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    target_spec.experimental_select_user_tf_ops = [
        "DummySeedGenerator"
    ]
    @authoring.compatible(converter_target_spec=target_spec)

    @authoring.compatible(converter_target_spec=target_spec)
    @tf.function
    def f():
      dataset = tf.data.Dataset.range(3)
      dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
      return dataset

    f()
    log_messages = f.get_compatibility_log()
    self.assertIn(
        "CompatibilityError: op 'RangeDataset, ShuffleDatasetV3' is(are) not "
        "natively supported by TensorFlow Lite. You need to provide a custom "
        "operator. https://www.tensorflow.org/lite/guide/ops_custom",
        log_messages)

  def test_allow_custom_ops(self):
    @authoring.compatible(converter_allow_custom_ops=True)
    @tf.function
    def f():
      dataset = tf.data.Dataset.range(3)
      dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
      return dataset

    f()
    log_messages = f.get_compatibility_log()
    self.assertEmpty(log_messages)

if __name__ == "__main__":
  tf.test.main()
