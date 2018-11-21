# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf 2.0 upgrader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tempfile
import six
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import tf_upgrade_v2


class TestUpgrade(test_util.TensorFlowTestCase):
  """Test various APIs that have been changed in 2.0.

  We also test whether a converted file is executable. test_file_v1_10.py
  aims to exhaustively test that API changes are convertible and actually
  work when run with current TensorFlow.
  """

  def _upgrade(self, old_file_text):
    in_file = six.StringIO(old_file_text)
    out_file = six.StringIO()
    upgrader = ast_edits.ASTCodeUpgrader(tf_upgrade_v2.TFAPIChangeSpec())
    count, report, errors = (
        upgrader.process_opened_file("test.py", in_file,
                                     "test_out.py", out_file))
    return count, report, errors, out_file.getvalue()

  def testParseError(self):
    _, report, unused_errors, unused_new_text = self._upgrade(
        "import tensorflow as tf\na + \n")
    self.assertTrue(report.find("Failed to parse") != -1)

  def testReport(self):
    text = "tf.assert_near(a)\n"
    _, report, unused_errors, unused_new_text = self._upgrade(text)
    # This is not a complete test, but it is a sanity test that a report
    # is generating information.
    self.assertTrue(report.find("Renamed function `tf.assert_near` to "
                                "`tf.debugging.assert_near`"))

  def testRename(self):
    text = "tf.conj(a)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.math.conj(a)\n")
    text = "tf.rsqrt(tf.log_sigmoid(3.8))\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.math.rsqrt(tf.math.log_sigmoid(3.8))\n")

  def testRenameConstant(self):
    text = "tf.MONOLITHIC_BUILD\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.sysconfig.MONOLITHIC_BUILD\n")
    text = "some_call(tf.MONOLITHIC_BUILD)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "some_call(tf.sysconfig.MONOLITHIC_BUILD)\n")

  def testRenameArgs(self):
    text = ("tf.nn.pool(input_a, window_shape_a, pooling_type_a, padding_a, "
            "dilation_rate_a, strides_a, name_a, data_format_a)\n")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text,
                     ("tf.nn.pool(input=input_a, window_shape=window_shape_a,"
                      " pooling_type=pooling_type_a, padding=padding_a, "
                      "dilations=dilation_rate_a, strides=strides_a, "
                      "name=name_a, data_format=data_format_a)\n"))

  def testReorder(self):
    text = "tf.boolean_mask(a, b, c, d)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text,
                     "tf.boolean_mask(tensor=a, mask=b, name=c, axis=d)\n")

  def testLearningRateDecay(self):
    for decay in ["tf.train.exponential_decay", "tf.train.piecewise_constant",
                  "tf.train.polynomial_decay", "tf.train.natural_exp_decay",
                  "tf.train.inverse_time_decay", "tf.train.cosine_decay",
                  "tf.train.cosine_decay_restarts",
                  "tf.train.linear_cosine_decay",
                  "tf.train.noisy_linear_cosine_decay"]:

      text = "%s(a, b)\n" % decay
      _, report, errors, new_text = self._upgrade(text)
      self.assertEqual(text, new_text)
      self.assertEqual(errors, ["test.py:1: %s requires manual check." % decay])
      self.assertIn("%s has been changed" % decay, report)

  def testEstimatorLossReductionChange(self):
    classes = [
        "LinearClassifier", "LinearRegressor", "DNNLinearCombinedClassifier",
        "DNNLinearCombinedRegressor", "DNNRegressor", "DNNClassifier",
        "BaselineClassifier", "BaselineRegressor"
    ]
    for c in classes:
      ns = "tf.estimator." + c
      text = ns + "(a, b)"
      _, report, errors, new_text = self._upgrade(text)
      self.assertEqual(text, new_text)
      self.assertEqual(errors, ["test.py:1: %s requires manual check." % ns])
      self.assertIn("loss_reduction has been changed", report)

  def testCountNonZeroChanges(self):
    text = (
        "tf.math.count_nonzero(input_tensor=input, dtype=dtype, name=name, "
        "reduction_indices=axis, keep_dims=keepdims)\n"
        )
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    expected_text = (
        "tf.math.count_nonzero(input=input, dtype=dtype, name=name, "
        "axis=axis, keepdims=keepdims)\n"
        )
    self.assertEqual(new_text, expected_text)

  def testRandomMultinomialToRandomCategorical(self):
    text = (
        "tf.random.multinomial(logits, samples, seed, name, output_dtype)\n"
        )
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    expected_text = (
        "tf.random.categorical(logits=logits, num_samples=samples, seed=seed, "
        "name=name, dtype=output_dtype)\n"
        )
    self.assertEqual(new_text, expected_text)

    text = (
        "tf.multinomial(logits, samples, seed, name, output_dtype)\n"
        )
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    expected_text = (
        "tf.random.categorical(logits=logits, num_samples=samples, seed=seed, "
        "name=name, dtype=output_dtype)\n"
        )
    self.assertEqual(new_text, expected_text)

  def testConvolutionOpUpdate(self):
    text = (
        "tf.nn.convolution(input, filter, padding, strides, dilation_rate, "
        "name, data_format)"
    )
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    expected_text = (
        "tf.nn.convolution(input=input, filters=filter, padding=padding, "
        "strides=strides, dilations=dilation_rate, name=name, "
        "data_format=data_format)"
    )
    self.assertEqual(new_text, expected_text)


class TestUpgradeFiles(test_util.TensorFlowTestCase):

  def testInplace(self):
    """Check to make sure we don't have a file system race."""
    temp_file = tempfile.NamedTemporaryFile("w", delete=False)
    original = "tf.conj(a)\n"
    upgraded = "tf.math.conj(a)\n"
    temp_file.write(original)
    temp_file.close()
    upgrader = ast_edits.ASTCodeUpgrader(tf_upgrade_v2.TFAPIChangeSpec())
    upgrader.process_file(temp_file.name, temp_file.name)
    self.assertAllEqual(open(temp_file.name).read(), upgraded)
    os.unlink(temp_file.name)


if __name__ == "__main__":
  test_lib.main()
