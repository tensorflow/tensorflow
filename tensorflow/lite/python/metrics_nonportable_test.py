# Lint as: python2, python3
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
"""TensorFlow Lite Python metrics helper TFLiteMetrics check."""
from tensorflow.lite.python import metrics_nonportable as metrics
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class MetricsNonportableTest(test_util.TensorFlowTestCase):

  def test_TFLiteMetrics_creation_no_arg_success(self):
    metrics.TFLiteMetrics()

  def test_TFLiteMetrics_creation_arg_success(self):
    metrics.TFLiteMetrics('md5', '/path/to/model')

  def test_TFLiteMetrics_creation_fails_with_only_md5(self):
    with self.assertRaises(ValueError):
      metrics.TFLiteMetrics(md5='md5')

  def test_TFLiteMetrics_creation_fail2_with_only_model_path(self):
    with self.assertRaises(ValueError):
      metrics.TFLiteMetrics(model_path='/path/to/model')

  def test_debugger_creation_counter_increase_success(self):
    stub = metrics.TFLiteMetrics()
    stub.increase_counter_debugger_creation()
    self.assertEqual(stub._counter_debugger_creation.get_cell().value(), 1)

  def test_interpreter_creation_counter_increase_success(self):
    stub = metrics.TFLiteMetrics()
    stub.increase_counter_interpreter_creation()
    self.assertEqual(
        stub._counter_interpreter_creation.get_cell('python').value(), 1)


if __name__ == '__main__':
  test.main()
