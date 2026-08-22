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
"""Tests for compile utilities."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.keras.engine import compile_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class MetricsContainerTest(test.TestCase):

  @test_util.run_deprecated_v1
  def test_unknown_shape_metric_selection_error(self):
    metric_container = compile_utils.MetricsContainer('accuracy')
    y_t = array_ops.placeholder(dtypes.float32, shape=None)
    y_p = array_ops.placeholder(dtypes.float32, shape=None)
    with self.assertRaisesRegex(
        ValueError,
        'Unable to automatically select a metric for tensors with unknown '
        'shapes'):
      metric_container.update_state(y_t, y_p)


if __name__ == '__main__':
  test.main()
