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
"""Tests for SavedModel metrics Python bindings."""

from tensorflow.python.eager import test
from tensorflow.python.saved_model.experimental.pywrap_libexport import metrics


class MetricsTest(test.TestCase):

  def test_increment_write(self):
    self.assertEqual(metrics.GetWrite(), 0)
    metrics.IncrementWriteApi("foo")
    self.assertEqual(metrics.GetWriteApi("foo"), 1)
    metrics.IncrementWrite()
    self.assertEqual(metrics.GetWrite(), 1)

  def test_increment_read(self):
    self.assertEqual(metrics.GetRead(), 0)
    metrics.IncrementReadApi("bar")
    self.assertEqual(metrics.GetReadApi("bar"), 1)
    metrics.IncrementRead()
    self.assertEqual(metrics.GetRead(), 1)


if __name__ == "__main__":
  test.main()
