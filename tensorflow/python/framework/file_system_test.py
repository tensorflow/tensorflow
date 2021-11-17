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
# =============================================================================
"""Tests for file_system."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.framework import load_library
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class FileSystemTest(test.TestCase):

  def setUp(self):
    file_system_library = resource_loader.get_path_to_datafile(
        "test_file_system.so")
    load_library.load_file_system_library(file_system_library)

  @test_util.run_deprecated_v1
  def testBasic(self):
    with self.cached_session() as sess:
      reader = io_ops.WholeFileReader("test_reader")
      queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
      queue.enqueue_many([["test://foo"]]).run()
      queue.close().run()
      key, value = self.evaluate(reader.read(queue))
    self.assertEqual(key, compat.as_bytes("test://foo"))
    self.assertEqual(value, compat.as_bytes("AAAAAAAAAA"))


if __name__ == "__main__":
  test.main()
