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
"""Tests for the MatchingFilesDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.ops.dataset_ops import MatchingFilesDataset
from tensorflow.python.platform import test


class MatchingFilesDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def _build_iterator_graph(self, test_patterns):
    return MatchingFilesDataset(test_patterns)

  def testMatchingFilesCore(self):
    tmp_dir = tempfile.mkdtemp()
    width = 16
    depth = 8
    for i in range(width):
      for j in range(depth):
        new_base = os.path.join(tmp_dir, str(i),
                                *[str(dir_name) for dir_name in range(j)])
        if not os.path.exists(new_base):
          os.makedirs(new_base)
        for f in ['a.txt', 'b.py', 'c.pyc']:
          filename = os.path.join(new_base, f)
          open(filename, 'w').close()

    patterns = []
    for i in range(depth):
      pattern = os.path.join(tmp_dir,
                             os.path.join(*['**' for _ in range(i + 1)]),
                             '*.txt')
      patterns.append(pattern)

    num_outputs = width * depth
    self.run_core_tests(
        lambda: self._build_iterator_graph(patterns),
        lambda: self._build_iterator_graph(patterns[0:depth // 2]), num_outputs)

    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
  test.main()
