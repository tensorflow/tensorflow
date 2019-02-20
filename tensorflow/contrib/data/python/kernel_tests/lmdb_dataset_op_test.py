#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for LMDBDatasetOp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

from tensorflow.contrib.data.python.ops import readers
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.util import compat

prefix_path = "tensorflow/core/lib"


@test_util.run_v1_only("deprecated API, no eager or V2 test coverage")
class LMDBDatasetTest(test_base.DatasetTestBase):

  def setUp(self):
    super(LMDBDatasetTest, self).setUp()
    # Copy database out because we need the path to be writable to use locks.
    path = os.path.join(prefix_path, "lmdb", "testdata", "data.mdb")
    self.db_path = os.path.join(self.get_temp_dir(), "data.mdb")
    shutil.copy(path, self.db_path)

  def testReadFromFile(self):
    filename = self.db_path

    filenames = constant_op.constant([filename], dtypes.string)
    num_repeats = 2

    dataset = readers.LMDBDataset(filenames).repeat(num_repeats)
    iterator = dataset_ops.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for _ in range(num_repeats):  # Dataset is repeated.
        for i in range(10):  # 10 records.
          k = compat.as_bytes(str(i))
          v = compat.as_bytes(str(chr(ord("a") + i)))
          self.assertEqual((k, v), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)


if __name__ == "__main__":
  test.main()
