# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""Tests for SequenceFileDataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.contrib.hadoop.python.ops import hadoop_dataset_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test


class SequenceFileDatasetTest(test.TestCase):

  def test_sequence_file_dataset(self):
    """Test case for SequenceFileDataset.

    The file is generated with `org.apache.hadoop.io.Text` for key/value.
    There are 25 records in the file with the format of:
    key = XXX
    value = VALUEXXX
    where XXX is replaced as the line number (starts with 001).
    """
    filename = os.path.join(resource_loader.get_data_files_path(),
                            "testdata", "string.seq")

    filenames = constant_op.constant([filename], dtypes.string)
    num_repeats = 2

    dataset = hadoop_dataset_ops.SequenceFileDataset(filenames).repeat(
        num_repeats)
    iterator = dataset_ops.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for _ in range(num_repeats):  # Dataset is repeated.
        for i in range(25):  # 25 records.
          v0 = b"%03d" % (i + 1)
          v1 = b"VALUE%03d" % (i + 1)
          self.assertEqual((v0, v1), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)


if __name__ == "__main__":
  test.main()
