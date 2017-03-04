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
"""Tests for learn.python.learn.dataframe.transforms.csv_parser."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.learn.python.learn.dataframe.transforms import csv_parser
from tensorflow.contrib.learn.python.learn.tests.dataframe import mocks
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test


class CSVParserTestCase(test.TestCase):

  def testParse(self):
    parser = csv_parser.CSVParser(
        column_names=["col0", "col1", "col2"], default_values=["", "", 1.4])
    csv_lines = ["one,two,2.5", "four,five,6.0"]
    csv_input = constant_op.constant(
        csv_lines, dtype=dtypes.string, shape=[len(csv_lines)])
    csv_column = mocks.MockSeries("csv", csv_input)
    expected_output = [
        np.array([b"one", b"four"]), np.array([b"two", b"five"]),
        np.array([2.5, 6.0])
    ]
    output_columns = parser(csv_column)
    self.assertEqual(3, len(output_columns))
    cache = {}
    output_tensors = [o.build(cache) for o in output_columns]
    self.assertEqual(3, len(output_tensors))
    with self.test_session() as sess:
      output = sess.run(output_tensors)
      for expected, actual in zip(expected_output, output):
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
  test.main()
