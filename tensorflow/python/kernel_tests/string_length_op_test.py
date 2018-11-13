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
"""Tests for string_length_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class StringLengthOpTest(test.TestCase):

  def testStringLength(self):
    strings = [[["1", "12"], ["123", "1234"], ["12345", "123456"]]]

    with self.test_session() as sess:
      lengths = string_ops.string_length(strings)
      values = sess.run(lengths)
      self.assertAllEqual(values, [[[1, 2], [3, 4], [5, 6]]])


if __name__ == "__main__":
  test.main()
