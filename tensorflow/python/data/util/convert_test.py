# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for utilities working with user input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.util import convert
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class ConvertTest(test.TestCase):

  def testInteger(self):
    resp = convert.optional_param_to_tensor("foo", 3)
    with self.test_session() as sess:
      self.assertEqual(3, sess.run(resp))

  def testIntegerDefault(self):
    resp = convert.optional_param_to_tensor("foo", None)
    with self.test_session() as sess:
      self.assertEqual(0, sess.run(resp))

  def testStringDefault(self):
    resp = convert.optional_param_to_tensor("bar", None, "default",
                                            dtypes.string)
    with self.test_session() as sess:
      self.assertEqual(compat.as_bytes("default"), sess.run(resp))

  def testString(self):
    resp = convert.optional_param_to_tensor("bar", "value", "default",
                                            dtypes.string)
    with self.test_session() as sess:
      self.assertEqual(compat.as_bytes("value"), sess.run(resp))


if __name__ == "__main__":
  test.main()
