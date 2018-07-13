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
"""Tests for tensorflow.python.framework.errors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import error_interpolation
from tensorflow.python.platform import test


class InterpolateTest(test.TestCase):

  def testNothingToDo(self):
    normal_string = "This is just a normal string"
    interpolated_string = error_interpolation.interpolate(normal_string)
    self.assertEqual(interpolated_string, normal_string)

  def testOneTag(self):
    one_tag_string = "^^node:Foo:${file}^^"
    interpolated_string = error_interpolation.interpolate(one_tag_string)
    self.assertEqual(interpolated_string, "${file}")

  def testTwoTagsNoSeps(self):
    two_tags_no_seps = "^^node:Foo:${file}^^^^node:Bar:${line}^^"
    interpolated_string = error_interpolation.interpolate(two_tags_no_seps)
    self.assertEqual(interpolated_string, "${file}${line}")

  def testTwoTagsWithSeps(self):
    two_tags_with_seps = "123^^node:Foo:${file}^^456^^node:Bar:${line}^^789"
    interpolated_string = error_interpolation.interpolate(two_tags_with_seps)
    self.assertEqual(interpolated_string, "123${file}456${line}789")


if __name__ == "__main__":
  test.main()
