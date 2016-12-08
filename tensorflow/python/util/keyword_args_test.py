# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Keyword args tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.util import keyword_args


class KeywordArgsTest(tf.test.TestCase):

  def test_keyword_args_only(self):
    def func_without_decorator(a, b):
      return a+b

    @keyword_args.keyword_args_only
    def func_with_decorator(a, b):
      return func_without_decorator(a, b)

    self.assertEqual(3, func_without_decorator(1, 2))
    self.assertEqual(3, func_without_decorator(a=1, b=2))
    self.assertEqual(3, func_with_decorator(a=1, b=2))

    # Providing non-keyword args should fail.
    with self.assertRaisesRegexp(
        ValueError, "Must use keyword args to call func_with_decorator."):
      self.assertEqual(3, func_with_decorator(1, 2))

    # Partially providing keyword args should fail.
    with self.assertRaisesRegexp(
        ValueError, "Must use keyword args to call func_with_decorator."):
      self.assertEqual(3, func_with_decorator(1, b=2))


if __name__ == "__main__":
  tf.test.main()
