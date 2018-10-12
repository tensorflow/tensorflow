# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================
"""Functional tests for UnicodeScript op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class UnicodeScriptOpTest(test.TestCase):

  def testValidScripts(self):
    inputs = [
        ord("a"),
        0x0411,  # CYRILLIC CAPITAL LETTER BE
        0x82b8,  # CJK UNIFIED IDEOGRAPH-82B8
        ord(",")
    ]
    with self.cached_session():
      input_vector = constant_op.constant(inputs, dtypes.int32)
      outputs = string_ops.unicode_script(input_vector).eval()
      self.assertAllEqual(
          outputs,
          [
              25,  # USCRIPT_LATIN (LATN)
              8,  # USCRIPT_CYRILLIC (CYRL)
              17,  # USCRIPT_HAN (HANI)
              0  # USCRIPT_COMMON (ZYYY)
          ])

  def testInvalidScript(self):
    inputs = [-100, 0xffffff]
    with self.cached_session():
      input_vector = constant_op.constant(inputs, dtypes.int32)
      outputs = string_ops.unicode_script(input_vector).eval()
      self.assertAllEqual(outputs, [-1, -1])


if __name__ == "__main__":
  test.main()
