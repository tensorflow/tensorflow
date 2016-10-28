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
"""Tests for TensorFlow Debugger command parser."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.debug.cli import command_parser
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class ParseCommandTest(test_util.TensorFlowTestCase):

  def testParseNoBracketsOrQuotes(self):
    command = ""
    self.assertEqual([], command_parser.parse_command(command))

    command = "a"
    self.assertEqual(["a"], command_parser.parse_command(command))

    command = "foo bar baz qux"
    self.assertEqual(["foo", "bar", "baz", "qux"],
                     command_parser.parse_command(command))

    command = "foo bar\tbaz\t qux"
    self.assertEqual(["foo", "bar", "baz", "qux"],
                     command_parser.parse_command(command))

  def testParseLeadingTrailingWhitespaces(self):
    command = "  foo bar baz qux   "
    self.assertEqual(["foo", "bar", "baz", "qux"],
                     command_parser.parse_command(command))

    command = "\nfoo bar baz qux\n"
    self.assertEqual(["foo", "bar", "baz", "qux"],
                     command_parser.parse_command(command))

  def testParseCommandsWithBrackets(self):
    command = "pt foo[1, 2, :]"
    self.assertEqual(["pt", "foo[1, 2, :]"],
                     command_parser.parse_command(command))
    command = "pt  foo[1, 2, :]   -a"
    self.assertEqual(["pt", "foo[1, 2, :]", "-a"],
                     command_parser.parse_command(command))

    command = "inject_value foo [1, 2,:] 0"
    self.assertEqual(["inject_value", "foo", "[1, 2,:]", "0"],
                     command_parser.parse_command(command))

  def testParseCommandWithTwoArgsContainingBrackets(self):
    command = "pt foo[1, :] bar[:, 2]"
    self.assertEqual(["pt", "foo[1, :]", "bar[:, 2]"],
                     command_parser.parse_command(command))

    command = "pt foo[] bar[:, 2]"
    self.assertEqual(["pt", "foo[]", "bar[:, 2]"],
                     command_parser.parse_command(command))

  def testParseCommandWithUnmatchedBracket(self):
    command = "pt  foo[1, 2, :"
    self.assertNotEqual(["pt", "foo[1, 2, :]"],
                        command_parser.parse_command(command))

  def testParseCommandsWithQuotes(self):
    command = "inject_value foo \"np.zeros([100, 500])\""
    self.assertEqual(["inject_value", "foo", "np.zeros([100, 500])"],
                     command_parser.parse_command(command))
    # The pair of double quotes should have been stripped.

    command = "\"command prefix with spaces\" arg1"
    self.assertEqual(["command prefix with spaces", "arg1"],
                     command_parser.parse_command(command))

  def testParseCommandWithTwoArgsContainingQuotes(self):
    command = "foo \"bar\" \"qux\""
    self.assertEqual(["foo", "bar", "qux"],
                     command_parser.parse_command(command))

    command = "foo \"\" \"qux\""
    self.assertEqual(["foo", "", "qux"],
                     command_parser.parse_command(command))


class ParseTensorNameTest(test_util.TensorFlowTestCase):

  def testParseTensorNameWithoutSlicing(self):
    (tensor_name,
     tensor_slicing) = command_parser.parse_tensor_name_with_slicing(
         "hidden/weights/Variable:0")

    self.assertEqual("hidden/weights/Variable:0", tensor_name)
    self.assertEqual("", tensor_slicing)

  def testParseTensorNameWithSlicing(self):
    (tensor_name,
     tensor_slicing) = command_parser.parse_tensor_name_with_slicing(
         "hidden/weights/Variable:0[:, 1]")

    self.assertEqual("hidden/weights/Variable:0", tensor_name)
    self.assertEqual("[:, 1]", tensor_slicing)


class ValidateSlicingStringTest(test_util.TensorFlowTestCase):

  def testValidateValidSlicingStrings(self):
    self.assertTrue(command_parser.validate_slicing_string("[1]"))
    self.assertTrue(command_parser.validate_slicing_string("[2,3]"))
    self.assertTrue(command_parser.validate_slicing_string("[4, 5, 6]"))
    self.assertTrue(command_parser.validate_slicing_string("[7,:, :]"))

  def testValidateInvalidSlicingStrings(self):
    self.assertFalse(command_parser.validate_slicing_string(""))
    self.assertFalse(command_parser.validate_slicing_string("[1,"))
    self.assertFalse(command_parser.validate_slicing_string("2,3]"))
    self.assertFalse(command_parser.validate_slicing_string("[4, foo()]"))
    self.assertFalse(command_parser.validate_slicing_string("[5, bar]"))


if __name__ == "__main__":
  googletest.main()
