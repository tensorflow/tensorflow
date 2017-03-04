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

import sys

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


class ExtractOutputFilePathTest(test_util.TensorFlowTestCase):

  def testNoOutputFilePathIsReflected(self):
    args, output_path = command_parser.extract_output_file_path(["pt", "a:0"])
    self.assertEqual(["pt", "a:0"], args)
    self.assertIsNone(output_path)

  def testHasOutputFilePathInOneArgsIsReflected(self):
    args, output_path = command_parser.extract_output_file_path(
        ["pt", "a:0", ">/tmp/foo.txt"])
    self.assertEqual(["pt", "a:0"], args)
    self.assertEqual(output_path, "/tmp/foo.txt")

  def testHasOutputFilePathInTwoArgsIsReflected(self):
    args, output_path = command_parser.extract_output_file_path(
        ["pt", "a:0", ">", "/tmp/foo.txt"])
    self.assertEqual(["pt", "a:0"], args)
    self.assertEqual(output_path, "/tmp/foo.txt")

  def testHasGreaterThanSignButNoFileNameCausesSyntaxError(self):
    with self.assertRaisesRegexp(SyntaxError, "Redirect file path is empty"):
      command_parser.extract_output_file_path(
          ["pt", "a:0", ">"])

  def testOutputPathMergedWithLastArgIsHandledCorrectly(self):
    args, output_path = command_parser.extract_output_file_path(
        ["pt", "a:0>/tmp/foo.txt"])
    self.assertEqual(["pt", "a:0"], args)
    self.assertEqual(output_path, "/tmp/foo.txt")

  def testOutputPathInLastArgGreaterThanInSecondLastIsHandledCorrectly(self):
    args, output_path = command_parser.extract_output_file_path(
        ["pt", "a:0>", "/tmp/foo.txt"])
    self.assertEqual(["pt", "a:0"], args)
    self.assertEqual(output_path, "/tmp/foo.txt")

  def testOneArgumentIsHandledCorrectly(self):
    args, output_path = command_parser.extract_output_file_path(["lt"])
    self.assertEqual(["lt"], args)
    self.assertIsNone(output_path)

  def testEmptyArgumentIsHandledCorrectly(self):
    args, output_path = command_parser.extract_output_file_path([])
    self.assertEqual([], args)
    self.assertIsNone(output_path)


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


class ParseIndicesTest(test_util.TensorFlowTestCase):

  def testParseValidIndicesStringsWithBrackets(self):
    self.assertEqual([0], command_parser.parse_indices("[0]"))
    self.assertEqual([0], command_parser.parse_indices(" [0] "))
    self.assertEqual([-1, 2], command_parser.parse_indices("[-1, 2]"))
    self.assertEqual([3, 4, -5],
                     command_parser.parse_indices("[3,4,-5]"))

  def testParseValidIndicesStringsWithoutBrackets(self):
    self.assertEqual([0], command_parser.parse_indices("0"))
    self.assertEqual([0], command_parser.parse_indices(" 0 "))
    self.assertEqual([-1, 2], command_parser.parse_indices("-1, 2"))
    self.assertEqual([3, 4, -5], command_parser.parse_indices("3,4,-5"))

  def testParseInvalidIndicesStringsWithoutBrackets(self):
    with self.assertRaisesRegexp(
        ValueError, r"invalid literal for int\(\) with base 10: 'a'"):
      self.assertEqual([0], command_parser.parse_indices("0,a"))

    with self.assertRaisesRegexp(
        ValueError, r"invalid literal for int\(\) with base 10: '2\]'"):
      self.assertEqual([0], command_parser.parse_indices("1, 2]"))

    with self.assertRaisesRegexp(
        ValueError, r"invalid literal for int\(\) with base 10: ''"):
      self.assertEqual([0], command_parser.parse_indices("3, 4,"))


class ParseRangesTest(test_util.TensorFlowTestCase):

  INF_VALUE = sys.float_info.max

  def testParseEmptyRangeString(self):
    self.assertEqual([], command_parser.parse_ranges(""))
    self.assertEqual([], command_parser.parse_ranges("  "))

  def testParseSingleRange(self):
    self.assertAllClose([[-0.1, 0.2]],
                        command_parser.parse_ranges("[-0.1, 0.2]"))
    self.assertAllClose([[-0.1, self.INF_VALUE]],
                        command_parser.parse_ranges("[-0.1, inf]"))
    self.assertAllClose([[-self.INF_VALUE, self.INF_VALUE]],
                        command_parser.parse_ranges("[-inf, inf]"))

  def testParseSingleListOfRanges(self):
    self.assertAllClose([[-0.1, 0.2], [10.0, 12.0]],
                        command_parser.parse_ranges("[[-0.1, 0.2], [10,  12]]"))
    self.assertAllClose(
        [[-self.INF_VALUE, -1.0], [1.0, self.INF_VALUE]],
        command_parser.parse_ranges("[[-inf, -1.0],[1.0, inf]]"))

  def testParseInvalidRangeString(self):
    with self.assertRaises(SyntaxError):
      command_parser.parse_ranges("[[1,2]")

    with self.assertRaisesRegexp(ValueError,
                                 "Incorrect number of elements in range"):
      command_parser.parse_ranges("[1,2,3]")

    with self.assertRaisesRegexp(ValueError,
                                 "Incorrect number of elements in range"):
      command_parser.parse_ranges("[inf]")

    with self.assertRaisesRegexp(ValueError,
                                 "Incorrect type in the 1st element of range"):
      command_parser.parse_ranges("[1j, 1]")

    with self.assertRaisesRegexp(ValueError,
                                 "Incorrect type in the 2nd element of range"):
      command_parser.parse_ranges("[1, 1j]")


class ParseReadableSizeStrTest(test_util.TensorFlowTestCase):

  def testParseNoUnitWorks(self):
    self.assertEqual(0, command_parser.parse_readable_size_str("0"))
    self.assertEqual(1024, command_parser.parse_readable_size_str("1024 "))
    self.assertEqual(2000, command_parser.parse_readable_size_str(" 2000 "))

  def testParseKiloBytesWorks(self):
    self.assertEqual(0, command_parser.parse_readable_size_str("0kB"))
    self.assertEqual(1024**2, command_parser.parse_readable_size_str("1024 kB"))
    self.assertEqual(1024**2 * 2,
                     command_parser.parse_readable_size_str("2048k"))
    self.assertEqual(1024**2 * 2,
                     command_parser.parse_readable_size_str("2048kB"))
    self.assertEqual(1024 / 4, command_parser.parse_readable_size_str("0.25k"))

  def testParseMegaBytesWorks(self):
    self.assertEqual(0, command_parser.parse_readable_size_str("0MB"))
    self.assertEqual(1024**3, command_parser.parse_readable_size_str("1024 MB"))
    self.assertEqual(1024**3 * 2,
                     command_parser.parse_readable_size_str("2048M"))
    self.assertEqual(1024**3 * 2,
                     command_parser.parse_readable_size_str("2048MB"))
    self.assertEqual(1024**2 / 4,
                     command_parser.parse_readable_size_str("0.25M"))

  def testParseGigaBytesWorks(self):
    self.assertEqual(0, command_parser.parse_readable_size_str("0GB"))
    self.assertEqual(1024**4, command_parser.parse_readable_size_str("1024 GB"))
    self.assertEqual(1024**4 * 2,
                     command_parser.parse_readable_size_str("2048G"))
    self.assertEqual(1024**4 * 2,
                     command_parser.parse_readable_size_str("2048GB"))
    self.assertEqual(1024**3 / 4,
                     command_parser.parse_readable_size_str("0.25G"))

  def testParseUnsupportedUnitRaisesException(self):
    with self.assertRaisesRegexp(
        ValueError, "Failed to parsed human-readable byte size str: \"0foo\""):
      command_parser.parse_readable_size_str("0foo")

    with self.assertRaisesRegexp(
        ValueError, "Failed to parsed human-readable byte size str: \"2E\""):
      command_parser.parse_readable_size_str("2EB")


if __name__ == "__main__":
  googletest.main()
