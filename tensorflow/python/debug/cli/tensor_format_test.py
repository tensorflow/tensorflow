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
"""Unit tests for tensor formatter."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import numpy as np

from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.debug.cli import cli_test_utils
from tensorflow.python.debug.cli import tensor_format
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class RichTextLinesTest(test_util.TensorFlowTestCase):

  def setUp(self):
    np.set_printoptions(
        precision=8, threshold=1000, edgeitems=3, linewidth=75)

  def _checkTensorMetadata(self, tensor, annotations):
    self.assertEqual(
        {"dtype": tensor.dtype, "shape": tensor.shape},
        annotations["tensor_metadata"])

  # Regular expression for text representation of float numbers, possibly in
  # engineering notation.
  _ELEMENT_REGEX = re.compile(
      r"([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?|nan|inf|-inf)")

  def _checkBeginIndicesAnnotations(self, out, a):
    """Check the beginning-index annotations of an ndarray representation.

    Args:
      out: An instance of RichTextLines representing a numpy.ndarray.
      a: The numpy.ndarray being represented.

    Raises:
      ValueError: if any ellipses ("...") are found in the lines representing
        the array.
    """
    begin_line_num = 0
    while not out.lines[begin_line_num].startswith("array"):
      begin_line_num += 1
    element_index = 0
    for line_num in range(begin_line_num, len(out.lines)):
      line = out.lines[line_num]
      if "..." in line:
        raise ValueError("Unexpected found ellipses in line representing array")
      matches = re.finditer(self._ELEMENT_REGEX, line)
      for line_item_index, _ in enumerate(matches):
        subscripts = list(np.unravel_index(element_index, a.shape))
        if line_item_index == 0:
          self.assertEqual({tensor_format.BEGIN_INDICES_KEY: subscripts},
                           out.annotations[line_num])
        element_index += 1
    self.assertEqual(element_index, np.size(a))

  def _checkTensorElementLocations(self, out, a):
    """Check the results of locate_tensor_element on an ndarray representation.

    that represents a numpy.ndaray.

    Args:
      out: An instance of RichTextLines representing a numpy.ndarray.
      a: The numpy.ndarray being represented.

    Raises:
      ValueError: if any ellipses ("...") are found in the lines representing
        the array.
    """
    # First, locate the beginning of the tensor value section.
    begin_line_num = 0
    while not out.lines[begin_line_num].startswith("array"):
      begin_line_num += 1
    # Second, find all matches to tensor-value regex.
    element_index = 0
    for line_num in range(begin_line_num, len(out.lines)):
      line = out.lines[line_num]
      if "..." in line:
        raise ValueError("Unexpected found ellipses in line representing array")
      matches = re.finditer(self._ELEMENT_REGEX, line)
      for match in matches:
        subscripts = list(np.unravel_index(element_index, a.shape))
        is_omitted, row, start_col, end_col = (
            tensor_format.locate_tensor_element(out, subscripts))
        self.assertFalse(is_omitted)
        self.assertEqual(line_num, row)
        self.assertEqual(match.start(), start_col)
        self.assertEqual(match.end(), end_col)
        element_index += 1
    self.assertEqual(element_index, np.size(a))

  def _findFirst(self, lines, string):
    """Find first occurrence of a string in a list of strings."""
    for i, line in enumerate(lines):
      find_index = line.find(string)
      if find_index >= 0:
        return i, find_index

  def _extractBoldNumbers(self, out, start_line):
    """Extract all numbers that have the bold font attribute.

    Args:
      out: An instance of RichTextLines.
      start_line: 0-based index to start from.

    Returns:
      A list of floats.
    """
    floats = []
    for i in range(start_line, len(out.lines)):
      if i not in out.font_attr_segs:
        continue
      line_attrs = out.font_attr_segs[i]
      for begin, end, attr_value in line_attrs:
        if attr_value == "bold":
          floats.append(float(out.lines[i][begin:end]))
    return floats

  def testFormatZeroDimensionTensor(self):
    a = np.array(42, dtype=np.int32)

    out = tensor_format.format_tensor(a, "a")

    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self, ["Tensor \"a\":", ""], out.lines[:2])
    self.assertTrue(out.lines[2].startswith("array(42"))
    self._checkTensorMetadata(a, out.annotations)

  def testFormatTensorHighlightsTensorNameWithoutDebugOp(self):
    tensor_name = "a_tensor:0"
    a = np.zeros(2)
    out = tensor_format.format_tensor(
        a, tensor_name, np_printoptions={"linewidth": 40})
    self.assertEqual([(8, 8 + len(tensor_name), "bold")], out.font_attr_segs[0])

  def testFormatTensorHighlightsTensorNameWithDebugOp(self):
    tensor_name = "a_tensor:0"
    debug_op = "DebugIdentity"
    a = np.zeros(2)
    out = tensor_format.format_tensor(
        a, "%s:%s" % (tensor_name, debug_op), np_printoptions={"linewidth": 40})
    self.assertEqual([(8, 8 + len(tensor_name), "bold"),
                      (8 + len(tensor_name) + 1,
                       8 + len(tensor_name) + 1 + len(debug_op), "yellow")],
                     out.font_attr_segs[0])

  def testFormatTensor1DNoEllipsis(self):
    a = np.zeros(20)

    out = tensor_format.format_tensor(
        a, "a", np_printoptions={"linewidth": 40})

    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self, ["Tensor \"a\":", ""], out.lines[:2])
    self.assertEqual(repr(a).split("\n"), out.lines[2:])

    self._checkTensorMetadata(a, out.annotations)

    # Check annotations for beginning indices of the lines.
    self._checkBeginIndicesAnnotations(out, a)

  def testFormatTensor2DNoEllipsisNoRowBreak(self):
    a = np.linspace(0.0, 1.0 - 1.0 / 16.0, 16).reshape([4, 4])

    out = tensor_format.format_tensor(a, "a")

    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self, ["Tensor \"a\":", ""], out.lines[:2])
    self.assertEqual(repr(a).split("\n"), out.lines[2:])

    self._checkTensorMetadata(a, out.annotations)
    self._checkBeginIndicesAnnotations(out, a)

  def testFormatTensorSuppressingTensorName(self):
    a = np.linspace(0.0, 1.0 - 1.0 / 16.0, 16).reshape([4, 4])

    out = tensor_format.format_tensor(a, None)
    self.assertEqual(repr(a).split("\n"), out.lines)

    self._checkTensorMetadata(a, out.annotations)
    self._checkBeginIndicesAnnotations(out, a)

  def testFormatTensorWithMetadata(self):
    a = np.linspace(0.0, 1.0 - 1.0 / 16.0, 16).reshape([4, 4])

    out = tensor_format.format_tensor(a, "a", include_metadata=True)

    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self,
        ["Tensor \"a\":",
         "  dtype: float64",
         "  shape: (4, 4)",
         ""], out.lines[:4])
    self.assertEqual(repr(a).split("\n"), out.lines[4:])

    self._checkTensorMetadata(a, out.annotations)
    self._checkBeginIndicesAnnotations(out, a)

  def testFormatTensor2DNoEllipsisWithRowBreak(self):
    a = np.linspace(0.0, 1.0 - 1.0 / 40.0, 40).reshape([2, 20])

    out = tensor_format.format_tensor(
        a, "a", np_printoptions={"linewidth": 50})

    self.assertEqual(
        {"dtype": a.dtype, "shape": a.shape},
        out.annotations["tensor_metadata"])

    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self, ["Tensor \"a\":", ""], out.lines[:2])
    self.assertEqual(repr(a).split("\n"), out.lines[2:])

    self._checkTensorMetadata(a, out.annotations)

    # Check annotations for the beginning indices of the lines.
    self._checkBeginIndicesAnnotations(out, a)

  def testFormatTensor3DNoEllipsis(self):
    a = np.linspace(0.0, 1.0 - 1.0 / 24.0, 24).reshape([2, 3, 4])

    out = tensor_format.format_tensor(a, "a")

    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self, ["Tensor \"a\":", ""], out.lines[:2])
    self.assertEqual(repr(a).split("\n"), out.lines[2:])

    self._checkTensorMetadata(a, out.annotations)
    self._checkBeginIndicesAnnotations(out, a)

  def testFormatTensor3DNoEllipsisWithArgwhereHighlightWithMatches(self):
    a = np.linspace(0.0, 1.0 - 1.0 / 24.0, 24).reshape([2, 3, 4])

    lower_bound = 0.26
    upper_bound = 0.5

    def highlight_filter(x):
      return np.logical_and(x > lower_bound, x < upper_bound)

    highlight_options = tensor_format.HighlightOptions(
        highlight_filter, description="between 0.26 and 0.5")
    out = tensor_format.format_tensor(
        a, "a", highlight_options=highlight_options)

    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self,
        ["Tensor \"a\": "
         "Highlighted(between 0.26 and 0.5): 5 of 24 element(s) (20.83%)",
         ""],
        out.lines[:2])
    self.assertEqual(repr(a).split("\n"), out.lines[2:])

    self._checkTensorMetadata(a, out.annotations)

    # Check annotations for beginning indices of the lines.
    self._checkBeginIndicesAnnotations(out, a)

    self.assertAllClose(
        [0.29166667, 0.33333333, 0.375, 0.41666667, 0.45833333],
        self._extractBoldNumbers(out, 2))

  def testFormatTensor3DNoEllipsisWithArgwhereHighlightWithNoMatches(self):
    a = np.linspace(0.0, 1.0 - 1.0 / 24.0, 24).reshape([2, 3, 4])

    def highlight_filter(x):
      return x > 10.0

    highlight_options = tensor_format.HighlightOptions(highlight_filter)
    out = tensor_format.format_tensor(
        a, "a", highlight_options=highlight_options)

    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self,
        ["Tensor \"a\": Highlighted: 0 of 24 element(s) (0.00%)", ""],
        out.lines[:2])
    self.assertEqual(repr(a).split("\n"), out.lines[2:])

    self._checkTensorMetadata(a, out.annotations)
    self._checkBeginIndicesAnnotations(out, a)

    # Check font attribute segments for highlighted elements.
    for i in range(2, len(out.lines)):
      self.assertNotIn(i, out.font_attr_segs)

  def testFormatTensorWithEllipses(self):
    a = (np.arange(11 * 11 * 11) + 1000).reshape([11, 11, 11]).astype(np.int32)

    out = tensor_format.format_tensor(
        a, "a", False, np_printoptions={"threshold": 100, "edgeitems": 2})

    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self, ["Tensor \"a\":", ""], out.lines[:2])
    self.assertEqual(repr(a).split("\n"), out.lines[2:])

    self._checkTensorMetadata(a, out.annotations)

    # Check annotations for beginning indices of the lines.
    actual_row_0_0_0, _ = self._findFirst(out.lines, "1000")
    self.assertEqual({tensor_format.BEGIN_INDICES_KEY: [0, 0, 0]},
                     out.annotations[actual_row_0_0_0])
    actual_row_0_1_0, _ = self._findFirst(out.lines, "1011")
    self.assertEqual({tensor_format.BEGIN_INDICES_KEY: [0, 1, 0]},
                     out.annotations[actual_row_0_1_0])
    # Find the first line that is completely omitted.
    omitted_line = 2
    while not out.lines[omitted_line].strip().startswith("..."):
      omitted_line += 1
    self.assertEqual({tensor_format.OMITTED_INDICES_KEY: [0, 2, 0]},
                     out.annotations[omitted_line])

    actual_row_10_10_0, _ = self._findFirst(out.lines, "2320")
    self.assertEqual({tensor_format.BEGIN_INDICES_KEY: [10, 10, 0]},
                     out.annotations[actual_row_10_10_0])
    # Find the last line that is completely omitted.
    omitted_line = len(out.lines) - 1
    while not out.lines[omitted_line].strip().startswith("..."):
      omitted_line -= 1
    self.assertEqual({tensor_format.OMITTED_INDICES_KEY: [10, 2, 0]},
                     out.annotations[omitted_line])

  def testFormatUninitializedTensor(self):
    tensor_proto = tensor_pb2.TensorProto(
        dtype=types_pb2.DataType.Value("DT_FLOAT"),
        tensor_shape=tensor_shape_pb2.TensorShapeProto(
            dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)]))
    out = tensor_format.format_tensor(
        debug_data.InconvertibleTensorProto(tensor_proto, False), "a")

    self.assertEqual(["Tensor \"a\":", "", "Uninitialized tensor:"],
                     out.lines[:3])
    self.assertEqual(str(tensor_proto).split("\n"), out.lines[3:])

  def testFormatResourceTypeTensor(self):
    tensor_proto = tensor_pb2.TensorProto(
        dtype=types_pb2.DataType.Value("DT_RESOURCE"),
        tensor_shape=tensor_shape_pb2.TensorShapeProto(
            dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)]))
    out = tensor_format.format_tensor(
        debug_data.InconvertibleTensorProto(tensor_proto), "a")

    self.assertEqual(["Tensor \"a\":", ""], out.lines[:2])
    self.assertEqual(str(tensor_proto).split("\n"), out.lines[2:])

  def testLocateTensorElement1DNoEllipsis(self):
    a = np.zeros(20)

    out = tensor_format.format_tensor(
        a, "a", np_printoptions={"linewidth": 40})

    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self, ["Tensor \"a\":", ""], out.lines[:2])
    self.assertEqual(repr(a).split("\n"), out.lines[2:])

    self._checkTensorElementLocations(out, a)

    with self.assertRaisesRegexp(
        ValueError, "Indices exceed tensor dimensions"):
      tensor_format.locate_tensor_element(out, [20])

    with self.assertRaisesRegexp(
        ValueError, "Indices contain negative"):
      tensor_format.locate_tensor_element(out, [-1])

    with self.assertRaisesRegexp(
        ValueError, "Dimensions mismatch"):
      tensor_format.locate_tensor_element(out, [0, 0])

  def testLocateTensorElement1DNoEllipsisBatchMode(self):
    a = np.zeros(20)

    out = tensor_format.format_tensor(
        a, "a", np_printoptions={"linewidth": 40})

    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self, ["Tensor \"a\":", ""], out.lines[:2])
    self.assertEqual(repr(a).split("\n"), out.lines[2:])

    self._checkTensorElementLocations(out, a)

  def testBatchModeWithErrors(self):
    a = np.zeros(20)

    out = tensor_format.format_tensor(
        a, "a", np_printoptions={"linewidth": 40})

    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self, ["Tensor \"a\":", ""], out.lines[:2])
    self.assertEqual(repr(a).split("\n"), out.lines[2:])

    with self.assertRaisesRegexp(ValueError, "Dimensions mismatch"):
      tensor_format.locate_tensor_element(out, [[0, 0], [0]])

    with self.assertRaisesRegexp(ValueError,
                                 "Indices exceed tensor dimensions"):
      tensor_format.locate_tensor_element(out, [[0], [20]])

    with self.assertRaisesRegexp(ValueError,
                                 r"Indices contain negative value\(s\)"):
      tensor_format.locate_tensor_element(out, [[0], [-1]])

    with self.assertRaisesRegexp(
        ValueError, "Input indices sets are not in ascending order"):
      tensor_format.locate_tensor_element(out, [[5], [0]])

  def testLocateTensorElement1DTinyAndNanValues(self):
    a = np.ones([3, 3]) * 1e-8
    a[1, 0] = np.nan
    a[1, 2] = np.inf

    out = tensor_format.format_tensor(
        a, "a", np_printoptions={"linewidth": 100})

    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self, ["Tensor \"a\":", ""], out.lines[:2])
    self.assertEqual(repr(a).split("\n"), out.lines[2:])

    self._checkTensorElementLocations(out, a)

  def testLocateTensorElement2DNoEllipsis(self):
    a = np.linspace(0.0, 1.0 - 1.0 / 16.0, 16).reshape([4, 4])

    out = tensor_format.format_tensor(a, "a")

    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self, ["Tensor \"a\":", ""], out.lines[:2])
    self.assertEqual(repr(a).split("\n"), out.lines[2:])

    self._checkTensorElementLocations(out, a)

    with self.assertRaisesRegexp(
        ValueError, "Indices exceed tensor dimensions"):
      tensor_format.locate_tensor_element(out, [1, 4])

    with self.assertRaisesRegexp(
        ValueError, "Indices contain negative"):
      tensor_format.locate_tensor_element(out, [-1, 2])

    with self.assertRaisesRegexp(
        ValueError, "Dimensions mismatch"):
      tensor_format.locate_tensor_element(out, [0])

  def testLocateTensorElement2DNoEllipsisWithNumericSummary(self):
    a = np.linspace(0.0, 1.0 - 1.0 / 16.0, 16).reshape([4, 4])

    out = tensor_format.format_tensor(a, "a", include_numeric_summary=True)

    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self,
        ["Tensor \"a\":",
         "",
         "Numeric summary:",
         "|  0  + | total |",
         "|  1 15 |    16 |",
         "|           min           max          mean           std |"],
        out.lines[:6])
    cli_test_utils.assert_array_lines_close(
        self, [0.0, 0.9375, 0.46875, 0.28811076429], out.lines[6:7])
    cli_test_utils.assert_array_lines_close(self, a, out.lines[8:])

    self._checkTensorElementLocations(out, a)

    with self.assertRaisesRegexp(
        ValueError, "Indices exceed tensor dimensions"):
      tensor_format.locate_tensor_element(out, [1, 4])

    with self.assertRaisesRegexp(
        ValueError, "Indices contain negative"):
      tensor_format.locate_tensor_element(out, [-1, 2])

    with self.assertRaisesRegexp(
        ValueError, "Dimensions mismatch"):
      tensor_format.locate_tensor_element(out, [0])

  def testLocateTensorElement3DWithEllipses(self):
    a = (np.arange(11 * 11 * 11) + 1000).reshape([11, 11, 11]).astype(np.int32)

    out = tensor_format.format_tensor(
        a, "a", False, np_printoptions={"threshold": 100, "edgeitems": 2})

    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self, ["Tensor \"a\":", ""], out.lines[:2])

    actual_row_0_0_0, actual_col_0_0_0 = self._findFirst(out.lines, "1000")
    is_omitted, row, start_col, end_col = tensor_format.locate_tensor_element(
        out, [0, 0, 0])
    self.assertFalse(is_omitted)
    self.assertEqual(actual_row_0_0_0, row)
    self.assertEqual(actual_col_0_0_0, start_col)
    self.assertEqual(actual_col_0_0_0 + 4, end_col)

    actual_row_0_0_10, _ = self._findFirst(out.lines, "1010")
    is_omitted, row, start_col, end_col = tensor_format.locate_tensor_element(
        out, [0, 0, 10])
    self.assertFalse(is_omitted)
    self.assertEqual(actual_row_0_0_10, row)
    self.assertIsNone(start_col)  # Passes ellipsis.
    self.assertIsNone(end_col)

    actual_row_0_1_0, actual_col_0_1_0 = self._findFirst(out.lines, "1011")
    is_omitted, row, start_col, end_col = tensor_format.locate_tensor_element(
        out, [0, 1, 0])
    self.assertFalse(is_omitted)
    self.assertEqual(actual_row_0_1_0, row)
    self.assertEqual(actual_col_0_1_0, start_col)
    self.assertEqual(actual_col_0_1_0 + 4, end_col)

    is_omitted, row, start_col, end_col = tensor_format.locate_tensor_element(
        out, [0, 2, 0])
    self.assertTrue(is_omitted)  # In omitted line.
    self.assertIsNone(start_col)
    self.assertIsNone(end_col)

    is_omitted, row, start_col, end_col = tensor_format.locate_tensor_element(
        out, [0, 2, 10])
    self.assertTrue(is_omitted)  # In omitted line.
    self.assertIsNone(start_col)
    self.assertIsNone(end_col)

    is_omitted, row, start_col, end_col = tensor_format.locate_tensor_element(
        out, [0, 8, 10])
    self.assertTrue(is_omitted)  # In omitted line.
    self.assertIsNone(start_col)
    self.assertIsNone(end_col)

    actual_row_0_10_1, actual_col_0_10_1 = self._findFirst(out.lines, "1111")
    is_omitted, row, start_col, end_col = tensor_format.locate_tensor_element(
        out, [0, 10, 1])
    self.assertFalse(is_omitted)
    self.assertEqual(actual_row_0_10_1, row)
    self.assertEqual(actual_col_0_10_1, start_col)
    self.assertEqual(actual_col_0_10_1 + 4, end_col)

    is_omitted, row, start_col, end_col = tensor_format.locate_tensor_element(
        out, [5, 1, 1])
    self.assertTrue(is_omitted)  # In omitted line.
    self.assertIsNone(start_col)
    self.assertIsNone(end_col)

    actual_row_10_10_10, _ = self._findFirst(out.lines, "2330")
    is_omitted, row, start_col, end_col = tensor_format.locate_tensor_element(
        out, [10, 10, 10])
    self.assertFalse(is_omitted)
    self.assertEqual(actual_row_10_10_10, row)
    self.assertIsNone(start_col)  # Past ellipsis.
    self.assertIsNone(end_col)

    with self.assertRaisesRegexp(
        ValueError, "Indices exceed tensor dimensions"):
      tensor_format.locate_tensor_element(out, [11, 5, 5])

    with self.assertRaisesRegexp(
        ValueError, "Indices contain negative"):
      tensor_format.locate_tensor_element(out, [-1, 5, 5])

    with self.assertRaisesRegexp(
        ValueError, "Dimensions mismatch"):
      tensor_format.locate_tensor_element(out, [5, 5])

  def testLocateTensorElement3DWithEllipsesBatchMode(self):
    a = (np.arange(11 * 11 * 11) + 1000).reshape([11, 11, 11]).astype(np.int32)

    out = tensor_format.format_tensor(
        a, "a", False, np_printoptions={"threshold": 100,
                                        "edgeitems": 2})

    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self, ["Tensor \"a\":", ""], out.lines[:2])
    self.assertEqual(repr(a).split("\n"), out.lines[2:])

    actual_row_0_0_0, actual_col_0_0_0 = self._findFirst(out.lines, "1000")
    actual_row_0_0_10, _ = self._findFirst(out.lines, "1010")
    actual_row_10_10_10, _ = self._findFirst(out.lines, "2330")

    (are_omitted, rows, start_cols,
     end_cols) = tensor_format.locate_tensor_element(out, [[0, 0, 0]])
    self.assertEqual([False], are_omitted)
    self.assertEqual([actual_row_0_0_0], rows)
    self.assertEqual([actual_col_0_0_0], start_cols)
    self.assertEqual([actual_col_0_0_0 + 4], end_cols)

    (are_omitted, rows, start_cols,
     end_cols) = tensor_format.locate_tensor_element(out,
                                                     [[0, 0, 0], [0, 0, 10]])
    self.assertEqual([False, False], are_omitted)
    self.assertEqual([actual_row_0_0_0, actual_row_0_0_10], rows)
    self.assertEqual([actual_col_0_0_0, None], start_cols)
    self.assertEqual([actual_col_0_0_0 + 4, None], end_cols)

    (are_omitted, rows, start_cols,
     end_cols) = tensor_format.locate_tensor_element(out,
                                                     [[0, 0, 0], [0, 2, 0]])
    self.assertEqual([False, True], are_omitted)
    self.assertEqual([2, 4], rows)
    self.assertEqual(2, len(start_cols))
    self.assertEqual(2, len(end_cols))

    (are_omitted, rows, start_cols,
     end_cols) = tensor_format.locate_tensor_element(out,
                                                     [[0, 0, 0], [10, 10, 10]])
    self.assertEqual([False, False], are_omitted)
    self.assertEqual([actual_row_0_0_0, actual_row_10_10_10], rows)
    self.assertEqual([actual_col_0_0_0, None], start_cols)
    self.assertEqual([actual_col_0_0_0 + 4, None], end_cols)

  def testLocateTensorElementAnnotationsUnavailable(self):
    tensor_proto = tensor_pb2.TensorProto(
        dtype=types_pb2.DataType.Value("DT_FLOAT"),
        tensor_shape=tensor_shape_pb2.TensorShapeProto(
            dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)]))
    out = tensor_format.format_tensor(
        debug_data.InconvertibleTensorProto(tensor_proto, False), "a")

    self.assertEqual(["Tensor \"a\":", "", "Uninitialized tensor:"],
                     out.lines[:3])

    with self.assertRaisesRegexp(
        AttributeError, "tensor_metadata is not available in annotations"):
      tensor_format.locate_tensor_element(out, [0])


class NumericSummaryTest(test_util.TensorFlowTestCase):

  def testNumericSummaryOnFloatFullHouse(self):
    x = np.array([np.nan, np.nan, -np.inf, np.inf, np.inf, np.inf, -2, -3, -4,
                  0, 1, 2, 2, 2, 2, 0, 0, 0, np.inf, np.inf, np.inf])
    out = tensor_format.numeric_summary(x)
    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self,
        ["|  nan -inf    -    0    + +inf | total |",
         "|    2    1    3    4    5    6 |    21 |",
         "|     min     max    mean    std |"], out.lines[:3])
    cli_test_utils.assert_array_lines_close(
        self, [-4.0, 2.0, 0.0, 1.95789002075], out.lines[3:4])

  def testNumericSummaryOnFloatMissingCategories(self):
    x = np.array([np.nan, np.nan])
    out = tensor_format.numeric_summary(x)
    self.assertEqual(2, len(out.lines))
    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self, ["| nan | total |", "|   2 |     2 |"], out.lines[:2])

    x = np.array([-np.inf, np.inf, 0, 0, np.inf, np.inf])
    out = tensor_format.numeric_summary(x)
    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self,
        ["| -inf    0 +inf | total |",
         "|    1    2    3 |     6 |",
         "|  min  max mean  std |"], out.lines[:3])
    cli_test_utils.assert_array_lines_close(
        self, [0.0, 0.0, 0.0, 0.0], out.lines[3:4])

    x = np.array([-120, 120, 130])
    out = tensor_format.numeric_summary(x)
    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self,
        ["| - + | total |",
         "| 1 2 |     3 |",
         "|       min       max     mean      std |"],
        out.lines[:3])
    cli_test_utils.assert_array_lines_close(
        self, [-120, 130, 43.3333333333, 115.566238822], out.lines[3:4])

  def testNumericSummaryOnEmptyFloat(self):
    x = np.array([], dtype=np.float32)
    out = tensor_format.numeric_summary(x)
    self.assertEqual(["No numeric summary available due to empty tensor."],
                     out.lines)

  def testNumericSummaryOnInt(self):
    x = np.array([-3] * 50 + [3] * 200 + [0], dtype=np.int32)
    out = tensor_format.numeric_summary(x)
    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self,
        ["|   -   0   + | total |",
         "|  50   1 200 |   251 |",
         "|      min     max    mean     std |"],
        out.lines[:3])
    cli_test_utils.assert_array_lines_close(
        self, [-3, 3, 1.79282868526, 2.39789673081], out.lines[3:4])

  def testNumericSummaryOnBool(self):
    x = np.array([False, True, True, False], dtype=np.bool)
    out = tensor_format.numeric_summary(x)
    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self,
        ["| False  True | total |", "|     2     2 |     4 |"], out.lines)

    x = np.array([True] * 10, dtype=np.bool)
    out = tensor_format.numeric_summary(x)
    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self, ["| True | total |", "|   10 |    10 |"], out.lines)

    x = np.array([False] * 10, dtype=np.bool)
    out = tensor_format.numeric_summary(x)
    cli_test_utils.assert_lines_equal_ignoring_whitespace(
        self, ["| False | total |", "|    10 |    10 |"], out.lines)

    x = np.array([], dtype=np.bool)
    out = tensor_format.numeric_summary(x)
    self.assertEqual(["No numeric summary available due to empty tensor."],
                     out.lines)

  def testNumericSummaryOnStrTensor(self):
    x = np.array(["spam", "egg"], dtype=np.object)
    out = tensor_format.numeric_summary(x)
    self.assertEqual(
        ["No numeric summary available due to tensor dtype: object."],
        out.lines)


if __name__ == "__main__":
  googletest.main()
