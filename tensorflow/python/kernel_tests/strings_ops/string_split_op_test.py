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
"""Tests for string_split_op."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class StringSplitOpTest(test.TestCase, parameterized.TestCase):

  def testStringSplit(self):
    strings = ["pigs on the wing", "animals"]

    with self.cached_session():
      tokens = string_ops.string_split(strings)
      indices, values, shape = self.evaluate(tokens)
      self.assertAllEqual(indices, [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0]])
      self.assertAllEqual(values, [b"pigs", b"on", b"the", b"wing", b"animals"])
      self.assertAllEqual(shape, [2, 4])

  @test_util.run_deprecated_v1
  def testStringSplitEmptyDelimiter(self):
    strings = ["hello", "hola", b"\xF0\x9F\x98\x8E"]  # Last string is U+1F60E

    with self.cached_session():
      tokens = string_ops.string_split(strings, delimiter="")
      indices, values, shape = self.evaluate(tokens)
      self.assertAllEqual(indices, [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
                                    [1, 0], [1, 1], [1, 2], [1, 3], [2, 0],
                                    [2, 1], [2, 2], [2, 3]])
      expected = np.array(
          [
              "h", "e", "l", "l", "o", "h", "o", "l", "a", b"\xf0", b"\x9f",
              b"\x98", b"\x8e"
          ],
          dtype="|S1")
      self.assertAllEqual(values.tolist(), expected)
      self.assertAllEqual(shape, [3, 5])

  def testStringSplitEmptyToken(self):
    strings = ["", " a", "b ", " c", " ", " d ", "  e", "f  ", "  g  ", "  "]

    with self.cached_session():
      tokens = string_ops.string_split(strings)
      indices, values, shape = self.evaluate(tokens)
      self.assertAllEqual(
          indices,
          [[1, 0], [2, 0], [3, 0], [5, 0], [6, 0], [7, 0], [8, 0]])
      self.assertAllEqual(values, [b"a", b"b", b"c", b"d", b"e", b"f", b"g"])
      self.assertAllEqual(shape, [10, 1])

  def testStringSplitOnSetEmptyToken(self):
    strings = ["", " a", "b ", " c", " ", " d ", ". e", "f .", " .g. ", " ."]

    with self.cached_session():
      tokens = string_ops.string_split(strings, delimiter=" .")
      indices, values, shape = self.evaluate(tokens)
      self.assertAllEqual(
          indices,
          [[1, 0], [2, 0], [3, 0], [5, 0], [6, 0], [7, 0], [8, 0]])
      self.assertAllEqual(values, [b"a", b"b", b"c", b"d", b"e", b"f", b"g"])
      self.assertAllEqual(shape, [10, 1])

  @test_util.run_deprecated_v1
  def testStringSplitWithDelimiter(self):
    strings = ["hello|world", "hello world"]

    with self.cached_session():
      self.assertRaises(
          ValueError, string_ops.string_split, strings, delimiter=["|", ""])

      self.assertRaises(
          ValueError, string_ops.string_split, strings, delimiter=["a"])

      tokens = string_ops.string_split(strings, delimiter="|")
      indices, values, shape = self.evaluate(tokens)
      self.assertAllEqual(indices, [[0, 0], [0, 1], [1, 0]])
      self.assertAllEqual(values, [b"hello", b"world", b"hello world"])
      self.assertAllEqual(shape, [2, 2])

      tokens = string_ops.string_split(strings, delimiter="| ")
      indices, values, shape = self.evaluate(tokens)
      self.assertAllEqual(indices, [[0, 0], [0, 1], [1, 0], [1, 1]])
      self.assertAllEqual(values, [b"hello", b"world", b"hello", b"world"])
      self.assertAllEqual(shape, [2, 2])

  @test_util.run_deprecated_v1
  def testStringSplitWithDelimiterTensor(self):
    strings = ["hello|world", "hello world"]

    with self.cached_session() as sess:
      delimiter = array_ops.placeholder(dtypes.string)

      tokens = string_ops.string_split(strings, delimiter=delimiter)

      with self.assertRaises(errors_impl.InvalidArgumentError):
        sess.run(tokens, feed_dict={delimiter: ["a", "b"]})
      with self.assertRaises(errors_impl.InvalidArgumentError):
        sess.run(tokens, feed_dict={delimiter: ["a"]})
      indices, values, shape = sess.run(tokens, feed_dict={delimiter: "|"})

      self.assertAllEqual(indices, [[0, 0], [0, 1], [1, 0]])
      self.assertAllEqual(values, [b"hello", b"world", b"hello world"])
      self.assertAllEqual(shape, [2, 2])

  @test_util.run_deprecated_v1
  def testStringSplitWithDelimitersTensor(self):
    strings = ["hello.cruel,world", "hello cruel world"]

    with self.cached_session() as sess:
      delimiter = array_ops.placeholder(dtypes.string)

      tokens = string_ops.string_split(strings, delimiter=delimiter)

      with self.assertRaises(errors_impl.InvalidArgumentError):
        sess.run(tokens, feed_dict={delimiter: ["a", "b"]})
      with self.assertRaises(errors_impl.InvalidArgumentError):
        sess.run(tokens, feed_dict={delimiter: ["a"]})
      indices, values, shape = sess.run(tokens, feed_dict={delimiter: ".,"})

      self.assertAllEqual(indices, [[0, 0], [0, 1], [0, 2], [1, 0]])
      self.assertAllEqual(values,
                          [b"hello", b"cruel", b"world", b"hello cruel world"])
      self.assertAllEqual(shape, [2, 3])

  def testStringSplitWithNoSkipEmpty(self):
    strings = ["#a", "b#", "#c#"]

    with self.cached_session():
      tokens = string_ops.string_split(strings, "#", skip_empty=False)
      indices, values, shape = self.evaluate(tokens)
      self.assertAllEqual(indices, [[0, 0], [0, 1],
                                    [1, 0], [1, 1],
                                    [2, 0], [2, 1], [2, 2]])
      self.assertAllEqual(values, [b"", b"a", b"b", b"", b"", b"c", b""])
      self.assertAllEqual(shape, [3, 3])

    with self.cached_session():
      tokens = string_ops.string_split(strings, "#")
      indices, values, shape = self.evaluate(tokens)
      self.assertAllEqual(values, [b"a", b"b", b"c"])
      self.assertAllEqual(indices, [[0, 0], [1, 0], [2, 0]])
      self.assertAllEqual(shape, [3, 1])

  @parameterized.named_parameters([
      dict(
          testcase_name="RaggedResultType",
          source=[b"pigs on the wing", b"animals"],
          result_type="RaggedTensor",
          expected=[[b"pigs", b"on", b"the", b"wing"], [b"animals"]]),
      dict(
          testcase_name="SparseResultType",
          source=[b"pigs on the wing", b"animals"],
          result_type="SparseTensor",
          expected=sparse_tensor.SparseTensorValue(
              [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0]],
              [b"pigs", b"on", b"the", b"wing", b"animals"], [2, 4])),
      dict(
          testcase_name="DefaultResultType",
          source=[b"pigs on the wing", b"animals"],
          expected=sparse_tensor.SparseTensorValue(
              [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0]],
              [b"pigs", b"on", b"the", b"wing", b"animals"], [2, 4])),
      dict(
          testcase_name="BadResultType",
          source=[b"pigs on the wing", b"animals"],
          result_type="BouncyTensor",
          error="result_type must be .*"),
      dict(
          testcase_name="WithSepAndAndSkipEmpty",
          source=[b"+hello+++this+is+a+test"],
          sep="+",
          skip_empty=False,
          result_type="RaggedTensor",
          expected=[[b"", b"hello", b"", b"", b"this", b"is", b"a", b"test"]]),
      dict(
          testcase_name="WithDelimiter",
          source=[b"hello world"],
          delimiter="l",
          result_type="RaggedTensor",
          expected=[[b"he", b"o wor", b"d"]]),
  ])
  def testRaggedStringSplitWrapper(self,
                                   source,
                                   sep=None,
                                   skip_empty=True,
                                   delimiter=None,
                                   result_type="SparseTensor",
                                   expected=None,
                                   error=None):
    if error is not None:
      with self.assertRaisesRegex(ValueError, error):
        ragged_string_ops.string_split(source, sep, skip_empty, delimiter,
                                       result_type)
    if expected is not None:
      result = ragged_string_ops.string_split(source, sep, skip_empty,
                                              delimiter, result_type)
      if isinstance(expected, sparse_tensor.SparseTensorValue):
        self.assertAllEqual(result.indices, expected.indices)
        self.assertAllEqual(result.values, expected.values)
        self.assertAllEqual(result.dense_shape, expected.dense_shape)
      else:
        self.assertAllEqual(result, expected)


class StringSplitV2OpTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      {"testcase_name": "Simple",
       "input": [b"pigs on the wing", b"animals"],
       "expected": [[b"pigs", b"on", b"the", b"wing"], [b"animals"]]},

      {"testcase_name": "MultiCharSeparator",
       "input": [b"1<>2<>3", b"<><>4<>5<><>6<>"],
       "sep": b"<>",
       "expected": [[b"1", b"2", b"3"],
                    [b"", b"", b"4", b"5", b"", b"6", b""]]},

      {"testcase_name": "SimpleSeparator",
       "input": [b"1,2,3", b"4,5,,6,"],
       "sep": b",",
       "expected": [[b"1", b"2", b"3"], [b"4", b"5", b"", b"6", b""]]},

      {"testcase_name": "EmptySeparator",
       "input": [b"1 2 3", b"  4  5    6  "],
       "expected": [[b"1", b"2", b"3"], [b"4", b"5", b"6"]]},

      {"testcase_name": "EmptySeparatorEmptyInputString",
       "input": [b""],
       "expected": [[]]},

      {"testcase_name": "EmptyInputVector",
       "input": [],
       "expected": []},

      {"testcase_name": "SimpleSeparatorMaxSplit",
       "input": [b"1,2,3", b"4,5,,6,"],
       "sep": b",",
       "maxsplit": 1,
       "expected": [[b"1", b"2,3"], [b"4", b"5,,6,"]]},

      {"testcase_name": "EmptySeparatorMaxSplit",
       "input": [b"1 2 3", b"  4  5    6  "],
       "maxsplit": 1,
       "expected": [[b"1", b"2 3"], [b"4", b"5    6  "]]},

      {"testcase_name": "ScalarInput",
       "input": b"1,2,3",
       "sep": b",",
       "expected": [b"1", b"2", b"3"]},

      {"testcase_name": "Dense2DInput",
       "input": [[b"1,2,3", b"4"], [b"5,6", b"7,8,9"]],
       "sep": b",",
       "expected": [[[b"1", b"2", b"3"], [b"4"]],
                    [[b"5", b"6"], [b"7", b"8", b"9"]]]},

      {"testcase_name": "Ragged2DInput",
       "input": [[b"1,2,3", b"4"], [b"5,6"]],
       "input_is_ragged": True,
       "sep": b",",
       "expected": [[[b"1", b"2", b"3"], [b"4"]], [[b"5", b"6"]]]},

      {"testcase_name": "Ragged3DInput",
       "input": [[[b"1,2,3", b"4"], [b"5,6"]], [[b"7,8,9"]]],
       "input_is_ragged": True,
       "sep": b",",
       "expected": [[[[b"1", b"2", b"3"], [b"4"]], [[b"5", b"6"]]],
                    [[[b"7", b"8", b"9"]]]]},

      {"testcase_name": "Ragged4DInput",
       "input": [[[[b"1,2,3", b"4"], [b"5,6"]], [[b"7,8,9"]]], [[[b""]]]],
       "input_is_ragged": True,
       "sep": b",",
       "expected": [[[[[b"1", b"2", b"3"], [b"4"]], [[b"5", b"6"]]],
                     [[[b"7", b"8", b"9"]]]], [[[[b""]]]]]},

      {"testcase_name": "Ragged4DInputEmptySeparator",
       "input": [[[[b"1 2 3", b"4"], [b"5 6"]], [[b"7 8 9"]]], [[[b""]]]],
       "input_is_ragged": True,
       "expected": [[[[[b"1", b"2", b"3"], [b"4"]], [[b"5", b"6"]]],
                     [[[b"7", b"8", b"9"]]]], [[[[]]]]]},

      ])  # pyformat: disable
  def testSplitV2(self,
                  input,
                  expected,
                  input_is_ragged=False,
                  **kwargs):  # pylint: disable=redefined-builtin
    # Check that we are matching the behavior of Python's str.split:
    self.assertEqual(expected, self._py_split(input, **kwargs))

    # Prepare the input tensor.
    if input_is_ragged:
      input = ragged_factory_ops.constant(input, dtype=dtypes.string)
    else:
      input = constant_op.constant(input, dtype=dtypes.string)

    # Check that the public version (which returns a RaggedTensor) works
    # correctly.
    expected_ragged = ragged_factory_ops.constant(
        expected, ragged_rank=input.shape.ndims)
    actual_ragged_v2 = ragged_string_ops.string_split_v2(input, **kwargs)
    actual_ragged_v2_input_kwarg = ragged_string_ops.string_split_v2(
        input=input, **kwargs)
    self.assertAllEqual(expected_ragged, actual_ragged_v2)
    self.assertAllEqual(expected_ragged, actual_ragged_v2_input_kwarg)

    # Check that the internal version (which returns a SparseTensor) works
    # correctly.  Note: the internal version oly supports vector inputs.
    if input.shape.ndims == 1:
      expected_sparse = self.evaluate(expected_ragged.to_sparse())
      actual_sparse_v2 = string_ops.string_split_v2(input, **kwargs)
      self.assertEqual(expected_sparse.indices.tolist(),
                       self.evaluate(actual_sparse_v2.indices).tolist())
      self.assertEqual(expected_sparse.values.tolist(),
                       self.evaluate(actual_sparse_v2.values).tolist())
      self.assertEqual(expected_sparse.dense_shape.tolist(),
                       self.evaluate(actual_sparse_v2.dense_shape).tolist())

  @parameterized.named_parameters([
      {"testcase_name": "Simple",
       "input": [b"pigs on the wing", b"animals"],
       "expected": [[b"pigs", b"on", b"the", b"wing"], [b"animals"]]},

      {"testcase_name": "MultiCharSeparator",
       "input": [b"1<>2<>3", b"<><>4<>5<><>6<>"],
       "sep": b"<>",
       "expected": [[b"1", b"2", b"3"],
                    [b"", b"", b"4", b"5", b"", b"6", b""]]},

      {"testcase_name": "SimpleSeparator",
       "input": [b"1,2,3", b"4,5,,6,"],
       "sep": b",",
       "expected": [[b"1", b"2", b"3"], [b"4", b"5", b"", b"6", b""]]},

      {"testcase_name": "EmptySeparator",
       "input": [b"1 2 3", b"  4  5    6  "],
       "expected": [[b"1", b"2", b"3"], [b"4", b"5", b"6"]]},

      {"testcase_name": "EmptySeparatorEmptyInputString",
       "input": [b""],
       "expected": [[]]},

      {"testcase_name": "SimpleSeparatorMaxSplit",
       "input": [b"1,2,3", b"4,5,,6,"],
       "sep": b",",
       "maxsplit": 1,
       "expected": [[b"1", b"2,3"], [b"4", b"5,,6,"]]},

      {"testcase_name": "EmptySeparatorMaxSplit",
       "input": [b"1 2 3", b"  4  5    6  "],
       "maxsplit": 1,
       "expected": [[b"1", b"2 3"], [b"4", b"5    6  "]]},

      {"testcase_name": "ScalarInput",
       "input": b"1,2,3",
       "sep": b",",
       "expected": [[b"1", b"2", b"3"]]},

      {"testcase_name": "Dense2DInput",
       "input": [[b"1,2,3", b"4"], [b"5,6", b"7,8,9"]],
       "sep": b",",
       "expected": [[[b"1", b"2", b"3"], [b"4"]],
                    [[b"5", b"6"], [b"7", b"8", b"9"]]]},

      {"testcase_name": "Ragged2DInput",
       "input": [[b"1,2,3", b"4"], [b"5,6"]],
       "input_is_ragged": True,
       "sep": b",",
       "expected": [[[b"1", b"2", b"3"], [b"4"]], [[b"5", b"6"]]]},

      {"testcase_name": "Ragged3DInput",
       "input": [[[b"1,2,3", b"4"], [b"5,6"]], [[b"7,8,9"]]],
       "input_is_ragged": True,
       "sep": b",",
       "expected": [[[[b"1", b"2", b"3"], [b"4"]], [[b"5", b"6"]]],
                    [[[b"7", b"8", b"9"]]]]},

      {"testcase_name": "Ragged4DInput",
       "input": [[[[b"1,2,3", b"4"], [b"5,6"]], [[b"7,8,9"]]], [[[b""]]]],
       "input_is_ragged": True,
       "sep": b",",
       "expected": [[[[[b"1", b"2", b"3"], [b"4"]], [[b"5", b"6"]]],
                     [[[b"7", b"8", b"9"]]]], [[[[b""]]]]]},

      {"testcase_name": "Ragged4DInputEmptySeparator",
       "input": [[[[b"1 2 3", b"4"], [b"5 6"]], [[b"7 8 9"]]], [[[b""]]]],
       "input_is_ragged": True,
       "expected": [[[[[b"1", b"2", b"3"], [b"4"]], [[b"5", b"6"]]],
                     [[[b"7", b"8", b"9"]]]], [[[[]]]]]},

      ])  # pyformat: disable
  def testSplitV1(self, input, expected, input_is_ragged=False, **kwargs):  # pylint: disable=redefined-builtin
    # Prepare the input tensor.
    if input_is_ragged:
      input = ragged_factory_ops.constant(input, dtype=dtypes.string)
    else:
      input = constant_op.constant(input, dtype=dtypes.string)

    expected_ragged = ragged_factory_ops.constant(expected)
    actual_ragged_v1 = ragged_string_ops.strings_split_v1(
        input, result_type="RaggedTensor", **kwargs)
    actual_ragged_v1_input_kwarg = ragged_string_ops.strings_split_v1(
        input=input, result_type="RaggedTensor", **kwargs)
    actual_ragged_v1_source_kwarg = ragged_string_ops.strings_split_v1(
        source=input, result_type="RaggedTensor", **kwargs)
    self.assertAllEqual(expected_ragged, actual_ragged_v1)
    self.assertAllEqual(expected_ragged, actual_ragged_v1_input_kwarg)
    self.assertAllEqual(expected_ragged, actual_ragged_v1_source_kwarg)
    expected_sparse = self.evaluate(expected_ragged.to_sparse())
    actual_sparse_v1 = ragged_string_ops.strings_split_v1(
        input, result_type="SparseTensor", **kwargs)
    self.assertEqual(expected_sparse.indices.tolist(),
                     self.evaluate(actual_sparse_v1.indices).tolist())
    self.assertEqual(expected_sparse.values.tolist(),
                     self.evaluate(actual_sparse_v1.values).tolist())
    self.assertEqual(expected_sparse.dense_shape.tolist(),
                     self.evaluate(actual_sparse_v1.dense_shape).tolist())

  def testSplitV1BadResultType(self):
    with self.assertRaisesRegex(ValueError, "result_type must be .*"):
      ragged_string_ops.strings_split_v1("foo", result_type="BouncyTensor")

  def _py_split(self, strings, **kwargs):
    if isinstance(strings, compat.bytes_or_text_types):
      # Note: str.split doesn't accept keyword args.
      if "maxsplit" in kwargs:
        return strings.split(kwargs.get("sep", None), kwargs["maxsplit"])
      else:
        return strings.split(kwargs.get("sep", None))
    else:
      return [self._py_split(s, **kwargs) for s in strings]


if __name__ == "__main__":
  test.main()
