# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.print with ragged tensors.

Note: ragged support for tf.print is implemented by RaggedPrintV2Dispatcher in
ragged_dispatch.py.
"""

import os.path
import tempfile
from absl.testing import parameterized
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedPrintV2Test(test_util.TensorFlowTestCase, parameterized.TestCase):

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters([
      dict(
          testcase_name='2d_int_values',
          inputs=lambda: [ragged_factory_ops.constant([[1, 2], [3]])],
          expected='[[1, 2], [3]]\n'),
      dict(
          testcase_name='3d_int_values',
          inputs=lambda: [ragged_factory_ops.constant([[[1, 2], [3]], [[4]]])],
          expected='[[[1, 2], [3]], [[4]]]\n'),
      dict(
          testcase_name='2d_str_values',
          inputs=lambda: [ragged_factory_ops.constant([['a', 'b'], ['c']])],
          expected="[['a', 'b'], ['c']]\n"),
      dict(
          testcase_name='2d_str_values_with_escaping',
          inputs=lambda: [ragged_factory_ops.constant([["a'b"], ['c"d']])],
          expected="[['a\\'b'], ['c\"d']]\n"),
      dict(
          testcase_name='two_ragged_values',
          inputs=lambda: [
              ragged_factory_ops.constant([[1, 2], [3]]),
              ragged_factory_ops.constant([[5], [], [6, 7, 8]])
          ],
          expected='[[1, 2], [3]] [[5], [], [6, 7, 8]]\n'),
      dict(
          testcase_name='ragged_value_and_non_tensor_values',
          inputs=lambda:
          ['a', 5, True,
           ragged_factory_ops.constant([[1, 2], [3]]), 'c'],
          expected='a 5 True [[1, 2], [3]] c\n'),
      dict(
          testcase_name='ragged_value_and_dense_value',
          inputs=lambda: [
              ragged_factory_ops.constant([[1, 2], [3]]),
              constant_op.constant([[1, 2], [3, 4]])
          ],
          expected='[[1, 2], [3]] [[1 2]\n [3 4]]\n'),
      dict(
          testcase_name='ragged_value_and_sparse_value',
          inputs=lambda: [
              ragged_factory_ops.constant([[1, 2], [3]]),
              sparse_ops.from_dense([[1]])
          ],
          expected=(
              '[[1, 2], [3]] '
              "'SparseTensor(indices=[[0 0]], values=[1], shape=[1 1])'\n")),
      dict(
          testcase_name='summarize_default',
          inputs=lambda: [
              ragged_factory_ops.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9], [10], [
              ], [], [], [], [11, 12]])
          ],
          expected=('[[1, 2, 3, ..., 7, 8, 9], [10], [], '
                    '..., '
                    '[], [], [11, 12]]\n')),
      dict(
          testcase_name='summarize_2',
          inputs=lambda: [
              ragged_factory_ops.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9], [10], [
              ], [], [], [], [11, 12]])
          ],
          summarize=2,
          expected='[[1, 2, ..., 8, 9], [10], ..., [], [11, 12]]\n'),
      dict(
          testcase_name='summarize_neg1',
          inputs=lambda: [
              ragged_factory_ops.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9], [10], [
              ], [], [], [], [11, 12]])
          ],
          summarize=-1,
          expected=('[[1, 2, 3, 4, 5, 6, 7, 8, 9], [10], '
                    '[], [], [], [], [11, 12]]\n')),
  ])
  def testRaggedPrint(self, inputs, expected, summarize=None):
    if callable(inputs):
      inputs = inputs()
    with tempfile.TemporaryDirectory() as tmpdirname:
      path = os.path.join(tmpdirname, 'print_output')
      kwargs = {'output_stream': 'file://{}'.format(path)}
      if summarize is not None:
        kwargs.update(summarize=summarize)
      self.evaluate(logging_ops.print_v2(*inputs, **kwargs))
      actual = open(path, 'r').read()
      self.assertEqual(repr(actual), repr(expected))


@test_util.run_all_in_graph_and_eager_modes
class RaggedToStringTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      ('2d_int', [[1, 2], [], [3, 4, 5]], '[[1, 2], [], [3, 4, 5]]'),
      ('2d_str', [['a'], ['b'], ['c', 'd']], "[['a'], ['b'], ['c', 'd']]"),
      ('3d_int', [[[1, 2], []], [[3, 4, 5]]], '[[[1, 2], []], [[3, 4, 5]]]'),
      ('escape', [["a'b"], [r'c\d']], r"[['a\'b'], ['c\\d']]"),
      dict(testcase_name='2d_empty', rt=[], ragged_rank=1, expected='[]'),
      dict(testcase_name='3d_empty', rt=[], ragged_rank=2, expected='[]'),
      dict(
          testcase_name='3d_rrank1',
          rt=[[[1, 2], [3, 4]], [], [[5, 6]]],
          ragged_rank=1,
          expected='[[[1, 2], [3, 4]], [], [[5, 6]]]'),
      dict(
          testcase_name='2d_empty_row', rt=[[]], ragged_rank=1,
          expected='[[]]'),
      dict(
          testcase_name='3d_empty_row', rt=[[]], ragged_rank=2,
          expected='[[]]'),
      dict(
          testcase_name='summarize_1',
          rt=[[1, 2, 3, 4, 5], [], [6], [7], [8, 9]],
          summarize=1,
          expected='[[1, ..., 5], ..., [8, 9]]'),
      dict(
          testcase_name='summarize_2',
          rt=[[1, 2, 3, 4, 5], [], [6], [7], [8, 9]],
          summarize=2,
          expected='[[1, 2, ..., 4, 5], [], ..., [7], [8, 9]]'),
  ])
  def testRaggedToString(self, rt, expected, summarize=None, ragged_rank=None):
    rt = ragged_factory_ops.constant(rt, ragged_rank=ragged_rank)
    actual = ragged_string_ops.ragged_tensor_to_string(rt, summarize=summarize)
    self.assertAllEqual(actual, expected)

  @parameterized.named_parameters([
      ('maxelts_BadType', [[1]], "Expected summarize .*, got 'foo'", 'foo'),
      ('maxelts_0', [[1]], 'Expected summarize to be .*, got 0', 0),
      ('maxelts_Neg2', [[1]], 'Expected summarize to be .*, got -2', -2),
  ])
  def testRaggedToStringErrors(self,
                               rt,
                               error,
                               summarize=None,
                               exception=ValueError):
    rt = ragged_factory_ops.constant(rt)
    with self.assertRaisesRegex(exception, error):
      self.evaluate(
          ragged_string_ops.ragged_tensor_to_string(rt, summarize=summarize))

  def testRaggedToStringUnknownRank(self):

    @def_function.function(
        input_signature=[ragged_tensor.RaggedTensorSpec(ragged_rank=1)])
    def f(rt):
      return ragged_string_ops.ragged_tensor_to_string(rt)

    with self.assertRaisesRegex(
        ValueError, 'RaggedTensor to_string requires '
        'that rt.shape.rank is not None'):
      f(ragged_factory_ops.constant([[1, 2], [3]]))


if __name__ == '__main__':
  googletest.main()
