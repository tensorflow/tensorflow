# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Raw ops tests."""

from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class RawOpsTest(test.TestCase, parameterized.TestCase):

  def testSimple(self):
    x = constant_op.constant(1)
    self.assertEqual([2], self.evaluate(gen_math_ops.Add(x=x, y=x)))

  def testRequiresKwargs(self):
    with self.assertRaisesRegex(TypeError, "only takes keyword args"):
      gen_math_ops.Add(1., 1.)

  def testRequiresKwargs_providesSuggestion(self):
    msg = "possible keys: \\['x', 'y', 'name'\\]"
    with self.assertRaisesRegex(TypeError, msg):
      gen_math_ops.Add(1., y=2.)

  def testName(self):
    x = constant_op.constant(1)
    op = gen_math_ops.Add(x=x, y=x, name="double")
    if not context.executing_eagerly():
      # `Tensor.name` is not available in eager.
      self.assertEqual(op.name, "double:0")

  def testDoc(self):
    self.assertEqual(gen_math_ops.add.__doc__, gen_math_ops.Add.__doc__)

  def testDefaults(self):
    x = constant_op.constant([[True]])
    self.assertAllClose(
        gen_math_ops.Any(input=x, axis=0),
        gen_math_ops.Any(input=x, axis=0, keep_dims=False))

  @parameterized.parameters([[0, 8]], [[-1, 6]])
  def testStringNGramsBadDataSplits(self, splits):
    data = ["aa", "bb", "cc", "dd", "ee", "ff"]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r"Invalid split value|First split value must be 0"):
      self.evaluate(
          gen_string_ops.string_n_grams(
              data=data,
              data_splits=splits,
              separator="",
              ngram_widths=[2],
              left_pad="",
              right_pad="",
              pad_width=0,
              preserve_short_sequences=False))

  def testStringSplit(self):
    data = ["123456"]
    data_splits = [0, 1]
    separator = "a" * 15
    ngram_widths = []
    pad_width = -5
    left_pad = right_pad = ""
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Pad width should be >= 0"):
      self.evaluate(gen_string_ops.string_n_grams(
          data=data,
          data_splits=data_splits,
          separator=separator,
          ngram_widths=ngram_widths,
          left_pad=left_pad,
          right_pad=right_pad,
          pad_width=pad_width,
          preserve_short_sequences=True))
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Pad width could lead to integer overflow"):
      self.evaluate(
          gen_string_ops.string_n_grams(
              data=["000.0", "000.0"],
              data_splits=[0, 2],
              separator="",
              ngram_widths=[2**30, 2**30],
              left_pad=" ",
              right_pad=" ",
              pad_width=-2**30,
              preserve_short_sequences=False))

  def testGetSessionHandle(self):
    if context.executing_eagerly():
      with self.assertRaisesRegex(
          errors.FailedPreconditionError,
          "GetSessionHandle called on null session state"):
        gen_data_flow_ops.GetSessionHandle(value=[1])


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
