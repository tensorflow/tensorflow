# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the b"License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an b"AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for the Tensorflow strings.ngrams op."""

from absl.testing import parameterized

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.platform import test


class StringNgramsTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def test_unpadded_ngrams(self):
    data = [[b"aa", b"bb", b"cc", b"dd"], [b"ee", b"ff"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=3, separator=b"|")
    result = self.evaluate(ngram_op)
    expected_ngrams = [[b"aa|bb|cc", b"bb|cc|dd"], []]
    self.assertAllEqual(expected_ngrams, result)

  def test_tuple_multi_ngrams(self):
    data = [[b"aa", b"bb", b"cc", b"dd"], [b"ee", b"ff"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=(2, 3), separator=b"|")
    result = self.evaluate(ngram_op)
    expected_ngrams = [[b"aa|bb", b"bb|cc", b"cc|dd", b"aa|bb|cc", b"bb|cc|dd"],
                       [b"ee|ff"]]
    self.assertAllEqual(expected_ngrams, result)

  def test_tuple_multi_ngrams_inverted_order(self):
    data = [[b"aa", b"bb", b"cc", b"dd"], [b"ee", b"ff"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=(3, 2), separator=b"|")
    result = self.evaluate(ngram_op)
    expected_ngrams = [[b"aa|bb|cc", b"bb|cc|dd", b"aa|bb", b"bb|cc", b"cc|dd"],
                       [b"ee|ff"]]
    self.assertAllEqual(expected_ngrams, result)

  def test_list_multi_ngrams(self):
    data = [[b"aa", b"bb", b"cc", b"dd"], [b"ee", b"ff"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=[2, 3], separator=b"|")
    result = self.evaluate(ngram_op)
    expected_ngrams = [[b"aa|bb", b"bb|cc", b"cc|dd", b"aa|bb|cc", b"bb|cc|dd"],
                       [b"ee|ff"]]
    self.assertAllEqual(expected_ngrams, result)

  def test_multi_ngram_ordering(self):
    data = [[b"aa", b"bb", b"cc", b"dd"], [b"ee", b"ff"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=[3, 2], separator=b"|")
    result = self.evaluate(ngram_op)
    expected_ngrams = [[b"aa|bb|cc", b"bb|cc|dd", b"aa|bb", b"bb|cc", b"cc|dd"],
                       [b"ee|ff"]]
    self.assertAllEqual(expected_ngrams, result)

  def test_fully_padded_ngrams(self):
    data = [[b"a"], [b"b", b"c", b"d"], [b"e", b"f"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=3, separator=b"|", pad_values=(b"LP", b"RP"))
    result = self.evaluate(ngram_op)
    expected_ngrams = [
        [b"LP|LP|a", b"LP|a|RP", b"a|RP|RP"],  # 0
        [b"LP|LP|b", b"LP|b|c", b"b|c|d", b"c|d|RP", b"d|RP|RP"],  # 1
        [b"LP|LP|e", b"LP|e|f", b"e|f|RP", b"f|RP|RP"]  # 2
    ]
    self.assertAllEqual(expected_ngrams, result)

  def test_ngram_padding_size_cap(self):
    # Validate that the padding size is never greater than ngram_size - 1.
    data = [[b"a"], [b"b", b"c", b"d"], [b"e", b"f"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor,
        ngram_width=3,
        separator=b"|",
        pad_values=(b"LP", b"RP"),
        padding_width=10)
    result = self.evaluate(ngram_op)
    expected_ngrams = [
        [b"LP|LP|a", b"LP|a|RP", b"a|RP|RP"],  # 0
        [b"LP|LP|b", b"LP|b|c", b"b|c|d", b"c|d|RP", b"d|RP|RP"],  # 1
        [b"LP|LP|e", b"LP|e|f", b"e|f|RP", b"f|RP|RP"]  # 2
    ]
    self.assertAllEqual(expected_ngrams, result)

  def test_singly_padded_ngrams(self):
    data = [[b"a"], [b"b", b"c", b"d"], [b"e", b"f"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor,
        ngram_width=5,
        separator=b"|",
        pad_values=(b"LP", b"RP"),
        padding_width=1)
    result = self.evaluate(ngram_op)
    expected_ngrams = [[], [b"LP|b|c|d|RP"], []]
    self.assertAllEqual(expected_ngrams, result)

  def test_singly_padded_ngrams_with_preserve_short(self):
    data = [[b"a"], [b"b", b"c", b"d"], [b"e", b"f"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor,
        ngram_width=5,
        separator=b"|",
        pad_values=(b"LP", b"RP"),
        padding_width=1,
        preserve_short_sequences=True)
    result = self.evaluate(ngram_op)
    expected_ngrams = [[b"LP|a|RP"], [b"LP|b|c|d|RP"], [b"LP|e|f|RP"]]
    self.assertAllEqual(expected_ngrams, result)

  def test_singly_padded_multiple_ngrams(self):
    data = [[b"a"], [b"b", b"c", b"d"], [b"e", b"f"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor,
        ngram_width=(1, 5),
        separator=b"|",
        pad_values=(b"LP", b"RP"),
        padding_width=1)
    result = self.evaluate(ngram_op)
    expected_ngrams = [[b"a"], [b"b", b"c", b"d", b"LP|b|c|d|RP"], [b"e", b"f"]]
    self.assertAllEqual(expected_ngrams, result)

  def test_single_padding_string(self):
    data = [[b"a"], [b"b", b"c", b"d"], [b"e", b"f"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor,
        ngram_width=5,
        separator=b"|",
        pad_values=b"[PAD]",
        padding_width=1)
    result = self.evaluate(ngram_op)
    expected_ngrams = [[], [b"[PAD]|b|c|d|[PAD]"], []]
    self.assertAllEqual(expected_ngrams, result)

  def test_explicit_multiply_padded_ngrams(self):
    data = [[b"a"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor,
        ngram_width=5,
        separator=b"|",
        pad_values=(b"LP", b"RP"),
        padding_width=2)
    result = self.evaluate(ngram_op)
    expected_ngrams = [[b"LP|LP|a|RP|RP"]]
    self.assertAllEqual(expected_ngrams, result)

  def test_ragged_inputs_with_multiple_ragged_dimensions(self):
    data = [[[[b"aa", b"bb", b"cc", b"dd"]], [[b"ee", b"ff"]]]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=3, separator=b"|")
    result = self.evaluate(ngram_op)
    expected_ngrams = [[[[b"aa|bb|cc", b"bb|cc|dd"]], [[]]]]
    self.assertAllEqual(expected_ngrams, result)

  def test_ragged_inputs_with_multiple_ragged_dimensions_and_preserve(self):
    data = [[[[b"aa", b"bb", b"cc", b"dd"]], [[b"ee", b"ff"]]]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor,
        ngram_width=3,
        separator=b"|",
        preserve_short_sequences=True)
    result = self.evaluate(ngram_op)
    expected_ngrams = [[[[b"aa|bb|cc", b"bb|cc|dd"]], [[b"ee|ff"]]]]
    self.assertAllEqual(expected_ngrams, result)

  def test_ragged_inputs_with_multiple_ragged_dimensions_bigrams(self):
    data = [[[[b"aa", b"bb", b"cc", b"dd"]], [[b"ee", b"ff"]]]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=2, separator=b"|")
    result = self.evaluate(ngram_op)
    expected_ngrams = [[[[b"aa|bb", b"bb|cc", b"cc|dd"]], [[b"ee|ff"]]]]
    self.assertAllEqual(expected_ngrams, result)

  def test_ragged_inputs_with_multiple_ragged_dimensions_and_multiple_ngrams(
      self):
    data = [[[[b"aa", b"bb", b"cc", b"dd"]], [[b"ee", b"ff"]]]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=(3, 4), separator=b"|")
    result = self.evaluate(ngram_op)
    expected_ngrams = [[[[b"aa|bb|cc", b"bb|cc|dd", b"aa|bb|cc|dd"]], [[]]]]
    self.assertAllEqual(expected_ngrams, result)

  def test_dense_input_rank_3(self):
    data = [[[b"a", b"z"], [b"b", b""]], [[b"b", b""], [b"e", b"f"]]]
    data_tensor = constant_op.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=3, separator=b"|", pad_values=(b"LP", b"RP"))
    result = self.evaluate(ngram_op)
    expected_ngrams = [[[b"LP|LP|a", b"LP|a|z", b"a|z|RP", b"z|RP|RP"],
                        [b"LP|LP|b", b"LP|b|", b"b||RP", b"|RP|RP"]],
                       [[b"LP|LP|b", b"LP|b|", b"b||RP", b"|RP|RP"],
                        [b"LP|LP|e", b"LP|e|f", b"e|f|RP", b"f|RP|RP"]]]
    self.assertIsInstance(ngram_op, ops.Tensor)
    self.assertAllEqual(expected_ngrams, result)

  def test_dense_input(self):
    data = [[b"a", b"z"], [b"b", b""], [b"e", b"f"]]
    data_tensor = constant_op.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=3, separator=b"|", pad_values=(b"LP", b"RP"))
    result = self.evaluate(ngram_op)
    expected_ngrams = [
        [b"LP|LP|a", b"LP|a|z", b"a|z|RP", b"z|RP|RP"],
        [b"LP|LP|b", b"LP|b|", b"b||RP", b"|RP|RP"],
        [b"LP|LP|e", b"LP|e|f", b"e|f|RP", b"f|RP|RP"],
    ]
    self.assertIsInstance(ngram_op, ops.Tensor)
    self.assertAllEqual(expected_ngrams, result)

  def test_input_list_input(self):
    data = [[b"a", b"z"], [b"b", b""], [b"e", b"f"]]
    ngram_op = ragged_string_ops.ngrams(
        data, ngram_width=3, separator=b"|", pad_values=(b"LP", b"RP"))
    result = self.evaluate(ngram_op)
    expected_ngrams = [
        [b"LP|LP|a", b"LP|a|z", b"a|z|RP", b"z|RP|RP"],
        [b"LP|LP|b", b"LP|b|", b"b||RP", b"|RP|RP"],
        [b"LP|LP|e", b"LP|e|f", b"e|f|RP", b"f|RP|RP"],
    ]
    self.assertAllEqual(expected_ngrams, result)

  def test_vector_input(self):
    data = [b"a", b"z"]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=3, separator=b"|", pad_values=(b"LP", b"RP"))
    result = self.evaluate(ngram_op)
    expected_ngrams = [b"LP|LP|a", b"LP|a|z", b"a|z|RP", b"z|RP|RP"]
    self.assertAllEqual(expected_ngrams, result)

  def test_dense_input_with_multiple_ngrams(self):
    data = [[b"a", b"b", b"c", b"d"], [b"e", b"f", b"g", b"h"]]
    data_tensor = ragged_factory_ops.constant(data)
    ngram_op = ragged_string_ops.ngrams(
        data_tensor, ngram_width=(1, 2, 3), separator=b"|")
    result = self.evaluate(ngram_op)
    expected_ngrams = [[
        b"a", b"b", b"c", b"d", b"a|b", b"b|c", b"c|d", b"a|b|c", b"b|c|d"
    ], [b"e", b"f", b"g", b"h", b"e|f", b"f|g", b"g|h", b"e|f|g", b"f|g|h"]]
    self.assertAllEqual(expected_ngrams, result)

  def test_input_with_no_values(self):
    data = ragged_factory_ops.constant([[], [], []], dtype=dtypes.string)
    ngram_op = ragged_string_ops.ngrams(data, (1, 2))
    result = self.evaluate(ngram_op)
    self.assertAllEqual([0, 0, 0, 0], result.row_splits)
    self.assertAllEqual(constant_op.constant([], dtype=dtypes.string),
                        result.values)

  @parameterized.parameters([
      dict(
          data=[b"a", b"z"],
          ngram_width=2,
          pad_values=5,
          exception=TypeError,
          error="pad_values must be a string, tuple of strings, or None."),
      dict(
          data=[b"a", b"z"],
          ngram_width=2,
          pad_values=[5, 3],
          exception=TypeError,
          error="pad_values must be a string, tuple of strings, or None."),
      dict(
          data=[b"a", b"z"],
          ngram_width=2,
          padding_width=0,
          pad_values="X",
          error="padding_width must be greater than 0."),
      dict(
          data=[b"a", b"z"],
          ngram_width=2,
          padding_width=1,
          error="pad_values must be provided if padding_width is set."),
      dict(
          data=b"hello",
          ngram_width=2,
          padding_width=1,
          pad_values="X",
          error="Data must have rank>0"),
      dict(
          data=[b"hello", b"world"],
          ngram_width=[1, 2, -1],
          padding_width=1,
          pad_values="X",
          error="All ngram_widths must be greater than 0. Got .*"),
  ])
  def test_error(self,
                 data,
                 ngram_width,
                 separator=" ",
                 pad_values=None,
                 padding_width=None,
                 preserve_short_sequences=False,
                 error=None,
                 exception=ValueError):
    with self.assertRaisesRegex(exception, error):
      ragged_string_ops.ngrams(data, ngram_width, separator, pad_values,
                               padding_width, preserve_short_sequences)

  def test_unknown_rank_error(self):
    # Use a tf.function that erases shape information.
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.string)])
    def f(v):
      return ragged_string_ops.ngrams(v, 2)

    with self.assertRaisesRegex(ValueError, "Rank of data must be known."):
      f([b"foo", b"bar"])


if __name__ == "__main__":
  test.main()
