# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

# encoding=utf-8
"""Tests for HubModuleTokenizer."""

from absl.testing import parameterized

from tensorflow.python.framework import test_util
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow_text.python.ops import hub_module_tokenizer  # pylint: disable=line-too-long


def _Utf8(char):
  return char.encode("utf-8")


@test_util.run_all_in_graph_and_eager_modes
class HubModuleTokenizerTest(parameterized.TestCase, test.TestCase):

  @parameterized.parameters([
      # Test scalar input.
      dict(
          text_input=_Utf8(u"新华社北京"),
          expected_tokens=[_Utf8(u"新华社"), _Utf8(u"北京")],
          expected_starts=[0, 9],
          expected_ends=[9, 15]
      ),
      # Test rank 1 input.
      dict(
          text_input=[_Utf8(u"新华社北京"), _Utf8(u"中文测试")],
          expected_tokens=[[_Utf8(u"新华社"), _Utf8(u"北京")],
                           [_Utf8(u"中文"), _Utf8(u"测试")]],
          expected_starts=[[0, 9], [0, 6]],
          expected_ends=[[9, 15], [6, 12]]
      ),
      # Test rank 2 ragged input.
      dict(
          text_input=ragged_factory_ops.constant_value(
              [[_Utf8(u"新华社北京"), _Utf8(u"中文测试")],
               [_Utf8(u"新华社上海")]]),
          expected_tokens=[[[_Utf8(u"新华社"), _Utf8(u"北京")],
                            [_Utf8(u"中文"), _Utf8(u"测试")]],
                           [[_Utf8(u"新华社"), _Utf8(u"上海")]]],
          expected_starts=[[[0, 9], [0, 6]], [[0, 9]]],
          expected_ends=[[[9, 15], [6, 12]], [[9, 15]]]
      ),
      # Test rank 2 dense input.
      dict(
          text_input=ragged_factory_ops.constant_value(
              [[_Utf8(u"新华社北京"), _Utf8(u"中文测试")],
               [_Utf8(u"新华社上海"), _Utf8(u"英国交通")]]),
          expected_tokens=[[[_Utf8(u"新华社"), _Utf8(u"北京")],
                            [_Utf8(u"中文"), _Utf8(u"测试")]],
                           [[_Utf8(u"新华社"), _Utf8(u"上海")],
                            [_Utf8(u"英国"), _Utf8(u"交通")]]],
          expected_starts=[[[0, 9], [0, 6]], [[0, 9], [0, 6]]],
          expected_ends=[[[9, 15], [6, 12]], [[9, 15], [6, 12]]]
      ),
      # Test ragged input with rank higher than 2.
      dict(
          text_input=ragged_factory_ops.constant_value(
              [
                  [[_Utf8(u"新华社北京")], [_Utf8(u"中文测试")]],
                  [[_Utf8(u"新华社上海")]]
              ]),
          expected_tokens=[
              [[[_Utf8(u"新华社"), _Utf8(u"北京")]],
               [[_Utf8(u"中文"), _Utf8(u"测试")]]],
              [[[_Utf8(u"新华社"), _Utf8(u"上海")]]]],
          expected_starts=[
              [[[0, 9]], [[0, 6]]],
              [[[0, 9]]]],
          expected_ends=[
              [[[9, 15]], [[6, 12]]],
              [[[9, 15]]]]
      )
  ])
  def testTokenize(self,
                   text_input,
                   expected_tokens,
                   expected_starts,
                   expected_ends):
    hub_module_handle = ("tensorflow_text/python/ops/test_data/"
                         "segmenter_hub_module")
    segmenter = hub_module_tokenizer.HubModuleTokenizer(hub_module_handle)
    tokens, starts, ends = segmenter.tokenize_with_offsets(text_input)
    tokens_no_offset = segmenter.tokenize(text_input)
    self.evaluate(lookup_ops.tables_initializer())
    self.evaluate(variables_lib.global_variables_initializer())
    self.assertAllEqual(expected_tokens, tokens)
    self.assertAllEqual(expected_starts, starts)
    self.assertAllEqual(expected_ends, ends)
    self.assertAllEqual(expected_tokens, tokens_no_offset)


if __name__ == "__main__":
  test.main()
