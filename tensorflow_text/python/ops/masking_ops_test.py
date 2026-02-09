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

"""Tests for masking ops."""
from absl.testing import parameterized

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.platform import test
from tensorflow_text.python.ops import item_selector_ops
from tensorflow_text.python.ops import masking_ops

_VOCAB = [
    b"[MASK]",
    b"[RANDOM]",
    b"[CLS]",
    b"[SEP]",
    b"##ack",
    b"##ama",
    b"##gers",
    b"##onge",
    b"##pants",
    b"##uare",
    b"##vel",
    b"##ven",
    b"A",
    b"Bar",
    b"Hates",
    b"Mar",
    b"Ob",
    b"Patrick",
    b"President",
    b"Sp",
    b"Sq",
    b"bob",
    b"box",
    b"has",
    b"highest",
    b"is",
    b"office",
    b"the",
]

_MASK_TOKEN = _VOCAB.index(b"[MASK]")
_RANDOM_TOKEN = _VOCAB.index(b"[RANDOM]")
_VOCAB_SIZE = len(_VOCAB)


def _create_table(vocab, num_oov=1):
  init = lookup_ops.KeyValueTensorInitializer(
      vocab,
      math_ops.range(
          array_ops.size(vocab, out_type=dtypes.int64), dtype=dtypes.int64),
      key_dtype=dtypes.string,
      value_dtype=dtypes.int64)
  return lookup_ops.StaticVocabularyTableV1(
      init, num_oov, lookup_key_dtype=dtypes.string)


class AlwaysRandomValuesChooser(masking_ops.MaskValuesChooser):

  def __init__(self,
               vocab_size,
               mask_token,
               random_token,
               mask_token_rate=0.8,
               random_token_rate=0.1):
    super(AlwaysRandomValuesChooser, self).__init__(1, 0, 0, 1)
    self._random_token = random_token

  def get_mask_values(self, masked_lm_ids, **kwargs):
    # If we're expecting all random tokens, set to all [RANDOM]
    if self.random_token_rate == 1:
      flat_mask_values = array_ops.tile(
          array_ops.expand_dims(self._random_token, -1),
          array_ops.shape(masked_lm_ids.flat_values))
      flat_mask_values = math_ops.cast(flat_mask_values, dtypes.int64)
    else:
      # Give them all [MASK] values.
      flat_mask_values = array_ops.tile(
          array_ops.expand_dims(self.mask_token, -1),
          array_ops.shape(masked_lm_ids.flat_values))
      flat_mask_values = math_ops.cast(flat_mask_values, dtypes.int64)
    return masked_lm_ids.with_flat_values(flat_mask_values)


@test_util.run_all_in_graph_and_eager_modes
class MaskingOpsTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      dict(
          description="Masking wordpieces",
          masking_inputs=[[
              b"Sp", b"##onge", b"bob", b"Sq", b"##uare", b"##pants"
          ], [b"Bar", b"##ack", b"Ob", b"##ama"],
                          [b"Mar", b"##vel", b"A", b"##ven", b"##gers"]],
          expected_masked_positions=[[0, 1], [0, 1], [0, 1]],
          expected_masked_ids=[[b"Sp", b"##onge"], [b"Bar", b"##ack"],
                               [b"Mar", b"##vel"]],
          expected_input_ids=[[
              b"[MASK]", b"[MASK]", b"bob", b"Sq", b"##uare", b"##pants"
          ], [b"[MASK]", b"[MASK]", b"Ob", b"##ama"],
                              [b"[MASK]", b"[MASK]", b"A", b"##ven",
                               b"##gers"]],
      ),
      dict(
          description="Masking wordpieces; allow all wordpieces",
          masking_inputs=[[
              b"Sp", b"##onge", b"bob", b"Sq", b"##uare", b"##pants"
          ], [b"Bar", b"##ack", b"Ob",
              b"##ama"], [b"Mar", b"##vel", b"A", b"##ven", b"##gers"]],
          expected_masked_positions=[[0, 1], [0, 1], [0, 1]],
          expected_masked_ids=[[b"Sp", b"##onge"], [b"Bar", b"##ack"],
                               [b"Mar", b"##vel"]],
          expected_input_ids=[[
              b"[MASK]", b"[MASK]", b"bob", b"Sq", b"##uare", b"##pants"
          ], [b"[MASK]", b"[MASK]", b"Ob", b"##ama"],
                              [b"[MASK]", b"[MASK]", b"A",
                               b"##ven", b"##gers"]],
          no_mask_ids=[],
      ),
      dict(
          description="Masking wordpieces w/ no_mask_ids",
          masking_inputs=[[
              b"Sp", b"##onge", b"bob", b"Sq", b"##uare", b"##pants"
          ], [b"Bar", b"##ack", b"Ob",
              b"##ama"], [b"Mar", b"##vel", b"A", b"##ven", b"##gers"]],
          no_mask_ids=[b"[CLS]", b"[SEP]", b"##onge", b"Mar"],
          expected_masked_positions=[[0, 2], [0, 1], [1, 2]],
          expected_masked_ids=[[b"Sp", b"bob"], [b"Bar", b"##ack"],
                               [b"##vel", b"A"]],
          expected_input_ids=[[
              b"[MASK]", b"##onge", b"[MASK]", b"Sq", b"##uare", b"##pants"
          ], [b"[MASK]", b"[MASK]", b"Ob",
              b"##ama"], [b"Mar", b"[MASK]", b"[MASK]", b"##ven", b"##gers"]],
      ),
      dict(
          description=b"Masking whole words, first masked tokens are selected" +
          b" as [MASK]",
          masking_inputs=[[[b"Sp", "##onge"], [b"bob"],
                           [b"Sq", b"##uare", b"##pants"]],
                          [[b"Bar", "##ack"], [b"Ob", b"##ama"]],
                          [[b"Mar", "##vel"], [b"A", b"##ven", b"##gers"]]],
          expected_masked_positions=[[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]],
          expected_masked_ids=[[b"Sp", b"##onge", b"bob"],
                               [b"Bar", b"##ack", b"Ob", b"##ama"],
                               [b"Mar", b"##vel", b"A", b"##ven", b"##gers"]],
          expected_input_ids=[
              [b"[MASK]", b"[MASK]", b"[MASK]", b"Sq", b"##uare", b"##pants"],
              [b"[MASK]", b"[MASK]", b"[MASK]", b"[MASK]"],
              [b"[MASK]", b"[MASK]", b"[MASK]", b"[MASK]", b"[MASK]"]
          ],
      ),
      dict(
          description=b"Masking whole words w/ no_mask_ids",
          masking_inputs=[[[b"Sp", b"##onge"], [b"bob"],
                           [b"Sq", b"##uare", b"##pants"]],
                          [[b"Bar", b"##ack"], [b"Ob", b"##ama"]],
                          [[b"Mar", b"##vel"], [b"A", b"##ven", b"##gers"]]],
          no_mask_ids=[b"##onge", b"Mar"],
          expected_masked_positions=[[2, 3, 4, 5], [0, 1, 2, 3], [2, 3, 4]],
          expected_masked_ids=[[b"bob", b"Sq", b"##uare", b"##pants"],
                               [b"Bar", b"##ack", b"Ob", b"##ama"],
                               [b"A", b"##ven", b"##gers"]],
          expected_input_ids=[[
              b"Sp", b"##onge", b"[MASK]", b"[MASK]", b"[MASK]", b"[MASK]"
          ], [b"[MASK]", b"[MASK]", b"[MASK]",
              b"[MASK]"], [b"Mar", b"##vel", b"[MASK]", b"[MASK]", b"[MASK]"]],
          axis=1,
      ),
      dict(
          description=b"Masking arbitrary spans",
          # [batch, (num_spans), (num_tokens), (num_wordpieces)]
          masking_inputs=[
              # "Sponge bob" is a single span
              [[[b"Sp", b"##onge"], [b"bob"]], [[b"Sq", b"##uare", b"##pants"]],
               [[b"Hates"]], [[b"Patrick"]]],
              # "Barack Obama"is a single span
              [[[b"Bar", b"##ack"], [b"Ob", b"##ama"]], [[b"is"]],
               [[b"President"]]],
              [[[b"Mar", b"##vel"]], [[b"A", b"##ven", b"##gers"]], [[b"has"]],
               [[b"the"]], [[b"highest"]], [[b"box"]], [[b"office"]]],
          ],
          expected_masked_positions=[[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4],
                                     [0, 1, 2, 3, 4]],
          expected_masked_ids=[[
              b"Sp", b"##onge", b"bob", b"Sq", b"##uare", b"##pants"
          ], [b"Bar", b"##ack", b"Ob", b"##ama", b"is"],
                               [b"Mar", b"##vel", b"A", b"##ven", b"##gers"]],
          expected_input_ids=[
              [
                  b"[MASK]", b"[MASK]", b"[MASK]", b"[MASK]", b"[MASK]",
                  b"[MASK]", b"Hates", b"Patrick"
              ],
              [
                  b"[MASK]", b"[MASK]", b"[MASK]", b"[MASK]", b"[MASK]",
                  b"President"
              ],
              [
                  b"[MASK]", b"[MASK]", b"[MASK]", b"[MASK]", b"[MASK]", b"has",
                  b"the", b"highest", b"box", b"office"
              ],
          ],
      ),
  ])
  def testMaskingOps(self,
                     masking_inputs,
                     expected_masked_positions,
                     description,
                     expected_input_ids=None,
                     expected_masked_ids=None,
                     selection_rate=None,
                     mask_token_rate=1,
                     random_token_rate=0,
                     shuffle_fn=None,
                     no_mask_ids=None,
                     max_selections_per_batch=10,
                     axis=1):

    if no_mask_ids:
      no_mask_ids = [_VOCAB.index(i) for i in no_mask_ids]
    item_selector = item_selector_ops.FirstNItemSelector(2, no_mask_ids)
    values_chooser = masking_ops.MaskValuesChooser(
        vocab_size=_VOCAB_SIZE,
        mask_token=_MASK_TOKEN,
        mask_token_rate=mask_token_rate,
        random_token_rate=random_token_rate)

    return self.runMaskingOpsTest(
        masking_inputs=masking_inputs,
        expected_masked_positions=expected_masked_positions,
        description=description,
        expected_input_ids=expected_input_ids,
        expected_masked_ids=expected_masked_ids,
        axis=axis,
        item_selector=item_selector,
        values_chooser=values_chooser,
    )

  @parameterized.parameters([
      dict(
          description="Masking wordpieces, no masking, nor random injection " +
          "allowed",
          masking_inputs=[[
              b"Sp", b"##onge", b"bob", b"Sq", b"##uare", b"##pants"
          ], [b"Bar", b"##ack", b"Ob", b"##ama"],
                          [b"Mar", b"##vel", b"A", b"##ven", b"##gers"]],
          expected_masked_positions=[[], [], []],
          expected_masked_ids=[[], [], []],
          expected_input_ids=[[
              b"Sp", b"##onge", b"bob", b"Sq", b"##uare", b"##pants"
          ], [b"Bar", b"##ack", b"Ob", b"##ama"],
                              [b"Mar", b"##vel", b"A", b"##ven", b"##gers"]],
      ),
  ])
  def testNothingSelectedMasker(self,
                                masking_inputs,
                                expected_masked_positions,
                                description,
                                expected_input_ids=None,
                                expected_masked_ids=None,
                                axis=1):
    item_selector = item_selector_ops.NothingSelector()
    values_chooser = masking_ops.MaskValuesChooser(_VOCAB_SIZE, _MASK_TOKEN,
                                                   0.9, 0.1)
    return self.runMaskingOpsTest(
        masking_inputs=masking_inputs,
        description=description,
        expected_input_ids=expected_input_ids,
        expected_masked_ids=expected_masked_ids,
        expected_masked_positions=expected_masked_positions,
        axis=axis,
        item_selector=item_selector,
        values_chooser=values_chooser,
    )

  @parameterized.parameters([
      dict(
          description=b"Masking wordpieces, all random",
          masking_inputs=[[
              b"Sp", b"##onge", b"bob", b"Sq", b"##uare", b"##pants"
          ], [b"Bar", b"##ack", b"Ob", b"##ama"],
                          [b"Mar", b"##vel", b"A", b"##ven", b"##gers"]],
          mask_token_rate=0.001,
          random_token_rate=0.9,
          expected_masked_positions=[[0, 1], [0, 1], [0, 1]],
          expected_masked_ids=[
              [b"Sp", b"##onge"],
              [b"Bar", b"##ack"],
              [b"Mar", b"##vel"],
          ],
          expected_input_ids=[[
              b"[RANDOM]", b"[RANDOM]", b"bob", b"Sq", b"##uare", b"##pants"
          ], [b"[RANDOM]", b"[RANDOM]", b"Ob",
              b"##ama"], [b"[RANDOM]", b"[RANDOM]", b"A", b"##ven", b"##gers"]],
      ),
      dict(
          description=b"Masking whole words w/ random injections",
          masking_inputs=[[[b"Sp", "##onge"], [b"bob"],
                           [b"Sq", b"##uare", b"##pants"]],
                          [[b"Bar", "##ack"], [b"Ob", b"##ama"]],
                          [[b"Mar", "##vel"], [b"A", b"##ven", b"##gers"]]],
          mask_token_rate=0,
          random_token_rate=1,
          expected_masked_positions=[[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]],
          expected_masked_ids=[[b"Sp", b"##onge", b"bob"],
                               [b"Bar", b"##ack", b"Ob", b"##ama"],
                               [b"Mar", b"##vel", b"A", b"##ven", b"##gers"]],
          expected_input_ids=[
              [
                  b"[RANDOM]", b"[RANDOM]", b"[RANDOM]", b"Sq", b"##uare",
                  b"##pants"
              ],
              [b"[RANDOM]", b"[RANDOM]", b"[RANDOM]", b"[RANDOM]"],
              [b"[RANDOM]", b"[RANDOM]", b"[RANDOM]", b"[RANDOM]", b"[RANDOM]"],
          ]),
  ])
  def testRandomMasking(self,
                        masking_inputs,
                        expected_masked_positions,
                        description,
                        expected_input_ids=None,
                        expected_masked_ids=None,
                        selection_rate=None,
                        mask_token_rate=1,
                        random_token_rate=None,
                        shuffle_fn=None,
                        no_mask_ids=None,
                        max_selections_per_batch=10,
                        axis=1):
    if no_mask_ids:
      no_mask_ids = [_VOCAB.index(i) for i in no_mask_ids]

    item_selector = item_selector_ops.FirstNItemSelector(2, no_mask_ids)
    values_chooser = AlwaysRandomValuesChooser(
        vocab_size=_VOCAB_SIZE,
        mask_token=_MASK_TOKEN,
        random_token=_RANDOM_TOKEN,
        mask_token_rate=mask_token_rate,
        random_token_rate=random_token_rate)
    return self.runMaskingOpsTest(
        masking_inputs=masking_inputs,
        expected_masked_positions=expected_masked_positions,
        description=description,
        expected_input_ids=expected_input_ids,
        expected_masked_ids=expected_masked_ids,
        axis=axis,
        item_selector=item_selector,
        values_chooser=values_chooser,
    )

  def runMaskingOpsTest(self,
                        masking_inputs,
                        expected_masked_positions,
                        description,
                        expected_input_ids=None,
                        expected_masked_ids=None,
                        axis=1,
                        item_selector=None,
                        values_chooser=None):
    masking_inputs = ragged_factory_ops.constant(masking_inputs)

    # Lookup int IDs
    table = _create_table(_VOCAB)
    self.evaluate(table.initializer)

    # Transform human-readable string wordpieces into int ids, which is what
    # will actually be tested.
    masking_inputs = (
        ragged_functional_ops.map_flat_values(table.lookup, masking_inputs))

    actual_input_ids, actual_masked_positions, actual_masked_ids = (
        masking_ops.mask_language_model(
            masking_inputs,
            axis=axis,
            item_selector=item_selector,
            mask_values_chooser=values_chooser))

    self.assertAllEqual(actual_masked_positions, expected_masked_positions)
    # Decode back into human readable wordpieces for comparison
    actual_masked_ids_flat = array_ops.gather(_VOCAB,
                                              actual_masked_ids.flat_values)
    actual_masked_ids = actual_masked_ids.with_flat_values(
        actual_masked_ids_flat)

    self.assertAllEqual(actual_masked_ids, expected_masked_ids)
    actual_input_ids_flat = array_ops.gather(_VOCAB,
                                             actual_input_ids.flat_values)
    actual_input_ids = actual_input_ids.with_flat_values(actual_input_ids_flat)
    self.assertAllEqual(actual_input_ids, expected_input_ids)

  def testInvalidRates(self):
    with self.assertRaises(errors.InvalidArgumentError):
      values_chooser = masking_ops.MaskValuesChooser(
          _VOCAB_SIZE, _MASK_TOKEN,
          0.9, 5.6)
      self.evaluate(values_chooser.get_mask_values(
          ragged_factory_ops.constant([
              [1, 2, 3], [4, 5]])))


if __name__ == "__main__":
  test.main()
