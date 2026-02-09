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

"""Tests for ItemSelectors."""
import functools

from absl.testing import parameterized

from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow_text.python.ops import item_selector_ops


@test_util.run_all_in_graph_and_eager_modes
class FirstNItemSelectorTest(test.TestCase, parameterized.TestCase):
  # pyformat: disable
  @parameterized.parameters([
      dict(
          description="Basic test on 2D `RaggedTensor`",
          masking_inputs=[
              [1, 2, 3, 4, 5, 6],
              [10, 20, 30, 40],
              [100, 200, 300, 400, 500]
          ],
          expected_selectable=[
              [1, 2],
              [10, 20],
              [100, 200]
          ],
      ),
      dict(
          description="Test broadcast",
          masking_inputs=[
              [[1, 2], [3], [4, 5, 6]],
              [[10, 20], [30, 40]],
              [[100, 200], [300, 400, 500]]
          ],
          expected_selectable=[
              [[1, 2], [3]],
              [[10, 20], [30, 40]],
              [[100, 200], [300, 400, 500]]
          ],
      ),
      dict(
          description="Select the first two items. Test broadcast and " +
          "dropping nonselectable ids.",
          masking_inputs=[
              [[1, 2], [3], [4, 5, 6]],
              [[10, 20], [30, 40]],
              [[100, 200], [300, 400, 500]]
          ],
          unselectable_ids=[1, 200],
          expected_selectable=[
              [[3], [4, 5, 6]],
              [[10, 20], [30, 40]],
              [[300, 400, 500]]],
          axis=1,
      ),
      dict(
          description="Select the first two items on axis=-1.",
          masking_inputs=[
              [[b"hello"], [b"there"]],
              [[b"name", b"is"]],
              [[b"what", b"time"], [b"is"], [b"it"], [b"?"]],
          ],
          expected_selectable=[
              [[b"hello"], [b"there"]],
              [[b"name", b"is"]],
              [[b"what", b"time"], [], [], []]],
          axis=-1,
      ),
      dict(
          description="Select the first two items on axis=1.",
          masking_inputs=[
              [[b"hello"], [b"there"]],
              [[b"name", b"is"]],
              [[b"what", b"time"], [b"is"], [b"it"], [b"?"]],
          ],
          expected_selectable=[
              [[b"hello"], [b"there"]],
              [[b"name", b"is"]],
              [[b"what", b"time"], [b"is"]]
          ],
          axis=1,
      ),
      dict(
          description="num_to_select is a 2D Tensor",
          masking_inputs=[
              [1, 2, 3],
              [4, 5],
              [6]
          ],
          expected_selectable=[
              [1, 2],
              [4],
              [6],
          ],
          num_to_select=[[2], [1], [1]],
          axis=-1,
      ),
  ])
  # pyformat: enable

  def testGetSelectable(self,
                        masking_inputs,
                        expected_selectable,
                        num_to_select=2,
                        unselectable_ids=None,
                        axis=1,
                        description=""):
    masking_inputs = ragged_factory_ops.constant(masking_inputs)
    item_selector = item_selector_ops.FirstNItemSelector(
        num_to_select=num_to_select, unselectable_ids=unselectable_ids)
    selectable = item_selector.get_selectable(masking_inputs, axis)
    actual_selection = ragged_array_ops.boolean_mask(masking_inputs, selectable)
    self.assertAllEqual(actual_selection, expected_selectable)


@test_util.run_all_in_graph_and_eager_modes
class LastNItemSelectorTest(test.TestCase, parameterized.TestCase):
  # pyformat: disable
  @parameterized.parameters([
      dict(
          description="Basic test on 2D `RaggedTensor`",
          masking_inputs=[
              [1, 2, 3, 4, 5, 6],
              [10, 20, 30, 40],
              [21],
              [100, 200, 300, 400, 500]
          ],
          expected_selectable=[
              [5, 6],
              [30, 40],
              [21],
              [400, 500]
          ],
      ),
      dict(
          description="Test broadcast",
          masking_inputs=[
              [[1, 2], [3], [4, 5, 6]],
              [[10, 20], [30, 40]],
              [[100, 200], [300, 400, 500]]
          ],
          expected_selectable=[
              [[3], [4, 5, 6]],
              [[10, 20], [30, 40]],
              [[100, 200], [300, 400, 500]]
          ],
      ),
      dict(
          description="Select the last two items. Test broadcast and " +
          "dropping nonselectable ids.",
          masking_inputs=[
              [[1, 2], [3], [4, 5, 6]],
              [[10, 20], [30, 40]],
              [[100, 200], [300, 400, 500]]
          ],
          unselectable_ids=[1, 200],
          expected_selectable=[
              [[3], [4, 5, 6]],
              [[10, 20], [30, 40]],
              [[300, 400, 500]]],
          axis=1,
      ),
      dict(
          description="Select the last two items on axis=-1.",
          masking_inputs=[
              [[b"hello"], [b"there"]],
              [[b"name", b"is"]],
              [[b"what"], [b"time"], [b"is", b"it"], [b"?"], []],
          ],
          expected_selectable=[
              [[b"hello"], [b"there"]],
              [[b"name", b"is"]],
              [[], [], [b"it"], [b"?"], []]
              ],
          axis=-1,
      ),
      dict(
          description="Select the last two items on axis=1.",
          masking_inputs=[
              [[b"hello"], [b"there"]],
              [[b"name", b"is"]],
              [[b"is"], [b"it"], [b"?", b"what"], [b"time"]],
          ],
          expected_selectable=[
              [[b"hello"], [b"there"]],
              [[b"name", b"is"]],
              [[b"?", b"what"], [b"time"]]
          ],
          axis=1,
      ),
      dict(
          description="num_to_select is a 2D Tensor",
          masking_inputs=[
              [1, 2, 3],
              [4, 5],
              [6]
          ],
          expected_selectable=[
              [2, 3],
              [5],
              [6],
          ],
          num_to_select=[[2], [1], [1]],
          axis=-1,
      ),
  ])
  # pyformat: enable

  def testGetSelectable(self,
                        masking_inputs,
                        expected_selectable,
                        num_to_select=2,
                        unselectable_ids=None,
                        axis=1,
                        description=""):
    masking_inputs = ragged_factory_ops.constant(masking_inputs)
    item_selector = item_selector_ops.LastNItemSelector(
        num_to_select=num_to_select, unselectable_ids=unselectable_ids)
    selectable = item_selector.get_selectable(masking_inputs, axis)
    actual_selection = ragged_array_ops.boolean_mask(masking_inputs, selectable)
    self.assertAllEqual(actual_selection, expected_selectable)


@test_util.run_all_in_graph_and_eager_modes
class RandomItemSelectorTest(test.TestCase, parameterized.TestCase):

  # pyformat: disable
  @parameterized.parameters([
      dict(
          description="Basic test on 2D `RaggedTensor`",
          masking_inputs=[
              [1, 2, 3, 4, 5, 6],
              [10, 20, 30, 40],
              [100, 200, 300, 400, 500]
          ],
          expected_selected_items=[
              [1, 2],
              [10, 20],
              [100, 200],
          ],
      ),
      dict(
          description="Test broadcast",
          masking_inputs=[
              [[1, 2], [3], [4, 5, 6]],
              [[10, 20], [30, 40]],
              [[100, 200], [300, 400, 500]]
          ],
          expected_selected_items=[
              [[1, 2], [3]],
              [[10, 20], [30, 40]],
              [[100, 200], [300, 400, 500]]
          ],
      ),
      dict(
          description="Select the first two items that don't have " +
          "unselectable ids; test that broadcasting works appropriately",
          masking_inputs=[
              [[1, 2], [3], [4, 5, 6]],
              [[10, 20], [30, 40]],
              [[100, 200], [300, 400, 500]]
          ],
          unselectable_ids=[1, 200],
          expected_selected_items=[
              [[3], [4, 5, 6]],
              [[10, 20], [30, 40]],
              [[300, 400, 500]]
          ],
          axis=1,
      ),
      dict(
          description="Test shape[:axis+1]",
          masking_inputs=[
              [[0, 1], [2, 3], [4, 5]],
              [],
              [[6, 7]]
          ],
          expected_selected_items=[
              [[0, 1], [2, 3]],
              [],
              [[6, 7]],
          ],
          axis=1,
      ),
      dict(
          description="Test rank 3 ragged tensor selecting on axis=1",
          masking_inputs=[
              [[101], [100], [2045], [1012], [102], [100], [2051],
               [2003], [2009], [1029], [102]],
              [[101], [100], [2292], [1996], [6077], [2041], [1029],
               [102], [100], [1029], [102]]],
          expected_selected_items=[
              [[101], [100]],
              [[101], [100]],
          ],
          axis=1,
      ),
      dict(
          description="Test rank 3 ragged tensor selecting on axis=1, but " +
          "w/ reverse shuffle_fn",
          masking_inputs=[
              [[101], [100], [2045], [1012], [102], [100], [2051],
               [2003], [2009], [1029], [102]],
              [[101], [100], [2292], [1996], [6077], [2041], [1029],
               [102], [100], [1029], [102]]],
          expected_selected_items=[
              [[1029], [102]],
              [[1029], [102]],
          ],
          axis=1,
          shuffle_fn="reverse",
      ),
  ])
  # pyformat: enable
  def testGetSelectionMask(self,
                           masking_inputs,
                           expected_selected_items,
                           unselectable_ids=None,
                           axis=1,
                           shuffle_fn="",
                           description=""):
    shuffle_fn = (
        functools.partial(array_ops.reverse, axis=[-1])
        if shuffle_fn == "reverse" else array_ops.identity)
    masking_inputs = ragged_factory_ops.constant(masking_inputs)
    item_selector = item_selector_ops.RandomItemSelector(
        max_selections_per_batch=2,
        selection_rate=1,
        shuffle_fn=shuffle_fn,
        unselectable_ids=unselectable_ids,
    )
    selection_mask = item_selector.get_selection_mask(masking_inputs, axis)
    selected_items = ragged_array_ops.boolean_mask(masking_inputs,
                                                   selection_mask)
    self.assertAllEqual(selected_items, expected_selected_items)


@test_util.run_all_in_graph_and_eager_modes
class NothingSelectorTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      dict(
          description="Basic test",
          masking_inputs=[[[1, 2], [3], [4, 5, 6]], [[10, 20], [30, 40]],
                          [[100, 200], [300, 400, 500]]],
          unselectable_ids=[1, 200],
          expected_selected_items=[[], [], []],
      ),
  ])
  def testNothingSelector(self,
                          masking_inputs,
                          unselectable_ids,
                          expected_selected_items,
                          num_to_select=2,
                          axis=1,
                          description=""):
    masking_inputs = ragged_factory_ops.constant(masking_inputs)
    item_selector = item_selector_ops.NothingSelector()
    selection_mask = item_selector.get_selectable(masking_inputs, axis)
    selected_items = ragged_array_ops.boolean_mask(masking_inputs,
                                                   selection_mask)
    self.assertAllEqual(selected_items, expected_selected_items)


if __name__ == "__main__":
  test.main()
