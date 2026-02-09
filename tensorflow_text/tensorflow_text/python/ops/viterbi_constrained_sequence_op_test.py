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

"""Tests for tensorflow_text.python.ops.viterbi_constrained_sequence_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow_text.python.numpy import viterbi_decode
from tensorflow_text.python.ops import viterbi_constrained_sequence_op as sequence_op


# TODO(b/122968457): Refactor this test logic.
@test_util.run_all_in_graph_and_eager_modes
class ViterbiConstrainedSequenceOpTest(test_util.TensorFlowTestCase):

  # Test a corner case when there is no data for a batch. In this case there
  # may be garbage data for the scores in the batch matrix, but the
  # sequence_length is zero for the batch.
  def test_empty_rows(self):
    use_log_space = True
    use_start_and_end_states = False
    # [batch, num_tokens, num_states]
    scores = np.array(
        [
            [[10.0, 12.0, 6.0, 4.0], [-13.0, -12.0, -11.0, -10.0]],
            # This batch should be ignored
            [[10.0, 12.0, 6.0, 4.0], [-13.0, -12.0, -11.0, -10.0]],
        ],
        dtype='f')

    # [batch] each indicating row size for batch. The second row has no data.
    sequence_length = [2, 0]

    # [num_states, num_states]
    allowed_transitions = np.array([[True, True, True, True],
                                    [True, True, True, True],
                                    [True, False, True, False],
                                    [True, True, True, True]])

    multiple_sequence_op = sequence_op.viterbi_constrained_sequence(
        scores,
        sequence_length,
        allowed_transitions=allowed_transitions,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)
    multiple_sequence_result = self.evaluate(multiple_sequence_op)
    self.assertAllEqual(multiple_sequence_result, [[1, 3], []])

  def test_sequence_in_exp_space_with_start_end_states_single_input(self):
    use_log_space = False
    use_start_and_end_states = True
    scores = np.array([[10.0, 12.0, 6.0, 4.0], [13.0, 12.0, 11.0, 10.0]])
    # pyformat: disable
    # pylint: disable=bad-whitespace
    # pylint: disable=bad-continuation
    transition_weights = np.array([[ .1,  .2,  .3,  .4, .1],
                                   [ .5,  .6,  .7,  .8, .1],
                                   [ .9,   1, .15,   1, .1],
                                   [.25, .35, .45, .55, .5],
                                   [ .1,  .5,  .1,  .1,  1]], dtype=np.float32)

    allowed_transitions = np.array([[True,  True,  True,  True,  True],
                                    [True,  True,  True,  True,  True],
                                    [True, False,  True, False,  True],
                                    [True,  True,  True,  True,  True],
                                    [True,  True,  True,  True, False]])
    # pyformat: enable
    # pylint: enable=bad-whitespace
    # pylint: enable=bad-continuation
    sequence, _ = viterbi_decode.decode(
        scores,
        transition_weights,
        allowed_transitions,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)

    # Test a multi-item batch.
    multiple_input = np.array([scores, scores, scores], dtype=np.float32)

    multiple_sequence_op = sequence_op.viterbi_constrained_sequence(
        multiple_input, [2, 2, 2],
        allowed_transitions=allowed_transitions,
        transition_weights=transition_weights,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)
    multiple_sequence_result = self.evaluate(multiple_sequence_op)
    self.assertAllEqual(multiple_sequence_result,
                        [sequence, sequence, sequence])

  def test_sequence_in_exp_space_with_start_end_states_multi_input(self):
    use_log_space = False
    use_start_and_end_states = True
    scores = np.array([[10.0, 12.0, 6.0, 4.0], [13.0, 12.0, 11.0, 10.0]])
    # pyformat: disable
    # pylint: disable=bad-whitespace
    # pylint: disable=bad-continuation
    transition_weights = np.array([[ .1,  .2,  .3,  .4, .1],
                                   [ .5,  .6,  .7,  .8, .1],
                                   [ .9,   1, .15,   1, .1],
                                   [.25, .35, .45, .55, .5],
                                   [ .1,  .5,  .1,  .1,  1]], dtype=np.float32)

    allowed_transitions = np.array([[True,  True,  True,  True,  True],
                                    [True,  True,  True,  True,  True],
                                    [True, False,  True, False,  True],
                                    [True,  True,  True,  True,  True],
                                    [True,  True,  True,  True, False]])
    # pyformat: enable
    # pylint: enable=bad-whitespace
    # pylint: enable=bad-continuation
    sequence, _ = viterbi_decode.decode(
        scores,
        transition_weights,
        allowed_transitions,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)

    # Test a multi-item batch.
    multiple_input = np.array([scores, scores, scores], dtype=np.float32)

    multiple_sequence_op = sequence_op.viterbi_constrained_sequence(
        multiple_input, [2, 2, 2],
        allowed_transitions=allowed_transitions,
        transition_weights=transition_weights,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)
    multiple_sequence_result = self.evaluate(multiple_sequence_op)
    self.assertAllEqual(multiple_sequence_result,
                        [sequence, sequence, sequence])

  def test_sequence_in_exp_space_without_start_end_states_single_input(self):
    use_log_space = False
    use_start_and_end_states = False
    scores = np.array([[10.0, 12.0, 6.0, 4.0], [13.0, 12.0, 11.0, 10.0]])
    # pyformat: disable
    # pylint: disable=bad-whitespace
    # pylint: disable=bad-continuation
    transition_weights = np.array([[ .1,  .2,  .3,  .4],
                                   [ .5,  .6,  .7,  .8],
                                   [ .9,  .1, .15,  .2],
                                   [.25, .35, .45, .55]], dtype=np.float32)

    allowed_transitions = np.array([[True,  True,  True,  True],
                                    [True,  True,  True,  True],
                                    [True, False,  True, False],
                                    [True,  True,  True,  True]])
    # pyformat: enable
    # pylint: enable=bad-whitespace
    # pylint: enable=bad-continuation
    sequence, _ = viterbi_decode.decode(
        scores,
        transition_weights,
        allowed_transitions,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)

    # Test a single-item batch.
    single_input = np.array([scores], dtype=np.float32)
    single_sequence_op = sequence_op.viterbi_constrained_sequence(
        single_input, [2],
        allowed_transitions=allowed_transitions,
        transition_weights=transition_weights,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)
    single_result = self.evaluate(single_sequence_op)
    self.assertAllEqual(single_result, [sequence])

  def test_sequence_in_exp_space_without_start_end_states_multi_input(self):
    use_log_space = False
    use_start_and_end_states = False
    scores = np.array([[10.0, 12.0, 6.0, 4.0], [13.0, 12.0, 11.0, 10.0]])
    # pyformat: disable
    # pylint: disable=bad-whitespace
    # pylint: disable=bad-continuation
    transition_weights = np.array([[ .1,  .2,  .3,  .4],
                                   [ .5,  .6,  .7,  .8],
                                   [ .9,  .1, .15,  .2],
                                   [.25, .35, .45, .55]], dtype=np.float32)

    allowed_transitions = np.array([[True,  True,  True,  True],
                                    [True,  True,  True,  True],
                                    [True, False,  True, False],
                                    [True,  True,  True,  True]])
    # pyformat: enable
    # pylint: enable=bad-whitespace
    # pylint: enable=bad-continuation
    sequence, _ = viterbi_decode.decode(
        scores,
        transition_weights,
        allowed_transitions,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)

    # Test a multi-item batch.
    multiple_input = np.array([scores, scores, scores], dtype=np.float32)

    multiple_sequence_op = sequence_op.viterbi_constrained_sequence(
        multiple_input, [2, 2, 2],
        allowed_transitions=allowed_transitions,
        transition_weights=transition_weights,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)
    multiple_sequence_result = self.evaluate(multiple_sequence_op)
    self.assertAllEqual(multiple_sequence_result,
                        [sequence, sequence, sequence])

  def test_sequence_in_log_space_with_start_end_states_single_input(self):
    use_log_space = True
    use_start_and_end_states = True
    scores = np.array([[10.0, 12.0, 7.0, 4.0], [13.0, 12.0, 11.0, 10.0]])
    # pyformat: disable
    # pylint: disable=bad-whitespace
    # pylint: disable=bad-continuation
    transition_weights = np.array([[-1.0,  1.0, -2.0,  2.0, 0.0],
                                   [ 3.0, -3.0,  4.0, -4.0, 0.0],
                                   [ 5.0,  1.0, 10.0,  1.0, 1.0],
                                   [-7.0,  7.0, -8.0,  8.0, 0.0],
                                   [ 0.0,  1.0,  2.0,  3.0, 0.0]],
                                  dtype=np.float32)

    allowed_transitions = np.array([[True,  True,  True,  True,  True],
                                    [True,  True,  True,  True,  True],
                                    [True, False,  True, False, False],
                                    [True,  True,  True,  True,  True],
                                    [True, False,  True,  True,  True]])
    # pyformat: enable
    # pylint: enable=bad-whitespace
    # pylint: enable=bad-continuation
    sequence, _ = viterbi_decode.decode(
        scores,
        transition_weights,
        allowed_transitions,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)

    # Test a single-item batch.
    single_input = np.array([scores], dtype=np.float32)
    single_sequence_op = sequence_op.viterbi_constrained_sequence(
        single_input, [2],
        allowed_transitions=allowed_transitions,
        transition_weights=transition_weights,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)
    single_result = self.evaluate(single_sequence_op)
    self.assertAllEqual(single_result, [sequence])

  def test_sequence_in_log_space_with_start_end_states_multi_input(self):
    use_log_space = True
    use_start_and_end_states = True
    scores = np.array([[10.0, 12.0, 7.0, 4.0], [13.0, 12.0, 11.0, 10.0]])
    # pyformat: disable
    # pylint: disable=bad-whitespace
    # pylint: disable=bad-continuation
    transition_weights = np.array([[-1.0,  1.0, -2.0,  2.0, 0.0],
                                   [ 3.0, -3.0,  4.0, -4.0, 0.0],
                                   [ 5.0,  1.0, 10.0,  1.0, 1.0],
                                   [-7.0,  7.0, -8.0,  8.0, 0.0],
                                   [ 0.0,  1.0,  2.0,  3.0, 0.0]],
                                  dtype=np.float32)

    allowed_transitions = np.array([[True,  True,  True,  True,  True],
                                    [True,  True,  True,  True,  True],
                                    [True, False,  True, False, False],
                                    [True,  True,  True,  True,  True],
                                    [True, False,  True,  True,  True]])
    # pyformat: enable
    # pylint: enable=bad-whitespace
    # pylint: enable=bad-continuation
    sequence, _ = viterbi_decode.decode(
        scores,
        transition_weights,
        allowed_transitions,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)

    # Test a multi-item batch.
    multiple_input = np.array([scores, scores, scores], dtype=np.float32)

    multiple_sequence_op = sequence_op.viterbi_constrained_sequence(
        multiple_input, [2, 2, 2],
        allowed_transitions=allowed_transitions,
        transition_weights=transition_weights,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)
    multiple_sequence_result = self.evaluate(multiple_sequence_op)
    self.assertAllEqual(multiple_sequence_result,
                        [sequence, sequence, sequence])

  def test_sequence_in_log_space_without_start_end_states_single_input(self):
    use_log_space = True
    use_start_and_end_states = False
    scores = np.array([[10.0, 12.0, 6.0, 4.0], [13.0, 12.0, 11.0, 10.0]])
    # pyformat: disable
    # pylint: disable=bad-whitespace
    # pylint: disable=bad-continuation
    transition_weights = np.array([[-1.0,  1.0, -2.0,  2.0],
                                   [ 3.0, -3.0,  4.0, -4.0],
                                   [ 5.0,  1.0, 10.0,  1.0],
                                   [-7.0,  7.0, -8.0,  8.0]], dtype=np.float32)

    allowed_transitions = np.array([[True,  True,  True,  True],
                                    [True,  True,  True,  True],
                                    [True, False,  True, False],
                                    [True,  True,  True,  True]])
    # pyformat: enable
    # pylint: enable=bad-whitespace
    # pylint: enable=bad-continuation
    sequence, _ = viterbi_decode.decode(
        scores,
        transition_weights,
        allowed_transitions,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)

    # Test a single-item batch.
    single_input = np.array([scores], dtype=np.float32)
    single_sequence_op = sequence_op.viterbi_constrained_sequence(
        single_input, [2],
        allowed_transitions=allowed_transitions,
        transition_weights=transition_weights,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)
    single_result = self.evaluate(single_sequence_op)
    self.assertAllEqual(single_result, [sequence])

  def test_sequence_in_log_space_without_start_end_states_multi_input(self):
    use_log_space = True
    use_start_and_end_states = False
    scores = np.array([[10.0, 12.0, 6.0, 4.0], [13.0, 12.0, 11.0, 10.0]])
    # pyformat: disable
    # pylint: disable=bad-whitespace
    # pylint: disable=bad-continuation
    transition_weights = np.array([[-1.0,  1.0, -2.0,  2.0],
                                   [ 3.0, -3.0,  4.0, -4.0],
                                   [ 5.0,  1.0, 10.0,  1.0],
                                   [-7.0,  7.0, -8.0,  8.0]], dtype=np.float32)

    allowed_transitions = np.array([[True,  True,  True,  True],
                                    [True,  True,  True,  True],
                                    [True, False,  True, False],
                                    [True,  True,  True,  True]])
    # pyformat: enable
    # pylint: enable=bad-whitespace
    # pylint: enable=bad-continuation
    sequence, _ = viterbi_decode.decode(
        scores,
        transition_weights,
        allowed_transitions,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)

    # Test a multi-item batch.
    multiple_input = np.array([scores, scores, scores], dtype=np.float32)

    multiple_sequence_op = sequence_op.viterbi_constrained_sequence(
        multiple_input, [2, 2, 2],
        allowed_transitions=allowed_transitions,
        transition_weights=transition_weights,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)
    multiple_sequence_result = self.evaluate(multiple_sequence_op)
    self.assertAllEqual(multiple_sequence_result,
                        [sequence, sequence, sequence])

  def test_sequence_with_none_weights_single_input(self):
    use_log_space = True
    use_start_and_end_states = False
    scores = np.array([[10.0, 12.0, 6.0, 4.0], [13.0, 12.0, 11.0, 10.0]])
    # pyformat: disable
    # pylint: disable=bad-whitespace
    # pylint: disable=bad-continuation
    transition_weights = np.array([[-1.0,  1.0, -2.0,  2.0],
                                   [ 3.0, -3.0,  4.0, -4.0],
                                   [ 5.0,  1.0, 10.0,  1.0],
                                   [-7.0,  7.0, -8.0,  8.0]], dtype=np.float32)

    allowed_transitions = np.array([[True,  True,  True,  True],
                                    [True,  True,  True,  True],
                                    [True, False,  True, False],
                                    [True,  True,  True,  True]])
    # pyformat: enable
    # pylint: enable=bad-whitespace
    # pylint: enable=bad-continuation
    sequence, _ = viterbi_decode.decode(
        scores,
        transition_weights,
        allowed_transitions,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)

    # Test a single-item batch.
    single_input = np.array([scores], dtype=np.float32)
    single_sequence_op = sequence_op.viterbi_constrained_sequence(
        single_input, [2],
        allowed_transitions=allowed_transitions,
        transition_weights=transition_weights,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)
    single_result = self.evaluate(single_sequence_op)
    self.assertAllEqual(single_result, [sequence])

  def test_sequence_with_none_weights_multi_input(self):
    use_log_space = True
    use_start_and_end_states = False
    scores = np.array([[10.0, 12.0, 6.0, 4.0], [13.0, 12.0, 11.0, 10.0]])
    # pyformat: disable
    # pylint: disable=bad-whitespace
    # pylint: disable=bad-continuation
    transition_weights = np.array([[-1.0,  1.0, -2.0,  2.0],
                                   [ 3.0, -3.0,  4.0, -4.0],
                                   [ 5.0,  1.0, 10.0,  1.0],
                                   [-7.0,  7.0, -8.0,  8.0]], dtype=np.float32)

    allowed_transitions = np.array([[True,  True,  True,  True],
                                    [True,  True,  True,  True],
                                    [True, False,  True, False],
                                    [True,  True,  True,  True]])
    # pyformat: enable
    # pylint: enable=bad-whitespace
    # pylint: enable=bad-continuation
    sequence, _ = viterbi_decode.decode(
        scores,
        transition_weights,
        allowed_transitions,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)

    # Test a multi-item batch.
    multiple_input = np.array([scores, scores, scores], dtype=np.float32)

    multiple_sequence_op = sequence_op.viterbi_constrained_sequence(
        multiple_input, [2, 2, 2],
        allowed_transitions=allowed_transitions,
        transition_weights=transition_weights,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)
    multiple_sequence_result = self.evaluate(multiple_sequence_op)
    self.assertAllEqual(multiple_sequence_result,
                        [sequence, sequence, sequence])

  def test_sequence_with_none_permissions_single_input(self):
    use_log_space = True
    use_start_and_end_states = False
    scores = np.array([[10.0, 12.0, 6.0, 4.0], [13.0, 12.0, 11.0, 10.0]])
    # pyformat: disable
    # pylint: disable=bad-whitespace
    # pylint: disable=bad-continuation
    transition_weights = np.array([[-1.0,  1.0, -2.0,  2.0],
                                   [ 3.0, -3.0,  4.0, -4.0],
                                   [ 5.0,  1.0, 10.0,  1.0],
                                   [-7.0,  7.0, -8.0,  8.0]], dtype=np.float32)

    # pyformat: enable
    # pylint: enable=bad-whitespace
    # pylint: enable=bad-continuation
    sequence, _ = viterbi_decode.decode(
        scores,
        transition_weights,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)

    # Test a single-item batch.
    single_input = np.array([scores], dtype=np.float32)
    single_sequence_op = sequence_op.viterbi_constrained_sequence(
        single_input, [2],
        transition_weights=transition_weights,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)
    single_result = self.evaluate(single_sequence_op)
    self.assertAllEqual(single_result, [sequence])

  def test_sequence_with_none_permissions_multi_input(self):
    use_log_space = True
    use_start_and_end_states = False
    scores = np.array([[10.0, 12.0, 6.0, 4.0], [13.0, 12.0, 11.0, 10.0]])
    # pyformat: disable
    # pylint: disable=bad-whitespace
    # pylint: disable=bad-continuation
    transition_weights = np.array([[-1.0,  1.0, -2.0,  2.0],
                                   [ 3.0, -3.0,  4.0, -4.0],
                                   [ 5.0,  1.0, 10.0,  1.0],
                                   [-7.0,  7.0, -8.0,  8.0]], dtype=np.float32)

    # pyformat: enable
    # pylint: enable=bad-whitespace
    # pylint: enable=bad-continuation
    sequence, _ = viterbi_decode.decode(
        scores,
        transition_weights,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)

    # Test a multi-item batch.
    multiple_input = np.array([scores, scores, scores], dtype=np.float32)

    multiple_sequence_op = sequence_op.viterbi_constrained_sequence(
        multiple_input, [2, 2, 2],
        transition_weights=transition_weights,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)
    multiple_sequence_result = self.evaluate(multiple_sequence_op)
    self.assertAllEqual(multiple_sequence_result,
                        [sequence, sequence, sequence])

  def test_multi_input_sequence_with_implicit_lengths(self):
    use_log_space = True
    use_start_and_end_states = False
    scores = np.array([[10.0, 12.0, 6.0, 4.0], [13.0, 12.0, 11.0, 10.0]])
    # pyformat: disable
    # pylint: disable=bad-whitespace
    # pylint: disable=bad-continuation
    transition_weights = np.array([[-1.0,  1.0, -2.0,  2.0],
                                   [ 3.0, -3.0,  4.0, -4.0],
                                   [ 5.0,  1.0, 10.0,  1.0],
                                   [-7.0,  7.0, -8.0,  8.0]], dtype=np.float32)

    # pyformat: enable
    # pylint: enable=bad-whitespace
    # pylint: enable=bad-continuation
    sequence, _ = viterbi_decode.decode(
        scores,
        transition_weights,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)

    # Test a multi-item batch.
    multiple_input = np.array([scores, scores, scores], dtype=np.float32)

    multiple_sequence_op = sequence_op.viterbi_constrained_sequence(
        multiple_input,
        transition_weights=transition_weights,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)
    multiple_sequence_result = self.evaluate(multiple_sequence_op)
    self.assertAllEqual(multiple_sequence_result,
                        [sequence, sequence, sequence])

  def test_single_input_sequence_with_implicit_lengths(self):
    use_log_space = True
    use_start_and_end_states = False
    scores = np.array([[10.0, 13.0, 6.0, 4.0], [13.0, 12.0, 11.0, 10.0],
                       [13.0, 12.0, 11.0, 10.0]])
    # pyformat: disable
    # pylint: disable=bad-whitespace
    # pylint: disable=bad-continuation
    transition_weights = np.array([[-1.0,  1.0, -2.0,  2.0],
                                   [ 3.0, -3.0,  4.0, -4.0],
                                   [ 5.0,  1.0, 10.0,  1.0],
                                   [-7.0,  7.0, -8.0,  8.0]], dtype=np.float32)

    # pyformat: enable
    # pylint: enable=bad-whitespace
    # pylint: enable=bad-continuation
    sequence, _ = viterbi_decode.decode(
        scores,
        transition_weights,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)

    # Test a multi-item batch.
    multiple_input = np.array([scores], dtype=np.float32)

    single_sequence_op = sequence_op.viterbi_constrained_sequence(
        multiple_input,
        transition_weights=transition_weights,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)
    single_sequence_result = self.evaluate(single_sequence_op)
    self.assertAllEqual(single_sequence_result, [sequence])

  def test_ragged_input_sequence(self):
    use_log_space = True
    use_start_and_end_states = False
    input_1 = np.array([[10.0, 13.0, 6.0, 4.0], [13.0, 12.0, 11.0, 10.0],
                        [13.0, 12.0, 11.0, 10.0]])
    input_2 = np.array([[10.0, 12.0, 6.0, 4.0], [13.0, 12.0, 11.0, 10.0]])
    # TODO(b/122968457): Extend RT support to lists-of-ndarrays.
    scores = ragged_factory_ops.constant([input_1.tolist(), input_2.tolist()])
    # pyformat: disable
    # pylint: disable=bad-whitespace
    # pylint: disable=bad-continuation
    transition_weights = np.array([[-1.0,  1.0, -2.0,  2.0],
                                   [ 3.0, -3.0,  4.0, -4.0],
                                   [ 5.0,  1.0, 10.0,  1.0],
                                   [-7.0,  7.0, -8.0,  8.0]], dtype=np.float32)

    # pyformat: enable
    # pylint: enable=bad-whitespace
    # pylint: enable=bad-continuation
    sequence_1, _ = viterbi_decode.decode(
        input_1,
        transition_weights,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)
    sequence_2, _ = viterbi_decode.decode(
        input_2,
        transition_weights,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)
    expected_sequence = ragged_factory_ops.constant([sequence_1, sequence_2])

    # Test a ragged batch
    single_sequence_op = sequence_op.viterbi_constrained_sequence(
        scores,
        transition_weights=transition_weights,
        use_log_space=use_log_space,
        use_start_and_end_states=use_start_and_end_states)
    single_sequence_result = self.evaluate(single_sequence_op)
    self.assertAllEqual(single_sequence_result, expected_sequence)


if __name__ == '__main__':
  test.main()
