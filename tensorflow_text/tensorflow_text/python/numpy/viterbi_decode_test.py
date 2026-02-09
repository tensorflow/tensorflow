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

"""Tests for tensorflow_text.python.numpy.viterbi_decode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import numpy as np

from tensorflow_text.python.numpy import viterbi_decode


class ViterbiDecodeTest(absltest.TestCase):

  def test_viterbi_in_log_space(self):
    scores = np.array([[10.0, 12.0, 6.0, 4.0], [13.0, 12.0, 11.0, 10.0]])
    x = -float('inf')
    # pyformat: disable
    # pylint: disable=bad-whitespace
    # pylint: disable=bad-continuation
    transition_params = np.array([[-1.0,  1.0, -2.0,  2.0],
                                  [ 3.0, -3.0,  4.0, -4.0],
                                  [ 5.0,    x, 10.0,    x],
                                  [-7.0,  7.0, -8.0,  8.0]])
    # pyformat: enable
    # pylint: enable=bad-whitespace
    # pylint: enable=bad-continuation

    # STEP 1:
    #   Starting scores are {10.0, 12.0, 6.0, 4.0}
    #   Raw scores are: {13.0, 12.0, 11.0, 10.0}
    #
    # To get the weighted scores, add the column of the final state to
    # the raw score.
    #
    # Final state 0: (13.0) Weighted scores are {12.0, 16.0, 18.0, 6.0}
    #      New totals are {22, 28, 24, 10} [max 28 from 1]
    #
    # Final state 1: (12.0) Weighted scores are {13.0, 9.0, X, 19.0},
    #      New totals are {23, 21, X, 23} [max 23 from 3]
    #
    # Final state 2: (11.0) Weighted scores are {9, 15, 21, 3},
    #      New totals are {19, 27, 27, 7} [max 27 from 2]
    #
    # Final state 3: (10.0) Weighted scores are {12, 6, X, 18},
    #      New totals are {19, 18, X, 22} [max 25 from 3]
    #
    #   Top scores are [28, 26, 27, 25] from [1, 3, 2, 3].
    #   Final state is [0] with a sequence of [1->0].

    sequence, score = viterbi_decode.decode(scores, transition_params)
    self.assertAlmostEqual(28.0, score)
    self.assertEqual([1, 0], sequence)

  def test_viterbi_with_allowed_transitions(self):
    scores = np.array([[10.0, 12.0, 6.0, 4.0], [13.0, 12.0, 11.0, 10.0]])
    # pyformat: disable
    # pylint: disable=bad-whitespace
    # pylint: disable=bad-continuation
    transition_params = np.array([[-1.0,    1.0, -2.0,    2.0],
                                  [ 3.0,   -3.0,  4.0,   -4.0],
                                  [ 5.0,  100.0, 10.0,  200.0],
                                  [-7.0,    7.0, -8.0,    8.0]])

    allowed_transitions = np.array([[ True,  True,  True,  True],
                                    [ True,  True,  True,  True],
                                    [ True, False,  True, False],
                                    [ True,  True,  True,  True]])
    # pyformat: enable
    # pylint: enable=bad-whitespace
    # pylint: enable=bad-continuation

    # STEP 1:
    #   Starting scores are {10.0, 12.0, 6.0, 4.0}
    #   Raw scores are: {13.0, 12.0, 11.0, 10.0}
    #
    # Final state 0: (13.0) Weighted scores are {12.0, 16.0, 18.0, 6.0}
    #      New totals are {22, 28, 24, 10} [max 28 from 1]
    #
    # Final state 1: (12.0) Weighted scores are {13.0, 9.0, X, 19.0},
    #      New totals are {23, 21, X, 23} [max 23 from 3]
    #
    # Final state 2: (11.0) Weighted scores are {9, 15, 21, 3},
    #      New totals are {19, 27, 27, 7} [max 27 from 2]
    #
    # Final state 3: (10.0) Weighted scores are {12, 6, X, 18},
    #      New totals are {19, 18, X, 22} [max 22 from 3]
    #
    #   Top scores are [28, 26, 27, 25] from [1, 3, 2, 3].
    #   Final state is [0] with a sequence of [1->0].

    sequence, score = viterbi_decode.decode(scores, transition_params,
                                            allowed_transitions)
    self.assertAlmostEqual(28.0, score)
    self.assertEqual([1, 0], sequence)

  def test_viterbi_in_log_space_with_start_and_end(self):
    scores = np.array([[10.0, 12.0, 7.0, 4.0], [13.0, 12.0, 11.0, 10.0]])
    x = -float('inf')
    # pyformat: disable
    # pylint: disable=bad-whitespace
    # pylint: disable=bad-continuation
    transition_params = np.array([[-1.0,  1.0, -2.0,  2.0, 0.0],
                                  [ 3.0, -3.0,  4.0, -4.0, 0.0],
                                  [ 5.0,    x, 10.0,    x,   x],
                                  [-7.0,  7.0, -8.0,  8.0, 0.0],
                                  [ 0.0,    x,  2.0,  3.0, 0.0]])
    # pyformat: enable
    # pylint: enable=bad-whitespace
    # pylint: enable=bad-continuation

    # STEP 1:
    # All scores should be summed with the last row in the weight tensor, so the
    # 'real' scores are:
    # B0: { 10.0, X,  9.0,  7.0}
    #
    # STEP 2:
    #   Raw scores are: {13.0, 12.0, 11.0, 10.0}
    #
    # Final state 0: (13.0) Weighted scores are {12.0, 16.0, 18.0, 6.0}
    #      New totals are {22, X, 27, 18} [max 27 from 2]
    #
    # Final state 1: (12.0) Weighted scores are {13.0, 9.0, X, 19.0},
    #      New totals are {23, X, X, 26} [max 26 from 3]
    #
    # Final state 2: (11.0) Weighted scores are {9, 15, 21, 3},
    #      New totals are {19, X, 30, 10} [max 30 from 2]
    #
    # Final state 3: (10.0) Weighted scores are {12, 6, X, 18},
    #      New totals are {19, X, X, 25} [max 25 from 3]
    #
    #   Top scores are [27, 26, 30, 25] from [2, 3, 2, 3].
    #   2->OUT is X, so final scores are  [27, 26, X, 25] for a
    #   final state of [0] with a sequence of [2->0].

    sequence, score = viterbi_decode.decode(
        scores, transition_params, use_start_and_end_states=True)
    self.assertAlmostEqual(27.0, score)
    self.assertEqual([2, 0], sequence)

  def test_viterbi_in_exp_space(self):
    scores = np.array([[10.0, 12.0, 6.0, 4.0], [13.0, 12.0, 11.0, 10.0]])
    x = 0.0
    # pyformat: disable
    # pylint: disable=bad-whitespace
    # pylint: disable=bad-continuation
    transition_params = np.array([[ .1,  .2,  .3,  .4],
                                  [ .5,  .6,  .7,  .8],
                                  [ .9,   x, .15,   x],
                                  [.25, .35, .45, .55]])
    # pyformat: enable
    # pylint: enable=bad-whitespace
    # pylint: enable=bad-continuation

    # STEP 1:
    #   Starting scores are {10.0, 12.0, 6.0, 4.0}
    #   Raw scores are: {13.0, 12.0, 11.0, 10.0}
    #
    # Final state 0: (13.0) Weighted scores are {1.3, 6.5, 11.7, 3.25}
    #      New totals are {13, 78, 70.2, 13} [max 78 from 1]
    #
    # Final state 1: (12.0) Weighted scores are {2.4, 7.2, 0, 4.2},
    #      New totals are {24, 86.4, 0, 16.8} [max 86.4 from 1]
    #
    # Final state 2: (11.0) Weighted scores are {3.3, 7.7, 1.65, 4.95},
    #      New totals are {33, 92.4, 9.9, 19.8} [max 92.4 from 1]
    #
    # Final state 3: (10.0) Weighted scores are {4, 8, 0, 5.5},
    #      New totals are {40, 96, 0, 22} [max 96 from 1]
    #
    #   Top scores are [78, 86.4, 92.4, 96] from [1, 1, 1, 1].
    #   Final state is [3] with a sequence of [1->3].

    sequence, score = viterbi_decode.decode(
        scores, transition_params, use_log_space=False)
    self.assertAlmostEqual(96.0, score)
    self.assertEqual([1, 3], sequence)

  def test_viterbi_in_exp_space_with_allowed_transitions(self):
    scores = np.array([[10.0, 12.0, 6.0, 4.0], [13.0, 12.0, 11.0, 10.0]])
    # pyformat: disable
    # pylint: disable=bad-whitespace
    # pylint: disable=bad-continuation
    transition_params = np.array([[ .1,  .2,  .3,  .4],
                                  [ .5,  .6,  .7,  .8],
                                  [ .9,  .5, .15,  .5],
                                  [.25, .35, .45, .55]])

    allowed_transitions = np.array([[ True,  True,  True,  True],
                                    [ True,  True,  True,  True],
                                    [ True, False,  True, False],
                                    [ True,  True,  True,  True]])
    # pyformat: enable
    # pylint: enable=bad-whitespace
    # pylint: enable=bad-continuation

    # STEP 1:
    #   Starting scores are {10.0, 12.0, 6.0, 4.0}
    #   Raw scores are: {13.0, 12.0, 11.0, 10.0}
    #
    # Final state 0: (13.0) Weighted scores are {1.3, 6.5, 11.7, 3.25}
    #      New totals are {13, 78, 70.2, 13} [max 78 from 1]
    #
    # Final state 1: (12.0) Weighted scores are {2.4, 7.2, 0, 4.2},
    #      New totals are {24, 86.4, 0, 16.8} [max 86.4 from 1]
    #
    # Final state 2: (11.0) Weighted scores are {3.3, 7.7, 1.65, 4.95},
    #      New totals are {33, 92.4, 9.9, 19.8} [max 92.4 from 1]
    #
    # Final state 3: (10.0) Weighted scores are {4, 8, 0, 5.5},
    #      New totals are {40, 96, 0, 22} [max 96 from 1]
    #
    #   Top scores are [78, 86.4, 92.4, 96] from [1, 1, 1, 1].
    #   Final state is [3] with a sequence of [1->3].

    sequence, score = viterbi_decode.decode(
        scores, transition_params, allowed_transitions, use_log_space=False)
    self.assertAlmostEqual(96.0, score)
    self.assertEqual([1, 3], sequence)

  def test_viterbi_in_exp_space_with_start_and_end(self):
    scores = np.array([[10.0, 12.0, 6.0, 4.0], [13.0, 12.0, 11.0, 10.0]])
    x = 0.0
    # pyformat: disable
    # pylint: disable=bad-whitespace
    # pylint: disable=bad-continuation
    transition_params = np.array([[ .1,  .2,  .3,  .4, .1],
                                  [ .5,  .6,  .7,  .8, .1],
                                  [ .9,   x, .15,   x, .1],
                                  [.25, .35, .45, .55, .5],
                                  [ .1,  .5,  .1,  .1,  x]])
    # pyformat: enable
    # pylint: enable=bad-whitespace
    # pylint: enable=bad-continuation

    # STEP 1:
    #   Starting scores are {.5, 6.0, .6, .4}
    #   Raw scores are: {13.0, 12.0, 11.0, 10.0}
    #
    # Final state 0: (13.0) Weighted scores are {1.3, 6.5, 11.7, 3.25}
    #      New totals are {0.13, 39, 7.02, 1.3} [max 39 from 1]
    #
    # Final state 1: (12.0) Weighted scores are {2.4, 7.2, 0, 4.2},
    #      New totals are {0.24, 43.2, 0, 1.68} [max 43.2 from 1]
    #
    # Final state 2: (11.0) Weighted scores are {3.3, 7.7, 1.65, 4.95},
    #      New totals are {0.33, 46.2, 0.99, 1.98} [max 46.2 from 1]
    #
    # Final state 3: (10.0) Weighted scores are {4, 8, 0, 5.5},
    #      New totals are {0.4, 48, 0, 2.2} [max 48 from 1]
    #
    #   Top scores are [39, 43.2, 46.2, 48] from [1, 1, 1, 1].
    #   Final multiplication results in [3.9, 4.32, 4.62, 24]
    #   Final state is [3] with a sequence of [1->3].

    sequence, score = viterbi_decode.decode(
        scores,
        transition_params,
        use_log_space=False,
        use_start_and_end_states=True)
    self.assertAlmostEqual(24.0, score)
    self.assertEqual([1, 3], sequence)

  def test_viterbi_in_exp_space_with_negative_weights_fails(self):
    scores = np.array([[10.0, 12.0, 6.0, 4.0], [13.0, 12.0, 11.0, 10.0]])
    x = 0.0
    # pyformat: disable
    # pylint: disable=bad-whitespace
    # pylint: disable=bad-continuation
    transition_params = np.array([[ .1,  .2,  .3,  .4],
                                  [ .5, -.6,  .7,  .8],
                                  [ .9,   x, .15,   x],
                                  [.25, .35, .45, .55]])
    # pyformat: enable
    # pylint: enable=bad-whitespace
    # pylint: enable=bad-continuation

    with self.assertRaises(ValueError):
      _, _ = viterbi_decode.decode(
          scores, transition_params, use_log_space=False)


if __name__ == '__main__':
  absltest.main()
