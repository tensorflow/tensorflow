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
"""Tests for text_similarity_metric_ops op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from absl.testing import parameterized
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow_text.python.metrics import text_similarity_metric_ops


def _tokenize_whitespace(text):
  """Tokenizes text by splitting on whitespace."""
  return text.split()


def _tokenize_155_compat(text):
  """Tokenizes text in a manner that is consistent with ROUGE-1.5.5.pl."""
  text = re.sub(r"-", " - ", text)
  text = re.sub(r"[^A-Za-z0-9\-]", " ", text)
  tokens = text.split()
  tokens = [t for t in tokens if re.match(r"^[a-z0-9$]", t)]
  return tokens


_TEST_HYPOTHESES = (
    "the #### transcript is a written version of each day 's cnn "
    "student news program use this transcript to help students with "
    "reading comprehension and vocabulary use the weekly newsquiz "
    "to test your knowledge of storie s you saw on cnn student "
    "news",
    "a u.s. citizen was killed in a a shootout in mississippi in "
    "#### he was shot in the head and died in a bath tub in omaha , "
    "louisiana authorities are investigating the death\",",
    "nelson mandela is a women 's advocate for women , nelson "
    "mandela says nelson mandela is a women 's advocate for women "
    "she says women do n't know how women are women",
    "the captain of the delta flight was en route to <unk> airport "
    ", the coast guard says the plane was carrying ### passengers "
    "and ## crew members the plane was en route from atlanta to the "
    "dominican republic")


_TEST_REFERENCES = (
    "this page includes the show transcript use the transcript to "
    "help students with reading comprehension and vocabulary at the "
    "bottom of the page , comment for a chance to be mentioned on "
    "cnn student news . you must be a teacher or a student age # # "
    "or older to request a mention on the cnn student news roll "
    "call . the weekly newsquiz tests students ' knowledge of even "
    "ts in the news",
    "the fugitive who killed the marshal was \" extremely dangerous "
    ", \" u.s. marshals service director says deputy u.s. marshal "
    "josie wells , ## , died after trying to arrest jamie croom \" "
    "before he 'd go back to jail , he said , he 'd rather be dead, "
    "\" croom 's sister says",
    "cnn 's kelly wallace wonders why women too often do n't lift "
    "each up in the workplace author of \" the woman code \" says "
    "women need to start operating like the boys women need to "
    "realize they win when they help other women get ahead , says "
    "author",
    "delta air lines flight #### skidded into a fence last week at "
    "a laguardia airport beset by winter weather the ntsb says the "
    "crew reported they did not sense any deceleration from the "
    "wheel brake upon landing")


class TextSimilarityMetricOpsTest(test_util.TensorFlowTestCase,
                                  parameterized.TestCase):

  @parameterized.parameters([
      # Corner-case
      dict(
          hyp=[[]],
          ref=[[]],
          expected_f_measures=[0],
          expected_p_measures=[0],
          expected_r_measures=[0],
          value_dtype=dtypes.int32,
      ),
      # Corner-case
      dict(
          hyp=[],
          ref=[],
          expected_f_measures=[],
          expected_p_measures=[],
          expected_r_measures=[],
          value_dtype=dtypes.int32,
      ),
      # Corner-case
      dict(
          hyp=[[]],
          ref=[[1, 2, 3]],
          expected_f_measures=[0],
          expected_p_measures=[0],
          expected_r_measures=[0],
          value_dtype=dtypes.int32,
      ),
      # Corner-case
      dict(
          hyp=[[1, 2, 3]],
          ref=[[]],
          expected_f_measures=[0],
          expected_p_measures=[0],
          expected_r_measures=[0],
          value_dtype=dtypes.int32,
      ),
      # Identical case
      dict(
          hyp=[[1, 2, 3, 4, 5, 1, 6, 7, 0],
               [1, 2, 3, 4, 5, 1, 6]],
          ref=[[1, 2, 3, 4, 5, 1, 6, 7, 0],
               [1, 2, 3, 4, 5, 1, 6]],
          expected_f_measures=[1.0, 1.0],
          expected_p_measures=[1.0, 1.0],
          expected_r_measures=[1.0, 1.0],
      ),
      # Disjoint case
      dict(
          hyp=[[1, 2, 3, 4, 5, 1, 6, 7, 0],
               [1, 2, 3, 4, 5, 1, 6, 8, 7]],
          ref=[[8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
               [9, 10, 11, 12, 13, 14, 15, 16, 17, 0]],
          expected_f_measures=[0.0, 0.0],
          expected_p_measures=[0.0, 0.0],
          expected_r_measures=[0.0, 0.0],
      ),
      # Basic case (alpha=-1)
      dict(
          hyp=[["a", "b",]],
          ref=[["b"]],
          expected_f_measures=[.555],
          expected_p_measures=[.5],
          expected_r_measures=[1.0],
          alpha=-1,
      ),
      # Basic case (alpha=0)
      dict(
          hyp=[["a", "b",]],
          ref=[["b"]],
          expected_f_measures=[1.0],
          expected_p_measures=[.5],
          expected_r_measures=[1.0],
          alpha=0,
      ),
      # Basic case (alpha=1)
      dict(
          hyp=[["a", "b",]],
          ref=[["b"]],
          expected_f_measures=[.5],
          expected_p_measures=[.5],
          expected_r_measures=[1.0],
          alpha=1,
      ),
      # Basic case (alpha=.5)
      dict(
          hyp=[["a", "b",]],
          ref=[["b"]],
          expected_f_measures=[.666],
          expected_p_measures=[.5],
          expected_r_measures=[1.0],
          alpha=.5,
      ),
      # Basic case (alpha=.8)
      dict(
          hyp=[["a", "b",]],
          ref=[["b"]],
          expected_f_measures=[.555],
          expected_p_measures=[.5],
          expected_r_measures=[1.0],
          alpha=.8,
      ),
      # Partial overlap case 1
      dict(
          hyp=[[1, 2, 3, 4, 5, 1, 6, 7, 0],
               [1, 2, 3, 4, 5, 1, 6, 8, 7]],
          ref=[[1, 9, 2, 3, 4, 5, 1, 10, 6, 7],
               [1, 9, 2, 3, 4, 5, 1, 10, 6, 7]],
          expected_f_measures=[.837, .837],
          expected_p_measures=[.889, .889],
          expected_r_measures=[.8, .8],
          alpha=-1,
      ),
      # Partial overlap case 2
      dict(
          hyp=[["12", "23", "34", "45"]],
          ref=[["12", "23"]],
          expected_f_measures=[.555],
          expected_p_measures=[.5],
          expected_r_measures=[1.0],
          alpha=-1,
      ),
      # Obscured sequence case
      dict(
          hyp=[[1, 2, 3]],
          ref=[[1, 2, 3, 2, 3]],
          expected_f_measures=[.671],
          expected_p_measures=[1.0],
          expected_r_measures=[.6],
          alpha=-1,
      ),
      # Thorough test case for Alpha=.5 (default; same as ROUGE-1.5.5.pl).
      #
      # The official ROUGE-1.5.5.pl script computes the following scores for
      # these examples:
      #
      #   f=[.345, .076, .177, .247]
      #   p=[.452, .091, .219, .243]
      #   r=[.279, .065, .149, .250]
      dict(
          hyp=_TEST_HYPOTHESES,
          ref=_TEST_REFERENCES,
          expected_f_measures=[.345, .076, .177, .253],
          expected_p_measures=[.452, .091, .219, .257],
          expected_r_measures=[.279, .065, .149, .250],
          tokenize_fn=_tokenize_155_compat,
      ),
      # Same as above case but with Alpha=0.
      dict(
          hyp=_TEST_HYPOTHESES,
          ref=_TEST_REFERENCES,
          expected_f_measures=[.279, .065, .149, .250],
          expected_p_measures=[.452, .091, .219, .257],
          expected_r_measures=[.279, .065, .149, .250],
          tokenize_fn=_tokenize_155_compat,
          alpha=0,
      ),
      # Same as above case but with Alpha=1
      dict(
          hyp=_TEST_HYPOTHESES,
          ref=_TEST_REFERENCES,
          expected_f_measures=[.452, .091, .219, .257],
          expected_p_measures=[.452, .091, .219, .257],
          expected_r_measures=[.279, .065, .149, .250],
          tokenize_fn=_tokenize_155_compat,
          alpha=1,
      ),
      # Thorough test case for Alpha=-1 (same as tensor2tensor).
      #
      # A popular unofficial implementation of ROUGE-L on Github also reports
      # these values:
      # https://github.com/pltrdy/rouge/blob/master/tests/data.json
      #
      #   f=[.287, .083, .137, .240]
      #   p=[.442, .118, .188, .237]
      #   r=[.257, .074, .122, .243]
      dict(
          hyp=_TEST_HYPOTHESES,
          ref=_TEST_REFERENCES,
          expected_f_measures=[.287, .083, .137, .240],
          expected_p_measures=[.442, .118, .188, .237],
          expected_r_measures=[.257, .074, .122, .243],
          alpha=-1,
          tokenize_fn=_tokenize_whitespace
      ),
  ])
  def testRougeLOp(self, hyp, ref, expected_f_measures, expected_p_measures,
                   expected_r_measures, value_dtype=None, alpha=None,
                   tokenize_fn=None):
    if tokenize_fn:
      hyp = [tokenize_fn(h) for h in hyp]
      ref = [tokenize_fn(r) for r in ref]
    tokens_hyp = ragged_factory_ops.constant(hyp, dtype=value_dtype,
                                             ragged_rank=1)
    tokens_ref = ragged_factory_ops.constant(ref, dtype=value_dtype,
                                             ragged_rank=1)
    forward = text_similarity_metric_ops.rouge_l(
        tokens_hyp, tokens_ref, alpha=alpha)
    # Check tuple ordering+naming.
    self.assertIs(forward.f_measure, forward[0])
    self.assertIs(forward.p_measure, forward[1])
    self.assertIs(forward.r_measure, forward[2])
    # Check actual vs expected values.
    self.assertAllClose(forward.f_measure, expected_f_measures, atol=1e-3)
    self.assertAllClose(forward.p_measure, expected_p_measures, atol=1e-3)
    self.assertAllClose(forward.r_measure, expected_r_measures, atol=1e-3)
    # Reverse alpha.
    if alpha is None or alpha < 0:
      reverse_alpha = alpha
    else:
      reverse_alpha = 1 - alpha
    # Now pass the arguments in reverse.
    reverse = text_similarity_metric_ops.rouge_l(tokens_ref, tokens_hyp,
                                                 alpha=reverse_alpha)
    self.assertAllClose(reverse.f_measure, expected_f_measures, atol=1e-3)
    self.assertAllClose(reverse.p_measure, expected_r_measures, atol=1e-3)
    self.assertAllClose(reverse.r_measure, expected_p_measures, atol=1e-3)

  @parameterized.parameters([
      # Corner-case (input not ragged)
      dict(
          hyp=[],
          ref=[],
      ),
      # Corner-case (input not ragged)
      dict(
          hyp=[[]],
          ref=[],
      ),
      # Corner-case (input not ragged)
      dict(
          hyp=[],
          ref=[[]],
      ),
  ])
  def testRougeLOp_notRagged(self, hyp, ref):
    # Note: ragged_factory_ops.constant returns a tf.Tensor for flat input lists
    tokens_hyp = ragged_factory_ops.constant(hyp, dtype=dtypes.int32)
    tokens_ref = ragged_factory_ops.constant(ref, dtype=dtypes.int32)
    with self.assertRaises(ValueError):
      text_similarity_metric_ops.rouge_l(tokens_hyp, tokens_ref)

  @parameterized.parameters([
      # Corner-case (ref is ragged rank 2)
      dict(
          hyp=[[1, 2, 3]],
          ref=[[[1], []], [[1, 2]]],
      ),
      # Corner-case (hyp is ragged rank 2)
      dict(
          hyp=[[[1], []], [[1, 2]]],
          ref=[[1, 2, 3]],
      ),
  ])
  def testRougeLOp_raggedRank2(self, hyp, ref):
    with self.assertRaises(ValueError):
      text_similarity_metric_ops.rouge_l(hyp, ref)

  def testRougeLOp_alphaValues(self):
    hyp = ragged_factory_ops.constant([[1, 2]], dtype=dtypes.int32)
    ref = ragged_factory_ops.constant([[2, 3]], dtype=dtypes.int32)
    text_similarity_metric_ops.rouge_l(hyp, ref, alpha=-1)
    text_similarity_metric_ops.rouge_l(hyp, ref, alpha=0)
    text_similarity_metric_ops.rouge_l(hyp, ref, alpha=.5)
    text_similarity_metric_ops.rouge_l(hyp, ref, alpha=1)
    with self.assertRaises(ValueError):
      text_similarity_metric_ops.rouge_l(hyp, ref, alpha=1.00001)


if __name__ == "__main__":
  test.main()
