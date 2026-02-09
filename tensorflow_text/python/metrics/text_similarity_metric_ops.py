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

"""Ops to compute similarity metrics between texts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops.ragged import ragged_tensor

# pylint: disable=g-bad-import-order
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_text_similarity_metric_ops = load_library.load_op_library(resource_loader.get_path_to_datafile('_text_similarity_metric_ops.so'))


def rouge_l(hypotheses, references, alpha=None):
  """Computes LCS-based similarity score between the hypotheses and references.

  The Rouge-L metric is a score from 0 to 1 indicating how similar two sequences
  are, based on the length of the longest common subsequence (LCS).  In
  particular, Rouge-L is the weighted harmonic mean (or f-measure) combining
  the LCS precision (the percentage of the hypothesis sequence covered by the
  LCS) and the LCS recall (the percentage of the reference sequence covered by
  the LCS).

  Source: https://www.microsoft.com/en-us/research/publication/
          rouge-a-package-for-automatic-evaluation-of-summaries/

  This method returns the F-measure, Precision, and Recall for each
  (hypothesis, reference) pair.

  Alpha is used as a weight for the harmonic mean of precision and recall. A
  value of 0 means recall is more important and 1 means precision is
  more important. Leaving alpha unset implies alpha=.5, which is the default in
  the official ROUGE-1.5.5.pl script. Setting alpha to a negative number
  triggers a compatibility mode with the tensor2tensor implementation of
  ROUGE-L.

  >>> hypotheses = tf.ragged.constant([["a","b"]])
  >>> references = tf.ragged.constant([["b"]])
  >>> f, p, r = rouge_l(hypotheses, references, alpha=1)
  >>> print("f: %s, p: %s, r: %s" % (f, p, r))
  f: tf.Tensor([0.5], shape=(1,), dtype=float32),
  p: tf.Tensor([0.5], shape=(1,), dtype=float32),
  r: tf.Tensor([1.], shape=(1,), dtype=float32)

  Args:
    hypotheses: A RaggedTensor with shape [N, (hyp_sentence_len)] and integer or
        string values.
    references: A RaggedTensor with shape [N, (ref_sentence_len)] and integer or
        string values.
    alpha: optional float parameter for weighting

  Returns:
    an (f_measure, p_measure, r_measure) tuple, where each element is a
      vector of floats with shape [N]. The i-th float in each vector contains
      the similarity measure of hypotheses[i] and references[i].
  """
  if not isinstance(hypotheses, ragged_tensor.RaggedTensor):
    raise ValueError('hypotheses must be a RaggedTensor')
  if not isinstance(references, ragged_tensor.RaggedTensor):
    raise ValueError('references must be a RaggedTensor')
  if hypotheses.ragged_rank != 1:
    raise ValueError('hypotheses.ragged_rank must be 1')
  if references.ragged_rank != 1:
    raise ValueError('references.ragged_rank must be 1')
  if alpha is None:
    alpha = .5
  if isinstance(alpha, (float, int)) and alpha > 1:
    raise ValueError('alpha cannot be greater than 1')
  with ops.name_scope(None, 'RougeL', [hypotheses, references]):
    return gen_text_similarity_metric_ops.rouge_l(
        hypotheses.values,
        hypotheses.row_splits,
        references.values,
        references.row_splits,
        alpha)
