# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Ops related to candidate sampling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn


def _rank_resample(weights, biases, inputs, sampled_values, num_resampled,
                   resampling_temperature, partition_strategy):
  """A helper function for rank_sampled_softmax_loss.

  This computes, for each i in `sampled_values`,

      log(sum_j exp((w_i * x_j + b_i) / resampling_temperature))

  where w_i, b_i are the weight and bias of the i-th class, repsectively,
  and j ranges over the rows of `inputs`. For efficiency, we rearrange the
  computation to

      log(sum_j exp(w_i * (x_j / resampling_temperature))) +
          b_i / resampling_temperature.

  This translates to the following batched computation using tensorflow ops:

      reduce_logsumexp(matmul(embeddings,
                       transpose(inputs / resampling_temperature))) +
          biases / resampling_temperature

  The computation of the first term is colocated with the embeddings using
  `transform_fn` in `embedding_ops._embedding_lookup_and_transform`. The second
  term, not the bottleneck, is computed at the worker.

  Args:
    weights: From `rank_sampled_softmax_loss`.
    biases: From `rank_sampled_softmax_loss`.
    inputs: From `rank_sampled_softmax_loss`.
    sampled_values: A tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `*_candidate_sampler` function.
    num_resampled: An `int`. This many values are selected from
        `sampled_values` using the adaptive resampling algorithm. The caller
        must ensure that `num_resampled` is less than the size of
        `sampled_values`.
    resampling_temperature: A scalar `Tensor` with the temperature parameter
        for the adaptive resampling algorithm.
    partition_strategy: From `rank_sampled_softmax_loss`.

  Returns:
    A tuple of (`resampled_candidates`, `true_expected_count`,
        `resampled_expected_count`), similar to `sampled_values` but sampled
        down to `num_resampled` values.
  """
  # This code supports passing a Tensor for num_resampled, but since it is only
  # called with an int, that's what we specify in the arg list. If this
  # function is ever externalized, we should change the doc to support Tensor.

  sampled, true_expected_count, sampled_expected_count = sampled_values

  sampled = math_ops.cast(array_ops.stop_gradient(sampled), dtypes.int64)
  true_expected_count = array_ops.stop_gradient(true_expected_count)
  sampled_expected_count = array_ops.stop_gradient(sampled_expected_count)

  reweighted_inputs = inputs / resampling_temperature

  def logsumexp_logit(embeddings):
    return math_ops.reduce_logsumexp(
        math_ops.matmul(embeddings, reweighted_inputs, transpose_b=True),
        axis=1,
        keep_dims=False)

  # Calling this protected form of embedding_lookup allows co-locating
  # the logsumexp computation with the partitioned weights, which yields
  # a large speedup in practice.
  sampled_logits = embedding_ops._embedding_lookup_and_transform(  # pylint: disable=protected-access
      weights, sampled, partition_strategy, transform_fn=logsumexp_logit)
  sampled_b = array_ops.reshape(
      embedding_ops.embedding_lookup(biases, sampled, partition_strategy), [-1])
  sampled_logits += sampled_b / resampling_temperature

  _, resampled_indices = nn.top_k(sampled_logits, k=num_resampled, sorted=False)
  resampled = array_ops.gather(sampled, indices=resampled_indices)
  resampled_expected_count = array_ops.gather(
      sampled_expected_count, indices=resampled_indices)

  return resampled, true_expected_count, resampled_expected_count


def rank_sampled_softmax_loss(weights,
                              biases,
                              labels,
                              inputs,
                              num_sampled,
                              num_resampled,
                              num_classes,
                              num_true,
                              sampled_values,
                              resampling_temperature,
                              remove_accidental_hits,
                              partition_strategy,
                              name=None):
  """Computes softmax loss using rank-based adaptive resampling.

  This has been shown to improve rank loss after training compared to
  @{tf.nn.sampled_softmax_loss}. For a description of the algorithm and some
  experimental results, please see: [TAPAS: Two-pass Approximate Adaptive
  Sampling for Softmax](https://arxiv.org/abs/1707.03073).

  Sampling follows two phases:
  * In the first phase, `num_sampled` classes are selected using
    @{tf.nn.learned_unigram_candidate_sampler} or supplied `sampled_values`.
    The logits are calculated on those sampled classes. This phases is
    similar to @{tf.nn.sampled_softmax_loss}.
  * In the second phase, the `num_resampled` classes with highest predicted
    probability are kept. Probabilities are
    `LogSumExp(logits / resampling_temperature)`, where the sum is over
    `inputs`.

  The `resampling_temperature` parameter controls the "adaptiveness" of the
  resampling. At lower temperatures, resampling is more adaptive because it
  picks more candidates close to the predicted classes. A common strategy is
  to decrease the temperature as training proceeds.

  See @{tf.nn.sampled_softmax_loss} for more documentation on sampling and
  for typical default values for some of the parameters.

  This operation is for training only. It is generally an underestimate of
  the full softmax loss.

  A common use case is to use this method for training, and calculate the full
  softmax loss for evaluation or inference. In this case, you must set
  `partition_strategy="div"` for the two losses to be consistent, as in the
  following example:

  ```python
  if mode == "train":
    loss = rank_sampled_softmax_loss(
        weights=weights,
        biases=biases,
        labels=labels,
        inputs=inputs,
        ...,
        partition_strategy="div")
  elif mode == "eval":
    logits = tf.matmul(inputs, tf.transpose(weights))
    logits = tf.nn.bias_add(logits, biases)
    labels_one_hot = tf.one_hot(labels, n_classes)
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels_one_hot,
        logits=logits)
  ```

  Args:
    weights: A `Tensor` or `PartitionedVariable` of shape `[num_classes, dim]`,
        or a list of `Tensor` objects whose concatenation along dimension 0
        has shape [num_classes, dim]. The (possibly-sharded) class embeddings.
    biases: A `Tensor` or `PartitionedVariable` of shape `[num_classes]`.
        The (possibly-sharded) class biases.
    labels: A `Tensor` of type `int64` and shape `[batch_size,
        num_true]`. The target classes. Note that this format differs from
        the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
    inputs: A `Tensor` of shape `[batch_size, dim]`. The forward
        activations of the input network.
    num_sampled: An `int`. The number of classes to randomly sample per batch.
    num_resampled: An `int`. The number of classes to select from the
        `num_sampled` classes using the adaptive resampling algorithm. Must be
        less than `num_sampled`.
    num_classes: An `int`. The number of possible classes.
    num_true: An `int`.  The number of target classes per training example.
    sampled_values: A tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `*_candidate_sampler` function.
        If None, default to `nn.learned_unigram_candidate_sampler`.
    resampling_temperature: A scalar `Tensor` with the temperature parameter
        for the adaptive resampling algorithm.
    remove_accidental_hits: A `bool`. Whether to remove "accidental hits"
        where a sampled class equals one of the target classes.
    partition_strategy: A string specifying the partitioning strategy, relevant
        if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
        See @{tf.nn.embedding_lookup} for more details.
    name: A name for the operation (optional).

  Returns:
    A `batch_size` 1-D tensor of per-example sampled softmax losses.

  Raises:
    ValueError: If `num_sampled <= num_resampled`.
  """
  if num_sampled > num_classes:
    raise ValueError("num_sampled ({}) cannot be greater than num_classes ({})".
                     format(num_sampled, num_classes))
  if num_sampled <= num_resampled:
    raise ValueError("num_resampled ({}) must be less than num_sampled ({})".
                     format(num_resampled, num_sampled))
  if partition_strategy not in ("div", "mod"):
    raise ValueError(
        "unsupported partition_strategy ({})".format(partition_strategy))
  with ops.name_scope(name, "rank_sampled_softmax_loss", [
      weights, biases, labels, inputs, sampled_values, resampling_temperature
  ]) as name:
    if not sampled_values:
      sampled_values = nn.learned_unigram_candidate_sampler(
          true_classes=labels,
          num_true=num_true,
          num_sampled=num_sampled,
          unique=True,
          range_max=num_classes)
    # From sampled_values, select the top num_resampled values using the
    # adaptive rank resampling strategy.
    resampled_values = _rank_resample(weights, biases, inputs, sampled_values,
                                      num_resampled, resampling_temperature,
                                      partition_strategy)
    return nn.sampled_softmax_loss(
        weights=weights,
        biases=biases,
        labels=labels,
        inputs=inputs,
        num_sampled=num_resampled,
        num_classes=num_classes,
        num_true=num_true,
        sampled_values=resampled_values,
        remove_accidental_hits=remove_accidental_hits,
        partition_strategy=partition_strategy,
        name=name)
