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
"""Resampling dataset transformations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.data.python.ops import batching
from tensorflow.contrib.data.python.ops import interleave_ops
from tensorflow.contrib.data.python.ops import scan_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


def rejection_resample(class_func, target_dist, initial_dist=None, seed=None):
  """A transformation that resamples a dataset to achieve a target distribution.

  **NOTE** Resampling is performed via rejection sampling; some fraction
  of the input values will be dropped.

  Args:
    class_func: A function mapping an element of the input dataset to a scalar
      `tf.int32` tensor. Values should be in `[0, num_classes)`.
    target_dist: A floating point type tensor, shaped `[num_classes]`.
    initial_dist: (Optional.)  A floating point type tensor, shaped
      `[num_classes]`.  If not provided, the true class distribution is
      estimated live in a streaming fashion.
    seed: (Optional.) Python integer seed for the resampler.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.data.Dataset.apply}.
  """
  def _apply_fn(dataset):
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    target_dist_t = ops.convert_to_tensor(target_dist, name="target_dist")
    class_values_ds = dataset.map(class_func)

    # Get initial distribution.
    if initial_dist is not None:
      initial_dist_t = math_ops.to_float(ops.convert_to_tensor(initial_dist, name="initial_dist"))
      acceptance_dist, prob_of_original = (
          _calculate_acceptance_probs_with_mixing(initial_dist_t,
                                                  target_dist_t))
      initial_dist_ds = dataset_ops.Dataset.from_tensors(
          initial_dist_t).repeat()
      acceptance_dist_ds = dataset_ops.Dataset.from_tensors(
          acceptance_dist).repeat()
      prob_of_original_ds = dataset_ops.Dataset.from_tensors(
          prob_of_original).repeat()
    else:
      initial_dist_ds = _estimate_initial_dist_ds(
          target_dist_t, class_values_ds)
      acceptance_and_original_prob_ds = initial_dist_ds.map(
          lambda initial: _calculate_acceptance_probs_with_mixing(
              initial, target_dist_t))
      acceptance_dist_ds = acceptance_and_original_prob_ds.map(
          lambda accept_prob, _: accept_prob)
      prob_of_original_ds = acceptance_and_original_prob_ds.map(
          lambda _, prob_original: prob_original)
    filtered_ds = _filter_ds(dataset, acceptance_dist_ds, initial_dist_ds,
                             class_values_ds, seed)
    # Prefetch filtered dataset for speed.
    filtered_ds = filtered_ds.prefetch(3)

    prob_original_static = _get_prob_original_static(
        initial_dist_t, target_dist_t) if initial_dist is not None else None
    if prob_original_static == 1:
      return dataset_ops.Dataset.zip((class_values_ds, dataset))
    elif prob_original_static == 0:
      return filtered_ds
    else:
      return interleave_ops.sample_from_datasets(
          [dataset_ops.Dataset.zip((class_values_ds, dataset)), filtered_ds],
          weights=prob_of_original_ds.map(lambda prob: [(prob, 1.0 - prob)]),
          seed=seed)

  return _apply_fn


def _get_prob_original_static(initial_dist_t, target_dist_t):
  """Returns the static probability of sampling from the original.

  `tensor_util.constant_value(prob_of_original)` returns `None` if it encounters
  an Op that it isn't defined for. We have some custom logic to avoid this.

  Args:
    initial_dist_t: A tensor of the initial distribution.
    target_dist_t: A tensor of the target distribution.

  Returns:
    The probability of sampling from the original distribution as a constant,
    if it is a constant, or `None`.
  """
  init_static = tensor_util.constant_value(initial_dist_t)
  target_static = tensor_util.constant_value(target_dist_t)

  if init_static is None or target_static is None:
    return None
  else:
    return np.min(target_static / init_static)


def _filter_ds(dataset, acceptance_dist_ds, initial_dist_ds, class_values_ds,
               seed):
  """Filters a dataset based on per-class acceptance probabilities.

  Args:
    dataset: The dataset to be filtered.
    acceptance_dist_ds: A dataset of acceptance probabilities.
    initial_dist_ds: A dataset of the initial probability distribution, given or
        estimated.
    class_values_ds: A dataset of the corresponding classes.
    seed: (Optional.) Python integer seed for the resampler.

  Returns:
    A dataset of (class value, data) after filtering.
  """
  def maybe_warn_on_large_rejection(accept_dist, initial_dist):
    proportion_rejected = math_ops.reduce_sum((1 - accept_dist) * initial_dist)
    return control_flow_ops.cond(
        math_ops.less(proportion_rejected, .5),
        lambda: accept_dist,
        lambda: logging_ops.Print(  # pylint: disable=g-long-lambda
            accept_dist, [proportion_rejected, initial_dist, accept_dist],
            message="Proportion of examples rejected by sampler is high: ",
            summarize=100,
            first_n=10))

  acceptance_dist_ds = (dataset_ops.Dataset.zip((acceptance_dist_ds,
                                                 initial_dist_ds))
                        .map(maybe_warn_on_large_rejection))

  def _gather_and_copy(class_val, acceptance_prob, data):
    return class_val, array_ops.gather(acceptance_prob, class_val), data

  current_probabilities_and_class_and_data_ds = dataset_ops.Dataset.zip(
      (class_values_ds, acceptance_dist_ds, dataset)).map(_gather_and_copy)
  filtered_ds = (
      current_probabilities_and_class_and_data_ds
      .filter(lambda _1, p, _2: random_ops.random_uniform([], seed=seed) < p))
  return filtered_ds.map(lambda class_value, _, data: (class_value, data))


def _estimate_initial_dist_ds(
    target_dist_t, class_values_ds, dist_estimation_batch_size=32,
    smoothing_constant=10):
  num_classes = (target_dist_t.shape[0].value or
                 array_ops.shape(target_dist_t)[0])
  initial_examples_per_class_seen = array_ops.fill(
      [num_classes], np.int64(smoothing_constant))

  def update_estimate_and_tile(num_examples_per_class_seen, c):
    updated_examples_per_class_seen, dist = _estimate_data_distribution(
        c, num_examples_per_class_seen)
    tiled_dist = array_ops.tile(
        array_ops.expand_dims(dist, 0), [dist_estimation_batch_size, 1])
    return updated_examples_per_class_seen, tiled_dist

  initial_dist_ds = (class_values_ds.batch(dist_estimation_batch_size)
                     .apply(scan_ops.scan(initial_examples_per_class_seen,
                                          update_estimate_and_tile))
                     .apply(batching.unbatch()))

  return initial_dist_ds


def _get_target_to_initial_ratio(initial_probs, target_probs):
  # Add tiny to initial_probs to avoid divide by zero.
  denom = (initial_probs + np.finfo(initial_probs.dtype.as_numpy_dtype).tiny)
  return target_probs / denom


def _estimate_data_distribution(c, num_examples_per_class_seen):
  """Estimate data distribution as labels are seen.

  Args:
    c: The class labels.  Type `int32`, shape `[batch_size]`.
    num_examples_per_class_seen: Type `int64`, shape `[num_classes]`,
      containing counts.

  Returns:
    num_examples_per_lass_seen: Updated counts.  Type `int64`, shape
      `[num_classes]`.
    dist: The updated distribution.  Type `float32`, shape `[num_classes]`.
  """
  num_classes = num_examples_per_class_seen.get_shape()[0].value
  # Update the class-count based on what labels are seen in batch.
  num_examples_per_class_seen = math_ops.add(
      num_examples_per_class_seen, math_ops.reduce_sum(
          array_ops.one_hot(c, num_classes, dtype=dtypes.int64), 0))
  init_prob_estimate = math_ops.truediv(
      num_examples_per_class_seen,
      math_ops.reduce_sum(num_examples_per_class_seen))
  dist = math_ops.cast(init_prob_estimate, dtypes.float32)
  return num_examples_per_class_seen, dist


def _calculate_acceptance_probs_with_mixing(initial_probs, target_probs):
  """Calculates the acceptance probabilities and mixing ratio.

  In this case, we assume that we can *either* sample from the original data
  distribution with probability `m`, or sample from a reshaped distribution
  that comes from rejection sampling on the original distribution. This
  rejection sampling is done on a per-class basis, with `a_i` representing the
  probability of accepting data from class `i`.

  This method is based on solving the following analysis for the reshaped
  distribution:

  Let F be the probability of a rejection (on any example).
  Let p_i be the proportion of examples in the data in class i (init_probs)
  Let a_i is the rate the rejection sampler should *accept* class i
  Let t_i is the target proportion in the minibatches for class i (target_probs)

  ```
  F = sum_i(p_i * (1-a_i))
    = 1 - sum_i(p_i * a_i)     using sum_i(p_i) = 1
  ```

  An example with class `i` will be accepted if `k` rejections occur, then an
  example with class `i` is seen by the rejector, and it is accepted. This can
  be written as follows:

  ```
  t_i = sum_k=0^inf(F^k * p_i * a_i)
      = p_i * a_j / (1 - F)    using geometric series identity, since 0 <= F < 1
      = p_i * a_i / sum_j(p_j * a_j)        using F from above
  ```

  Note that the following constraints hold:
  ```
  0 <= p_i <= 1, sum_i(p_i) = 1
  0 <= a_i <= 1
  0 <= t_i <= 1, sum_i(t_i) = 1
  ```

  A solution for a_i in terms of the other variables is the following:
    ```a_i = (t_i / p_i) / max_i[t_i / p_i]```

  If we try to minimize the amount of data rejected, we get the following:

  M_max = max_i [ t_i / p_i ]
  M_min = min_i [ t_i / p_i ]

  The desired probability of accepting data if it comes from class `i`:

  a_i = (t_i/p_i - m) / (M_max - m)

  The desired probability of pulling a data element from the original dataset,
  rather than the filtered one:

  m = M_min

  Args:
    initial_probs: A Tensor of the initial probability distribution, given or
      estimated.
    target_probs: A Tensor of the corresponding classes.

  Returns:
    (A 1D Tensor with the per-class acceptance probabilities, the desired
    probability of pull from the original distribution.)
  """
  ratio_l = _get_target_to_initial_ratio(initial_probs, target_probs)
  max_ratio = math_ops.reduce_max(ratio_l)
  min_ratio = math_ops.reduce_min(ratio_l)

  # Target prob to sample from original distribution.
  m = min_ratio

  # TODO(joelshor): Simplify fraction, if possible.
  a_i = (ratio_l - m) / (max_ratio - m)
  return math_ops.to_float(a_i), math_ops.to_float(m)