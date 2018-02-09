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
from tensorflow.contrib.data.python.ops import scan_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
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
    dist_estimation_batch_size = 32
    target_dist_t = ops.convert_to_tensor(target_dist, name="initial_dist")
    class_values_ds = dataset.map(class_func)
    if initial_dist is not None:
      initial_dist_t = ops.convert_to_tensor(initial_dist, name="initial_dist")
      acceptance_dist = _calculate_acceptance_probs(initial_dist_t,
                                                    target_dist_t)
      initial_dist_ds = dataset_ops.Dataset.from_tensors(
          initial_dist_t).repeat()
      acceptance_dist_ds = dataset_ops.Dataset.from_tensors(
          acceptance_dist).repeat()
    else:
      num_classes = (target_dist_t.shape[0].value or
                     array_ops.shape(target_dist_t)[0])
      smoothing_constant = 10
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
      acceptance_dist_ds = initial_dist_ds.map(
          lambda initial: _calculate_acceptance_probs(initial, target_dist_t))

    def maybe_warn_on_large_rejection(accept_dist, initial_dist):
      proportion_rejected = math_ops.reduce_sum(
          (1 - accept_dist) * initial_dist)
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

    current_probabilities_ds = dataset_ops.Dataset.zip(
        (acceptance_dist_ds, class_values_ds)).map(array_ops.gather)
    filtered_ds = (
        dataset_ops.Dataset.zip((class_values_ds, current_probabilities_ds,
                                 dataset))
        .filter(lambda _1, p, _2: random_ops.random_uniform([], seed=seed) < p))
    return filtered_ds.map(lambda class_value, _, data: (class_value, data))

  return _apply_fn


def _calculate_acceptance_probs(initial_probs, target_probs):
  """Calculate the per-class acceptance rates.

  Args:
    initial_probs: The class probabilities of the data.
    target_probs: The desired class proportion in minibatches.
  Returns:
    A list of the per-class acceptance probabilities.

  This method is based on solving the following analysis:

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


  A solution for a_i in terms of the other variabes is the following:
    ```a_i = (t_i / p_i) / max_i[t_i / p_i]```
  """
  # Add tiny to initial_probs to avoid divide by zero.
  denom = (initial_probs + np.finfo(initial_probs.dtype.as_numpy_dtype).tiny)
  ratio_l = target_probs / denom

  # Calculate list of acceptance probabilities.
  max_ratio = math_ops.reduce_max(ratio_l)
  return ratio_l / max_ratio


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
