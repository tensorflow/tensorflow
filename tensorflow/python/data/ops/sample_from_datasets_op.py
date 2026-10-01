# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""The implementation of `tf.data.Dataset.sample_from_datasets`."""

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import directed_interleave_op
from tensorflow.python.data.ops import map_op
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_stateless_random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.types import data as data_types


def _sample_from_datasets(datasets,  # pylint: disable=unused-private-name
                          weights=None,
                          seed=None,
                          stop_on_empty_dataset=False,
                          rerandomize_each_iteration=None):
  """See `Dataset.sample_from_datasets()` for details."""

  def _skip_datasets_with_zero_weight(datasets, weights):
    datasets_and_weights = [(dataset, weight)
                            for (dataset, weight) in zip(datasets, weights)
                            if weight > 0]
    return (zip(*datasets_and_weights) if datasets_and_weights else
            ([datasets[0].take(0)], [1.]))

  def _empty_datasets_with_zero_weight(datasets, positive):
    # Weights only known at runtime can't drop datasets up front, so make
    # each dataset whose weight isn't positive empty instead, as list weights
    # would drop it. `take(-1)` keeps every element and `take(0)` none.
    take_counts = array_ops_stack.unstack(
        -math_ops.cast(positive, dtypes.int64), num=len(datasets))
    return [
        dataset.take(count) for dataset, count in zip(datasets, take_counts)
    ]

  def _logits_with_zero_weight(weights, positive, stop_on_empty_dataset):
    # Logits for the datasets above, in float64 so that the ones for empty
    # datasets neither underflow nor overflow. The multinomial kernel
    # computes in float64 anyway, so positive weights are sampled exactly as
    # before. Non-positive weights are replaced first, so that no logit is
    # infinite or NaN.
    def float64(value):
      return constant_op.constant(value, dtype=dtypes.float64)

    one = constant_op.constant(1, dtype=weights.dtype)
    logits = math_ops.cast(
        math_ops.log(array_ops.where_v2(positive, weights, one)),
        dtypes.float64)
    weights = math_ops.cast(weights, dtypes.float64)
    total = math_ops.reduce_sum(
        array_ops.where_v2(positive, weights, float64(0)))
    has_positive = math_ops.reduce_any(positive)
    if stop_on_empty_dataset:
      # Selecting an empty dataset would end sampling, so keep them
      # unselectable.
      fill = float64(-np.inf)
    else:
      # An unselectable dataset is never found empty, so sampling wouldn't
      # end once the others are exhausted. Give the empty datasets a tenth of
      # the draws instead: selecting one only skips it, so the others keep
      # their relative probabilities.
      num_empty = math_ops.reduce_sum(
          math_ops.cast(math_ops.logical_not(positive), dtypes.float64))
      fill = math_ops.log(
          array_ops.where_v2(has_positive, total, float64(1)) /
          (float64(9) * math_ops.maximum(num_empty, float64(1))))
    # With no positive weight every dataset is empty, so let any be selected.
    fill = array_ops.where_v2(has_positive, fill, float64(0))
    return array_ops.where_v2(positive, logits, fill)

  if not datasets:
    raise ValueError("Invalid `datasets`. `datasets` should not be empty.")

  if not isinstance(weights, data_types.DatasetV2):
    if weights is None:
      # Select inputs with uniform probability.
      logits = [[1.0] * len(datasets)]

    else:
      if isinstance(weights, tensor.Tensor):
        if not weights.shape.is_compatible_with([len(datasets)]):
          raise ValueError(f"Invalid `weights`. The shape of `weights` "
                           f"should be compatible with `[len(datasets)]` "
                           f"but is {weights.shape}.")
      else:
        if len(datasets) != len(weights):
          raise ValueError(f"Invalid `weights`. `weights` should have the "
                           f"same length as `datasets` but got "
                           f"`len(weights)={len(weights)}` vs. "
                           f"`len(datasets)={len(datasets)}`.")

      # Use the given `weights` as the probability of choosing the respective
      # input.
      # A list of bfloat16 scalars can't be converted to a tensor, but an
      # array of them can. A list with tensors in it is converted as is.
      if isinstance(weights, (list, tuple)) and not any(
          isinstance(weight, tensor.Tensor) for weight in weights):
        weights = np.asarray(weights)
      weights = ops.convert_to_tensor(weights, name="weights")
      if weights.dtype not in (dtypes.float16, dtypes.bfloat16, dtypes.float32,
                               dtypes.float64):
        raise TypeError(f"Invalid `weights`. `weights` type must be "
                        f"`tf.float16`, `tf.bfloat16`, `tf.float32` or "
                        f"`tf.float64` but is {weights.dtype}.")
      weights_value = tensor_util.constant_value(weights)
      if weights_value is not None and not (weights_value > 0).all():
        datasets, weights = _skip_datasets_with_zero_weight(
            datasets, weights_value)
        weights = ops.convert_to_tensor(np.asarray(weights), name="weights")

      if weights_value is None:
        positive = weights > 0
        datasets = _empty_datasets_with_zero_weight(datasets, positive)
      # Return before building the logits, which a single dataset doesn't
      # need.
      if len(datasets) == 1:
        return datasets[0]
      # The `stateless_multinomial()` op expects log-probabilities, as opposed
      # to weights.
      if weights_value is None:
        logits = _logits_with_zero_weight(weights, positive,
                                          stop_on_empty_dataset)
      else:
        logits = math_ops.log(weights, name="logits")
      logits = array_ops.expand_dims(logits, 0)

    # NOTE(mrry): We only specialize when `weights` is not a `Dataset`. When
    # it is a `Dataset`, it is possible that evaluating it has a side effect
    # the user depends on.
    if len(datasets) == 1:
      return datasets[0]

    def select_dataset_constant_logits(seed):
      return array_ops.squeeze(
          gen_stateless_random_ops.stateless_multinomial(
              logits, 1, seed=seed),
          axis=[0, 1])

    selector_input = map_op._MapDataset(  # pylint: disable=protected-access
        dataset_ops.Dataset.random(
            seed=seed,
            rerandomize_each_iteration=rerandomize_each_iteration).batch(2),
        select_dataset_constant_logits,
        use_inter_op_parallelism=False)

  else:  # isinstance(weights, DatasetV2)
    # Use each element of the given `weights` dataset as the probability of
    # choosing the respective input.
    #
    # The `stateless_multinomial()` op expects log-probabilities, as opposed
    # to weights.
    logits_ds = weights.map(lambda *p: math_ops.log(p, name="logits"))

    def select_dataset_varying_logits(logits, seed):
      return array_ops.squeeze(
          gen_stateless_random_ops.stateless_multinomial(
              logits, 1, seed=seed),
          axis=[0, 1])

    logits_and_seeds = dataset_ops.Dataset.zip(
        (logits_ds,
         dataset_ops.Dataset.random(
             seed=seed,
             rerandomize_each_iteration=rerandomize_each_iteration).batch(2)))
    selector_input = map_op._MapDataset(  # pylint: disable=protected-access
        logits_and_seeds,
        select_dataset_varying_logits,
        use_inter_op_parallelism=False)

  return directed_interleave_op._directed_interleave(  # pylint: disable=protected-access
      selector_input, datasets, stop_on_empty_dataset
  )
