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

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import directed_interleave_op
from tensorflow.python.data.ops import map_op
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
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

  if not datasets:
    raise ValueError("Invalid `datasets`. `datasets` should not be empty.")

  zero_indices_ds = None

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

      # Filter zero-weight datasets before computing logits.
      # - For static weights (including a `tf.constant`), resolve the values
      #   statically via `tensor_util.constant_value` and remove the
      #   zero-weight datasets, so -inf logits never reach the selector.
      # - For dynamic tensors (e.g. `tf.function` arguments) the values are
      #   unknown here. A zero-weight dataset is never sampled, so its
      #   iterator would never reach EOF and `directed_interleave` would spin
      #   forever. Instead, map zero-weight datasets to empty datasets and
      #   have the selector pick each of them once up front so they are
      #   marked exhausted.
      weights_to_filter = weights
      if isinstance(weights, tensor.Tensor):
        static_weights = tensor_util.constant_value(weights)
        if static_weights is not None:
          weights_to_filter = static_weights.tolist()
        else:
          weights_to_filter = None

      if weights_to_filter is not None:
        datasets, weights = _skip_datasets_with_zero_weight(
            datasets, weights_to_filter)
      elif not stop_on_empty_dataset:
        # With `stop_on_empty_dataset=True` an unsampled zero-weight dataset
        # cannot cause a hang, and selecting an empty dataset up front would
        # end the iteration immediately.
        is_zero = math_ops.equal(weights, math_ops.cast(0, weights.dtype))
        zero_indices = math_ops.cast(
            array_ops.squeeze(array_ops.where_v2(is_zero), axis=1),
            dtypes.int64)
        zero_indices_ds = dataset_ops.Dataset.from_tensor_slices(zero_indices)
        datasets = list(datasets)
        for i in range(len(datasets)):
          datasets[i] = datasets[i].take(
              array_ops.where_v2(
                  is_zero[i],
                  constant_op.constant(0, dtype=dtypes.int64),
                  constant_op.constant(-1, dtype=dtypes.int64)))

      weights = ops.convert_to_tensor(weights, name="weights")
      if weights.dtype not in (dtypes.float32, dtypes.float64):
        raise TypeError(f"Invalid `weights`. `weights` type must be either "
                        f"`tf.float32` or `tf.float64` but is "
                        f"{weights.dtype}.")

      # The `stateless_multinomial()` op expects log-probabilities, as opposed
      # to weights.
      logits = array_ops.expand_dims(math_ops.log(weights, name="logits"), 0)

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

  if zero_indices_ds is not None:
    selector_input = zero_indices_ds.concatenate(selector_input)

  return directed_interleave_op._directed_interleave(  # pylint: disable=protected-access
      selector_input, datasets, stop_on_empty_dataset
  )
