# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
import itertools
import math

from absl.testing import parameterized
import numpy as np

from tensorflow.python.compat import v2_compat
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test
from tensorflow.python.tpu.ops import gen_xla_ops as xla_ops


def _get_combiner_scale_contribution(x, combiner):
  if combiner == "sum":
    return 1.0
  elif combiner == "mean":
    return x
  else:
    return x * x


def _get_combiner_scale_transform(x, combiner):
  if combiner == "sum":
    return 1.0
  elif combiner == "mean":
    return 0 if x == 0 else (1 / x)
  else:
    return 0 if x == 0 else (1 / math_ops.sqrt(x))


def convert_input_to_coo_tensor(
    indices_or_row_splits, values, weight, sample_count, combiner
):
  if indices_or_row_splits.shape.rank >= 2:
    row_ids_before_dedup = indices_or_row_splits[:, 0]
  elif indices_or_row_splits.shape.rank == 1:
    row_ids_before_dedup = []
    current_row_id = -1
    for i, _ in enumerate(values):
      while i == indices_or_row_splits[current_row_id + 1]:
        current_row_id += 1
      row_ids_before_dedup.append(current_row_id)
  else:
    row_ids_before_dedup = math_ops.range(0, values.shape[0])

  col_ids = []
  row_ids = []
  gains = []
  gains_rescale = [0] * sample_count
  for i in range(values.shape[0]):
    gains_rescale[row_ids_before_dedup[i]] += _get_combiner_scale_contribution(
        weight[i], combiner
    )
    if (
        row_ids
        and row_ids[-1] == row_ids_before_dedup[i]
        and col_ids[-1] == values[i]
    ):
      gains[-1] += weight[i]
    else:
      row_ids.append(row_ids_before_dedup[i])
      col_ids.append(values[i])
      gains.append(weight[i])

  for i in range(sample_count):
    gains_rescale[i] = _get_combiner_scale_transform(gains_rescale[i], combiner)
  for i in range(len(row_ids)):
    gains[i] *= gains_rescale[row_ids[i]]

  row_ids = ops.convert_to_tensor(row_ids, dtype=dtypes.int32)
  col_ids = ops.convert_to_tensor(col_ids, dtype=dtypes.int32)
  gains = ops.convert_to_tensor(gains, dtype=dtypes.float32)

  return row_ids, col_ids, gains


class TpuEmbeddingV3CPUOpsTest(parameterized.TestCase, test.TestCase):

  @parameterized.parameters(
      *list(
          itertools.product(
              [16, 32],
              [1024, 2048],
              ["sum", "mean", "sqrtn"],
              [0, 56, 1600000],
              [0, 12, 20],
          )
      )
  )
  def test_convert_to_list_of_sparse_core_coo_tensor(
      self, sample_count, token_count, combiner, col_offset, col_shift
  ):
    sparse_feature = sparse_ops.sparse_reorder(
        sparse_tensor.SparseTensor(
            indices=[
                [i % sample_count, i]
                for i in np.random.randint(low=0, high=1024, size=token_count)
            ],
            values=np.random.randint(low=0, high=1024, size=token_count),
            dense_shape=[sample_count, 1024],
        )
    )

    num_sc_per_chip = 4
    num_chip = 128

    row_offset = sample_count
    num_sc_shards = num_sc_per_chip * num_chip
    stacked_table_sample_count = sample_count * 4

    num_sc_shards_bit = int(math.log2(num_sc_shards))
    num_sc_shards_bit_mod = (1 << num_sc_shards_bit) - 1
    num_sc_shards_bit_mod_inv = bitwise_ops.invert(num_sc_shards_bit_mod)

    row_ids, col_ids, gains = convert_input_to_coo_tensor(
        indices_or_row_splits=sparse_feature.indices,
        values=sparse_feature.values,
        weight=np.ones(shape=token_count),
        sample_count=sample_count,
        combiner=combiner,
    )

    golden_row_ids = (
        row_ids % (sample_count // num_sc_per_chip)
        + int(row_offset // num_sc_per_chip)
        + int(stacked_table_sample_count // num_sc_per_chip)
        * (row_ids // (sample_count // num_sc_per_chip))
    )
    golden_col_ids = (
        bitwise_ops.bitwise_and(col_ids + col_shift, num_sc_shards_bit_mod)
        + bitwise_ops.bitwise_and(col_ids, num_sc_shards_bit_mod_inv)
        + col_offset
    )

    row_ids_list, col_ids_list, gains_list = (
        xla_ops.convert_to_list_of_sparse_core_coo_tensors(
            indices_or_row_splits=math_ops.cast(
                sparse_feature.indices, dtype=dtypes.int32
            ),
            values=math_ops.cast(sparse_feature.values, dtype=dtypes.int32),
            weights=1.0,
            sample_count=sample_count,
            combiner=combiner,
            num_sc_per_chip=4,
            row_offset=row_offset,
            col_offset=col_offset,
            col_shift=col_shift,
            num_sc_shards=num_sc_shards,
            stacked_table_sample_count=stacked_table_sample_count,
        )
    )

    self.assertAllClose(golden_row_ids, array_ops.concat(row_ids_list, axis=0))
    self.assertAllClose(golden_col_ids, array_ops.concat(col_ids_list, axis=0))
    self.assertAllClose(gains, array_ops.concat(gains_list, axis=0))


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  test.main()
