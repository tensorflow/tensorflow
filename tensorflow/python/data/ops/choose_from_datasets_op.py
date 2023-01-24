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
"""The implementation of `tf.data.Dataset.choose_from_datasets`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import directed_interleave_op
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec


def _choose_from_datasets(  # pylint: disable=unused-private-name
    datasets, choice_dataset, stop_on_empty_dataset=True
):
  """See `Dataset.choose_from_datasets()` for details."""

  if not datasets:
    raise ValueError("Invalid `datasets`. `datasets` should not be empty.")
  if not isinstance(choice_dataset, dataset_ops.DatasetV2):
    raise TypeError(
        "Invalid `choice_dataset`. `choice_dataset` should be a "
        f"`tf.data.Dataset` but is {type(choice_dataset)}."
    )
  if not structure.are_compatible(
      choice_dataset.element_spec, tensor_spec.TensorSpec([], dtypes.int64)
  ):
    raise TypeError(
        "Invalid `choice_dataset`. Elements of `choice_dataset` "
        "must be scalar `tf.int64` tensors but are "
        f"{choice_dataset.element_spec}."
    )
  # Replicates the `choice_dataset` component so that each split makes choices
  # independently. This avoids the need for prohibitively expensive
  # cross-split coordination.
  # pylint: disable=protected-access
  choice_dataset = dataset_ops._apply_rewrite(
      choice_dataset, "replicate_on_split"
  )
  return directed_interleave_op._directed_interleave(  # pylint: disable=protected-access
      choice_dataset, datasets, stop_on_empty_dataset
  )
