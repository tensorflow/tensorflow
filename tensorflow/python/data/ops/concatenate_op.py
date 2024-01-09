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
"""The implementation of `tf.data.Dataset.concatenate`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.util import nest as tf_nest


def _concatenate(input_dataset, dataset_to_concatenate, name):  # pylint: disable=unused-private-name
  return _ConcatenateDataset(input_dataset, dataset_to_concatenate, name)


class _ConcatenateDataset(dataset_ops.DatasetV2):
  """A `Dataset` that concatenates its input with given dataset."""

  def __init__(self, input_dataset, dataset_to_concatenate, name=None):
    """See `Dataset.concatenate()` for details."""
    self._input_dataset = input_dataset
    self._dataset_to_concatenate = dataset_to_concatenate

    def common_supertype(a, b):
      result = a.most_specific_common_supertype([b])
      if result is None:
        raise TypeError(f"No common supertype of {a} and {b}.")
      return result

    try:
      self._structure = tf_nest.map_structure(
          common_supertype, input_dataset.element_spec,
          dataset_to_concatenate.element_spec)
    except (TypeError, ValueError) as e:
      raise TypeError(f"Incompatible dataset elements:\n"
                      f"  {input_dataset.element_spec} vs. "
                      f"  {dataset_to_concatenate.element_spec}") from e

    self._input_datasets = [input_dataset, dataset_to_concatenate]
    self._name = name
    # pylint: disable=protected-access
    variant_tensor = gen_dataset_ops.concatenate_dataset(
        input_dataset._variant_tensor, dataset_to_concatenate._variant_tensor,
        **self._common_args)
    # pylint: enable=protected-access
    super().__init__(variant_tensor)

  def _inputs(self):
    return self._input_datasets

  @property
  def element_spec(self):
    return self._structure
