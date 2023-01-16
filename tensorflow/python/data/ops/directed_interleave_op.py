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
"""The implementation of `tf.data.Dataset.shuffle`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops


def _directed_interleave(  # pylint: disable=unused-private-name
    selector_input, data_inputs, stop_on_empty_dataset=False
):
  return _DirectedInterleaveDataset(
      selector_input, data_inputs, stop_on_empty_dataset=stop_on_empty_dataset
  )


class _DirectedInterleaveDataset(dataset_ops.DatasetV2):
  """A substitute for `Dataset.interleave()` on a fixed list of datasets."""

  def __init__(self, selector_input, data_inputs, stop_on_empty_dataset=False):
    self._selector_input = selector_input
    self._data_inputs = list(data_inputs)
    self._stop_on_empty_dataset = stop_on_empty_dataset

    spec = self._data_inputs[0].element_spec
    for i, data_input in enumerate(self._data_inputs[1:]):
      def common_supertype(a, b):
        result = a.most_specific_common_supertype([b])
        if result is None:
          raise TypeError(f"No common supertype of {a} and {b}.")
        return result

      try:
        spec = nest.map_structure(common_supertype, spec,
                                  data_input.element_spec)
      except (TypeError, ValueError) as e:
        raise TypeError(f"Invalid `datasets`. `datasets` must have compatible "
                        f"element specs.\n Dataset 0 "
                        f"element_spec={data_inputs[0].element_spec}.\n"
                        f"Dataset {i+1} "
                        f"element_spec={data_input.element_spec}.") from e
    self._element_spec = spec

    # pylint: disable=protected-access
    variant_tensor = (
        ged_ops.directed_interleave_dataset(
            self._selector_input._variant_tensor,
            [data_input._variant_tensor for data_input in self._data_inputs],
            stop_on_empty_dataset=self._stop_on_empty_dataset,
            **self._flat_structure))

    super().__init__(variant_tensor)

  def _inputs(self):
    return [self._selector_input] + self._data_inputs

  @property
  def element_spec(self):
    return self._element_spec
