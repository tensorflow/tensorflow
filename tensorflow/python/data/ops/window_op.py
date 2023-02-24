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
"""The implementation of `tf.data.Dataset.window`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops


def _window(input_dataset, size, shift, stride, drop_remainder, name):
  if shift is None:
    shift = size
  return _WindowDataset(
      input_dataset, size, shift, stride, drop_remainder, name=name)


class _WindowDataset(dataset_ops.UnaryDataset):
  """A dataset that creates window datasets from the input elements."""

  def __init__(self,
               input_dataset,
               size,
               shift,
               stride,
               drop_remainder,
               name=None):
    """See `window()` for more details."""
    self._input_dataset = input_dataset
    self._size = ops.convert_to_tensor(size, dtype=dtypes.int64, name="size")
    self._shift = ops.convert_to_tensor(shift, dtype=dtypes.int64, name="shift")
    self._stride = ops.convert_to_tensor(
        stride, dtype=dtypes.int64, name="stride")
    self._drop_remainder = ops.convert_to_tensor(
        drop_remainder, dtype=dtypes.bool, name="drop_remainder")
    self._structure = nest.pack_sequence_as(
        dataset_ops.get_legacy_output_classes(input_dataset),
        [
            dataset_ops.DatasetSpec(  # pylint: disable=g-complex-comprehension
                structure.convert_legacy_structure(output_type, output_shape,
                                                   output_class))
            for output_class, output_shape, output_type in zip(
                nest.flatten(
                    dataset_ops.get_legacy_output_classes(input_dataset)),
                nest.flatten(
                    dataset_ops.get_legacy_output_shapes(input_dataset)),
                nest.flatten(
                    dataset_ops.get_legacy_output_types(input_dataset)))
        ])
    self._name = name
    variant_tensor = gen_dataset_ops.window_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        size=self._size,
        shift=self._shift,
        stride=self._stride,
        drop_remainder=self._drop_remainder,
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._structure
