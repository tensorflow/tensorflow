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
"""The implementation of `tf.data.Dataset.zip`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.types import data as data_types


def _zip(datasets, name):  # pylint: disable=redefined-builtin
  return _ZipDataset(datasets, name)


class _ZipDataset(dataset_ops.DatasetV2):
  """A `Dataset` that zips its inputs together."""

  def __init__(self, datasets, name=None):
    """See `Dataset.zip()` for details."""
    for ds in nest.flatten(datasets):
      if not isinstance(ds, data_types.DatasetV2):
        if isinstance(ds, list):
          raise TypeError("Invalid `datasets`. `datasets` is expected to be a "
                          "(nested) structure of `tf.data.Dataset` objects. "
                          "Python `list` is not supported and you should use "
                          "`tuple` instead.")
        else:
          raise TypeError(f"Invalid `datasets`. `datasets` is expected to be a "
                          f"(nested) structure of `tf.data.Dataset` objects "
                          f"but encountered object of type {type(ds)}.")
    self._datasets = datasets
    self._structure = nest.pack_sequence_as(
        self._datasets,
        [ds.element_spec for ds in nest.flatten(self._datasets)])
    self._name = name
    variant_tensor = gen_dataset_ops.zip_dataset(
        [ds._variant_tensor for ds in nest.flatten(self._datasets)],
        **self._common_args)
    super().__init__(variant_tensor)

  def _inputs(self):
    return nest.flatten(self._datasets)

  @property
  def element_spec(self):
    return self._structure
