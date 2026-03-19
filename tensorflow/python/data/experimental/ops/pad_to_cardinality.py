# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""The implementation of `tf.data.experimental.pad_to_cardinality`."""

from collections.abc import Mapping

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.pad_to_cardinality")
def pad_to_cardinality(cardinality, mask_key="valid"):
  """Pads a dataset with fake elements to reach the desired cardinality.

  The dataset to pad must have a known and finite cardinality and contain
  dictionary elements. The `mask_key` will be added to differentiate between
  real and padding elements -- real elements will have a `<mask_key>=True` entry
  while padding elements will have a `<mask_key>=False` entry.

  Example usage:

  ds = tf.data.Dataset.from_tensor_slices({'a': [1, 2]})
  ds = ds.apply(tf.data.experimental.pad_to_cardinality(3))
  list(ds.as_numpy_iterator())
  [{'a': 1, 'valid': True}, {'a': 2, 'valid': True}, {'a': 0, 'valid': False}]

  This can be useful, e.g. during eval, when partial batches are undesirable but
  it is also important not to drop any data.

  ```
  ds = ...
  # Round up to the next full batch.
  target_cardinality = -(-ds.cardinality() // batch_size) * batch_size
  ds = ds.apply(tf.data.experimental.pad_to_cardinality(target_cardinality))
  # Set `drop_remainder` so that batch shape will be known statically. No data
  # will actually be dropped since the batch size divides the cardinality.
  ds = ds.batch(batch_size, drop_remainder=True)
  ```

  Args:
    cardinality: The cardinality to pad the dataset to.
    mask_key: The key to use for identifying real vs padding elements.

  Returns:
    A dataset transformation that can be applied via `Dataset.apply()`.
  """

  def make_filler_dataset(ds):
    padding = cardinality - ds.cardinality()

    filler_element = nest.map_structure(
        lambda spec: array_ops.zeros(spec.shape, spec.dtype), ds.element_spec
    )
    filler_element[mask_key] = False
    filler_dataset = dataset_ops.Dataset.from_tensors(filler_element)
    filler_dataset = filler_dataset.repeat(padding)
    return filler_dataset

  def apply_valid_mask(x):
    x[mask_key] = True
    return x

  def _apply_fn(dataset):
    # The cardinality tensor is unknown during tracing, so we only check it
    # in eager mode.
    if context.executing_eagerly():
      if dataset.cardinality() < 0:
        raise ValueError(
            "The dataset passed into `pad_to_cardinality` must "
            "have a known cardinalty, but has cardinality "
            f"{dataset.cardinality()}"
        )
      if dataset.cardinality() > cardinality:
        raise ValueError(
            "The dataset passed into `pad_to_cardinality` must "
            "have a cardinalty less than the target cardinality "
            f"({cardinality}), but has cardinality "
            f"{dataset.cardinality()}"
        )
    if not isinstance(dataset.element_spec, Mapping):
      raise ValueError(
          "`pad_to_cardinality` requires its input dataset to "
          "be a dictionary."
      )
    filler = make_filler_dataset(dataset)
    dataset = dataset.map(apply_valid_mask)
    dataset = dataset.concatenate(filler)
    return dataset

  return _apply_fn
