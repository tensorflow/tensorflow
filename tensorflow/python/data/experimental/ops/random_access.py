# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Python API for random indexing into a dataset."""

from tensorflow.python.data.util import structure
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.at", v1=[])
def at(dataset, index):
  """Returns the element at a specific index in a datasest.

  Currently, random access is supported for the following tf.data operations:

     - `tf.data.Dataset.from_tensor_slices`,
     - `tf.data.Dataset.from_tensors`,
     - `tf.data.Dataset.shuffle`,
     - `tf.data.Dataset.batch`,
     - `tf.data.Dataset.shard`,
     - `tf.data.Dataset.map`,
     - `tf.data.Dataset.range`,
     - `tf.data.Dataset.zip`,
     - `tf.data.Dataset.skip`,
     - `tf.data.Dataset.repeat`,
     - `tf.data.Dataset.list_files`,
     - `tf.data.Dataset.SSTableDataset`,
     - `tf.data.Dataset.concatenate`,
     - `tf.data.Dataset.enumerate`,
     - `tf.data.Dataset.parallel_map`,
     - `tf.data.Dataset.prefetch`,
     - `tf.data.Dataset.take`,
     - `tf.data.Dataset.cache` (in-memory only)

     Users can use the cache operation to enable random access for any dataset,
     even one comprised of transformations which are not on this list.
     E.g., to get the third element of a TFDS dataset:

       ```python
       ds = tfds.load("mnist", split="train").cache()
       elem = tf.data.Dataset.experimental.at(ds, 3)
       ```

  Args:
    dataset: A `tf.data.Dataset` to determine whether it supports random access.
    index: The index at which to fetch the element.

  Returns:
      A (nested) structure of values matching `tf.data.Dataset.element_spec`.

   Raises:
     UnimplementedError: If random access is not yet supported for a dataset.
  """
  # pylint: disable=protected-access
  return structure.from_tensor_list(
      dataset.element_spec,
      gen_experimental_dataset_ops.get_element_at_index(
          dataset._variant_tensor,
          index,
          output_types=structure.get_flat_tensor_types(dataset.element_spec),
          output_shapes=structure.get_flat_tensor_shapes(dataset.element_spec)))
