# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Python wrappers for Datasets and Iterators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.data.python.ops import grouping
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import sparse
from tensorflow.python.ops import gen_dataset_ops


def get_single_element(dataset):
  """Returns the single element in `dataset` as a nested structure of tensors.

  This function enables you to use a `tf.data.Dataset` in a stateless
  "tensor-in tensor-out" expression, without creating a `tf.data.Iterator`.
  This can be useful when your preprocessing transformations are expressed
  as a `Dataset`, and you want to use the transformation at serving time.
  For example:

  ```python
  input_batch = tf.placeholder(tf.string, shape=[BATCH_SIZE])

  def preprocessing_fn(input_str):
    # ...
    return image, label

  dataset = (tf.data.Dataset.from_tensor_slices(input_batch)
             .map(preprocessing_fn, num_parallel_calls=BATCH_SIZE)
             .batch(BATCH_SIZE))

  image_batch, label_batch = tf.contrib.data.get_single_element(dataset)
  ```

  Args:
    dataset: A `tf.data.Dataset` object containing a single element.

  Returns:
    A nested structure of `tf.Tensor` objects, corresponding to the single
    element of `dataset`.

  Raises:
    TypeError: if `dataset` is not a `tf.data.Dataset` object.
    InvalidArgumentError (at runtime): if `dataset` does not contain exactly
      one element.
  """
  if not isinstance(dataset, dataset_ops.Dataset):
    raise TypeError("`dataset` must be a `tf.data.Dataset` object.")

  nested_ret = nest.pack_sequence_as(
      dataset.output_types, gen_dataset_ops.dataset_to_single_element(
          dataset._as_variant_tensor(),  # pylint: disable=protected-access
          **dataset_ops.flat_structure(dataset)))
  return sparse.deserialize_sparse_tensors(
      nested_ret, dataset.output_types, dataset.output_shapes,
      dataset.output_classes)


def reduce_dataset(dataset, reducer):
  """Returns the result of reducing the `dataset` using `reducer`.

  Args:
    dataset: A `tf.data.Dataset` object.
    reducer: A `tf.contrib.data.Reducer` object representing the reduce logic.

  Returns:
    A nested structure of `tf.Tensor` objects, corresponding to the result
    of reducing `dataset` using `reducer`.

  Raises:
    TypeError: if `dataset` is not a `tf.data.Dataset` object.
  """
  if not isinstance(dataset, dataset_ops.Dataset):
    raise TypeError("`dataset` must be a `tf.data.Dataset` object.")

  # The sentinel dataset is used in case the reduced dataset is empty.
  sentinel_dataset = dataset_ops.Dataset.from_tensors(
      reducer.finalize_func(reducer.init_func(np.int64(0))))
  reduced_dataset = dataset.apply(
      grouping.group_by_reducer(lambda x: np.int64(0), reducer))

  return get_single_element(
      reduced_dataset.concatenate(sentinel_dataset).take(1))
