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

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import sparse
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.util import nest as tf_nest


class _PrependFromQueueAndPaddedBatchDataset(dataset_ops.Dataset):
  """A `Dataset` that prepends a queue to another `Dataset`.

  A vector of handles to the queue is returned as the first component of
  the associated iterator.  This vector can be passed to
  `enqueue_in_queue_dataset` to add new elements to the queue.
  """

  def __init__(self, input_dataset, batch_size, padded_shapes, padding_values):
    """Initialize `PrependFromQueueAndPaddedBatchDataset`."""
    super(_PrependFromQueueAndPaddedBatchDataset, self).__init__()
    if sparse.any_sparse(input_dataset.output_classes):
      raise TypeError(
          "Batching of padded sparse tensors is not currently supported")
    self._input_dataset = input_dataset
    self._batch_size = ops.convert_to_tensor(
        batch_size, dtype=dtypes.int64, name="batch_size")
    # pylint: disable=protected-access
    if padded_shapes is None:
      self._padded_shapes = nest.map_structure(
          dataset_ops._partial_shape_to_tensor, input_dataset.output_shapes)
    else:
      self._padded_shapes = nest.map_structure_up_to(
          input_dataset.output_shapes, dataset_ops._partial_shape_to_tensor,
          padded_shapes)
    padding_values = (
        padding_values if padding_values is not None else
        dataset_ops._default_padding(input_dataset))
    self._padding_values = nest.map_structure_up_to(
        input_dataset.output_shapes, dataset_ops._padding_value_to_tensor,
        padding_values, input_dataset.output_types)
    # pylint: enable=protected-access

  def _as_variant_tensor(self):
    # pylint: disable=protected-access
    return gen_dataset_ops.prepend_from_queue_and_padded_batch_dataset(
        self._input_dataset._as_variant_tensor(),
        batch_size=self._batch_size,
        padded_shapes=[
            ops.convert_to_tensor(s, dtype=dtypes.int64)
            for s in nest.flatten(self._padded_shapes)
        ],
        padding_values=nest.flatten(self._padding_values),
        output_shapes=nest.flatten(
            sparse.as_dense_shapes(self.output_shapes, self.output_classes)))
    # pylint: enable=protected-access

  @property
  def output_classes(self):
    return (ops.Tensor, self._input_dataset.output_classes)

  def _as_batch_shape(self, shape_like):
    return tensor_shape.vector(None).concatenate(
        tensor_util.constant_value_as_shape(shape_like))

  @property
  def output_shapes(self):
    # First output is a variant representing the Queue
    return (tensor_shape.vector(None),
            nest.map_structure(self._as_batch_shape, self._padded_shapes))

  @property
  def output_types(self):
    # First output is a variant representing the Queue
    return (dtypes.variant, self._input_dataset.output_types)


def prepend_from_queue_and_padded_batch_dataset(batch_size,
                                                padding_values=None,
                                                padded_shapes=None):
  """A transformation that prepends a queue to a `Dataset` and batches results.

  A vector of handles to the queue is returned as the first component of the
  associated iterator.  This vector can be passed to `enqueue_in_queue_dataset`
  to add new elements to the queue.

  Below is an example of how this dataset might be used to split incoming
  variable-length sequences into "head" and "rest" parts, where "rest" parts
  are re-enqueued back into the dataset.  A more realistic example would
  perform some calculation on the "head" and modify some components of "rest"
  with the result (before re-enqueueing).

  ```python
  dataset = tf.data.Dataset.from_tensor_slices([2*x for x in range(10)])
  # Make a dataset of variable-length vectors and their lengths.
  dataset = dataset.map(lambda count: (count, tf.ones((count,))))
  # Emit a queue we can prepend to, and counts/values as padded batch.
  dataset = dataset.apply(
      tf.contrib.training.prepend_from_queue_and_padded_batch_dataset(
        batch_size=10))
  dataset = dataset.prefetch(1)

  iterator = dataset.make_one_shot_iterator()
  queue, (count, padded_value) = iterator.get_next()

  # Split the padded_value into two pieces: head and rest
  rest_indices = tf.squeeze(tf.where(count > 3), axis=1)
  bound = tf.minimum(3, tf.reduce_max(count))
  value_head = padded_value[:, :bound]
  count_rest = tf.gather(count - 3, rest_indices)
  value_rest = tf.gather(padded_value[:, bound:], rest_indices)
  queue_rest = tf.gather(queue, rest_indices)
  enqueue_rest_op = tf.contrib.training.enqueue_in_queue_dataset(
    queue_rest, (count_rest, value_rest))
  with tf.control_dependencies([enqueue_rest_op]):
    calculation = fn(value_head)

  while True:  # Will raise OutOfRange when finished with all pieces.
    session.run(calculation)
  ```

  Args:
    batch_size: `int64` scalar tensor.  The batch size to use when performing
      padded batching.
    padding_values: (optional) Nested tuple of scalar tensors.  If provided,
      the structure and dtypes of padding_values should match that of
      incoming dataset's `output_types`.
    padded_shapes: (optional) Nested tuple of `int64` vector tensors.
      If provided, the structure must match that of the incoming dataset's
      `output_types`.  If not provided, the incoming dataset's `output_shapes`
      is used.  Any unknown (`None` or `-1`) dimensions in the shapes are
      treated as being unique per-batch: for each batch time, an unknown
      dimension is replaced with the maximum given value of this dimension
      across all tensors for the given component in the batch.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.data.Dataset.apply}.
  """

  def _apply_fn(dataset):
    return _PrependFromQueueAndPaddedBatchDataset(
        dataset,
        batch_size=batch_size,
        padding_values=padding_values,
        padded_shapes=padded_shapes)

  return _apply_fn


def enqueue_in_queue_dataset(queue, components):
  """Enqueue components into queue from `PrependFromQueueAndPaddedBatchDataset`.

  The components' dtypes and shapes must be compatible with the `output_shapes`
  attribute of the `dataset` created by
  `prepend_from_queue_and_padded_batch_dataset`.  This operation supports both
  non-batched and batched modes.

  For more details, see the example in the docstring for
  `prepend_from_queue_and_padded_batch_dataset`.

  Args:
    queue: `variant` scalar or vector tensor.
      The tensor emitted by the first component of the iterator associated with
      `prepend_from_queue_and_padded_batch_dataset`.  If this is a scalar,
      then the `components` input tensors should not have a prepended batch
      dimension.
    components: Nested tuple of tensors, each with a leading batch dimension
      if `queue` is a vector.  The structure, dtypes, and shapes
      (excluding batch dimension) must match the nested tuples
      `dataset.output_types[1]` and `dataset.output_shapes[1]` (the non-queue
      output types and shapes) of the `dataset` emitted by
      the original `prepend_from_queue_and_padded_batch_dataset` call.

  Returns:
    An `Operation` that enqueues `components` into the dataset(s) associated
    with entries of `queue`.
  """
  return gen_dataset_ops.enqueue_in_queue_dataset(
      queue=queue, components=tf_nest.flatten(components))
