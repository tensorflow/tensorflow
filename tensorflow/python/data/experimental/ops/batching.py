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
"""Enhanced batching dataset transformations for TensorFlow."""

import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import convert
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util  # Added missing import
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util.tf_export import tf_export
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tf_export("data.experimental.dense_to_ragged_batch")
def dense_to_ragged_batch(batch_size,
                          drop_remainder=False,
                          row_splits_dtype=dtypes.int64):
    """Batches elements into `tf.RaggedTensor`s, handling ragged dimensions.

    This transformation combines multiple consecutive elements of the input
    dataset into a single element, where elements may have different shapes.
    It is similar to `tf.data.Dataset.batch`, but allows for ragged dimensions.

    Args:
        batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
            consecutive elements of this dataset to combine in a single batch.
        drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
            whether the last batch should be dropped if it has fewer than
            `batch_size` elements. Defaults to `False`.
        row_splits_dtype: The dtype to use for the `row_splits` of any new
            `tf.RaggedTensor`s. Defaults to `tf.int64`.

    Returns:
        A `Dataset` transformation function, which can be passed to
        `tf.data.Dataset.apply`.

    Example:
    ```python
    dataset = tf.data.Dataset.from_tensor_slices([tf.range(i) for i in range(5)])
    dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(2))
    for batch in dataset:
        print(batch)
    ```
    """
    def _apply_fn(dataset):
        logger.info(f"Applying dense_to_ragged_batch with batch_size={batch_size}, "
                    f"drop_remainder={drop_remainder}, row_splits_dtype={row_splits_dtype}")
        return dataset.ragged_batch(batch_size, drop_remainder, row_splits_dtype)

    return _apply_fn


@tf_export("data.experimental.dense_to_sparse_batch")
def dense_to_sparse_batch(batch_size, row_shape):
    """Batches elements into `tf.sparse.SparseTensor`s, handling varying shapes.

    This transformation combines multiple consecutive elements of the dataset,
    which might have different shapes, into a single `tf.sparse.SparseTensor`.

    Args:
        batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
            consecutive elements of this dataset to combine in a single batch.
        row_shape: A `tf.TensorShape` or `tf.int64` vector tensor-like object
            representing the dense shape of each row in the resulting
            `tf.sparse.SparseTensor`.

    Returns:
        A `Dataset` transformation function, which can be passed to
        `tf.data.Dataset.apply`.

    Example:
    ```python
    dataset = tf.data.Dataset.from_tensor_slices([tf.range(i) for i in range(1, 5)])
    dataset = dataset.apply(tf.data.experimental.dense_to_sparse_batch(2, [5]))
    for sparse_tensor in dataset:
        print(tf.sparse.to_dense(sparse_tensor))
    ```
    """
    def _apply_fn(dataset):
        logger.info(f"Applying dense_to_sparse_batch with batch_size={batch_size}, "
                    f"row_shape={row_shape}")
        return dataset.sparse_batch(batch_size, row_shape)

    return _apply_fn


@tf_export("data.experimental.map_and_batch")
def map_and_batch(map_func,
                  batch_size,
                  num_parallel_calls=None,
                  drop_remainder=False):
    """Fused implementation of `map` and `batch` with enhanced features.

    Maps `map_func` across elements of the dataset and then combines them into
    batches. This transformation is more efficient than using `map` followed
    by `batch` due to potential optimizations.

    Args:
        map_func: A function mapping a nested structure of tensors to another
            nested structure of tensors.
        batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
            consecutive elements of this dataset to combine in a single batch.
        num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
            representing the number of elements to process in parallel.
            If not specified, elements will be processed sequentially.
            If `tf.data.AUTOTUNE` is used, the number of parallel calls is
            set dynamically based on available resources.
        drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
            whether the last batch should be dropped if it has fewer than
            `batch_size` elements. Defaults to `False`.

    Returns:
        A `Dataset` transformation function, which can be passed to
        `tf.data.Dataset.apply`.

    Example:
    ```python
    dataset = tf.data.Dataset.range(10)
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(lambda x: x * 2, batch_size=4))
    for batch in dataset:
        print(batch)
    ```
    """
    def _apply_fn(dataset):
        logger.info(f"Applying map_and_batch with batch_size={batch_size}, "
                    f"num_parallel_calls={num_parallel_calls}, drop_remainder={drop_remainder}")
        return _EnhancedMapAndBatchDataset(dataset, map_func, batch_size,
                                           num_parallel_calls, drop_remainder)

    return _apply_fn


@tf_export("data.experimental.unbatch")
def unbatch():
    """Splits elements of a dataset into multiple elements along the batch dimension.

    For example, if elements of the dataset are shaped `[B, a0, a1, ...]`,
    where `B` may vary for each input element, then the unbatched dataset will
    contain elements of shape `[a0, a1, ...]`.

    Returns:
        A `Dataset` transformation function, which can be passed to
        `tf.data.Dataset.apply`.

    Example:
    ```python
    dataset = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4], [5]])
    dataset = dataset.apply(tf.data.experimental.unbatch())
    for element in dataset:
        print(element)
    ```
    """
    def _apply_fn(dataset):
        logger.info("Applying unbatch")
        return dataset.unbatch()

    return _apply_fn


class _EnhancedMapAndBatchDataset(dataset_ops.UnaryDataset):
    """An enhanced Dataset that maps a function over a batch of elements efficiently."""

    def __init__(self, input_dataset, map_func, batch_size,
                 num_parallel_calls, drop_remainder):
        """Initializes the `_EnhancedMapAndBatchDataset`.

        Args:
            input_dataset: The input `Dataset`.
            map_func: A function mapping a nested structure of tensors to another
                nested structure of tensors.
            batch_size: A scalar representing the number of elements to combine.
            num_parallel_calls: (Optional.) A scalar representing the number of
                elements to process in parallel.
            drop_remainder: A boolean indicating whether to drop the last batch
                if it has fewer than `batch_size` elements.
        """
        logger.debug("Initializing EnhancedMapAndBatchDataset")
        self._input_dataset = input_dataset

        self._map_func = structured_function.StructuredFunctionWrapper(
            map_func,
            "tf.data.experimental.map_and_batch()",
            dataset=input_dataset
        )
        self._batch_size = ops.convert_to_tensor(
            batch_size, dtype=dtypes.int64, name="batch_size")
        self._drop_remainder = ops.convert_to_tensor(
            drop_remainder, dtype=dtypes.bool, name="drop_remainder")

        if num_parallel_calls is None:
            self._num_parallel_calls = tf.data.AUTOTUNE
        else:
            self._num_parallel_calls = ops.convert_to_tensor(
                num_parallel_calls, dtype=dtypes.int64, name="num_parallel_calls")

        constant_drop_remainder = tensor_util.constant_value(self._drop_remainder)
        if constant_drop_remainder:
            self._element_spec = nest.map_structure(
                lambda spec: spec._batch(tensor_util.constant_value(self._batch_size)),
                self._map_func.output_structure
            )
        else:
            self._element_spec = nest.map_structure(
                lambda spec: spec._batch(None),
                self._map_func.output_structure
            )

        variant_tensor = ged_ops.map_and_batch_dataset(
            self._input_dataset._variant_tensor,  # pylint: disable=protected-access
            self._map_func.function.captured_inputs,
            f=self._map_func.function,
            batch_size=self._batch_size,
            num_parallel_calls=self._num_parallel_calls,
            drop_remainder=self._drop_remainder,
            preserve_cardinality=True,
            **self._flat_structure
        )
        super().__init__(input_dataset, variant_tensor)

        logger.info("EnhancedMapAndBatchDataset initialized successfully")

    def _inputs(self):
        return [self._input_dataset]

    def _functions(self):
        return [self._map_func]

    @property
    def element_spec(self):
        return self._element_spec

    def __iter__(self):
        logger.debug("Creating iterator for EnhancedMapAndBatchDataset")
        try:
            for batch in super().__iter__():
                yield batch
        except Exception as e:
            logger.error(f"Error while iterating over dataset: {e}")
            raise RuntimeError(f"Error while iterating over dataset: {e}")


# Additional Enhancements

@tf_export("data.experimental.parallel_interleave")
def parallel_interleave(map_func,
                        cycle_length,
                        block_length=1,
                        num_parallel_calls=None,
                        deterministic=None):
    """Interleaves the elements produced by `map_func` in parallel.

    This transformation maps `map_func` across the elements of the input
    dataset and interleaves the results.

    Args:
        map_func: A function mapping a dataset element to a dataset.
        cycle_length: The number of input elements that will be processed
            concurrently.
        block_length: The number of consecutive elements to produce from each
            input element before cycling to another input element.
        num_parallel_calls: (Optional.) The number of parallel calls.
            If not specified, the transformation will be executed sequentially.
        deterministic: (Optional.) Controls the order in which elements are
            produced. If `True`, elements are produced in deterministic order.
            If `False`, the transformation is allowed to produce elements in
            any order.

    Returns:
        A `Dataset` transformation function.

    Example:
    ```python
    dataset = tf.data.Dataset.range(4)
    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            lambda x: tf.data.Dataset.range(10 * x, 10 * x + 5),
            cycle_length=2, block_length=1))
    for element in dataset:
        print(element)
    ```
    """
    def _apply_fn(dataset):
        logger.info(f"Applying parallel_interleave with cycle_length={cycle_length}, "
                    f"block_length={block_length}, num_parallel_calls={num_parallel_calls}, "
                    f"deterministic={deterministic}")
        return dataset.interleave(
            map_func,
            cycle_length=cycle_length,
            block_length=block_length,
            num_parallel_calls=num_parallel_calls,
            deterministic=deterministic
        )

    return _apply_fn


@tf_export("data.experimental.prefetch_to_device")
def prefetch_to_device(device, buffer_size=None):
    """A transformation that prefetches dataset elements to the given device.

    Args:
        device: A string. The name of a device to which elements will be prefetched.
        buffer_size: (Optional.) The maximum number of elements to buffer
            on the device. Defaults to an automatically chosen value.

    Returns:
        A `Dataset` transformation function.

    Example:
    ```python
    dataset = tf.data.Dataset.range(10)
    dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/gpu:0"))
    for element in dataset:
        print(element)
    ```
    """
    def _apply_fn(dataset):
        logger.info(f"Applying prefetch_to_device with device={device}, buffer_size={buffer_size}")
        if buffer_size is None:
            buffer_size = tf.data.AUTOTUNE
        return dataset.apply(
            tf.data.experimental.copy_to_device(device)
        ).prefetch(buffer_size)

    return _apply_fn


# Improved Error Handling and Logging

class EnhancedDatasetIterator:
    """Iterator class with enhanced logging for datasets."""

    def __init__(self, dataset):
        self._dataset = dataset
        self._iterator = iter(dataset)
        logger.debug("EnhancedDatasetIterator initialized")

    def __iter__(self):
        return self

    def __next__(self):
        try:
            element = next(self._iterator)
            logger.debug("Fetched next element from dataset")
            return element
        except StopIteration:
            logger.info("Reached end of dataset")
            raise
        except Exception as e:
            logger.error(f"Error while fetching next element: {e}")
            raise RuntimeError(f"Error while fetching next element: {e}")
def example_usage():
    """Example usage of the datasets."""
    dataset = tf.data.Dataset.range(10)
    logger.info("Created dataset from range 0 to 9")

    # Using the enhanced map and batch
    dataset = dataset.apply(
        map_and_batch(
            map_func=lambda x: x * 2,
            batch_size=4,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    )
    logger.info("Applied map_and_batch to dataset")

    # Prefetching to device
    dataset = dataset.apply(prefetch_to_device("/gpu:0"))
    logger.info("Applied prefetch_to_device to dataset")

    # Iterating over the dataset with enhanced iterator
    enhanced_iterator = EnhancedDatasetIterator(dataset)
    for batch in enhanced_iterator:
        logger.info(f"Processing batch: {batch}")
        print(batch)

if __name__ == "__main__":
    example_usage()
