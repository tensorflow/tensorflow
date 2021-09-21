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
"""Non-deterministic dataset transformations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@deprecation.deprecated(
    None,
    "Use `tf.data.Dataset.interleave(map_func, cycle_length, block_length, "
    "num_parallel_calls=tf.data.AUTOTUNE)` instead. If sloppy "
    "execution is desired, use `tf.data.Options.deterministic`.")
@tf_export("data.experimental.parallel_interleave")
def parallel_interleave(map_func,
                        cycle_length,
                        block_length=1,
                        sloppy=False,
                        buffer_output_elements=None,
                        prefetch_input_elements=None):
  """A parallel version of the `Dataset.interleave()` transformation.

  `parallel_interleave()` maps `map_func` across its input to produce nested
  datasets, and outputs their elements interleaved. Unlike
  `tf.data.Dataset.interleave`, it gets elements from `cycle_length` nested
  datasets in parallel, which increases the throughput, especially in the
  presence of stragglers. Furthermore, the `sloppy` argument can be used to
  improve performance, by relaxing the requirement that the outputs are produced
  in a deterministic order, and allowing the implementation to skip over nested
  datasets whose elements are not readily available when requested.

  Example usage:

  ```python
  # Preprocess 4 files concurrently.
  filenames = tf.data.Dataset.list_files("/path/to/data/train*.tfrecords")
  dataset = filenames.apply(
      tf.data.experimental.parallel_interleave(
          lambda filename: tf.data.TFRecordDataset(filename),
          cycle_length=4))
  ```

  WARNING: If `sloppy` is `True`, the order of produced elements is not
  deterministic.

  Args:
    map_func: A function mapping a nested structure of tensors to a `Dataset`.
    cycle_length: The number of input `Dataset`s to interleave from in parallel.
    block_length: The number of consecutive elements to pull from an input
      `Dataset` before advancing to the next input `Dataset`.
    sloppy: A boolean controlling whether determinism should be traded for
      performance by allowing elements to be produced out of order.  If `sloppy`
      is `None`, the `tf.data.Options.deterministic` dataset option (`True` by
      default) is used to decide whether to enforce a deterministic order.
    buffer_output_elements: The number of elements each iterator being
      interleaved should buffer (similar to the `.prefetch()` transformation for
      each interleaved iterator).
    prefetch_input_elements: The number of input elements to transform to
      iterators before they are needed for interleaving.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    return readers.ParallelInterleaveDataset(dataset, map_func, cycle_length,
                                             block_length, sloppy,
                                             buffer_output_elements,
                                             prefetch_input_elements)

  return _apply_fn


@deprecation.deprecated(None,
                        "Use `tf.data.Dataset.sample_from_datasets(...)`.")
@tf_export("data.experimental.sample_from_datasets", v1=[])
def sample_from_datasets_v2(datasets,
                            weights=None,
                            seed=None,
                            stop_on_empty_dataset=False):
  """Samples elements at random from the datasets in `datasets`.

  Creates a dataset by interleaving elements of `datasets` with `weight[i]`
  probability of picking an element from dataset `i`. Sampling is done without
  replacement. For example, suppose we have 2 datasets:

  ```python
  dataset1 = tf.data.Dataset.range(0, 3)
  dataset2 = tf.data.Dataset.range(100, 103)
  ```

  Suppose also that we sample from these 2 datasets with the following weights:

  ```python
  sample_dataset = tf.data.Dataset.sample_from_datasets(
      [dataset1, dataset2], weights=[0.5, 0.5])
  ```

  One possible outcome of elements in sample_dataset is:

  ```
  print(list(sample_dataset.as_numpy_iterator()))
  # [100, 0, 1, 101, 2, 102]
  ```

  Args:
    datasets: A non-empty list of `tf.data.Dataset` objects with compatible
      structure.
    weights: (Optional.) A list or Tensor of `len(datasets)` floating-point
      values where `weights[i]` represents the probability to sample from
      `datasets[i]`, or a `tf.data.Dataset` object where each element is such a
      list. Defaults to a uniform distribution across `datasets`.
    seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
      seed that will be used to create the distribution. See
      `tf.random.set_seed` for behavior.
    stop_on_empty_dataset: If `True`, sampling stops if it encounters an empty
      dataset. If `False`, it skips empty datasets. It is recommended to set it
      to `True`. Otherwise, the distribution of samples starts off as the user
      intends, but may change as input datasets become empty. This can be
      difficult to detect since the dataset starts off looking correct. Default
      to `False` for backward compatibility.

  Returns:
    A dataset that interleaves elements from `datasets` at random, according to
    `weights` if provided, otherwise with uniform probability.

  Raises:
    TypeError: If the `datasets` or `weights` arguments have the wrong type.
    ValueError:
      - If `datasets` is empty, or
      - If `weights` is specified and does not match the length of `datasets`.
  """
  return dataset_ops.Dataset.sample_from_datasets(
      datasets=datasets,
      weights=weights,
      seed=seed,
      stop_on_empty_dataset=stop_on_empty_dataset)


@deprecation.deprecated(None,
                        "Use `tf.data.Dataset.sample_from_datasets(...)`.")
@tf_export(v1=["data.experimental.sample_from_datasets"])
def sample_from_datasets_v1(datasets,
                            weights=None,
                            seed=None,
                            stop_on_empty_dataset=False):
  return dataset_ops.DatasetV1Adapter(
      sample_from_datasets_v2(datasets, weights, seed, stop_on_empty_dataset))


sample_from_datasets_v1.__doc__ = sample_from_datasets_v2.__doc__


@deprecation.deprecated(
    None, "Use `tf.data.Dataset.choose_from_datasets(...)` instead. Note that, "
    "unlike the experimental endpoint, the non-experimental endpoint "
    "sets `stop_on_empty_dataset=True` by default. You should set this "
    "argument explicitly in case you would like to match the behavior of the "
    "experimental endpoint.")
@tf_export("data.experimental.choose_from_datasets", v1=[])
def choose_from_datasets_v2(datasets,
                            choice_dataset,
                            stop_on_empty_dataset=False):
  """Creates a dataset that deterministically chooses elements from `datasets`.

  For example, given the following datasets:

  ```python
  datasets = [tf.data.Dataset.from_tensors("foo").repeat(),
              tf.data.Dataset.from_tensors("bar").repeat(),
              tf.data.Dataset.from_tensors("baz").repeat()]

  # Define a dataset containing `[0, 1, 2, 0, 1, 2, 0, 1, 2]`.
  choice_dataset = tf.data.Dataset.range(3).repeat(3)

  result = tf.data.experimental.choose_from_datasets(datasets, choice_dataset)
  ```

  The elements of `result` will be:

  ```
  "foo", "bar", "baz", "foo", "bar", "baz", "foo", "bar", "baz"
  ```

  Args:
    datasets: A non-empty list of `tf.data.Dataset` objects with compatible
      structure.
    choice_dataset: A `tf.data.Dataset` of scalar `tf.int64` tensors between `0`
      and `len(datasets) - 1`.
    stop_on_empty_dataset: If `True`, selection stops if it encounters an empty
      dataset. If `False`, it skips empty datasets. It is recommended to set it
      to `True`. Otherwise, the selected elements start off as the user intends,
      but may change as input datasets become empty. This can be difficult to
      detect since the dataset starts off looking correct. Default to `False`
      for backward compatibility.

  Returns:
    A dataset that interleaves elements from `datasets` according to the values
    of `choice_dataset`.

  Raises:
    TypeError: If `datasets` or `choice_dataset` has the wrong type.
    ValueError: If `datasets` is empty.
  """
  return dataset_ops.Dataset.choose_from_datasets(
      datasets=datasets,
      choice_dataset=choice_dataset,
      stop_on_empty_dataset=stop_on_empty_dataset)


@deprecation.deprecated(
    None, "Use `tf.data.Dataset.choose_from_datasets(...)` instead. Note that, "
    "unlike the experimental endpoint, the non-experimental endpoint "
    "sets `stop_on_empty_dataset=True` by default. You should set this "
    "argument explicitly in case you would like to match the behavior of the "
    "experimental endpoint.")
@tf_export(v1=["data.experimental.choose_from_datasets"])
def choose_from_datasets_v1(datasets,
                            choice_dataset,
                            stop_on_empty_dataset=False):
  return dataset_ops.DatasetV1Adapter(
      choose_from_datasets_v2(datasets, choice_dataset, stop_on_empty_dataset))


choose_from_datasets_v1.__doc__ = choose_from_datasets_v2.__doc__

if tf2.enabled():
  choose_from_datasets = choose_from_datasets_v2
  sample_from_datasets = sample_from_datasets_v2
else:
  choose_from_datasets = choose_from_datasets_v1
  sample_from_datasets = sample_from_datasets_v1
