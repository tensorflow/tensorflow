# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Experimental API for optimizing `tf.data` pipelines."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops


# TODO(jsimsa): Support RE matching for both individual transformation (e.g. to
# account for indexing) and transformation sequence.
def assert_next(transformations):
  """A transformation that asserts which transformations happen next.

  Args:
    transformations: A `tf.string` vector `tf.Tensor` identifying the
      transformations that are expected to happen next.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    return _AssertNextDataset(dataset, transformations)

  return _apply_fn


def model():
  """A transformation that models performance.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    return dataset_ops._ModelDataset(dataset)  # pylint: disable=protected-access

  return _apply_fn


def non_serializable():
  """A non-serializable identity transformation.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    return _NonSerializableDataset(dataset)

  return _apply_fn


def optimize(optimizations=None):
  """A transformation that applies optimizations.

  Args:
    optimizations: (Optional.) A `tf.string` vector `tf.Tensor` identifying
      optimizations to use. If not specified, the default set of optimizations
      is applied.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    return dataset_ops._OptimizeDataset(dataset, optimizations)  # pylint: disable=protected-access

  return _apply_fn


class _AssertNextDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` that asserts which transformations happen next."""

  def __init__(self, input_dataset, transformations):
    """See `assert_next()` for details."""
    self._input_dataset = input_dataset
    if transformations is None:
      raise ValueError("At least one transformation should be specified")
    self._transformations = ops.convert_to_tensor(
        transformations, dtype=dtypes.string, name="transformations")
    variant_tensor = (
        gen_experimental_dataset_ops.assert_next_dataset(
            self._input_dataset._variant_tensor,  # pylint: disable=protected-access
            self._transformations,
            **self._flat_structure))
    super(_AssertNextDataset, self).__init__(input_dataset, variant_tensor)


class _NonSerializableDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` that performs non-serializable identity transformation."""

  def __init__(self, input_dataset):
    """See `non_serializable()` for details."""
    self._input_dataset = input_dataset
    variant_tensor = (
        gen_experimental_dataset_ops.non_serializable_dataset(
            self._input_dataset._variant_tensor,  # pylint: disable=protected-access
            **self._flat_structure))
    super(_NonSerializableDataset, self).__init__(input_dataset, variant_tensor)


class _ChooseFastestDataset(dataset_ops.DatasetV2):
  """A `Dataset` that merges two input datasets."""

  def __init__(self, datasets, num_experiments=10):
    """Chooses the fastest of some input datasets.

    Given input datasets, produces elements as quickly as the fastest of the
    inputs. Note that this dataset assumes that input datasets have the same
    elements in the same order, though this is not enforced besides checking
    that the input datasets have compatible output types, output shapes, and
    cardinality at runtime. The resulting dataset produces elements that are
    identical to the input elements, and in the same order.

    Note that the time to first iteration is longer when this dataset is used
    due to the overhead of dynamically picking the faster dataset. Namely,
    for the first num_experiments iterations, this dataset will pull from all
    of its inputs simultaneously in order to determine which input is the
    fastest. For all subsequent iterations, that input will be used.

    Args:
      datasets: A list of `Datasets` that all have the same elements in the same
        order.
      num_experiments: The number of experiments to run before deciding which
        dataset is fastest. In each "experiment" iteration, the dataset will
        call from all its inputs simultaneously, and update its knowledge of
        which input is the fastest.

    Returns:
      A `Dataset` that has the same elements the inputs.
    """
    self._datasets = list(datasets)
    self._element_spec = self._datasets[0].element_spec
    variant_tensor = (
        gen_experimental_dataset_ops.choose_fastest_dataset(
            [dataset._variant_tensor for dataset in self._datasets],  # pylint: disable=protected-access
            num_experiments=num_experiments,
            **self._flat_structure))
    super(_ChooseFastestDataset, self).__init__(variant_tensor)

  def _inputs(self):
    return self._datasets

  @property
  def element_spec(self):
    return self._element_spec


class _ChooseFastestBranchDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that merges two input datasets."""

  def __init__(self,
               input_dataset,
               functions,
               ratio_numerator=1,
               ratio_denominator=1,
               num_elements_per_branch=None):
    """Chooses the fastest of some dataset functions.

    Given dataset functions that take input_dataset as input and output
    another dataset, produces elements as quickly as the fastest of these
    output datasets. Note that datasets in the dataset functions are assumed
    to be stateless, and the iterators created by the functions' output datasets
    will, given the same input elements, all produce the same output elements.
    Datasets in the functions are also expected to iterate over the input
    dataset at most once. The violation of these conditions may lead to
    undefined behavior.

    For example:
    ```python
    dataset = tf.data.Dataset.range(100)
    dataset = _ChooseFastestDataset(
        dataset,
        [
            lambda ds: ds.map(lambda x: tf.reshape(x, [1])).batch(10),
            lambda ds: ds.batch(10).map(lambda x: tf.reshape(x, [10, 1]))
        ],
        ratio=10,
        num_elements_per_branch=10
    )
    ```
    The resulting dataset will produce elements equivalent to
    `tf.data.Dataset.range(100).map(lambda x: tf.reshape(x, [1])).batch(10)`, or
    `tf.data.Dataset.range(100).batch(10).map(lambda x: tf.reshape(x, [10, 1]))`

    Note that the first `num_elements_per_branch` iterations may be slower due
    to the
    overhead of dynamically picking the fastest dataset. Namely, for these
    iterations, the dataset will produce elements from any of branches to
    determine which input is the fastest. For all subsequent iterations, that
    input will be used.

    Args:
      input_dataset: A `Dataset` that can be used as input to `functions`.
      functions: A list of callables, each of which takes a `Dataset` as input
        and returns a `Dataset`.
      ratio_numerator: The numerator in the ratio of input elements consumed to
        output elements produced for each function. This should be the same for
        all functions. For example, if the function is
        `lambda ds: ds.batch(10)`, the ratio is 10:1, i.e. the input dataset
          must produce 10 elements for every element of the output dataset. In
          this case, ratio_numerator should be 10.
      ratio_denominator: The denominator in the ratio of input elements consumed
        to output elements produced for each function. This should be the same
        for all functions. For example, if the function is
        `lambda ds: ds.batch(10)`, the ratio is 10:1, i.e. the input dataset
          must produce 10 elements for every element of the output dataset. In
          this case, ratio_denominator should be 1.
      num_elements_per_branch: The number of elements to get from each branch
        before deciding which dataset is fastest. In the first len(functions) *
        num_elements_per_branch iterations, the dataset will call from one of
        the branches, and update its knowledge of which input is the fastest.
        Note that (num_elements_per_branch * ratio) is expected to be an
        integer.

    Returns:
      A `Dataset` that has the same elements the inputs.
    """
    input_structure = dataset_ops.DatasetSpec(input_dataset.element_spec)
    self._funcs = [
        dataset_ops.StructuredFunctionWrapper(
            f, "ChooseFastestV2", input_structure=input_structure)
        for f in functions
    ]
    self._element_spec = self._funcs[0].output_structure._element_spec  # pylint: disable=protected-access

    self._captured_arguments = []
    for f in self._funcs:
      self._captured_arguments.extend(f.function.captured_inputs)
    self._capture_lengths = [
        len(f.function.captured_inputs) for f in self._funcs
    ]

    if ratio_numerator <= 0 or ratio_denominator <= 0:
      raise ValueError("ratio must be positive.")

    if num_elements_per_branch is None:
      # Pick a sensible default based on `ratio_denominator`
      num_elements_per_branch = 10 * ratio_denominator

    variant_tensor = (
        gen_experimental_dataset_ops.choose_fastest_branch_dataset(
            input_dataset._variant_tensor,  # pylint: disable=protected-access
            ratio_numerator=ratio_numerator,
            ratio_denominator=ratio_denominator,
            other_arguments=self._captured_arguments,
            num_elements_per_branch=num_elements_per_branch,
            branches=[f.function for f in self._funcs],
            other_arguments_lengths=self._capture_lengths,
            **self._flat_structure))
    super(_ChooseFastestBranchDataset, self).__init__(input_dataset,
                                                      variant_tensor)

  @property
  def element_spec(self):
    return self._element_spec
