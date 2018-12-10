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
from tensorflow.python.util.tf_export import tf_export


# A constant that can be used to enable auto-tuning.
AUTOTUNE = -1
tf_export("data.experimental.AUTOTUNE").export_constant(__name__, "AUTOTUNE")


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
    super(_AssertNextDataset, self).__init__(input_dataset)
    self._input_dataset = input_dataset
    if transformations is None:
      raise ValueError("At least one transformation should be specified")
    self._transformations = ops.convert_to_tensor(
        transformations, dtype=dtypes.string, name="transformations")

  def _as_variant_tensor(self):
    return gen_experimental_dataset_ops.experimental_assert_next_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        self._transformations,
        **dataset_ops.flat_structure(self))


class _NonSerializableDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` that performs non-serializable identity transformation."""

  def __init__(self, input_dataset):
    """See `non_serializable()` for details."""
    super(_NonSerializableDataset, self).__init__(input_dataset)
    self._input_dataset = input_dataset

  def _as_variant_tensor(self):
    return gen_experimental_dataset_ops.experimental_non_serializable_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        **dataset_ops.flat_structure(self))
