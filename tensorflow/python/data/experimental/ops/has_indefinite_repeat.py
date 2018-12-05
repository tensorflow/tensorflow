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
"""Finiteness analysis of `Dataset` objects."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.util.tf_export import tf_export


class _AssumeFiniteDataset(dataset_ops.UnaryUnchangedStructureDataset):

  def __init__(self, input_dataset):
    super(_AssumeFiniteDataset, self).__init__(input_dataset)
    self._input_dataset = input_dataset

  def _as_variant_tensor(self):
    return self._input_dataset._as_variant_tensor()  # pylint: disable=protected-access


@tf_export("data.experimental.assume_finite")
def assume_finite():
  """Assume that the input is finite, even if it contains `Dataset.repeat()`.

  Training libraries may analyze a `tf.data.Dataset` to determine if it is
  finite or infinite (e.g. because it contains an indefinite
  `tf.data.Dataset.repeat` transformation). Since that analysis may be
  imprecise, this transformation allows the user to annotate a dataset
  explicitly as being finite.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    return _AssumeFiniteDataset(dataset)

  return _apply_fn


def has_indefinite_repeat(dataset):
  """Returns `True` if `dataset` or any of its inputs is `Dataset.repeat()`.

  NOTE: For simplicity, this analysis does not attempt to analyze nested
  datasets (e.g. in a function passed to `tf.data.Dataset.flat_map`). If the
  analysis is incorrect, you can apply `tf.data.experimental.assume_finite()`
  to the dataset to override it.

  Args:
    dataset: A `tf.data.Dataset`.

  Returns:
    `True` if `dataset` or any of its inputs is repeated indefinitely.
  """
  # pylint: disable=protected-access
  if isinstance(dataset, dataset_ops.DatasetV1Adapter):
    return has_indefinite_repeat(dataset._dataset)
  elif isinstance(dataset, dataset_ops.RepeatDataset):
    count = tensor_util.constant_value(dataset._count)
    return count == -1 or has_indefinite_repeat(dataset._inputs()[0])
  elif isinstance(dataset, _AssumeFiniteDataset):
    return False
  else:
    return any(
        has_indefinite_repeat(input_dataset)
        for input_dataset in dataset._inputs())
