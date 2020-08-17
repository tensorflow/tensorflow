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
"""Ignore_errors dataset transformations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.compat import compat


@tf_export("data.experimental.ignore_errors")
def ignore_errors(log_warning=False):
  """Creates a `Dataset` from another `Dataset` and silently ignores any errors.

  Use this transformation to produce a dataset that contains the same elements
  as the input, but silently drops any elements that caused an error. For
  example:

  ```python
  dataset = tf.data.Dataset.from_tensor_slices([1., 2., 0., 4.])

  # Computing `tf.debugging.check_numerics(1. / 0.)` will raise an
  InvalidArgumentError.
  dataset = dataset.map(lambda x: tf.debugging.check_numerics(1. / x, "error"))

  # Using `ignore_errors()` will drop the element that causes an error.
  dataset =
      dataset.apply(tf.data.experimental.ignore_errors())  # ==> {1., 0.5, 0.2}
  ```
  Args:
     log_warning: (Optional.) A 'tf.bool' scalar indicating whether ignored
      errors should be logged to stderr. Defaults to 'False'.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    return _IgnoreErrorsDataset(dataset, log_warning)

  return _apply_fn


class _IgnoreErrorsDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` that silently ignores errors when computing its input."""

  def __init__(self, input_dataset, log_warning):
    """See `Dataset.ignore_errors()` for details."""
    self._input_dataset = input_dataset
    if compat.forward_compatible(2020, 8, 26) or log_warning:
      variant_tensor = (
          gen_experimental_dataset_ops.ignore_errors_dataset(
              self._input_dataset._variant_tensor,  # pylint: disable=protected-access
              log_warning=log_warning,
              **self._flat_structure))
    else:
      variant_tensor = (
          gen_experimental_dataset_ops.ignore_errors_dataset(
              self._input_dataset._variant_tensor,  # pylint: disable=protected-access
              **self._flat_structure))
    super(_IgnoreErrorsDataset, self).__init__(input_dataset, variant_tensor)
