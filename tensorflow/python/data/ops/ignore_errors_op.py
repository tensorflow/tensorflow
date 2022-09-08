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
"""The implementation of `tf.data.Dataset.ignore_errors`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops


def ignore_errors(self, log_warning=False, name=None):
  return IgnoreErrorsDataset(self, log_warning, name)


class IgnoreErrorsDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` that drops erroneous elements from its input."""

  def __init__(self, input_dataset, log_warning, name=None):
    """See `Dataset.ignore_errors` for details."""
    self._input_dataset = input_dataset
    self._name = name
    variant_tensor = (
        gen_experimental_dataset_ops.ignore_errors_dataset(
            self._input_dataset._variant_tensor,  # pylint: disable=protected-access
            log_warning=log_warning,
            **self._flat_structure))
    super().__init__(input_dataset, variant_tensor)
