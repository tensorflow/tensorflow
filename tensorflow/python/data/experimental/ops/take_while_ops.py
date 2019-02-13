# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""take-while dataset transformation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure as structure_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.util.tf_export import tf_export


class _TakeWhileDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A dataset that stops iteration when `predicate` returns false."""

  def __init__(self, input_dataset, predicate):
    """See `take_while()` for details."""

    self._input_dataset = input_dataset
    wrapped_func = dataset_ops.StructuredFunctionWrapper(
        predicate,
        "tf.data.experimental.take_while()",
        dataset=self._input_dataset)

    if not wrapped_func.output_structure.is_compatible_with(
        structure_lib.TensorStructure(dtypes.bool, [])):
      raise ValueError("`predicate` must return a scalar boolean tensor.")

    self._predicate = wrapped_func
    var_tensor = gen_experimental_dataset_ops.experimental_take_while_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        other_arguments=self._predicate.function.captured_inputs,
        predicate=self._predicate.function,
        **dataset_ops.flat_structure(self))
    super(_TakeWhileDataset, self).__init__(input_dataset, var_tensor)

  def _functions(self):
    return [self._predicate]


@tf_export("data.experimental.take_while")
def take_while(predicate):
  """A transformation that stops dataset iteration based on a `predicate`.

  Args:
    predicate: A function that maps a nested structure of tensors (having shapes
      and types defined by `self.output_shapes` and `self.output_types`) to a
      scalar `tf.bool` tensor.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    return _TakeWhileDataset(dataset, predicate)

  return _apply_fn
