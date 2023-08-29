# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""NoneTensor and NoneTensorSpec classes."""

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry


# TODO(b/149584798): add tests for non-tf.data functionality.
class NoneTensor(composite_tensor.CompositeTensor):
  """Composite tensor representation for `None` value."""

  @property
  def _type_spec(self):
    return NoneTensorSpec()


# TODO(b/149584798): add tests for non-tf.data functionality.
@type_spec_registry.register("tf.NoneTensorSpec")
class NoneTensorSpec(type_spec.BatchableTypeSpec):
  """Type specification for `None` value."""

  @property
  def value_type(self):
    return NoneTensor

  def _serialize(self):
    return ()

  @property
  def _component_specs(self):
    return []

  def _to_components(self, value):
    return []

  def _from_components(self, components):
    return

  def _to_tensor_list(self, value):
    return []

  @staticmethod
  def from_value(value):
    return NoneTensorSpec()

  def _batch(self, batch_size):
    return NoneTensorSpec()

  def _unbatch(self):
    return NoneTensorSpec()

  def _to_batched_tensor_list(self, value):
    return []

  def _to_legacy_output_types(self):
    return self

  def _to_legacy_output_shapes(self):
    return self

  def _to_legacy_output_classes(self):
    return self

  def most_specific_compatible_shape(self, other):
    if type(self) is not type(other):
      raise ValueError("No `TypeSpec` is compatible with both {} and {}".format(
          self, other))
    return self


type_spec.register_type_spec_from_value_converter(type(None),
                                                  NoneTensorSpec.from_value)
