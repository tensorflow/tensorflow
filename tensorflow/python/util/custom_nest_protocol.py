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

"""Protocol class for custom tf.nest support."""

import typing
from typing import Protocol


@typing.runtime_checkable
class CustomNestProtocol(Protocol):
  """Protocol for adding custom tf.nest support in user-defined classes.

  User classes should implement the two methods defined in this protocol in
  order to be supported by nest functions.
    - `__tf_flatten__` for generating the flattened components and the metadata
      of the current object.
    - `__tf_unflatten__` for creating a new object based on the input metadata
      and the components.
  See the method doc for details.

  In terms of support level, classes implementing this protocol
    - are supported by tf.nest and tf.data functions.
    - have limited support from tf.function, which requires writing a custom
      TraceType subclass to be used as the input or output of a tf.function.
    - are NOT supported by SavedModel.

  Code Examples:

  >>> import dataclasses
  >>> @dataclasses.dataclass
  ... class MaskedTensor:
  ...   mask: bool
  ...   value: tf.Tensor
  ...
  ...   def __tf_flatten__(self):
  ...     metadata = (self.mask,)  # static config.
  ...     components = (self.value,)  # dynamic values.
  ...     return metadata, components
  ...
  ...   @classmethod
  ...   def __tf_unflatten__(cls, metadata, components):
  ...     mask = metadata[0]
  ...     value = components[0]
  ...     return MaskedTensor(mask=mask, value=value)
  ...
  >>> mt = MaskedTensor(mask=True, value=tf.constant([1]))
  >>> mt
  MaskedTensor(mask=True, value=<tf.Tensor: ... numpy=array([1], dtype=int32)>)
  >>> tf.nest.is_nested(mt)
  True
  >>> mt2 = MaskedTensor(mask=False, value=tf.constant([2]))
  >>> tf.nest.assert_same_structure(mt, mt2)

  >>> leaves = tf.nest.flatten(mt)
  >>> leaves
  [<tf.Tensor: shape=(1,), dtype=int32, numpy=array([1], dtype=int32)>]

  >>> mt3 = tf.nest.pack_sequence_as(mt, leaves)
  >>> mt3
  MaskedTensor(mask=True, value=<tf.Tensor: ... numpy=array([1], dtype=int32)>)
  >>> bool(mt == mt3)
  True

  >>> tf.nest.map_structure(lambda x: x * 2, mt)
  MaskedTensor(mask=True, value=<tf.Tensor: ... numpy=array([2], dtype=int32)>)

  More examples are available in the unit tests (nest_test.py).
  """

  def __tf_flatten__(self):
    """Flatten current object into (metadata, components).

    Returns:
      A `tuple` of (metadata, components), where
        - metadata is a custom Python object that stands for the static config
          of the current object, which is supposed to be fixed and not affected
          by data transformation.
        - components is a `tuple` that contains the modifiable fields of the
          current object.

    Implementation Note:
    - This method should not invoke any TensorFlow ops.
    - This method only needs to flatten the current level. If current object has
      an attribute that also need custom flattening, nest functions (such as
      `nest.flatten`) will utilize this method to do recursive flattening.
    - Components must ba a `tuple`, not a `list`
    """

  @classmethod
  def __tf_unflatten__(cls, metadata, components):
    """Create a user-defined object from (metadata, components).

    Args:
      metadata: a custom Python objet that stands for the static config for
        reconstructing a new object of the current class.
      components: a `tuple` that contains the dynamic data fields of the current
        class, for object reconstruction.

    Returns:
      The user-defined object, with the same class of the current object.

    Implementation Note:
    - This method should not invoke any TensorFlow ops.
    - This method only needs to unflatten the current level. If the object has
      an attribute that also need custom unflattening, nest functions will
      utilize this method to do recursive unflattening.
    """
