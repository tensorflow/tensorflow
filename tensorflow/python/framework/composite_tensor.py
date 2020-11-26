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
"""Tensor-like objects that are composed from tf.Tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.python import _pywrap_utils
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


@tf_export("__internal__.CompositeTensor", v1=[])
@six.add_metaclass(abc.ABCMeta)
class CompositeTensor(object):
  """Abstract base class for Tensor-like objects that are composed from Tensors.

  Each `CompositeTensor` can be decomposed into a structured collection of
  component `tf.Tensor`s, and reconstructed from those components.

  The `tensorflow.python.util.nest` module has support for treating composite
  tensors as structure, which makes it easy to flatten and reconstruct
  composite tensors (or larger structures that contain composite tensors).
  E.g.:

  ```python
  ct = ...  # Create a composite tensor.
  flat_list_of_tensors = nest.flatten(ct, expand_composites=True)
  transformed_list_of_tensors = ...  # do something with the flat tensors.
  result = nest.pack_sequence_as(ct, transformed_list_of_tensors,
                                 expand_composites=True)
  ```
  """

  @abc.abstractproperty
  def _type_spec(self):
    """A `TypeSpec` describing the type of this value."""
    raise NotImplementedError("%s._type_spec()" % type(self).__name__)

  def _shape_invariant_to_type_spec(self, shape):
    """Returns a TypeSpec given a shape invariant (used by `tf.while_loop`).

    Args:
      shape: A `tf.TensorShape` object.  The shape invariant for this
        `CompositeTensor`, or `None` if a default shape invariant should be used
        (based on the value of this `CompositeTensor`).

    Returns:
      A nested structure whose values are `tf.TensorShape` objects, specifying
      the shape invariants for the tensors that comprise this `CompositeTensor`.
    """
    # New TypeSpec subclasses generally do not need to implement this --
    # this method is used for backwards compatibility.  Users of tf.while_loop
    # can specify a type by passing in TypeSpec instead.
    raise NotImplementedError("%s._shape_invariant_to_type_spec" %
                              type(self).__name__)

  def _consumers(self):
    """Returns a list of `Operation`s that consume this `CompositeTensor`.

    Returns:
      A list of `Operation`s.

    Raises:
      RuntimeError: If this method is called while executing eagerly.
    """
    consumers = nest.flatten([
        component.consumers()
        for component in nest.flatten(self, expand_composites=True)
        if getattr(component, "graph", None) is not None
    ])
    return list(set(consumers))


_pywrap_utils.RegisterType("CompositeTensor", CompositeTensor)


def replace_composites_with_components(structure):
  """Recursively replaces CompositeTensors with their components.

  Args:
    structure: A `nest`-compatible structure, possibly containing composite
      tensors.

  Returns:
    A copy of `structure`, where each composite tensor has been replaced by
    its components.  The result will contain no composite tensors.
    Note that `nest.flatten(replace_composites_with_components(structure))`
    returns the same value as `nest.flatten(structure)`.
  """
  if isinstance(structure, CompositeTensor):
    return replace_composites_with_components(
        structure._type_spec._to_components(structure))  # pylint: disable=protected-access
  elif not nest.is_sequence(structure):
    return structure
  else:
    return nest.map_structure(
        replace_composites_with_components, structure, expand_composites=False)


# @TODO(edloper): Can we replace convert_to_tensor_or_xyz with just
# convert_to_tensor_or_composite?  Alternatively, should composite tensors
# register a dispatch override for tf.convert_to_tensor?
