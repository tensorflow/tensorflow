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

from tensorflow.python import pywrap_tensorflow


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
  result = nest.pack_sequence_as(ct, transformed_list_of_tensors)
  ```
  """

  @abc.abstractmethod
  def _to_components(self):
    """Decomposes this composite tensor into its components.

    Returns:
      The components that comprise this composite tensor: a nested structure
      (as defined by `tf.python.util.nest`) whose values are `tf.Tensor`s or
      `CompositeTensor`s.
    """
    raise NotImplementedError("CompositeTensor._to_components")

  @abc.abstractmethod
  def _from_components(cls, components):  # pylint: disable=no-self-argument
    """Creates a composite tensor of type `cls` from components.

    Args:
      components: The components that should be used to form the
        composite tensor: a nested structure (as defined by
        `tf.python.util.nest`) whose values are tf.Tensors or composite
        tensors.

    Returns:
      A `CompositeTensor` of type `cls`.
    """
    raise NotImplementedError("CompositeTensor._from_components")

  @abc.abstractmethod
  def _shape_invariant_to_components(self, shape=None):
    """Converts a shape invariant into invariants for individual components.

    Args:
      shape: A `tf.TensorShape` object.  The shape invariant for this
        `CompositeTensor`, or `None` if a default shape invariant should be
        used (based on the value of this `CompositeTensor`).

    Returns:
      A nested structure whose values are `tf.TensorShape` objects, specifying
      the shape invariants for the tensors that comprise this `CompositeTensor`.
    """
    raise NotImplementedError("CompositeTensor._shape_invariant_to_components")

  @abc.abstractproperty
  def _is_graph_tensor(self):
    """Returns True if this tensor's components belong to a TF graph."""
    raise NotImplementedError("CompositeTensor._is_symbolic_tensor")


pywrap_tensorflow.RegisterType("CompositeTensor", CompositeTensor)


# @TODO(edloper): Can we replace convert_to_tensor_or_xyz with just
# convert_to_tensor_or_composite?  Alternatively, should composite tensors
# register a dispatch override for tf.convert_to_tensor?
