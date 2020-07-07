# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Keras Input Tensor used to track functional API Topology."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec as type_spec_module
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity

_KERAS_TENSORS_ENABLED = False


def enable_keras_tensors():
  """Enable using KerasTensors in Keras's functional API."""
  global _KERAS_TENSORS_ENABLED
  _KERAS_TENSORS_ENABLED = True


def disable_keras_tensors():
  """Disable using KerasTensors in Keras's functional API."""
  global _KERAS_TENSORS_ENABLED
  _KERAS_TENSORS_ENABLED = False


def keras_tensors_enabled():
  """Return a bool specifying if KerasTensors are enabled."""
  return _KERAS_TENSORS_ENABLED and ops.executing_eagerly_outside_functions()


class KerasTensor(object):
  """A representation of a Keras in/output during Functional API construction.

  `KerasTensor`s are an alternative representation for Keras `Inputs`
  and for intermediate outputs of layers during Functional API construction of
  models. They are a lightweight data structure comprised of only the
  `tf.TypeSpec` of the Tensor that will be consumed/produced in the
  corresponding position of the model.

  They implement just small subset of `tf.Tensor`'s attributes and
  methods, and also overload
  the same operators as `tf.Tensor` and automatically turn them into
  Keras layers in the model.

  `KerasTensor`s are still internal-only and are a work in progress, but they
  have several advantages over using a graph `tf.Tensor` to represent
  symbolic values in functional models.
  - Unlike symbolic tensors, they do not need to refer to a graph. This means
    Keras does not need to maintain a never-deleted global background graph
    containing all layers ever called during functional model construction when
    constructing Functional Models with KerasTensors. These memory savings
    can be significant.

  - Triggering Keras functional model construction is simpler
    when it just has to check whether something is a KerasTensor, rather
    than trying to infer if a tensor was meant to be a symbolic keras
    representation or just a value produced during function tracing.

  - Autolambda layers (converting tf ops on symbolic Keras tensors to lambda
    Keras layers in the model) use TF's internal dispatching mechanism, instead
    of trying to manually walk a graph and extract nodes from it.
    The dispatching mechanism is simpler, works more reliably, and is less
    likely to run into issues with composite tensors or strange tf ops/nodes.

    (And when it fails, it's by design: because dispatch is explicitly not
    supported on the op & it's more obvious that dispatch doesn't support the
    setting).

  - Because they support arbitrary typespecs, models/layers that use
    KerasTensors are generally more friendly to composite tensors of different
    types than using symbolic graph tensors (which must have a TensorSpec and
    can't have arbitrary typespecs)

  To experiment with using KerasTensors instead of symbolic graph `tf.Tensors`,
  import keras_tensor directly and call `keras_tensor.enable_keras_tensors()`
  """

  def __init__(self, type_spec, inferred_shape_value=None, name=None):
    """Construct a KerasTensor from a type_spec and an optional name."""
    if not isinstance(type_spec, type_spec_module.TypeSpec):
      raise ValueError('KerasTensors must be constructed with a `tf.TypeSpec`.')

    self._type_spec = type_spec
    self._inferred_shape_value = inferred_shape_value
    if name is None and hasattr(type_spec, 'name'):
      name = type_spec.name
    self._name = name

  @property
  def type_spec(self):
    """Returns the `TypeSpec` that represents this Tensor."""
    return self._type_spec

  @property
  def shape(self):
    """Returns the `TensorShape` that represents the shape of the tensor."""
    # TODO(kaftan): This is only valid for normal/sparse/ragged tensors.
    # may need to raise an error when it's not valid for a type_spec,
    # but some keras code (e.g. build-related stuff) will likely fail when
    # it can't access shape or dtype
    return self._type_spec._shape  # pylint: disable=protected-access

  def get_shape(self):
    return self.shape

  @property
  def dtype(self):
    """Returns the `dtype` of elements in the tensor."""
    # TODO(kaftan): This is only valid for normal/sparse/ragged tensors.
    # may need to raise an error when it's not valid for a type_spec,
    # but some keras code (e.g. build-related stuff) will likely fail when
    # it can't access shape or dtype
    return self._type_spec._dtype  # pylint: disable=protected-access

  def ref(self):
    """Returns a hashable reference object to this KerasTensor.

    The primary use case for this API is to put KerasTensors in a
    set/dictionary. We can't put tensors in a set/dictionary as
    `tensor.__hash__()` is not available and tensor equality (`==`) is supposed
    to produce a tensor representing if the two inputs are equal.

    See the documentation of `tf.Tensor.ref()` for more info.
    """
    return object_identity.Reference(self)

  def __iter__(self):
    shape = None
    if self.shape.ndims is not None:
      shape = [dim.value for dim in self.shape.dims]

    if shape is None:
      raise TypeError('Cannot iterate over a KerasTensor with unknown shape.')
    if not shape:
      raise TypeError('Cannot iterate over a scalar.')
    if shape[0] is None:
      raise TypeError(
          'Cannot iterate over a KerasTensor with unknown first dimension.')
    return _KerasTensorIterator(self, shape[0])

  @property
  def name(self):
    """Returns the (optionally provided) name of the described tensor."""
    return self._name

  @classmethod
  def _overload_all_operators(cls):  # pylint: disable=invalid-name
    """Register overloads for all operators."""
    for operator in ops.Tensor.OVERLOADABLE_OPERATORS:
      cls._overload_operator(operator)

  @classmethod
  def _overload_operator(cls, operator):  # pylint: disable=invalid-name
    """Overload an operator with the same overloading as `ops.Tensor`.

    We pull the operator out of ops.Tensor dynamically to avoid ordering issues.

    Args:
      operator: string. The operator name.
    """
    tensor_oper = getattr(ops.Tensor, operator)

    # Compatibility with Python 2:
    # Python 2 unbound methods have type checks for the first arg,
    # so we need to extract the underlying function
    tensor_oper = getattr(tensor_oper, '__func__', tensor_oper)

    setattr(cls, operator, tensor_oper)


KerasTensor._overload_all_operators()  # pylint: disable=protected-access


class _KerasTensorIterator(object):
  """Iterates over the leading dim of a KerasTensor. Performs 0 error checks."""

  def __init__(self, tensor, dim0):
    self._tensor = tensor
    self._index = 0
    self._limit = dim0

  def __iter__(self):
    return self

  def __next__(self):
    if self._index == self._limit:
      raise StopIteration
    result = self._tensor[self._index]
    self._index += 1
    return result

  next = __next__  # python2.x compatibility.


def keras_tensor_to_placeholder(x):
  """Construct a graph placeholder to represent a KerasTensor when tracing."""
  if isinstance(x, KerasTensor):
    spec = x.type_spec

    if x._inferred_shape_value is not None:  # pylint: disable=protected-access
      # If we suspect this KerasTensor might be representing a shape tensor,
      # and we were able to extract value information with TensorFlow's shape
      # handling when making the KerasTensor, we construct the placeholder by
      # re-injecting the inferred value information into the graph.
      # Even though keras layers each trace in their own scratch
      # graph, this shape value info injection allows us to capture
      # a sizable and useful subset of the C++ shape value inference TF can do
      # if all tf ops appear in the same graph when using shape ops.
      #
      # Examples of things this cannot infer concrete dimensions for
      # that the full single-graph C++ shape inference sometimes can are:
      # * cases where the shape tensor is cast out of int32 before being
      #   manipulated w/ floating point numbers then converted back
      # * cases where int32 tensors w/ rank > 2 are manipulated before being
      #   used as a shape tensor
      inferred_shape_value = array_ops.shape(
          array_ops.placeholder(
              shape=x._inferred_shape_value, dtype=dtypes.int32))  # pylint: disable=protected-access
      if spec.shape.rank == 0:
        # `tf.shape` always returns a rank-1, we may need to turn it back to a
        # scalar.
        inferred_shape_value = inferred_shape_value[0]
      return inferred_shape_value  # pylint: disable=protected-access

    if isinstance(spec, sparse_tensor.SparseTensorSpec):
      # nest.map_structure loses dense shape information for sparse tensors.
      # So, we special-case sparse placeholder creation.
      # This only preserves shape information for top-level sparse tensors;
      # not for sparse tensors that are nested inside another composite
      # tensor.
      return array_ops.sparse_placeholder(dtype=spec.dtype, shape=spec.shape)

    def component_to_placeholder(component):
      return array_ops.placeholder(component.dtype, component.shape)

    ph = nest.map_structure(
        component_to_placeholder, spec, expand_composites=True)
    return ph
  else:
    return x


def keras_tensor_from_tensor(x):
  """Convert a traced (composite)tensor to a representative KerasTensor."""
  name = getattr(x, 'name', None)
  inferred_shape_value = None
  type_spec = type_spec_module.type_spec_from_value(x)

  if (isinstance(type_spec, tensor_spec.TensorSpec)
      and type_spec.dtype == dtypes.int32
      and type_spec.shape.rank < 2):
    # If this tensor might be representing shape information,
    # (dtype=int32, rank of 0 or 1)
    # we attempt to capture any value information tensorflow's
    # shape handling can extract from the current scratch graph.
    #
    # Even though keras layers each trace in their own scratch
    # graph, this shape value info extraction allows us to capture
    # a sizable and useful subset of the C++ shape value inference TF can do
    # if all tf ops appear in the same graph when using shape ops.
    #
    # Examples of things this cannot infer concrete dimensions for
    # that the full single-graph C++ shape inference sometimes can are:
    # * cases where the shape tensor is cast out of int32 before being
    #   manipulated w/ floating point numbers then converted back
    # * cases where int32 tensors w/ rank > 2 are manipulated before being
    #   used as a shape tensor
    inferred_shape_value = array_ops.ones(shape=x).shape
    if inferred_shape_value.dims:
      inferred_shape_value = inferred_shape_value.as_list()
    else:
      inferred_shape_value = None

  out = KerasTensor(type_spec,
                    inferred_shape_value=inferred_shape_value, name=name)
  if hasattr(x, '_keras_mask'):
    out._keras_mask = KerasTensor(  # pylint: disable=protected-access
        type_spec_module.type_spec_from_value(x._keras_mask))  # pylint: disable=protected-access

  return out
