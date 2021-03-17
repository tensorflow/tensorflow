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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec as type_spec_module
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_operators  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import nest

# pylint: disable=g-classes-have-attributes


# Tensorflow tensors have a maximum rank of 254
# (See `MaxDimensions()` in //tensorflow/core/framework/tensor_shape.h )
# So we do not try to infer values for int32 tensors larger than this,
# As they cannot represent shapes.
_MAX_TENSOR_RANK = 254


class KerasTensor(object):
  """A representation of a Keras in/output during Functional API construction.

  `KerasTensor`s are tensor-like objects that represent the symbolic inputs
  and outputs of Keras layers during Functional model construction. They are
  comprised of the `tf.TypeSpec` of the (Composite)Tensor that will be
  consumed/produced in the corresponding location of the Functional model.

  KerasTensors are intended as a private API, so users should never need to
  directly instantiate `KerasTensor`s.

  **Building Functional Models with KerasTensors**
  `tf.keras.Input` produces `KerasTensor`s that represent the symbolic inputs
  to your model.

  Passing a `KerasTensor` to a `tf.keras.Layer` `__call__` lets the layer know
  that you are building a Functional model. The layer __call__ will
  infer the output signature and return `KerasTensor`s with `tf.TypeSpec`s
  corresponding to the symbolic outputs of that layer call. These output
  `KerasTensor`s will have all of the internal KerasHistory metadata attached
  to them that Keras needs to construct a Functional Model.

  Currently, layers infer the output signature by:
    * creating a scratch `FuncGraph`
    * making placeholders in the scratch graph that match the input typespecs
    * Calling `layer.call` on these placeholders
    * extracting the signatures of the outputs before clearing the scratch graph

  (Note: names assigned to KerasTensors by this process are not guaranteed to
  be unique, and are subject to implementation details).

  `tf.nest` methods are used to insure all of the inputs/output data
  structures get maintained, with elements swapped between KerasTensors and
  placeholders.

  In rare cases (such as when directly manipulating shapes using Keras layers),
  the layer may be able to partially infer the value of the output in addition
  to just inferring the signature.
  When this happens, the returned KerasTensor will also contain the inferred
  value information. Follow-on layers can use this information.
  during their own output signature inference.
  E.g. if one layer produces a symbolic `KerasTensor` that the next layer uses
  as the shape of its outputs, partially knowing the value helps infer the
  output shape.

  **Automatically converting TF APIs to layers**:
  If you passing a `KerasTensor` to a TF API that supports dispatching,
  Keras will automatically turn that API call into a lambda
  layer in the Functional model, and return KerasTensors representing the
  symbolic outputs.

  Most TF APIs that take only tensors as input and produce output tensors
  will support dispatching.

  Calling a `tf.function` does not support dispatching, so you cannot pass
  `KerasTensor`s as inputs to a `tf.function`.

  Higher-order APIs that take methods which produce tensors (e.g. `tf.while`,
  `tf.map_fn`, `tf.cond`) also do not currently support dispatching. So, you
  cannot directly pass KerasTensors as inputs to these APIs either. If you
  want to use these APIs inside of a Functional model, you must put them inside
  of a custom layer.

  Args:
    type_spec: The `tf.TypeSpec` for the symbolic input created by
      `tf.keras.Input`, or symbolically inferred for the output
      during a symbolic layer `__call__`.
    inferred_value: (Optional) a non-symbolic static value, possibly partially
      specified, that could be symbolically inferred for the outputs during
      a symbolic layer `__call__`. This will generally only happen when
      grabbing and manipulating `tf.int32` shapes directly as tensors.
      Statically inferring values in this way and storing them in the
      KerasTensor allows follow-on layers to infer output signatures
      more effectively. (e.g. when using a symbolic shape tensor to later
      construct a tensor with that shape).
    name: (optional) string name for this KerasTensor. Names automatically
      generated by symbolic layer `__call__`s are not guaranteed to be unique,
      and are subject to implementation details.
  """

  def __init__(self, type_spec, inferred_value=None, name=None):
    """Constructs a KerasTensor."""
    if not isinstance(type_spec, type_spec_module.TypeSpec):
      raise ValueError('KerasTensors must be constructed with a `tf.TypeSpec`.')

    self._type_spec = type_spec
    self._inferred_value = inferred_value
    self._name = name

  @property
  def type_spec(self):
    """Returns the `tf.TypeSpec` symbolically inferred for this Keras output."""
    return self._type_spec

  @property
  def shape(self):
    """Returns the `TensorShape` symbolically inferred for this Keras output."""
    # TODO(kaftan): This is only valid for normal/sparse/ragged tensors.
    # may need to raise an error when it's not valid for a type_spec,
    # but some keras code (e.g. build-related stuff) will likely fail when
    # it can't access shape or dtype
    return self._type_spec._shape  # pylint: disable=protected-access

  @classmethod
  def from_tensor(cls, tensor):
    """Convert a traced (composite)tensor to a representative KerasTensor."""
    if isinstance(tensor, ops.Tensor):
      name = getattr(tensor, 'name', None)
      type_spec = type_spec_module.type_spec_from_value(tensor)
      inferred_value = None
      if (type_spec.dtype == dtypes.int32 and type_spec.shape.rank is not None
          and type_spec.shape.rank < 2):
        # If this tensor might be representing shape information,
        # (dtype=int32, rank of 0 or 1, not too large to represent a shape)
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
        # * cases where int32 tensors w/ rank >= 2 are manipulated before being
        #   used as a shape tensor
        # * cases where int32 tensors too large to represent shapes are
        #   manipulated to a smaller size before being used as a shape tensor
        inferred_value = array_ops.ones(shape=tensor).shape
        if inferred_value.dims:
          inferred_value = inferred_value.as_list()
          if len(inferred_value) > _MAX_TENSOR_RANK:
            inferred_value = None
        else:
          inferred_value = None

      return KerasTensor(type_spec, inferred_value=inferred_value, name=name)
    else:
      # Fallback to the generic arbitrary-typespec KerasTensor
      name = getattr(tensor, 'name', None)
      type_spec = type_spec_module.type_spec_from_value(tensor)
      return cls(type_spec, name=name)

  @classmethod
  def from_type_spec(cls, type_spec, name=None):
    return cls(type_spec=type_spec, name=name)

  def _to_placeholder(self):
    """Convert this KerasTensor to a placeholder in a graph."""
    # If there is an inferred value for this tensor, inject the inferred value
    if self._inferred_value is not None:
      # If we suspect this KerasTensor might be representing a shape tensor,
      # and we were able to extract value information with TensorFlow's shape
      # handling when making the KerasTensor, we construct the placeholder by
      # re-injecting the inferred value information into the graph. We
      # do this injection through the shape of a placeholder, because that
      # allows us to specify partially-unspecified shape values.
      #
      # See the comment on value extraction inside `from_tensor` for more info.
      inferred_value = array_ops.shape(
          array_ops.placeholder(
              shape=self._inferred_value, dtype=dtypes.int32))
      if self.type_spec.shape.rank == 0:
        # `tf.shape` always returns a rank-1, we may need to turn it back to a
        # scalar.
        inferred_value = inferred_value[0]
      return inferred_value

    # Use the generic conversion from typespec to a placeholder.
    def component_to_placeholder(component):
      return array_ops.placeholder(component.dtype, component.shape)

    return nest.map_structure(
        component_to_placeholder, self.type_spec, expand_composites=True)

  def get_shape(self):
    return self.shape

  def __len__(self):
    raise TypeError('Keras symbolic inputs/outputs do not '
                    'implement `__len__`. You may be '
                    'trying to pass Keras symbolic inputs/outputs '
                    'to a TF API that does not register dispatching, '
                    'preventing Keras from automatically '
                    'converting the API call to a lambda layer '
                    'in the Functional Model. This error will also get raised '
                    'if you try asserting a symbolic input/output directly.')

  @property
  def op(self):
    raise TypeError('Keras symbolic inputs/outputs do not '
                    'implement `op`. You may be '
                    'trying to pass Keras symbolic inputs/outputs '
                    'to a TF API that does not register dispatching, '
                    'preventing Keras from automatically '
                    'converting the API call to a lambda layer '
                    'in the Functional Model.')

  def __hash__(self):
    raise TypeError('Tensors are unhashable. (%s)'
                    'Instead, use tensor.ref() as the key.' % self)

  # Note: This enables the KerasTensor's overloaded "right" binary
  # operators to run when the left operand is an ndarray, because it
  # accords the Tensor class higher priority than an ndarray, or a
  # numpy matrix.
  # In the future explore chaning this to using numpy's __numpy_ufunc__
  # mechanism, which allows more control over how Tensors interact
  # with ndarrays.
  __array_priority__ = 100

  def __array__(self):
    raise TypeError(
        'Cannot convert a symbolic Keras input/output to a numpy array. '
        'This error may indicate that you\'re trying to pass a symbolic value '
        'to a NumPy call, which is not supported. Or, '
        'you may be trying to pass Keras symbolic inputs/outputs '
        'to a TF API that does not register dispatching, '
        'preventing Keras from automatically '
        'converting the API call to a lambda layer '
        'in the Functional Model.')

  @property
  def is_tensor_like(self):
    return True

  def set_shape(self, shape):
    """Updates the shape of this KerasTensor. Mimics `tf.Tensor.set_shape()`."""
    if not isinstance(shape, tensor_shape.TensorShape):
      shape = tensor_shape.TensorShape(shape)
    if shape.dims is not None:
      dim_list = [dim.value for dim in shape.dims]
      for dim in range(len(dim_list)):
        if dim_list[dim] is None and self.shape.dims is not None:
          dim_list[dim] = self.shape.dims[dim]
      shape = tensor_shape.TensorShape(dim_list)
    if not self.shape.is_compatible_with(shape):
      raise ValueError(
          "Keras symbolic input/output's shape %s is not"
          "compatible with supplied shape %s" %
          (self.shape, shape))
    else:
      self._type_spec._shape = shape  # pylint: disable=protected-access

  def __str__(self):
    symbolic_description = ''
    inferred_value_string = ''
    name_string = ''

    if hasattr(self, '_keras_history'):
      layer = self._keras_history.layer
      symbolic_description = (
          ', description="created by layer \'%s\'"' % (layer.name,))
    if self._inferred_value is not None:
      inferred_value_string = (
          ', inferred_value=%s' % self._inferred_value)
    if self.name is not None:
      name_string = ', name=\'%s\'' % self._name
    return 'KerasTensor(type_spec=%s%s%s%s)' % (
        self.type_spec, inferred_value_string,
        name_string, symbolic_description)

  def __repr__(self):
    symbolic_description = ''
    inferred_value_string = ''
    if isinstance(self.type_spec, tensor_spec.TensorSpec):
      type_spec_string = 'shape=%s dtype=%s' % (self.shape, self.dtype.name)
    else:
      type_spec_string = 'type_spec=%s' % self.type_spec

    if hasattr(self, '_keras_history'):
      layer = self._keras_history.layer
      symbolic_description = ' (created by layer \'%s\')' % (layer.name,)
    if self._inferred_value is not None:
      inferred_value_string = (
          ' inferred_value=%s' % self._inferred_value)
    return '<KerasTensor: %s%s%s>' % (
        type_spec_string, inferred_value_string, symbolic_description)

  @property
  def dtype(self):
    """Returns the `dtype` symbolically inferred for this Keras output."""
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
      raise TypeError('Cannot iterate over a Tensor with unknown shape.')
    if not shape:
      raise TypeError('Cannot iterate over a scalar.')
    if shape[0] is None:
      raise TypeError(
          'Cannot iterate over a Tensor with unknown first dimension.')
    return _KerasTensorIterator(self, shape[0])

  @property
  def name(self):
    """Returns the (non-unique, optional) name of this symbolic Keras value."""
    return self._name

  @classmethod
  def _overload_all_operators(cls, tensor_class):  # pylint: disable=invalid-name
    """Register overloads for all operators."""
    for operator in ops.Tensor.OVERLOADABLE_OPERATORS:
      cls._overload_operator(tensor_class, operator)

    # We include `experimental_ref` for versions of TensorFlow that
    # still include the deprecated method in Tensors.
    if hasattr(tensor_class, 'experimental_ref'):
      cls._overload_operator(tensor_class, 'experimental_ref')

  @classmethod
  def _overload_operator(cls, tensor_class, operator):  # pylint: disable=invalid-name
    """Overload an operator with the same implementation as a base Tensor class.

    We pull the operator out of the class dynamically to avoid ordering issues.

    Args:
      tensor_class: The (Composite)Tensor to get the method from.
      operator: string. The operator name.
    """
    tensor_oper = getattr(tensor_class, operator)

    # Compatibility with Python 2:
    # Python 2 unbound methods have type checks for the first arg,
    # so we need to extract the underlying function
    tensor_oper = getattr(tensor_oper, '__func__', tensor_oper)

    setattr(cls, operator, tensor_oper)


KerasTensor._overload_all_operators(ops.Tensor)  # pylint: disable=protected-access


class SparseKerasTensor(KerasTensor):
  """A specialized KerasTensor representation for `tf.sparse.SparseTensor`s.

  Specifically, it specializes the conversion to a placeholder in order
  to maintain dense shape information.
  """

  def _to_placeholder(self):
    spec = self.type_spec

    # nest.map_structure loses dense shape information for sparse tensors.
    # So, we special-case sparse placeholder creation.
    # This only preserves shape information for top-level sparse tensors;
    # not for sparse tensors that are nested inside another composite
    # tensor.
    return array_ops.sparse_placeholder(dtype=spec.dtype, shape=spec.shape)


class RaggedKerasTensor(KerasTensor):
  """A specialized KerasTensor representation for `tf.RaggedTensor`s.

  Specifically, it:

  1. Specializes the conversion to a placeholder in order
  to maintain shape information for non-ragged dimensions.
  2. Overloads the KerasTensor's operators with the RaggedTensor versions
  when they don't match the `tf.Tensor` versions
  3. Exposes some of the instance method/attribute that are unique to
  the RaggedTensor API (such as ragged_rank).
  """

  def _to_placeholder(self):
    ragged_spec = self.type_spec
    if ragged_spec.ragged_rank == 0 or ragged_spec.shape.rank is None:
      return super(RaggedKerasTensor, self)._to_placeholder()

    flat_shape = ragged_spec.shape[ragged_spec.ragged_rank:]
    result = array_ops.placeholder(ragged_spec.dtype, flat_shape)

    known_num_splits = []
    prod = 1
    for axis_size in ragged_spec.shape:
      if prod is not None:
        if axis_size is None or (
            getattr(axis_size, 'value', True) is None):
          prod = None
        else:
          prod = prod * axis_size
      known_num_splits.append(prod)

    for axis in range(ragged_spec.ragged_rank, 0, -1):
      axis_size = ragged_spec.shape[axis]
      if axis_size is None or (getattr(axis_size, 'value', True) is None):
        num_splits = known_num_splits[axis-1]
        if num_splits is not None:
          num_splits = num_splits + 1
        splits = array_ops.placeholder(
            ragged_spec.row_splits_dtype, [num_splits])
        result = ragged_tensor.RaggedTensor.from_row_splits(
            result, splits, validate=False)
      else:
        rowlen = constant_op.constant(axis_size, ragged_spec.row_splits_dtype)
        result = ragged_tensor.RaggedTensor.from_uniform_row_length(
            result, rowlen, validate=False)
    return result

  @property
  def ragged_rank(self):
    return self.type_spec.ragged_rank

RaggedKerasTensor._overload_operator(ragged_tensor.RaggedTensor, '__add__')  # pylint: disable=protected-access
RaggedKerasTensor._overload_operator(ragged_tensor.RaggedTensor, '__radd__')  # pylint: disable=protected-access
RaggedKerasTensor._overload_operator(ragged_tensor.RaggedTensor, '__mul__')  # pylint: disable=protected-access
RaggedKerasTensor._overload_operator(ragged_tensor.RaggedTensor, '__rmul__')  # pylint: disable=protected-access


# TODO(b/161487382):
# Special-case user-registered symbolic objects (registered by the
# private `register_symbolic_tensor_type` method) by passing them between
# scratch graphs directly.
# This is needed to not break Tensorflow probability
# while they finish migrating to composite tensors.
class UserRegisteredSpec(type_spec_module.TypeSpec):
  """TypeSpec to represent user-registered symbolic objects."""

  def __init__(self, shape, dtype):
    self.shape = shape
    self._dtype = dtype
    self.dtype = dtype

  def _component_specs(self):
    raise NotImplementedError

  def _from_components(self, components):
    raise NotImplementedError

  def _serialize(self):
    raise NotImplementedError

  def _to_components(self, value):
    raise NotImplementedError

  def value_type(self):
    raise NotImplementedError


# TODO(b/161487382):
# Special-case user-registered symbolic objects (registered by the
# private `register_symbolic_tensor_type` method) by passing them between
# scratch graphs directly.
# This is needed to not break Tensorflow probability
# while they finish migrating to composite tensors.
class UserRegisteredTypeKerasTensor(KerasTensor):
  """KerasTensor that represents legacy register_symbolic_tensor_type."""

  def __init__(self, user_registered_symbolic_object):
    x = user_registered_symbolic_object
    self._user_registered_symbolic_object = x
    type_spec = UserRegisteredSpec(x.shape, x.dtype)
    name = getattr(x, 'name', None)

    super(UserRegisteredTypeKerasTensor, self).__init__(type_spec, name)

  @classmethod
  def from_tensor(cls, tensor):
    return cls(tensor)

  @classmethod
  def from_type_spec(cls, type_spec, name=None):
    raise NotImplementedError('You cannot instantiate a KerasTensor '
                              'directly from TypeSpec: %s' % type_spec)

  def _to_placeholder(self):
    return self._user_registered_symbolic_object


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


# Specify the mappings of tensor class to KerasTensor class.
# This is specifically a list instead of a dict for now because
# 1. we do a check w/ isinstance because a key lookup based on class
#    would miss subclasses
# 2. a list allows us to control lookup ordering
# We include ops.Tensor -> KerasTensor in the first position as a fastpath,
# *and* include object -> KerasTensor at the end as a catch-all.
# We can re-visit these choices in the future as needed.
keras_tensor_classes = [
    (ops.Tensor, KerasTensor),
    (sparse_tensor.SparseTensor, SparseKerasTensor),
    (ragged_tensor.RaggedTensor, RaggedKerasTensor),
    (object, KerasTensor)
]


def register_keras_tensor_specialization(cls, keras_tensor_subclass):
  """Register a specialized KerasTensor subclass for a Tensor type."""
  # We always leave (object, KerasTensor) at the end as a generic fallback
  keras_tensor_classes.insert(-1, (cls, keras_tensor_subclass))


def keras_tensor_to_placeholder(x):
  """Construct a graph placeholder to represent a KerasTensor when tracing."""
  if isinstance(x, KerasTensor):
    return x._to_placeholder()  # pylint: disable=protected-access
  else:
    return x


def keras_tensor_from_tensor(tensor):
  """Convert a traced (composite)tensor to a representative KerasTensor."""
  # Create a specialized KerasTensor that supports instance methods,
  # operators, and additional value inference if possible
  keras_tensor_cls = None
  for tensor_type, cls in keras_tensor_classes:
    if isinstance(tensor, tensor_type):
      keras_tensor_cls = cls
      break

  out = keras_tensor_cls.from_tensor(tensor)

  if hasattr(tensor, '_keras_mask'):
    out._keras_mask = keras_tensor_from_tensor(tensor._keras_mask)  # pylint: disable=protected-access
  return out


def keras_tensor_from_type_spec(type_spec, name=None):
  """Convert a TypeSpec to a representative KerasTensor."""
  # Create a specialized KerasTensor that supports instance methods,
  # operators, and additional value inference if possible
  keras_tensor_cls = None
  value_type = type_spec.value_type
  for tensor_type, cls in keras_tensor_classes:
    if issubclass(value_type, tensor_type):
      keras_tensor_cls = cls
      break

  return keras_tensor_cls.from_type_spec(type_spec, name=name)
