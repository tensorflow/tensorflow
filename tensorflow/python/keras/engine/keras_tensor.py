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
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec as type_spec_module
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity

# pylint: disable=g-classes-have-attributes

_KERAS_TENSORS_ENABLED = True


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
  the layer may be able to partially infer the value of of the output in
  addition to just inferring the signature.
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

  Higher-order apis that take methods which produce tensors (e.g. `tf.while`,
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
  def _overload_all_operators(cls):  # pylint: disable=invalid-name
    """Register overloads for all operators."""
    for operator in ops.Tensor.OVERLOADABLE_OPERATORS:
      cls._overload_operator(operator)

    # We include `experimental_ref` for versions of TensorFlow that
    # still include the deprecated method in Tensors.
    if hasattr(ops.Tensor, 'experimental_ref'):
      cls._overload_operator('experimental_ref')

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
  if hasattr(x, '_user_registered_symbolic_object'):
    return x._user_registered_symbolic_object  # pylint: disable=protected-access

  if isinstance(x, KerasTensor):
    spec = x.type_spec

    if x._inferred_value is not None:  # pylint: disable=protected-access
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
      inferred_value = array_ops.shape(
          array_ops.placeholder(
              shape=x._inferred_value, dtype=dtypes.int32))  # pylint: disable=protected-access
      if spec.shape.rank == 0:
        # `tf.shape` always returns a rank-1, we may need to turn it back to a
        # scalar.
        inferred_value = inferred_value[0]
      return inferred_value  # pylint: disable=protected-access

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

# Tensorflow tensors have a maximum dimension of 254
# (See //tensorflow/core/framework/tensor_shape.h )
# So we do not try to infer values for int32 tensors larger than this,
# As they cannot represent shapes.
_MAX_TENSOR_DIMS = 254


def keras_tensor_from_tensor(x):
  """Convert a traced (composite)tensor to a representative KerasTensor."""
  name = getattr(x, 'name', None)
  inferred_value = None

  # TODO(b/161487382):
  # Special-case user-registered symbolic objects (registered by the
  # private `register_symbolic_tensor_type` method) by passing them between
  # scratch graphs directly.
  # This is needed to not break Tensorflow probability
  # while they finish migrating to composite tensors.
  user_registered_symbolic = False
  try:
    from tensorflow.python.keras.utils import tf_utils  # pylint: disable=g-import-not-at-top to prevent circular imports
    if isinstance(x, tuple(tf_utils._user_convertible_tensor_types)):  # pylint: disable=protected-access
      user_registered_symbolic = True
  except ImportError:
    pass
  if user_registered_symbolic:
    type_spec = UserRegisteredSpec(x.shape, x.dtype)
  else:
    type_spec = type_spec_module.type_spec_from_value(x)

  if (isinstance(type_spec, tensor_spec.TensorSpec)
      and type_spec.dtype == dtypes.int32
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
    # * cases where int32 tensors w/ rank > 2 are manipulated before being
    #   used as a shape tensor
    # * cases where int32 tensors too large to represent shapes are manipulated
    #   to a smaller size before being used as a shape tensor
    inferred_value = array_ops.ones(shape=x).shape
    if inferred_value.dims:
      inferred_value = inferred_value.as_list()
      if len(inferred_value) > _MAX_TENSOR_DIMS:
        inferred_value = None
    else:
      inferred_value = None

  out = KerasTensor(type_spec,
                    inferred_value=inferred_value, name=name)
  if user_registered_symbolic:
    out._user_registered_symbolic_object = x  # pylint: disable=protected-access

  if hasattr(x, '_keras_mask'):
    out._keras_mask = KerasTensor(  # pylint: disable=protected-access
        type_spec_module.type_spec_from_value(x._keras_mask))  # pylint: disable=protected-access

  return out
