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
"""An in-process, local XLA client in Python, supporting AOT compilation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import enum  # pylint: disable=g-bad-import-order
import inspect
import itertools
import os

import numpy as np

from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.compiler.xla.python import pywrap_xla as c_api
from tensorflow.compiler.xla.service import hlo_pb2


# Most functions are snake_case for consistency with other modules, whereas
# method names of ComputationBuilder and LocalComputation are CamelCase for
# consistency with XLA.
# pylint: disable=invalid-name


_OP_METADATA_FIELDS = [
    'op_type',
    'op_name',
    'source_file',
    'source_line',
]
OpMetadata = collections.namedtuple('OpMetadata', _OP_METADATA_FIELDS)


def OpMetadataToProto(pyobj):
  proto = xla_data_pb2.OpMetadata()
  for field in _OP_METADATA_FIELDS:
    attr = getattr(pyobj, field)
    if attr is not None:
      setattr(proto, field, attr)
  return proto


def CurrentSourceInfoMetadata(op_type=None, op_name=None, skip_frames=1):
  """Helper for use in source mapping that returns an OpMetadata object."""
  full_filename, lineno = inspect.stack()[skip_frames][1:3]
  filename = os.path.basename(full_filename)
  return OpMetadata(
      op_type=op_type,
      op_name=op_name,
      source_file=filename,
      source_line=lineno)


class PaddingType(enum.Enum):
  VALID = 1
  SAME = 2


def _convert_padding_type_to_pad_values(padding_type, lhs_dims, rhs_dims,
                                        window_strides):
  """Maps PaddingType (VALID or SAME) to pad values (list of pairs of ints)."""
  if padding_type == PaddingType.VALID:
    return [(0, 0)] * len(window_strides)

  out_shape = np.ceil(np.true_divide(lhs_dims, window_strides)).astype(int)
  pad_sizes = [max((out_size - 1) * stride + filter_size - in_size, 0)
               for out_size, stride, filter_size, in_size
               in zip(out_shape, window_strides, rhs_dims, lhs_dims)]
  return [(pad_size // 2, pad_size - pad_size // 2)
          for pad_size in pad_sizes]


_UNARY_OPS = [
    'Not',
    'Abs',
    'Exp',
    'Expm1',
    'Floor',
    'Round',
    'Ceil',
    'Log',
    'Log1p',
    'Sign',
    'Cos',
    'Sin',
    'Tanh',
    'SqrtF32',
    'SquareF32',
    'IsFinite',
    'ReciprocalF32',
    'Neg',
    'Sort',
]

_BINARY_OPS = [
    'Eq',
    'Ne',
    'Ge',
    'Gt',
    'Lt',
    'Le',
    'Add',
    'Sub',
    'Mul',
    'Div',
    'Rem',
    'Max',
    'Min',
    'And',
    'Or',
    'Xor',
    'Pow',
]


XLA_ELEMENT_TYPE_TO_DTYPE = {
    xla_data_pb2.PRED: np.dtype('bool'),
    xla_data_pb2.S8: np.dtype('int8'),
    xla_data_pb2.S16: np.dtype('int16'),
    xla_data_pb2.S32: np.dtype('int32'),
    xla_data_pb2.S64: np.dtype('int64'),
    xla_data_pb2.U8: np.dtype('uint8'),
    xla_data_pb2.U16: np.dtype('uint16'),
    xla_data_pb2.U32: np.dtype('uint32'),
    xla_data_pb2.U64: np.dtype('uint64'),
    xla_data_pb2.F16: np.dtype('float16'),
    xla_data_pb2.F32: np.dtype('float32'),
    xla_data_pb2.F64: np.dtype('float64'),
    xla_data_pb2.C64: np.dtype('complex64'),
    xla_data_pb2.TUPLE: np.dtype(np.object),
}

# Note the conversion on the key. Numpy has a known issue wherein dtype hashing
# doesn't work as expected (https://github.com/numpy/numpy/issues/7242). Thus,
# when keying by dtype in this dict, we use the string form of dtypes.
DTYPE_TO_XLA_ELEMENT_TYPE = {str(dt): et
                             for et, dt in XLA_ELEMENT_TYPE_TO_DTYPE.items()}


def dtype_to_etype(dtype):
  """Convenience function for reading DTYPE_TO_XLA_ELEMENT_TYPE."""
  return DTYPE_TO_XLA_ELEMENT_TYPE[str(np.dtype(dtype))]


class LocalBuffer(object):
  """Represents a handle to data owned by XLA.

  The referent is ready for use in executing a local, compiled
  Computation. On XLA platforms involving a device (e.g. GPU), this
  means the referent is in device memory.
  """

  def __init__(self, c_local_shaped_buffer):
    self.c_local_shaped_buffer = c_local_shaped_buffer
    self._delete = c_api.DeleteLocalShapedBuffer

  @staticmethod
  def from_pyval(pyval, layout_fn=None):
    pyval = require_numpy_array_layout(pyval)
    if layout_fn:
      shape = Shape.from_pyval(pyval)
      shape = shape.map_leaves(layout_fn)
    else:
      shape = None
    return LocalBuffer(c_api.LocalShapedBuffer.FromLiteral(pyval, shape))

  def to_py(self):
    return self.c_local_shaped_buffer.ToLiteral()

  def delete(self):
    if self.c_local_shaped_buffer is not None:
      self._delete(self.c_local_shaped_buffer)
      self.c_local_shaped_buffer = None

  def destructure(self):
    assert self.c_local_shaped_buffer is not None
    result = c_api.DestructureLocalShapedBufferTuple(self.c_local_shaped_buffer)
    self.c_local_shaped_buffer = None
    size = result.size()
    destructured = tuple(LocalBuffer(result.Release(i)) for i in xrange(size))
    return destructured

  def is_deleted(self):
    return self.c_local_shaped_buffer is None

  def __del__(self):
    self.delete()


class Shape(object):
  """Represents an XLA shape.

  A shape is either an array shape, having rank-many integer
  dimensions and an element type (represented by a Numpy dtype), or it
  is a tuple shape, having a shape for every tuple component:

    type shape =
        TupleShape of shape list
      | ArrayShape of { dimensions: int list; element_type: dtype }

  Callers are expected to instantiate this class only via the static
  constructors: tuple_shape, array_shape, and from_pyval.
  """

  @staticmethod
  def tuple_shape(tuple_shapes):
    """Construct a tuple shape."""
    if (not isinstance(tuple_shapes, (tuple, list)) or
        not all(isinstance(t, Shape) for t in tuple_shapes)):
      raise TypeError('tuple_shapes must be a tuple of Shapes')
    return Shape(tuple_shapes, tuple)

  @staticmethod
  def array_shape(element_type, dimensions, minor_to_major=None):
    """Construct an array shape."""
    if (not isinstance(dimensions, tuple) or
        not all(isinstance(i, int) for i in dimensions)):
      dimensions = tuple(int(i) for i in dimensions)
    return Shape(dimensions, np.dtype(element_type),
                 minor_to_major=minor_to_major)

  @staticmethod
  def from_pyval(pyval):
    def convert(pyval):
      if isinstance(pyval, tuple):
        return Shape.tuple_shape(tuple(convert(elt) for elt in pyval))
      else:
        pyval = require_numpy_array_layout(pyval)
        return Shape.array_shape(pyval.dtype, np.shape(pyval))
    return convert(pyval)

  def __init__(self, dimensions, dtype, minor_to_major=None):
    assert isinstance(dimensions, tuple)
    self._dimensions = dimensions
    self._dtype = dtype
    self._is_tuple = dtype == tuple
    self._minor_to_major = minor_to_major
    self._check_minor_to_major()

  def __eq__(self, other):
    # pylint: disable=protected-access
    return (self._dtype == other._dtype and
            self._dimensions == other._dimensions and
            self._minor_to_major == other._minor_to_major)

  def __ne__(self, other):
    return not self == other

  def __repr__(self):
    return ('xla_client.Shape(_dtype={!r}, _dimensions={!r}, '
            '_is_tuple={!r}, _minor_to_major={!r})').format(
                self._dtype, self._dimensions, self._is_tuple,
                self._minor_to_major)

  def is_tuple(self):
    return self._is_tuple

  def is_array(self):
    return not self._is_tuple

  def tuple_shapes(self):
    if not self.is_tuple():
      raise ValueError('not a tuple shape')
    return self._dimensions

  def numpy_dtype(self):
    """Like element_type(), but returns dtype('O') in case of a tuple shape."""
    if self.is_tuple():
      return np.dtype(np.object)
    else:
      return self.element_type()

  def xla_element_type(self):
    return DTYPE_TO_XLA_ELEMENT_TYPE[str(self.numpy_dtype())]

  def element_type(self):
    if not self.is_array():
      raise ValueError('not an array shape')
    return self._dtype

  def dimensions(self):
    if not self.is_array():
      raise ValueError('not an array shape')
    return self._dimensions

  def rank(self):
    return len(self.dimensions())

  def minor_to_major(self):
    return self._minor_to_major

  def map_leaves(self, f):
    """Map f over each leaf-level array subshape.

    Args:
      f: The function to apply. Whenever f returns None, the identity is
        applied instead.

    Returns:
      A new Shape with the mapped leaves.
    """
    if self.is_tuple():
      children = tuple(child.map_leaves(f) for child in self.tuple_shapes())
      return Shape.tuple_shape(children)
    else:
      mapped = f(self)
      return self if mapped is None else mapped

  def _check_minor_to_major(self):
    mtm = self._minor_to_major
    if self.is_tuple():
      assert mtm is None, self
    if mtm is not None:
      assert self.rank() == len(mtm), self
      assert sorted(mtm) == range(len(mtm)), self

  def update_minor_to_major(self, minor_to_major):
    if not self.is_array():
      raise ValueError('not an array shape')
    if not isinstance(minor_to_major, tuple):
      raise TypeError('minor_to_major must be a tuple')
    updated = Shape.array_shape(
        self.element_type(), self.dimensions(), minor_to_major)
    updated._check_minor_to_major()  # pylint: disable=protected-access
    return updated


def _wrap_shape(shape_info):
  dtype, dims = shape_info
  element_type = DTYPE_TO_XLA_ELEMENT_TYPE[str(dtype)]
  if element_type == xla_data_pb2.TUPLE:
    shapes = tuple(_wrap_shape(subshape_info) for subshape_info in dims)
    return Shape.tuple_shape(shapes)
  else:
    return Shape.array_shape(dtype, dims)


def require_numpy_array_layout(value):
  if isinstance(value, tuple):
    return tuple(require_numpy_array_layout(x) for x in value)
  else:
    return np.require(value, requirements=['C', 'A'])


class CompileOptions(object):
  """Python object for XLA compile options.

  These options can be passed to the 'compile' step when using a local XLA
  client.
  """

  def __init__(self):
    self.generate_hlo_graph = None
    self.dump_optimized_hlo_proto_to = None
    self.dump_unoptimized_hlo_proto_to = None
    self.dump_per_pass_hlo_proto_to = None
    self.hlo_profile = False


def transfer_to_infeed(value, replica_number=None):
  """Transfers the given value into the XLA infeed queue.

  XLA's infeed queue is a single queue that feeds the "XLA virtual machine" with
  a totally ordered stream of values. This is dequeued from XLA computations via
  the Infeed() operation.

  Args:
    value: the value that the caller would like to enqueue into the XLA infeed
      queue
    replica_number: the replica number to infeed the value to -- if not
      provided, then the default replica (trivially replica 0) is used.
  """
  if replica_number is None:
    c_api.TransferToInfeedLocal(require_numpy_array_layout(value))
  else:
    c_api.TransferToInfeedLocalReplica(
        require_numpy_array_layout(value), replica_number)


def transfer_from_outfeed(shape, replica_number=None):
  """Transfers a literal of the given shape from replica_number's outfeed.

  Args:
    shape: The shape of the value to transfer from outfeed.
    replica_number: The replica number ordinal to transfer the outfeed value
      from. (Each replica has a distinct outfeed queue.)

  Returns:
    The literal value that is produced from the outfeed queue.
  """
  return c_api.TransferFromOutfeedLocalReplica(shape, replica_number or 0)


class LocalComputation(object):
  """Python wrapper for a local XLA Computation.

  A LocalComputation can be executed if it is compiled. Otherwise, it
  can still be used as a Computation where required by the
  ComputationBuilder methods.
  """

  def __init__(self, c_local_computation, is_compiled):
    self.c_local_computation = c_local_computation
    self.is_compiled = is_compiled

    # Ensure a reference to C-based destructor for use in __del__.
    if is_compiled:
      assert isinstance(c_local_computation, c_api.CompiledLocalComputation)
      self._delete = c_api.DeleteCompiledLocalComputation
    else:
      assert isinstance(c_local_computation, c_api.LocalComputation)
      self._delete = c_api.DeleteLocalComputation

  def GetProto(self):
    """Get the HloModuleProto proto object in this local computation.

    Returns:
       An HloModuleProto proto object that has the whole-graph information.
    """

    serialized = self.c_local_computation.GetSerializedProto()
    proto = hlo_pb2.HloModuleProto.FromString(serialized)
    return proto

  def Compile(self, argument_shapes=(), compile_options=None, layout_fn=None):
    """Compiles an un-compiled local computation.

    Local computations are the result of a "LocalComputationBuild'ing" process
    -- they start in uncompiled form, and via a call to Compile() turn into a
    compiled local computation.

    Raises:
      ValueError: if this is already a compiled local computation.

    Arguments:
      argument_shapes: parameter shapes -- they are first laid out by layout_fn
        if layout_fn is provided. Otherwise, the default layout for those shapes
        will be used.
      compile_options: options to use for compilation, includes an optional
        laid out result shape for the computation.
      layout_fn: lambda that is used to lay out the argument/result shapes.

    Returns:
      A newly *compiled* local computation instance.
    """
    if self.is_compiled:
      raise ValueError('Attempt to compile a compiled local XLA computation.')

    if layout_fn:
      argument_shapes = [
          shape.map_leaves(layout_fn) for shape in argument_shapes
      ]
      result_shape = _wrap_shape(self.c_local_computation.GetReturnValueShape())
      result_shape = result_shape.map_leaves(layout_fn)
      compile_options = compile_options or CompileOptions()
      compile_options.result_shape = result_shape
    return LocalComputation(
        self.c_local_computation.Compile(argument_shapes, compile_options),
        is_compiled=True)

  def CompileWithExampleArguments(self,
                                  arguments=(),
                                  compile_options=None,
                                  layout_fn=None):
    return self.Compile(
        argument_shapes=[Shape.from_pyval(arg) for arg in arguments],
        compile_options=compile_options,
        layout_fn=layout_fn)

  def Execute(self, arguments=(), layout_fn=None):
    """Execute with Python values as arguments and return value."""
    if not self.is_compiled:
      raise ValueError('Cannot execute an uncompiled local XLA computation.')
    argument_shapes = [Shape.from_pyval(arg) for arg in arguments]
    if layout_fn:
      argument_shapes = [
          shape.map_leaves(layout_fn) for shape in argument_shapes
      ]
    else:
      argument_shapes = [None for shape in argument_shapes]
    arguments = tuple(map(require_numpy_array_layout, arguments))
    return self.c_local_computation.Execute(arguments, argument_shapes)

  def ExecuteWithLocalBuffers(self, arguments=()):
    """Execute with LocalBuffer arguments and return value."""
    if not self.is_compiled:
      raise ValueError('Cannot execute an uncompiled local XLA computation.')
    arguments = tuple(arguments)
    if any(arg.is_deleted() for arg in arguments):
      raise ValueError('Executing with deleted local buffer argument')
    return LocalBuffer(
        self.c_local_computation.ExecuteWithShapedBuffers(
            [arg.c_local_shaped_buffer for arg in arguments]))

  def __del__(self):
    self._delete(self.c_local_computation)


class ComputationBuilder(object):
  """XLA computation builder.

  Enqueues XLA ops in sequence and in order to build a
  LocalComputation, which in turn can be compiled into a
  CompiledLocalComputation, which in turn can be locally executed.
  """

  # The methods of this class map 1-to-1 onto the XLA C++
  # computation builder API. Therefore, there's no need to laboriously list
  # arguments and return values for every method, especially where it's obvious.
  #
  # pylint: disable=g-doc-return-or-yield
  # pylint: disable=g-doc-args

  def __init__(self, name):
    self._client = c_api.LocalComputationBuilder(name.encode('utf8'))
    self._parameter_numbering = itertools.count()

  def Build(self):
    return LocalComputation(self._client.Build(), is_compiled=False)

  def SetOpMetadata(self, op_metadata):
    """Set metadata for operations that are about to be enqueued."""
    self._client.SetOpMetadata(op_metadata)

  def ClearOpMetadata(self):
    """Clear metadata for operations that are about to be enqueued."""
    self._client.ClearOpMetadata()

  def Infeed(self, shape):
    """Enqueues an infeed op onto the computation.

    Infeed operations dequeue data of the given shape from the device's infeed
    queue for subsequent use in the computation.

    Returns:
      A LocalOp.
    """
    return self._client.Infeed(shape)

  def Outfeed(self, operand):
    """Enqueues an outfeed op onto the computation.

    Outfeed operations enqueue data, using the given operand, onto the XLA
    outfeed queue for subsequent dequeue via the client API.
    """
    self._client.Outfeed(operand, self.GetShape(operand), ''.encode('utf-8'))

  def Constant(self, value):
    """Enqueues a constant op onto the computation.

    Args:
      value: value for the constant, as a np.array with an explicit dtype set
             to one of the supported types.

    Returns:
      A LocalOp.
    """
    value = require_numpy_array_layout(value)
    return self._client.ConstantLiteral(value)

  def ConstantF32Scalar(self, value):
    """Convenience method to enqueue a scalar F32 constant op.

    Args:
      value: a floating-point number.

    Returns:
      A LocalOp.
    """
    return self.Constant(np.array(value, dtype=np.float32))

  def ConstantF64Scalar(self, value):
    """Convenience method to enqueue a scalar F32 constant op.

    Args:
      value: a floating-point number.

    Returns:
      A LocalOp.
    """
    return self.Constant(np.array(value, dtype=np.float64))

  def ConstantS32Scalar(self, value):
    """Convenience method to enqueue a scalar S32 constant op.

    Args:
      value: a floating-point number.

    Returns:
      A LocalOp.
    """
    return self.Constant(np.array(value, dtype=np.int32))

  def ConstantS64Scalar(self, value):
    """Convenience method to enqueue a scalar S64 constant op.

    Args:
      value: a floating-point number.

    Returns:
      A LocalOp.
    """
    return self.Constant(np.array(value, dtype=np.int64))

  def ConstantPredScalar(self, value):
    """Convenience method to enqueue a scalar PRED constant op.

    Args:
      value: a boolean value.

    Returns:
      A LocalOp.
    """
    return self.Constant(np.array(value, dtype=np.bool))

  def ParameterWithShape(self, shape, name=None, parameter_num=None):
    """Enqueues a Parameter op onto the computation, given a shape.

    Args:
      shape: the parameter's shape as a Shape object.
      name: optional string name for the parameter.
      parameter_num: parameter number in the computation function. If None,
        the next linear parameter number is used. The default value capability
        can be used for auto-numbering. If you're using auto-numbering for some
        parameters, use it for *all* parameters to avoid clashes.

    Returns:
      A LocalOp.
    """
    if name is None:
      name = ''
    if parameter_num is None:
      parameter_num = next(self._parameter_numbering)

    return self._client.Parameter(parameter_num, shape, name.encode('utf8'))

  def ParameterFromNumpy(self, value, name=None, parameter_num=None):
    """Enqueues a Parameter op onto the computation.

    Args:
      value: a Numpy array, or a nested tuple thereof, from which the
        shape is inferred.
      name: as in ParameterWithShape.
      parameter_num: as in ParameterWithShape.

    Returns:
      A LocalOp.
    """
    return self.ParameterWithShape(
        Shape.from_pyval(value), name=name, parameter_num=parameter_num)

  def Broadcast(self, operand, sizes):
    """Enqueues a broadcast operation onto the computation.

    Args:
      operand: the operand LocalOp to broadcast.
      sizes: an iterable of broadcast sizes.

    Returns:
      A LocalOp representing the added broadcast op.
    """
    return self._client.Broadcast(operand, sizes)

  def Concatenate(self, operands, dimension):
    """Enqueues a concatenate operation onto the computation.

    Args:
      operands: the operands to concatenate.
      dimension: the dimension in which to perform the concatenation.

    Returns:
      A LocalOp representing the added concatenate op.
    """
    return self._client.ConcatInDim(operands, dimension)

  def ConvertElementType(self, operand, new_element_type):
    """Enqueues an element type conversion operation onto the computation.

    Args:
      operand: the operand to convert.
      new_element_type: the target primitive type.

    Returns:
      A LocalOp representing the added conversion op.
    """
    return self._client.ConvertElementType(operand, new_element_type)

  def GetShape(self, operand):
    return _wrap_shape(self._client.GetShape(operand))

  def GetReturnValueShape(self):
    return _wrap_shape(self._client.GetReturnValueShape())

  def GetComputationStats(self):
    raise NotImplementedError()

  def Pad(self, operand, padding_value, padding_config):
    """Enqueues a Pad operation onto the computation.

    Args:
      operand: LocalOp representing the array to pad.
      padding_value: LocalOp representing the scalar pad value.
      padding_config: either an xla_data_pb2.PaddingConfig or a list of integer
        triples (edge_padding_low, edge_padding_high, interior_padding)
        representing the configuration of the padding operation.

    Returns:
      A LocalOp representing the added Pad op.
    """
    if not isinstance(padding_config, xla_data_pb2.PaddingConfig):
      padding_config = GetPaddingConfigFromTriples(padding_config)
    return self._client.Pad(operand, padding_value, padding_config)

  def Reshape(self, operand, dimensions, new_sizes):
    """Enqueues a reshape op onto the computation.

    Args:
      operand: LocalOp representing the array to be reshaped.
      dimensions: sequence of integers encoding the order in which dimensions
        are collapsed or None, in which case dimensions are flattened in order.
      new_sizes: sequence of integers encoding the new dimension sizes (shape).

    Returns:
      A LocalOp representing the added Reshape op.
    """
    if dimensions is None:
      ndim = len(self.GetShape(operand).dimensions())
      dimensions = tuple(range(ndim))
    return self._client.Reshape(operand, dimensions, new_sizes)

  def CrossReplicaSum(self, operand):
    """CrossReplicaSum op.

    Args:
      operand: the operand to sum across replica instances.

    Returns:
      A LocalOp that has the sum of the value among all replicas.
    """
    return self._client.CrossReplicaSum(operand)

  def Collapse(self, operand, dimensions):
    """Collapse op."""
    return self._client.Collapse(operand, dimensions)

  def Trans(self, operand):
    """Specialized matrix transpose op."""
    return self._client.Transpose(operand, [1, 0])

  def Transpose(self, operand, permutation):
    """Transpose op."""
    return self._client.Transpose(operand, permutation)

  def Rev(self, operand, dimensions):
    """Rev op."""
    return self._client.Rev(operand, dimensions)

  def Clamp(self, min, operand, max):  # pylint: disable=redefined-builtin
    """Clamp op."""
    return self._client.Clamp(min, operand, max)

  def SelectAndScatter(self, operand, select, window_dimensions, window_strides,
                       padding, source, init_value, scatter):
    """Select and scatter op, used by the gradient of ReduceWindow.

    Args:
      operand: LocalOp for array of dimension N and type T over
        which the windows slide.
      select: Computation of type (T, T) -> Pred to apply to the elements of
        each window to indicate which element is selected.
      window_dimensions: sequence of N integers for dimensions of the window.
      window_strides: sequence of N integers for the strides of the window.
      padding: PaddingType representing either 'SAME' or 'VALID ' padding.
      source: LocalOp for array of type T with values to scatter.
      init_value: LocalOp of scalar type T for initial out value.
      scatter: Computation of type (T, T) -> T to apply to each scatter source
        element with its destination element.

    Returns:
      A LocalOp representing the added SelectAndScatter op.
    """
    pads = _convert_padding_type_to_pad_values(
        padding, self.GetShape(operand).dimensions(),
        window_dimensions, window_strides)
    return self._client.SelectAndScatterWithGeneralPadding(
        operand, select.c_local_computation, window_dimensions, window_strides,
        pads, source, init_value, scatter.c_local_computation)

  def Select(self, pred, on_true, on_false):
    """Element-wise selection op.

    Constructs an output array from elements of two input arrays, based on the
    values of a predicate array.
    """
    return self._client.Select(pred, on_true, on_false)

  def Slice(self, operand, start_indices, limit_indices, strides=None):
    """Enqueues a slice operation onto the computation.

    Args:
      operand: LocalOp for the N dimensional array to be sliced.
      start_indices: iterable of N integers containing the starting indices of
        the slice for each dimension.
      limit_indices: iterable of N integers containing the ending indices
        (exclusive) of the slice for each dimension.
      strides: optional iterable of N integers containing the stride sizes for
        each dimension.

    Returns:
      A LocalOp representing the added Slice op.
    """
    if strides is None:
      start_indices = list(start_indices)
      strides = [1] * len(start_indices)
    return self._client.Slice(operand, start_indices, limit_indices, strides)

  def SliceInDim(self, operand, start_index, limit_index, stride, dimno):
    """Enqueues a slice-in-dimension operation onto the computation.

    Args:
      operand: LocalOp for the N dimensional array to be sliced.
      start_index: an integer containing the start index of the slice.
      limit_index: an integer containing the end index of the slice.
      stride: an integer containing the stride size for the slice.
      dimno: an integer indicating the dimension along which to slice.

    Returns:
      A LocalOp representing the added Slice op.
    """
    return self._client.SliceInDim(operand, start_index, limit_index, stride,
                                   dimno)

  def DynamicSlice(self, operand, start_indices, slice_sizes):
    """Enqueues a slice op with dynamic start indices onto the computation.

    Args:
      operand: LocalOp for the N dimensional array to be sliced.
      start_indices: LocalOp for the 1D array of N integers
        containing the starting indices of the slice.
      slice_sizes: iterable of N integers containing the slice sizes in each
        dimension.

    Returns:
      A LocalOp representing the added DynamicSlice op.
    """
    return self._client.DynamicSlice(operand, start_indices, slice_sizes)

  def DynamicUpdateSlice(self, operand, update, start_indices):
    """Enqueues a dynamic update slice operation onto the computation.

    Args:
      operand: LocalOp for the N dimensional array to be updated.
      update: N dimensional array comprising the slice update.
      start_indices: Rank-1 array of N integers comprising the starting indices
        of the slice along each dimension.
    Returns:
      A LocalOp representing the added DynamicUpdateSlice op.
    """
    return self._client.DynamicUpdateSlice(operand, update, start_indices)

  def Tuple(self, *ops):
    """Enqueues a tuple operation onto the computation.

    Args:
      ops: a sequence of tuple operands (each a LocalOp).

    Returns:
      A LocalOp representing the added Tuple op.
    """
    return self._client.Tuple(ops)

  def GetTupleElement(self, tup, index):
    """Enqueues a 'get tuple element' operation onto the computation.

    Args:
      tup: the tuple operand (a LocalOp).
      index: numeric index to select from the tuple.

    Returns:
      A LocalOp representing the added GetTupleElement op.
    """
    return self._client.GetTupleElement(tup, index)

  def Call(self, computation_to_apply, operands):
    """Enqueues a call operation onto the computation.

    Args:
      computation_to_apply: a Computation object.
      operands: an iterable of LocalOp. The number and types of
        operands must match the arity of computation_to_apply.

    Returns:
      A LocalOp representing the added call op.
    """
    return self._client.Call(computation_to_apply.c_local_computation, operands)

  def Map(self, operands, computation_to_apply, dimensions):
    """Enqueues a map operation onto the computation.

    Args:
      operands: an iterable of LocalOp.
      computation_to_apply: a Computation object.
      dimensions: dimensions over which to apply map the function.

    Returns:
      A LocalOp representing the added Map op.
    """
    return self._client.Map(operands, computation_to_apply.c_local_computation,
                            dimensions)

  def Reduce(self, operand, init_value, computation_to_apply, dimensions):
    """Enqueues a reduction operation onto the computation.

    Args:
      operand: reduction operand (LocalOp).
      init_value: reduction initial value (LocalOp).
      computation_to_apply: a Computation object - binary reduction function.
      dimensions: sequence of dimensions (integers) to reduce on.

    Returns:
      A LocalOp representing the added Reduce op.
    """
    return self._client.Reduce(operand, init_value,
                               computation_to_apply.c_local_computation,
                               dimensions)

  def ReduceWindow(self, operand, init_value, computation_to_apply,
                   window_dimensions, window_strides, padding):
    """Enqueues a windowed reduction operation onto the computation.

    Args:
      operand: reduction operand (LocalOp).
      init_value: reduction initial value (LocalOp).
      computation_to_apply: a binary reduction function (Computation).
      window_dimensions: dimensions of window (sequence of integers).
      window_strides: strides for window (sequence of integers).
      padding: PaddingType representing either 'SAME' or 'VALID' padding.

    Returns:
      A LocalOp representing the added ReduceWindow op.
    """
    pads = _convert_padding_type_to_pad_values(
        padding, self.GetShape(operand).dimensions(), window_dimensions,
        window_strides)
    return self._client.ReduceWindowWithGeneralPadding(
        operand, init_value, computation_to_apply.c_local_computation,
        window_dimensions, window_strides, pads)

  def RngNormal(self, mu, sigma, dims):
    """Enqueues an RngNormal operation onto the computation.

    Args:
      mu: A LocalOp to an F32 scalar specifying the mean.
      sigma: A LocalOp to an F32 scalar specifying the standard
        deviation.
      dims: A 1D array-like of nonnegative integers specifying the dimensions.

    Returns: a LocalOp to the generated array of F32 values.
    """
    shape = Shape.array_shape(self.GetShape(mu).element_type(), dims)
    return self._client.RngNormal(mu, sigma, shape)

  def RngUniform(self, a, b, dims):
    """Enqueues an RngUniform operation onto the computation.

    Args:
      a: a LocalOp to an F32, S32, or U32 scalar (consistent with
        the type of b) specifying the low end of the interval [a, b) over which
        values are generated.
      b: a LocalOp to an F32, S32, or U32 scalar (consistent with
        the type of a) specifying the high end of the interval [a, b) over which
        values are generated.
      dims: A 1D array-like of nonnegative integers specifying the dimensions.

    Returns: a LocalOp to the generated array of values with the
      same numeric type (F32, S32, or U32) as the arguments a and b.
    """
    shape = Shape.array_shape(self.GetShape(a).element_type(), dims)
    return self._client.RngUniform(a, b, shape)

  def While(self, cond, body, init):
    """Enqueues a While operation onto the computation.

    Args:
      cond: a Computation for the loop condition, which has type T -> PRED
      body: a Computation for the loop body, which has type T -> T
      init: a LocalOp for the initial parameter, which has type T

    Returns: a LocalOp representing the While operation.
    """
    return self._client.While(cond.c_local_computation,
                              body.c_local_computation, init)

  def Conditional(self, pred, true_operand, true_computation, false_operand,
                  false_computation):
    """Enqueues a Conditional operation onto the computation.

    Args:
      predicate: a LocalOp to test, which has scalar type PRED
      true_operand: a LocalOp of type T_0
      true_computation: a Computation to apply to true_operand, type T_0 -> S
      false_operand: a ComputationDatahandle of type T_1
      false_computation: a Computation to apply to false_operand, type T_1 -> S

    Returns: a LocalOp representing the Conditional operation.
    """
    return self._client.Conditional(
        pred, true_operand, true_computation.c_local_computation, false_operand,
        false_computation.c_local_computation)

  def IsConstant(self, operand):
    """Checks whether the given operand is a compile-time constant.

    Args:
      operand: a ComputationDataHandle to test.

    Returns: bool indicating whether `operand` is a compile-time constant,
      meaning its value does not depend on any parametersor, or on stateful
      operators such as `RngNormal` or `Infeed`.
    """
    return self._client.IsConstant(operand)

  def BuildConstantSubGraph(self, operand):
    """Builds a constant sub graph.

    Args:
      operand: a LocalOp to test.
    Returns: a LocalComputation that is rooted on the given `operand` which is a
      compile-time constant.
    """
    return self._client.BuildConstantSubGraph(operand)

  def Dot(self, lhs, rhs):
    """Enqueues a dot operation onto the computation.

    Args:
      lhs: LocalOp for the rank 1 or rank 2 left-hand-side array.
      rhs: LocalOp for the rank 1 or rank 2 right-hand-side array.

    Returns: a LocalOp representing the Dot operation.
    """
    return self._client.Dot(lhs, rhs)

  def DotGeneral(self, lhs, rhs, dimension_numbers):
    """Enqueues a general dot operation onto the computation.

    Args:
      lhs: LocalOp for the left-hand-side array.
      rhs: LocalOp for the right-hand-side array.
      dimension_numbers: either an xla_data_pb2.DotDimensionNumbers or a nested
        tuple ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) of lists of
        integers representing the dimensions to treat as contracting dimensions
        and batch dimensions on each input operand.

    Returns: a LocalOp representing the DotGeneral operation.
    """
    if not isinstance(dimension_numbers, xla_data_pb2.DotDimensionNumbers):
      dimension_numbers = GetDotDimensionsFromLists(dimension_numbers)
    return self._client.DotGeneral(lhs, rhs, dimension_numbers)

  def Conv(self, lhs, rhs, window_strides, padding):
    """Enqueues a Conv operation onto the computation.

    Args:
      lhs: LocalOp for the rank N+2 array of inputs.
      rhs: LocalOp for the rank N+2 array of kernel weights.
      window_strides: length-N array-like of integer kernel strides.
      padding: PaddingType representing either 'SAME' or 'VALID' padding.

    Returns: a LocalOp representing the Conv operation.
    """
    pads = _convert_padding_type_to_pad_values(
        padding, self.GetShape(lhs).dimensions()[2:],
        self.GetShape(rhs).dimensions()[2:], window_strides)
    dimension_numbers = self._GetConvDimensionNumbers(len(window_strides))
    return self._client.ConvGeneralDilated(lhs, rhs, window_strides, pads, (),
                                           (), dimension_numbers)

  def ConvWithGeneralPadding(self, lhs, rhs, window_strides, padding,
                             lhs_dilation, rhs_dilation):
    """Enqueues a ConvWithGeneralPadding operation onto the computation.

    Args:
      lhs: LocalOp for the rank N+2 array of inputs.
      rhs: LocalOp for the rank N+2 array of kernel weights.
      window_strides: length-N array-like of kernel strides.
      padding: length-N array-like of pairs of integers of (low, high) padding.
      lhs_dilation: length-N array-like of dilation factors.
      rhs_dilation: length-N array-like of dilation factors.

    Returns:
      A ComputationdataHandle representing the added ConvWithGeneralPadding op.
    """
    dimension_numbers = self._GetConvDimensionNumbers(len(window_strides))
    return self._client.ConvGeneralDilated(lhs, rhs, window_strides, padding,
                                           lhs_dilation, rhs_dilation,
                                           dimension_numbers)

  def _GetConvDimensionNumbers(self, num_spatial_dims):
    """Create ConvolutionDimensionNumbers proto for convolutions."""
    nd = num_spatial_dims
    dimension_numbers = xla_data_pb2.ConvolutionDimensionNumbers()
    dimension_numbers.input_batch_dimension = 0
    dimension_numbers.input_feature_dimension = 1
    dimension_numbers.output_batch_dimension = 0
    dimension_numbers.output_feature_dimension = 1
    dimension_numbers.kernel_output_feature_dimension = 0
    dimension_numbers.kernel_input_feature_dimension = 1
    dimension_numbers.input_spatial_dimensions.extend(range(2, 2 + nd))
    dimension_numbers.kernel_spatial_dimensions.extend(range(2, 2 + nd))
    dimension_numbers.output_spatial_dimensions.extend(range(2, 2 + nd))
    return dimension_numbers

  def ConvGeneralDilated(self, lhs, rhs, window_strides, padding, lhs_dilation,
                         rhs_dilation, dimension_numbers):
    """Enqueues a ConvGeneralDilated operation onto the computation.

    Args:
      lhs: LocalOp for the rank N+2 array of inputs.
      rhs: LocalOp for the rank N+2 array of kernel weights.
      window_strides: length-N array-like of integer kernel strides.
      padding: length-N array-like of pairs of integers of (low, high) padding.
      lhs_dilation: length-N array-like of integer dilation factors.
      rhs_dilation: length-N array-like of integer dilation factors.
      dimension_numbers: either an xla_data_pb2.ConvolutionDimensionNumbers or a
        triple (lhs_spec, rhs_spec, out_spec) where each element is a string of
        length N+2 identifying by position (1) batch dimensions in lhs, rhs, and
        the output with the character 'N', (2) feature dimensions in lhs and the
        output with the character 'C', (3) input and output feature dimensions
        in rhs with the characters 'I' and 'O' respectively, and (4) spatial
        dimension correspondences between lhs, rhs, and the output using any
        distinct characters. For example, to indicate dimension numbers
        consistent with the Conv operation with two spatial dimensions, one
        could use ('NCHW', 'OIHW', 'NCHW'). As another example, to indicate
        dimension numbers consistent with the TensorFlow Conv2D operation, one
        could use ('NHWC', 'HWIO', 'NHWC'). When using the latter form of
        convolution dimension specification, window strides are associated with
        spatial dimension character labels according to the order in which the
        labels appear in the rhs_spec string, so that window_strides[0] is
        matched with the dimension corresponding to the first character
        appearing in rhs_spec that is not 'I' or 'O'.

    Returns: a LocalOp representing the ConvGenralDilated operation.
    """
    if not isinstance(dimension_numbers,
                      xla_data_pb2.ConvolutionDimensionNumbers):
      lhs_spec, rhs_spec, out_spec = dimension_numbers
      dimension_numbers = xla_data_pb2.ConvolutionDimensionNumbers()

      dimension_numbers.input_batch_dimension = lhs_spec.index('N')
      dimension_numbers.input_feature_dimension = lhs_spec.index('C')
      dimension_numbers.output_batch_dimension = out_spec.index('N')
      dimension_numbers.output_feature_dimension = out_spec.index('C')
      dimension_numbers.kernel_output_feature_dimension = rhs_spec.index('O')
      dimension_numbers.kernel_input_feature_dimension = rhs_spec.index('I')

      dimension_numbers.kernel_spatial_dimensions.extend(
          i for i, c in enumerate(rhs_spec) if c not in {'I', 'O'})
      dimension_numbers.input_spatial_dimensions.extend(
          sorted((i for i, c in enumerate(lhs_spec) if c not in {'N', 'C'}),
                 key=lambda i: rhs_spec.index(lhs_spec[i])))
      dimension_numbers.output_spatial_dimensions.extend(
          sorted((i for i, c in enumerate(out_spec) if c not in {'N', 'C'}),
                 key=lambda i: rhs_spec.index(out_spec[i])))
    return self._client.ConvGeneralDilated(lhs, rhs, window_strides, padding,
                                           lhs_dilation, rhs_dilation,
                                           dimension_numbers)


def _forward_methods_to_local_builder():
  """Forward remaining ComputationBuilder methods to the C API.

  Set up methods, corresponding to unary and binary XLA operations,
  whose calls are forwarded in a boilerplate manner to the underlying
  LocalComputationBuilder C-extension API.
  """

  def forward_to_local_builder_with_handles(target_method, is_binop=False):
    """Generate a forwarding method that wraps/unwraps data handles."""

    def forward(self, *args, **kwargs):
      arg_list = list(args)

      if is_binop and len(arg_list) < 3:
        arg_list.append(kwargs.get('broadcast_dimensions', ()))

      return target_method(
          self._client,  # pylint: disable=protected-access
          *arg_list)

    return forward

  for method_name in _UNARY_OPS:
    forward = forward_to_local_builder_with_handles(
        getattr(c_api.LocalComputationBuilder, method_name))
    forward.__name__ = method_name
    setattr(ComputationBuilder, method_name, forward)

  for method_name in _BINARY_OPS:
    forward = forward_to_local_builder_with_handles(
        getattr(c_api.LocalComputationBuilder, method_name), is_binop=True)
    forward.__name__ = method_name
    setattr(ComputationBuilder, method_name, forward)


_forward_methods_to_local_builder()


def initialize_replica_count(replica_count):
  """Initializes the desired replica count to use on XLA service init.

  Args:
    replica_count: number of replicas that are desired for set up during XLA
      initialization.

  Raises:
    A runtime exception if the XLA service has already been initialized.
  """
  c_api.InitializeReplicaCount(replica_count)


def get_replica_count():
  """Returns the current replica count used for the XLA service.

  Note: this will return a value whether the XLA service has been initialized
  yet or not.
  """
  return c_api.GetReplicaCount()


def GetPaddingConfigFromTriples(triples):
  """Create PaddingConfig proto from list of triples of integers."""
  padding_config = xla_data_pb2.PaddingConfig()
  for lo, hi, interior in triples:
    dimension = padding_config.dimensions.add()
    dimension.edge_padding_low = lo
    dimension.edge_padding_high = hi
    dimension.interior_padding = interior
  return padding_config


def GetDotDimensionsFromLists(dimension_numbers):
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  dot_dims_proto = xla_data_pb2.DotDimensionNumbers()
  dot_dims_proto.lhs_contracting_dimensions.extend(lhs_contract)
  dot_dims_proto.rhs_contracting_dimensions.extend(rhs_contract)
  dot_dims_proto.lhs_batch_dimensions.extend(lhs_batch)
  dot_dims_proto.rhs_batch_dimensions.extend(rhs_batch)
  return dot_dims_proto
