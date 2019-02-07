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
"""An XLA client in Python, supporting AOT compilation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import enum  # pylint: disable=g-bad-import-order
import inspect
import itertools
import os

import numpy as np

import six
from six.moves import xrange

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


@six.add_metaclass(abc.ABCMeta)
class Backend(object):
  """Abstract base class for XLA backends."""

  @abc.abstractmethod
  def buffer_from_pyval(self, pyval, device=0):
    """Allocates a fresh buffer and populates it with `pyval`."""

  @abc.abstractmethod
  def delete_buffer(self, c_buffer):
    """Deletes buffer `c_buffer`."""

  @abc.abstractmethod
  def destructure_tuple(self, c_buffer):
    """Destructures a tuple buffer into a sequence of buffers."""

  @abc.abstractmethod
  def compile(self, computation, argument_shapes, compile_options):
    """Compiles a computation. Returns an executable."""

  @abc.abstractmethod
  def delete_executable(self, executable):
    """Deletes an executable."""

  @abc.abstractmethod
  def execute(self, executable, args):
    """Runs an executable without replication."""

  @abc.abstractmethod
  def execute_replicated(self, executable, per_replica_args):
    """Runs an executable in a replicated manner."""


class XlaLocalBackend(Backend):
  """XLA backend implemented using the in-process xla::LocalClient API."""

  def buffer_from_pyval(self, pyval, device=0):
    return c_api.LocalShapedBuffer.FromLiteral(pyval, None, device)

  def delete_buffer(self, c_buffer):
    c_api.DeleteLocalShapedBuffer(c_buffer)

  def destructure_tuple(self, c_buffer):
    result = c_api.DestructureLocalShapedBufferTuple(c_buffer)
    return [result.Release(i) for i in xrange(result.size())]

  def compile(self, c_computation, argument_shapes, compile_options):
    return c_computation.Compile(argument_shapes, compile_options)

  def delete_executable(self, executable):
    assert isinstance(executable, c_api.CompiledLocalComputation)
    c_api.DeleteCompiledLocalComputation(executable)

  def execute(self, executable, args):
    return executable.Execute(args)

  def execute_replicated(self, executable, per_replica_args):
    output_buffer_tup = executable.ExecutePerReplica(per_replica_args)
    size = output_buffer_tup.size()
    return [output_buffer_tup.Release(i) for i in xrange(size)]


class XrtBackend(Backend):
  """XLA backend implemented using XRT."""

  def __init__(self, target):
    self.target = target

  def buffer_from_pyval(self, pyval, device=0):
    if device != 0:
      raise NotImplementedError(
          'Multi-replica execution is not yet supported via the XRT backend.')
    return c_api.XrtAllocation.FromLiteral(pyval,
                                           _maybe_encode_string(self.target))

  def delete_buffer(self, c_buffer):
    c_api.DeleteXrtAllocation(c_buffer)

  def destructure_tuple(self, c_buffer):
    result = c_api.DestructureXrtAllocationTuple(
        c_buffer, _maybe_encode_string(self.target))
    return [result.Release(i) for i in xrange(result.size())]

  def compile(self, c_computation, argument_shapes, compile_options):
    return c_computation.CompileForXrt(argument_shapes,
                                       _maybe_encode_string(self.target))

  def delete_executable(self, executable):
    assert isinstance(executable, c_api.CompiledXrtComputation)
    c_api.DeleteCompiledXrtComputation(executable)

  def execute(self, executable, args):
    return executable.Execute(args)

  def execute_replicated(self, executable, per_replica_args):
    if len(per_replica_args) != 1:
      raise NotImplementedError(
          'Multi-replica execution is not yet supported via the XRT backend.')
    return [executable.Execute(per_replica_args[0])]


XLA_LOCAL_BACKEND = XlaLocalBackend()


class BackendType(enum.Enum):
  XLA_LOCAL = 1
  XRT = 2


def BackendSpec(backend, target):
  """Compatibility wrapper to support older clients. Do not use in new code."""
  if backend == BackendType.XLA_LOCAL:
    return XLA_LOCAL_BACKEND
  elif backend == BackendType.XRT:
    return XrtBackend(target)
  else:
    raise ValueError('Unknown backend {}'.format(backend))


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


def _maybe_encode_string(s):
  if six.PY3:
    return s.encode('utf-8')
  else:
    return s


class PaddingType(enum.Enum):
  VALID = 1
  SAME = 2


def _convert_padding_type_to_pad_values(padding_type, lhs_dims, rhs_dims,
                                        window_strides):
  """Maps PaddingType or string to pad values (list of pairs of ints)."""
  if not isinstance(padding_type, (str, PaddingType)):
    msg = 'padding_type must be str or PaddingType, got {}.'
    raise TypeError(msg.format(type(padding_type)))

  if isinstance(padding_type, str):
    if padding_type.upper() == 'VALID':
      padding_type = PaddingType.VALID
    elif padding_type.upper() == 'SAME':
      padding_type = PaddingType.SAME
    else:
      msg = 'Unknown padding type string: expected "VALID" or "SAME", got {}.'
      raise ValueError(msg.format(padding_type))

  if padding_type == PaddingType.VALID:
    return [(0, 0)] * len(window_strides)
  elif padding_type == PaddingType.SAME:
    out_shape = np.ceil(np.true_divide(lhs_dims, window_strides)).astype(int)
    pad_sizes = [max((out_size - 1) * stride + filter_size - in_size, 0)
                 for out_size, stride, filter_size, in_size
                 in zip(out_shape, window_strides, rhs_dims, lhs_dims)]
    return [(pad_size // 2, pad_size - pad_size // 2)
            for pad_size in pad_sizes]
  else:
    msg = 'Unexpected PaddingType value: {}'
    raise ValueError(msg.format(padding_type))


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
    'IsFinite',
    'Sqrt',
    'Rsqrt',
    'Square',
    'Reciprocal',
    'Neg',
    'Erf',
    'Erfc',
    'ErfInv',
    'Lgamma',
    'Digamma',
    'Acos',
    'Asin',
    'Atan',
    'Tan',
    'Acosh',
    'Asinh',
    'Atanh',
    'Cosh',
    'Sinh',
    'Real',
    'Imag',
    'Conj',
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
    'ShiftLeft',
    'ShiftRightArithmetic',
    'ShiftRightLogical',
    'Atan2',
    'Complex',
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
    xla_data_pb2.C128: np.dtype('complex128'),
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

  def __init__(self, c_buffer, backend, replica):
    self.c_buffer = c_buffer
    self._backend = backend
    self._replica = replica

  @staticmethod
  def from_pyval(pyval, replica=0, backend=XLA_LOCAL_BACKEND):
    """Allocate and copy to XLA the given python value."""
    pyval = require_numpy_array_layout(pyval)
    num_replicas = get_replica_count()
    if not 0 <= replica < num_replicas:
      raise ValueError(
          'Attempt to place buffer on replica {} when the replica count is {}'
          .format(replica, num_replicas))
    cbuf = backend.buffer_from_pyval(pyval, replica)
    return LocalBuffer(cbuf, backend, replica)

  def to_py(self):
    return self.c_buffer.ToLiteral()

  def shape(self):
    return _wrap_shape(self.c_buffer.shape())

  def replica(self):
    return self._replica

  def delete(self):
    if self.c_buffer is not None:
      self._backend.delete_buffer(self.c_buffer)
      self.c_buffer = None

  def destructure(self):
    """Assuming a tuple buffer, unpack it into constituent tuple elements."""
    assert self.c_buffer is not None
    result = self._backend.destructure_tuple(self.c_buffer)
    self.delete()
    return tuple(
        LocalBuffer(sub_buffer, replica=self._replica, backend=self._backend)
        for sub_buffer in result)

  def is_deleted(self):
    return self.c_buffer is None

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

  def __hash__(self):
    return hash((self._dtype, self._dimensions, self._minor_to_major))

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
      assert sorted(mtm) == list(range(len(mtm))), self

  def update_minor_to_major(self, minor_to_major):
    if not self.is_array():
      raise ValueError('not an array shape')
    if not isinstance(minor_to_major, tuple):
      raise TypeError('minor_to_major must be a tuple')
    updated = Shape.array_shape(
        self.element_type(), self.dimensions(), minor_to_major)
    updated._check_minor_to_major()  # pylint: disable=protected-access
    return updated

  def serialize(self, proto):
    """Serializes 'shape' into proto."""
    if self.is_tuple():
      proto.element_type = xla_data_pb2.TUPLE
      for shape in self.tuple_shapes():
        shape.serialize(proto.tuple_shapes.add())
    else:
      proto.element_type = dtype_to_etype(self.element_type())
      proto.dimensions.extend(self.dimensions())
      proto.is_dynamic_dimension.extend([False for _ in self.dimensions()])
      if self.minor_to_major():
        proto.layout.format = xla_data_pb2.DENSE
        proto.layout.minor_to_major.extend(self.minor_to_major())


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
    self.num_replicas = get_replica_count()


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

  def __init__(self, c_computation, is_compiled, backend=XLA_LOCAL_BACKEND):
    self._c_computation = c_computation
    self._backend = backend
    self._is_compiled = is_compiled

  @property
  def computation(self):
    if self._is_compiled:
      raise ValueError(
          'Attempt to read the XLA computation of a compiled LocalComputation.')
    return self._c_computation

  def GetProto(self):
    """Get the HloModuleProto proto object in this local computation.

    Returns:
       An HloModuleProto proto object that has the whole-graph information.
    """
    serialized = self.computation.GetSerializedProto()
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
    if self._is_compiled:
      raise ValueError('Attempt to compile a compiled local XLA computation.')

    result_shape = _wrap_shape(self.computation.GetReturnValueShape())

    if layout_fn:
      argument_shapes = [
          shape.map_leaves(layout_fn) for shape in argument_shapes
      ]
      result_shape = result_shape.map_leaves(layout_fn)

    argument_shapes = list(argument_shapes)

    compile_options = compile_options or CompileOptions()
    compile_options.result_shape = result_shape
    c = self._backend.compile(self.computation, argument_shapes,
                              compile_options)
    return LocalComputation(c, is_compiled=True, backend=self._backend)

  def CompileWithExampleArguments(self,
                                  arguments=(),
                                  compile_options=None,
                                  layout_fn=None):
    return self.Compile(
        argument_shapes=[Shape.from_pyval(arg) for arg in arguments],
        compile_options=compile_options,
        layout_fn=layout_fn)

  def GetReturnValueShape(self):
    return _wrap_shape(self._c_computation.GetReturnValueShape())

  def Execute(self, arguments=(), check_for_deleted_args=True):
    """Execute on one replica with LocalBuffer arguments and return value."""
    if check_for_deleted_args and any(arg.is_deleted() for arg in arguments):
      raise ValueError('Executing with deleted local buffer argument')
    raw_args = [arg.c_buffer for arg in arguments]
    output_buffer = self._backend.execute(self._c_computation, raw_args)
    return LocalBuffer(output_buffer, backend=self._backend, replica=0)

  def ExecutePerReplica(self, arguments=None):
    """Execute on many replicas with LocalBuffer arguments and return value.

    Args:
      arguments: A sequence of sequences of LocalBuffers. The i'th inner
        sequence comprises the arguments for execution on the i'th replica.

    Returns:
      A list of the computation's outputs on each replica, as a LocalBuffer. If
      a shallow sequence of arguments was passed in for `arguments`, then the
      sole, zero'th replica's output is returned instead, as a LocalBuffer.
    """
    if not self._is_compiled:
      raise ValueError('Cannot execute an uncompiled local XLA computation.')
    if arguments is None:
      arguments = ((),) * get_replica_count()
    else:
      arguments = [list(replica_args) for replica_args in arguments]

    # Check arguments
    for replica, replica_args in enumerate(arguments):
      for arg in replica_args:
        if arg.is_deleted():
          raise ValueError('Executing with deleted local buffer argument')
        if arg.replica() != replica:
          raise ValueError(
              'Executing on replica {} with argument from replica {}'.format(
                  replica, arg.replica()))

    # Pull out argument buffer handles
    stripped_args = [
        [arg.c_buffer for arg in replica_args] for replica_args in arguments
    ]

    # Execute
    output_buffers = self._backend.execute_replicated(
        self._c_computation, stripped_args)

    # Wrap output handles in LocalBuffer instances
    return tuple(
        LocalBuffer(output_buffer, backend=self._backend, replica=replica)
        for replica, output_buffer in enumerate(output_buffers))

  def ExecuteWithPythonValues(self, arguments=()):
    """Execute on one replica with Python values as arguments and output."""

    def put(arg):
      return LocalBuffer.from_pyval(arg, backend=self._backend)

    arguments = [put(arg) for arg in arguments]
    return self.Execute(arguments).to_py()

  def ExecuteWithPythonValuesPerReplica(self, arguments):
    """Execute on many replicas with Python values as arguments and output."""

    def put(arg, replica):
      return LocalBuffer.from_pyval(arg, replica, backend=self._backend)

    arguments = [[put(arg, replica)
                  for arg in replica_args]
                 for replica, replica_args in enumerate(arguments)]
    return [out.to_py() for out in self.ExecutePerReplica(arguments)]

  def __del__(self):
    # Ensure a reference to C-based destructor for use in __del__.
    if self._is_compiled:
      self._backend.delete_executable(self._c_computation)
    else:
      assert isinstance(self._c_computation, c_api.LocalComputation)
      c_api.DeleteLocalComputation(self._c_computation)


def _make_replica_group_proto(replica_group):
  replica_group_proto = xla_data_pb2.ReplicaGroup()
  replica_group_proto.replica_ids.extend(replica_group)
  return replica_group_proto


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

  def Build(self, root=None, backend=XLA_LOCAL_BACKEND):
    if root is not None:
      return LocalComputation(
          self._client.BuildWithRoot(root), is_compiled=False, backend=backend)
    else:
      return LocalComputation(
          self._client.Build(), is_compiled=False, backend=backend)

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

  def Iota(self, dtype, size):
    """Enqueues an iota constant onto the computation.

    Args:
      dtype: expected numpy dtype of the output.
      size: integer, the number of elements in the array.

    Returns:
      A LocalOp representing the added iota constant.
    """
    element_type = DTYPE_TO_XLA_ELEMENT_TYPE[str(np.dtype(dtype))]
    return self._client.Iota(element_type, size)

  def BroadcastedIota(self, dtype, shape, dimension):
    """Enqueues a broadcasted iota constant onto the computation.

    Args:
      dtype: expected numpy dtype of the output.
      shape: tuple of integers, the expected output shape (dimensions).
      dimension: positive integer, dimension along which to increment values.

    Returns:
      A LocalOp representing the added broadcasted iota constant.
    """
    xla_shape = Shape.array_shape(dtype, shape)
    return self._client.BroadcastedIota(xla_shape, dimension)

  def Broadcast(self, operand, sizes):
    """Enqueues a broadcast operation onto the computation.

    Args:
      operand: the operand LocalOp to broadcast.
      sizes: an iterable of broadcast sizes.

    Returns:
      A LocalOp representing the added broadcast op.
    """
    return self._client.Broadcast(operand, sizes)

  def BroadcastInDim(self, operand, shape, broadcast_dimensions):
    """Enqueues a broadcast-in-dimensions operation onto the computation.

    Args:
      operand: the operand LocalOp to broadcast.
      shape: tuple of integers, the expected output shape.
      broadcast_dimensions: tuple of integers identifying which dimensions
        of the output are to be broadcast into.

    Returns:
      A LocalOp representing the added broadcast-in-dimensions op.
    """
    return self._client.BroadcastInDim(operand, shape, broadcast_dimensions)

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

  def BitcastConvertType(self, operand, new_element_type):
    """Enqueues a bitcast type conversion operation onto the computation.

    Args:
      operand: the operand to convert.
      new_element_type: the target primitive type.

    Returns:
      A LocalOp representing the added conversion op.
    """
    return self._client.BitcastConvertType(operand, new_element_type)

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

  def AllToAll(self,
               operand,
               split_dimension,
               concat_dimension,
               replica_groups=None):
    """AllToAll op.

    Args:
      operand: LocalOp representing the input array
      split_dimension: the dimension along which the operand is split
      concat_dimension: the dimension along which the split blocks are
        concatenated
      replica_groups: optional, list of lists of ints encoding a partition of
        the set {0, 1, ..., num_replicas} into equally-sized replica groups
        within which the all-to-all is performed. If not supplied or None (the
        default), all replicas belong to the same group.

    Returns:
      A LocalOp that represents the all-to-all concatenation.
    """
    if replica_groups is None:
      replica_groups_protos = []  # special value for XLA API
    else:
      replica_groups = list(replica_groups)
      replica_groups_protos = [
          _make_replica_group_proto(group) for group in replica_groups]
    if not replica_groups:
      split_count = get_replica_count()
    else:
      split_count = len(replica_groups[0])
      if not all(split_count == len(g) for g in replica_groups):
        raise ValueError('Replica groups must be equally sized')
    return self._client.AllToAll(operand, split_dimension, concat_dimension,
                                 split_count, replica_groups_protos)

  def CrossReplicaSum(self, operand, replica_groups=None):
    """CrossReplicaSum op.

    Args:
      operand: the operand to sum across replica instances.
      replica_groups: optional, list of lists of ints encoding a partition of
        the set {0, 1, ..., num_replicas} into equally-sized replica groups
        within which the cross-replica sum is performed. If not supplied or None
        (the default), all replicas belong to the same group.

    Returns:
      A LocalOp that represents on each replica the sum of its group's values.
    """
    if replica_groups is None:
      replica_groups = []  # special value for XLA API
    else:
      replica_groups = [
          _make_replica_group_proto(group) for group in replica_groups]
    return self._client.CrossReplicaSum(operand, replica_groups)

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
        operand, select.computation, window_dimensions, window_strides, pads,
        source, init_value, scatter.computation)

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
    return self._client.Call(computation_to_apply.computation, operands)

  def CustomCall(self,
                 call_target_name,
                 operands,
                 shape_with_layout,
                 operand_shapes_with_layout,
                 opaque=None):
    """Enqueues a custom call operation onto the computation.

    Args:
      call_target_name: the name of the function to call.
      operands: an iterable of LocalOp. The number and types of operands must
        match the arity of `operand_shapes_with_layout`.
      shape_with_layout: the shape of the operator's output, with layout.
      operand_shapes_with_layout: the shapes of `operands`, including the
        expected layouts.
      opaque: an opaque string passed to the backend.

    Returns:
      A LocalOp representing the added custom call op.
    """
    opaque = opaque or b''
    return self._client.CustomCall(call_target_name, operands,
                                   shape_with_layout,
                                   operand_shapes_with_layout, opaque)

  def Map(self, operands, computation_to_apply, dimensions):
    """Enqueues a map operation onto the computation.

    Args:
      operands: an iterable of LocalOp.
      computation_to_apply: a Computation object.
      dimensions: dimensions over which to apply map the function.

    Returns:
      A LocalOp representing the added Map op.
    """
    return self._client.Map(operands, computation_to_apply.computation,
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
                               computation_to_apply.computation, dimensions)

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
        operand, init_value, computation_to_apply.computation,
        window_dimensions, window_strides, (), (), pads)

  def ReduceWindowWithGeneralPadding(
      self, operand, init_value, computation_to_apply, window_dimensions,
      window_strides, base_dilations, window_dilations, padding):
    """Enqueues a windowed reduction operation onto the computation.

    Args:
      operand: reduction operand (LocalOp).
      init_value: reduction initial value (LocalOp).
      computation_to_apply: a binary reduction function (Computation).
      window_dimensions: dimensions of window (sequence of integers).
      window_strides: strides for window (sequence of integers).
      base_dilations: dilations for the base (sequence of integers).
      window_dilations: dilations for window (sequence of integers).
      padding: length-N array-like of pairs of integers of (low, high) padding.

    Returns:
      A LocalOp representing the added ReduceWindow op.
    """
    return self._client.ReduceWindowWithGeneralPadding(
        operand, init_value, computation_to_apply.computation,
        window_dimensions, window_strides, base_dilations, window_dilations,
        padding)

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
    return self._client.While(cond.computation, body.computation, init)

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
        pred, true_operand, true_computation.computation, false_operand,
        false_computation.computation)

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

  def Conv(self, lhs, rhs, window_strides, padding, feature_group_count=1):
    """Enqueues a Conv operation onto the computation.

    Args:
      lhs: LocalOp for the rank N+2 array of inputs.
      rhs: LocalOp for the rank N+2 array of kernel weights.
      window_strides: length-N array-like of integer kernel strides.
      padding: PaddingType representing either 'SAME' or 'VALID' padding.
      feature_group_count: number of feature groups for grouped convolution.

    Returns: a LocalOp representing the Conv operation.
    """
    pads = _convert_padding_type_to_pad_values(
        padding, self.GetShape(lhs).dimensions()[2:],
        self.GetShape(rhs).dimensions()[2:], window_strides)
    return self.ConvGeneralDilated(
        lhs, rhs, window_strides, pads, (), (),
        dimension_numbers=None, feature_group_count=feature_group_count)

  def ConvWithGeneralPadding(self, lhs, rhs, window_strides, padding,
                             lhs_dilation, rhs_dilation, feature_group_count=1):
    """Enqueues a ConvWithGeneralPadding operation onto the computation.

    Args:
      lhs: LocalOp for the rank N+2 array of inputs.
      rhs: LocalOp for the rank N+2 array of kernel weights.
      window_strides: length-N array-like of kernel strides.
      padding: length-N array-like of pairs of integers of (low, high) padding.
      lhs_dilation: length-N array-like of dilation factors.
      rhs_dilation: length-N array-like of dilation factors.
      feature_group_count: number of feature groups for grouped convolution.

    Returns:
      A ComputationdataHandle representing the added ConvWithGeneralPadding op.
    """
    return self.ConvGeneralDilated(
        lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
        dimension_numbers=None, feature_group_count=feature_group_count)

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
                         rhs_dilation, dimension_numbers=None,
                         feature_group_count=1):
    """Enqueues a ConvGeneralDilated operation onto the computation.

    Args:
      lhs: LocalOp for the rank N+2 array of inputs.
      rhs: LocalOp for the rank N+2 array of kernel weights.
      window_strides: length-N array-like of integer kernel strides.
      padding: length-N array-like of pairs of integers of (low, high) padding.
      lhs_dilation: length-N array-like of integer dilation factors.
      rhs_dilation: length-N array-like of integer dilation factors.
      dimension_numbers: optional, either an
        xla_data_pb2.ConvolutionDimensionNumbers proto instance or a tuple
        (lhs_spec, rhs_spec, out_spec) where each element is a string of length
        N+2 identifying by position (1) batch dimensions in lhs, rhs, and the
        output with the character 'N', (2) feature dimensions in lhs and the
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
        appearing in rhs_spec that is not 'I' or 'O'. By default, use the same
        dimension numbering as Conv and ConvWithGeneralPadding.
      feature_group_count: number of feature groups for grouped convolution.

    Returns: a LocalOp representing the ConvGenralDilated operation.
    """
    if dimension_numbers is None:
      dimension_numbers = self._GetConvDimensionNumbers(len(window_strides))
    elif not isinstance(dimension_numbers,
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
                                           dimension_numbers,
                                           feature_group_count)

  def Sort(self, operand, dimension=-1):
    """Enqueues a sort operation onto the computation."""
    return self._client.Sort(operand, dimension)

  def SortKeyVal(self, keys, values, dimension=-1):
    """Enqueues a key-value sort operation onto the computation."""
    return self._client.SortKeyVal(keys, values, dimension)

  def Cholesky(self, a):
    """Enqueues a Cholesky decomposition onto the computation."""
    return self._client.Cholesky(a)

  def QR(self, a, full_matrices=True):
    """Enqueues a QR decomposition onto the computation."""
    return self._client.QR(a, full_matrices)

  def TriangularSolve(self,
                      a,
                      b,
                      left_side=False,
                      lower=False,
                      transpose_a=False,
                      conjugate_a=False,
                      unit_diagonal=False):
    """Enqueues a triangular-solve operation onto the computation."""
    return self._client.TriangularSolve(a, b, left_side, lower, transpose_a,
                                        conjugate_a, unit_diagonal)

  def Gather(self, a, start_indices, dimension_numbers, slice_sizes):
    """Enqueues a Gather operation onto the computation."""
    return self._client.Gather(a, start_indices, dimension_numbers,
                               slice_sizes)

  def Scatter(self, a, scatter_indices, updates, update_computation,
              dimension_numbers):
    """Enqueues a Scatter operation onto the computation."""
    return self._client.Scatter(
        a, scatter_indices, updates, update_computation.computation,
        dimension_numbers,)


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


def initialize_platform_name(platform_name):
  """Initializes the desired platform name to use on XLA service init.

  Args:
    platform_name: string name of platform.

  Raises:
    A runtime exception if the XLA service has already been initialized.
    A runtime exception if the platform does not exist, or there are no devices
    with that platform.
  """
  platform_name = _maybe_encode_string(platform_name)
  c_api.InitializePlatformName(platform_name)


def get_replica_count():
  """Returns the current replica count used for the XLA service.

  Note: this will return a value whether the XLA service has been initialized
  yet or not.
  """
  return c_api.GetReplicaCount()


def register_cpu_custom_call_target(name, fn):
  """Registers a CPU custom call target.

  Args:
    name: bytes containing the name of the function.
    fn: a PyCapsule object containing the function pointer.
  """
  c_api.RegisterCpuCustomCallTarget(name, fn)


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
