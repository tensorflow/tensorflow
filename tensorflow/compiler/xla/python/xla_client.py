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
"""An XLA client in Python."""

import atexit
import contextlib
import enum  # pylint: disable=g-bad-import-order
import gzip
import inspect
import logging
import os
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

from . import xla_extension as _xla
import numpy as np

# Note this module does *not* depend on any Python protocol buffers. The XLA
# Python bindings are currently packaged both as part of jaxlib and as part
# of TensorFlow. If we use protocol buffers here, then importing both jaxlib
# and TensorFlow may fail with duplicate protocol buffer message definitions.

# Most functions are snake_case for consistency with other modules, some
# method names are CamelCase for consistency with XLA.
# pylint: disable=invalid-name

# Pylint has false positives for type annotations.
# pylint: disable=invalid-sequence-index

ops = _xla.ops
profiler = _xla.profiler

# Just an internal arbitrary increasing number to help with backward-compatible
# changes.
_version = 138

# Version number for MLIR:Python components.
mlir_api_version = 45

xla_platform_names = {
    'cpu': 'Host',
    'gpu': 'CUDA',
}

logger = logging.getLogger(__name__)

_NameValueMapping = Mapping[str, Union[str, int, List[int], float]]


def make_interpreter_client():
  return _xla.get_interpreter_client()


def make_cpu_client(*, use_tfrt: bool = True) -> ...:
  assert use_tfrt
  return _xla.get_tfrt_cpu_client(asynchronous=True)


def make_gpu_client(distributed_client=None, node_id=0, platform_name=None,
                    allowed_devices=None):
  """Returns a GPU client. BFC allocator is used by default."""
  allocator = os.getenv('XLA_PYTHON_CLIENT_ALLOCATOR', 'default').lower()
  memory_fraction = os.getenv('XLA_PYTHON_CLIENT_MEM_FRACTION')
  preallocate = os.getenv('XLA_PYTHON_CLIENT_PREALLOCATE')
  if allocator not in ('default', 'platform', 'bfc', 'cuda_async'):
    raise ValueError(
        'XLA_PYTHON_CLIENT_ALLOCATOR env var must be "default", "platform", '
        '"bfc", or "cuda_async", got "%s"' % allocator)
  config = _xla.GpuAllocatorConfig()
  if allocator == 'default':
    config.kind = _xla.GpuAllocatorConfig.Kind.DEFAULT
  if allocator == 'platform':
    config.kind = _xla.GpuAllocatorConfig.Kind.PLATFORM
  if allocator == 'bfc':
    config.kind = _xla.GpuAllocatorConfig.Kind.BFC
  if allocator == 'cuda_async':
    config.kind = _xla.GpuAllocatorConfig.Kind.CUDA_ASYNC
  if memory_fraction:
    config.memory_fraction = float(memory_fraction)
  config.preallocate = preallocate not in ('0', 'false', 'False')

  return _xla.get_gpu_client(
      asynchronous=True,
      allocator_config=config,
      distributed_client=distributed_client,
      node_id=node_id,
      platform_name=platform_name,
      allowed_devices=allowed_devices)


def make_tfrt_tpu_c_api_client(options: Optional[_NameValueMapping] = None):
  if options is None:
    options = {}
  return _xla.get_c_api_client('tpu', options)


def load_pjrt_plugin_dynamically(plugin_name: str, library_path: str) -> None:
  _xla.load_pjrt_plugin(plugin_name, library_path)


def make_c_api_client(
    plugin_name: str,
    options: Optional[_NameValueMapping] = None,
):
  """Creates a PJRT C API client for a PJRT plugin.

  It is required that load_pjrt_plugin_dynamically is called once with the same
  plugin_name before this method is called.

  Args:
     plugin_name: the name of the PJRT plugin.
     options: extra platform-specific options.

  Returns:
     A PJRT C API client for plugin_name.
  """
  if options is None:
    options = {}
  return _xla.get_c_api_client(plugin_name, options)


def _use_pjrt_c_api() -> bool:
  use_pjrt_c_api = os.getenv('JAX_USE_PJRT_C_API_ON_TPU', 'false')
  if use_pjrt_c_api not in ('1', '0', 'true', 'false'):
    raise ValueError(
        'JAX_USE_PJRT_C_API_ON_TPU env var must be "0", "1", "true" or '
        f'"false", got "{use_pjrt_c_api}"')
  return use_pjrt_c_api in ('1', 'true')


def make_tpu_client(use_pjrt_c_api: bool = False):
  """Returns a TPU client. Defaults to allowing 32 in-flight computations."""
  if use_pjrt_c_api or _use_pjrt_c_api():
    library_path = os.getenv('TPU_LIBRARY_PATH', 'libtpu.so')
    load_pjrt_plugin_dynamically('tpu', library_path)
    return make_tfrt_tpu_c_api_client()

  max_inflight_computations = os.getenv(
      'JAX_TPU_MAX_INFLIGHT_COMPUTATIONS', '32')
  try:
    max_inflight_computations = int(max_inflight_computations)
  except ValueError as e:
    raise ValueError(
        f'JAX_TPU_MAX_INFLIGHT_COMPUTATIONS env var must be an int, '
        f'got {max_inflight_computations}') from e
  return _xla.get_tpu_client(
      max_inflight_computations=max_inflight_computations)


def make_plugin_device_client():
  """Returns a plugin device client."""
  try:
    return _xla.get_plugin_device_client()
  except AttributeError as e:
    raise AttributeError(
        'xla_extension has no attributes named get_plugin_device_client. '
        'Compile TensorFlow with '
        '//tensorflow/compiler/xla/python:enable_plugin_device set to true '
        '(defaults to false) to enable this.') from e


class OpMetadata:
  """Python representation of a xla.OpMetadata protobuf."""
  __slots__ = ('op_type', 'op_name', 'source_file', 'source_line')

  def __init__(self, op_type='', op_name='', source_file='', source_line=0):
    self.op_type = op_type
    self.op_name = op_name
    self.source_file = source_file
    self.source_line = source_line


def CurrentSourceInfoMetadata(op_type=None, op_name=None, skip_frames=1):
  """Helper for use in source mapping that returns an OpMetadata object."""
  full_filename, lineno = inspect.stack()[skip_frames][1:3]
  filename = os.path.basename(full_filename)
  return OpMetadata(
      op_type=op_type,
      op_name=op_name,
      source_file=filename,
      source_line=lineno)


PrimitiveType = _xla.PrimitiveType

bfloat16 = _xla.bfloat16_dtype()
float8_e4m3fn = _xla.float8_e4m3fn_dtype()
float8_e5m2 = _xla.float8_e5m2_dtype()

XLA_ELEMENT_TYPE_TO_DTYPE = {
    PrimitiveType.PRED: np.dtype('bool'),
    PrimitiveType.S8: np.dtype('int8'),
    PrimitiveType.S16: np.dtype('int16'),
    PrimitiveType.S32: np.dtype('int32'),
    PrimitiveType.S64: np.dtype('int64'),
    PrimitiveType.U8: np.dtype('uint8'),
    PrimitiveType.U16: np.dtype('uint16'),
    PrimitiveType.U32: np.dtype('uint32'),
    PrimitiveType.U64: np.dtype('uint64'),
    PrimitiveType.F8E4M3FN: np.dtype(float8_e4m3fn),
    PrimitiveType.F8E5M2: np.dtype(float8_e5m2),
    PrimitiveType.BF16: np.dtype(bfloat16),
    PrimitiveType.F16: np.dtype('float16'),
    PrimitiveType.F32: np.dtype('float32'),
    PrimitiveType.F64: np.dtype('float64'),
    PrimitiveType.C64: np.dtype('complex64'),
    PrimitiveType.C128: np.dtype('complex128'),
    PrimitiveType.TUPLE: np.dtype(np.object_),
    PrimitiveType.TOKEN: np.dtype(np.object_),
}

# Note the conversion on the key. Numpy has a known issue wherein dtype hashing
# doesn't work as expected (https://github.com/numpy/numpy/issues/7242). Thus,
# when keying by dtype in this dict, we use the string form of dtypes.
DTYPE_TO_XLA_ELEMENT_TYPE = {
    str(dt): et for et, dt in XLA_ELEMENT_TYPE_TO_DTYPE.items()
}


def dtype_to_etype(dtype):
  """Convenience function for reading DTYPE_TO_XLA_ELEMENT_TYPE."""
  return DTYPE_TO_XLA_ELEMENT_TYPE[str(np.dtype(dtype))]


Shape = _xla.Shape
Shape.__doc__ = """
A Shape is an object defined in C++ that duck types like the following class:

class Shape:
  '''Represents an XLA shape.

  A shape is either an array shape, having rank-many integer
  dimensions and an element type (represented by a Numpy dtype), or it
  is a tuple shape, having a shape for every tuple component:

    type shape =
        TupleShape of shape list
      | ArrayShape of { dimensions: int list; element_type: dtype }
  '''

  @staticmethod
  def tuple_shape(tuple_shapes) -> Shape:
    "Construct a tuple shape."

  @staticmethod
  def array_shape(element_type, dimensions, minor_to_major=None) -> Shape:

  @staticmethod
  def from_pyval(pyval) -> Shape:
    "Returns a Shape that describes a tuple-tree of Numpy arrays."

  def __init__(self, str) -> Shape:
    "Parses a shape string."
  def __eq__(self, other: Shape) -> bool:
  def __ne__(self, other: Shape) -> bool:
  def __hash__(self):
  def __repr__(self):
  def is_tuple(self) -> bool:
  def is_array(self) -> bool:
  def tuple_shapes(self) -> [Shape]:
  def numpy_dtype(self) -> np.dtype:
    "Like element_type(), but returns dtype('O') for a tuple shape."
  def xla_element_type(self) -> PrimitiveType:
  def element_type(self) -> np.dtype:
  def dimensions(self) -> (int, int, ...):
  def rank(self) -> int:
  def with_major_to_minor_layout_if_absent(self) -> Shape:
    "Returns a copy with missing layouts set to major-to-minor."

  def to_serialized_proto(self) -> bytes:
    "Returns 'shape' as a serialized proto."
"""

ProgramShape = _xla.ProgramShape
ProgramShape.__doc__ = """
A ProgramShape is a C++ object that duck types like the following class.

class ProgramShape:
  def __init__(self, parameter_shapes, result_shape):
  def parameter_shapes(self) -> [Shape]:
  def result_shape(self) -> Shape:
  def __repr__(self):
"""

ShapeIndex = _xla.ShapeIndex
ShapeIndex.__doc__ = """
A Shape is an object defined in C++ that duck types like the following class:

class ShapeIndex:
  '''Represents an XLA ShapeIndex.

  An index for specifying a particular nested subshape within a shape. Used in
  ShapeUtil::GetSubshape and other interfaces. ShapeIndex defines a path through
  the Shape tree where each element of ShapeIndex indexes into a tuple (or
  nested tuple) within the shape. For a non-nested tuple, an index has a single
  element.
  '''

  def __init__(self, List[int]) -> ShapeIndex:
  def __eq__(self, other: Shape) -> bool:
  def __ne__(self, other: Shape) -> bool:
  def __hash__(self):
  def __repr__(self):
"""


def shape_from_pyval(pyval):
  """Returns a Shape that describes a tuple-tree of Numpy arrays."""

  def convert(pyval):
    if isinstance(pyval, tuple):
      return Shape.tuple_shape(tuple(convert(elt) for elt in pyval))
    else:
      return Shape.array_shape(pyval.dtype, np.shape(pyval))

  return convert(pyval)


DeviceAssignment = _xla.DeviceAssignment
DeviceAssignment.__doc__ = """
A DeviceAssignment is a C++ object with the following signature.

def create(assignment):
  '''Builds a device assignment.

   Args:
     assignment: a 2D numpy array of device ordinal integers, indexed by
       [replica][computation_in_replica].
   Returns:
     A device assignment.
  '''

def replica_count():
  '''Returns the number of replicas.'''
def computation_count():
  '''Returns the number of computations per replica.'''
"""

Device = _xla.Device
CompileOptions = _xla.CompileOptions

HostBufferSemantics = _xla.HostBufferSemantics

# An Executable is a C++ class that duck types with the following API:
# class Executable:
#   def local_devices(self) -> [Device]:
#   def execute(self, arguments : [Buffer]) -> Buffer:
#     """Execute on one replica with Buffer arguments and return value."""
#
#   def size_of_generated_code_in_bytes(self) -> int:
#     """Return generated binary size, or -1 if not known."""
#
#   def execute_sharded_on_local_devices(self, arguments: [[Buffer]])
#       -> [Buffer]:
#     """Execute on many replicas with Buffer arguments and return value.
#
#     Args:
#       arguments: A sequence of sequences of Buffers. The i'th element of each
#         sequence comprises the arguments for execution on the i'th local
#         device.
#
#     Returns:
#       A list of the computation's outputs as a list of Buffers for each
#       device.
#     """
#
# There are different implementations of Executable for different backends.


def execute_with_python_values(executable, arguments, backend):
  """Execute on one replica with Python values as arguments and output."""

  def put(arg):
    return backend.buffer_from_pyval(arg, device=executable.local_devices()[0])

  arguments = [put(arg) for arg in arguments]
  outputs = executable.execute(arguments)
  return [np.asarray(x) for x in outputs]


def execute_with_python_values_replicated(executable, arguments, backend):
  """Execute on many replicas with Python values as arguments and output.

  Args:
    executable: the program to run.
    arguments: a list of lists of Python values indexed by `[replica][arg_num]`
      to pass as inputs.
    backend: the backend we are targeting.

  Returns:
    A list of python values, one per replica.
  """
  devices = executable.local_devices()

  # pylint: disable=g-complex-comprehension
  def copy_to_devices(pyvals):
    return [backend.buffer_from_pyval(v, d) for v, d in zip(pyvals, devices)]

  inputs = [copy_to_devices(pyvals) for pyvals in zip(*arguments)]
  outputs = executable.execute_sharded_on_local_devices(inputs)
  return [[np.asarray(x) for x in xs] for xs in zip(*outputs)]


class PaddingType(enum.Enum):
  VALID = 1
  SAME = 2


def window_padding_type_to_pad_values(padding_type, lhs_dims, rhs_dims,
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
    pad_sizes = [
        max((out_size - 1) * stride + filter_size - in_size, 0)
        for out_size, stride, filter_size, in_size in zip(
            out_shape, window_strides, rhs_dims, lhs_dims)
    ]
    return [(pad_size // 2, pad_size - pad_size // 2) for pad_size in pad_sizes]
  else:
    msg = 'Unexpected PaddingType value: {}'
    raise ValueError(msg.format(padding_type))


XlaBuilder = _xla.XlaBuilder
XlaComputation = _xla.XlaComputation
XlaOp = _xla.XlaOp
FftType = _xla.FftType
Client = _xla.Client
Buffer = _xla.Buffer
ArrayImpl = _xla.ArrayImpl
DeviceArrayBase = _xla.DeviceArrayBase
LoadedExecutable = _xla.LoadedExecutable
OpSharding = _xla.OpSharding
HloSharding = _xla.HloSharding
Sharding = _xla.Sharding
XLACompatibleSharding = _xla.XLACompatibleSharding
NamedSharding = _xla.NamedSharding
SingleDeviceSharding = _xla.SingleDeviceSharding
PmapSharding = _xla.PmapSharding
GSPMDSharding = _xla.GSPMDSharding


def register_custom_call_target(
    name: str, fn: Any, platform: str = 'cpu'
) -> None:
  """Registers a custom call target.

  Args:
    name: bytes containing the name of the function.
    fn: a PyCapsule object containing the function pointer.
    platform: the target platform.
  """
  # To support AMD GPUs, we need to have xla_platform_names["gpu"] == "ROCM"
  # Since that is hardcoded to CUDA, we are using the following as workaround.
  _xla.register_custom_call_target(name, fn,
                                   xla_platform_names.get(platform, platform))


# Deprecated. Use register_custom_call_target instead.
register_cpu_custom_call_target = register_custom_call_target
register_custom_call_partitioner = _xla.register_custom_call_partitioner
hlo_sharding_util = _xla.hlo_sharding_util


class PaddingConfigDimension:
  """Python representation of a xla.PaddingConfigDimension protobuf."""
  __slots__ = ('edge_padding_low', 'edge_padding_high', 'interior_padding')

  edge_padding_low: int
  edge_padding_high: int
  interior_padding: int

  def __init__(self):
    self.edge_padding_low = 0
    self.edge_padding_high = 0
    self.interior_padding = 0


class PaddingConfig:
  """Python representation of a xla.PaddingConfig protobuf."""
  __slots__ = ('dimensions',)

  def __init__(self):
    self.dimensions = []


def make_padding_config(
    padding_config: Union[PaddingConfig, Sequence[Tuple[int, int, int]]]
) -> PaddingConfig:
  """Create PaddingConfig proto from list of triples of integers.

  Args:
    padding_config: either a PaddingConfig or a list of integer triples
      (edge_padding_low, edge_padding_high, interior_padding) representing the
      configuration of the padding operation.

  Returns:
    A `PaddingConfig` object.
  """
  if not isinstance(padding_config, PaddingConfig):
    triples = padding_config
    padding_config = PaddingConfig()
    for lo, hi, interior in triples:
      dimension = PaddingConfigDimension()
      dimension.edge_padding_low = lo
      dimension.edge_padding_high = hi
      dimension.interior_padding = interior
      padding_config.dimensions.append(dimension)
  return padding_config


class DotDimensionNumbers:
  """Python representation of a xla.DotDimensionNumbers protobuf."""
  __slots__ = ('lhs_contracting_dimensions', 'rhs_contracting_dimensions',
               'lhs_batch_dimensions', 'rhs_batch_dimensions')

  def __init__(self):
    self.lhs_contracting_dimensions = []
    self.rhs_contracting_dimensions = []
    self.lhs_batch_dimensions = []
    self.rhs_batch_dimensions = []


def make_dot_dimension_numbers(
    dimension_numbers: Union[DotDimensionNumbers,
                             Tuple[Tuple[List[int], List[int]],
                                   Tuple[List[int], List[int]]]]
) -> DotDimensionNumbers:
  """Builds a DotDimensionNumbers object from a specification.

  Args:
    dimension_numbers: either a `DotDimensionNumbers` or a nested tuple
      `((lhs_contract, rhs_contract), (lhs_batch, rhs_batch))` of lists of
      integers representing the dimensions to treat as contracting dimensions
      and batch dimensions on each input operand.

  Returns:
    A `DotDimensionNumbers` object.
  """
  if isinstance(dimension_numbers, (list, tuple)):
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    dot_dims_proto = DotDimensionNumbers()
    dot_dims_proto.lhs_contracting_dimensions.extend(lhs_contract)
    dot_dims_proto.rhs_contracting_dimensions.extend(rhs_contract)
    dot_dims_proto.lhs_batch_dimensions.extend(lhs_batch)
    dot_dims_proto.rhs_batch_dimensions.extend(rhs_batch)
    return dot_dims_proto
  else:
    return dimension_numbers


class ConvolutionDimensionNumbers:
  """Python representation of a xla.ConvolutionDimensionNumbers protobuf."""
  __slots__ = ('input_batch_dimension', 'input_feature_dimension',
               'input_spatial_dimensions', 'kernel_input_feature_dimension',
               'kernel_output_feature_dimension', 'kernel_spatial_dimensions',
               'output_batch_dimension', 'output_feature_dimension',
               'output_spatial_dimensions')

  def __init__(self):
    self.input_batch_dimension = 0
    self.input_feature_dimension = 0
    self.input_spatial_dimensions = []
    self.kernel_input_feature_dimension = 0
    self.kernel_output_feature_dimension = 0
    self.kernel_spatial_dimensions = []
    self.output_batch_dimension = 0
    self.output_feature_dimension = 0
    self.output_spatial_dimensions = []


def make_convolution_dimension_numbers(
    dimension_numbers: Union[None, ConvolutionDimensionNumbers, Tuple[str, str,
                                                                      str]],
    num_spatial_dimensions: int) -> ConvolutionDimensionNumbers:
  """Builds a ConvolutionDimensionNumbers object from a specification.

  Args:
    dimension_numbers: optional, either a ConvolutionDimensionNumbers object or
      a tuple (lhs_spec, rhs_spec, out_spec). Each element is a string of
      length N+2 identifying by position: (1) batch dimensions in lhs, rhs, and
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
        appearing in rhs_spec that is not 'I' or 'O'. By default, use the same
        dimension numbering as Conv and ConvWithGeneralPadding.
    num_spatial_dimensions: the number of spatial dimensions.

  Returns:
    A `ConvolutionDimensionNumbers` object.
  """
  if dimension_numbers is None:
    nd = num_spatial_dimensions
    dimension_numbers = ConvolutionDimensionNumbers()
    dimension_numbers.input_batch_dimension = 0
    dimension_numbers.input_feature_dimension = 1
    dimension_numbers.output_batch_dimension = 0
    dimension_numbers.output_feature_dimension = 1
    dimension_numbers.kernel_output_feature_dimension = 0
    dimension_numbers.kernel_input_feature_dimension = 1
    dimension_numbers.input_spatial_dimensions.extend(range(2, 2 + nd))
    dimension_numbers.kernel_spatial_dimensions.extend(range(2, 2 + nd))
    dimension_numbers.output_spatial_dimensions.extend(range(2, 2 + nd))
  elif isinstance(dimension_numbers, tuple):
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    dimension_numbers = ConvolutionDimensionNumbers()

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
  return dimension_numbers


class PrecisionConfig:
  """Python representation of a xla.PrecisionConfig protobuf."""
  __slots__ = ('operand_precision',)

  Precision = _xla.PrecisionConfig_Precision

  def __init__(self):
    self.operand_precision = []


class GatherDimensionNumbers:
  """Python representation of a xla.GatherDimensionNumbers protobuf."""
  __slots__ = ('offset_dims', 'collapsed_slice_dims', 'start_index_map',
               'index_vector_dim')

  def __init__(self):
    self.offset_dims = []
    self.collapsed_slice_dims = []
    self.start_index_map = []
    self.index_vector_dim = 0


class ScatterDimensionNumbers:
  """Python representation of a xla.ScatterDimensionNumbers protobuf."""
  __slots__ = ('update_window_dims', 'inserted_window_dims',
               'scatter_dims_to_operand_dims', 'index_vector_dim')

  def __init__(self):
    self.update_window_dims = []
    self.inserted_window_dims = []
    self.scatter_dims_to_operand_dims = []
    self.index_vector_dim = 0


class ReplicaGroup:
  """Python representation of a xla.ReplicaGroup protobuf."""
  __slots__ = ('replica_ids',)

  def __init__(self):
    self.replica_ids = []


def _make_replica_group_proto(replica_group):
  replica_group_proto = ReplicaGroup()
  replica_group_proto.replica_ids.extend(replica_group)
  return replica_group_proto


def make_replica_groups(replica_groups):
  if replica_groups is None:
    replica_groups_protos = []  # special value for XLA API
  else:
    replica_groups = list(replica_groups)
    replica_groups_protos = [
        _make_replica_group_proto(group) for group in replica_groups
    ]
  return replica_groups_protos


Traceback = _xla.Traceback
Frame = _xla.Frame


@contextlib.contextmanager
def tracebacks(enabled=True):
  """Context manager that enables or disables traceback collection."""
  saved = Traceback.enabled
  Traceback.enabled = enabled
  try:
    yield
  finally:
    Traceback.enabled = saved


def heap_profile(client: Client) -> bytes:
  """Returns a gzipped pprof protocol buffer containing a heap profile."""
  return gzip.compress(client.heap_profile())


XlaRuntimeError = _xla.XlaRuntimeError

# Perform one last garbage collection of deferred Python references. This is
# mostly to keep ASAN happy.
atexit.register(_xla.collect_garbage)

weakref_lru_cache = _xla.weakref_lru_cache
array_result_handler = _xla.array_result_handler
copy_array_to_devices_with_sharding = _xla.copy_array_to_devices_with_sharding
batched_device_put = _xla.batched_device_put
