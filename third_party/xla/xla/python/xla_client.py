# Copyright 2017 The OpenXLA Authors.
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

from __future__ import annotations

import atexit
from collections.abc import Mapping, Sequence
import contextlib
import enum  # pylint: disable=g-bad-import-order
import gzip
import inspect
import logging
import os
import threading
from typing import Any, Protocol, Union

import ml_dtypes
import numpy as np

from . import xla_extension as _xla

# Note this module does *not* depend on any Python protocol buffers. The XLA
# Python bindings are currently packaged both as part of jaxlib and as part
# of TensorFlow. If we use protocol buffers here, then importing both jaxlib
# and TensorFlow may fail with duplicate protocol buffer message definitions.

# Most functions are snake_case for consistency with other modules, some
# method names are CamelCase for consistency with XLA.
# pylint: disable=invalid-name

# Pylint has false positives for type annotations.
# pylint: disable=invalid-sequence-index

ifrt_programs = _xla.ifrt_programs
ops = _xla.ops
profiler = _xla.profiler

# Just an internal arbitrary increasing number to help with backward-compatible
# changes. In JAX, reference this via jax._src.lib.xla_extension_version.
_version = 317

# Version number for MLIR:Python components.
mlir_api_version = 58

xla_platform_names = {
    'cpu': 'Host',
    'gpu': 'CUDA',
}

logger = logging.getLogger(__name__)

_NameValueMapping = Mapping[str, Union[str, int, list[int], float, bool]]


def make_cpu_client(
    asynchronous=True,
    distributed_client=None,
    node_id=0,
    num_nodes=1,
    collectives=None,
    num_devices=None,
) -> ...:
  register_custom_call_handler('cpu', _xla.register_custom_call_target)
  register_custom_type_id_handler('cpu', _xla.register_custom_type_id)
  return _xla.get_tfrt_cpu_client(
      asynchronous=asynchronous,
      distributed_client=distributed_client,
      node_id=node_id,
      num_nodes=num_nodes,
      collectives=collectives,
      num_devices=num_devices,
  )


def make_gpu_client(
    distributed_client=None,
    node_id=0,
    num_nodes=1,
    platform_name=None,
    allowed_devices=None,
    mock=False,
    mock_gpu_topology=None,
):
  """Returns a GPU client. BFC allocator is used by default."""
  options = generate_pjrt_gpu_plugin_options()
  allocator = options['allocator']
  config = _xla.GpuAllocatorConfig()
  if allocator == 'default':
    config.kind = _xla.GpuAllocatorConfig.Kind.DEFAULT
  if allocator == 'platform':
    config.kind = _xla.GpuAllocatorConfig.Kind.PLATFORM
  if allocator == 'bfc':
    config.kind = _xla.GpuAllocatorConfig.Kind.BFC
  if allocator == 'cuda_async':
    config.kind = _xla.GpuAllocatorConfig.Kind.CUDA_ASYNC
  if 'memory_fraction' in options:
    config.memory_fraction = options['memory_fraction']
  if 'preallocate' in options:
    config.preallocate = options['preallocate']
  if 'collective_memory_size' in options:
    config.collective_memory_size = options['collective_memory_size']
  register_custom_call_handler('CUDA', _xla.register_custom_call_target)
  register_custom_call_handler('ROCM', _xla.register_custom_call_target)
  register_custom_type_id_handler('CUDA', _xla.register_custom_type_id)
  register_custom_type_id_handler('ROCM', _xla.register_custom_type_id)

  return _xla.get_gpu_client(
      asynchronous=True,
      allocator_config=config,
      distributed_client=distributed_client,
      node_id=node_id,
      num_nodes=num_nodes,
      platform_name=platform_name,
      allowed_devices=allowed_devices,
      mock=mock,
      mock_gpu_topology=mock_gpu_topology,
  )


def make_tfrt_tpu_c_api_client(options: _NameValueMapping | None = None):
  assert pjrt_plugin_loaded('tpu')
  if not pjrt_plugin_initialized('tpu'):
    initialize_pjrt_plugin('tpu')
  if options is None:
    options = {}
  return _xla.get_c_api_client('tpu', options)


DeviceTopology = _xla.DeviceTopology
get_topology_for_devices = _xla.get_topology_for_devices


def make_tfrt_tpu_c_api_device_topology(
    topology_name: str = '', **kwargs
) -> DeviceTopology:
  """Creates a PJRT C API TopologyDescription."""
  return _xla.get_default_c_api_topology('tpu', topology_name, dict(**kwargs))


def make_c_api_device_topology(
    c_api: Any, topology_name: str = '', **kwargs
) -> DeviceTopology:
  """Creates a PJRT C API TopologyDescription."""
  return _xla.get_c_api_topology(c_api, topology_name, dict(**kwargs))


def pjrt_plugin_loaded(plugin_name: str) -> bool:
  return _xla.pjrt_plugin_loaded(plugin_name)


def load_pjrt_plugin_dynamically(plugin_name: str, library_path: str) -> Any:
  return _xla.load_pjrt_plugin(plugin_name, library_path, c_api=None)


def load_pjrt_plugin_with_c_api(plugin_name: str, c_api: Any) -> None:
  return _xla.load_pjrt_plugin(plugin_name, None, c_api)


def pjrt_plugin_initialized(plugin_name: str) -> bool:
  return _xla.pjrt_plugin_initialized(plugin_name)


def initialize_pjrt_plugin(plugin_name: str) -> None:
  """Initializes a PJRT plugin.

  The plugin needs to be loaded first (through load_pjrt_plugin_dynamically or
  static linking) before this method is called.
  Args:
    plugin_name: the name of the PJRT plugin.
  """
  _xla.initialize_pjrt_plugin(plugin_name)


def make_c_api_client(
    plugin_name: str,
    options: _NameValueMapping | None = None,
    distributed_client: _xla.DistributedRuntimeClient | None = None,
):
  """Creates a PJRT C API client for a PJRT plugin.

  It is required that load_pjrt_plugin_dynamically is called once with the same
  plugin_name before this method is called.

  Args:
     plugin_name: the name of the PJRT plugin.
     options: extra platform-specific options.
     distributed_client: distributed client.

  Returns:
     A PJRT C API client for plugin_name.
  """
  if options is None:
    options = {}
  return _xla.get_c_api_client(plugin_name, options, distributed_client)


def make_tpu_client(
    library_path: str | None = None, options: _NameValueMapping | None = None
):
  """Returns a TPU client. Defaults to allowing 32 in-flight computations."""
  if not pjrt_plugin_loaded('tpu'):
    c_api = load_pjrt_plugin_dynamically('tpu', library_path or 'libtpu.so')
    profiler.register_plugin_profiler(c_api)
  return make_tfrt_tpu_c_api_client(options)


def generate_pjrt_gpu_plugin_options() -> _NameValueMapping:
  """Generates the PjRt GPU plugin options.

  Returns:
    A dictionary of plugin options.
  """

  options = {}
  options['platform_name'] = 'cuda'
  allocator = os.getenv('XLA_PYTHON_CLIENT_ALLOCATOR', 'default').lower()
  memory_fraction = os.getenv('XLA_CLIENT_MEM_FRACTION', '')
  deprecated_memory_fraction = os.getenv('XLA_PYTHON_CLIENT_MEM_FRACTION', '')
  if deprecated_memory_fraction:
    if memory_fraction:
      raise ValueError(
          'XLA_CLIENT_MEM_FRACTION is specified together '
          'with XLA_PYTHON_CLIENT_MEM_FRACTION. '
          'Remove the latter one, it is deprecated.'
      )
    else:
      memory_fraction = deprecated_memory_fraction
  preallocate = os.getenv('XLA_PYTHON_CLIENT_PREALLOCATE', '')
  collective_memory_size = os.getenv(
      'XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB', ''
  )
  if allocator not in ('default', 'platform', 'bfc', 'cuda_async'):
    raise ValueError(
        'XLA_PYTHON_CLIENT_ALLOCATOR env var must be "default", "platform", '
        '"bfc", or "cuda_async", got "%s"' % allocator
    )
  options['allocator'] = allocator
  if memory_fraction:
    options['memory_fraction'] = float(memory_fraction)
  if preallocate:
    options['preallocate'] = preallocate not in ('false', 'False', '0')
  if collective_memory_size:
    options['collective_memory_size'] = int(collective_memory_size) * (1 << 20)
  return options


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
      op_type=op_type, op_name=op_name, source_file=filename, source_line=lineno
  )


PrimitiveType = _xla.PrimitiveType

bfloat16 = ml_dtypes.bfloat16
# TODO(reedwm): Uncomment once the minimum ml_dtypes in JAX is >= 0.5.0.
# Also, it would be better to conditionally import these based on whether they
# are in the current version of ml_dtypes.
# float4_e2m1fn = ml_dtypes.float4_e2m1fn
# float8_e3m4 = ml_dtypes.float8_e3m4
# float8_e4m3 = ml_dtypes.float8_e4m3
# float8_e8m0fnu = ml_dtypes.float8_e8m0fnu
float8_e4m3fn = ml_dtypes.float8_e4m3fn
float8_e4m3b11fnuz = ml_dtypes.float8_e4m3b11fnuz
float8_e4m3fnuz = ml_dtypes.float8_e4m3fnuz
float8_e5m2 = ml_dtypes.float8_e5m2
float8_e5m2fnuz = ml_dtypes.float8_e5m2fnuz

XLA_ELEMENT_TYPE_TO_DTYPE = {
    PrimitiveType.PRED: np.dtype('bool'),
    PrimitiveType.S4: np.dtype('int4'),
    PrimitiveType.S8: np.dtype('int8'),
    PrimitiveType.S16: np.dtype('int16'),
    PrimitiveType.S32: np.dtype('int32'),
    PrimitiveType.S64: np.dtype('int64'),
    PrimitiveType.U4: np.dtype('uint4'),
    PrimitiveType.U8: np.dtype('uint8'),
    PrimitiveType.U16: np.dtype('uint16'),
    PrimitiveType.U32: np.dtype('uint32'),
    PrimitiveType.U64: np.dtype('uint64'),
    # TODO(reedwm): Uncomment once the minimum ml_dtypes in JAX is >= 0.5.0.
    # PrimitiveType.F4E2M1FN: np.dtype(float4_e2m1fn),
    # PrimitiveType.F8E3M4: np.dtype(float8_e3m4),
    # PrimitiveType.F8E4M3: np.dtype(float8_e4m3),
    # PrimitiveType.F8E8M0FNU: np.dtype(float8_e8m0fnu),
    PrimitiveType.F8E4M3FN: np.dtype(float8_e4m3fn),
    PrimitiveType.F8E4M3B11FNUZ: np.dtype(float8_e4m3b11fnuz),
    PrimitiveType.F8E5M2: np.dtype(float8_e5m2),
    PrimitiveType.F8E4M3FNUZ: np.dtype(float8_e4m3fnuz),
    PrimitiveType.F8E5M2FNUZ: np.dtype(float8_e5m2fnuz),
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


def shape_from_pyval(pyval, layout: Sequence[int] | None = None):
  """Returns a Shape that describes a tuple-tree of Numpy arrays."""

  def convert(pyval):
    if isinstance(pyval, tuple):
      if layout is not None:
        raise NotImplementedError(
            'shape_from_pyval does not support layouts for tuple shapes'
        )
      return Shape.tuple_shape(tuple(convert(elt) for elt in pyval))
    else:
      return Shape.array_shape(pyval.dtype, np.shape(pyval), layout)

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


class PaddingType(enum.Enum):
  VALID = 1
  SAME = 2


def window_padding_type_to_pad_values(
    padding_type, lhs_dims, rhs_dims, window_strides
):
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
            out_shape, window_strides, rhs_dims, lhs_dims
        )
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
Memory = _xla.Memory
ArrayImpl = _xla.ArrayImpl
LoadedExecutable = _xla.LoadedExecutable
DeviceList = _xla.DeviceList
OpSharding = _xla.OpSharding
HloSharding = _xla.HloSharding
Sharding = _xla.Sharding
NamedSharding = _xla.NamedSharding
SingleDeviceSharding = _xla.SingleDeviceSharding
PmapSharding = _xla.PmapSharding
GSPMDSharding = _xla.GSPMDSharding
PjRtLayout = _xla.PjRtLayout
AutotuneCacheMode = _xla.AutotuneCacheMode
ResultAccuracyMode = _xla.ResultAccuracy_Mode


def LoadedExecutable_execute(self, arguments, device=None):
  del device
  results = self.execute_sharded(arguments)
  return [x[0] for x in results.disassemble_into_single_device_arrays()]


def LoadedExecutable_execute_with_token(self, arguments, device=None):
  del device
  results = self.execute_sharded(arguments, with_tokens=True)
  return (
      [x[0] for x in results.disassemble_into_single_device_arrays()],
      results.consume_token().get_token(0),
  )


LoadedExecutable.execute = LoadedExecutable_execute
LoadedExecutable.execute_with_token = LoadedExecutable_execute_with_token


class CustomCallTargetTraits(enum.IntFlag):
  DEFAULT = 0
  # Calls to custom call are safe to trace into the command buffer. It means
  # that calls to custom call always launch exactly the same device operations
  # (can depend on attribute values) that can be captured and then replayed.
  #
  # Supported only for custom calls implemented with XLA FFI.
  COMMAND_BUFFER_COMPATIBLE = 1


class CustomCallHandler(Protocol):

  def __call__(
      self,
      name: str,
      fn: Any,
      platform: str,
      /,
      api_version: int = ...,
      traits: CustomCallTargetTraits = ...,
  ) -> None:
    ...


_custom_callback_handler: dict[str, CustomCallHandler] = {}
# Key is xla_platform_name, value is (function_name, function, api_version)
_custom_callback: dict[
    str, list[tuple[str, Any, int, CustomCallTargetTraits]]
] = {}
_custom_callback_lock = threading.Lock()


def register_custom_call_target(
    name: str,
    fn: Any,
    platform: str = 'cpu',
    api_version: int = 0,
    traits: CustomCallTargetTraits = CustomCallTargetTraits.DEFAULT,
) -> None:
  """Registers a custom call target.

  Args:
    name: bytes containing the name of the function.
    fn: a PyCapsule object containing the function pointer.
    platform: the target platform.
    api_version: the XLA FFI version to use. Supported versions are: 0 for the
      untyped FFI and 1 for the typed FFI.
    traits: custom call traits corresponding to XLA FFI handler traits.
  """
  # To support AMD GPUs, we need to have xla_platform_names["gpu"] == "ROCM"
  # Since that is hardcoded to CUDA, we are using the following as workaround.
  xla_platform_name = xla_platform_names.get(platform, platform)
  with _custom_callback_lock:
    if xla_platform_name in _custom_callback_handler:
      _custom_callback_handler[xla_platform_name](
          name, fn, xla_platform_name, api_version, traits
      )
    else:
      _custom_callback.setdefault(xla_platform_name, []).append(
          (name, fn, api_version, traits)
      )


def register_custom_call_handler(
    platform: str, handler: CustomCallHandler
) -> None:
  """Registers a custom handler and use it to register existing custom calls.

  If a custom call handler for the platform already exist, calling this method
  is a no-op and it will not register a new handler.

  Args:
    platform: the target platform.
    handler: the function to register a custom call.
  """
  xla_platform_name = xla_platform_names.get(platform, platform)
  with _custom_callback_lock:
    if xla_platform_name in _custom_callback_handler:
      logger.debug(
          'Custom call handler for %s is already register. Will not register a'
          ' new one',
          xla_platform_name,
      )
      return
    _custom_callback_handler[xla_platform_name] = handler
    if xla_platform_name in _custom_callback:
      for name, fn, api_version, traits in _custom_callback[xla_platform_name]:
        handler(name, fn, xla_platform_name, api_version, traits)
      del _custom_callback[xla_platform_name]


class CustomTypeIdHandler(Protocol):

  def __call__(self, name: str, capsule: Any) -> None:
    ...


_custom_type_id_handler: dict[str, CustomTypeIdHandler] = {}
_custom_type_id: dict[str, Any] = {}
_custom_type_id_lock = threading.Lock()


def register_custom_type_id(
    type_name: str,
    type_id: Any,
    platform: str = 'cpu',
) -> None:
  """Register a custom type id for use with the FFI.

  Args:
    type_name: a unique name for the type.
    type_id: a PyCapsule object containing a pointer to the ``ffi::TypeId``.
    platform: the target platform.
  """
  xla_platform_name = xla_platform_names.get(platform, platform)
  with _custom_type_id_lock:
    if xla_platform_name in _custom_type_id_handler:
      _custom_type_id_handler[xla_platform_name](type_name, type_id)
    else:
      _custom_type_id.setdefault(xla_platform_name, []).append(
          (type_name, type_id)
      )


def register_custom_type_id_handler(
    platform: str, handler: CustomTypeIdHandler
) -> None:
  """Register a custom type id handler and use it to register existing type ids.

  If a custom type id handler for the platform already exist, calling this
  method is a no-op and it will not register a new handler.

  Args:
    platform: the target platform.
    handler: the function to register a custom type id.
  """
  xla_platform_name = xla_platform_names.get(platform, platform)
  with _custom_callback_lock:
    if xla_platform_name in _custom_type_id_handler:
      logger.debug(
          'Custom type id handler for %s is already register. Will not '
          'register a new one',
          xla_platform_name,
      )
      return
    _custom_type_id_handler[xla_platform_name] = handler
    if xla_platform_name in _custom_type_id:
      for name, capsule in _custom_type_id[xla_platform_name]:
        handler(name, capsule)
      del _custom_type_id[xla_platform_name]


register_custom_call_partitioner = _xla.register_custom_call_partitioner
encode_inspect_sharding_callback = _xla.encode_inspect_sharding_callback
hlo_sharding_util = _xla.hlo_sharding_util
register_custom_call_as_batch_partitionable = (
    _xla.register_custom_call_as_batch_partitionable
)


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
    padding_config: Union[PaddingConfig, Sequence[tuple[int, int, int]]]
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

  __slots__ = (
      'lhs_contracting_dimensions',
      'rhs_contracting_dimensions',
      'lhs_batch_dimensions',
      'rhs_batch_dimensions',
  )

  def __init__(self):
    self.lhs_contracting_dimensions = []
    self.rhs_contracting_dimensions = []
    self.lhs_batch_dimensions = []
    self.rhs_batch_dimensions = []


def make_dot_dimension_numbers(
    dimension_numbers: Union[
        DotDimensionNumbers,
        tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]],
    ]
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

  __slots__ = (
      'input_batch_dimension',
      'input_feature_dimension',
      'input_spatial_dimensions',
      'kernel_input_feature_dimension',
      'kernel_output_feature_dimension',
      'kernel_spatial_dimensions',
      'output_batch_dimension',
      'output_feature_dimension',
      'output_spatial_dimensions',
  )

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
    dimension_numbers: Union[
        None, ConvolutionDimensionNumbers, tuple[str, str, str]
    ],
    num_spatial_dimensions: int,
) -> ConvolutionDimensionNumbers:
  """Builds a ConvolutionDimensionNumbers object from a specification.

  Args:
    dimension_numbers: optional, either a ConvolutionDimensionNumbers object or
      a tuple (lhs_spec, rhs_spec, out_spec). Each element is a string of length
      N+2 identifying by position: (1) batch dimensions in lhs, rhs, and the
      output with the character 'N', (2) feature dimensions in lhs and the
      output with the character 'C', (3) input and output feature dimensions in
      rhs with the characters 'I' and 'O' respectively, and (4) spatial
      dimension correspondences between lhs, rhs, and the output using any
      distinct characters. For example, to indicate dimension numbers consistent
      with the Conv operation with two spatial dimensions, one could use
      ('NCHW', 'OIHW', 'NCHW'). As another example, to indicate dimension
      numbers consistent with the TensorFlow Conv2D operation, one could use
      ('NHWC', 'HWIO', 'NHWC'). When using the latter form of convolution
      dimension specification, window strides are associated with spatial
      dimension character labels according to the order in which the labels
      appear in the rhs_spec string, so that window_strides[0] is matched with
      the dimension corresponding to the first character appearing in rhs_spec
      that is not 'I' or 'O'. By default, use the same dimension numbering as
      Conv and ConvWithGeneralPadding.
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
        i for i, c in enumerate(rhs_spec) if c not in {'I', 'O'}
    )
    dimension_numbers.input_spatial_dimensions.extend(
        sorted(
            (i for i, c in enumerate(lhs_spec) if c not in {'N', 'C'}),
            key=lambda i: rhs_spec.index(lhs_spec[i]),
        )
    )
    dimension_numbers.output_spatial_dimensions.extend(
        sorted(
            (i for i, c in enumerate(out_spec) if c not in {'N', 'C'}),
            key=lambda i: rhs_spec.index(out_spec[i]),
        )
    )
  return dimension_numbers


class PrecisionConfig:
  """Python representation of a xla.PrecisionConfig protobuf."""

  __slots__ = ('operand_precision',)

  Precision = _xla.PrecisionConfig_Precision

  def __init__(self):
    self.operand_precision = []


class ResultAccuracy:
  """Python representation of a xla.ResultAccuracy protobuf."""

  __slots__ = ('mode', 'atol', 'rtol', 'ulps')

  def __init__(self):
    self.mode = _xla.ResultAccuracy_Mode.DEFAULT
    self.atol = 0.0
    self.rtol = 0.0
    self.ulps = 0


class GatherDimensionNumbers:
  """Python representation of a xla.GatherDimensionNumbers protobuf."""

  __slots__ = (
      'offset_dims',
      'collapsed_slice_dims',
      'start_index_map',
      'index_vector_dim',
  )

  def __init__(self):
    self.offset_dims = []
    self.collapsed_slice_dims = []
    self.start_index_map = []
    self.index_vector_dim = 0


class ScatterDimensionNumbers:
  """Python representation of a xla.ScatterDimensionNumbers protobuf."""

  __slots__ = (
      'update_window_dims',
      'inserted_window_dims',
      'scatter_dims_to_operand_dims',
      'index_vector_dim',
  )

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
batched_copy_array_to_devices_with_sharding = (
    _xla.batched_copy_array_to_devices_with_sharding
)
batched_device_put = _xla.batched_device_put
reorder_shards = _xla.reorder_shards
batched_block_until_ready = _xla.batched_block_until_ready
check_and_canonicalize_memory_kind = _xla.check_and_canonicalize_memory_kind
Layout = _xla.Layout
custom_call_targets = _xla.custom_call_targets
ArrayCopySemantics = _xla.ArrayCopySemantics
