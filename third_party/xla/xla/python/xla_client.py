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

# pylint: disable=unused-import

import enum
import inspect
import os
from typing import Sequence

from jax.jaxlib.xla.xla_client import *  # pylint: disable=wildcard-import
from jax.jaxlib.xla.xla_client import _xla
from jax.jaxlib.xla.xla_client import PrimitiveType
from jax.jaxlib.xla.xla_client import Shape

import ml_dtypes
import numpy as np


from . import _ops as ops
from . import _profiler as profiler
from ._xla_builder import XlaBuilder
from ._xla_builder import XlaOp


XLA_ELEMENT_TYPE_TO_DTYPE = {
    PrimitiveType.PRED: np.dtype('bool'),
    PrimitiveType.S4: np.dtype(ml_dtypes.int4),
    PrimitiveType.S8: np.dtype('int8'),
    PrimitiveType.S16: np.dtype('int16'),
    PrimitiveType.S32: np.dtype('int32'),
    PrimitiveType.S64: np.dtype('int64'),
    PrimitiveType.U4: np.dtype(ml_dtypes.uint4),
    PrimitiveType.U8: np.dtype('uint8'),
    PrimitiveType.U16: np.dtype('uint16'),
    PrimitiveType.U32: np.dtype('uint32'),
    PrimitiveType.U64: np.dtype('uint64'),
    PrimitiveType.F4E2M1FN: np.dtype(ml_dtypes.float4_e2m1fn),
    PrimitiveType.F8E3M4: np.dtype(ml_dtypes.float8_e3m4),
    PrimitiveType.F8E4M3: np.dtype(ml_dtypes.float8_e4m3),
    PrimitiveType.F8E4M3FN: np.dtype(ml_dtypes.float8_e4m3fn),
    PrimitiveType.F8E4M3B11FNUZ: np.dtype(ml_dtypes.float8_e4m3b11fnuz),
    PrimitiveType.F8E4M3FNUZ: np.dtype(ml_dtypes.float8_e4m3fnuz),
    PrimitiveType.F8E5M2: np.dtype(ml_dtypes.float8_e5m2),
    PrimitiveType.F8E5M2FNUZ: np.dtype(ml_dtypes.float8_e5m2fnuz),
    PrimitiveType.F8E8M0FNU: np.dtype(ml_dtypes.float8_e8m0fnu),
    PrimitiveType.BF16: np.dtype(ml_dtypes.bfloat16),
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


class PrecisionConfig:
  """Python representation of a xla.PrecisionConfig protobuf."""

  __slots__ = ('operand_precision',)

  Precision = ops.PrecisionConfig_Precision  # pylint: disable=invalid-name

  def __init__(self):
    self.operand_precision = []


FftType = ops.FftType
ShapeIndex = ops.ShapeIndex
ResultAccuracyMode = ops.ResultAccuracy_Mode


class ResultAccuracy:
  """Python representation of a xla.ResultAccuracy protobuf."""

  __slots__ = ('mode', 'atol', 'rtol', 'ulps')

  def __init__(self):
    self.mode = ops.ResultAccuracy_Mode.DEFAULT
    self.atol = 0.0
    self.rtol = 0.0
    self.ulps = 0


class OpMetadata:
  """Python representation of a xla.OpMetadata protobuf."""

  __slots__ = ('op_type', 'op_name', 'source_file', 'source_line')

  def __init__(self, op_type='', op_name='', source_file='', source_line=0):
    self.op_type = op_type
    self.op_name = op_name
    self.source_file = source_file
    self.source_line = source_line


def current_source_info_metadata(op_type=None, op_name=None, skip_frames=1):
  """Helper for use in source mapping that returns an OpMetadata object."""
  full_filename, lineno = inspect.stack()[skip_frames][1:3]
  filename = os.path.basename(full_filename)
  return OpMetadata(
      op_type=op_type, op_name=op_name, source_file=filename, source_line=lineno
  )


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
    padding_config: PaddingConfig | Sequence[tuple[int, int, int]],
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
    dimension_numbers: (
        DotDimensionNumbers
        | tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]
    ),
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
    dimension_numbers: (
        None | ConvolutionDimensionNumbers | tuple[str, str, str]
    ),
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
