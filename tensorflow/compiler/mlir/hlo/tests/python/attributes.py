# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Test for Python APIs accessing MHLO attributes."""

# pylint: disable=wildcard-import,undefined-variable

from mlir.dialects.mhlo import *
from mlir.ir import *


def run(f):
  with Context() as context:
    register_mhlo_dialect(context)
    f()
  return f


@run
def test_scatter_dimension_numbers():
  """Check that ScatterDimensionNumbers attributes is available and usable."""

  attr = ScatterDimensionNumbers.get(
      update_window_dims=[1, 2, 3],
      inserted_window_dims=[4, 6],
      scattered_dims_to_operand_dims=[6, 7],
      index_vector_dim=8)
  assert attr is not None
  assert str(attr) == ("#mhlo.scatter<update_window_dims = [1, 2, 3], "
                       "inserted_window_dims = [4, 6], "
                       "scatter_dims_to_operand_dims = [6, 7], "
                       "index_vector_dim = 8>")
  assert attr.update_window_dims == [1, 2, 3]
  assert attr.inserted_window_dims == [4, 6]
  assert attr.scattered_dims_to_operand_dims == [6, 7]
  assert attr.index_vector_dim == 8


@run
def test_gather_dimension_numbers():
  """Check that GatherDimensionNumbers attributes is available and usable."""

  attr = GatherDimensionNumbers.get(
      offset_dims=[1, 2],
      collapsed_slice_dims=[3, 4, 5],
      start_index_map=[6],
      index_vector_dim=7)
  assert attr is not None
  assert str(attr) == ("#mhlo.gather<offset_dims = [1, 2], "
                       "collapsed_slice_dims = [3, 4, 5], "
                       "start_index_map = [6], "
                       "index_vector_dim = 7>")
  assert attr.offset_dims == [1, 2]
  assert attr.collapsed_slice_dims == [3, 4, 5]
  assert attr.start_index_map == [6]
  assert attr.index_vector_dim == 7


@run
def test_dot_dimension_numbers():
  """Check that DotDimensionNumbers attributes is available and usable."""

  attr = DotDimensionNumbers.get(
      lhs_batching_dimensions=[0, 1],
      rhs_batching_dimensions=[2, 3],
      lhs_contracting_dimensions=[4, 5],
      rhs_contracting_dimensions=[6, 7])
  assert attr is not None
  assert str(attr) == ("#mhlo.dot<lhs_batching_dimensions = [0, 1], "
                       "rhs_batching_dimensions = [2, 3], "
                       "lhs_contracting_dimensions = [4, 5], "
                       "rhs_contracting_dimensions = [6, 7]>")
  assert attr.lhs_batching_dimensions == [0, 1]
  assert attr.rhs_batching_dimensions == [2, 3]
  assert attr.lhs_contracting_dimensions == [4, 5]
  assert attr.rhs_contracting_dimensions == [6, 7]


@run
def test_conv_dimension_numbers():
  """Check that DotDimensionNumbers attributes is available and usable."""

  attr = ConvDimensionNumbers.get(
      input_batch_dimension=0,
      input_feature_dimension=4,
      input_spatial_dimensions=[1, 2, 3],
      kernel_input_feature_dimension=1,
      kernel_output_feature_dimension=2,
      kernel_spatial_dimensions=[0, 3],
      output_batch_dimension=1,
      output_feature_dimension=3,
      output_spatial_dimensions=[0, 2])
  assert str(attr) == "#mhlo.conv<[b, 0, 1, 2, f]x[0, i, o, 1]->[0, b, 1, f]>"
  assert attr is not None
  assert attr.input_batch_dimension == 0
  assert attr.input_feature_dimension == 4
  assert attr.input_spatial_dimensions == [1, 2, 3]
  assert attr.kernel_input_feature_dimension == 1
  assert attr.kernel_output_feature_dimension == 2
  assert attr.kernel_spatial_dimensions == [0, 3]
  assert attr.output_batch_dimension == 1
  assert attr.output_feature_dimension == 3
  assert attr.output_spatial_dimensions == [0, 2]


@run
def test_comparison_direction():
  """Check that ComparisonDirection attribute is available and usable."""

  attr = ComparisonDirectionAttr.get("EQ")
  assert attr is not None
  assert str(attr) == ("#mhlo<\"comparison_direction EQ\">")
  assert attr.comparison_direction == "EQ"


@run
def test_comparison_type():
  """Check that ComparisonType attribute is available and usable."""

  attr = ComparisonTypeAttr.get("TOTALORDER")
  assert attr is not None
  assert str(attr) == ("#mhlo<\"comparison_type TOTALORDER\">")
  assert attr.comparison_type == "TOTALORDER"


@run
def test_precision():
  """Check that Precision attribute is available and usable."""

  attr = PrecisionAttr.get("DEFAULT")
  assert attr is not None
  assert str(attr) == ("#mhlo<\"precision DEFAULT\">")
  assert attr.precision_type == "DEFAULT"


@run
def test_fft_type():
  """Check that FftType attribute is available and usable."""

  attr = FftTypeAttr.get("FFT")
  assert attr is not None
  assert str(attr) == ("#mhlo<\"fft_type FFT\">")
  assert attr.fft_type == "FFT"


@run
def test_dequantize_mode():
  """Check that DequantizeMode attribute is available and usable."""

  attr = DequantizeModeAttr.get("MIN_COMBINED")
  assert attr is not None
  assert str(attr) == ("#mhlo<\"dequantize_mode MIN_COMBINED\">")
  assert attr.dequantize_mode == "MIN_COMBINED"


@run
def test_transpose_type():
  """Check that Transpose attribute is available and usable."""

  attr = TransposeAttr.get("TRANSPOSE")
  assert attr is not None
  assert str(attr) == ("#mhlo<\"transpose TRANSPOSE\">")
  assert attr.transpose_type == "TRANSPOSE"


@run
def test_fusion_kind():
  """Check that FusionKind attribute is available and usable."""

  attr = FusionKindAttr.get("kLoop")
  assert attr is not None
  assert str(attr) == ("#mhlo<\"fusion_kind kLoop\">")
  assert attr.fusion_kind == "kLoop"

@run
def test_rng_algorithm():
  """Check that RngAlgorithm attribute is available and usable."""

  attr = RngAlgorithmAttr.get("DEFAULT")
  assert attr is not None
  assert str(attr) == ("#mhlo<\"rng_algorithm DEFAULT\">")
  assert attr.rng_algorithm == "DEFAULT"
