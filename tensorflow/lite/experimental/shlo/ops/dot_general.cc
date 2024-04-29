/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/experimental/shlo/ops/dot_general.h"

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

absl::Status CheckParameters(
    const Tensor& lhs, const Tensor& rhs,
    const absl::Span<int64_t> lhs_batching_dimensions,
    const absl::Span<int64_t> rhs_batching_dimensions,
    const absl::Span<int64_t> lhs_contracting_dimensions,
    const absl::Span<int64_t> rhs_contracting_dimensions, Tensor& output,
    std::array<PrecisionTypes, 2>& precision_configs) {
  const DimensionSize lhsb_size = lhs_batching_dimensions.size();
  const DimensionSize rhsb_size = rhs_batching_dimensions.size();
  const DimensionSize lhsc_size = lhs_contracting_dimensions.size();
  const DimensionSize rhsc_size = rhs_contracting_dimensions.size();
  const size_t lhs_rank = lhs.Rank();
  const size_t rhs_rank = rhs.Rank();
  const size_t output_rank = output.Rank();
  absl::InlinedVector<size_t, 6> lhs_result_dims;
  absl::InlinedVector<size_t, 6> rhs_result_dims;
  absl::InlinedVector<size_t, 6> output_shape_check;

  if (precision_configs.size() != 2) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: Size of precision_config must be two.");
  }
  if (lhsb_size != rhsb_size) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: Size of lhs_batching_dimensions and "
        "rhs_batching_dimensions must be same.");
  } else if (lhsc_size != rhsc_size) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: Size of lhs_contracting_dimensions and "
        "rhs_contracting_dimensions must be same.");
  }
  for (DimensionSize i = 0; i < lhsb_size; ++i) {
    for (DimensionSize j = 0; j < lhsc_size; ++j) {
      if (lhs_batching_dimensions[i] == lhs_contracting_dimensions[j]) {
        return absl::FailedPreconditionError(
            "stablehlo.dot_general: The lhs_batching_dimensions and "
            "lhs_contracting_dimensions must be unique.");
      }
    }
  }
  for (DimensionSize i = 0; i < rhsb_size; ++i) {
    for (DimensionSize j = 0; j < rhsc_size; ++j) {
      if (rhs_batching_dimensions[i] == rhs_contracting_dimensions[j]) {
        return absl::FailedPreconditionError(
            "stablehlo.dot_general: The rhs_batching_dimensions and "
            "rhs_contracting_dimensions must be unique.");
      }
    }
  }
  for (DimensionSize i = 0; i < lhsb_size; ++i) {
    if (lhs_batching_dimensions[i] >= lhs_rank ||
        lhs_batching_dimensions[i] < 0) {
      return absl::FailedPreconditionError(
          "stablehlo.dot_general: Invalid lhs_batching_dimensions index.");
    }
    output_shape_check.push_back(lhs.shape().Dim(lhs_batching_dimensions[i]));
  }
  for (DimensionSize i = 0; i < lhsc_size; ++i) {
    if (lhs_contracting_dimensions[i] >= lhs_rank ||
        lhs_contracting_dimensions[i] < 0) {
      return absl::FailedPreconditionError(
          "stablehlo.dot_general: Invalid lhs_contracting_dimensions index.");
    }
  }
  for (DimensionSize i = 0; i < rhsb_size; ++i) {
    if (rhs_batching_dimensions[i] >= rhs_rank ||
        rhs_batching_dimensions[i] < 0) {
      return absl::FailedPreconditionError(
          "stablehlo.dot_general: Invalid rhs_batching_dimensions index.");
    }
  }
  for (DimensionSize i = 0; i < rhsc_size; ++i) {
    if (rhs_contracting_dimensions[i] >= rhs_rank ||
        rhs_contracting_dimensions[i] < 0) {
      return absl::FailedPreconditionError(
          "stablehlo.dot_general: Invalid rhs_contracting_dimensions index.");
    }
  }
  for (DimensionSize i = 0; i < lhsb_size; ++i) {
    if (lhs.shape().Dim(lhs_batching_dimensions[i]) !=
        rhs.shape().Dim(rhs_batching_dimensions[i])) {
      return absl::FailedPreconditionError(
          "stablehlo.dot_general: The lhs and rhs tensors should have same "
          "batch dimension size.");
    }
  }
  for (DimensionSize i = 0; i < lhsc_size; ++i) {
    if (lhs.shape().Dim(lhs_contracting_dimensions[i]) !=
        rhs.shape().Dim(rhs_contracting_dimensions[i])) {
      return absl::FailedPreconditionError(
          "stablehlo.dot_general: The lhs and rhs tensors should have same "
          "contracting dimension size.");
    }
  }
  for (size_t i = 0; i < lhs_rank; ++i) {
    if ((std::count(lhs_batching_dimensions.begin(),
                    lhs_batching_dimensions.end(), i) == 0) &&
        (std::count(lhs_contracting_dimensions.begin(),
                    lhs_contracting_dimensions.end(), i) == 0)) {
      lhs_result_dims.push_back(i);
    }
  }
  for (size_t i = 0; i < rhs_rank; ++i) {
    if ((std::count(rhs_batching_dimensions.begin(),
                    rhs_batching_dimensions.end(), i) == 0) &&
        (std::count(rhs_contracting_dimensions.begin(),
                    rhs_contracting_dimensions.end(), i) == 0)) {
      rhs_result_dims.push_back(i);
    }
  }
  for (size_t i = 0; i < lhs_result_dims.size(); ++i) {
    output_shape_check.push_back(lhs.shape().Dim(lhs_result_dims[i]));
  }
  for (size_t i = 0; i < rhs_result_dims.size(); ++i) {
    output_shape_check.push_back(rhs.shape().Dim(rhs_result_dims[i]));
  }
  if (output_shape_check.size()) {
    for (size_t i = 0; i < output_rank; ++i) {
      if (output.shape().Dim(i) != output_shape_check[i]) {
        return absl::FailedPreconditionError(
            "stablehlo.dot_general: Invalid output shape.");
      }
    }
  }
  if (lhs.IsPerAxisQuantized()) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: The lhs tensor cannot be per-axis quantized.");
  }
  if (!lhs.IsPerTensorQuantized() && !rhs.IsQuantized()) {
    if (lhs.tensor_element_type() != rhs.tensor_element_type()) {
      return absl::FailedPreconditionError(
          "stablehlo.dot_general: For non-quantized tensors the element type "
          "of lhs and rhs must be the same.");
    }
  }
  if (lhs.IsPerTensorQuantized()) {
    if (rhs.IsQuantized() && !output.IsQuantized()) {
      return absl::FailedPreconditionError(
          "stablehlo.dot_general: If lhs and rhs are quantized tensors, than "
          "the output tensor should also be quantized.");
    } else if (lhs.StorageType() != rhs.StorageType()) {
      return absl::FailedPreconditionError(
          "stablehlo.dot_general: If the lhs and rhs are quantized tensors, "
          "than they should have the same storage type.");
    } else if (rhs.IsPerTensorQuantized()) {
      if (!output.IsPerTensorQuantized()) {
        return absl::FailedPreconditionError(
            "stablehlo.dot_general: If lhs and rhs are per-tensor quantized "
            "than output should also be per-tensor quantized.");
      }
      if (lhs.quantized_per_tensor_element_type().ExpressedType() ==
          rhs.quantized_per_tensor_element_type().ExpressedType()) {
        if (lhs.quantized_per_tensor_element_type().ExpressedType() !=
            output.quantized_per_tensor_element_type().ExpressedType()) {
          return absl::FailedPreconditionError(
              "stablehlo.dot_general: The expressed_type of output tensor must "
              "be the same as the expressed_type of lhs and rhs tensors.");
        }
      }
      auto check_zero_point_value_is_zero = [](const auto& zero_point) -> bool {
        return std::visit(
            [](const auto& v) { return v == static_cast<decltype(v)>(0); },
            zero_point);
      };
      if (!check_zero_point_value_is_zero(
              rhs.quantized_per_tensor_element_type().ZeroPoint())) {
        return absl::FailedPreconditionError(
            "stablehlo.dot_general: The rhs per-tensor should have zero points "
            "as 0.");
      }
    } else if (rhs.IsPerAxisQuantized()) {
      if (output.IsPerTensorQuantized()) {
        if (lhs.quantized_per_tensor_element_type().ExpressedType() ==
            rhs.quantized_per_axis_element_type().ExpressedType()) {
          if (lhs.quantized_per_tensor_element_type().ExpressedType() !=
              output.quantized_per_tensor_element_type().ExpressedType()) {
            return absl::FailedPreconditionError(
                "stablehlo.dot_general: The expressed_type of output must be "
                "the same as the expressed_type of lhs and rhs.");
          }
        }
      } else if (output.IsPerAxisQuantized()) {
        if (lhs.quantized_per_tensor_element_type().ExpressedType() ==
            rhs.quantized_per_axis_element_type().ExpressedType()) {
          if (lhs.quantized_per_tensor_element_type().ExpressedType() !=
              output.quantized_per_axis_element_type().ExpressedType()) {
            return absl::FailedPreconditionError(
                "stablehlo.dot_general: The expressed_type of output must be "
                "the same as the expressed_type of lhs and rhs.");
          }
        }
      }
      auto check_zero_points_values_are_zero =
          [](const auto& zero_points) -> bool {
        return std::visit(
            [](const auto& v) {
              return std::all_of(v.begin(), v.end(), [](const auto value) {
                return value == static_cast<decltype(value)>(0);
              });
            },
            zero_points);
      };
      if (!check_zero_points_values_are_zero(
              rhs.quantized_per_axis_element_type().ZeroPoints())) {
        return absl::FailedPreconditionError(
            "stablehlo.dot_general: The rhs per-axis should have zero points "
            "as 0.");
      }
    } else if (rhs.IsPerAxisQuantized()) {
      for (DimensionSize i = 0; i < rhsc_size; ++i) {
        if (rhs_contracting_dimensions[i] ==
            rhs.quantized_per_axis_element_type().QuantizedDimension()) {
          return absl::FailedPreconditionError(
              "stablehlo.dot_general: If the rhs is per-axis quantized than "
              "the quantization_dimensions of rhs should not be in "
              "rhs_contracting_dimensions.");
        }
      }
    }
  }
  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status EvaluateImpl(DotGeneralOp& op, const Tensor& lhs,
                          const Tensor& rhs,
                          absl::Span<int64_t> lhs_batching_dimensions,
                          absl::Span<int64_t> rhs_batching_dimensions,
                          absl::Span<int64_t> lhs_contracting_dimensions,
                          absl::Span<int64_t> rhs_contracting_dimensions,
                          Tensor& output) {
  using StorageT = StorageType<storage_type>;
  const StorageT* lhs_data = lhs.GetDataAs<storage_type>();
  const StorageT* rhs_data = rhs.GetDataAs<storage_type>();
  StorageT* output_data = output.GetDataAs<storage_type>();
  const DimensionSize lhs_size = lhs.NumElements();
  const DimensionSize rhs_size = rhs.NumElements();
  const DimensionSize output_size = output.NumElements();
  const size_t lhs_rank = lhs.Rank();
  const size_t rhs_rank = rhs.Rank();
  const size_t output_rank = output.Rank();
  const DimensionSize lhsb_size = lhs_batching_dimensions.size();
  const DimensionSize lhsc_size = lhs_contracting_dimensions.size();

  // function to generate indices for output
  auto GenerateIndices = [&](size_t index) -> void {
    size_t rank = op.output_shape.size();
    size_t divisor = 1;
    for (size_t i = 0, j = rank - 1; i < rank; ++i, --j) {
      op.output_index[j] = (index / divisor) % op.output_shape[j];
      divisor *= op.output_shape[j];
    }
  };
  // function to incremement lhs and rhs indices
  auto IncrementIndices = [&]() -> bool {
    if (lhsc_size == 0) return false;
    for (DimensionSize i = lhsc_size - 1; i >= 0; --i) {
      op.lhs_index[lhs_contracting_dimensions[i]]++;
      op.rhs_index[rhs_contracting_dimensions[i]]++;
      if (op.lhs_index[lhs_contracting_dimensions[i]] <
          lhs.shape().Dim(lhs_contracting_dimensions[i]))
        return true;
      if (i == 0) return false;
      op.lhs_index[lhs_contracting_dimensions[i]] = 0;
      op.rhs_index[rhs_contracting_dimensions[i]] = 0;
    }
    return true;
  };
  // pre compute helper for lhs and rhs indices
  DimensionSize lhs_dim_accumulator = 1, rhs_dim_accumulator = 1;
  for (size_t i = 0; i < lhs_rank; ++i) {
    lhs_dim_accumulator *= lhs.shape().Dim(i);
    op.lhs_index_helper[i] = lhs_size / lhs_dim_accumulator;
  }
  for (size_t i = 0; i < rhs_rank; ++i) {
    rhs_dim_accumulator *= rhs.shape().Dim(i);
    op.rhs_index_helper[i] = rhs_size / rhs_dim_accumulator;
  }

  StorageT output_element(0);
  DimensionSize lhs_element_index = 0, rhs_element_index = 0;
  for (DimensionSize k = 0; k < output_size; ++k, ++output_data) {
    GenerateIndices(k);
    absl::c_fill(op.lhs_index, 0);
    absl::c_fill(op.rhs_index, 0);
    size_t result_dim = 0;
    for (size_t i = 0; i < lhsb_size; ++i, ++result_dim) {
      op.lhs_index[lhs_batching_dimensions[i]] = op.output_index[result_dim];
      op.rhs_index[rhs_batching_dimensions[i]] = op.output_index[result_dim];
    }
    for (size_t i = 0; i < op.lhs_result_dims.size(); ++i, ++result_dim) {
      op.lhs_index[op.lhs_result_dims[i]] = op.output_index[result_dim];
    }
    for (size_t i = 0; i < op.rhs_result_dims.size(); ++i, ++result_dim) {
      op.rhs_index[op.rhs_result_dims[i]] = op.output_index[result_dim];
    }
    output_element = 0;
    while (true) {
      lhs_element_index = 0;
      rhs_element_index = 0;
      for (size_t i = 0; i < lhs_rank; ++i) {
        lhs_element_index += op.lhs_index[i] * op.lhs_index_helper[i];
      }
      for (size_t i = 0; i < rhs_rank; ++i) {
        rhs_element_index += op.rhs_index[i] * op.rhs_index_helper[i];
      }
      output_element +=
          lhs_data[lhs_element_index] * rhs_data[rhs_element_index];
      if (!IncrementIndices()) {
        break;
      }
    }
    *output_data = output_element;
  }
  return absl::OkStatus();
}

template <DataType storage_type, DataType expressed_type>
void DequantizeOpQuantizePerTensor(DotGeneralOp& op, const Tensor& lhs,
                                   const Tensor& rhs, Tensor& output) {
  using StorageT = StorageType<storage_type>;
  using ExpressedT = StorageType<expressed_type>;

  std::vector<ExpressedT> lhs_values(lhs.NumElements());
  const Shape lhs_shape = lhs.shape();
  Tensor lhs_dequantized{
      .type = TensorType{.shape = lhs_shape, .element_type = expressed_type},
      .data = lhs_values.data()};
  std::vector<ExpressedT> rhs_values(rhs.NumElements());
  const Shape rhs_shape = rhs.shape();
  Tensor rhs_dequantized{
      .type = TensorType{.shape = rhs_shape, .element_type = expressed_type},
      .data = rhs_values.data()};
  std::vector<ExpressedT> output_values(output.NumElements());
  const Shape result_shape = output.shape();
  Tensor output_dequantized{
      .type = TensorType{.shape = result_shape, .element_type = expressed_type},
      .data = output_values.data()};

  const DimensionSize lhs_num_elements = lhs.NumElements();
  const StorageT lhs_zero_point =
      lhs.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT lhs_scale =
      lhs.quantized_per_tensor_element_type().ScaleAs<expressed_type>();
  const DimensionSize rhs_num_elements = rhs.NumElements();
  const StorageT rhs_zero_point =
      rhs.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT rhs_scale =
      rhs.quantized_per_tensor_element_type().ScaleAs<expressed_type>();
  const DimensionSize output_num_elements = output.NumElements();
  const StorageT output_zero_point =
      output.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT output_scale =
      output.quantized_per_tensor_element_type().ScaleAs<expressed_type>();

  const StorageT* lhs_data = lhs.GetDataAs<storage_type>();
  ExpressedT* lhs_dequantized_data =
      lhs_dequantized.GetDataAs<expressed_type>();
  const StorageT* rhs_data = rhs.GetDataAs<storage_type>();
  ExpressedT* rhs_dequantized_data =
      rhs_dequantized.GetDataAs<expressed_type>();
  StorageT* output_data = output.GetDataAs<storage_type>();
  ExpressedT* output_dequantized_data =
      output_dequantized.GetDataAs<expressed_type>();

  for (DimensionSize i = 0; i < lhs_num_elements;
       ++i, ++lhs_data, ++lhs_dequantized_data)
    *lhs_dequantized_data = Dequantize(*lhs_data, lhs_zero_point, lhs_scale);

  for (DimensionSize i = 0; i < rhs_num_elements;
       ++i, ++rhs_data, ++rhs_dequantized_data)
    *rhs_dequantized_data = Dequantize(*rhs_data, rhs_zero_point, rhs_scale);

  auto status =
      Evaluate(op, lhs_dequantized, rhs_dequantized, output_dequantized);
  const ExpressedT inv_scale = static_cast<ExpressedT>(1) / output_scale;
  for (DimensionSize i = 0; i < output_num_elements;
       ++i, ++output_dequantized_data, ++output_data) {
    *output_data = Quantize<storage_type, expressed_type>(
        *output_dequantized_data, output_zero_point, inv_scale);
  }
}

template <typename StorageT, typename ExpressedT>
void DequantizeOpQuantizePerAxisImpl(
    const Shape& shape, const Axis quantization_dimension,
    const StorageT quantization_min, const StorageT quantization_max,
    const absl::Span<const StorageT> input_zero_points,
    const absl::Span<const ExpressedT> input_scales, const Strides& strides,
    const StorageT* input_data, ExpressedT* input_dequantized_data,
    const size_t depth, size_t quantization_index) {
  const DimensionSize dim = shape.Dim(depth);
  if (depth + 1 >= shape.Rank()) {
    for (DimensionSize i = 0; i < dim; ++i) {
      if (depth == quantization_dimension) {
        quantization_index = i;
      }
      *input_dequantized_data =
          Dequantize(*input_data, input_zero_points[quantization_index],
                     input_scales[quantization_index]);
      input_data += strides[depth];
      input_dequantized_data += strides[depth];
    }
  } else {
    for (DimensionSize i = 0; i < dim; ++i) {
      if (depth == quantization_dimension) {
        quantization_index = i;
      }
      DequantizeOpQuantizePerAxisImpl(
          shape, quantization_dimension, quantization_min, quantization_max,
          input_zero_points, input_scales, strides, input_data,
          input_dequantized_data, depth + 1, quantization_index);
      input_data += strides[depth];
      input_dequantized_data += strides[depth];
    }
  }
}

template <typename StorageT, typename ExpressedT>
void QuantizeOpQuantizePerAxisImpl(
    const Shape& shape, const Axis quantization_dimension,
    const StorageT quantization_min, const StorageT quantization_max,
    const absl::Span<const StorageT> input_zero_points,
    const absl::Span<const ExpressedT> input_scales, const Strides& strides,
    StorageT* input_data, ExpressedT* input_dequantized_data,
    const size_t depth, size_t quantization_index) {
  const DimensionSize dim = shape.Dim(depth);
  if (depth + 1 >= shape.Rank()) {
    for (DimensionSize i = 0; i < dim; ++i) {
      if (depth == quantization_dimension) {
        quantization_index = i;
      }
      *input_data = Quantize<StorageT, ExpressedT>(
          *input_dequantized_data, input_zero_points[quantization_index],
          static_cast<ExpressedT>(1 / input_scales[quantization_index]),
          quantization_min, quantization_max);
      input_data += strides[depth];
      input_dequantized_data += strides[depth];
    }
  } else {
    for (DimensionSize i = 0; i < dim; ++i) {
      if (depth == quantization_dimension) {
        quantization_index = i;
      }
      QuantizeOpQuantizePerAxisImpl(
          shape, quantization_dimension, quantization_min, quantization_max,
          input_zero_points, input_scales, strides, input_data,
          input_dequantized_data, depth + 1, quantization_index);
      input_data += strides[depth];
      input_dequantized_data += strides[depth];
    }
  }
}

template <DataType storage_type, DataType expressed_type>
void DequantizeOpQuantizePerAxis(DotGeneralOp& op, const Tensor& lhs,
                                 const Tensor& rhs, Tensor& result) {
  using StorageT = StorageType<storage_type>;
  using ExpressedT = StorageType<expressed_type>;

  std::vector<ExpressedT> lhs_values(lhs.NumElements());
  const Shape lhs_shape = lhs.shape();
  Tensor lhs_dequantized{
      .type = TensorType{.shape = lhs_shape, .element_type = expressed_type},
      .data = lhs_values.data()};
  std::vector<ExpressedT> rhs_values(rhs.NumElements());
  const Shape rhs_shape = rhs.shape();
  Tensor rhs_dequantized{
      .type = TensorType{.shape = rhs_shape, .element_type = expressed_type},
      .data = rhs_values.data()};
  std::vector<ExpressedT> result_values(result.NumElements());
  const Shape result_shape = result.shape();
  Tensor result_dequantized{
      .type = TensorType{.shape = result_shape, .element_type = expressed_type},
      .data = result_values.data()};

  const StorageT* lhs_data = lhs.GetDataAs<storage_type>();
  ExpressedT* lhs_dequantized_data =
      lhs_dequantized.GetDataAs<expressed_type>();
  const StorageT* rhs_data = rhs.GetDataAs<storage_type>();
  ExpressedT* rhs_dequantized_data =
      rhs_dequantized.GetDataAs<expressed_type>();
  StorageT* result_data = result.GetDataAs<storage_type>();
  ExpressedT* result_dequantized_data =
      result_dequantized.GetDataAs<expressed_type>();

  const DimensionSize lhs_num_elements = lhs.NumElements();
  const StorageT lhs_zero_point =
      lhs.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT lhs_scale =
      lhs.quantized_per_tensor_element_type().ScaleAs<expressed_type>();
  for (DimensionSize i = 0; i < lhs_num_elements;
       ++i, ++lhs_data, ++lhs_dequantized_data)
    *lhs_dequantized_data = Dequantize(*lhs_data, lhs_zero_point, lhs_scale);

  const Shape& shape = rhs.shape();
  const Axis rhs_quantization_dimension =
      rhs.quantized_per_axis_element_type().QuantizedDimension();
  const absl::Span<const StorageT> rhs_zero_points =
      rhs.quantized_per_axis_element_type().ZeroPointsAs<storage_type>();
  const absl::Span<const ExpressedT> rhs_scales =
      rhs.quantized_per_axis_element_type().ScalesAs<expressed_type>();

  const Strides& strides = ComputeStrides(shape);

  DequantizeOpQuantizePerAxisImpl(
      shape, rhs_quantization_dimension, Storage<storage_type>::kMinValue,
      Storage<storage_type>::kMaxValue, rhs_zero_points, rhs_scales, strides,
      rhs_data, rhs_dequantized_data, /*depth=*/0, /*quantization_index=*/0);
  auto status =
      Evaluate(op, lhs_dequantized, rhs_dequantized, result_dequantized);

  if (result.IsPerAxisQuantized()) {
    const Shape& shape = result.shape();
    const Axis result_quantization_dimension =
        result.quantized_per_axis_element_type().QuantizedDimension();
    const absl::Span<const StorageT> result_zero_points =
        result.quantized_per_axis_element_type().ZeroPointsAs<storage_type>();
    const absl::Span<const ExpressedT> result_scales =
        result.quantized_per_axis_element_type().ScalesAs<expressed_type>();
    const Strides& strides = ComputeStrides(shape);
    QuantizeOpQuantizePerAxisImpl(
        shape, result_quantization_dimension, Storage<storage_type>::kMinValue,
        Storage<storage_type>::kMaxValue, result_zero_points, result_scales,
        strides, result_data, result_dequantized_data, /*depth=*/0,
        /*quantization_index=*/0);
  } else {
    const DimensionSize result_num_elements = result.NumElements();
    const StorageT result_zero_point =
        result.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
    const ExpressedT result_scale =
        result.quantized_per_tensor_element_type().ScaleAs<expressed_type>();
    const ExpressedT inv_scale = static_cast<ExpressedT>(1 / result_scale);
    for (DimensionSize i = 0; i < result_num_elements;
         ++i, ++result_dequantized_data, ++result_data) {
      *result_data = Quantize<storage_type, expressed_type>(
          *result_dequantized_data, result_zero_point, inv_scale);
    }
  }
}

DotGeneralOp Create(DotGeneralOp::Attributes attributes) {
  return {.attributes = attributes};
}

absl::Status Prepare(DotGeneralOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(
      CheckParameters(lhs, rhs, op.attributes.lhs_batching_dimensions,
                      op.attributes.rhs_batching_dimensions,
                      op.attributes.lhs_contracting_dimensions,
                      op.attributes.rhs_contracting_dimensions, output,
                      op.attributes.precision_configs));

  for (size_t i = 0; i < lhs.Rank(); ++i) {
    if ((std::count(op.attributes.lhs_batching_dimensions.begin(),
                    op.attributes.lhs_batching_dimensions.end(), i) == 0) &&
        (std::count(op.attributes.lhs_contracting_dimensions.begin(),
                    op.attributes.lhs_contracting_dimensions.end(), i) == 0)) {
      op.lhs_result_dims.push_back(i);
    }
  }
  for (size_t i = 0; i < rhs.Rank(); ++i) {
    if ((std::count(op.attributes.rhs_batching_dimensions.begin(),
                    op.attributes.rhs_batching_dimensions.end(), i) == 0) &&
        (std::count(op.attributes.rhs_contracting_dimensions.begin(),
                    op.attributes.rhs_contracting_dimensions.end(), i) == 0)) {
      op.rhs_result_dims.push_back(i);
    }
  }

  op.lhs_index.resize(lhs.Rank());
  op.rhs_index.resize(rhs.Rank());
  op.lhs_index_helper.resize(lhs.Rank());
  op.rhs_index_helper.resize(rhs.Rank());
  op.output_index.resize(output.Rank());
  for (size_t i = 0; i < output.Rank(); ++i) {
    op.output_shape.push_back(output.shape().Dim(i));
  }
  return absl::OkStatus();
}

absl::Status Evaluate(DotGeneralOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output) {
  if (lhs.IsPerTensorQuantized()) {
    if (rhs.IsPerTensorQuantized()) {
      DISPATCH_QUANTIZED(
          DequantizeOpQuantizePerTensor,
          lhs.quantized_per_tensor_element_type().StorageType(),
          lhs.quantized_per_tensor_element_type().ExpressedType(), op, lhs, rhs,
          output);
    } else if (rhs.IsPerAxisQuantized()) {
      DISPATCH_QUANTIZED(
          DequantizeOpQuantizePerAxis,
          lhs.quantized_per_tensor_element_type().StorageType(),
          lhs.quantized_per_tensor_element_type().ExpressedType(), op, lhs, rhs,
          output);
    }
  } else {
    DISPATCH_BOOL_INT_FLOAT(EvaluateImpl, output.tensor_element_type(), op, lhs,
                            rhs, op.attributes.lhs_batching_dimensions,
                            op.attributes.rhs_batching_dimensions,
                            op.attributes.lhs_contracting_dimensions,
                            op.attributes.rhs_contracting_dimensions, output);
  }
  return absl::FailedPreconditionError(
      "stablehlo.dot_general: Unsupported tensor type.");
}

}  // namespace shlo_ref
