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
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

absl::Status CheckParameters(const Tensor& lhs, const Tensor& rhs,
                             const Tensor& lhs_batching_dimensions,
                             const Tensor& rhs_batching_dimensions,
                             const Tensor& lhs_contracting_dimensions,
                             const Tensor& rhs_contracting_dimensions,
                             Tensor& output) {
  const int32_t* lhsb = lhs_batching_dimensions.GetDataAs<DataType::kSI32>();
  const int32_t* rhsb = rhs_batching_dimensions.GetDataAs<DataType::kSI32>();
  const int32_t* lhsc = lhs_contracting_dimensions.GetDataAs<DataType::kSI32>();
  const int32_t* rhsc = rhs_contracting_dimensions.GetDataAs<DataType::kSI32>();
  const DimensionSize lhsb_size = lhs_batching_dimensions.NumElements();
  const DimensionSize rhsb_size = rhs_batching_dimensions.NumElements();
  const DimensionSize lhsc_size = lhs_contracting_dimensions.NumElements();
  const DimensionSize rhsc_size = rhs_contracting_dimensions.NumElements();
  const size_t lhs_rank = lhs.Rank();
  const size_t rhs_rank = rhs.Rank();
  const size_t output_rank = output.Rank();
  std::vector<size_t> lhs_result_dims;
  std::vector<size_t> rhs_result_dims;
  std::vector<size_t> output_shape_check;

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
      if (lhsb[i] == lhsc[j]) {
        return absl::FailedPreconditionError(
            "stablehlo.dot_general: The lhs_batching_dimensions and "
            "lhs_contracting_dimensions must be unique.");
      }
    }
  }
  for (DimensionSize i = 0; i < rhsb_size; ++i) {
    for (DimensionSize j = 0; j < rhsc_size; ++j) {
      if (rhsb[i] == rhsc[j]) {
        return absl::FailedPreconditionError(
            "stablehlo.dot_general: The rhs_batching_dimensions and "
            "rhs_contracting_dimensions must be unique.");
      }
    }
  }
  for (DimensionSize i = 0; i < lhsb_size; ++i) {
    if (lhsb[i] >= lhs_rank || lhsb[i] < 0) {
      return absl::FailedPreconditionError(
          "stablehlo.dot_general: Invalid lhs_batching_dimensions index.");
    }
    output_shape_check.push_back(lhs.shape().Dim(lhsb[i]));
  }
  for (DimensionSize i = 0; i < lhsc_size; ++i) {
    if (lhsc[i] >= lhs_rank || lhsc[i] < 0) {
      return absl::FailedPreconditionError(
          "stablehlo.dot_general: Invalid lhs_contracting_dimensions index.");
    }
  }
  for (DimensionSize i = 0; i < rhsb_size; ++i) {
    if (rhsb[i] >= rhs_rank || rhsb[i] < 0) {
      return absl::FailedPreconditionError(
          "stablehlo.dot_general: Invalid rhs_batching_dimensions index.");
    }
  }
  for (DimensionSize i = 0; i < rhsc_size; ++i) {
    if (rhsc[i] >= rhs_rank || rhsc[i] < 0) {
      return absl::FailedPreconditionError(
          "stablehlo.dot_general: Invalid rhs_contracting_dimensions index.");
    }
  }
  for (DimensionSize i = 0; i < lhsb_size; ++i) {
    if (lhs.shape().Dim(lhsb[i]) != rhs.shape().Dim(rhsb[i])) {
      return absl::FailedPreconditionError(
          "stablehlo.dot_general: The lhs and rhs tensors should have same "
          "batch dimensions.");
    }
  }
  for (DimensionSize i = 0; i < lhsc_size; ++i) {
    if (lhs.shape().Dim(lhsc[i]) != rhs.shape().Dim(rhsc[i])) {
      return absl::FailedPreconditionError(
          "stablehlo.dot_general: The lhs and rhs tensors should have same "
          "contracting dimensions.");
    }
  }
  for (size_t i = 0; i < lhs_rank; ++i) {
    if ((std::count(lhsb, lhsb + lhsb_size, i) == 0) &&
        (std::count(lhsc, lhsc + lhsc_size, i) == 0)) {
      lhs_result_dims.push_back(i);
    }
  }
  for (size_t i = 0; i < rhs_rank; ++i) {
    if ((std::count(rhsb, rhsb + rhsb_size, i) == 0) &&
        (std::count(rhsc, rhsc + rhsc_size, i) == 0)) {
      rhs_result_dims.push_back(i);
    }
  }
  for (size_t i = 0; i < lhs_result_dims.size(); ++i) {
    output_shape_check.push_back(lhs.shape().Dim(lhs_result_dims[i]));
  }
  for (size_t i = 0; i < rhs_result_dims.size(); ++i) {
    output_shape_check.push_back(rhs.shape().Dim(rhs_result_dims[i]));
  }
  for (size_t i = 0; i < output_rank; ++i) {
    if (output.shape().Dim(i) != output_shape_check[i]) {
      return absl::FailedPreconditionError(
          "stablehlo.dot_general: Invalid output shape.");
    }
  }
  if (lhs.IsPerAxisQuantized()) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: The lhs tensor cannot be per-axis quantized.");
  }
  if (!lhs.IsPerTensorQuantized() && !rhs.IsQuantized() &&
      lhs.tensor_element_type() != rhs.tensor_element_type()) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: For non-quantized tensors the element type of "
        "lhs and rhs must be the same.");
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
      auto check_zero_point_value_is_zero =
          [](auto zero_point) -> bool {
        if (std::holds_alternative<I4>(zero_point)) {
          return std::get<I4>(zero_point) == static_cast<I4>(0);
        } else if (std::holds_alternative<int8_t>(zero_point)) {
          return std::get<int8_t>(zero_point) == static_cast<int8_t>(0);
        } else if (std::holds_alternative<int16_t>(zero_point)) {
          return std::get<int16_t>(zero_point) == static_cast<int16_t>(0);
        }
        return false;
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
          [](auto zero_points) -> bool {
        if (std::holds_alternative<absl::InlinedVector<I4, 8>>(zero_points)) {
          auto zero_point_value =
              std::get<absl::InlinedVector<I4, 8>>(zero_points);
          return std::all_of(
              zero_point_value.begin(), zero_point_value.end(),
              [](I4 value) { return value == static_cast<I4>(0); });
        } else if (std::holds_alternative<absl::InlinedVector<int8_t, 8>>(
                       zero_points)) {
          auto zero_point_value =
              std::get<absl::InlinedVector<int8_t, 8>>(zero_points);
          return std::all_of(
              zero_point_value.begin(), zero_point_value.end(),
              [](int8_t value) { return value == static_cast<int8_t>(0); });
        } else if (std::holds_alternative<absl::InlinedVector<int16_t, 8>>(
                       zero_points)) {
          auto zero_point_value =
              std::get<absl::InlinedVector<int16_t, 8>>(zero_points);
          return std::all_of(
              zero_point_value.begin(), zero_point_value.end(),
              [](int16_t value) { return value == static_cast<int16_t>(0); });
        }
        return false;
      };
      if (!check_zero_points_values_are_zero(
              rhs.quantized_per_axis_element_type().ZeroPoints())) {
        return absl::FailedPreconditionError(
            "stablehlo.dot_general: The rhs per-axis should have zero points "
            "as 0.");
      }
    } else if (rhs.IsPerAxisQuantized()) {
      for (DimensionSize i = 0; i < rhsc_size; ++i) {
        if (rhsc[i] ==
                rhs.quantized_per_axis_element_type().QuantizedDimension() ||
            rhsb[i] ==
                rhs.quantized_per_axis_element_type().QuantizedDimension()) {
          return absl::FailedPreconditionError(
              "stablehlo.dot_general: If the rhs is per-axis quantized than "
              "the quantization_dimensions of rhs should not be in "
              "rhs_contracting_dimensions and rhs_batching_dimensions.");
        }
      }
    }
  }
  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status EvaluateImpl(DotGeneralOp& op, const Tensor& lhs,
                          const Tensor& rhs,
                          const Tensor& lhs_batching_dimensions,
                          const Tensor& rhs_batching_dimensions,
                          const Tensor& lhs_contracting_dimensions,
                          const Tensor& rhs_contracting_dimensions,
                          Tensor& output) {
  using StorageT = StorageType<storage_type>;
  const StorageT* lhs_data = lhs.GetDataAs<storage_type>();
  const StorageT* rhs_data = rhs.GetDataAs<storage_type>();
  StorageT* output_data = output.GetDataAs<storage_type>();
  const size_t lhs_size = lhs.NumElements();
  const DimensionSize rhs_size = rhs.NumElements();
  const DimensionSize output_size = output.NumElements();
  const size_t lhs_rank = lhs.Rank();
  const size_t rhs_rank = rhs.Rank();
  const size_t output_rank = output.Rank();

  const int32_t* lhs_batching_dimensions_data =
      lhs_batching_dimensions.GetDataAs<DataType::kSI32>();
  const int32_t* rhs_batching_dimensions_data =
      rhs_batching_dimensions.GetDataAs<DataType::kSI32>();
  const int32_t* lhs_contracting_dimensions_data =
      lhs_contracting_dimensions.GetDataAs<DataType::kSI32>();
  const int32_t* rhs_contracting_dimensions_data =
      rhs_contracting_dimensions.GetDataAs<DataType::kSI32>();
  const DimensionSize lhsb_size = lhs_batching_dimensions.NumElements();
  const DimensionSize lhsc_size = lhs_contracting_dimensions.NumElements();
  const DimensionSize rhsb_size = rhs_batching_dimensions.NumElements();
  const DimensionSize rhsc_size = rhs_contracting_dimensions.NumElements();

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
      op.lhs_index[lhs_contracting_dimensions_data[i]]++;
      op.rhs_index[rhs_contracting_dimensions_data[i]]++;
      if (op.lhs_index[lhs_contracting_dimensions_data[i]] <
          lhs.shape().Dim(lhs_contracting_dimensions_data[i]))
        return true;
      if (i == 0) return false;
      op.lhs_index[lhs_contracting_dimensions_data[i]] = 0;
      op.rhs_index[rhs_contracting_dimensions_data[i]] = 0;
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
    std::fill(op.lhs_index.begin(), op.lhs_index.end(), 0);
    std::fill(op.rhs_index.begin(), op.rhs_index.end(), 0);
    size_t result_dim = 0;
    for (size_t i = 0; i < lhsb_size; ++i, ++result_dim) {
      op.lhs_index[lhs_batching_dimensions_data[i]] =
          op.output_index[result_dim];
      op.rhs_index[rhs_batching_dimensions_data[i]] =
          op.output_index[result_dim];
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

DotGeneralOp Create(DotGeneralOp::Attributes attributes) {
  return {.attributes = attributes};
}

absl::Status Prepare(DotGeneralOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output) {
  if (absl::Status status =
          CheckParameters(lhs, rhs, op.attributes.lhs_batching_dimensions,
                          op.attributes.rhs_batching_dimensions,
                          op.attributes.lhs_contracting_dimensions,
                          op.attributes.rhs_contracting_dimensions, output);
      !status.ok()) {
    return status;
  }

  const int32_t* lhs_batching_dimensions_data =
      op.attributes.lhs_batching_dimensions.GetDataAs<DataType::kSI32>();
  const int32_t* rhs_batching_dimensions_data =
      op.attributes.rhs_batching_dimensions.GetDataAs<DataType::kSI32>();
  const int32_t* lhs_contracting_dimensions_data =
      op.attributes.lhs_contracting_dimensions.GetDataAs<DataType::kSI32>();
  const int32_t* rhs_contracting_dimensions_data =
      op.attributes.rhs_contracting_dimensions.GetDataAs<DataType::kSI32>();
  const DimensionSize lhsb_size =
      op.attributes.lhs_batching_dimensions.NumElements();
  const DimensionSize lhsc_size =
      op.attributes.lhs_contracting_dimensions.NumElements();
  const DimensionSize rhsb_size =
      op.attributes.rhs_batching_dimensions.NumElements();
  const DimensionSize rhsc_size =
      op.attributes.rhs_contracting_dimensions.NumElements();

  for (size_t i = 0; i < lhs.Rank(); ++i) {
    if ((std::count(lhs_batching_dimensions_data,
                    lhs_batching_dimensions_data + lhsb_size, i) == 0) &&
        (std::count(lhs_contracting_dimensions_data,
                    lhs_contracting_dimensions_data + lhsc_size, i) == 0)) {
      op.lhs_result_dims.push_back(i);
    }
  }
  for (size_t i = 0; i < rhs.Rank(); ++i) {
    if ((std::count(rhs_batching_dimensions_data,
                    rhs_batching_dimensions_data + rhsb_size, i) == 0) &&
        (std::count(rhs_contracting_dimensions_data,
                    rhs_contracting_dimensions_data + rhsc_size, i) == 0)) {
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
  // TODO: Add support for Quantized types
  if (!lhs.IsQuantized() && !rhs.IsQuantized()) {
    DISPATCH_INT_FLOAT(EvaluateImpl, output.tensor_element_type(), op, lhs, rhs,
                       op.attributes.lhs_batching_dimensions,
                       op.attributes.rhs_batching_dimensions,
                       op.attributes.lhs_contracting_dimensions,
                       op.attributes.rhs_contracting_dimensions, output);
  }
  return absl::FailedPreconditionError(
      "stablehlo.dot_general: Unsupported tensor type.");
}

}  // namespace shlo_ref
