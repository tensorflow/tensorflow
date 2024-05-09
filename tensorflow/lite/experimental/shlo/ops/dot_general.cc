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

bool HasInvalidDimension(absl::Span<const Axis> dimensions, const size_t rank) {
  return std::any_of(dimensions.begin(), dimensions.end(),
                     [=](Axis dim) { return dim < 0 || dim >= rank; });
}

bool ContainsDimension(absl::Span<const Axis> dimensions, Axis dimension) {
  return std::find(dimensions.begin(), dimensions.end(), dimension) !=
         dimensions.end();
}

bool HasUniqueDimension(absl::Span<const Axis> batching_dimensions,
                        absl::Span<const Axis> contracting_dimensions,
                        const size_t batch_size, const size_t contract_size) {
  std::unordered_set<Axis> batching_elements;
  std::unordered_set<Axis> contracting_elements;

  batching_elements.insert(batching_dimensions.begin(),
                           batching_dimensions.end());
  if (batching_dimensions.size() != batching_elements.size()) {
    return false;
  }
  contracting_elements.insert(contracting_dimensions.begin(),
                              contracting_dimensions.end());
  if (contracting_dimensions.size() != contracting_elements.size()) {
    return false;
  }
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < contract_size; ++j) {
      if (batching_dimensions[i] == contracting_dimensions[j]) {
        return false;
      }
    }
  }
  return true;
}

bool HasSameDimensionSize(const Tensor& lhs, const Tensor& rhs,
                          absl::Span<const Axis> lhs_dimensions,
                          absl::Span<const Axis> rhs_dimensions,
                          const size_t size) {
  for (size_t i = 0; i < size; ++i) {
    if (lhs.shape().Dim(lhs_dimensions[i]) !=
        rhs.shape().Dim(rhs_dimensions[i])) {
      return false;
    }
  }
  return true;
}

absl::InlinedVector<Axis, kMaxNumDimensions> CalculateResultDimensions(
    Axis rank, absl::Span<const Axis> batching_dimensions,
    absl::Span<const Axis> contracting_dimensions) {
  absl::InlinedVector<Axis, kMaxNumDimensions> result_dims;
  for (Axis i = 0; i < rank; ++i) {
    if (!ContainsDimension(batching_dimensions, i) &&
        !ContainsDimension(contracting_dimensions, i)) {
      result_dims.push_back(i);
    }
  }
  return result_dims;
}

bool CheckZeroPoint(
    const QuantizedElementTypePerTensor::ZeroPointVariant& zero_point) {
  return std::visit(
      [](const auto& v) { return v == static_cast<decltype(v)>(0); },
      zero_point);
}

bool CheckZeroPoints(
    const QuantizedElementTypePerAxis::ZeroPointsVariant& zero_points) {
  return std::visit(
      [](const auto& v) {
        return std::all_of(v.begin(), v.end(), [](const auto value) {
          return value == static_cast<decltype(value)>(0);
        });
      },
      zero_points);
}

bool IncrementIndices(const Tensor& lhs,
                      absl::InlinedVector<Axis, kMaxNumDimensions>& lhs_index,
                      absl::InlinedVector<Axis, kMaxNumDimensions>& rhs_index,
                      absl::Span<const Axis> lhs_contracting_dimensions,
                      absl::Span<const Axis> rhs_contracting_dimensions,
                      const size_t lhsc_size) {
  if (lhsc_size == 0) {
    return false;
  }
  for (int64_t i = static_cast<int64_t>(lhsc_size) - 1; i >= 0; --i) {
    lhs_index[lhs_contracting_dimensions[i]]++;
    rhs_index[rhs_contracting_dimensions[i]]++;
    if (lhs_index[lhs_contracting_dimensions[i]] <
        lhs.shape().Dim(lhs_contracting_dimensions[i])) {
      return true;
    }
    if (i == 0) {
      return false;
    }
    lhs_index[lhs_contracting_dimensions[i]] = 0;
    rhs_index[rhs_contracting_dimensions[i]] = 0;
  }
  return true;
}

absl::Status CheckParameters(
    const Tensor& lhs, const Tensor& rhs,
    absl::Span<const Axis> lhs_batching_dimensions,
    absl::Span<const Axis> rhs_batching_dimensions,
    absl::Span<const Axis> lhs_contracting_dimensions,
    absl::Span<const Axis> rhs_contracting_dimensions, Tensor& output,
    const std::array<PrecisionTypes, 2>& precision_configs) {
  const size_t lhsb_size = lhs_batching_dimensions.size();
  const size_t rhsb_size = rhs_batching_dimensions.size();
  const size_t lhsc_size = lhs_contracting_dimensions.size();
  const size_t rhsc_size = rhs_contracting_dimensions.size();
  const size_t lhs_rank = lhs.Rank();
  const size_t rhs_rank = rhs.Rank();
  const size_t output_rank = output.Rank();
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> expected_output_shape;

  if (precision_configs.size() != 2) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: Size of precision_config must be two.");
  }
  if (lhsb_size != rhsb_size) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: Size of lhs_batching_dimensions and "
        "rhs_batching_dimensions must be same.");
  }
  if (lhsc_size != rhsc_size) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: Size of lhs_contracting_dimensions and "
        "rhs_contracting_dimensions must be same.");
  }
  if (!HasUniqueDimension(lhs_batching_dimensions, lhs_contracting_dimensions,
                          lhsb_size, lhsc_size)) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: The lhs_batching_dimensions and "
        "lhs_contracting_dimensions must be unique.");
  }
  if (!HasUniqueDimension(rhs_batching_dimensions, rhs_contracting_dimensions,
                          rhsb_size, rhsc_size)) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: The rhs_batching_dimensions and "
        "rhs_contracting_dimensions must be unique.");
  }
  if (HasInvalidDimension(lhs_batching_dimensions, lhs_rank)) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: Invalid lhs_batching_dimensions index.");
  }
  if (HasInvalidDimension(lhs_contracting_dimensions, lhs_rank)) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: Invalid lhs_contracting_dimensions index.");
  }
  if (HasInvalidDimension(rhs_batching_dimensions, rhs_rank)) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: Invalid rhs_batching_dimensions index.");
  }
  if (HasInvalidDimension(rhs_contracting_dimensions, rhs_rank)) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: Invalid rhs_contracting_dimensions index.");
  }
  if (!HasSameDimensionSize(lhs, rhs, lhs_batching_dimensions,
                            rhs_batching_dimensions, lhsb_size)) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: The lhs and rhs tensors should have same batch "
        "dimension size.");
  }
  if (!HasSameDimensionSize(lhs, rhs, lhs_contracting_dimensions,
                            rhs_contracting_dimensions, lhsc_size)) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: The lhs and rhs tensors should have same "
        "contracting dimension size.");
  }

  absl::InlinedVector<Axis, kMaxNumDimensions> lhs_result_dim =
      CalculateResultDimensions(lhs_rank, lhs_batching_dimensions,
                                lhs_contracting_dimensions);
  absl::InlinedVector<Axis, kMaxNumDimensions> rhs_result_dim =
      CalculateResultDimensions(rhs_rank, rhs_batching_dimensions,
                                rhs_contracting_dimensions);
  absl::Span<const Axis> lhs_result_dims(lhs_result_dim);
  absl::Span<const Axis> rhs_result_dims(rhs_result_dim);
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> lhs_batch_dims_size =
      lhs.shape().Dims(lhs_batching_dimensions);
  expected_output_shape.insert(expected_output_shape.end(),
                               lhs_batch_dims_size.begin(),
                               lhs_batch_dims_size.end());
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> lhs_result_dims_size =
      lhs.shape().Dims(lhs_result_dims);
  expected_output_shape.insert(expected_output_shape.end(),
                               lhs_result_dims_size.begin(),
                               lhs_result_dims_size.end());
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> rhs_batch_dims_size =
      rhs.shape().Dims(rhs_result_dims);
  expected_output_shape.insert(expected_output_shape.end(),
                               rhs_batch_dims_size.begin(),
                               rhs_batch_dims_size.end());

  const Shape expected_output_check(expected_output_shape);
  if (expected_output_shape.size()) {
    if (expected_output_check != output.shape()) {
      return absl::FailedPreconditionError(
          "stablehlo.dot_general: Invalid output shape.");
    }
  }
  if (lhs.IsPerAxisQuantized()) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: The lhs tensor cannot be per-axis quantized.");
  }
  if ((!lhs.IsQuantized() && !rhs.IsQuantized()) &&
      (lhs.tensor_element_type() != rhs.tensor_element_type())) {
    return absl::FailedPreconditionError(
        "stablehlo.dot_general: For non-quantized tensors the element type of "
        "lhs and rhs must be the same.");
  }
  if (lhs.IsQuantized()) {
    if (!rhs.IsQuantized() || !output.IsQuantized()) {
      return absl::FailedPreconditionError(
          "stablehlo.dot_general: If lhs and rhs are quantized tensors, than "
          "the output tensor should also be quantized.");
    }
    if (lhs.StorageType() != rhs.StorageType()) {
      return absl::FailedPreconditionError(
          "stablehlo.dot_general: If the lhs and rhs are quantized tensors, "
          "than they should have the same storage type.");
    }
    if (rhs.IsPerTensorQuantized()) {
      if (!output.IsPerTensorQuantized()) {
        return absl::FailedPreconditionError(
            "stablehlo.dot_general: If lhs and rhs are per-tensor quantized "
            "than output should also be per-tensor quantized.");
      }
      if ((lhs.quantized_per_tensor_element_type().ExpressedType() ==
           rhs.quantized_per_tensor_element_type().ExpressedType()) &&
          (lhs.quantized_per_tensor_element_type().ExpressedType() !=
           output.quantized_per_tensor_element_type().ExpressedType())) {
        return absl::FailedPreconditionError(
            "stablehlo.dot_general: The expressed_type of output tensor must "
            "be the same as the expressed_type of lhs and rhs tensors.");
      }
      if (!CheckZeroPoint(
              rhs.quantized_per_tensor_element_type().ZeroPoint())) {
        return absl::FailedPreconditionError(
            "stablehlo.dot_general: The rhs per-tensor should have zero points "
            "as 0.");
      }
    } else if (rhs.IsPerAxisQuantized()) {
      if (output.IsPerTensorQuantized()) {
        if ((lhs.quantized_per_tensor_element_type().ExpressedType() ==
             rhs.quantized_per_axis_element_type().ExpressedType()) &&
            (lhs.quantized_per_tensor_element_type().ExpressedType() !=
             output.quantized_per_tensor_element_type().ExpressedType())) {
          return absl::FailedPreconditionError(
              "stablehlo.dot_general: The expressed_type of output must be the "
              "same as the expressed_type of lhs and rhs.");
        }
      } else if (output.IsPerAxisQuantized()) {
        if ((lhs.quantized_per_tensor_element_type().ExpressedType() ==
             rhs.quantized_per_axis_element_type().ExpressedType()) &&
            (lhs.quantized_per_tensor_element_type().ExpressedType() !=
             output.quantized_per_axis_element_type().ExpressedType())) {
          return absl::FailedPreconditionError(
              "stablehlo.dot_general: The expressed_type of output must be the "
              "same as the expressed_type of lhs and rhs.");
        }
      }
      if (!CheckZeroPoints(
              rhs.quantized_per_axis_element_type().ZeroPoints())) {
        return absl::FailedPreconditionError(
            "stablehlo.dot_general: The rhs per-axis should have zero points "
            "as 0.");
      }
      if (ContainsDimension(
              rhs_contracting_dimensions,
              rhs.quantized_per_axis_element_type().QuantizedDimension())) {
        return absl::FailedPreconditionError(
            "stablehlo.dot_general: If the rhs is per-axis quantized than the "
            "quantization_dimensions of rhs should not be in "
            "rhs_contracting_dimensions.");
      }
    }
  }
  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status PrepareTensors(DotGeneralOp& op, const Tensor& lhs,
                            const Tensor& rhs, Tensor& output) {
  using StorageT = StorageType<storage_type>;
  const DimensionSize lhs_size = lhs.NumElements();
  const DimensionSize rhs_size = rhs.NumElements();
  const DimensionSize output_size = output.NumElements();

  // quantized tensor prepare
  if (lhs.IsQuantized()) {
    op.lhs_dequantized_data =
        std::vector<std::byte>(lhs_size * sizeof(StorageT));
    const Shape lhs_dequantized_shape = lhs.shape();
    Tensor lhs_dequantized{.type = TensorType{.shape = lhs_dequantized_shape,
                                              .element_type = storage_type},
                           .data = op.lhs_dequantized_data.data()};
    op.rhs_dequantized_data =
        std::vector<std::byte>(rhs_size * sizeof(StorageT));
    const Shape rhs_dequantized_shape = rhs.shape();
    Tensor rhs_dequantized{.type = TensorType{.shape = rhs_dequantized_shape,
                                              .element_type = storage_type},
                           .data = op.rhs_dequantized_data.data()};
    op.output_dequantized_data =
        std::vector<std::byte>(output_size * sizeof(StorageT));
    const Shape output_dequantized_shape = output.shape();
    Tensor output_dequantized{
        .type = TensorType{.shape = output_dequantized_shape,
                           .element_type = storage_type},
        .data = op.output_dequantized_data.data()};

    op.lhs_dequantized = std::move(lhs_dequantized);
    op.rhs_dequantized = std::move(rhs_dequantized);
    op.output_dequantized = std::move(output_dequantized);
  }
  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status EvaluateImpl(DotGeneralOp& op, const Tensor& lhs,
                          const Tensor& rhs,
                          absl::Span<const Axis> lhs_batching_dimensions,
                          absl::Span<const Axis> rhs_batching_dimensions,
                          absl::Span<const Axis> lhs_contracting_dimensions,
                          absl::Span<const Axis> rhs_contracting_dimensions,
                          Tensor& output) {
  using StorageT = StorageType<storage_type>;
  StorageT* output_data = output.GetDataAs<storage_type>();
  const DimensionSize output_size = output.NumElements();
  const size_t lhs_rank = lhs.Rank();
  const size_t rhs_rank = rhs.Rank();
  const size_t output_rank = output.Rank();
  const size_t lhsb_size = lhs_batching_dimensions.size();
  const size_t lhsc_size = lhs_contracting_dimensions.size();

  absl::InlinedVector<Axis, kMaxNumDimensions> lhs_index(lhs_rank);
  absl::InlinedVector<Axis, kMaxNumDimensions> rhs_index(rhs_rank);
  absl::InlinedVector<Axis, kMaxNumDimensions> output_index(output_rank);

  StorageT output_element(0);
  for (size_t k = 0; k < output_size; ++k, ++output_data) {
    output.GetNdIndex(k, output_index);
    absl::c_fill(lhs_index, 0);
    absl::c_fill(rhs_index, 0);

    DimensionSize result_dim = 0;
    for (size_t i = 0; i < lhsb_size; ++i, ++result_dim) {
      lhs_index[lhs_batching_dimensions[i]] = output_index[result_dim];
      rhs_index[rhs_batching_dimensions[i]] = output_index[result_dim];
    }
    for (size_t i = 0; i < op.lhs_result_dims.size(); ++i, ++result_dim) {
      lhs_index[op.lhs_result_dims[i]] = output_index[result_dim];
    }
    for (size_t i = 0; i < op.rhs_result_dims.size(); ++i, ++result_dim) {
      rhs_index[op.rhs_result_dims[i]] = output_index[result_dim];
    }

    output_element = 0;
    while (true) {
      output_element +=
          lhs.Get<storage_type>(lhs_index) * rhs.Get<storage_type>(rhs_index);
      if (!IncrementIndices(lhs, lhs_index, rhs_index,
                            lhs_contracting_dimensions,
                            rhs_contracting_dimensions, lhsc_size)) {
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

  const StorageT* lhs_data = lhs.GetDataAs<storage_type>();
  ExpressedT* lhs_dequantized_data =
      op.lhs_dequantized.GetDataAs<expressed_type>();
  const StorageT* rhs_data = rhs.GetDataAs<storage_type>();
  ExpressedT* rhs_dequantized_data =
      op.rhs_dequantized.GetDataAs<expressed_type>();
  StorageT* output_data = output.GetDataAs<storage_type>();
  ExpressedT* output_dequantized_data =
      op.output_dequantized.GetDataAs<expressed_type>();

  const DimensionSize lhs_num_elements = lhs.NumElements();
  const StorageT lhs_zero_point =
      lhs.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT lhs_scale =
      lhs.quantized_per_tensor_element_type().ScaleAs<expressed_type>();

  for (DimensionSize i = 0; i < lhs_num_elements;
       ++i, ++lhs_data, ++lhs_dequantized_data) {
    *lhs_dequantized_data = Dequantize(*lhs_data, lhs_zero_point, lhs_scale);
  }

  const DimensionSize rhs_num_elements = rhs.NumElements();
  const StorageT rhs_zero_point =
      rhs.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT rhs_scale =
      rhs.quantized_per_tensor_element_type().ScaleAs<expressed_type>();

  for (DimensionSize i = 0; i < rhs_num_elements;
       ++i, ++rhs_data, ++rhs_dequantized_data) {
    *rhs_dequantized_data = Dequantize(*rhs_data, rhs_zero_point, rhs_scale);
  }

  absl::Status status = Evaluate(op, op.lhs_dequantized, op.rhs_dequantized,
                                 op.output_dequantized);

  const DimensionSize output_num_elements = output.NumElements();
  const StorageT output_zero_point =
      output.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT output_scale =
      output.quantized_per_tensor_element_type().ScaleAs<expressed_type>();
  const ExpressedT inv_scale = static_cast<ExpressedT>(1 / output_scale);

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
    const StorageT* input_data, ExpressedT* inputDeQuantized_data,
    const size_t depth, size_t quantization_index) {
  const DimensionSize dim = shape.Dim(depth);
  if (depth + 1 >= shape.Rank()) {
    for (DimensionSize i = 0; i < dim; ++i) {
      if (depth == quantization_dimension) {
        quantization_index = i;
      }
      *inputDeQuantized_data =
          Dequantize(*input_data, input_zero_points[quantization_index],
                     input_scales[quantization_index]);
      input_data += strides[depth];
      inputDeQuantized_data += strides[depth];
    }
  } else {
    for (DimensionSize i = 0; i < dim; ++i) {
      if (depth == quantization_dimension) {
        quantization_index = i;
      }
      DequantizeOpQuantizePerAxisImpl(
          shape, quantization_dimension, quantization_min, quantization_max,
          input_zero_points, input_scales, strides, input_data,
          inputDeQuantized_data, depth + 1, quantization_index);
      input_data += strides[depth];
      inputDeQuantized_data += strides[depth];
    }
  }
}

template <typename StorageT, typename ExpressedT>
void QuantizeOpQuantizePerAxisImpl(
    const Shape& shape, const Axis quantization_dimension,
    const StorageT quantization_min, const StorageT quantization_max,
    const absl::Span<const StorageT> input_zero_points,
    const absl::Span<const ExpressedT> input_scales, const Strides& strides,
    StorageT* input_data, const ExpressedT* inputDequantized_data,
    const size_t depth, size_t quantization_index) {
  const DimensionSize dim = shape.Dim(depth);
  if (depth + 1 >= shape.Rank()) {
    for (DimensionSize i = 0; i < dim; ++i) {
      if (depth == quantization_dimension) {
        quantization_index = i;
      }
      *input_data = Quantize<StorageT, ExpressedT>(
          *inputDequantized_data, input_zero_points[quantization_index],
          static_cast<ExpressedT>(1 / input_scales[quantization_index]),
          quantization_min, quantization_max);
      input_data += strides[depth];
      inputDequantized_data += strides[depth];
    }
  } else {
    for (DimensionSize i = 0; i < dim; ++i) {
      if (depth == quantization_dimension) {
        quantization_index = i;
      }
      QuantizeOpQuantizePerAxisImpl(
          shape, quantization_dimension, quantization_min, quantization_max,
          input_zero_points, input_scales, strides, input_data,
          inputDequantized_data, depth + 1, quantization_index);
      input_data += strides[depth];
      inputDequantized_data += strides[depth];
    }
  }
}

template <DataType storage_type, DataType expressed_type>
void DequantizeOpQuantizePerAxis(DotGeneralOp& op, const Tensor& lhs,
                                 const Tensor& rhs, Tensor& output) {
  using StorageT = StorageType<storage_type>;
  using ExpressedT = StorageType<expressed_type>;

  const StorageT* lhs_data = lhs.GetDataAs<storage_type>();
  ExpressedT* lhs_dequantized_data =
      op.lhs_dequantized.GetDataAs<expressed_type>();
  const StorageT* rhs_data = rhs.GetDataAs<storage_type>();
  ExpressedT* rhs_dequantized_data =
      op.rhs_dequantized.GetDataAs<expressed_type>();
  StorageT* output_data = output.GetDataAs<storage_type>();
  ExpressedT* output_dequantized_data =
      op.output_dequantized.GetDataAs<expressed_type>();

  const DimensionSize lhs_num_elements = lhs.NumElements();
  const StorageT lhs_zero_point =
      lhs.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT lhs_scale =
      lhs.quantized_per_tensor_element_type().ScaleAs<expressed_type>();

  for (DimensionSize i = 0; i < lhs_num_elements;
       ++i, ++lhs_data, ++lhs_dequantized_data) {
    *lhs_dequantized_data = Dequantize(*lhs_data, lhs_zero_point, lhs_scale);
  }

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

  absl::Status status = Evaluate(op, op.lhs_dequantized, op.rhs_dequantized,
                                 op.output_dequantized);
  if (output.IsPerAxisQuantized()) {
    const Shape& shape = output.shape();
    const Axis output_quantization_dimension =
        output.quantized_per_axis_element_type().QuantizedDimension();
    const absl::Span<const StorageT> output_zero_points =
        output.quantized_per_axis_element_type().ZeroPointsAs<storage_type>();
    const absl::Span<const ExpressedT> output_scales =
        output.quantized_per_axis_element_type().ScalesAs<expressed_type>();
    const Strides& strides = ComputeStrides(shape);
    QuantizeOpQuantizePerAxisImpl(
        shape, output_quantization_dimension, Storage<storage_type>::kMinValue,
        Storage<storage_type>::kMaxValue, output_zero_points, output_scales,
        strides, output_data, output_dequantized_data, /*depth=*/0,
        /*quantization_index=*/0);
  } else {
    const DimensionSize output_num_elements = output.NumElements();
    const StorageT output_zero_point =
        output.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
    const ExpressedT output_scale =
        output.quantized_per_tensor_element_type().ScaleAs<expressed_type>();
    const ExpressedT inv_scale = static_cast<ExpressedT>(1 / output_scale);

    for (DimensionSize i = 0; i < output_num_elements;
         ++i, ++output_dequantized_data, ++output_data) {
      *output_data = Quantize<storage_type, expressed_type>(
          *output_dequantized_data, output_zero_point, inv_scale);
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

  const size_t lhs_rank = lhs.Rank();
  const size_t rhs_rank = rhs.Rank();
  op.lhs_result_dims =
      CalculateResultDimensions(lhs_rank, op.attributes.lhs_batching_dimensions,
                                op.attributes.lhs_contracting_dimensions);
  op.rhs_result_dims =
      CalculateResultDimensions(rhs_rank, op.attributes.rhs_batching_dimensions,
                                op.attributes.rhs_contracting_dimensions);

  if (lhs.IsQuantized()) {
    DISPATCH_BOOL_INT_FLOAT(
        PrepareTensors, lhs.quantized_per_tensor_element_type().ExpressedType(),
        op, lhs, rhs, output);
  } else {
    DISPATCH_BOOL_INT_FLOAT(PrepareTensors, lhs.StorageType(), op, lhs, rhs,
                            output);
  }
  return absl::OkStatus();
}

absl::Status Evaluate(DotGeneralOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output) {
  if (lhs.IsQuantized()) {
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
