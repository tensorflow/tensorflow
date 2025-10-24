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

#include "tensorflow/lite/experimental/shlo/ops/convolution.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/dot_general.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

template <class T>
using DimVector = absl::InlinedVector<T, 6>;

bool IsUnique(DimVector<int64_t>& vec) {
  std::sort(vec.begin(), vec.end());
  return std::unique(vec.begin(), vec.end()) == vec.end();
}

bool IsInRange(DimVector<int64_t>& vec, size_t N) {
  for (int64_t dim : vec) {
    if (dim >= N || dim < 0) {
      return false;
    }
  }
  return true;
}

template <DataType storage_type>
absl::Status PrepareImpl(ConvolutionOp& op, const Tensor& lhs,
                         const Tensor& rhs, Tensor& output) {
  using StorageT = StorageType<storage_type>;

  // Transpose prepare
  const int64_t* window_spacial_pointer =
      op.attributes.kernel_spacial_dimensions.GetDataAs<DataType::kSI64>();
  const int64_t* output_spacial_pointer =
      op.attributes.output_spacial_dimensions.GetDataAs<DataType::kSI64>();
  const int64_t* input_spacial_pointer =
      op.attributes.input_spacial_dimensions.GetDataAs<DataType::kSI64>();

  std::vector<StorageT> lhs_permutation_values(
      static_cast<int64_t>(lhs.Rank()));
  lhs_permutation_values[0] = op.attributes.input_batch_dimension;
  lhs_permutation_values[1] = op.attributes.input_feature_dimension;
  DimVector<DimensionSize> lhs_shape_dims(lhs.Rank());
  lhs_shape_dims[0] =
      lhs.shape().Dim(static_cast<size_t>(op.attributes.input_batch_dimension));
  lhs_shape_dims[1] = lhs.shape().Dim(
      static_cast<size_t>(op.attributes.input_feature_dimension));
  for (size_t i = 0; i < lhs.Rank() - 2; ++i) {
    lhs_shape_dims[i + 2] =
        lhs.shape().Dim(static_cast<size_t>(input_spacial_pointer[i]));
    lhs_permutation_values[i + 2] = input_spacial_pointer[i];
  }
  // malloc is used to have the storage space available out of prepare function
  // scope and it's pointer is stored in class data member to
  // deallocate the memory in destructor.
  op.lhs_permutation_data =
      malloc(lhs_permutation_values.size() * sizeof(StorageT));
  memmove(op.lhs_permutation_data, lhs_permutation_values.data(),
          lhs_permutation_values.size() * sizeof(StorageT));
  const Shape lhs_permutation_shape({static_cast<int64_t>(lhs.Rank())});
  Tensor lhs_permutations{.type = TensorType{.shape = lhs_permutation_shape,
                                             .element_type = storage_type},
                          .data = op.lhs_permutation_data};

  op.lhs_transposed_data = malloc(lhs.NumElements() * sizeof(StorageT));
  const Shape lhs_transposed_shape(lhs_shape_dims);
  Tensor lhs_transposed{.type = TensorType{.shape = lhs_transposed_shape,
                                           .element_type = storage_type},
                        .data = op.lhs_transposed_data};

  std::vector<StorageT> rhs_permutation_values(
      static_cast<int64_t>(rhs.Rank()));
  rhs_permutation_values[0] = op.attributes.kernel_output_feature_dimension;
  rhs_permutation_values[1] = op.attributes.kernel_input_feature_dimension;
  DimVector<DimensionSize> rhs_shape_dims(rhs.Rank());
  rhs_shape_dims[0] = rhs.shape().Dim(
      static_cast<size_t>(op.attributes.kernel_output_feature_dimension));
  rhs_shape_dims[1] = rhs.shape().Dim(
      static_cast<size_t>(op.attributes.kernel_input_feature_dimension));
  for (size_t i = 0; i < rhs.Rank() - 2; ++i) {
    rhs_shape_dims[i + 2] =
        rhs.shape().Dim(static_cast<size_t>(window_spacial_pointer[i]));
    rhs_permutation_values[i + 2] = window_spacial_pointer[i];
  }
  op.rhs_permutation_data = malloc(rhs.Rank() * sizeof(StorageT));
  memmove(op.rhs_permutation_data, rhs_permutation_values.data(),
          rhs_permutation_values.size() * sizeof(StorageT));
  const Shape rhs_permutation_shape({static_cast<int64_t>(rhs.Rank())});
  Tensor rhs_permutations{.type = TensorType{.shape = rhs_permutation_shape,
                                             .element_type = storage_type},
                          .data = op.rhs_permutation_data};

  op.rhs_transposed_data = malloc(rhs.NumElements() * sizeof(StorageT));
  const Shape rhs_transposed_shape(rhs_shape_dims);
  Tensor rhs_transposed{.type = TensorType{.shape = rhs_transposed_shape,
                                           .element_type = storage_type},
                        .data = op.rhs_transposed_data};

  std::vector<StorageT> output_permutation_values(
      static_cast<int64_t>(output.Rank()));
  output_permutation_values[0] = op.attributes.output_batch_dimension;
  output_permutation_values[1] = op.attributes.output_feature_dimension;
  DimVector<DimensionSize> output_shape_dims(output.Rank());
  output_shape_dims[0] = output.shape().Dim(
      static_cast<size_t>(op.attributes.output_batch_dimension));
  output_shape_dims[1] = output.shape().Dim(
      static_cast<size_t>(op.attributes.output_feature_dimension));
  for (size_t i = 0; i < output.Rank() - 2; ++i) {
    output_shape_dims[i + 2] =
        output.shape().Dim(static_cast<size_t>(output_spacial_pointer[i]));
    output_permutation_values[i + 2] = output_spacial_pointer[i];
  }
  op.output_permutation_data = malloc(output.Rank() * sizeof(StorageT));
  memmove(op.output_permutation_data, output_permutation_values.data(),
          output_permutation_values.size() * sizeof(StorageT));
  const Shape output_permutation_shape({static_cast<int64_t>(output.Rank())});
  Tensor output_permutations{
      .type = TensorType{.shape = output_permutation_shape,
                         .element_type = storage_type},
      .data = op.output_permutation_data};

  op.output_transposed_data = malloc(output.NumElements() * sizeof(StorageT));
  const Shape output_transposed_shape(output_shape_dims);
  Tensor output_transposed{.type = TensorType{.shape = output_transposed_shape,
                                              .element_type = storage_type},
                           .data = op.output_transposed_data};
  // transpose prepare end

  // DotGeneral prepare
  DimVector<DimensionSize> dims(rhs_transposed.Rank());
  size_t rhs_transposed_tensor_size = 1;
  dims[0] = 1;
  for (size_t i = 1; i < rhs_transposed.Rank(); ++i) {
    dims[i] = rhs_transposed.shape().Dim(i);
    rhs_transposed_tensor_size *= rhs_transposed.shape().Dim(i);
  }
  const Shape rhs_dot_general_shape(dims);
  op.rhs_dot_general_data =
      malloc(rhs_transposed_tensor_size * sizeof(StorageT));
  Tensor rhs_dot_general{.type = TensorType{.shape = rhs_dot_general_shape,
                                            .element_type = storage_type},
                         .data = op.rhs_dot_general_data};

  op.lhs_dot_general_data =
      malloc(rhs_transposed_tensor_size * sizeof(StorageT));
  Tensor lhs_dot_general{.type = TensorType{.shape = rhs_dot_general_shape,
                                            .element_type = storage_type},
                         .data = op.lhs_dot_general_data};

  std::vector<typename Storage<DataType::kSI64>::Type>
      lhs_contracting_dimensions_values(lhs_transposed.Rank() - 1);
  for (size_t i = 0; i < lhs_transposed.Rank() - 1; ++i) {
    lhs_contracting_dimensions_values[i] = i + 1;
  }
  op.lhs_contracting_dimensions_data =
      malloc((lhs_transposed.Rank() - 1) * sizeof(int64_t));
  memmove(op.lhs_contracting_dimensions_data,
          lhs_contracting_dimensions_values.data(),
          lhs_contracting_dimensions_values.size() * sizeof(int64_t));
  const Shape lhs_contracting_dimensions_shape(
      {static_cast<int64_t>(lhs_transposed.Rank() - 1)});
  Tensor lhs_contracting_dimensions{
      .type = TensorType{.shape = lhs_contracting_dimensions_shape,
                         .element_type = DataType::kSI64},
      .data = op.lhs_contracting_dimensions_data};

  std::vector<typename Storage<DataType::kSI64>::Type>
      rhs_contracting_dimensions_values(rhs_transposed.Rank() - 1);
  for (size_t i = 0; i < rhs_transposed.Rank() - 1; ++i) {
    rhs_contracting_dimensions_values[i] = i + 1;
  }
  op.rhs_contracting_dimensions_data =
      malloc((rhs_transposed.Rank() - 1) * sizeof(int64_t));
  memmove(op.rhs_contracting_dimensions_data,
          rhs_contracting_dimensions_values.data(),
          rhs_contracting_dimensions_values.size() * sizeof(int64_t));
  Tensor rhs_contracting_dimensions{
      .type = TensorType{.shape = lhs_contracting_dimensions_shape,
                         .element_type = DataType::kSI8},
      .data = op.rhs_contracting_dimensions_data};

  std::vector<StorageT> dor_general_output_values(1);
  dor_general_output_values[0] = 0;
  op.output_dot_general_data = malloc(1 * sizeof(StorageT));
  memmove(op.output_dot_general_data, dor_general_output_values.data(),
          dor_general_output_values.size() * sizeof(StorageT));
  const Shape dor_general_output_shape{{1}};
  Tensor output_dot_general{
      .type = TensorType{.shape = dor_general_output_shape,
                         .element_type = storage_type},
      .data = op.output_dot_general_data};

  Tensor lhs_batching_dimensions{
      .type = TensorType{.shape = Shape(), .element_type = DataType::kSI8},
      .data = {}};
  Tensor rhs_batching_dimensions{
      .type = TensorType{.shape = Shape(), .element_type = DataType::kSI8},
      .data = {}};

  absl::InlinedVector<PrecisionTypes, 2> precision_configs = {
      PrecisionTypes::DEFAULT, PrecisionTypes::DEFAULT};
  op.dot_general_op = Create(DotGeneralOp::Attributes{
      .lhs_batching_dimensions = lhs_batching_dimensions,
      .rhs_batching_dimensions = rhs_batching_dimensions,
      .lhs_contracting_dimensions = lhs_contracting_dimensions,
      .rhs_contracting_dimensions = rhs_contracting_dimensions,
      .precision_configs = precision_configs});

  auto state = Prepare(op.dot_general_op, lhs_dot_general, rhs_dot_general,
                       output_dot_general);
  // Dot general prepare end

  // padding prepare
  const int64_t* lhs_dilation_buffer =
      op.attributes.lhs_dilation.GetDataAs<DataType::kSI64>();
  const int64_t* padding_buffer =
      op.attributes.padding.GetDataAs<DataType::kSI64>();

  int lhs_padded_spacials[lhs_transposed.Rank() - 2];
  int lhs_padded_tensor_size = 1;
  for (size_t i = lhs_transposed.Rank() - 1; i > 1; --i) {
    lhs_padded_spacials[i - 2] =
        lhs_transposed.shape().Dim(i) +
        (lhs_dilation_buffer[i - 2] - 1) * (lhs_transposed.shape().Dim(i) - 1) +
        padding_buffer[2 * (i - 2)] + padding_buffer[(2 * (i - 2)) + 1];
    lhs_padded_tensor_size *= lhs_padded_spacials[i - 2];
  }

  lhs_padded_tensor_size *=
      lhs_transposed.shape().Dim(0) * lhs_transposed.shape().Dim(1);
  op.lhs_padded_data = malloc(lhs_padded_tensor_size * sizeof(StorageT));
  DimVector<DimensionSize> lhs_padding_shape_dims(lhs_transposed.Rank());
  lhs_padding_shape_dims[0] = lhs_transposed.shape().Dim(0);
  lhs_padding_shape_dims[1] = lhs_transposed.shape().Dim(1);
  for (size_t i = 0; i < lhs_transposed.Rank() - 2; ++i) {
    lhs_padding_shape_dims[i + 2] =
        static_cast<int64_t>(lhs_padded_spacials[i]);
  }
  const Shape lhs_padding_shape(lhs_padding_shape_dims);
  Tensor lhs_padded{.type = TensorType{.shape = lhs_padding_shape,
                                       .element_type = storage_type},
                    .data = op.lhs_padded_data};
  // padding prepare end

  // Split prepare
  int64_t num_splits =
      op.attributes.batch_group_count * op.attributes.feature_group_count;
  int64_t split_dimension = 0;
  for (int64_t i = 0; i < num_splits; ++i) {
    DimVector<DimensionSize> rhs_split_dims(rhs_transposed.Rank());
    for (size_t i = 0; i < rhs_transposed.Rank(); ++i) {
      if (i == split_dimension) {
        rhs_split_dims[i] = (rhs_transposed.shape().Dim(i) / num_splits);
      } else {
        rhs_split_dims[i] = rhs_transposed.shape().Dim(i);
      }
    }
    const Shape rhs_split_shape(rhs_split_dims);
    void* rhs_split_data =
        malloc((rhs_transposed.NumElements() / num_splits) * sizeof(StorageT));
    op.rhs_splits_data.push_back(rhs_split_data);
    Tensor rhs_split{.type = TensorType{.shape = rhs_split_shape,
                                        .element_type = storage_type},
                     .data = rhs_split_data};
    op.rhs_splits.push_back(rhs_split);
  }

  if (op.attributes.feature_group_count > 1) {
    split_dimension = 1;
  }

  for (int64_t i = 0; i < num_splits; ++i) {
    DimVector<DimensionSize> lhs_split_dims(lhs_padded.Rank());
    for (size_t i = 0; i < lhs_padded.Rank(); ++i) {
      if (i == split_dimension) {
        lhs_split_dims[i] = (lhs_padded.shape().Dim(i) / num_splits);
      } else {
        lhs_split_dims[i] = lhs_padded.shape().Dim(i);
      }
    }
    const Shape lhs_split_shape(lhs_split_dims);
    void* lhs_split_data =
        malloc((lhs_padded.NumElements() / num_splits) * sizeof(StorageT));
    op.lhs_splits_data.push_back(lhs_split_data);
    Tensor lhs_split{.type = TensorType{.shape = lhs_split_shape,
                                        .element_type = storage_type},
                     .data = lhs_split_data};
    op.lhs_splits.push_back(lhs_split);
  }
  // split prepare end

  // quantized tensors prepare
  if (lhs.IsQuantized()) {
    op.lhs_dequantized_data = malloc(lhs.NumElements() * sizeof(StorageT));
    const Shape lhs_dequantized_shape = lhs.shape();
    Tensor lhs_dequantized{.type = TensorType{.shape = lhs_dequantized_shape,
                                              .element_type = storage_type},
                           .data = op.lhs_dequantized_data};
    op.rhs_dequantized_data = malloc(rhs.NumElements() * sizeof(StorageT));
    const Shape rhs_dequantized_shape = rhs.shape();
    Tensor rhs_dequantized{.type = TensorType{.shape = rhs_dequantized_shape,
                                              .element_type = storage_type},
                           .data = op.rhs_dequantized_data};
    op.output_dequantized_data =
        malloc(output.NumElements() * sizeof(StorageT));
    const Shape output_dequantized_shape = output.shape();
    Tensor output_dequantized{
        .type = TensorType{.shape = output_dequantized_shape,
                           .element_type = storage_type},
        .data = op.output_dequantized_data};

    op.lhs_dequantized = std::move(lhs_dequantized);
    op.rhs_dequantized = std::move(rhs_dequantized);
    op.output_dequantized = std::move(output_dequantized);
  }
  // quantized tensors prepare end

  op.lhs_permutations = std::move(lhs_permutations);
  op.lhs_transposed = std::move(lhs_transposed);
  op.rhs_permutations = std::move(rhs_permutations);
  op.rhs_transposed = std::move(rhs_transposed);
  op.output_permutations = std::move(output_permutations);
  op.output_transposed = std::move(output_transposed);
  op.lhs_dot_general = std::move(lhs_dot_general);
  op.rhs_dot_general = std::move(rhs_dot_general);
  op.output_dot_general = std::move(output_dot_general);
  op.lhs_padded = std::move(lhs_padded);

  const int64_t* rhs_dilation_buffer =
      op.attributes.rhs_dilation.GetDataAs<DataType::kSI64>();
  const int64_t* window_strides_pointer =
      op.attributes.window_strides.GetDataAs<DataType::kSI64>();

  // Constraints Check
  if (op.attributes.precision_configs.size() != 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution: Size of precision_config must be two.");
  }
  if (op.attributes.precision_configs[0] != PrecisionTypes::DEFAULT &&
      op.attributes.precision_configs[1] != PrecisionTypes::DEFAULT) {
    return absl::UnimplementedError(
        "stablehlo.convolution: Currently the precision_config supports "
        "DEFAULT configuration only.");
  }
  size_t rank = lhs.Rank();
  if (lhs.Rank() != rhs.Rank()) {
    return absl::FailedPreconditionError(
        "Constraint violation: rank(lhs) == rank(rhs)");
  } else if (output.Rank() != lhs.Rank()) {
    return absl::FailedPreconditionError(
        "Constraint violation: rank(output) == lhs.Rank()");
  }
  if (!lhs.IsQuantized()) {
    SHLO_REF_RETURN_ON_ERROR(
        CheckSameBaselineType(CheckCtx("Convolution"), lhs, rhs));
    SHLO_REF_RETURN_ON_ERROR(
        CheckSameBaselineType(CheckCtx("Convolution"), lhs, output));
  }
  if (op.attributes.window_strides.shape().Dim(0) != rank - 2) {
    return absl::FailedPreconditionError(
        "Constraint violation: size(windowStride)=rank-2");
  }

  const int64_t* check_buffer =
      op.attributes.window_strides.GetDataAs<DataType::kSI64>();
  bool is_greater_than_zero = true;
  size_t n = op.attributes.window_strides.NumElements();
  for (size_t i = 0; i < n; ++i) {
    if (check_buffer[i] == 0) {
      is_greater_than_zero = false;
      exit;
    }
  }

  if (!is_greater_than_zero) {
    return absl::FailedPreconditionError(
        "Constraint violation: 0<windowStride");
  } else if (op.attributes.padding.shape().Dim(0) != rank - 2 ||
             op.attributes.padding.shape().Dim(1) != 2) {
    return absl::FailedPreconditionError(
        "Constraint violation: shape(padding)=[rank-2,2]");
  } else if (op.attributes.lhs_dilation.shape().Dim(0) != rank - 2) {
    return absl::FailedPreconditionError(
        "Contraint violation: shape(lhs_dilation) == rank-2");
  }

  check_buffer = op.attributes.lhs_dilation.GetDataAs<DataType::kSI64>();
  n = op.attributes.lhs_dilation.NumElements();
  for (size_t i = 0; i < n; ++i) {
    if (check_buffer[i] == 0) {
      is_greater_than_zero = false;
      exit;
    }
  }

  if (!is_greater_than_zero) {
    return absl::FailedPreconditionError(
        "Constraint violation: 0<lhs_dilation");
  } else if (op.attributes.rhs_dilation.shape().Dim(0) != rank - 2) {
    return absl::FailedPreconditionError(
        "Constraint violation: shape(rhs_dilation) == rank-2");
  }

  check_buffer = op.attributes.rhs_dilation.GetDataAs<DataType::kSI64>();
  n = op.attributes.rhs_dilation.NumElements();
  for (size_t i = 0; i < n; ++i) {
    if (check_buffer[i] == 0) {
      is_greater_than_zero = false;
      exit;
    }
  }

  if (!is_greater_than_zero) {
    return absl::FailedPreconditionError(
        "Constraint violation: 0<rhs_dilation");
  } else if (op.attributes.window_reversal.shape().Dim(0) != rank - 2) {
    return absl::FailedPreconditionError(
        "Constraint violation: shape(window_reversal) == rank-2");
  } else if (lhs.shape().Dim(
                 static_cast<size_t>(op.attributes.input_batch_dimension)) %
                 op.attributes.batch_group_count !=
             0) {
    return absl::FailedPreconditionError(
        "Contraint violation: Dim(lhs,input_batch_dimension)%batch_group_count "
        "= "
        "0");
  } else if (lhs.shape().Dim(
                 static_cast<size_t>(op.attributes.input_feature_dimension)) %
                 op.attributes.feature_group_count !=
             0) {
    return absl::FailedPreconditionError(
        "Contraint violation: "
        "Dim(lhs,input_feature_dimension)%(feature_group_count) = 0");
  } else if (op.attributes.input_spacial_dimensions.shape().Dim(0) !=
             rank - 2) {
    return absl::FailedPreconditionError(
        "Constarint violation: size(input_spacial_dimensions) = rank-2");
  }

  DimVector<int64_t> vec;
  vec.push_back(op.attributes.input_batch_dimension);
  vec.push_back(op.attributes.input_feature_dimension);
  check_buffer =
      op.attributes.input_spacial_dimensions.GetDataAs<DataType::kSI64>();
  n = op.attributes.input_spacial_dimensions.NumElements();
  for (size_t i = 0; i < n; ++i) {
    vec.push_back(check_buffer[i]);
  }

  if (!(IsUnique(vec))) {
    return absl::FailedPreconditionError(
        "Constraint violation: isUnique(inputDimensions)");
  } else if (!(IsInRange(vec, rank))) {
    return absl::FailedPreconditionError(
        "Constraint violation: 0<= inputDimensions < rank");
  } else if (rhs.shape().Dim(static_cast<size_t>(
                 op.attributes.kernel_input_feature_dimension)) !=
             lhs.shape().Dim(
                 static_cast<size_t>(op.attributes.input_feature_dimension)) /
                 op.attributes.feature_group_count) {
    return absl::FailedPreconditionError(
        "Constraint violation: Dim(rhs,kernel_input_feature_dimension) = "
        "Dim(lhs,input_feature_dimension)/feature_group_count");
  } else if (rhs.shape().Dim(static_cast<size_t>(
                 op.attributes.kernel_output_feature_dimension)) %
                 op.attributes.batch_group_count !=
             0) {
    return absl::FailedPreconditionError(
        "Constarint violation: "
        "Dim(rhs,kernel_output_feature_dimension)%batch_group_count=0");
  } else if (rhs.shape().Dim(static_cast<size_t>(
                 op.attributes.kernel_output_feature_dimension)) %
                 op.attributes.feature_group_count !=
             0) {
    return absl::FailedPreconditionError(
        "Constraint violation: "
        "Dim(rhs,kernel_output_feature_dimension)%(feature_group_count)=0");
  } else if (op.attributes.kernel_spacial_dimensions.shape().Dim(0) !=
             rank - 2) {
    return absl::FailedPreconditionError(
        "Constraint violation: size(kernel_spacial_dimensions) = rank-2");
  }

  vec.clear();
  vec.push_back(op.attributes.kernel_input_feature_dimension);
  vec.push_back(op.attributes.kernel_output_feature_dimension);
  check_buffer =
      op.attributes.kernel_spacial_dimensions.GetDataAs<DataType::kSI64>();
  n = op.attributes.kernel_spacial_dimensions.NumElements();
  for (size_t i = 0; i < n; ++i) {
    vec.push_back(check_buffer[i]);
  }

  if (!(IsUnique(vec))) {
    return absl::FailedPreconditionError(
        "Constraint violation: isUnique(kernelDimensions)");
  } else if (!(IsInRange(vec, rank))) {
    return absl::FailedPreconditionError(
        "Constraint violation: 0<= kernelDimensions < rank");
  } else if (op.attributes.output_spacial_dimensions.shape().Dim(0) !=
             rank - 2) {
    return absl::FailedPreconditionError(
        "Constraint violation: size(output_spacial_dimensions) = rank-2");
  }

  vec.clear();
  vec.push_back(op.attributes.output_batch_dimension);
  vec.push_back(op.attributes.output_feature_dimension);
  check_buffer =
      op.attributes.output_spacial_dimensions.GetDataAs<DataType::kSI64>();
  n = op.attributes.output_spacial_dimensions.NumElements();
  for (size_t i = 0; i < n; ++i) {
    vec.push_back(check_buffer[i]);
  }

  if (!(IsUnique(vec))) {
    return absl::FailedPreconditionError(
        "Constraint violation: isUnique(outputDimensions)");
  } else if (!(IsInRange(vec, rank))) {
    return absl::FailedPreconditionError(
        "Constraint violation: 0<= outputDimensions < rank");
  } else if (op.attributes.feature_group_count <= 0) {
    return absl::FailedPreconditionError(
        "Constraint violation: 0<feature_group_count");
  } else if (op.attributes.batch_group_count <= 0) {
    return absl::FailedPreconditionError(
        "Constraint violation: 0<batch_group_count");
  } else if (op.attributes.batch_group_count != 1 &&
             op.attributes.feature_group_count != 1) {
    return absl::FailedPreconditionError(
        "Constraint violation: batch_group_count == 1 or feature_group_count "
        "== 1");
  } else if (output.shape().Dim(
                 static_cast<size_t>(op.attributes.output_batch_dimension)) !=
             lhs.shape().Dim(
                 static_cast<size_t>(op.attributes.input_batch_dimension)) /
                 op.attributes.batch_group_count) {
    return absl::FailedPreconditionError(
        "Constraint violation: output.shape().Dim(output_batch_dimension) == "
        "lhs.shape().Dim(input_batch_dimension)/batch_group_count");
  } else if (output.shape().Dim(
                 static_cast<size_t>(op.attributes.output_feature_dimension)) !=
             rhs.shape().Dim(static_cast<size_t>(
                 op.attributes.kernel_output_feature_dimension))) {
    return absl::FailedPreconditionError(
        "Constraint violation: output.shape().Dim(outputFeatureDimention) == "
        "rhs.shape().Dim(kernel_output_feature_dimension)");
  }

  for (int64_t i = 0; i < output.Rank() - 2; ++i) {
    if (output.shape().Dim(static_cast<size_t>(output_spacial_pointer[i])) !=
        std::floor(
            (((lhs_dilation_buffer[i] *
               (lhs.shape().Dim(static_cast<size_t>(input_spacial_pointer[i])) -
                1)) +
              1 + padding_buffer[2 * i] + padding_buffer[2 * i + 1] -
              ((rhs_dilation_buffer[i] * (rhs.shape().Dim(static_cast<size_t>(
                                              window_spacial_pointer[i])) -
                                          1)) +
               1)) /
             window_strides_pointer[i]) +
            1)) {
      return absl::FailedPreconditionError(
          "Constraint violation: output.shape().Dim(spacial_dim) is not "
          "properly set");
    }
  }

  if (lhs.IsQuantized() || rhs.IsQuantized() || output.IsQuantized()) {
    if (!(lhs.IsQuantized() && rhs.IsQuantized() && output.IsQuantized())) {
      return absl::FailedPreconditionError(
          "Constraint violation: lhs.IsQuantized() && rhs.IsQuantized() && "
          "output.IsQuantized()");
    }
    if (rhs.IsPerTensorQuantized()) {
      if (!(output.IsPerTensorQuantized())) {
        return absl::FailedPreconditionError(
            "Constraint violation: If is_per_tensor_quantized(rhs), then "
            "is_per_tensor_quantized(output)");
      }
    }
    if (rhs.IsPerAxisQuantized()) {
      if (rhs.quantized_per_axis_element_type().QuantizedDimension() !=
          op.attributes.kernel_output_feature_dimension) {
        return absl::FailedPreconditionError(
            "Constraint violation:  If is_per_axis_quantized(rhs), then "
            "quantization_dimension(rhs) = "
            "op.attributes.kernel_output_feature_dimension");
      }
    }
    if (output.IsPerAxisQuantized()) {
      if (output.quantized_per_axis_element_type().QuantizedDimension() !=
          op.attributes.output_feature_dimension) {
        return absl::FailedPreconditionError(
            "Constraint violation:  If is_per_axis_quantized(output), then "
            "quantization_dimension(output) = "
            "op.attributes.output_feature_dimension");
      }
    }
  }

  return absl::OkStatus();
}

// Slice op basic implimentation in context of Convolution
template <DataType storage_type>
void EvalDynamicSliceOp(const Tensor& operand, int64_t num_outputs,
                        size_t start_indices, size_t inner_dimensions_size,
                        size_t outer_dimensions_size, int64_t dimension,
                        Tensor& output) {
  using StorageT = StorageType<storage_type>;

  StorageT* output_buffer = output.GetDataAs<storage_type>();
  const StorageT* operand_buffer = operand.GetDataAs<storage_type>();

  size_t i = start_indices;
  size_t k = 0;
  while (i < operand.NumElements()) {
    for (size_t j = 0;
         j < (output.shape().Dim(dimension) * inner_dimensions_size);
         ++j, ++k) {
      output_buffer[k] = operand_buffer[i + j];
    }
    i += outer_dimensions_size;
  }
}

template <DataType storage_type>
void split(const Tensor& operand, int64_t num_outputs, int64_t dimension,
           std::vector<Tensor>& outputs) {
  size_t start_indices = 0;
  size_t inner_dimensions_size = 1;
  size_t outer_dimensions_size = 1;
  size_t dimension_size = operand.shape().Dim(dimension) / num_outputs;

  for (size_t i = operand.Rank() - 1; i > dimension; --i) {
    inner_dimensions_size *= operand.shape().Dim(i);
  }
  outer_dimensions_size *=
      inner_dimensions_size * operand.shape().Dim(dimension);

  for (int64_t i = 0; i < num_outputs; ++i) {
    start_indices += (i)*dimension_size * inner_dimensions_size;
    EvalDynamicSliceOp<storage_type>(
        operand, num_outputs, start_indices, inner_dimensions_size,
        outer_dimensions_size, dimension, outputs[i]);
  }
}

// Transpose op basic implimentation in context of Convolution
DimVector<int> GenerateIndices(int i, const DimVector<int>& temp) {
  int rank = temp.size();
  DimVector<int> indices(rank, 0);
  int divisor = 1;
  for (int64_t j = rank - 1; j >= 0; --j) {
    indices[j] = (i / divisor) % temp[j];
    divisor *= temp[j];
  }
  return indices;
}

template <DataType data_type>
absl::Status TransposeEvaluateImpl(const Tensor& operand,
                                   const Tensor& permutation, Tensor& output) {
  using StorageT = StorageType<data_type>;
  if (permutation.NumElements() != operand.Rank()) {
    return absl::FailedPreconditionError(
        "Rank of output and permutation doesn't match");
  }
  const StorageT* operand_buffer = operand.GetDataAs<data_type>();
  const StorageT* permutation_buffer = permutation.GetDataAs<data_type>();
  StorageT* output_buffer = output.GetDataAs<data_type>();

  int64_t operand_product = 1, output_product = 1;
  for (int64_t i = 0; i < operand.Rank(); ++i) {
    operand_product *= operand.shape().Dim(i);
    output_product *= output.shape().Dim(i);
  }

  DimVector<int> temp;
  for (int64_t i = 0; i < operand.Rank(); ++i) {
    temp.push_back(operand.shape().Dim(i));
  }

  for (size_t k = 0; k < operand.NumElements(); ++k) {
    DimVector<int> operand_index = GenerateIndices(k, temp);
    DimVector<int> output_index(output.Rank(), 0);
    for (size_t d = 0; d < output.Rank(); ++d) {
      output_index[d] = operand_index[permutation_buffer[d]];
    }

    int operand_element_index = 0, output_element_index = 0;
    int64_t temp1 = 1, temp2 = 1;
    for (int64_t i = 0; i < operand.Rank(); i++) {
      temp1 *= operand.shape().Dim(i);
      operand_element_index += operand_index[i] * (operand_product / temp1);
      temp2 *= output.shape().Dim(i);
      output_element_index += output_index[i] * (output_product / temp2);
    }
    output_buffer[output_element_index] = operand_buffer[operand_element_index];
  }
  return absl::OkStatus();
}

// Padding op basic implimentation in context of Convolution
template <DataType storage_type>
void PaddingOp(ConvolutionOp& op, const Tensor& x, const Tensor& padding,
               const Tensor& lhs_dilations) {
  using StorageT = StorageType<storage_type>;
  using int64_t = StorageType<DataType::kSI64>;
  const StorageT* x_buffer = x.GetDataAs<storage_type>();
  const int64_t* lhs_dilation_buffer =
      lhs_dilations.GetDataAs<DataType::kSI64>();
  const int64_t* padding_buffer = padding.GetDataAs<DataType::kSI64>();
  StorageT* lhs_buffer = op.lhs_padded.GetDataAs<storage_type>();
  size_t j = 0;
  for (size_t i = 0; i < op.lhs_padded.NumElements(); ++i) {
    int x_spacials[x.Rank() - 2];
    size_t depth = 1;

    for (int64_t m = x.Rank() - 3; m >= 0; --m) {
      x_spacials[m] =
          (i / depth) % static_cast<size_t>(op.lhs_padded.shape().Dim(m + 2));
      depth *= static_cast<size_t>(op.lhs_padded.shape().Dim(m + 2));
    }
    bool check = true;
    for (int64_t k = x.Rank() - 3; k >= 0; --k) {
      check *= x_spacials[k] >= (padding_buffer[2 * k]) &&
               x_spacials[k] < x.shape().Dim(k + 2) +
                                   (lhs_dilation_buffer[k] - 1) *
                                       (x.shape().Dim(k + 2) - 1) +
                                   padding_buffer[2 * k];
    }

    if (check) {
      for (int64_t k = x.Rank() - 3; k >= 0; --k) {
        check *= static_cast<size_t>(lhs_dilation_buffer[k]) != 0;
      }
      if (check) {
        for (int64_t k = x.Rank() - 3; k >= 0; --k) {
          check *= static_cast<size_t>(x_spacials[k] - padding_buffer[2 * k]) %
                       static_cast<size_t>(lhs_dilation_buffer[k]) ==
                   0;
        }
        if (check) {
          lhs_buffer[i] = x_buffer[j++];
        }
      }
    }
  }
}

// Convolution
template <DataType storage_type>
absl::Status ConvolutionImp(ConvolutionOp& op, size_t& output_channel,
                            const Tensor& lhs, const Tensor& rhs,
                            Tensor& output) {
  using StorageT = StorageType<storage_type>;
  using int64_t = StorageType<DataType::kSI64>;
  const StorageT* lhs_buffer = lhs.GetDataAs<storage_type>();
  const StorageT* rhs_buffer = rhs.GetDataAs<storage_type>();
  StorageT* output_buffer = output.GetDataAs<storage_type>();

  size_t rhs_tensor_size = 1;
  size_t rhs_spacial_size = 1;
  size_t output_spacial_size = 1;
  for (size_t i = 1; i < rhs.Rank(); ++i) {
    rhs_tensor_size *= rhs.shape().Dim(i);
    if (i > 1) {
      output_spacial_size *= output.shape().Dim(i);
      rhs_spacial_size *= rhs.shape().Dim(i);
    }
  }

  Tensor lhs_slice = op.lhs_dot_general;
  Tensor rhs_slice = op.rhs_dot_general;
  Tensor dor_general_output = op.output_dot_general;

  const int64_t* window_spacial_pointer =
      op.attributes.kernel_spacial_dimensions.GetDataAs<DataType::kSI64>();
  const int64_t* output_spacial_pointer =
      op.attributes.output_spacial_dimensions.GetDataAs<DataType::kSI64>();
  const int64_t* input_spacial_pointer =
      op.attributes.input_spacial_dimensions.GetDataAs<DataType::kSI64>();
  const int64_t* rhs_dilation_pointer =
      op.attributes.rhs_dilation.GetDataAs<DataType::kSI64>();
  const int64_t* window_strides_pointer =
      op.attributes.window_strides.GetDataAs<DataType::kSI64>();
  StorageT* lhs_slice_pointer = lhs_slice.GetDataAs<storage_type>();
  StorageT* rhs_slice_pointer = rhs_slice.GetDataAs<storage_type>();

  for (size_t i = 0; i < lhs.shape().Dim(0); ++i) {
    for (size_t j = 0; j < output_spacial_size; ++j) {
      int output_dims[output.Rank()];
      size_t output_depth = 1;
      for (size_t m = output.Rank() - 1; m > 1; --m) {
        output_dims[m] = (j / output_depth) % output.shape().Dim(m);
        output_depth *= output.shape().Dim(m);
      }
      for (size_t k = 0; k < lhs.shape().Dim(1); ++k) {
        for (size_t l = 0; l < rhs_spacial_size; ++l) {
          int filter_spacials[rhs.Rank() - 2];
          size_t depth = 1;
          for (size_t m = rhs.Rank() - 1; m > 1; --m) {
            filter_spacials[m - 2] = (l / depth) % rhs.shape().Dim(m);
            depth *= rhs.shape().Dim(m);
          }

          int lhs_dims[lhs.Rank()];
          lhs_dims[0] = i;
          lhs_dims[1] = k;
          depth = 1;
          size_t lhs_index = 0;
          for (int64_t m = lhs.Rank() - 1; m >= 0; --m) {
            if (m > 1)
              lhs_dims[m] =
                  output_dims[m] * window_strides_pointer[m - 2] +
                  filter_spacials[m - 2] * rhs_dilation_pointer[m - 2];
            lhs_index += lhs_dims[m] * depth;
            depth *= lhs.shape().Dim(m);
          }

          l += k * rhs_spacial_size;
          lhs_slice_pointer[l] = lhs_buffer[lhs_index];
          l -= k * rhs_spacial_size;
        }
      }
      for (size_t k = 0; k < rhs.shape().Dim(0); ++k) {
        for (size_t l = 0; l < rhs_tensor_size; ++l) {
          size_t batchSkip = k * rhs_tensor_size;
          rhs_slice_pointer[l] = rhs_buffer[l + batchSkip];
        }

        auto state = Evaluate(op.dot_general_op, lhs_slice, rhs_slice,
                              dor_general_output);

        StorageT* dor_general_output_buffer =
            dor_general_output.GetDataAs<storage_type>();

        output_dims[0] = i;
        output_dims[1] = output_channel + k;
        output_depth = 1;
        size_t output_index = 0;
        for (int64_t m = output.Rank() - 1; m >= 0; --m) {
          output_index += output_dims[m] * output_depth;
          output_depth *= output.shape().Dim(m);
        }
        output_buffer[output_index] = dor_general_output_buffer[0];
      }
    }
  }
  output_channel += rhs.shape().Dim(0);

  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status EvaluateImpl(ConvolutionOp& op, const Tensor& lhs,
                          const Tensor& rhs, Tensor& output) {
  auto transpose_status = TransposeEvaluateImpl<storage_type>(
      lhs, op.lhs_permutations, op.lhs_transposed);

  transpose_status = TransposeEvaluateImpl<storage_type>(
      rhs, op.rhs_permutations, op.rhs_transposed);

  PaddingOp<storage_type>(op, op.lhs_transposed, op.attributes.padding,
                          op.attributes.lhs_dilation);

  // spliting the lhs and rhs
  size_t output_channel = 0;

  if (op.attributes.feature_group_count > 1) {
    split<storage_type>(op.lhs_padded, op.attributes.feature_group_count, 1,
                        op.lhs_splits);

    split<storage_type>(op.rhs_transposed, op.attributes.feature_group_count, 0,
                        op.rhs_splits);

    for (int64_t i = 0; i < op.attributes.feature_group_count; ++i) {
      auto status =
          ConvolutionImp<storage_type>(op, output_channel, op.lhs_splits[i],
                                       op.rhs_splits[i], op.output_transposed);
    }
    transpose_status = TransposeEvaluateImpl<storage_type>(
        op.output_transposed, op.output_permutations, output);
    return absl::OkStatus();
  } else if (op.attributes.batch_group_count > 1) {
    split<storage_type>(op.lhs_padded, op.attributes.batch_group_count, 0,
                        op.lhs_splits);
    split<storage_type>(op.rhs_transposed, op.attributes.batch_group_count, 0,
                        op.rhs_splits);

    for (int64_t i = 0; i < op.attributes.batch_group_count; ++i) {
      auto status =
          ConvolutionImp<storage_type>(op, output_channel, op.lhs_splits[i],
                                       op.rhs_splits[i], op.output_transposed);
    }
    transpose_status = TransposeEvaluateImpl<storage_type>(
        op.output_transposed, op.output_permutations, output);
    return absl::OkStatus();
  }

  auto status =
      ConvolutionImp<storage_type>(op, output_channel, op.lhs_padded,
                                   op.rhs_transposed, op.output_transposed);
  transpose_status = TransposeEvaluateImpl<storage_type>(
      op.output_transposed, op.output_permutations, output);

  return absl::OkStatus();
}

template <DataType storage_type, DataType expressed_type>
void DequantizeOpQuantizePerTensor(const Tensor& lhs, const Tensor& rhs,
                                   Tensor& output, ConvolutionOp& op) {
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

  auto status = Evaluate(op, op.lhs_dequantized, op.rhs_dequantized,
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
void DequantizeOpQuantizePerAxis(const Tensor& lhs, const Tensor& rhs,
                                 Tensor& output, ConvolutionOp& op) {
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

  auto status = Evaluate(op, op.lhs_dequantized, op.rhs_dequantized,
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

ConvolutionOp Create(const ConvolutionOp::Attributes& attributes) {
  return {.attributes = attributes};
}

absl::Status Prepare(ConvolutionOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output) {
  if (lhs.IsQuantized()) {
    DISPATCH_INT_FLOAT(PrepareImpl,
                       lhs.quantized_per_tensor_element_type().ExpressedType(),
                       op, lhs, rhs, output);
  }
  if (!lhs.IsQuantized()) {
    DISPATCH_INT_FLOAT(PrepareImpl, lhs.StorageType(), op, lhs, rhs, output);
    return absl::OkStatus();
  }
}

absl::Status Evaluate(ConvolutionOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output) {
  if (lhs.IsQuantized()) {
    if (rhs.IsPerTensorQuantized()) {
      DISPATCH_QUANTIZED(
          DequantizeOpQuantizePerTensor,
          output.quantized_per_tensor_element_type().StorageType(),
          output.quantized_per_tensor_element_type().ExpressedType(), lhs, rhs,
          output, op);
    }
    if (rhs.IsPerAxisQuantized()) {
      DISPATCH_QUANTIZED(DequantizeOpQuantizePerAxis,
                         rhs.quantized_per_axis_element_type().StorageType(),
                         rhs.quantized_per_axis_element_type().ExpressedType(),
                         lhs, rhs, output, op);
    }
  } else {
    DISPATCH_INT_FLOAT(EvaluateImpl, output.tensor_element_type(), op, lhs, rhs,
                       output);
  }
}
}  // namespace shlo_ref