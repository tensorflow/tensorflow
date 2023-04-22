/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/xla/attribute_importer.h"

#include <vector>

#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

static mlir::DenseIntElementsAttr Convert(llvm::ArrayRef<int64_t> elements,
                                          mlir::Builder* builder) {
  return mlir::DenseIntElementsAttr::get(
      mlir::RankedTensorType::get(elements.size(), builder->getIntegerType(64)),
      elements);
}

mlir::ArrayAttr ConvertPrecisionConfig(const PrecisionConfig* config,
                                       mlir::Builder* builder) {
  if (!config) return {};

  // TODO(b/129709049) The HLO text format elides this in the all DEFAULT
  // case and the parser sticks it in. Maybe we should too.
  llvm::SmallVector<mlir::Attribute, 4> operand_precision_attrs;

  for (auto prec : config->operand_precision()) {
    operand_precision_attrs.push_back(
        builder->getStringAttr(PrecisionConfig_Precision_Name(prec)));
  }
  return builder->getArrayAttr(operand_precision_attrs);
}

// Converts the gather dimensions to attributes.
mlir::mhlo::GatherDimensionNumbers ConvertGatherDimensionNumbers(
    const xla::GatherDimensionNumbers& dnums, mlir::Builder* builder) {
  std::vector<int64_t> offset_dims(dnums.offset_dims().begin(),
                                   dnums.offset_dims().end());
  std::vector<int64_t> collapsed_slice_dims(
      dnums.collapsed_slice_dims().begin(), dnums.collapsed_slice_dims().end());
  std::vector<int64_t> start_index_map(dnums.start_index_map().begin(),
                                       dnums.start_index_map().end());
  return mlir::mhlo::GatherDimensionNumbers::get(
      Convert(offset_dims, builder), Convert(collapsed_slice_dims, builder),
      Convert(start_index_map, builder),
      builder->getI64IntegerAttr(dnums.index_vector_dim()),
      builder->getContext());
}

mlir::mhlo::ScatterDimensionNumbers ConvertScatterDimensionNumbers(
    const xla::ScatterDimensionNumbers& dnums, mlir::Builder* builder) {
  std::vector<int64_t> update_window_dims(dnums.update_window_dims().begin(),
                                          dnums.update_window_dims().end());
  std::vector<int64_t> inserted_window_dims(
      dnums.inserted_window_dims().begin(), dnums.inserted_window_dims().end());
  std::vector<int64_t> scatter_dims_to_operand_dims(
      dnums.scatter_dims_to_operand_dims().begin(),
      dnums.scatter_dims_to_operand_dims().end());
  return mlir::mhlo::ScatterDimensionNumbers::get(
      Convert(update_window_dims, builder),
      Convert(inserted_window_dims, builder),
      Convert(scatter_dims_to_operand_dims, builder),
      builder->getI64IntegerAttr(dnums.index_vector_dim()),
      builder->getContext());
}

mlir::mhlo::DotDimensionNumbers ConvertDotDimensionNumbers(
    const DotDimensionNumbers& dnums, mlir::Builder* builder) {
  std::vector<int64_t> rhs_contracting_dimensions(
      dnums.rhs_contracting_dimensions().begin(),
      dnums.rhs_contracting_dimensions().end());
  std::vector<int64_t> lhs_contracting_dimensions(
      dnums.lhs_contracting_dimensions().begin(),
      dnums.lhs_contracting_dimensions().end());
  std::vector<int64_t> rhs_batch_dimensions(
      dnums.rhs_batch_dimensions().begin(), dnums.rhs_batch_dimensions().end());
  std::vector<int64_t> lhs_batch_dimensions(
      dnums.lhs_batch_dimensions().begin(), dnums.lhs_batch_dimensions().end());

  // Push the attributes into our new DictionaryAttr.
  auto lhs_batch_dims_attr = Convert(lhs_batch_dimensions, builder);
  auto rhs_batch_dims_attr = Convert(rhs_batch_dimensions, builder);
  auto lhs_contracting_dims_attr = Convert(lhs_contracting_dimensions, builder);
  auto rhs_contracting_dims_attr = Convert(rhs_contracting_dimensions, builder);

  return mlir::mhlo::DotDimensionNumbers::get(
      lhs_batch_dims_attr, rhs_batch_dims_attr, lhs_contracting_dims_attr,
      rhs_contracting_dims_attr, builder->getContext());
}

mlir::mhlo::ConvDimensionNumbers ConvertConvDimensionNumbers(
    const xla::ConvolutionDimensionNumbers& dnums, mlir::Builder* builder) {
  llvm::SmallVector<int64_t, 4> input_spatial_dims(
      dnums.input_spatial_dimensions().begin(),
      dnums.input_spatial_dimensions().end());
  llvm::SmallVector<int64_t, 4> kernel_spatial_dims(
      dnums.kernel_spatial_dimensions().begin(),
      dnums.kernel_spatial_dimensions().end());
  llvm::SmallVector<int64_t, 4> output_spatial_dims(
      dnums.output_spatial_dimensions().begin(),
      dnums.output_spatial_dimensions().end());
  return mlir::mhlo::ConvDimensionNumbers::get(
      builder->getI64IntegerAttr(dnums.input_batch_dimension()),
      builder->getI64IntegerAttr(dnums.input_feature_dimension()),
      Convert(input_spatial_dims, builder),
      builder->getI64IntegerAttr(dnums.kernel_input_feature_dimension()),
      builder->getI64IntegerAttr(dnums.kernel_output_feature_dimension()),
      Convert(kernel_spatial_dims, builder),
      builder->getI64IntegerAttr(dnums.output_batch_dimension()),
      builder->getI64IntegerAttr(dnums.output_feature_dimension()),
      Convert(output_spatial_dims, builder), builder->getContext());
}

StatusOr<mlir::mhlo::FftType> ConvertFftType(FftType type) {
  switch (type) {
    case FftType::FFT:
      return mlir::mhlo::FftType::FFT;
    case FftType::IFFT:
      return mlir::mhlo::FftType::IFFT;
    case FftType::RFFT:
      return mlir::mhlo::FftType::RFFT;
    case FftType::IRFFT:
      return mlir::mhlo::FftType::IRFFT;
    default:
      return InvalidArgument("Unknown FFT type enum value #%d", type);
  }
}

StatusOr<mlir::mhlo::Transpose> ConvertTranspose(
    xla::TriangularSolveOptions_Transpose transpose) {
  switch (transpose) {
    case TriangularSolveOptions::NO_TRANSPOSE:
      return mlir::mhlo::Transpose::NO_TRANSPOSE;
    case TriangularSolveOptions::TRANSPOSE:
      return mlir::mhlo::Transpose::TRANSPOSE;
    case TriangularSolveOptions::ADJOINT:
      return mlir::mhlo::Transpose::ADJOINT;
    case TriangularSolveOptions::TRANSPOSE_INVALID:
      return mlir::mhlo::Transpose::TRANSPOSE_INVALID;
    default:
      return InvalidArgument("Unknown transpose enum value #%d", transpose);
  }
}

}  // namespace xla
