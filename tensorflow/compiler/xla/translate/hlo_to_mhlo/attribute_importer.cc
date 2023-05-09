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

#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/attribute_importer.h"

#include <sys/types.h>

#include <optional>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/hlo/ir/dynamic_parameter_binding.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

mlir::ArrayAttr ConvertPrecisionConfig(const PrecisionConfig* config,
                                       mlir::Builder* builder) {
  if (!config) return {};

  // TODO(b/129709049) The HLO text format elides this in the all DEFAULT
  // case and the parser sticks it in. Maybe we should too.
  llvm::SmallVector<mlir::Attribute, 4> operand_precision_attrs;

  for (auto prec : config->operand_precision()) {
    operand_precision_attrs.push_back(mlir::mhlo::PrecisionAttr::get(
        builder->getContext(),
        mlir::mhlo::symbolizePrecision(PrecisionConfig_Precision_Name(prec))
            .value()));
  }
  return builder->getArrayAttr(operand_precision_attrs);
}

// Converts the gather dimensions to attributes.
mlir::mhlo::GatherDimensionNumbersAttr ConvertGatherDimensionNumbers(
    const xla::GatherDimensionNumbers& dnums, mlir::Builder* builder) {
  std::vector<int64_t> offset_dims(dnums.offset_dims().begin(),
                                   dnums.offset_dims().end());
  std::vector<int64_t> collapsed_slice_dims(
      dnums.collapsed_slice_dims().begin(), dnums.collapsed_slice_dims().end());
  std::vector<int64_t> start_index_map(dnums.start_index_map().begin(),
                                       dnums.start_index_map().end());
  return mlir::mhlo::GatherDimensionNumbersAttr::get(
      builder->getContext(), offset_dims, collapsed_slice_dims, start_index_map,
      dnums.index_vector_dim());
}

mlir::mhlo::ScatterDimensionNumbersAttr ConvertScatterDimensionNumbers(
    const xla::ScatterDimensionNumbers& dnums, mlir::Builder* builder) {
  std::vector<int64_t> update_window_dims(dnums.update_window_dims().begin(),
                                          dnums.update_window_dims().end());
  std::vector<int64_t> inserted_window_dims(
      dnums.inserted_window_dims().begin(), dnums.inserted_window_dims().end());
  std::vector<int64_t> scatter_dims_to_operand_dims(
      dnums.scatter_dims_to_operand_dims().begin(),
      dnums.scatter_dims_to_operand_dims().end());
  return mlir::mhlo::ScatterDimensionNumbersAttr::get(
      builder->getContext(), update_window_dims, inserted_window_dims,
      scatter_dims_to_operand_dims, dnums.index_vector_dim());
}

mlir::mhlo::DotDimensionNumbersAttr ConvertDotDimensionNumbers(
    const DotDimensionNumbers& dnums, mlir::Builder* builder) {
  auto arrayref = [](absl::Span<const int64_t> array) {
    return llvm::ArrayRef<int64_t>{array.data(), array.size()};
  };
  return mlir::mhlo::DotDimensionNumbersAttr::get(
      builder->getContext(), arrayref(dnums.lhs_batch_dimensions()),
      arrayref(dnums.rhs_batch_dimensions()),
      arrayref(dnums.lhs_contracting_dimensions()),
      arrayref(dnums.rhs_contracting_dimensions()));
}

mlir::mhlo::ConvDimensionNumbersAttr ConvertConvDimensionNumbers(
    const xla::ConvolutionDimensionNumbers& dnums, mlir::Builder* builder) {
  auto arrayref = [](absl::Span<const int64_t> array) {
    return llvm::ArrayRef<int64_t>{array.data(), array.size()};
  };
  llvm::SmallVector<int64_t, 4> input_spatial_dims(
      dnums.input_spatial_dimensions().begin(),
      dnums.input_spatial_dimensions().end());
  llvm::SmallVector<int64_t, 4> kernel_spatial_dims(
      dnums.kernel_spatial_dimensions().begin(),
      dnums.kernel_spatial_dimensions().end());
  llvm::SmallVector<int64_t, 4> output_spatial_dims(
      dnums.output_spatial_dimensions().begin(),
      dnums.output_spatial_dimensions().end());
  return mlir::mhlo::ConvDimensionNumbersAttr::get(
      builder->getContext(), dnums.input_batch_dimension(),
      dnums.input_feature_dimension(),
      arrayref(dnums.input_spatial_dimensions()),
      dnums.kernel_input_feature_dimension(),
      dnums.kernel_output_feature_dimension(),
      arrayref(dnums.kernel_spatial_dimensions()),
      dnums.output_batch_dimension(), dnums.output_feature_dimension(),
      arrayref(dnums.output_spatial_dimensions()));
}

mlir::ArrayAttr ConvertOutputOperandAliasing(
    const std::vector<std::pair<xla::ShapeIndex,
                                std::pair<int64_t, xla::ShapeIndex>>>& aliaInfo,
    mlir::Builder* builder) {
  auto arrayref = [](absl::Span<const int64_t> array) {
    return llvm::ArrayRef<int64_t>{array.data(), array.size()};
  };
  std::vector<mlir::Attribute> attrs;
  for (auto& aliasing : aliaInfo) {
    auto attr = mlir::mhlo::OutputOperandAliasAttr::get(
        builder->getContext(), arrayref(aliasing.first), aliasing.second.first,
        arrayref(aliasing.second.second));
    attrs.push_back(attr);
  }
  return builder->getArrayAttr(attrs);
}

mlir::ArrayAttr ConvertDynamicParameterBindings(
    const DynamicParameterBinding dpb, mlir::Builder* builder) {
  llvm::SmallVector<mlir::Attribute, 4> bindings;
  (void)dpb.ForEachBinding(
      [&](const DynamicParameterBinding::DynamicParameter& source,
          const DynamicParameterBinding::DynamicDimension& target) {
        llvm::SmallVector<int64_t, 4> dpis;
        for (auto dpi : source.parameter_index) dpis.push_back(dpi);
        llvm::SmallVector<int64_t, 4> tpis;
        for (auto tpi : target.parameter_index) tpis.push_back(tpi);
        bindings.push_back(mlir::mhlo::DynamicParameterBindingAttr::get(
            builder->getContext(), source.parameter_num, dpis,
            target.parameter_num, tpis, target.dimension));
        return OkStatus();
      });
  return mlir::ArrayAttr::get(builder->getContext(), bindings);
}

mlir::ArrayAttr ConvertCrossProgramPrefetches(
    const absl::Span<const xla::HloModule::CrossProgramPrefetchInfo> prefetches,
    mlir::Builder* builder) {
  llvm::SmallVector<mlir::Attribute, 4> shapes;
  for (auto [parameter, index, alt_memory_offset] : prefetches) {
    llvm::SmallVector<int64_t, 4> dims;
    for (auto dim : index) dims.push_back(dim);
    std::optional<int64_t> offset =
        alt_memory_offset ? std::optional<int64_t>(*alt_memory_offset)
                          : std::nullopt;
    shapes.push_back(mlir::mhlo::CrossProgramPrefetchAttr::get(
        builder->getContext(), parameter, dims, offset));
  }

  return mlir::ArrayAttr::get(builder->getContext(), shapes);
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

StatusOr<mlir::mhlo::CustomCallApiVersion> ConvertCustomCallApiVersion(
    xla::CustomCallApiVersion api_version) {
  switch (api_version) {
    case xla::CustomCallApiVersion::API_VERSION_UNSPECIFIED:
      return mlir::mhlo::CustomCallApiVersion::API_VERSION_UNSPECIFIED;
    case xla::CustomCallApiVersion::API_VERSION_ORIGINAL:
      return mlir::mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL;
    case xla::CustomCallApiVersion::API_VERSION_STATUS_RETURNING:
      return mlir::mhlo::CustomCallApiVersion::API_VERSION_STATUS_RETURNING;
    case xla::CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED:
      return mlir::mhlo::CustomCallApiVersion::
          API_VERSION_STATUS_RETURNING_UNIFIED;
    case xla::CustomCallApiVersion::API_VERSION_TYPED_FFI:
      return mlir::mhlo::CustomCallApiVersion::API_VERSION_TYPED_FFI;
    default:
      return InvalidArgument("Unknown CustomCallApiVersion enum value #%d (%s)",
                             api_version,
                             xla::CustomCallApiVersion_Name(api_version));
  }
}

StatusOr<mlir::ArrayAttr> ExtractLayoutsFromShapes(
    const absl::Span<const Shape> shapes_with_layouts, mlir::Builder* builder) {
  std::vector<mlir::Attribute> layouts;
  for (auto& shape_and_layout : shapes_with_layouts) {
    if (shape_and_layout.IsTuple())
      return Unimplemented(
          "Layout support for nested tuples is not implemented.");
    // XLA can have invalid layout for certain values (such as token types).
    // These are imported as empty layout in MHLO.
    if (!shape_and_layout.IsArray()) {
      layouts.push_back(builder->getIndexTensorAttr({}));
      continue;
    }

    // Only a subset of layout specification in XLA is supported in MHLO
    // currently. The layout has to be dense, and only specify the order of
    // dimensions. Sparse, tiled layout or non-default memory space fields
    // cannot be expressed in MHLO layout yet.
    if (!xla::LayoutUtil::IsDenseArray(shape_and_layout)) {
      return Unimplemented("Only dense arrays are supported.");
    }

    const xla::Layout& xla_layout = shape_and_layout.layout();
    if (!xla_layout.tiles().empty())
      return Unimplemented("Tiled layout is not supported yet");
    if (xla_layout.memory_space() != xla::Layout::kDefaultMemorySpace)
      return Unimplemented(
          "Layout support for non-default memory space is not yet implemented");

    llvm::SmallVector<int64_t> layout;
    for (int64_t dim_index : xla_layout.minor_to_major())
      layout.push_back(dim_index);
    layouts.push_back(builder->getIndexTensorAttr(layout));
  }
  return builder->getArrayAttr(layouts);
}

StatusOr<mlir::ArrayAttr> ExtractLayoutsFromTuple(const Shape shape,
                                                  mlir::Builder* builder) {
  if (!shape.IsTuple()) return InvalidArgument("Expected shape to be Tuple");
  return ExtractLayoutsFromShapes(shape.tuple_shapes(), builder);
}

}  // namespace xla
