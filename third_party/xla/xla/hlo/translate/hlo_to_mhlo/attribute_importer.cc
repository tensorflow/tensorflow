/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/hlo/translate/hlo_to_mhlo/attribute_importer.h"

#include <sys/types.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace stablehlo {

mlir::stablehlo::GatherDimensionNumbersAttr ConvertGatherDimensionNumbers(
    const xla::GatherDimensionNumbers& dnums, mlir::Builder* builder) {
  std::vector<int64_t> offset_dims(dnums.offset_dims().begin(),
                                   dnums.offset_dims().end());
  std::vector<int64_t> collapsed_slice_dims(
      dnums.collapsed_slice_dims().begin(), dnums.collapsed_slice_dims().end());
  std::vector<int64_t> operand_batching_dims(
      dnums.operand_batching_dims().begin(),
      dnums.operand_batching_dims().end());
  std::vector<int64_t> start_indices_batching_dims(
      dnums.start_indices_batching_dims().begin(),
      dnums.start_indices_batching_dims().end());
  std::vector<int64_t> start_index_map(dnums.start_index_map().begin(),
                                       dnums.start_index_map().end());
  return mlir::stablehlo::GatherDimensionNumbersAttr::get(
      builder->getContext(), offset_dims, collapsed_slice_dims,
      operand_batching_dims, start_indices_batching_dims, start_index_map,
      dnums.index_vector_dim());
}

mlir::stablehlo::ScatterDimensionNumbersAttr ConvertScatterDimensionNumbers(
    const xla::ScatterDimensionNumbers& dnums, mlir::Builder* builder) {
  std::vector<int64_t> update_window_dims(dnums.update_window_dims().begin(),
                                          dnums.update_window_dims().end());
  std::vector<int64_t> inserted_window_dims(
      dnums.inserted_window_dims().begin(), dnums.inserted_window_dims().end());
  std::vector<int64_t> input_batching_dims(dnums.input_batching_dims().begin(),
                                           dnums.input_batching_dims().end());
  std::vector<int64_t> scatter_indices_batching_dims(
      dnums.scatter_indices_batching_dims().begin(),
      dnums.scatter_indices_batching_dims().end());
  std::vector<int64_t> scatter_dims_to_operand_dims(
      dnums.scatter_dims_to_operand_dims().begin(),
      dnums.scatter_dims_to_operand_dims().end());
  return mlir::stablehlo::ScatterDimensionNumbersAttr::get(
      builder->getContext(), update_window_dims, inserted_window_dims,
      input_batching_dims, scatter_indices_batching_dims,
      scatter_dims_to_operand_dims, dnums.index_vector_dim());
}

mlir::NamedAttribute ConvertChannelHandle(const ChannelHandle& channel,
                                          mlir::Builder* builder) {
  return builder->getNamedAttr(
      "channel_handle",
      mlir::stablehlo::ChannelHandleAttr::get(
          builder->getContext(), channel.handle(), channel.type()));
}

mlir::NamedAttribute ConvertChannelHandle(std::optional<int64_t> channel_id,
                                          mlir::Builder* builder) {
  ChannelHandle channel_handle;
  if (channel_id) channel_handle.set_handle(*channel_id);
  return stablehlo::ConvertChannelHandle(channel_handle, builder);
}

mlir::stablehlo::ConvDimensionNumbersAttr ConvertConvDimensionNumbers(
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
  return mlir::stablehlo::ConvDimensionNumbersAttr::get(
      builder->getContext(), dnums.input_batch_dimension(),
      dnums.input_feature_dimension(),
      arrayref(dnums.input_spatial_dimensions()),
      dnums.kernel_input_feature_dimension(),
      dnums.kernel_output_feature_dimension(),
      arrayref(dnums.kernel_spatial_dimensions()),
      dnums.output_batch_dimension(), dnums.output_feature_dimension(),
      arrayref(dnums.output_spatial_dimensions()));
}

absl::StatusOr<mlir::stablehlo::CustomCallApiVersion>
ConvertCustomCallApiVersion(xla::CustomCallApiVersion api_version) {
  switch (api_version) {
    case xla::CustomCallApiVersion::API_VERSION_UNSPECIFIED:
      return mlir::stablehlo::CustomCallApiVersion::API_VERSION_UNSPECIFIED;
    case xla::CustomCallApiVersion::API_VERSION_ORIGINAL:
      return mlir::stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL;
    case xla::CustomCallApiVersion::API_VERSION_STATUS_RETURNING:
      return mlir::stablehlo::CustomCallApiVersion::
          API_VERSION_STATUS_RETURNING;
    case xla::CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED:
      return mlir::stablehlo::CustomCallApiVersion::
          API_VERSION_STATUS_RETURNING_UNIFIED;
    case xla::CustomCallApiVersion::API_VERSION_TYPED_FFI:
      return mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI;
    default:
      return InvalidArgument("Unknown CustomCallApiVersion enum value #%d (%s)",
                             api_version,
                             xla::CustomCallApiVersion_Name(api_version));
  }
}

mlir::stablehlo::DotAlgorithmAttr ConvertDotAlgorithm(
    const PrecisionConfig::Algorithm algorithm, mlir::Builder* builder) {
  mlir::Type lhs, rhs, accum;
  int64_t lhsComponentCount = 1, rhsComponentCount = 1,
          numPrimitiveOperations = 1;
  bool allowImpreciseAccumulation = false;
  switch (algorithm) {
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32: {
      lhs = rhs = builder->getType<mlir::Float8E5M2Type>();
      accum = builder->getF32Type();
      break;
    }
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM: {
      lhs = rhs = builder->getType<mlir::Float8E5M2Type>();
      accum = builder->getF32Type();
      allowImpreciseAccumulation = true;
      break;
    }
    case PrecisionConfig::ALG_DOT_F16_F16_F16: {
      lhs = rhs = accum = builder->getF16Type();
      break;
    }
    case PrecisionConfig::ALG_DOT_F16_F16_F32: {
      lhs = rhs = builder->getF16Type();
      accum = builder->getF32Type();
      break;
    }
    case PrecisionConfig::ALG_DOT_BF16_BF16_BF16: {
      lhs = rhs = accum = builder->getBF16Type();
      break;
    }
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32: {
      lhs = rhs = builder->getBF16Type();
      accum = builder->getF32Type();
      break;
    }
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3: {
      lhs = rhs = builder->getBF16Type();
      accum = builder->getF32Type();
      numPrimitiveOperations = 3;
      break;
    }
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6: {
      lhs = rhs = builder->getBF16Type();
      accum = builder->getF32Type();
      numPrimitiveOperations = 6;
      break;
    }
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32: {
      lhs = rhs = builder->getTF32Type();
      accum = builder->getF32Type();
      break;
    }
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3: {
      lhs = rhs = builder->getTF32Type();
      accum = builder->getF32Type();
      numPrimitiveOperations = 3;
      break;
    }
    case PrecisionConfig::ALG_DOT_F32_F32_F32: {
      lhs = rhs = accum = builder->getF32Type();
      break;
    }
    case PrecisionConfig::ALG_DOT_F64_F64_F64: {
      lhs = rhs = accum = builder->getF64Type();
      break;
    }
    default:
      // Unset, sentinels
      return mlir::stablehlo::DotAlgorithmAttr{};
  }
  return mlir::stablehlo::DotAlgorithmAttr::get(
      builder->getContext(), lhs, rhs, accum, lhsComponentCount,
      rhsComponentCount, numPrimitiveOperations, allowImpreciseAccumulation);
}

mlir::stablehlo::DotDimensionNumbersAttr ConvertDotDimensionNumbers(
    const DotDimensionNumbers& dnums, mlir::Builder* builder) {
  auto arrayref = [](absl::Span<const int64_t> array) {
    return llvm::ArrayRef<int64_t>{array.data(), array.size()};
  };
  return mlir::stablehlo::DotDimensionNumbersAttr::get(
      builder->getContext(), arrayref(dnums.lhs_batch_dimensions()),
      arrayref(dnums.rhs_batch_dimensions()),
      arrayref(dnums.lhs_contracting_dimensions()),
      arrayref(dnums.rhs_contracting_dimensions()));
}

mlir::ArrayAttr ConvertOutputOperandAliasing(
    const std::vector<std::pair<
        xla::ShapeIndex, std::pair<int64_t, xla::ShapeIndex>>>& aliasInfo,
    mlir::Builder* builder) {
  auto arrayref = [](absl::Span<const int64_t> array) {
    return llvm::ArrayRef<int64_t>{array.data(), array.size()};
  };
  std::vector<mlir::Attribute> attrs;
  for (auto& [output_tuple_idx, operand_idx] : aliasInfo) {
    auto attr = mlir::stablehlo::OutputOperandAliasAttr::get(
        builder->getContext(), arrayref(output_tuple_idx), operand_idx.first,
        arrayref(operand_idx.second));
    attrs.push_back(attr);
  }
  return builder->getArrayAttr(attrs);
}

mlir::ArrayAttr ConvertPrecisionConfig(const PrecisionConfig* config,
                                       mlir::Builder* builder) {
  if (!config) return {};

  // TODO(b/129709049) The HLO text format elides this in the all DEFAULT
  // case and the parser sticks it in. Maybe we should too.
  llvm::SmallVector<mlir::Attribute, 4> operand_precision_attrs;

  for (auto prec : config->operand_precision()) {
    operand_precision_attrs.push_back(mlir::stablehlo::PrecisionAttr::get(
        builder->getContext(), mlir::stablehlo::symbolizePrecision(
                                   PrecisionConfig_Precision_Name(prec))
                                   .value()));
  }
  return builder->getArrayAttr(operand_precision_attrs);
}

mlir::stablehlo::ResultAccuracyAttr ConvertResultAccuracy(
    const ResultAccuracy& result_accuracy, mlir::Builder* builder) {
  if (result_accuracy.has_tolerance()) {
    return mlir::stablehlo::ResultAccuracyAttr::get(
        builder->getContext(),
        llvm::APFloat(result_accuracy.tolerance().atol()),
        llvm::APFloat(result_accuracy.tolerance().rtol()),
        result_accuracy.tolerance().ulps(),
        // Explicitly set the mode to TOLERANCE since ResultAccuracy has no
        // TOLERANCE enum.
        mlir::stablehlo::ResultAccuracyModeAttr::get(
            builder->getContext(),
            mlir::stablehlo::ResultAccuracyMode::TOLERANCE));
  }
  return mlir::stablehlo::ResultAccuracyAttr::get(
      builder->getContext(), llvm::APFloat(0.0), llvm::APFloat(0.0), 0,
      mlir::stablehlo::ResultAccuracyModeAttr::get(
          builder->getContext(),
          mlir::stablehlo::symbolizeResultAccuracyMode(result_accuracy.mode())
              .value()));
}

}  // namespace stablehlo

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

// Converts the gather dimensions to attribute.
mlir::mhlo::GatherDimensionNumbersAttr ConvertGatherDimensionNumbers(
    const xla::GatherDimensionNumbers& dnums, mlir::Builder* builder) {
  std::vector<int64_t> offset_dims(dnums.offset_dims().begin(),
                                   dnums.offset_dims().end());
  std::vector<int64_t> collapsed_slice_dims(
      dnums.collapsed_slice_dims().begin(), dnums.collapsed_slice_dims().end());
  std::vector<int64_t> operand_batching_dims(
      dnums.operand_batching_dims().begin(),
      dnums.operand_batching_dims().end());
  std::vector<int64_t> start_indices_batching_dims(
      dnums.start_indices_batching_dims().begin(),
      dnums.start_indices_batching_dims().end());
  std::vector<int64_t> start_index_map(dnums.start_index_map().begin(),
                                       dnums.start_index_map().end());
  return mlir::mhlo::GatherDimensionNumbersAttr::get(
      builder->getContext(), offset_dims, collapsed_slice_dims,
      operand_batching_dims, start_indices_batching_dims, start_index_map,
      dnums.index_vector_dim());
}

mlir::mhlo::ScatterDimensionNumbersAttr ConvertScatterDimensionNumbers(
    const xla::ScatterDimensionNumbers& dnums, mlir::Builder* builder) {
  std::vector<int64_t> update_window_dims(dnums.update_window_dims().begin(),
                                          dnums.update_window_dims().end());
  std::vector<int64_t> inserted_window_dims(
      dnums.inserted_window_dims().begin(), dnums.inserted_window_dims().end());
  std::vector<int64_t> input_batching_dims(dnums.input_batching_dims().begin(),
                                           dnums.input_batching_dims().end());
  std::vector<int64_t> scatter_indices_batching_dims(
      dnums.scatter_indices_batching_dims().begin(),
      dnums.scatter_indices_batching_dims().end());
  std::vector<int64_t> scatter_dims_to_operand_dims(
      dnums.scatter_dims_to_operand_dims().begin(),
      dnums.scatter_dims_to_operand_dims().end());
  return mlir::mhlo::ScatterDimensionNumbersAttr::get(
      builder->getContext(), update_window_dims, inserted_window_dims,
      input_batching_dims, scatter_indices_batching_dims,
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

mlir::mhlo::RaggedDotDimensionNumbersAttr ConvertRaggedDotDimensionNumbers(
    const RaggedDotDimensionNumbers& dnums, mlir::Builder* builder) {
  auto arrayref = [](absl::Span<const int64_t> array) {
    return llvm::ArrayRef<int64_t>{array.data(), array.size()};
  };
  return mlir::mhlo::RaggedDotDimensionNumbersAttr::get(
      builder->getContext(),
      ConvertDotDimensionNumbers(dnums.dot_dimension_numbers(), builder),
      arrayref(dnums.lhs_ragged_dimensions()),
      arrayref(dnums.rhs_group_dimensions()));
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

absl::StatusOr<mlir::mhlo::SparsityDescriptorAttr> ConvertSparsityDescriptor(
    xla::SparsityDescriptor sparsity_descriptor, mlir::Builder* builder) {
  switch (sparsity_descriptor.type()) {
    case SPARSITY_STRUCTURED_N_M:
      return mlir::mhlo::SparsityDescriptorAttr::get(
          builder->getContext(), sparsity_descriptor.dimension(),
          sparsity_descriptor.n(), sparsity_descriptor.m());
    default:
      return InvalidArgument("Unknown sparsity descriptor type");
  }
}

absl::StatusOr<mlir::mhlo::CustomCallApiVersion> ConvertCustomCallApiVersion(
    xla::CustomCallApiVersion api_version) {
  TF_ASSIGN_OR_RETURN(auto stablehlo_api_version,
                      stablehlo::ConvertCustomCallApiVersion(api_version));
  auto mhlo_api_version = mlir::mhlo::symbolizeCustomCallApiVersion(
      mlir::stablehlo::stringifyCustomCallApiVersion(stablehlo_api_version));
  if (!mhlo_api_version.has_value())
    return InvalidArgument("Unknown CustomCallApiVersion enum value #%d",
                           api_version);
  return mhlo_api_version.value();
}

mlir::NamedAttribute ConvertChannelHandle(const ChannelHandle& channel,
                                          mlir::Builder* builder) {
  return builder->getNamedAttr(
      "channel_handle",
      mlir::mhlo::ChannelHandleAttr::get(builder->getContext(),
                                         channel.handle(), channel.type()));
}

mlir::NamedAttribute ConvertChannelHandle(std::optional<int64_t> channel_id,
                                          mlir::Builder* builder) {
  ChannelHandle channel_handle;
  if (channel_id) channel_handle.set_handle(*channel_id);
  return ConvertChannelHandle(channel_handle, builder);
}

mlir::NamedAttribute ConvertReplicaGroups(
    absl::Span<const ReplicaGroup> replica_groups, mlir::Builder* builder) {
  const int64_t num_groups = replica_groups.size();
  // Replica groups in HLO can be non-uniform in size, for example:
  // replica_groups={{0},{1,2},{3}}. Since we are representing them as a 2D
  // tensor, pad the smaller sized replica groups with -1.
  const int64_t group_size = absl::c_accumulate(
      replica_groups, static_cast<int64_t>(0),
      [](int64_t current, const ReplicaGroup& g) {
        return std::max<int64_t>(current, g.replica_ids_size());
      });
  // Initialize all elements to -1 to support non-uniform replica groups.
  std::vector<int64_t> attr(num_groups * group_size, -1);
  for (int i = 0; i < num_groups; ++i) {
    int index = i * group_size;
    for (const int64_t& id : replica_groups[i].replica_ids())
      attr[index++] = id;
  }
  auto type = mlir::RankedTensorType::get({num_groups, group_size},
                                          builder->getIntegerType(64));
  return builder->getNamedAttr("replica_groups",
                               mlir::DenseIntElementsAttr::get(type, attr));
}

mlir::NamedAttribute ConvertSourceTargetPairs(
    const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
    mlir::Builder* builder) {
  std::vector<int64_t> attr(source_target_pairs.size() * 2);
  for (const auto& p : llvm::enumerate(source_target_pairs)) {
    attr[2 * p.index()] = p.value().first;
    attr[2 * p.index() + 1] = p.value().second;
  }
  auto type = mlir::RankedTensorType::get(
      {static_cast<int64_t>(attr.size() / 2), 2}, builder->getIntegerType(64));
  return builder->getNamedAttr("source_target_pairs",
                               mlir::DenseIntElementsAttr::get(type, attr));
}

mlir::NamedAttribute ConvertUseGlobalDeviceIds(mlir::Builder* builder) {
  return builder->getNamedAttr("use_global_device_ids", builder->getUnitAttr());
}

absl::StatusOr<mlir::ArrayAttr> ExtractLayoutsFromShapes(
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
    if (!shape_and_layout.IsArray()) {
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

absl::StatusOr<mlir::ArrayAttr> ExtractLayoutsFromTuple(
    const Shape shape, mlir::Builder* builder) {
  if (!shape.IsTuple()) return InvalidArgument("Expected shape to be Tuple");
  return ExtractLayoutsFromShapes(shape.tuple_shapes(), builder);
}

}  // namespace xla
