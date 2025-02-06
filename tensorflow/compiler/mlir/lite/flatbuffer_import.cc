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

#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/AssumeBundleQueries.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/core/absl_error_model_builder.h"
#include "tensorflow/compiler/mlir/lite/experimental/remat/metadata_util.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_operator.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/offset_buffer.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/schema/mutable/debug_metadata_generated.h"
#include "tensorflow/compiler/mlir/lite/schema/mutable/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_utils.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"
#include "tensorflow/compiler/mlir/lite/utils/const_tensor_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/control_edges.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/lite/utils/size_utils.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_traits.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

using absl::Status;
using absl::StatusOr;
using llvm::ArrayRef;
using mlir::Builder;
using mlir::DenseElementsAttr;
using mlir::Location;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationState;
using mlir::OwningOpRef;
using mlir::RankedTensorType;
using mlir::Value;
using mlir::func::FuncOp;
using tflite::OperatorT;
using tflite::TensorT;

namespace errors = tensorflow::errors;
namespace tfl = mlir::TFL;

namespace {

constexpr char kScatterRegionFuncName[] = "update_computation_func_name";

using ::mlir::tf_saved_model::kTfSavedModelExportedNamesAttr;
using ::mlir::tf_saved_model::kTfSavedModelIndexPathAttr;
using ::tflite::IsValidBufferOffset;

struct DebugMetadata {
  // Debug metadata locations.
  std::vector<mlir::Location> debug_metadata_locations;

  // Maps from operator (subgraph_debug_metadata_idx,
  // operator_debug_metadata_idx) to its top-level location index in
  // `debug_metadata_locations`, which is:
  // <<subgraph_debug_metadata_idx, operator_debug_metadata_idx>, location_idx>.
  absl::flat_hash_map<int, absl::flat_hash_map<int, int>> operator_location_map;
};

// Create the MLIR NamedLoc location corresponding to a given tensor
Location TensorLoc(const TensorT& tensor, Builder builder, Location base) {
  if (tensor.name.empty()) {
    return base;
  }
  return mlir::NameLoc::get(builder.getStringAttr(tensor.name), base);
}

// Build and return the MLIR location.
StatusOr<mlir::Location> BuildLocation(
    Builder builder, const debug_metadata::Location& location,
    const std::vector<mlir::Location>& debug_metadata_locations,
    const absl::flat_hash_map<unsigned int, unsigned int>&
        attribute_location_idx_map) {
  switch (location.location_type()) {
    // FileLineColLoc.
    case debug_metadata::LocationType_FileLineColLoc: {
      auto file_line_col_loc =
          static_cast<const debug_metadata::FileLineColLoc*>(
              location.location());
      return mlir::FileLineColLoc::get(
          builder.getContext(),
          builder.getStringAttr(file_line_col_loc->filename()->string_view()),
          file_line_col_loc->line(), file_line_col_loc->column());
    }
    // CallSiteLoc.
    case debug_metadata::LocationType_CallSiteLoc: {
      auto callsite_loc =
          static_cast<const debug_metadata::CallSiteLoc*>(location.location());
      if (!attribute_location_idx_map.contains(callsite_loc->callee_index()) ||
          !attribute_location_idx_map.contains(callsite_loc->caller_index())) {
        return absl::InternalError(
            "Invalid/corrupt DebugMetadata, expected invariant broken (callee "
            "or caller index of a CallSiteLoc is not valid)");
      }
      return mlir::CallSiteLoc::get(
          debug_metadata_locations[attribute_location_idx_map.at(
              callsite_loc->callee_index())],
          debug_metadata_locations[attribute_location_idx_map.at(
              callsite_loc->caller_index())]);
    }
    // NameLoc.
    case debug_metadata::LocationType_NameLoc: {
      auto name_loc =
          static_cast<const debug_metadata::NameLoc*>(location.location());
      if (!attribute_location_idx_map.contains(name_loc->child_index())) {
        return absl::InternalError(
            "Invalid/corrupt DebugMetadata, expected invariant broken (child "
            "index of a NameLoc is not valid)");
      }
      return mlir::NameLoc::get(
          builder.getStringAttr(name_loc->name()->string_view()),
          debug_metadata_locations[attribute_location_idx_map.at(
              name_loc->child_index())]);
    }
    // FusedLoc.
    case debug_metadata::LocationType_FusedLoc: {
      auto fused_loc =
          static_cast<const debug_metadata::FusedLoc*>(location.location());
      auto fused_location_indexes = fused_loc->location_indexes();
      std::vector<mlir::Location> fused_locations;
      fused_locations.reserve(fused_location_indexes->size());
      for (int fused_loc_idx = 0;
           fused_loc_idx < fused_location_indexes->size(); ++fused_loc_idx) {
        if (!attribute_location_idx_map.contains(
                fused_location_indexes->Get(fused_loc_idx))) {
          return absl::InternalError(
              "Invalid/corrupt DebugMetadata, expected invariant broken "
              "(location index of a FusedLoc is not valid)");
        }
        fused_locations.push_back(
            debug_metadata_locations[attribute_location_idx_map.at(
                fused_location_indexes->Get(fused_loc_idx))]);
      }
      return mlir::FusedLoc::get(
          fused_locations, mlir::StringAttr::get(builder.getContext(), ""),
          builder.getContext());
    }
    default: {
      return mlir::UnknownLoc::get(builder.getContext());
    }
  }
}

// Parses all locations in ConversionDebugMetadata, build the mlir::location
// counterparts, and put them inside debug_metadata_. Additionally, maintain a
// map that maps the top location index of each operator.
Status ParseAndBuildLocation(
    Builder builder,
    const debug_metadata::ConversionDebugMetadata* conversion_debug_metadata,
    DebugMetadata& debug_metadata_var) {
  auto attribute_types = conversion_debug_metadata->attributes_type();
  auto attributes = conversion_debug_metadata->attributes();

  auto& debug_metadata_locations = debug_metadata_var.debug_metadata_locations;
  debug_metadata_locations.reserve(attribute_types->size());

  // Map index in the attribute_vector to the index in the data structure we
  // are building: DebugMetadata::debug_metadata_locations.
  absl::flat_hash_map<unsigned int, unsigned int> attribute_location_idx_map;

  for (int i = 0; i < attribute_types->size(); ++i) {
    if (attribute_types->Get(i) == debug_metadata::Attribute_Location) {
      auto location =
          static_cast<const debug_metadata::Location*>(attributes->Get(i));
      TF_ASSIGN_OR_RETURN(
          auto mlir_location,
          BuildLocation(builder, *location, debug_metadata_locations,
                        attribute_location_idx_map));
      debug_metadata_locations.push_back(mlir_location);

      // Create index mapping.
      attribute_location_idx_map[i] = debug_metadata_locations.size() - 1;
    }
  }

  // Collect the top location idx of each operator.
  auto subgraphs_debug_metadata =
      conversion_debug_metadata->subgraphs_debug_metadata();
  for (int subgraph_idx = 0; subgraph_idx < subgraphs_debug_metadata->size();
       ++subgraph_idx) {
    const auto* subgraph_debug_metadata =
        subgraphs_debug_metadata->Get(subgraph_idx);
    auto operators_debug_metadata =
        subgraph_debug_metadata->operators_debug_metadata();
    for (int operator_idx = 0; operator_idx < operators_debug_metadata->size();
         ++operator_idx) {
      const auto* operator_debug_metadata =
          operators_debug_metadata->Get(operator_idx);
      // Find the location attribute of the operator. Note that there should
      // be at most one idx pointing to location attribute for each operator.
      std::vector<unsigned int> location_attribute_idxs;
      for (int i = 0;
           i < operator_debug_metadata->attribute_metadata_indexes()->size();
           ++i) {
        auto attribute_idx =
            operator_debug_metadata->attribute_metadata_indexes()->Get(i);
        if (attribute_types->Get(attribute_idx) ==
            debug_metadata::Attribute_Location) {
          location_attribute_idxs.push_back(attribute_idx);
        }
      }
      if (location_attribute_idxs.size() > 1) {
        return absl::InternalError(
            "Invalid/corrupt DebugMetadata, expected invariant broken (more "
            "than one location attribute for an operator)");
      }
      if (location_attribute_idxs.empty()) {
        continue;
      }

      if (!attribute_location_idx_map.contains(location_attribute_idxs[0])) {
        return absl::InternalError(
            "Invalid/corrupt DebugMetadata, expected invariant broken "
            "(location attribute index of an operator is not valid)");
      }
      debug_metadata_var.operator_location_map[subgraph_idx][operator_idx] =
          attribute_location_idx_map[location_attribute_idxs[0]];
    }
  }

  return absl::OkStatus();
}

// Parse the DebugMetadata flatbuffer and store debug metadata in struct
// `debug_metadata`.
Status ParseDebugMetadata(Builder builder, const char* data, size_t size,
                          DebugMetadata& debug_metadata_var) {
  auto debug_metadata_fb = debug_metadata::GetDebugMetadata(data);

  if (debug_metadata_fb->debug_metadata_type()->size() !=
      debug_metadata_fb->debug_metadata()->size()) {
    return absl::InternalError(
        "Invalid/corrupt DebugMetadata, expected invariant broken (size of "
        "debug_metadata_type and debug_metadata not equal)");
  }

  for (int i = 0; i < debug_metadata_fb->debug_metadata_type()->size(); ++i) {
    if (debug_metadata_fb->debug_metadata_type()->Get(i) ==
        debug_metadata::DebugMetadataType_ConversionDebugMetadata) {
      auto conversion_debug_metadata =
          static_cast<const debug_metadata::ConversionDebugMetadata*>(
              debug_metadata_fb->debug_metadata()->Get(i));
      TF_RETURN_IF_ERROR(ParseAndBuildLocation(
          builder, conversion_debug_metadata, debug_metadata_var));
    } else {
      LOG(WARNING) << "Unsupported DebugMetadataType: "
                   << debug_metadata_fb->debug_metadata_type()->Get(i);
    }
  }

  return absl::OkStatus();
}

// Return MLIR location if it exists in the debug metadata. Otherwise, create a
// MLIR location by fusing its output tensor names.
Location OpLoc(const OperatorT& op, Builder builder,
               DebugMetadata& debug_metadata, const tflite::SubGraphT& subgraph,
               Location base) {
  const int subgraph_debug_metadata_idx = subgraph.debug_metadata_index;
  if (debug_metadata.operator_location_map.contains(
          subgraph_debug_metadata_idx) &&
      debug_metadata.operator_location_map[subgraph_debug_metadata_idx]
          .contains(op.debug_metadata_index)) {
    int location_idx =
        debug_metadata.operator_location_map[subgraph_debug_metadata_idx]
                                            [op.debug_metadata_index];
    return debug_metadata.debug_metadata_locations[location_idx];
  }

  if (op.outputs.empty()) return base;

  llvm::SmallVector<Location, 4> locations;
  locations.reserve(op.outputs.size());
  for (auto tensor_index : op.outputs) {
    locations.push_back(
        TensorLoc(*subgraph.tensors[tensor_index], builder, base));
  }
  return mlir::FusedLoc::get(builder.getContext(), locations);
}

// Extract the min max information in the tensor and create the quant stats op.
// If the input `tensor` has scale/zero_point, `res` should have quantized type,
// thus none stats op is required and nullptr is returned. If the min max
// information is invalid, nullptr is returned.
mlir::Operation* ConvertMinMaxToStatsOp(const TensorT& tensor, OpBuilder b,
                                        Value res) {
  // If the `tensor` has scale/zero_point, it must have been quantized, then the
  // min/max stats is just for comments, so ignore it.
  if (!tensor.quantization || tfl::IsQuantized(tensor)) return nullptr;
  // If the result isn't float and unquantizable, the min/max is ignored.
  if (!res.getType()
           .cast<mlir::ShapedType>()
           .getElementType()
           .isa<mlir::FloatType>()) {
    return nullptr;
  }
  auto mins = tensor.quantization->min;
  auto maxs = tensor.quantization->max;
  if (mins.size() != maxs.size() || mins.empty()) return nullptr;

  llvm::SmallVector<llvm::APFloat, 4> min_maxs;
  min_maxs.reserve(mins.size() * 2);
  for (int i = 0, end = mins.size(); i < end; ++i) {
    llvm::APFloat min(mins[i]);
    llvm::APFloat max(maxs[i]);
    min_maxs.push_back(min);
    min_maxs.push_back(max);
  }
  // The layer stats contain only the first min/max pairs.
  mlir::ElementsAttr layer_stats = mlir::DenseFPElementsAttr::get(
      tensorflow::GetTypeFromTFTensorShape({2}, b.getF32Type()),
      {min_maxs[0], min_maxs[1]});
  mlir::ElementsAttr axis_stats;
  mlir::IntegerAttr axis;
  if (mins.size() > 1) {
    llvm::SmallVector<int64_t, 4> axis_stats_shape{
        static_cast<int64_t>(mins.size()), 2};
    axis_stats = mlir::DenseFPElementsAttr::get(
        tensorflow::GetTypeFromTFTensorShape(axis_stats_shape, b.getF32Type()),
        min_maxs);
    // TODO(fengliuai): this quantization dimension isn't correct.
    axis = b.getI64IntegerAttr(tensor.quantization->quantized_dimension);
  }
  return b.create<mlir::quantfork::StatisticsOp>(b.getUnknownLoc(), res,
                                                 layer_stats, axis_stats, axis);
}

// Returns true if this is a basic LSTM op.
bool IsBasicLSTMOp(tflite::BuiltinOptionsUnion op_union) {
  if (const auto* op = op_union.AsLSTMOptions()) {
    return op->kernel_type == tflite::LSTMKernelType_BASIC;
  } else {
    return false;
  }
}

// Gets the MLIR op name with the dialect name for the flatbuffer operator.
std::string GetMlirOpName(const tflite::OperatorT& op,
                          const tflite::OperatorCodeT& op_code) {
  if (IsBasicLSTMOp(op.builtin_options)) {
    return std::string("tfl.basic_lstm");
  }
  return mlir::GetMlirOpNameFromOpCode(op_code);
}

StatusOr<Operation*> BuildExternalConstOp(const tflite::TensorT& tensor,
                                          int32_t buffer_index,
                                          OpBuilder builder, Location loc) {
  TF_ASSIGN_OR_RETURN(mlir::TensorType type,
                      tfl::GetTensorType(tensor, builder,
                                         /*is_constant=*/true));
  auto shaped_type = type.dyn_cast<mlir::RankedTensorType>();
  if (!shaped_type) {
    return errors::Internal("Constant doesn't have a shape");
  }
  auto op = builder.create<tfl::ExternalConstOp>(
      loc, shaped_type, builder.getI32IntegerAttr(buffer_index));
  return op.getOperation();
}

// TODO(b/172664358): Creates a new op instead of reusing constant op.
// Creates a constant op with "tfl.is_variable" attribute to represent stateful
// variable. The function static variable `stateful_variable_idx` is used as a
// unique value for each constant to avoid CSEed. `tensor` is the data structure
// of flatbuffer. `shaped_type` is the ShapedType for the const op.
StatusOr<Operation*> BuildVariableOp(const tflite::TensorT& tensor,
                                     OpBuilder builder, Location loc) {
  TF_ASSIGN_OR_RETURN(mlir::TensorType type,
                      tfl::GetTensorType(tensor, builder,
                                         /*is_constant=*/true));
  auto shaped_type = type.dyn_cast<mlir::RankedTensorType>();
  if (!shaped_type) {
    return errors::Internal("Constant doesn't have a shape");
  }

  static int stateful_variable_idx = 0;
  mlir::ElementsAttr value =
      tfl::GetSplat(shaped_type, stateful_variable_idx++, builder);
  if (tfl::IsQuantized(tensor)) {
    auto op = builder.create<tfl::QConstOp>(
        loc, mlir::TypeAttr::get(shaped_type), value);
    return op.getOperation();
  }
  auto op = builder.create<tfl::ConstOp>(loc, value);
  op->setAttr("tfl.is_variable", builder.getUnitAttr());
  if (tensor.quantization && !tensor.quantization->min.empty()) {
    if (auto stats_op =
            ConvertMinMaxToStatsOp(tensor, builder, op.getResult())) {
      return stats_op;
    }
  }
  return op.getOperation();
}

static StatusOr<std::vector<int32_t>> ConvertSparseIndexVector(
    const tflite::SparseIndexVectorUnion& sparse_index_vector) {
  if (sparse_index_vector.type == tflite::SparseIndexVector_Int32Vector) {
    return sparse_index_vector.AsInt32Vector()->values;
  } else if (sparse_index_vector.type ==
             tflite::SparseIndexVector_Uint16Vector) {
    const auto& inputs = sparse_index_vector.AsUint16Vector()->values;
    std::vector<int32_t> outputs(inputs.size());
    std::transform(inputs.begin(), inputs.end(), outputs.begin(),
                   [](auto x) { return static_cast<int32_t>(x); });
    return outputs;
  } else if (sparse_index_vector.type ==
             tflite::SparseIndexVector_Uint8Vector) {
    const auto& inputs = sparse_index_vector.AsUint8Vector()->values;
    std::vector<int32_t> outputs(inputs.size());
    std::transform(inputs.begin(), inputs.end(), outputs.begin(),
                   [](auto x) { return static_cast<int32_t>(x); });
    return outputs;
  } else {
    return errors::Unimplemented("Unsupported SparseIndexVector type");
  }
}

static StatusOr<Operation*> BuildSparseConstOp(
    const tflite::TensorT& tensor, const std::vector<uint8_t>& buffer,
    OpBuilder& builder, Location loc) {
  TF_ASSIGN_OR_RETURN(mlir::TensorType type,
                      tfl::GetTensorType(tensor, builder,
                                         /*is_constant=*/true));
  auto shaped_type = type.dyn_cast<mlir::RankedTensorType>();
  if (!shaped_type) {
    return errors::Internal("Constant doesn't have a shape");
  }

  TF_ASSIGN_OR_RETURN(type, tfl::GetTensorType(tensor, builder,
                                               /*is_constant=*/true,
                                               /*is_intermediate=*/false,
                                               /*get_storage=*/true));
  auto value_type = mlir::dyn_cast<mlir::RankedTensorType>(type);

  tensorflow::TensorProto repr = tfl::ConvertTfliteConstTensor(tensor, buffer);
  repr.clear_tensor_shape();
  if (tfl::IsQuantized(tensor)) {
    repr.mutable_tensor_shape()->add_dim()->set_size(buffer.size());
    repr.set_dtype(tensorflow::DT_INT8);
  } else {
    repr.mutable_tensor_shape()->add_dim()->set_size(
        buffer.size() / (shaped_type.getElementTypeBitWidth() / CHAR_BIT));
  }
  TF_ASSIGN_OR_RETURN(mlir::ElementsAttr compressed_data,
                      tensorflow::ConvertTensorProto(repr, &builder));

  const int dim_metadata_size = tensor.sparsity->dim_metadata.size();
  std::vector<mlir::TFL::DimensionMetadataAttr> dim_metadata(dim_metadata_size);
  for (int i = 0; i < dim_metadata_size; i++) {
    if (tensor.sparsity->dim_metadata[i]->format ==
        tflite::DimensionType_DENSE) {
      dim_metadata[i] = tfl::DimensionMetadataAttr::get(
          builder.getContext(),
          mlir::TFL::DimensionTypeAttr::get(builder.getContext(),
                                            tfl::DimensionType::DENSE),
          tensor.sparsity->dim_metadata[i]->dense_size, {}, {});
    } else if (tensor.sparsity->dim_metadata[i]->format ==
               tflite::DimensionType_SPARSE_CSR) {
      TF_ASSIGN_OR_RETURN(
          auto segments, ConvertSparseIndexVector(
                             tensor.sparsity->dim_metadata[i]->array_segments));
      TF_ASSIGN_OR_RETURN(auto indices,
                          ConvertSparseIndexVector(
                              tensor.sparsity->dim_metadata[i]->array_indices));
      dim_metadata[i] = tfl::DimensionMetadataAttr::get(
          builder.getContext(),
          mlir::TFL::DimensionTypeAttr::get(builder.getContext(),
                                            tfl::DimensionType::SPARSE_CSR),
          0, segments, indices);
    } else {
      return errors::Unimplemented("Unsupported dimension metadata type");
    }
  }
  auto s_param = tfl::SparsityParameterAttr::get(
      builder.getContext(), tensor.sparsity->traversal_order,
      tensor.sparsity->block_map, dim_metadata);

  std::vector<char> dense_buffer(
      value_type.getElementType().getIntOrFloatBitWidth() / CHAR_BIT);
  mlir::TypedAttr dummy_value =
      mlir::DenseIntOrFPElementsAttr::getFromRawBuffer(value_type,
                                                       dense_buffer);

  if (tfl::IsQuantized(tensor)) {
    return builder
        .create<tfl::SparseQConstOp>(loc, mlir::TypeAttr::get(shaped_type),
                                     dummy_value, s_param, compressed_data)
        .getOperation();
  }
  return builder
      .create<tfl::SparseConstOp>(loc, dummy_value, s_param, compressed_data)
      .getOperation();
}

StatusOr<Operation*> BuildConstOp(const tflite::TensorT& tensor,
                                  const std::vector<uint8_t>& buffer,
                                  bool is_variable, OpBuilder builder,
                                  Location loc, bool use_stablehlo_constant) {
  if (tensor.sparsity != nullptr) {
    return BuildSparseConstOp(tensor, buffer, builder, loc);
  }

  if (is_variable) {
    return BuildVariableOp(tensor, builder, loc);
  }

  TF_ASSIGN_OR_RETURN(mlir::TensorType type,
                      tfl::GetTensorType(tensor, builder,
                                         /*is_constant=*/true,
                                         /*is_intermediate=*/false,
                                         /*get_storage=*/true));
  auto shaped_type = type.dyn_cast<mlir::RankedTensorType>();
  if (!shaped_type) {
    return errors::Internal("Constant doesn't have a shape");
  }

  mlir::ElementsAttr value;
  if (tfl::IsQuantized(tensor)) {
    bool truncate = shaped_type.getElementType().getIntOrFloatBitWidth() == 64;
    TF_ASSIGN_OR_RETURN(value,
                        tfl::ConvertIntBuffer(shaped_type, buffer, truncate));
    TF_ASSIGN_OR_RETURN(
        mlir::quant::QuantizedType type,
        tfl::GetQuantizedType(tensor, builder, /*is_constant=*/true,
                              /*storage_type=*/value.getElementType()));
    shaped_type = shaped_type.clone(type);
    auto op = builder.create<tfl::QConstOp>(
        loc, mlir::TypeAttr::get(shaped_type), value);
    return op.getOperation();
  }

  auto elem_type = shaped_type.getElementType();
  if (auto float_type = elem_type.dyn_cast<mlir::FloatType>()) {
    TF_ASSIGN_OR_RETURN(value, tfl::ConvertFloatBuffer(shaped_type, buffer));
  } else if (elem_type.isa<mlir::IntegerType>()) {
    TF_ASSIGN_OR_RETURN(value, tfl::ConvertIntBuffer(shaped_type, buffer));
  } else if (elem_type.isa<mlir::TF::StringType>()) {
    tensorflow::TensorProto repr =
        tfl::ConvertTfliteConstTensor(tensor, buffer);
    std::vector<llvm::StringRef> refs;
    refs.reserve(repr.string_val_size());

    for (const auto& ref : repr.string_val())
      refs.push_back({ref.data(), ref.size()});

    value = mlir::DenseStringElementsAttr::get(shaped_type, refs);
  } else if (elem_type.isa<mlir::ComplexType, mlir::TF::TensorFlowType>()) {
    tensorflow::TensorProto repr =
        tfl::ConvertTfliteConstTensor(tensor, buffer);
    std::string mangled = tensorflow::mangling_util::MangleTensor(repr);

    value = mlir::TF::TensorProtoAttr::get(shaped_type, mangled);
  } else {
    return errors::Unimplemented("Constant of unsupported type");
  }

  if (use_stablehlo_constant) {
    mlir::StablehloVhloTypeConverter vhlo_type_converter;
    llvm::ArrayRef<char> val_ref(reinterpret_cast<const char*>(buffer.data()),
                                 buffer.size());
    auto vhlo_val = mlir::vhlo::TensorV1Attr::get(
        builder.getContext(), vhlo_type_converter.convertType(shaped_type),
        val_ref);
    auto op =
        builder.create<mlir::vhlo::ConstantOpV1>(loc, shaped_type, vhlo_val);
    return op.getOperation();
  }
  auto op = builder.create<tfl::ConstOp>(loc, value);
  return op.getOperation();
}

StatusOr<llvm::SmallVector<mlir::NamedAttribute, 4>>
ConvertSubgraphIdxsToFunctionAttrs(tflite::BuiltinOptionsUnion options,
                                   const std::vector<std::string>& func_names,
                                   Builder builder) {
  if (auto* opts = options.AsCallOnceOptions()) {
    uint32_t init_idx = opts->init_subgraph_index;
    if (init_idx >= func_names.size()) {
      return errors::InvalidArgument("subgraph with index not found: ",
                                     init_idx);
    }
    auto init_attr = builder.getStringAttr(func_names.at(init_idx));

    return llvm::SmallVector<mlir::NamedAttribute, 4>{
        builder.getNamedAttr("session_init_function", init_attr)};
  }
  if (auto* opts = options.AsIfOptions()) {
    uint32_t then_idx = opts->then_subgraph_index;
    if (then_idx >= func_names.size()) {
      return errors::InvalidArgument("subgraph with index not found: ",
                                     then_idx);
    }
    auto then_attr =
        mlir::SymbolRefAttr::get(builder.getContext(), func_names.at(then_idx));
    uint32_t else_idx = opts->else_subgraph_index;
    if (else_idx >= func_names.size()) {
      return errors::InvalidArgument("subgraph with index not found: ",
                                     else_idx);
    }
    auto else_attr =
        mlir::SymbolRefAttr::get(builder.getContext(), func_names.at(else_idx));

    return llvm::SmallVector<mlir::NamedAttribute, 4>{
        builder.getNamedAttr("then_branch", then_attr),
        builder.getNamedAttr("else_branch", else_attr),
        // TODO(b/139667752): Analyze statelessness correctly
        builder.getNamedAttr("is_stateless", builder.getBoolAttr(false))};
  }
  if (auto* opts = options.AsWhileOptions()) {
    uint32_t cond_idx = opts->cond_subgraph_index;
    if (cond_idx >= func_names.size()) {
      return errors::InvalidArgument("subgraph with index not found: ",
                                     cond_idx);
    }
    auto cond_attr =
        mlir::SymbolRefAttr::get(builder.getContext(), func_names.at(cond_idx));
    uint32_t body_idx = opts->body_subgraph_index;
    if (body_idx >= func_names.size()) {
      return errors::InvalidArgument("subgraph with index not found: ",
                                     body_idx);
    }
    auto body_attr =
        mlir::SymbolRefAttr::get(builder.getContext(), func_names.at(body_idx));

    return llvm::SmallVector<mlir::NamedAttribute, 4>{
        builder.getNamedAttr("cond", cond_attr),
        builder.getNamedAttr("body", body_attr)};
  }
  return llvm::SmallVector<mlir::NamedAttribute, 4>{};
}

Status ConvertSubgraphIdxToStablehloRegion(
    const tflite::OperatorT& op, const std::vector<std::string>& func_names,
    Builder builder, OperationState& op_state) {
  if (auto* opts = op.builtin_options_2.AsStablehloReduceOptions()) {
    int32_t body_idx = opts->body_subgraph_index;
    if (body_idx >= func_names.size()) {
      return absl::AbortedError("subgraph with index not found: " +
                                std::to_string(body_idx));
    }
    auto body_attr =
        mlir::SymbolRefAttr::get(builder.getContext(), func_names.at(body_idx));

    op_state.addAttribute("body", body_attr);

    return absl::OkStatus();
  }
  if (auto* opts = op.builtin_options_2.AsStablehloReduceWindowOptions()) {
    int32_t body_idx = opts->body_subgraph_index;
    if (body_idx >= func_names.size()) {
      return absl::AbortedError("subgraph with index not found: " +
                                std::to_string(body_idx));
    }
    auto body_attr =
        mlir::SymbolRefAttr::get(builder.getContext(), func_names.at(body_idx));

    op_state.addAttribute("body", body_attr);

    return absl::OkStatus();
  }
  if (auto* opts = op.builtin_options_2.AsStablehloSortOptions()) {
    int32_t comparator_idx = opts->comparator_subgraph_index;
    if (comparator_idx >= func_names.size()) {
      return absl::AbortedError("subgraph with index not found: " +
                                std::to_string(comparator_idx));
    }
    auto comparator_attr = mlir::SymbolRefAttr::get(
        builder.getContext(), func_names.at(comparator_idx));

    op_state.addAttribute("comparator", comparator_attr);

    return absl::OkStatus();
  }
  if (auto* opts = op.builtin_options_2.AsStablehloWhileOptions()) {
    int32_t body_idx = opts->body_subgraph_index;
    int32_t cond_idx = opts->cond_subgraph_index;
    if (body_idx >= func_names.size()) {
      return absl::AbortedError("subgraph with index not found: " +
                                std::to_string(body_idx));
    }
    if (cond_idx >= func_names.size()) {
      return absl::AbortedError("subgraph with index not found: " +
                                std::to_string(cond_idx));
    }
    auto body_attr =
        mlir::SymbolRefAttr::get(builder.getContext(), func_names.at(body_idx));
    auto cond_attr =
        mlir::SymbolRefAttr::get(builder.getContext(), func_names.at(cond_idx));

    op_state.addAttribute("body", body_attr);
    op_state.addAttribute("cond", cond_attr);

    return absl::OkStatus();
  }
  if (auto* opts = op.builtin_options_2.AsStablehloScatterOptions()) {
    uint32_t subgraph_idx = opts->update_computation_subgraph_index;

    if (subgraph_idx >= func_names.size()) {
      return absl::AbortedError(
          absl::StrCat("subgraph with index not found: ", subgraph_idx));
    }
    mlir::FlatSymbolRefAttr subgraph_attr = mlir::SymbolRefAttr::get(
        builder.getContext(), func_names.at(subgraph_idx));

    op_state.addAttribute(kScatterRegionFuncName, subgraph_attr);

    return absl::OkStatus();
  }
  // skip if not supported
  return absl::OkStatus();
}

Status AddOpIntermediatesForLstm(
    const tflite::OperatorT& op,
    const std::vector<mlir::TensorType>& intermediate_types,
    OperationState& op_state, Location loc, OpBuilder& builder) {
  if (!op.intermediates.empty()) {
    if (op.intermediates.size() != 5) {
      auto err = errors::InvalidArgument(
          "operator has intermediate tensors but the number of them is not "
          "five.");
      return emitError(loc, err.ToString()), err;
    }
    // Create intermediate value

    const llvm::SmallVector<llvm::StringRef, 5> kIntermediateNames = {
        "input_to_input_intermediate", "input_to_forget_intermediate",
        "input_to_cell_intermediate", "input_to_output_intermediate",
        "effective_hidden_scale_intermediate"};
    for (auto type_and_name :
         llvm::zip(intermediate_types, kIntermediateNames)) {
      mlir::TypeAttr type_attr =
          mlir::TypeAttr::get(std::get<0>(type_and_name));
      auto named_attr =
          builder.getNamedAttr(std::get<1>(type_and_name), type_attr);
      op_state.addAttribute(named_attr.getName(), named_attr.getValue());
    }
  }
  return absl::OkStatus();
}

// TODO(krzysd) Handle function calls
StatusOr<Operation*> ConvertOp(
    const tflite::OperatorT& op, const std::vector<Value>& vals_map,
    const std::vector<mlir::TensorType>& intermediate_types,
    Value optional_arg_marker,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& op_codes,
    const std::vector<std::string>& func_names,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tensors, Location loc,
    OpBuilder builder,
    const std::unique_ptr<tfl::FlatBufferModelAbslError>& model_ptr) {
  llvm::SmallVector<Value, 4> operands;
  llvm::SmallVector<mlir::Type, 2> outputTypes;

  const tflite::OperatorCodeT& op_code = *op_codes.at(op.opcode_index);

  const std::string op_name = GetMlirOpName(op, op_code);

  OperationState op_state(loc, op_name);

  for (auto input_num : op.inputs) {
    if (input_num == -1) {
      assert(optional_arg_marker != nullptr);
      op_state.addOperands({optional_arg_marker});
    } else {
      op_state.addOperands({vals_map.at(input_num)});
    }
  }

  for (auto output_num : op.outputs) {
    auto& tensor = *tensors.at(output_num);
    auto type_or_err = tfl::GetTensorType(tensor, builder);
    if (!type_or_err.ok()) {
      return emitError(loc, type_or_err.status().ToString()),
             type_or_err.status();
    }
    auto type = std::move(type_or_err).value();

    if (op_name == "tfl.quantize") {
      // Special case for quantize: return type must also be in qtype attribute
      op_state.addAttribute("qtype", mlir::TypeAttr::get(type));
    } else if (op_name == "tfl.reshape" && op_state.operands.size() == 1) {
      // Special case for reshape: the second op is optional in the old
      // converter and kernel, so we create the second operand, which is
      // required by the new converter, from the reshape op's option.
      auto new_shape = op.builtin_options.AsReshapeOptions()->new_shape;
      auto shape_type = tensorflow::GetTypeFromTFTensorShape(
          {static_cast<int64_t>(new_shape.size())}, builder.getIntegerType(32));

      mlir::SmallVector<mlir::Attribute, 4> shape;
      for (auto s : new_shape) {
        shape.push_back(
            builder.getI32IntegerAttr(mlir::TFL::ConvertToTfliteSize(s)));
      }
      auto output_shape = DenseElementsAttr::get(shape_type, shape);
      auto shape_op = builder.create<tfl::ConstOp>(loc, output_shape);
      op_state.addOperands({shape_op});
    }

    op_state.addTypes({type});
  }

  // While the last several tensors could be optional tensors for an tfl op, the
  // number of input operands could vary. Gets the min/max number of operands
  // from tflite op name.
  // Also, since the above code special-handles the `tfl.reshape` op and add an
  // additional input, we put these function block here.
  llvm::MinMax input_min_max = mlir::OperandNumbersMinMax(op_name);
  int input_max_num = input_min_max.Max;
  int op_input_num = op_state.operands.size();
  if (input_max_num != 0 && input_max_num > op_input_num) {
    // If the number of current inputs is less than the op definition, fill in
    // with `none` value,
    llvm::SmallVector<Value, 4> none_operands(
        input_max_num - op_input_num,
        builder.create<mlir::TFL::NoValueOp>(loc, builder.getNoneType(),
                                             builder.getUnitAttr()));
    op_state.addOperands(ArrayRef<Value>(none_operands));
  }

  if (op_name == "tfl.lstm") {
    // TODO(b/147587779): add the right region if region is empty.
    op_state.addRegion();
    TF_CHECK_OK(AddOpIntermediatesForLstm(op, intermediate_types, op_state, loc,
                                          builder));
  }
  if (op_name == "tfl.while") {
    // Adds two empty regions for "tfl.while". We will fill the regions after
    // creating the callee functions because the "tfl.while" input/output types
    // may be different with the callee functions, and the call ops need to sync
    // with callee function types.
    op_state.addRegion();
    op_state.addRegion();
  }
  if (op_name == "tfl.unidirectional_sequence_lstm") {
    TF_CHECK_OK(AddOpIntermediatesForLstm(op, intermediate_types, op_state, loc,
                                          builder));
  }
  if (op_name == "tfl.reshape") {
    // Flattens reshape ops when more than one dimension shape operand is given.
    mlir::DenseIntElementsAttr shape_attr;
    if (matchPattern(op_state.operands[1], m_Constant(&shape_attr))) {
      auto shape_ty =
          op_state.operands[1].getType().dyn_cast<RankedTensorType>();
      if (shape_ty != nullptr && shape_ty.hasRank() && shape_ty.getRank() > 1) {
        llvm::SmallVector<mlir::Attribute, 4> shape;
        int32_t dim_size = 0;
        for (const auto& dim :
             llvm::enumerate(shape_attr.getValues<llvm::APInt>())) {
          shape.push_back(builder.getI32IntegerAttr(
              mlir::TFL::ConvertToTfliteSize(dim.value().getSExtValue())));
          ++dim_size;
        }
        auto shape_type = tensorflow::GetTypeFromTFTensorShape(
            {static_cast<int32_t>(dim_size)}, builder.getIntegerType(32));
        auto output_shape = mlir::DenseElementsAttr::get(shape_type, shape);
        auto shape_op = builder.create<tfl::ConstOp>(loc, output_shape);
        op_state.operands[1] = shape_op;
      }
    }
  }
  if (op_name == "vhlo.reduce_v1" || op_name == "vhlo.reduce_window_v1" ||
      op_name == "vhlo.sort_v1" || op_name == "vhlo.scatter_v1") {
    op_state.addRegion();
  }
  if (op_name == "vhlo.while_v1") {
    op_state.addRegion();
    op_state.addRegion();
  }

  llvm::SmallVector<mlir::NamedAttribute, 2> attrs;
  auto builtin_code = tflite::GetBuiltinCode(&op_code);
  if (builtin_code == tflite::BuiltinOperator_CUSTOM) {
    auto status = absl::OkStatus();

    std::vector<uint8_t> custom_options;

    if (IsValidBufferOffset(op.large_custom_options_offset)) {
      custom_options.resize(op.large_custom_options_size);
      memcpy(custom_options.data(),
             reinterpret_cast<const uint8_t*>(model_ptr->allocation()->base()) +
                 op.large_custom_options_offset,
             op.large_custom_options_size);
    } else {
      custom_options = op.custom_options;
    }

    status = mlir::CustomOptionsToAttributes(
        op_code.custom_code, custom_options, builder, loc, &attrs);
    if (!status.ok()) {
      return emitError(loc, status.ToString()), status;
    }
  } else {
    mlir::BuiltinOptionsToAttributes(op.builtin_options, builder, attrs);
    mlir::BuiltinOptions2ToAttributes(op.builtin_options_2, builder, attrs);
  }

  if (builtin_code == tflite::BuiltinOperator_STABLEHLO_COMPOSITE) {
    auto composite_options = op.builtin_options_2.AsStableHLOCompositeOptions();
    std::string decomposition = "";
    if (composite_options->decomposition_subgraph_index > -1) {
      decomposition =
          func_names.at(composite_options->decomposition_subgraph_index);
    }

    attrs.emplace_back(builder.getNamedAttr(
        "decomposition",
        mlir::vhlo::StringV1Attr::get(builder.getContext(), decomposition)));
  }

  op_state.addAttributes(attrs);

  // Handle the conversion from subgraph index to functions for If and While. We
  // will add CallOps in the region to call the functions later for While.
  TF_ASSIGN_OR_RETURN(auto function_ref_attrs,
                      ConvertSubgraphIdxsToFunctionAttrs(op.builtin_options,
                                                         func_names, builder));
  op_state.addAttributes(function_ref_attrs);
  // Handle conversion from subgraph to regions in StableHLO ops.
  auto status =
      ConvertSubgraphIdxToStablehloRegion(op, func_names, builder, op_state);
  if (!status.ok()) {
    return emitError(loc, status.ToString()), status;
  }

  return builder.create(op_state);
}

// Returns indices of the given tensors in the subgraph. Returns error if a
// tensor name cannot be found in the subgraph.
StatusOr<std::vector<int>> GetTensorIndices(
    const tflite::SubGraphT& subgraph,
    const std::vector<std::string>& tensor_names) {
  absl::flat_hash_map<std::string, int> name_to_index;
  for (const auto& index_and_tensor : llvm::enumerate(subgraph.tensors)) {
    name_to_index[index_and_tensor.value()->name] = index_and_tensor.index();
  }

  std::vector<int> indices;
  indices.reserve(tensor_names.size());

  for (const auto& name : tensor_names) {
    auto found = name_to_index.find(name);
    if (found != name_to_index.end()) {
      indices.push_back(found->second);
    } else {
      return errors::InvalidArgument("could not find tensor in subgraph: ",
                                     name);
    }
  }

  return indices;
}

// Given a list of tensor indices, returns true if any of the tensors have
// non-empty name strings.
bool HasNonEmptyNames(const tflite::SubGraphT& subgraph,
                      ArrayRef<int32_t> indices) {
  return llvm::any_of(
      indices, [&](int i) { return !subgraph.tensors.at(i)->name.empty(); });
}

// Given a list of tensor indices, returns a string of concatenated tensor names
// wrapped in a NamedAttribute.
mlir::NamedAttribute BuildTFEntryFunctionAttribute(
    const tflite::SubGraphT& subgraph, Builder* builder,
    const std::string& name, ArrayRef<int32_t> indices) {
  auto tensor_names = llvm::map_range(
      indices, [&](int i) { return subgraph.tensors.at(i)->name; });
  return builder->getNamedAttr(
      name, builder->getStringAttr(llvm::join(tensor_names, ",")));
}

// Traverses the subgraph from output_indices to input_indices and returns the
// set of ops that are visited.
StatusOr<absl::flat_hash_set<const tflite::OperatorT*>> PruneSubgraph(
    const tflite::SubGraphT& subgraph, ArrayRef<int32_t> input_indices,
    ArrayRef<int32_t> output_indices) {
  // Create a map from tensor index to defining op.
  absl::flat_hash_map<int32_t, const tflite::OperatorT*> defining_op;
  for (const auto& op : subgraph.operators) {
    for (int32_t output : op->outputs) {
      if (!llvm::is_contained(input_indices, output)) {
        defining_op[output] = op.get();
      }
    }
  }

  std::vector<const tflite::OperatorT*> queue;
  for (int32_t output : output_indices) {
    if (auto& op = defining_op[output]) {
      queue.push_back(op);
    }
  }

  // Traverse the graph towards inputs.
  absl::flat_hash_set<const tflite::OperatorT*> visited;
  while (!queue.empty()) {
    const tflite::OperatorT* op = queue.back();
    queue.pop_back();
    if (!visited.insert(op).second) {
      // The node has already been visited.
      continue;
    }

    for (int32_t input : op->inputs) {
      // Input tensor may not have a defining op in case it is a subgraph input
      // or a constant tensor.
      if (auto& op = defining_op[input]) {
        queue.push_back(op);
      }
    }
  }

  return visited;
}

// We want to adjust the func op according to some cross ops information.
static StatusOr<FuncOp> PostProcessFuncOp(FuncOp func) {
  OpBuilder builder(func);
  // When a quantized constant is imported, its quantization parameter is set
  // to be narrow range. Here revert to be the fully range if the user doesn't
  // require narrow range.
  func.walk([&](tfl::QConstOp cst) {
    Value value = cst.getResult();
    Value full_range_const = value;
    auto qtype = mlir::quant::UniformQuantizedType::getQuantizedElementType(
        value.getType());
    // Only the 8-bit constants are imported with narrow range.
    if (!qtype || qtype.getStorageTypeIntegralWidth() != 8 ||
        !(qtype.isa<mlir::quant::UniformQuantizedType>() ||
          qtype.isa<mlir::quant::UniformQuantizedPerAxisType>())) {
      return;
    }
    for (auto& use : value.getUses()) {
      Operation* user = use.getOwner();
      if (user->hasTrait<mlir::OpTrait::IsTerminator>()) continue;

      auto affine_user = llvm::dyn_cast<mlir::AffineQuantizedOpInterface>(user);
      if (affine_user &&
          affine_user.GetAffineOperandIndex() == use.getOperandNumber() &&
          affine_user.RequiredNarrowRangeAffineOperand())
        continue;
      // Create a fully range quantized constant.
      if (full_range_const == value) {
        mlir::quant::QuantizedType new_qtype;
        if (auto per_axis =
                qtype.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
          new_qtype = mlir::quant::UniformQuantizedPerAxisType::get(
              per_axis.getFlags(), per_axis.getStorageType(),
              per_axis.getExpressedType(), per_axis.getScales(),
              per_axis.getZeroPoints(), per_axis.getQuantizedDimension(),
              per_axis.getStorageTypeMin() - 1, per_axis.getStorageTypeMax());
        } else if (auto per_tensor =
                       qtype.dyn_cast<mlir::quant::UniformQuantizedType>()) {
          new_qtype = mlir::quant::UniformQuantizedType::get(
              per_tensor.getFlags(), per_tensor.getStorageType(),
              per_tensor.getExpressedType(), per_tensor.getScale(),
              per_tensor.getZeroPoint(), per_tensor.getStorageTypeMin() - 1,
              per_tensor.getStorageTypeMax());
        } else {
          return;  // Should not reach here, as it's already checked.
        }
        auto new_output_type = new_qtype.castFromExpressedType(
            mlir::quant::UniformQuantizedType::castToExpressedType(
                value.getType()));
        builder.setInsertionPointAfter(cst.getOperation());
        auto new_op = builder.create<tfl::QConstOp>(
            cst.getLoc(), new_output_type, mlir::TypeAttr::get(new_output_type),
            cst.getValueAttr());
        full_range_const = new_op.getOutput();
      }
      use.set(full_range_const);
    }
    if (cst.use_empty()) cst.erase();
  });
  return func;
}

// Helper method that returns the index of the tensor with name 'tensor_name'
// in the list of tensor names 'tensors'. It allows excluding some indices.
int GetTensorIndex(const std::string& tensor_name,
                   llvm::SmallVector<llvm::StringRef, 2> tensors,
                   const std::set<int>& exclude_indices = {}) {
  for (const auto& tensor_index_pair : llvm::enumerate(tensors)) {
    if (tensor_index_pair.value() == tensor_name &&
        exclude_indices.find(tensor_index_pair.index()) ==
            exclude_indices.end())
      return tensor_index_pair.index();
  }
  return -1;
}

// Helper method that returns list of all strings in a StringAttr identified
// by 'attr_key' and values are separated by a comma.
llvm::SmallVector<llvm::StringRef, 2> GetStringsFromAttrWithSeparator(
    mlir::DictionaryAttr attr, const std::string& attr_key) {
  llvm::SmallVector<llvm::StringRef, 2> result;
  if (auto str = attr.get(attr_key).dyn_cast_or_null<mlir::StringAttr>()) {
    str.getValue().split(result, ',', /*MaxSplit=*/-1,
                         /*KeepEmpty=*/false);
  }
  return result;
}

// Sets signature attributes on the function.
void SetSignature(
    FuncOp func, const tflite::SignatureDefT* signature,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tensors) {
  auto* context = func->getContext();
  static const char kEntryFunctionAttributes[] = "tf.entry_function";

  auto dict_attr =
      func->getAttrOfType<mlir::DictionaryAttr>(kEntryFunctionAttributes);
  if (!dict_attr) return;

  // Get Input and output tensor names from attribute.
  llvm::SmallVector<llvm::StringRef, 2> input_names =
      GetStringsFromAttrWithSeparator(dict_attr, /*attr_key=*/"inputs");
  llvm::SmallVector<llvm::StringRef, 2> output_names =
      GetStringsFromAttrWithSeparator(dict_attr, /*attr_key=*/"outputs");

  for (const auto& input_pair : llvm::enumerate(signature->inputs)) {
    const int arg_index = GetTensorIndex(
        tensors[input_pair.value()->tensor_index]->name, input_names);
    if (arg_index == -1) {
      func->emitWarning("Invalid signature tensors specified.");
      return;
    }
    func.setArgAttr(
        arg_index, kTfSavedModelIndexPathAttr,
        mlir::ArrayAttr::get(context, {mlir::StringAttr::get(
                                          context, input_pair.value()->name)}));
  }
  // Multiple signature outputs can refer to the same tensor. Avoid setting
  // signature output attribute at the same index by maintaining a set.
  std::set<int> seen_indices;
  for (const auto& output_pair : llvm::enumerate(signature->outputs)) {
    const int arg_index =
        GetTensorIndex(tensors[output_pair.value()->tensor_index]->name,
                       output_names, seen_indices);
    if (arg_index == -1) {
      func->emitWarning("Invalid signature tensors specified.");
      return;
    }
    func.setResultAttr(arg_index, kTfSavedModelIndexPathAttr,
                       mlir::ArrayAttr::get(
                           context, {mlir::StringAttr::get(
                                        context, output_pair.value()->name)}));
    seen_indices.insert(arg_index);
  }
  func->setAttr(
      kTfSavedModelExportedNamesAttr,
      mlir::ArrayAttr::get(
          context, {mlir::StringAttr::get(context, signature->signature_key)}));
}

// There are control nodes at each end of each control edge. For each of them,
// we store the source vertices of the incoming edges (if any) and the control
// node's output token. To improve testability, we use an ordered set for the
// source vertices.
struct ControlNodeDesc {
  std::set<int> incoming;
  std::optional<mlir::Value> outgoing;
};

using ControlNodes = llvm::DenseMap<int, ControlNodeDesc>;

// Helper function: After op has been emitted as the MLIR representation of
// a subgraph's operators[op_index], check *control_nodes whether it needs to be
// wrapped in a ControlNode because it's at either end of a control edge from
// the metadata. If it is, wrap it in a ControlNode, store the resulting
// ControlType token in *control_nodes, and return the non-ControlType (i.e.,
// tensor) results.  If it isn't, just return the original operator's results.
mlir::ResultRange MaybeWrapInControlNode(mlir::Operation* op,
                                         OpBuilder op_builder, int op_index,
                                         Location op_loc,
                                         ControlNodes* control_nodes) {
  const ControlNodes::iterator maybe_control_node =
      control_nodes->find(op_index);
  if (maybe_control_node == control_nodes->end()) {
    return op->getResults();
  }
  mlir::Region region;
  region.push_back(new mlir::Block);
  auto saved_pos = op_builder.saveInsertionPoint();
  op_builder.setInsertionPointToEnd(&region.front());
  mlir::Operation* cloned_op = op_builder.clone(*op);
  // Add the yield operation.
  op_builder.create<mlir::TFL::YieldOp>(op_loc, cloned_op->getResults());
  // Now emit into the function body again.
  op_builder.restoreInsertionPoint(saved_pos);

  // The ControlNodeOp depends on all control tokens emitted by the nodes at the
  // other end of the incoming edges. Since we're proceding in a valid
  // topological order, all lookups of these tokens in
  // (*control_nodes)[incoming] should be valid. However, we might (in theory)
  // have pruned an operator above, so we only emit values that have been
  // populated.
  llvm::SmallVector<Value, 2> control_tokens;
  for (const int incoming : maybe_control_node->second.incoming) {
    if (const auto& outgoing = (*control_nodes)[incoming].outgoing; outgoing) {
      control_tokens.push_back(*outgoing);
    }
  }

  // Create the ControlNodeOp.
  auto ctrl_op = op_builder.create<mlir::TFL::ControlNodeOp>(
      op_loc, cloned_op->getResultTypes(),
      mlir::TFL::ControlType::get(op->getContext()), control_tokens);
  ctrl_op.getBody().takeBody(region);

  // Store the control_token output for use by downstream nodes.
  maybe_control_node->second.outgoing = ctrl_op.getControl();

  // Remove the original op.
  op->replaceAllUsesWith(ctrl_op.getOutputs());
  op->erase();
  return ctrl_op.getOutputs();
}

// Build a FuncOp from a tflite SubGraph
// The buffers are directly taken
// from the deserialized flatbuffer as we do not have the type information to
// interpret them until this point. The base_loc parameter is the location of
// the flatbuffer as a whole (usually a file). If ordered_output_arrays is not
// empty, then the imported mlir function will only return nodes in
// ordered_output_arrays in the same order. If signature is not null, then the
// inputs/outputs in signature will be attached to the FuncOp.
StatusOr<FuncOp> ConvertSubgraph(
    const tflite::SubGraphT& subgraph, llvm::StringRef name,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& op_codes,
    const std::vector<std::string>& func_names,
    const std::vector<std::unique_ptr<tflite::BufferT>>& buffers,
    Location base_loc, Builder builder, bool is_entry_point,
    bool use_external_constant,
    const std::vector<std::string>& ordered_input_arrays,
    const std::vector<std::string>& ordered_output_arrays,
    bool experimental_prune_unreachable_nodes_unconditionally,
    const tflite::SignatureDefT* signature,
    const tflite::ControlEdges& control_edges,
    const std::unique_ptr<tfl::FlatBufferModelAbslError>& model_ptr,
    bool use_stablehlo_constant, DebugMetadata& debug_metadata) {
  // Populate from metadata.
  ControlNodes control_nodes;
  for (const auto [from, to] : control_edges) {
    control_nodes.try_emplace(from);
    control_nodes[to].incoming.insert(from);
  }

  llvm::SmallVector<mlir::Type, 2> ret_types;
  llvm::SmallVector<mlir::Type, 4> input_types;

  auto func_loc = mlir::NameLoc::get(builder.getStringAttr(name), base_loc);
  std::vector<int> func_inputs = subgraph.inputs;
  if (is_entry_point && !ordered_input_arrays.empty()) {
    if (!experimental_prune_unreachable_nodes_unconditionally) {
      // TODO(b/149922113): Resolve input-arrays/pruning flags interaction.
      return errors::InvalidArgument(
          "input-arrays should be used with experimental pruning flag");
    }
    TF_ASSIGN_OR_RETURN(func_inputs,
                        GetTensorIndices(subgraph, ordered_input_arrays));
  }

  for (int input : func_inputs) {
    auto& tensor = *subgraph.tensors.at(input);
    auto type_or_err = tfl::GetTensorType(tensor, builder);
    if (!type_or_err.ok()) {
      emitError(func_loc, "error reading argument types")
          << type_or_err.status().ToString();
      return type_or_err.status();
    }
    auto type = std::move(type_or_err).value();
    input_types.push_back(type);
  }

  llvm::SmallVector<bool, 16> is_op_output(subgraph.tensors.size(), false);
  for (auto& op : subgraph.operators) {
    for (auto output : op->outputs) {
      is_op_output[output] = true;
    }
  }

  std::vector<int> func_outputs = subgraph.outputs;
  if (is_entry_point && !ordered_output_arrays.empty()) {
    TF_ASSIGN_OR_RETURN(func_outputs,
                        GetTensorIndices(subgraph, ordered_output_arrays));
  }

  for (auto output : func_outputs) {
    const bool is_func_input = std::find(func_inputs.begin(), func_inputs.end(),
                                         output) != func_inputs.end();
    bool is_constant = !is_op_output[output] && !is_func_input;

    auto type_or_err =
        tfl::GetTensorType(*subgraph.tensors.at(output), builder, is_constant);
    if (!type_or_err.ok()) {
      emitError(func_loc, "error reading return types")
          << type_or_err.status().ToString();
      return type_or_err.status();
    }
    auto type = std::move(type_or_err).value();
    ret_types.push_back(type);
  }
  auto func_type = builder.getFunctionType(input_types, ret_types);

  // Construct function object
  auto func = FuncOp::create(func_loc, name, func_type, /* attrs= */ {});
  func.addEntryBlock();
  auto& body = func.getBody();
  OpBuilder op_builder{body};

  std::vector<Value> vals_map(subgraph.tensors.size(), nullptr);
  Value maybe_optional_arg_marker = nullptr;

  // Get or construct MLIR values for each input
  for (int i = 0, e = func_inputs.size(); i < e; i++) {
    auto input_tensor = func_inputs[i];
    const auto& tensor = *subgraph.tensors.at(input_tensor);
    auto loc = TensorLoc(tensor, builder, base_loc);
    if (vals_map[input_tensor]) {
      auto err = errors::FailedPrecondition("duplicate input arguments");
      return emitError(loc, err.ToString()), err;
    }
    Value input_value = func.getArgument(i);

    // If the `tensor` has min/max and doesn't have scale/zero_point
    // information, a stats op is created to use the input_value, then the
    // `tensor` should be mapped to the result of this new stats op.
    if (auto stats_op =
            ConvertMinMaxToStatsOp(tensor, op_builder, input_value)) {
      vals_map[input_tensor] = stats_op->getResult(0);
    } else {
      vals_map[input_tensor] = input_value;
    }
  }

  // Set tf.entry_function attribute
  if (is_entry_point) {
    llvm::SmallVector<mlir::NamedAttribute, 2> attributes;
    if (HasNonEmptyNames(subgraph, func_inputs)) {
      attributes.push_back(BuildTFEntryFunctionAttribute(
          subgraph, &builder, "inputs", func_inputs));
    }
    if (HasNonEmptyNames(subgraph, func_outputs)) {
      attributes.push_back(BuildTFEntryFunctionAttribute(
          subgraph, &builder, "outputs", func_outputs));
    }
    if (!attributes.empty()) {
      func->setAttr("tf.entry_function", builder.getDictionaryAttr(attributes));
    }
  } else {
    func.setPrivate();
  }

  // Set signature on function.
  if (signature) {
    SetSignature(func, signature, subgraph.tensors);
  }

  absl::flat_hash_set<const tflite::OperatorT*> pruned_subgraph_ops;
  if (experimental_prune_unreachable_nodes_unconditionally) {
    TF_ASSIGN_OR_RETURN(pruned_subgraph_ops,
                        PruneSubgraph(subgraph, func_inputs, func_outputs));
  }

  // Construct MLIR operators from TFLite operators
  for (const auto& it : llvm::enumerate(subgraph.operators)) {
    auto& op = it.value();

    if (experimental_prune_unreachable_nodes_unconditionally &&
        !pruned_subgraph_ops.contains(op)) {
      continue;
    }

    for (auto input_num : op->inputs) {
      // The operators in a graph are topologically sorted
      // and so if no previous operation has produced a tensor
      // it must be a constant.
      if (input_num == -1) {
        if (maybe_optional_arg_marker == nullptr) {
          maybe_optional_arg_marker =
              op_builder
                  .create<mlir::TFL::NoValueOp>(base_loc, builder.getNoneType(),
                                                builder.getUnitAttr())
                  .getResult();
        }
      } else if (!vals_map.at(input_num)) {
        auto& const_tensor = *subgraph.tensors[input_num];
        auto const_loc = TensorLoc(const_tensor, builder, base_loc);
        StatusOr<Operation*> op_or_err;
        std::vector<uint8_t> buffer;
        // Check if constant tensor is stored outside of the flatbuffers.
        if (IsValidBufferOffset(buffers[const_tensor.buffer]->offset)) {
          const uint8_t* file_begin_ptr =
              reinterpret_cast<const uint8_t*>(model_ptr->allocation()->base());
          buffer = std::vector<uint8_t>(
              file_begin_ptr + buffers[const_tensor.buffer]->offset,
              file_begin_ptr + buffers[const_tensor.buffer]->offset +
                  buffers[const_tensor.buffer]->size);
        } else {
          buffer = buffers[const_tensor.buffer]->data;
        }
        op_or_err =
            use_external_constant
                ? BuildExternalConstOp(const_tensor, const_tensor.buffer,
                                       op_builder, const_loc)
                : BuildConstOp(const_tensor, buffer, const_tensor.is_variable,
                               op_builder, const_loc, use_stablehlo_constant);
        if (!op_or_err.ok()) {
          return emitError(const_loc, op_or_err.status().ToString()),
                 op_or_err.status();
        }
        vals_map[input_num] = op_or_err.value()->getResult(0);
      }
    }

    // Intermediate tensors for LSTMs are used to carry quantization range
    // in their types, so we only need and extract their types.
    std::vector<mlir::TensorType> intermediate_types;
    intermediate_types.reserve(5);
    for (auto intermediate : op->intermediates) {
      TF_ASSIGN_OR_RETURN(
          mlir::TensorType type,
          tfl::GetTensorType(*subgraph.tensors[intermediate], builder,
                             /*is_constant=*/false,
                             /*is_intermediate=*/true));
      intermediate_types.emplace_back(type);
    }

    auto op_loc = OpLoc(*op, builder, debug_metadata, subgraph, base_loc);

    // If there's an optional argument, maybe_optional_arg_marker has been set
    // to a valid Value
    TF_ASSIGN_OR_RETURN(
        auto* mlir_op,
        ConvertOp(*op, vals_map, intermediate_types, maybe_optional_arg_marker,
                  op_codes, func_names, subgraph.tensors, op_loc, op_builder,
                  model_ptr));

    // Add the results to the value maps. There are two cases: 1. the result
    // tensor does not have min/max values, the original op result is used
    // directly; 2. the result tensor has some min/max values, a stats op is
    // created, then the result of the stats op is used.
    for (const auto& pair : llvm::enumerate(MaybeWrapInControlNode(
             mlir_op, op_builder, it.index(), op_loc, &control_nodes))) {
      int output_tensor_index = op->outputs[pair.index()];
      auto& tensor = *subgraph.tensors[output_tensor_index];
      if (auto stats_op =
              ConvertMinMaxToStatsOp(tensor, op_builder, pair.value())) {
        vals_map[output_tensor_index] = stats_op->getResult(0);
      } else {
        vals_map[output_tensor_index] = pair.value();
      }
    }
  }

  // Construct return values
  llvm::SmallVector<Value, 4> return_operands;
  for (auto index : func_outputs) {
    if (!vals_map.at(index)) {
      auto& const_tensor = *subgraph.tensors[index];
      auto const_loc = TensorLoc(const_tensor, builder, base_loc);
      StatusOr<Operation*> op_or_err;
      std::vector<uint8_t> buffer;
      // Check if constant tensor is stored outside of the flatbuffers.
      if (IsValidBufferOffset(buffers[const_tensor.buffer]->offset)) {
        const uint8_t* file_begin_ptr =
            reinterpret_cast<const uint8_t*>(model_ptr->allocation()->base());

        buffer = std::vector<uint8_t>(
            file_begin_ptr + buffers[const_tensor.buffer]->offset,
            file_begin_ptr + buffers[const_tensor.buffer]->offset +
                buffers[const_tensor.buffer]->size);
      } else {
        buffer = buffers[const_tensor.buffer]->data;
      }
      op_or_err =
          use_external_constant
              ? BuildExternalConstOp(const_tensor, const_tensor.buffer,
                                     op_builder, const_loc)
              : BuildConstOp(const_tensor, buffer, const_tensor.is_variable,
                             op_builder, const_loc, use_stablehlo_constant);
      if (!op_or_err.ok()) {
        return emitError(const_loc, op_or_err.status().ToString()),
               op_or_err.status();
      }
      vals_map[index] = op_or_err.value()->getResult(0);
    }
    return_operands.push_back(vals_map[index]);
  }

  op_builder.create<mlir::func::ReturnOp>(base_loc, return_operands);

  return PostProcessFuncOp(func);
}

// TFLite subgraphs do not necessarily have names, though MLIR functions must
// have them, so we generate a name for subgraphs that are missing one here.
// Note: in TFLite, the first subgraph is the entry point, and in MLIR that
// represents TFLite, this entry point must be called "main"
std::string SubgraphName(bool set_implicit_main_func, unsigned index,
                         const tflite::SubGraphT& subgraph) {
  if (index == 0 && set_implicit_main_func) {
    return "main";
  }
  if (subgraph.name.empty()) {
    return llvm::formatv("fn_{0}", index).str();
  }
  return subgraph.name;
}

// Adds a CallOp in `region` to call the `func` and returns the results of
// CallOp.
void AddCallOpInWhileOpRegion(mlir::Region& region, mlir::func::FuncOp func) {
  OpBuilder op_builder{region};
  region.push_back(new mlir::Block());
  Location loc = region.getLoc();
  auto inputs = func.getFunctionType().getInputs();
  region.addArguments(inputs, mlir::SmallVector<Location>(inputs.size(), loc));
  op_builder.setInsertionPointToStart(&region.front());
  auto call_op = op_builder.create<mlir::func::CallOp>(
      loc, func.getFunctionType().getResults(), func.getSymName(),
      region.getArguments());
  op_builder.create<mlir::TFL::YieldOp>(loc, call_op.getResults());
}

void InlineStablehloOpRegion(mlir::Region& region, mlir::func::FuncOp func) {
  OpBuilder op_builder{region};
  mlir::IRMapping mapper;
  func.getBody().cloneInto(&region, mapper);
  mlir::Operation& return_op = region.back().back();
  mlir::Location loc = return_op.getLoc();
  op_builder.setInsertionPointToEnd(&region.back());
  op_builder.create<mlir::stablehlo::ReturnOp>(loc, return_op.getOperands());
  return_op.erase();
}

void InlineVhloOpRegion(mlir::Region& region, mlir::func::FuncOp func) {
  OpBuilder op_builder{region};
  mlir::IRMapping mapper;
  func.getBody().cloneInto(&region, mapper);
  mlir::Operation& return_op = region.back().back();
  mlir::Location loc = return_op.getLoc();
  op_builder.setInsertionPointToEnd(&region.back());
  op_builder.create<mlir::vhlo::ReturnOpV1>(loc, return_op.getOperands());
  return_op.erase();
}

// TFL::WhileOp has regions, so we add CallOp to call the FuncOp in the regions
// if we have while ops.
void AddRegionsForTflWhileOp(mlir::ModuleOp module) {
  mlir::SymbolTable symbol_table(module);
  module.walk([&](mlir::TFL::WhileOp while_op) {
    auto cond = symbol_table.lookup<mlir::func::FuncOp>(
        while_op->getAttr("cond").cast<mlir::FlatSymbolRefAttr>().getValue());
    AddCallOpInWhileOpRegion(while_op.getCond(), cond);
    while_op->removeAttr("cond");
    auto body = symbol_table.lookup<mlir::func::FuncOp>(
        while_op->getAttr("body").cast<mlir::FlatSymbolRefAttr>().getValue());
    AddCallOpInWhileOpRegion(while_op.getBody(), body);
    while_op->removeAttr("body");
  });
}

void AddRegionsForStableHLOOp(mlir::ModuleOp module) {
  mlir::SymbolTable symbol_table(module);
  std::vector<mlir::func::FuncOp> to_delete_funcs;
  module.walk([&](mlir::vhlo::ReduceOpV1 reduce_op) {
    auto body = symbol_table.lookup<mlir::func::FuncOp>(
        reduce_op->getAttr("body").cast<mlir::FlatSymbolRefAttr>().getValue());
    InlineVhloOpRegion(reduce_op.getBody(), body);
    reduce_op->removeAttr("body");
    to_delete_funcs.push_back(body);
  });
  module.walk([&](mlir::vhlo::ReduceWindowOpV1 reduce_window_op) {
    auto body = symbol_table.lookup<mlir::func::FuncOp>(
        reduce_window_op->getAttr("body")
            .cast<mlir::FlatSymbolRefAttr>()
            .getValue());
    InlineVhloOpRegion(reduce_window_op.getBody(), body);
    reduce_window_op->removeAttr("body");
    to_delete_funcs.push_back(body);
  });
  module.walk([&](mlir::vhlo::ScatterOpV1 scatter_op) {
    auto update_computation = symbol_table.lookup<mlir::func::FuncOp>(
        scatter_op->getAttr(kScatterRegionFuncName)
            .cast<mlir::FlatSymbolRefAttr>()
            .getValue());
    InlineVhloOpRegion(scatter_op.getUpdateComputation(), update_computation);
    scatter_op->removeAttr(kScatterRegionFuncName);
    to_delete_funcs.push_back(update_computation);
  });
  module.walk([&](mlir::vhlo::SortOpV1 sort_op) {
    auto comparator = symbol_table.lookup<mlir::func::FuncOp>(
        sort_op->getAttr("comparator")
            .cast<mlir::FlatSymbolRefAttr>()
            .getValue());
    InlineVhloOpRegion(sort_op.getComparator(), comparator);
    sort_op->removeAttr("comparator");
    to_delete_funcs.push_back(comparator);
  });
  module.walk([&](mlir::vhlo::WhileOpV1 while_op) {
    auto cond = symbol_table.lookup<mlir::func::FuncOp>(
        while_op->getAttr("cond").cast<mlir::FlatSymbolRefAttr>().getValue());
    InlineVhloOpRegion(while_op.getCond(), cond);
    while_op->removeAttr("cond");
    auto body = symbol_table.lookup<mlir::func::FuncOp>(
        while_op->getAttr("body").cast<mlir::FlatSymbolRefAttr>().getValue());
    InlineVhloOpRegion(while_op.getBody(), body);
    while_op->removeAttr("body");
    to_delete_funcs.push_back(body);
    to_delete_funcs.push_back(cond);
  });
  for (auto& func : to_delete_funcs) {
    func.erase();
  }
}
}  // namespace

OwningOpRef<mlir::ModuleOp> tflite::FlatBufferToMlir(
    absl::string_view buffer, MLIRContext* context, Location base_loc,
    bool use_external_constant,
    const std::vector<std::string>& ordered_input_arrays,
    const std::vector<std::string>& ordered_output_arrays,
    bool experimental_prune_unreachable_nodes_unconditionally,
    const bool disable_vhlo_to_stablehlo) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::quant::QuantDialect,
                  mlir::quantfork::QuantizationForkDialect,
                  mlir::TFL::TensorFlowLiteDialect, mlir::TF::TensorFlowDialect,
                  mlir::stablehlo::StablehloDialect, mlir::vhlo::VhloDialect>();
  mlir::func::registerAllExtensions(registry);
  context->appendDialectRegistry(registry);

  context->loadDialect<
      mlir::arith::ArithDialect, mlir::func::FuncDialect,
      mlir::quant::QuantDialect,
      mlir::quantfork::QuantizationForkDialect,
      mlir::TFL::TensorFlowLiteDialect, mlir::TF::TensorFlowDialect,
      mlir::stablehlo::StablehloDialect, mlir::vhlo::VhloDialect>();

  auto model_ptr = tfl::FlatBufferModelAbslError::VerifyAndBuildFromBuffer(
      buffer.data(), buffer.length());
  if (nullptr == model_ptr) {
    return emitError(base_loc, "couldn't parse flatbuffer"), nullptr;
  }

  std::unique_ptr<ModelT> model(model_ptr->GetModel()->UnPack());

  auto builder = Builder(context);

  tflite::ModelControlDependencies model_control_dependencies(
      model->subgraphs.size());

  bool use_stablehlo_constant = false;

  llvm::SmallVector<mlir::NamedAttribute> metadata_attrs;
  mlir::StringSet<> seen_attr;
  DebugMetadata debug_metadata;
  for (const auto& metadata : model->metadata) {
    if (metadata->name == tflite::kModelControlDependenciesMetadataKey) {
      const std::vector<uint8_t>& data = model->buffers[metadata->buffer]->data;
      if (!ParseModelControlDependencies(
              reinterpret_cast<const char*>(data.data()), data.size(),
              &model_control_dependencies)) {
        return emitError(base_loc,
                         "invalid model_control_dependencies metadata"),
               nullptr;
      }
      continue;
    }

    // Skip already seen attributes. Ideally there should be no duplicates here.
    if (!seen_attr.try_emplace(metadata->name).second) continue;

    // check if the model is serialized using stablehlo constant tensor
    if (metadata->name == tflite::kModelUseStablehloTensorKey) {
      use_stablehlo_constant = true;
      metadata_attrs.emplace_back(builder.getStringAttr(metadata->name),
                                  builder.getStringAttr("true"));
      continue;
    }

    if (metadata->name == "debug_metadata") {
      const std::vector<uint8_t>& data = model->buffers[metadata->buffer]->data;
      auto status = ParseDebugMetadata(
          builder, reinterpret_cast<const char*>(data.data()), data.size(),
          debug_metadata);
      if (!status.ok()) {
        return emitError(base_loc, std::string(status.message())), nullptr;
      }
      continue;
    }

    std::vector<uint8_t> buffer = model->buffers[metadata->buffer]->data;
    metadata_attrs.emplace_back(
        builder.getStringAttr(metadata->name),
        builder.getStringAttr(llvm::StringRef(
            reinterpret_cast<char*>(buffer.data()), buffer.size())));
  }

  std::vector<std::string> func_names;
  for (auto& subgraph : model->subgraphs) {
    func_names.push_back(subgraph->name);
  }

  auto module = mlir::ModuleOp::create(base_loc);
  // We currently don't use this to make decisions, but we could
  // use it in exports or if there are breaking changes
  module->setAttr("tfl.schema_version",
                  builder.getI32IntegerAttr(model->version));
  if (!model->description.empty()) {
    module->setAttr("tfl.description",
                    builder.getStringAttr(model->description));
  }

  if (!metadata_attrs.empty()) {
    module->setAttr("tfl.metadata", builder.getDictionaryAttr(metadata_attrs));
  }

  if (!model->signature_defs.empty()) {
    module->setAttr("tf_saved_model.semantics",
                    mlir::UnitAttr::get(builder.getContext()));
  }

  absl::flat_hash_map<uint32_t, tflite::SignatureDefT*>
      subgraph_to_signature_map;
  for (int i = 0; i < model->signature_defs.size(); i++) {
    auto* signature_def = model->signature_defs[i].get();
    const uint32_t subgraph_index = signature_def->subgraph_index;
    subgraph_to_signature_map[subgraph_index] = signature_def;
  }

  const bool set_implicit_main_func = subgraph_to_signature_map.size() <= 1;
  for (const auto& e : llvm::enumerate(model->subgraphs)) {
    auto& subgraph = e.value();
    std::string name =
        SubgraphName(set_implicit_main_func, e.index(), *subgraph);
    uint32_t subgraph_index = static_cast<uint32_t>(e.index());
    auto func_or_error = ConvertSubgraph(
        *subgraph, name, model->operator_codes, func_names, model->buffers,
        base_loc, builder,
        /*is_entry_point=*/
        set_implicit_main_func
            ? e.index() == 0
            : subgraph_to_signature_map.contains(subgraph_index),
        /*use_external_constant=*/use_external_constant, ordered_input_arrays,
        ordered_output_arrays,
        experimental_prune_unreachable_nodes_unconditionally,
        subgraph_to_signature_map.contains(subgraph_index)
            ? subgraph_to_signature_map.at(subgraph_index)
            : nullptr,
        model_control_dependencies[subgraph_index], model_ptr,
        use_stablehlo_constant, debug_metadata);
    if (!func_or_error.ok()) {
      return emitError(base_loc, "could not translate function ")
                 << subgraph->name << ": " << func_or_error.status().message(),
             nullptr;
    }
    module.push_back(std::move(func_or_error).value());
  }
  AddRegionsForTflWhileOp(module);
  AddRegionsForStableHLOOp(module);
  if (!disable_vhlo_to_stablehlo) {
    mlir::PassManager pass_manager(module.getContext());
    pass_manager.addPass(mlir::odml::createLegalizeVhloToStablehloPass());
    pass_manager.addPass(mlir::createReconcileUnrealizedCastsPass());
    auto result = pass_manager.run(module);
    if (failed(result)) {
      return nullptr;
    }
  }
  return OwningOpRef<mlir::ModuleOp>(module);
}
