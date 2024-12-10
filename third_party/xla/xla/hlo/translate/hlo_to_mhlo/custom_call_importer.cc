/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/translate/hlo_to_mhlo/custom_call_importer.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/util.h"

namespace xla {
namespace {

absl::StatusOr<mlir::Operation*> ImportDynamicBroadcastInDimOp(
    mlir::StringRef backend_config, mlir::Location loc, mlir::Type result_type,
    mlir::ValueRange operands, mlir::OpBuilder* builder) {
  if (backend_config.empty()) {
    return Internal("backend_config attribute cannot be empty.");
  }

  auto attr = mlir::dyn_cast<mlir::DictionaryAttr>(
      mlir::parseAttribute(backend_config, builder->getContext()));
  if (!attr) {
    return Internal(
        "Couldn't parse backend config into a dictionary attribute");
  }

  auto broadcast_dimensions_attr =
      mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr.get("broadcast_dimensions"));
  if (!broadcast_dimensions_attr) {
    return Internal("broadcast_dimensions attribute is required.");
  }

  std::vector<int64_t> broadcast_dimensions(broadcast_dimensions_attr.size());
  for (auto [i, broadcast_dimension] :
       llvm::enumerate(broadcast_dimensions_attr)) {
    broadcast_dimensions[i] =
        mlir::cast<mlir::IntegerAttr>(broadcast_dimension).getInt();
  }

  return builder
      ->create<mlir::mhlo::DynamicBroadcastInDimOp>(
          loc, result_type, operands[0], operands[1],
          builder->getI64TensorAttr(broadcast_dimensions))
      .getOperation();
}

absl::StatusOr<mlir::Operation*> ImportDynamicReshapeOp(
    mlir::StringRef backend_config, mlir::Location loc, mlir::Type result_type,
    mlir::ValueRange operands, mlir::OpBuilder* builder) {
  if (!backend_config.empty()) {
    return Internal("backend_config attribute must be empty.");
  }
  return builder
      ->create<mlir::mhlo::DynamicReshapeOp>(loc, result_type, operands)
      .getOperation();
}

absl::StatusOr<mlir::Operation*> ImportRealDynamicSliceOp(
    mlir::StringRef backend_config, mlir::Location loc, mlir::Type result_type,
    mlir::ValueRange operands, mlir::OpBuilder* builder) {
  if (!backend_config.empty()) {
    return Internal("backend_config attribute must be empty.");
  }
  return builder
      ->create<mlir::mhlo::RealDynamicSliceOp>(loc, result_type, operands)
      .getOperation();
}

}  // namespace

mlir::Type getQuantizedType(mlir::DictionaryAttr& backend_config) {
  std::vector<double> scales;
  std::vector<int64_t> zero_points;
  int64_t quantization_dimension = -1, storage_max = 0, storage_min = 0;
  mlir::Type storage_type, expressed_type;

  if (const mlir::Attribute scales_attr = backend_config.get("scale");
      scales_attr) {
    for (auto scale_attr : mlir::cast<mlir::ArrayAttr>(scales_attr)) {
      scales.push_back(
          mlir::cast<mlir::FloatAttr>(scale_attr).getValueAsDouble());
    }
  }

  auto zero_points_attr = backend_config.get("zero_point");
  if (zero_points_attr) {
    for (auto zero_point_attr : mlir::cast<mlir::ArrayAttr>(zero_points_attr)) {
      zero_points.push_back(
          mlir::cast<mlir::IntegerAttr>(zero_point_attr).getInt());
    }
  }

  auto quantization_dimension_attr =
      backend_config.get("quantization_dimension");
  if (quantization_dimension_attr) {
    quantization_dimension =
        mlir::cast<mlir::IntegerAttr>(quantization_dimension_attr).getInt();
  }

  auto storage_max_attr = backend_config.get("storage_max");
  if (storage_max_attr) {
    storage_max = mlir::cast<mlir::IntegerAttr>(storage_max_attr).getInt();
  }

  auto storage_min_attr = backend_config.get("storage_min");
  if (storage_min_attr) {
    storage_min = mlir::cast<mlir::IntegerAttr>(storage_min_attr).getInt();
  }

  auto storage_type_attr = backend_config.get("storage_type");
  if (storage_type_attr) {
    storage_type = mlir::cast<mlir::TypeAttr>(storage_type_attr).getValue();
  }

  auto expressed_type_attr = backend_config.get("expressed_type");
  if (expressed_type_attr) {
    expressed_type = mlir::cast<mlir::TypeAttr>(expressed_type_attr).getValue();
  }

  auto is_signed = mlir::cast<mlir::IntegerType>(storage_type).isSignless();

  if (quantization_dimension != -1) {
    return mlir::quant::UniformQuantizedPerAxisType::get(
        is_signed, storage_type, expressed_type, scales, zero_points,
        quantization_dimension, storage_min, storage_max);
  } else {
    return mlir::quant::UniformQuantizedType::get(
        is_signed, storage_type, expressed_type, scales[0], zero_points[0],
        storage_min, storage_max);
  }
}

absl::StatusOr<mlir::Operation*> ImportCustomCallAsOp(
    const HloCustomCallInstruction* instruction, mlir::Location loc,
    mlir::Type result_type, mlir::ValueRange operands,
    mlir::OpBuilder* builder) {
  const std::string& custom_call_target = instruction->custom_call_target();
  const std::string& backend_config_str =
      instruction->raw_backend_config_string();
  if (custom_call_target == "mhlo.dynamic_broadcast_in_dim") {
    return ImportDynamicBroadcastInDimOp(backend_config_str, loc, result_type,
                                         operands, builder);
  }
  if (custom_call_target == "mhlo.dynamic_reshape") {
    return ImportDynamicReshapeOp(backend_config_str, loc, result_type,
                                  operands, builder);
  }
  if (custom_call_target == "mhlo.real_dynamic_slice") {
    return ImportRealDynamicSliceOp(backend_config_str, loc, result_type,
                                    operands, builder);
  }

  auto backend_config = mlir::dyn_cast<mlir::DictionaryAttr>(
      mlir::parseAttribute(backend_config_str, builder->getContext()));
  if (!backend_config) {
    return Internal(
        "Couldn't parse backend config into a dictionary attribute");
  }

  if (custom_call_target == "mhlo.uniform_quantize") {
    return builder
        ->create<mlir::mhlo::UniformQuantizeOp>(
            loc,
            mlir::RankedTensorType::get(
                mlir::cast<mlir::RankedTensorType>(result_type).getShape(),
                getQuantizedType(backend_config)),
            operands)
        .getOperation();
  }

  if (custom_call_target == "mhlo.uniform_dequantize") {
    return builder
        ->create<mlir::mhlo::UniformDequantizeOp>(loc, result_type, operands)
        .getOperation();
  }
  return InvalidArgument("Unsupported MHLO op custom_call %s",
                         custom_call_target);
}

bool IsOpEncodedCustomCall(const HloCustomCallInstruction* instruction) {
  return absl::StartsWith(instruction->custom_call_target(), "mhlo.");
}

}  // namespace xla
