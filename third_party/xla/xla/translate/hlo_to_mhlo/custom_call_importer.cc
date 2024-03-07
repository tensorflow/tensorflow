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

#include "xla/translate/hlo_to_mhlo/custom_call_importer.h"

#include <cstdint>
#include <vector>

#include "absl/strings/match.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/AsmParser/AsmParser.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/status.h"
#include "xla/util.h"

namespace xla {
namespace {

absl::StatusOr<mlir::Operation*> ImportDynamicBroadcastInDimOp(
    mlir::StringRef backend_config, mlir::Location loc, mlir::Type result_type,
    mlir::ValueRange operands, mlir::OpBuilder* builder) {
  if (backend_config.empty()) {
    return Internal("backend_config attribute cannot be empty.");
  }

  auto attr = mlir::parseAttribute(backend_config, builder->getContext())
                  .dyn_cast<mlir::DictionaryAttr>();
  if (!attr) {
    return Internal(
        "Couldn't parse backend config into a dictionary attribute");
  }

  auto broadcast_dimensions_attr =
      attr.get("broadcast_dimensions").dyn_cast_or_null<mlir::ArrayAttr>();
  if (!broadcast_dimensions_attr) {
    return Internal("broadcast_dimensions attribute is required.");
  }

  std::vector<int64_t> broadcast_dimensions(broadcast_dimensions_attr.size());
  for (auto [i, broadcast_dimension] :
       llvm::enumerate(broadcast_dimensions_attr)) {
    broadcast_dimensions[i] =
        broadcast_dimension.cast<mlir::IntegerAttr>().getInt();
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
  return InvalidArgument("Unsupported MHLO op custom_call %s",
                         custom_call_target);
}

bool IsOpEncodedCustomCall(const HloCustomCallInstruction* instruction) {
  return absl::StartsWith(instruction->custom_call_target(), "mhlo.");
}

}  // namespace xla
