/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/spmd/shardy/extensions/custom_call_extensions.h"

#include <optional>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/service/custom_call_sharding_helper.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h"

namespace xla {
namespace sdy {

namespace {

constexpr char kInspectShardingTargetName[] = "InspectSharding";

// Converts `value`'s Shardy tensor sharding into an `xla::HloSharding`
// (defaulting to replicated when unannotated), or emits an error on `op` and
// returns `std::nullopt` if the referenced mesh symbol does not exist.
std::optional<xla::HloSharding> convertValueShardingToHloSharding(
    mlir::Operation* op, mlir::Value value,
    const mlir::SymbolTable& symbolTable) {
  auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value);
  mlir::sdy::TensorShardingAttr sdySharding =
      (blockArg &&
       (!blockArg.getOwner() || !blockArg.getOwner()->getParentOp()))
          ? nullptr
          : mlir::sdy::getSharding(value);
  if (!sdySharding) {
    return xla::HloSharding::Replicate();
  }
  mlir::sdy::MeshAttr mesh = sdySharding.getMesh(symbolTable);
  if (!mesh) {
    op->emitOpError("unknown mesh: ") << sdySharding.getMeshOrRef();
    return std::nullopt;
  }
  return xla::sdy::convertToHloSharding(
      sdySharding, [&](mlir::sdy::TensorShardingAttr) { return mesh; });
}

// Extracts the raw `backend_config` byte slice from `customCall` without
// null-terminated C-string conversion, or returns an empty view if absent.
absl::string_view getRawBackendConfig(
    mlir::stablehlo::CustomCallOp customCall) {
  if (std::optional<mlir::Attribute> backendConfigAttr =
          customCall.getBackendConfig()) {
    if (auto strAttr = llvm::dyn_cast<mlir::StringAttr>(*backendConfigAttr)) {
      llvm::StringRef ref = strAttr.getValue();
      return absl::string_view(ref.data(), ref.size());
    }
  }
  return {};
}

// Resolves the input sharding of an `@InspectSharding` custom call and converts
// it to an `xla::HloSharding`. If `operand(0)` is a `BlockArgument` whose
// owning block was already unlinked by `FuncOp` or `ManualComputationOp`
// signature conversion, `sdy::getSharding(operand(0))` cannot walk to the
// parent op; in that case, reads the propagated inner sharding from `result(0)`
// (which has manual axes stripped inside `sdy.manual_computation`) or falls
// back to `convertedOperand` (which is attached to the converted `FuncOp` entry
// block).
std::optional<xla::HloSharding> getInspectShardingInputHloSharding(
    mlir::stablehlo::CustomCallOp customCall, mlir::Value convertedOperand,
    const mlir::SymbolTable& symbolTable) {
  mlir::Value value = customCall.getOperand(0);
  if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value);
      blockArg &&
      (!blockArg.getOwner() || !blockArg.getOwner()->getParentOp())) {
    value = mlir::sdy::getSharding(customCall.getResult(0))
                ? customCall.getResult(0)
                : convertedOperand;
  }
  return convertValueShardingToHloSharding(customCall, value, symbolTable);
}

// Lowers `stablehlo.custom_call @InspectSharding` by invoking the registered
// sharding-inspection callback on the input operand's sharding and replacing
// the op with its localized operand.
mlir::LogicalResult convertInspectShardingToLocal(
    mlir::stablehlo::CustomCallOp customCall, mlir::ValueRange operands,
    mlir::ConversionPatternRewriter& rewriter,
    const mlir::SymbolTable& symbolTable) {
  if (customCall.getNumOperands() != 1 || customCall.getNumResults() != 1 ||
      operands.size() != 1) {
    return customCall.emitOpError(
        "expected InspectSharding custom call to have 1 operand and 1 result");
  }

  std::optional<xla::HloSharding> hloSharding =
      getInspectShardingInputHloSharding(customCall, operands[0], symbolTable);
  if (!hloSharding.has_value()) {
    return mlir::failure();
  }

  if (const xla::CustomCallPartitioner* partitioner =
          xla::GetCustomCallPartitioner(kInspectShardingTargetName)) {
    absl::Status status = partitioner->InspectSharding(
        *hloSharding, getRawBackendConfig(customCall));
    if (!status.ok()) {
      return customCall.emitOpError("InspectSharding callback failed: ")
             << status.message();
    }
  }

  rewriter.replaceOp(customCall, operands[0]);
  return mlir::success();
}

using ConvertGlobalToLocalHandler = mlir::LogicalResult (*)(
    mlir::stablehlo::CustomCallOp, mlir::ValueRange,
    mlir::ConversionPatternRewriter&, const mlir::SymbolTable&);

// Dispatches `stablehlo.custom_call` global-to-local lowering by
// `call_target_name`, returning `std::nullopt` for unhandled targets so Shardy
// falls back to generic op localization.
struct CustomCallPartitioningOpInterface
    : public mlir::sdy::PartitioningOpInterface::ExternalModel<
          CustomCallPartitioningOpInterface, mlir::stablehlo::CustomCallOp> {
  std::optional<mlir::LogicalResult> convertGlobalToLocal(
      mlir::Operation* op, mlir::ValueRange operands,
      mlir::ConversionPatternRewriter& rewriter,
      const mlir::TypeConverter& /*typeConverter*/,
      const mlir::SymbolTable& symbolTable) const {
    auto customCall = llvm::cast<mlir::stablehlo::CustomCallOp>(op);
    ConvertGlobalToLocalHandler handler =
        llvm::StringSwitch<ConvertGlobalToLocalHandler>(
            customCall.getCallTargetName())
            .Case(kInspectShardingTargetName, &convertInspectShardingToLocal)
            .Default(nullptr);
    if (!handler) {
      return std::nullopt;
    }
    return handler(customCall, operands, rewriter, symbolTable);
  }
};

}  // namespace

void registerCustomCallExtensions(mlir::DialectRegistry& registry) {
  registry.addExtension(
      +[](mlir::MLIRContext* ctx, mlir::stablehlo::StablehloDialect*) {
        mlir::stablehlo::CustomCallOp::attachInterface<
            CustomCallPartitioningOpInterface>(*ctx);
      });
}

}  // namespace sdy
}  // namespace xla
