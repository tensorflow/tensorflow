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

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/Serialization.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/tf_passes.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/stablehlo/utils/bfloat16_type.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir::tf_quant::stablehlo {

absl::StatusOr<std::string> ConvertSerializedStableHloModuleToBfloat16(
    const StringRef serialized_stablehlo_module) {
  // StableHLO module is empty often because the XlaCallModuleOp is already
  // deserialized, e.g. after invoking XlaCallModuleDeserializationPass. We
  // don't handle this situation.
  if (serialized_stablehlo_module.empty()) {
    return absl::InvalidArgumentError("StableHLO module is empty.");
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> stablehlo_module_op =
      mlir::stablehlo::deserializePortableArtifact(serialized_stablehlo_module,
                                                   &context);
  auto version =
      mlir::stablehlo::getPortableArtifactVersion(serialized_stablehlo_module);
  if (failed(version)) {
    return absl::InternalError(
        "Failed to get the deserialized StableHLO version, XlaCallModuleOp "
        "must have a valid StableHLO module serialized using "
        "stablehlo::serializePortableArtifact APIs.");
  }

  // Convert the StableHLO module to bfloat16.
  PassManager pm(&context);
  pm.addNestedPass<func::FuncOp>(createConvertFuncToBfloat16Pass());
  if (failed(pm.run(stablehlo_module_op.get()))) {
    return absl::InternalError(
        "Failed to convert StableHLO module to bfloat16.");
  }

  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  if (failed(mlir::stablehlo::serializePortableArtifact(
          stablehlo_module_op.get(), version.value().toString(), os))) {
    return absl::InternalError("Failed to serialize StableHLO module.");
  }
  return bytecode;
}

#define GEN_PASS_DEF_CONVERTXLACALLMODULEOPTOBFLOAT16PASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/tf_passes.h.inc"

namespace {
class ConvertXlaCallModuleOpToBfloat16Pass
    : public impl::ConvertXlaCallModuleOpToBfloat16PassBase<
          ConvertXlaCallModuleOpToBfloat16Pass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ConvertXlaCallModuleOpToBfloat16Pass)

  explicit ConvertXlaCallModuleOpToBfloat16Pass() = default;

 private:
  void runOnOperation() override;
};

void ConvertXlaCallModuleOpToBfloat16Pass::runOnOperation() {
  Operation* func_op = getOperation();
  SymbolTableCollection symbol_table;
  OpBuilder builder(&getContext());

  auto result = func_op->walk([&](TF::XlaCallModuleOp op) {
    // Converts the serialized StableHLO module to bfloat16.
    auto result =
        ConvertSerializedStableHloModuleToBfloat16(op.getModuleAttr());
    if (!result.ok()) {
      llvm::errs() << "Failed to convert StableHLO module to bfloat16: "
                   << result.status().message();
      return WalkResult::interrupt();
    }
    op.setModuleAttr(StringAttr::get(&getContext(), *result));

    // Convert the `tf.XlaCallModuleOp` to bfloat16 and add casts around it.
    builder.setInsertionPoint(op);
    for (auto& op_operand : op->getOpOperands()) {
      if (quant::stablehlo::IsLargeFloatType(op_operand.get().getType())) {
        op_operand.set(builder.create<TF::CastOp>(
            op->getLoc(),
            quant::stablehlo::ToBfloat16Type(op_operand.get().getType()),
            op_operand.get()));
      }
    }
    builder.setInsertionPointAfter(op);
    for (auto op_result : op->getOpResults()) {
      if (quant::stablehlo::IsLargeFloatType(op_result.getType())) {
        const Type original_type = op_result.getType();
        op_result.setType(quant::stablehlo::ToBfloat16Type(original_type));
        const Value cast =
            builder.create<TF::CastOp>(op->getLoc(), original_type, op_result);
        op_result.replaceAllUsesExcept(cast, cast.getDefiningOp());
      }
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) return signalPassFailure();
}

}  // namespace
}  // namespace mlir::tf_quant::stablehlo
