/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/tf_lift_as_function_call.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_call_module_attrs.h"

//===----------------------------------------------------------------------===//
// The stablehlo-restore-function-name Pass.
//===----------------------------------------------------------------------===//

namespace mlir::tf_quant::stablehlo {

#define GEN_PASS_DEF_RESTOREFUNCTIONNAMEPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/tf_passes.h.inc"

namespace {

// Restores entry function name from XlaCallModuleOp attribute.
// This restoration is required because StableHLO functions are renamed during
// the XlaCallModuleSerialization.
class RestoreFunctionNamePass
    : public impl::RestoreFunctionNamePassBase<RestoreFunctionNamePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RestoreFunctionNamePass)

  explicit RestoreFunctionNamePass() = default;

  void runOnOperation() override;
};

void RestoreFunctionNameFromXlaCallModuleOp(TF::XlaCallModuleOp& call_op,
                                            SymbolTable& symbol_table) {
  if (!call_op->hasAttr(kOriginalStablehloEntryFunctionAttrName)) {
    return;
  }

  const auto original_function_name = call_op->getAttrOfType<StringAttr>(
      kOriginalStablehloEntryFunctionAttrName);
  const auto current_function_name = call_op->getAttrOfType<FlatSymbolRefAttr>(
      TF::kStablehloEntryFunctionAttrName);

  if (!original_function_name || !current_function_name) {
    return;
  }

  auto function =
      symbol_table.lookup<func::FuncOp>(current_function_name.getValue());
  if (function) {
    function.setName(original_function_name);
  }

  call_op->setAttr(TF::kStablehloEntryFunctionAttrName,
                   FlatSymbolRefAttr::get(original_function_name));
}

void RestoreFunctionNamePass::runOnOperation() {
  ModuleOp module_op = getOperation();

  MLIRContext* ctx = module_op.getContext();
  OpBuilder builder(ctx);
  SymbolTable symbol_table(module_op);

  // TODO - b/298966126: Improve this logic if needed.
  module_op.walk([&](TF::XlaCallModuleOp call_op) {
    RestoreFunctionNameFromXlaCallModuleOp(call_op, symbol_table);
  });
}
}  // namespace

}  // namespace mlir::tf_quant::stablehlo
