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

#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Block.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/OperationSupport.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/SymbolTable.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassRegistry.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {
namespace {

constexpr char kTfLiteFunctionName[] = "_tflite_function_name";
constexpr char kUnidirectionalSequenceRnn[] = "UnidirectionalSequenceRnn";

// This pass is used for converting to TFLite composite op like
// UnidirectionalSequenceRNN, UnidirectionalSequenceLSTM or SVDF Op. Currently,
// this pass is only for ophint converted function op only. See below diagram:
//
// InputOp1      InputOp2 ...
//    \            /
//     \          /
//    call funcOp (say UnidirectionalSequenceRNN)
//           |
//           |
//        OutputOp1
//
//   funcOp() { '_tflite_function_name' = 'UnidirectionalSequenceRNN'}
//
//          ||
//          ||
//         \ /
//
// InputOp1      InputOp2 ...
//    \            /
//     \          /
//    tfl.UnidirectionalSequenceRNN
//           |
//           |
//        OutputOp1
struct LegalizeOphintFuncOpPass : public ModulePass<LegalizeOphintFuncOpPass> {
  void runOnModule() override;
};

llvm::StringMap<FuncOp> FindCompositeFuncOps(ModuleOp module) {
  llvm::StringMap<FuncOp> composite_func_ops;
  for (FuncOp func : module.getOps<FuncOp>()) {
    if (func.getAttr(kTfLiteFunctionName))
      composite_func_ops[func.getName()] = func;
  }
  return composite_func_ops;
}

LogicalResult BuildUnidirectionalSequenceRnnOp(FuncOp composite_func_op,
                                               CallOp* call_op,
                                               OpBuilder* builder,
                                               Operation** fused_op) {
  // UnidirectionalSequenceRnn takes exactly 5 inputs.
  if (composite_func_op.getNumArguments() != 5) return failure();
  if (call_op->getNumOperands() != 5) return failure();
  // UnidirectionalSequenceRnn has exactly 1 input.
  if (call_op->getNumResults() != 1) return failure();

  // Inputs is indexed at 0.
  Value* input = call_op->getOperand(0);
  // Input_weight is indexed at 1.
  Value* weight = call_op->getOperand(1);
  // Recurrent_weight is indexed at 2.
  Value* recurrent_weight = call_op->getOperand(2);
  // Bias is indexed at 3.
  Value* bias = call_op->getOperand(3);
  // Hidden_state is indexed at 4.
  Value* hidden_state = call_op->getOperand(4);

  // Build Output.
  auto output_type = call_op->getResult(0)->getType();

  // Currently, ophinted RNN only supports time_major = True.
  const bool time_major = true;
  // Activation will always be TanH.
  StringAttr fused_activation_function = builder->getStringAttr("TANH");

  builder->setInsertionPoint(call_op->getOperation());
  *fused_op = builder->create<TFL::UnidirectionalSequenceRNNOp>(
      call_op->getLoc(), output_type, input, weight, recurrent_weight, bias,
      hidden_state, builder->getBoolAttr(time_major),
      fused_activation_function);
  return success();
}

LogicalResult ConvertTfLiteFusedOpIfAvaiable(StringRef func_name,
                                             FuncOp composite_func_op,
                                             CallOp* call_op,
                                             OpBuilder* builder) {
  Operation* fused_op = nullptr;
  if (func_name == kUnidirectionalSequenceRnn) {
    // TODO(renjieliu): Validate the func op inputs.
    LogicalResult build_fused_op_result = BuildUnidirectionalSequenceRnnOp(
        composite_func_op, call_op, builder, &fused_op);
    if (failed(build_fused_op_result)) return build_fused_op_result;
  } else {  // If we support more fused op, we should add the conversion here.
    return failure();
  }

  call_op->replaceAllUsesWith(fused_op);

  // Delete call op.
  Operation* call = call_op->getOperation();
  call->dropAllDefinedValueUses();
  call->dropAllReferences();
  call->erase();
  return success();
}

LogicalResult ConvertCallOps(llvm::StringMap<FuncOp>* composite_func_ops,
                             ModuleOp* module) {
  for (auto func : module->getOps<FuncOp>()) {
    // Ideally it will be much simpler if we can just use walk, but we also
    // want to early return if encounter errors. :(
    OpBuilder builder(func.getBody());
    // The call_op replacement within this loop works like an in-place
    // replacement, so it should be safe to do so.
    for (auto call_op :
         llvm::make_early_inc_range(builder.getBlock()->getOps<CallOp>())) {
      auto it = composite_func_ops->find(call_op.getCallee());
      if (it == composite_func_ops->end()) return failure();

      // Replace the call op with TfLite fused op.
      // Currently it's only handled case by case, but ideally it would be
      // much better if we can do this automatically.
      FuncOp composite_func_op = it->second;
      StringRef func_name = composite_func_op.getAttr(kTfLiteFunctionName)
                                .cast<StringAttr>()
                                .getValue();
      if (failed(ConvertTfLiteFusedOpIfAvaiable(func_name, composite_func_op,
                                                &call_op, &builder)))
        return failure();

      composite_func_ops->erase(it);
      // Delete func op.
      Operation* func = composite_func_op.getOperation();
      func->erase();
    }
  }
  return success();
}

void LegalizeOphintFuncOpPass::runOnModule() {
  ModuleOp module = getModule();
  // Find all composite funcs, then for every call op inside every func op
  // within the module, we go ahead and replace the callop with the tflite
  // corresponding op and destroy the func op. This two-phase processing is
  // intended:
  //
  // Every func op is meant to be used exactly once.
  // Instead of finding the composite func then loop through the graph and
  // convert the call op immediately, we break finding & converting into two
  // phases. This changes the complexity from O(op_in_module *
  // function_in_module * attr_in_func) to O(op_in_module) * O(map_look_up) +
  // O(function_in_module * attr_in_func). O(op_in_module) is the dominant
  // factor here and map look up should be very cheap.
  llvm::StringMap<FuncOp> composite_func_ops = FindCompositeFuncOps(module);
  if (composite_func_ops.empty()) return;
  if (failed(ConvertCallOps(&composite_func_ops, &module))) {
    module.emitError() << "Legalize ophint: ConvertCallOps failed.";
    return signalPassFailure();
  }
}

}  // namespace

/// Creates an instance of the TensorFlow Lite dialect LegalizeOphintFuncOpPass
/// pass.
std::unique_ptr<ModulePassBase> CreateLegalizeOphintFuncOpPass() {
  return std::make_unique<LegalizeOphintFuncOpPass>();
}

static PassRegistration<LegalizeOphintFuncOpPass> pass(
    "tfl-legalize-ophint-func-op", "Convert composite op for TfLite dialect.");

}  // namespace TFL
}  // namespace mlir
