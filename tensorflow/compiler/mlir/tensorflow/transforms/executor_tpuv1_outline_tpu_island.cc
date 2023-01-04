/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"

namespace mlir {
namespace tf_executor {

namespace {
constexpr llvm::StringRef kNestedModule = "_tpu_v1_compat_outlined";
constexpr llvm::StringRef kOutlinedFuncPrefix = "_tpu_v1_compat_outlined_func";

#define GEN_PASS_DEF_TPUBRIDGEEXECUTORISLANDOUTLININGPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

// Extract the islands containing a TPU cluster computation into an outlined
// function in a nested module. This will allow to run the usual bridge on this
// nested module which exhibit a more friendly "V2-like" structure.
// This is only intended for V1 compatibility mode where the bridge runs without
// feed/fetches on session create/extend.
struct TPUBridgeExecutorIslandOutlining
    : public impl::TPUBridgeExecutorIslandOutliningPassBase<
          TPUBridgeExecutorIslandOutlining> {
  void runOnOperation() override;
};

// Move FuncOp referenced by `symbol_ref` from one symbol table to another.
void MoveFuncOp(FlatSymbolRefAttr &symbol_ref, SymbolTable &from,
                SymbolTable &to) {
  if (to.lookup<func::FuncOp>(symbol_ref.getValue())) return;
  func::FuncOp callee = from.lookup<func::FuncOp>(symbol_ref.getValue());
  callee.getOperation()->getBlock()->getOperations().remove(
      callee.getOperation());
  to.insert(callee);
}

void TPUBridgeExecutorIslandOutlining::runOnOperation() {
  MLIRContext *ctx = &getContext();

  SymbolTable symbol_table(getOperation());
  if (Operation *nested_module = symbol_table.lookup(kNestedModule)) {
    nested_module->emitOpError("unexpected already present outlined module.");
    return signalPassFailure();
  }
  ModuleOp outlined_module = ModuleOp::create(getOperation().getLoc());
  outlined_module->setAttrs(getOperation()->getAttrDictionary());
  outlined_module->setAttr(SymbolTable::getSymbolAttrName(),
                           StringAttr::get(ctx, kNestedModule));
  symbol_table.insert(outlined_module);
  SymbolTable outlined_symbol_table(outlined_module);

  // Find every island that contains a TPU node and extract it into a new module
  // to run the V1 bridge there.
  llvm::SmallVector<IslandOp, 8> islands_to_outline;
  getOperation().walk([&](IslandOp island_op) {
    auto parent_func = island_op->getParentOfType<func::FuncOp>();
    auto skip_island_outlining =
        parent_func->getAttrOfType<BoolAttr>(mlir::TF::kSkipIslandOutlining);
    if (skip_island_outlining && skip_island_outlining.getValue()) {
      // Island was marked to be skipped.
      return WalkResult::advance();
    }
    for (Operation &op : island_op.GetBody().without_terminator()) {
      if (isa<TF::TPUReplicateMetadataOp>(&op)) {
        // Handle replicated TPU case.
        islands_to_outline.push_back(island_op);
        break;
      }
      auto device_type =
          op.getAttrOfType<StringAttr>(TF::kCompileDeviceTypeAttr);
      if (device_type && device_type.getValue() == TF::kTpuDevice &&
          !op.hasAttrOfType<StringAttr>(TF::kReplicationInfoAttr)) {
        // Handle single-core TPU case (no `TPUReplicateMetadataOp`).
        islands_to_outline.push_back(island_op);
        break;
      }
    }
    return WalkResult::advance();
  });
  int prefix_id = 0;
  for (IslandOp island_op : islands_to_outline) {
    // Build the function signature.

    // First the captured values in the island are function arguments
    llvm::SetVector<Value> operands;
    getUsedValuesDefinedAbove(island_op.getBody(), operands);

    SmallVector<Type, 16> func_operand_types;
    func_operand_types.reserve(operands.size());
    for (Value operand : operands)
      func_operand_types.push_back(operand.getType());

    // Function results are the yield operands
    SmallVector<Type, 16> func_result_types;
    for (Value operand : island_op.GetYield().getOperands())
      func_result_types.push_back(operand.getType());
    FunctionType func_type =
        FunctionType::get(ctx, func_operand_types, func_result_types);

    // Create the outlined function
    SmallString<32> name = kOutlinedFuncPrefix;
    name += llvm::Twine(prefix_id++).str();
    auto outlined_func = OpBuilder(ctx).create<func::FuncOp>(island_op.getLoc(),
                                                             name, func_type);
    outlined_symbol_table.insert(outlined_func);
    outlined_func.setNested();

    // We will "steal" the body of the island and replace it with a call to the
    // new function later.
    {
      YieldOp yield_op = island_op.GetYield();
      outlined_func.getBody().takeBody(island_op.getBody());

      // Replace the yield with a return
      OpBuilder replacer(yield_op);
      island_op.getBody().push_back(new Block);
      replacer.create<mlir::func::ReturnOp>(yield_op.getLoc(),
                                            yield_op.getOperands());
      yield_op.erase();
    }

    // Remap the captured operands in the (former) island block with newly
    // created entry block arguments in the function body.
    {
      Block &entry_block = outlined_func.getBody().front();
      auto loc = outlined_func.getLoc();
      for (Value operand : operands) {
        BlockArgument newArg = entry_block.addArgument(operand.getType(), loc);
        replaceAllUsesInRegionWith(operand, newArg, outlined_func.getBody());
      }
    }

    // The function is in place in the nested module, create a call and yield in
    // the original island.
    OpBuilder builder = OpBuilder::atBlockEnd(&island_op.GetBody());
    auto call_op = builder.create<mlir::TF::PartitionedCallOp>(
        island_op.getLoc(), func_result_types, operands.getArrayRef(),
        SymbolRefAttr::get(
            builder.getContext(), kNestedModule,
            SymbolRefAttr::get(builder.getContext(), outlined_func.getName())),
        /*config=*/builder.getStringAttr(""),
        /*config_proto=*/builder.getStringAttr(""),
        /*executor_type=*/builder.getStringAttr(""));
    SmallVector<Value, 16> yield_operands(call_op.getResults());
    builder.create<YieldOp>(island_op.getLoc(), yield_operands);
  }

  // Outline all the transitively called functions by moving them in the
  // outlined module.
  for (func::FuncOp func : outlined_module.getOps<func::FuncOp>()) {
    func.walk([&](Operation *op) {
      for (NamedAttribute attr : op->getAttrs()) {
        if (auto symbol_ref = attr.getValue().dyn_cast<FlatSymbolRefAttr>()) {
          MoveFuncOp(symbol_ref, symbol_table, outlined_symbol_table);
          continue;
        }
        if (auto array_attr = attr.getValue().dyn_cast<ArrayAttr>()) {
          for (const Attribute &attribute : array_attr) {
            auto symbol_ref = attribute.dyn_cast<FlatSymbolRefAttr>();
            if (!symbol_ref) continue;
            MoveFuncOp(symbol_ref, symbol_table, outlined_symbol_table);
          }
        }
      }
    });
  }
  // Remove `kSkipIslandOutlining` attributes.
  for (func::FuncOp func_op : getOperation().getOps<func::FuncOp>()) {
    if (func_op->hasAttr(mlir::TF::kSkipIslandOutlining)) {
      func_op->removeAttr(mlir::TF::kSkipIslandOutlining);
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTFExecutorTPUV1IslandOutliningPass() {
  return std::make_unique<TPUBridgeExecutorIslandOutlining>();
}

}  // namespace tf_executor
}  // namespace mlir
