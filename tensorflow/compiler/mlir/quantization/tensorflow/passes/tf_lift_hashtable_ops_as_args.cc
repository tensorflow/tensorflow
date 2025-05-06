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
#include <memory>
#include <utility>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/constants.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/manipulate_model_attr.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"

namespace mlir {
namespace tf_quant {
namespace {

constexpr StringRef kSharedNameAttr = "shared_name";

class TFLiftHashTableOpsAsArgsPass
    : public PassWrapper<TFLiftHashTableOpsAsArgsPass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TFLiftHashTableOpsAsArgsPass)
  explicit TFLiftHashTableOpsAsArgsPass() = default;

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tf-quant-lift-hashtable-ops-as-args";
  }
  StringRef getDescription() const final {
    return "Lifts HashTable ops as function arguments.";
  }

  void runOnOperation() override;
};

// Checks if the given op is a Hashtable op.
bool IsHashTableOp(Operation* op) {
  return llvm::isa<TF::HashTableOp, TF::HashTableV2Op,
                   TF::MutableHashTableV2Op>(op);
}

// Checks if the function is the main or initializer function.
bool IsMainOrInitializerFunction(ModuleOp module, func::FuncOp func) {
  if (func.getSymName() ==
          llvm::StringRef(tensorflow::kImportModelDefaultGraphFuncName) ||
      func.getSymName() == quant::kTfQuantSaveFuncName) {
    return true;
  }

  for (func::FuncOp init_func :
       tf_saved_model::GetInitializerFunctions(module)) {
    if (func.getSymName() == init_func.getSymName()) {
      return true;
    }
  }
  return false;
}

// Checks if the function is only used by supported ops. Returns false when the
// function has no uses. Currently, only PartitionedCall is supported.
// TODO(b/284222309): Support lifting for functions called by control flow.
bool UsedBySupportedOps(ModuleOp module, func::FuncOp func) {
  auto function_uses =
      SymbolTable::getSymbolUses(func, &module.getBodyRegion());
  if (!function_uses.has_value()) return false;
  for (auto& function_use : function_uses.value()) {
    if (!llvm::isa<TF::PartitionedCallOp, TF::StatefulPartitionedCallOp>(
            function_use.getUser())) {
      return false;
    }
  }
  return true;
}

// Returns the `shared_name` attribute value if exists. If not, returns an
// empty string.
StringRef GetSharedName(Operation* op) {
  if (!op->hasAttrOfType<StringAttr>(kSharedNameAttr)) return "";
  return op->getAttrOfType<StringAttr>(kSharedNameAttr).getValue();
}

// Checks if the HashTable is initialized. This function assumes that the
// HashTable is initialized if it appears in the initializer since it can't
// check the actual value.
bool IsResourceInitialized(ModuleOp module_op, Operation* hash_table) {
  StringRef shared_name = GetSharedName(hash_table);
  if (shared_name.empty()) return false;

  for (func::FuncOp init_func_op :
       tf_saved_model::GetInitializerFunctions(module_op)) {
    for (Operation& op : init_func_op.getBody().getOps()) {
      StringRef other_shared_name = GetSharedName(&op);
      if (IsHashTableOp(&op) && other_shared_name == shared_name) {
        return true;
      }
    }
  }
  return false;
}

// Lifts HashTable ops in the target function as function arguments and returns
// the lifted ops. These ops  will then be added to the caller function and
// passed to the target function.
LogicalResult LiftHashTableOpsToArguments(ModuleOp module_op,
                                          func::FuncOp target_func) {
  if (!llvm::hasSingleElement(target_func)) return success();
  if (!UsedBySupportedOps(module_op, target_func)) return success();
  if (IsMainOrInitializerFunction(module_op, target_func)) return success();

  llvm::StringMap<int> shared_name_to_arg_idx;
  llvm::SmallVector<std::pair<Operation*, int>> lifted_op_and_arg_idx;
  Block& block = target_func.front();
  auto func_type = target_func.getFunctionType();

  for (Operation& op : block.without_terminator()) {
    StringRef shared_name = GetSharedName(&op);
    if (shared_name.empty() || !IsHashTableOp(&op)) continue;
    if (!IsResourceInitialized(module_op, &op)) continue;

    auto it =
        shared_name_to_arg_idx.insert({shared_name, block.getNumArguments()});
    if (it.second) {
      auto resource_type = op.getResult(0).getType();
      op.getResult(0).replaceAllUsesWith(
          block.addArgument(resource_type, op.getLoc()));
      quant::AddEntryFunctionInput(
          absl::StrCat("hash_table_", it.first->getValue(), ":0"), target_func);
      // Avoid deleting the op here, clone it to the caller function first.
      lifted_op_and_arg_idx.emplace_back(&op, it.first->getValue());
    } else {
      op.getResult(0).replaceAllUsesWith(
          block.getArgument(it.first->getValue()));
      op.erase();
    }
  }
  if (lifted_op_and_arg_idx.empty()) return success();

  // Update the function signature as well as its uses.
  target_func.setType(FunctionType::get(target_func.getContext(),
                                        block.getArgumentTypes(),
                                        func_type.getResults()));

  IRMapping mapping;
  OpBuilder builder(module_op);
  OpBuilder::InsertionGuard g(builder);
  // The function has been checked to have at least one use.
  auto function_uses =
      SymbolTable::getSymbolUses(target_func, &module_op.getBodyRegion());
  for (auto& function_use : function_uses.value()) {
    auto call_op = function_use.getUser();
    auto caller_func = call_op->getParentOfType<func::FuncOp>();
    if (!caller_func) return failure();

    builder.setInsertionPoint(call_op);
    for (auto [lifted_op, arg_idx] : lifted_op_and_arg_idx) {
      auto new_op = builder.clone(*lifted_op, mapping);
      call_op->insertOperands(arg_idx, new_op->getResult(0));
    }

    // Try to lift recursively until the main function.
    if (failed(LiftHashTableOpsToArguments(module_op, caller_func))) {
      return failure();
    }
  }

  // Erase the lifted operations explicitly.
  for (auto [lifted_op, arg_idx] : lifted_op_and_arg_idx) {
    lifted_op->erase();
  }

  return success();
}

void TFLiftHashTableOpsAsArgsPass::runOnOperation() {
  auto module_op = getOperation();

  for (auto func_op : module_op.getOps<func::FuncOp>()) {
    if (failed(LiftHashTableOpsToArguments(module_op, func_op))) {
      signalPassFailure();
      return;
    }
  }
}

static PassRegistration<TFLiftHashTableOpsAsArgsPass> pass;

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTFLiftHashTableOpsAsArgsPass() {
  return std::make_unique<TFLiftHashTableOpsAsArgsPass>();
}

}  // namespace tf_quant
}  // namespace mlir
