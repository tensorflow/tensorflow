/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <memory>
#include <vector>

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

#define DEBUG_TYPE "unfreeze-mutable-global-tensor"

namespace mlir {
namespace tf_saved_model {

namespace {

#define GEN_PASS_DEF_UNFREEZEMUTABLEGLOBALTENSORSPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_savedmodel_passes.h.inc"
struct UnfreezeMutableGlobalTensorsPass
    : public impl::UnfreezeMutableGlobalTensorsPassBase<
          UnfreezeMutableGlobalTensorsPass> {
  UnfreezeMutableGlobalTensorsPass() = default;
  UnfreezeMutableGlobalTensorsPass(const UnfreezeMutableGlobalTensorsPass&) {};
  void runOnOperation() override;
};

struct GlobalTensorAndUseIndex {
  GlobalTensorOp global_tensor;
  size_t arg_idx;
};

using GlobalTensorUsesMap =
    llvm::DenseMap<func::FuncOp, std::vector<GlobalTensorAndUseIndex>>;

GlobalTensorUsesMap CreateGlobalTensorUsesMap(ModuleOp module) {
  GlobalTensorUsesMap global_tensor_uses;

  SymbolTable symbol_table(module);
  for (auto func : module.getOps<func::FuncOp>()) {
    global_tensor_uses[func] = {};
    for (size_t i = 0, e = func.getNumArguments(); i < e; i++) {
      mlir::SymbolRefAttr sym =
          func.getArgAttrOfType<SymbolRefAttr>(i, "tf_saved_model.bound_input");
      if (!sym) {
        continue;
      }
      GlobalTensorOp global_tensor = symbol_table.lookup<GlobalTensorOp>(
          mlir::cast<FlatSymbolRefAttr>(sym).getValue());
      if (!global_tensor || !global_tensor.getIsMutable()) {
        continue;
      }
      global_tensor_uses[func].push_back({global_tensor, i});
    }
  }

  return global_tensor_uses;
}

// Adds the symbol to the "initializers" attribute of the session_initializer
// op.
void AddSymbolToInitializersAttr(SessionInitializerOp session_init_op,
                                 FlatSymbolRefAttr symbol) {
  const auto prev_initializers = session_init_op.getInitializersAttr();
  llvm::SmallVector<Attribute> initializers_attrs(prev_initializers.begin(),
                                                  prev_initializers.end());
  initializers_attrs.emplace_back(symbol);

  session_init_op.setInitializersAttr(
      ArrayAttr::get(session_init_op.getContext(), initializers_attrs));
}

// Returns the session_initializer op in the module if exists. Otherwise,
// creates a new session_initializer op and returns it.
SessionInitializerOp GetOrCreateSessionInitializerOp(ModuleOp module_op) {
  SessionInitializerOp session_init_op = GetSessionInitializerOp(module_op);

  // Create one if it doesn't exist.
  if (!session_init_op) {
    OpBuilder builder(&module_op.getBodyRegion());

    session_init_op = SessionInitializerOp::create(
        builder, module_op.getLoc(), /*initializers=*/builder.getArrayAttr({}));
  }

  return session_init_op;
}

// Create the initializer function right after the SessionInitializer op.
// Returns the newly created initializer function. The initializer function's
// initializer_type is set to "init_op" since it essentially serves as a
// variable initialization function.
func::FuncOp CreateInitializerFunc(ModuleOp module_op) {
  SessionInitializerOp session_init_op =
      GetOrCreateSessionInitializerOp(module_op);

  func::FuncOp init_func = GetInitializerFunction(
      module_op, /*initializer_type=*/kTfSavedModelInitializerInitType);
  if (init_func) {
    return init_func;
  }

  OpBuilder builder(module_op.getContext());
  builder.setInsertionPointAfter(session_init_op);

  const Location loc = builder.getUnknownLoc();
  const auto func_type = builder.getFunctionType(/*inputs=*/{}, /*results=*/{});

  init_func =
      func::FuncOp::create(builder, loc, /*sym_name=*/"NoOp", func_type);
  builder.createBlock(&init_func.getBody(), /*insertPt=*/init_func.begin(),
                      /*arg_types=*/{}, /*arg_locs=*/{});

  init_func->setAttr(
      kTfSavedModelExportedNamesAttr,
      builder.getStrArrayAttr({"__tf_saved_model_session_initializer_NoOp"}));
  init_func->setAttr(kTfSavedModelInitializerTypeAttr,
                     builder.getStringAttr(kTfSavedModelInitializerInitType));

  builder.setInsertionPointToStart(&init_func.front());
  func::ReturnOp::create(builder, loc, /*operands=*/ValueRange());

  SymbolTable symbol_table(module_op);
  symbol_table.insert(init_func);

  AddSymbolToInitializersAttr(
      session_init_op, FlatSymbolRefAttr::get(init_func.getSymNameAttr()));

  return init_func;
}

// Returns the initializer function whose tf_saved_model.initializer_type
// is "init_op". Creates and returns a new initializer function iff such
// `FuncOp` is not found. The newly created initializer function's
// initializer_type is "init_op" and its symbol will be added to the symbol
// table and session_initializer op's "intializer" attribute.
func::FuncOp GetOrCreateInitializerFunc(ModuleOp module_op) {
  if (auto init_func_op = GetInitializerFunction(
          module_op, /*initializer_type=*/kTfSavedModelInitializerInitType);
      init_func_op) {
    return init_func_op;
  } else {
    // Create a new initializer function if the init function is not found.
    return CreateInitializerFunc(module_op);
  }
}

// Collects the mutable GlobalTensorOps to unfreeze.
std::vector<GlobalTensorOp> GetTargetGlobalTensorOps(ModuleOp module_op) {
  std::vector<GlobalTensorOp> target_global_tensor_ops;
  for (auto global_tensor : module_op.getOps<GlobalTensorOp>()) {
    if (global_tensor.getIsMutable()) {
      target_global_tensor_ops.push_back(global_tensor);
    }
  }

  return target_global_tensor_ops;
}

// Inside `session_init_func`, creates AssignVariableOps(VarHandleOp, ConstOp)
// for each VarHandleOp that replaces a ConstOp. The `session_init_func` will
// essentially behave like init_op for the newly created VarHandleOps whose
// shared names are the values of `const_op_name_map`.
void CreateAssignVariableOps(
    const std::vector<GlobalTensorOp>& target_global_tensor_ops,
    func::FuncOp session_init_func) {
  OpBuilder builder(&session_init_func.getBody());

  for (auto global_tensor : target_global_tensor_ops) {
    llvm::StringRef shared_name = global_tensor.getSymName();
    const auto element_type = TF::ResourceType::get(
        {mlir::cast<mlir::TensorType>(global_tensor.getType())},
        builder.getContext());

    const auto ranked_tensor_type = RankedTensorType::get(
        /*shape=*/{}, /*elementType=*/element_type);
    auto var_handle_op =
        TF::VarHandleOp::create(builder, global_tensor.getLoc(),
                                /*resource=*/ranked_tensor_type,
                                /*container=*/"", shared_name);

    // Assign the ConstOp to each VarHandleOp. These will be used to save the
    // variable values to the checkpoint.
    auto const_op_copy = TF::ConstOp::create(builder, global_tensor.getLoc(),
                                             *global_tensor.getValue());

    TF::AssignVariableOp::create(builder, global_tensor.getLoc(),
                                 /*resource=*/var_handle_op,
                                 /*value=*/const_op_copy.getOutput());
  }
}

void UnfreezeMutableGlobalTensorsPass::runOnOperation() {
  ModuleOp module_op = getOperation();

  // Find the mutable GlobalTensorOps to "unfreeze" into VarHandleOps.
  const std::vector<GlobalTensorOp> target_global_tensor_ops =
      GetTargetGlobalTensorOps(module_op);

  if (target_global_tensor_ops.empty()) {
    return;
  }

  func::FuncOp session_init_func = GetOrCreateInitializerFunc(module_op);

  // In the session initializer function, assign the const op's values to the
  // corresponding VarHandleOps.
  CreateAssignVariableOps(target_global_tensor_ops, session_init_func);

  // Remove the references to the mutable GlobalTensorOps.
  GlobalTensorUsesMap global_tensor_uses = CreateGlobalTensorUsesMap(module_op);
  for (auto& [func, global_tensor_and_use_indices] : global_tensor_uses) {
    llvm::BitVector args_to_erase(func.getNumArguments());

    for (auto& [global_tensor, arg_idx] : global_tensor_and_use_indices) {
      args_to_erase.set(arg_idx);

      OpBuilder builder(func.getBody());
      builder.setInsertionPointToStart(&func.getBody().front());

      llvm::StringRef shared_name = global_tensor.getSymName();
      const auto element_type = TF::ResourceType::get(
          {mlir::cast<mlir::TensorType>(global_tensor.getType())},
          global_tensor.getContext());
      const auto ranked_tensor_type = RankedTensorType::get(
          /*shape=*/{}, /*elementType=*/element_type);

      auto var_handle_op =
          TF::VarHandleOp::create(builder, global_tensor.getLoc(),
                                  /*resource=*/ranked_tensor_type,
                                  /*container=*/"", shared_name);

      auto arg = func.getArguments()[arg_idx];
      arg.replaceAllUsesWith(var_handle_op->getResults()[0]);
    }

    if (failed(func.eraseArguments(args_to_erase))) {
      return signalPassFailure();
    }
  }

  // Erase the mutable GlobalTensorOps that are replaced by VarHandleOps.
  for (auto global_tensor_op : target_global_tensor_ops) {
    global_tensor_op.erase();
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateUnfreezeMutableGlobalTensorsPass() {
  return std::make_unique<UnfreezeMutableGlobalTensorsPass>();
}

}  // namespace tf_saved_model
}  // namespace mlir
