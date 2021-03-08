/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/ADT/None.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace TFL {
namespace {
constexpr char kTfSavedModelSessionInitNameAttr[] =
    "__tf_saved_model_session_initializer";
constexpr char kTfSavedModelExportedNameAttr[] =
    "tf_saved_model.exported_names";

// Returns Value representing the resource_id.
Value GetResourceIDAsI32(int resource_id, Location loc,
                         mlir::OpBuilder& rewriter) {
  return rewriter.create<ConstOp>(
      loc, DenseElementsAttr::get(
               RankedTensorType::get({1}, rewriter.getIntegerType(32)),
               resource_id));
}

// Helper method that fetches the global tensor that 'op' points to it.
template <typename T>
tf_saved_model::GlobalTensorOp GetGlobalTensor(const SymbolTable& symbol_table,
                                               T op, FuncOp func) {
  auto block_arg = op.resource().template dyn_cast<BlockArgument>();
  if (!block_arg) return nullptr;
  int index = block_arg.getArgNumber();
  auto sym = func.template getArgAttrOfType<FlatSymbolRefAttr>(
      index, "tf_saved_model.bound_input");
  if (!sym) {
    return nullptr;
  }
  return symbol_table.lookup<tf_saved_model::GlobalTensorOp>(sym.getValue());
}

// Pass which Initializes TF variables which are already passed as bounded
// arguments to functions, to a TFLite variables.
class InitializeVariablesPass
    : public PassWrapper<InitializeVariablesPass, OperationPass<ModuleOp>> {
 public:
  InitializeVariablesPass() = default;
  InitializeVariablesPass(const InitializeVariablesPass&) {}

  // Initializes a single variable identified by 'var_id' with value 'value'
  // in 'session_init' function.
  void InitializeVariable(int var_id, ElementsAttr value, FuncOp session_init) {
    // TODO(b/149099381): Initialize using TF::AssignVariableOp instead
    // and let legalization be handled by Legalize variables pass.
    mlir::OpBuilder builder(&getContext());
    builder.setInsertionPoint(&session_init.getBlocks().front().front());
    auto resource_op =
        GetResourceIDAsI32(var_id, session_init.body().getLoc(), builder);
    auto value_op =
        builder.create<ConstOp>(session_init.body().getLoc(), value);
    builder.create<TFL::AssignVariableOp>(session_init.body().getLoc(),
                                          resource_op, value_op);
  }

  tf_saved_model::GlobalTensorOp GetGlobalTensorOp(mlir::Operation* op,
                                                   SymbolTable symbol_table,
                                                   FuncOp func) {
    if (auto read_var = llvm::dyn_cast_or_null<TF::ReadVariableOp>(op))
      return GetGlobalTensor<TF::ReadVariableOp>(symbol_table, read_var, func);
    else if (auto assign_var = llvm::dyn_cast_or_null<TF::AssignVariableOp>(op))
      return GetGlobalTensor<TF::AssignVariableOp>(symbol_table, assign_var,
                                                   func);
    return nullptr;
  }

  // Initializes all variables in the module.
  void InitializeVariables(const std::map<std::string, int>& global_tensor_id,
                           SymbolTable symbol_table) {
    auto module = getOperation();
    // Check if there is Session init func already, if not create one.
    FuncOp session_init_func = nullptr;
    for (auto func : module.getOps<FuncOp>()) {
      if (auto attr = func->getAttr(kTfSavedModelExportedNameAttr)) {
        auto exported_names = attr.dyn_cast<ArrayAttr>();
        if (!exported_names) continue;
        for (auto exported_name : exported_names) {
          if (auto name = exported_name.dyn_cast_or_null<StringAttr>())
            if (name.getValue() == kTfSavedModelSessionInitNameAttr)
              session_init_func = func;
        }
        if (session_init_func) break;
      }
    }
    // TODO(b/149099381): Refactor to separate function in saved model util.
    if (!session_init_func) session_init_func = CreateSessionInitFunc();

    std::set<tf_saved_model::GlobalTensorOp> tensors_to_initialize;
    for (auto func : module.getOps<FuncOp>()) {
      func->walk([&](Operation* op) {
        // TODO(b/149099381): Make sure to verify flex compatability
        // with ops that accepts resource as input.
        if (!llvm::isa<TF::ReadVariableOp, TF::AssignVariableOp>(op))
          return WalkResult::advance();
        tensors_to_initialize.insert(GetGlobalTensorOp(op, symbol_table, func));
        return WalkResult::advance();
      });
    }
    for (auto global_tensor : tensors_to_initialize) {
      InitializeVariable(global_tensor_id.at(global_tensor.sym_name().str()),
                         global_tensor.value(), session_init_func);
    }
  }
  // Create a new function in the module which is SessionInitializerOp.
  FuncOp CreateSessionInitFunc() {
    constexpr char kSessionInitFuncName[] = "SessionInitializerFunction";
    auto module = getOperation();

    mlir::OpBuilder builder(module.body());
    auto func_type = FunctionType::get(&getContext(), {}, {});
    auto func = builder.create<FuncOp>(module->getLoc(), kSessionInitFuncName,
                                       func_type);
    func->setAttr(kTfSavedModelExportedNameAttr,
                  builder.getStrArrayAttr({kSessionInitFuncName}));
    func.setVisibility(mlir::FuncOp::Visibility::Public);
    auto funcBuilder = OpBuilder::atBlockBegin(func.addEntryBlock());
    funcBuilder.create<mlir::ReturnOp>(func.getLoc());
    builder.create<tf_saved_model::SessionInitializerOp>(
        module->getLoc(),
        builder.getArrayAttr(builder.getSymbolRefAttr(kSessionInitFuncName)));
    return func;
  }

  void runOnOperation() override {
    auto module = getOperation();
    // Use ordered container to make sure ids are deterministic if we got tensor
    // ids from different part, since we have different passes that touches
    // variables.
    // TODO(b/149099381): Remove integer IDs after adding the new variable
    // handle type.
    std::map<std::string, int> global_tensor_id;
    int id = 0;
    for (auto global_tensor : module.getOps<tf_saved_model::GlobalTensorOp>()) {
      global_tensor_id[global_tensor.sym_name().str()];
    }
    for (auto& tensor : global_tensor_id) tensor.second = id++;
    SymbolTable symbol_table(module);

    // Initialize all variables.
    InitializeVariables(global_tensor_id, symbol_table);
  }
};
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateInitializeVariablesPass() {
  return std::make_unique<InitializeVariablesPass>();
}

static PassRegistration<InitializeVariablesPass> pass(
    "tfl-initialize-variables-tf",
    "Initialize TensorFlow variables to TensorFlow Lite dialect");

}  // namespace TFL
}  // namespace mlir
