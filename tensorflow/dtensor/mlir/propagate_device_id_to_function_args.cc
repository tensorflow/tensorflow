/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/dtensor/mlir/device_utils.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes_classes.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {
namespace {

// Holds information on functions to rewrite. `function` is the function
// definition or function that needs to be updated and `callsite_ops` holds a
// list of ops that calls the `function`.
struct FunctionToChangeInfo {
  mlir::func::FuncOp function;
  llvm::SmallVector<mlir::Operation*, 4> callsite_ops;
};

// Finds all functions in graph that is not a public functions and retrieves
// their callsite operations.
llvm::SmallVector<FunctionToChangeInfo, 4> FindFunctionsToRewrite(
    mlir::ModuleOp module) {
  llvm::SmallVector<FunctionToChangeInfo, 4> functions_to_change;
  module.walk([&](mlir::Operation* op) {
    if (!llvm::isa<mlir::TF::StatefulPartitionedCallOp,
                   mlir::TF::PartitionedCallOp>(op))
      return;

    // Extract function symbol from PartitionedCall or StatefulPartitionedCall
    // op.
    llvm::StringRef symbol;
    if (auto call_op =
            llvm::dyn_cast<mlir::TF::StatefulPartitionedCallOp>(op)) {
      symbol = call_op.f();
    } else {
      auto symbol_ref = llvm::dyn_cast<mlir::TF::PartitionedCallOp>(op).f();
      if (!symbol_ref.isa<mlir::FlatSymbolRefAttr>()) return;
      symbol = symbol_ref.getRootReference().getValue();
    }

    // If function definition could be found, then extract all function usages.
    auto function = MaybeFindFunction(op);
    if (!function || function->isPublic()) return;

    auto function_uses = mlir::SymbolTable::getSymbolUses(
        mlir::StringAttr::get(module.getContext(), symbol),
        &module.getBodyRegion());
    if (!function_uses) return;

    llvm::SmallVector<mlir::Operation*, 4> function_use_ops;
    for (auto function_use : *function_uses)
      function_use_ops.emplace_back(function_use.getUser());

    functions_to_change.emplace_back(
        FunctionToChangeInfo{function.value(), function_use_ops});
  });

  return functions_to_change;
}

// Rewrites function such that 0th argument of type `type` is added to
// `function`.
void PrependArgumentToFunction(mlir::func::FuncOp function, mlir::Type type,
                               mlir::OpBuilder* builder) {
  auto& function_body = function.front();
  function_body.insertArgument(static_cast<unsigned>(0), type,
                               function.getLoc());
  auto new_argument_types =
      llvm::to_vector<4>(function_body.getArgumentTypes());
  function.setType(
      mlir::FunctionType::get(builder->getContext(), new_argument_types,
                              function.getFunctionType().getResults()));
}

// Rewrites function callsites ops. As function signatures are already updated,
// simply add 0th argument of the parent function to 0th operand of the callsite
// operation.
mlir::LogicalResult PrependDeviceIdToCallsites(mlir::OpBuilder* builder,
                                               mlir::Operation* op) {
  auto device_id_or_status = DeviceId(op);
  if (!device_id_or_status.ok())
    return op->emitOpError(
        "Failed during PropagateDeviceIdToFunctionArgs pass. All functions "
        "must have device id as 0th argument.");

  auto new_operands = llvm::to_vector<4>(op->getOperands());
  new_operands.insert(new_operands.begin(), device_id_or_status.ValueOrDie());

  builder->setInsertionPoint(op);
  mlir::Operation* new_call = nullptr;
  if (auto stateful_partitioned_call =
          llvm::dyn_cast<mlir::TF::StatefulPartitionedCallOp>(op)) {
    new_call = builder->create<mlir::TF::StatefulPartitionedCallOp>(
        op->getLoc(), op->getResultTypes(), new_operands,
        stateful_partitioned_call.f(), stateful_partitioned_call.config(),
        stateful_partitioned_call.config_proto(),
        stateful_partitioned_call.executor_type());
  } else {
    auto partitioned_call = llvm::cast<mlir::TF::PartitionedCallOp>(op);
    new_call = builder->create<mlir::TF::PartitionedCallOp>(
        op->getLoc(), op->getResultTypes(), new_operands, partitioned_call.f(),
        partitioned_call.config(), partitioned_call.config_proto(),
        partitioned_call.executor_type());
  }

  for (auto results : llvm::zip(op->getResults(), new_call->getResults()))
    std::get<0>(results).replaceAllUsesWith(std::get<1>(results));

  op->erase();

  return mlir::success();
}

// Pass that rewrites the functions in graph so that 0th argument of the main
// function (i.e. device_id) is present on all functions in the graph.
struct DTensorPropagateDeviceIdToFunctionArgs
    : public DTensorPropagateDeviceIdToFunctionArgsBase<
          DTensorPropagateDeviceIdToFunctionArgs> {
  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    auto module = getOperation();
    mlir::OpBuilder builder(&context);

    // Extracts device id argument from main function.
    mlir::func::FuncOp main_func =
        module.lookupSymbol<mlir::func::FuncOp>("main");
    auto device_id_or_status = DeviceId(&main_func.getBody().front().front());
    if (!device_id_or_status.ok()) {
      main_func.emitOpError(
          "Error in PropagateDeviceIdToFunctionArgs pass. Main function must "
          "have device id as 0th function argument.");
      return signalPassFailure();
    }
    auto device_id_from_main_function = device_id_or_status.ValueOrDie();
    // First iterate through all functions to rewrite and update the signatures
    // first.
    const auto functions_to_update = FindFunctionsToRewrite(module);
    for (const auto& function_to_update : functions_to_update)
      PrependArgumentToFunction(function_to_update.function,
                                device_id_from_main_function.getType(),
                                &builder);

    // Once all function signatures are updated, rewrite the callsite ops.
    for (const auto& function_to_update : functions_to_update) {
      for (auto call_site_op : function_to_update.callsite_ops) {
        if (mlir::failed(PrependDeviceIdToCallsites(&builder, call_site_op)))
          return signalPassFailure();
      }
    }
  };
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorPropagateDeviceIdToFunctionArgs() {
  return std::make_unique<DTensorPropagateDeviceIdToFunctionArgs>();
}

}  // namespace dtensor
}  // namespace tensorflow
