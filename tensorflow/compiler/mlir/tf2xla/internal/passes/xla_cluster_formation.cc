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

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/call_graph_util.h"
#include "tensorflow/core/common_runtime/inline_function_utils.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

using mlir::Block;
using mlir::CallInterfaceCallable;
using mlir::CallOpInterface;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationPass;
using mlir::SymbolTable;
using mlir::SymbolTableCollection;
using mlir::SymbolUserOpInterface;
using mlir::func::FuncOp;

#define GEN_PASS_DEF_XLACLUSTERFORMATIONPASS
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/clustering_passes.h.inc"

// Outlines partitioned call ops with `_XlaMustCompile` to device clusters.
struct XlaClusterFormationPass
    : public impl::XlaClusterFormationPassBase<XlaClusterFormationPass> {
  void runOnOperation() override;
};

void EncapsulatePartitionedCall(Operation *call_op,
                                mlir::StringAttr callee_name) {
  OpBuilder builder(call_op);
  auto cluster = builder.create<mlir::tf_device::ClusterOp>(
      call_op->getLoc(), call_op->getResultTypes());
  cluster.getBody().push_back(new Block);
  call_op->replaceAllUsesWith(cluster.getResults());
  call_op->moveBefore(&cluster.GetBody(), cluster.GetBody().end());
  builder.setInsertionPointToEnd(&cluster.GetBody());
  builder.create<mlir::tf_device::ReturnOp>(call_op->getLoc(),
                                            call_op->getResults());
  // Propagate necessary attributes to the cluster so that when it's outlined,
  // the function will have correct attributes.
  mlir::TF::CopyDeviceAndUnderscoredAttributes(call_op, cluster);
  cluster->setAttr(mlir::TF::kClusterOutlinedFunctionNameAttr, callee_name);
  cluster->setAttr(mlir::TF::kAllowSoftPlacementAttr,
                   builder.getBoolAttr(true));
}

// Encapsulate the first partitioned call that can be reached from
// `func` and is with compilation markers in a device cluster. For nested calls,
// if the outermost one has the markers, encapsulates the outermost call and
// returns. Otherwise, we'll keep going through inner calls until we found one.
mlir::LogicalResult EncapsulateFirstXlaCompilablePartitionedCalls(
    FuncOp func, SymbolTableCollection &symbol_table_collection,
    SymbolTable &symtab) {
  auto has_no_compile_device_type = [](SymbolUserOpInterface op) {
    return !op->hasAttr(mlir::TF::kCompileDeviceTypeAttr);
  };

  mlir::OpBuilder builder(func.getContext());
  auto noinline_attr_name = absl::StrCat("tf.", tensorflow::kNoInlineAttr);
  llvm::SmallVector<SymbolUserOpInterface> noinline_pcall_ops,
      outermost_pcall_ops;
  if (mlir::failed(
          mlir::GetOpsOfTypeUntilMiss<mlir::TF::StatefulPartitionedCallOp,
                                      mlir::TF::PartitionedCallOp>(
              func, symtab, /*predicate*/ has_no_compile_device_type,
              /*hits*/ noinline_pcall_ops,
              /*first_misses*/ outermost_pcall_ops))) {
    return mlir::failure();
  }
  // Cluster outermost partitioned calls with _xla_compile_device_type
  // attribute.
  for (auto &pcall_op : outermost_pcall_ops) {
    auto call = llvm::cast<CallOpInterface>(pcall_op.getOperation());
    CallInterfaceCallable callable = call.getCallableForCallee();
    auto sym = callable.get<mlir::SymbolRefAttr>();
    EncapsulatePartitionedCall(pcall_op, sym.getRootReference());
  }
  // Partitioned calls are executed asynchronous. The calls outside of
  // device clusters therefore should not be inlined to perserve run-time
  // performance.
  for (auto &pcall_op : noinline_pcall_ops) {
    auto call = llvm::cast<CallOpInterface>(pcall_op.getOperation());
    auto callee = llvm::cast<FuncOp>(
        call.resolveCallableInTable(&symbol_table_collection));
    callee->setAttr(noinline_attr_name, builder.getBoolAttr(true));
  }
  return mlir::success();
}

void XlaClusterFormationPass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTableCollection symbol_table_collection;
  SymbolTable symtab = symbol_table_collection.getSymbolTable(module);
  llvm::SmallVector<FuncOp> entry_funcs = GetEntryFunctions(module);
  for (auto &entry_func : entry_funcs) {
    if (EncapsulateFirstXlaCompilablePartitionedCalls(
            entry_func, symbol_table_collection, symtab)
            .failed()) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<ModuleOp>> CreateXlaClusterFormationPass() {
  return std::make_unique<XlaClusterFormationPass>();
}

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
