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

#include <functional>
#include <stack>
#include <vector>

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/call_graph_util.h"
#include "tensorflow/core/common_runtime/inline_function_utils.h"

namespace mlir {

namespace {

#define GEN_PASS_DEF_XLACLUSTERFORMATIONPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_device_passes.h.inc"

constexpr char kAllowSoftPlacementAttr[] = "allow_soft_placement";

// Outlines partitioned call ops with `_XlaMustCompile` to device clusters.
struct XlaClusterFormationPass
    : public impl::XlaClusterFormationPassBase<XlaClusterFormationPass> {
  void runOnOperation() override;
};

void CopyAttribute(const llvm::StringRef attr, Operation *src,
                   Operation *dest) {
  if (src->hasAttr(attr)) {
    dest->setAttr(attr, src->getAttr(attr));
  }
}

std::string getClusterOutlinedFunctionName(func::FuncOp func) {
  return func.getSymName().str() + "_cluster_func";
}

void AddClusterAttributes(OpBuilder &builder, func::FuncOp entry_func,
                          tf_device::ClusterOp cluster) {
  TF::CopyDeviceAndUnderscoredAttributes(entry_func, cluster);
  CopyAttribute(kAllowSoftPlacementAttr, entry_func, cluster);
  cluster->setAttr(
      TF::kClusterOutlinedFunctionNameAttr,
      builder.getStringAttr(getClusterOutlinedFunctionName(entry_func)));
}

// Wrap the body of `func` in a device cluster. `func` must have a single
// region and a single block.
LogicalResult EncapsulateEntryFunctionBody(func::FuncOp entry_func) {
  // We've verified the input graph has single-entry and single-block entry
  // functions. This is just in case passes in the pipeline uninteionally break
  // the assumption, and not expected to happen in practice.
  if (!HasSingleBlock(entry_func)) {
    entry_func->emitError() << "TF2XLA MLIR CPU/GPU MLIR phase 1 bridge "
                               "expects single region and single "
                               "block in an entry function.";
    return failure();
  }
  std::vector<Operation *> ops_without_terminator;
  for (auto &op : entry_func.front().without_terminator()) {
    ops_without_terminator.push_back(&op);
  }
  Operation *original_return_op = entry_func.front().getTerminator();
  OpBuilder builder(entry_func.getContext());
  builder.setInsertionPointToEnd(&entry_func.front());
  auto cluster = builder.create<tf_device::ClusterOp>(
      entry_func.getLoc(), entry_func.getResultTypes());
  cluster.getBody().push_back(new Block);
  for (auto &op : ops_without_terminator) {
    op->moveBefore(&cluster.GetBody(), cluster.GetBody().end());
  }
  builder.setInsertionPointToEnd(&cluster.GetBody());
  builder.create<tf_device::ReturnOp>(original_return_op->getLoc(),
                                      original_return_op->getResultTypes(),
                                      original_return_op->getOperands());
  original_return_op->erase();
  builder.setInsertionPointToEnd(&entry_func.front());
  builder.create<func::ReturnOp>(entry_func->getLoc(), cluster->getResults());
  AddClusterAttributes(builder, entry_func, cluster);
  return success();
}

void EncapsulatePartitionedCall(Operation *call_op, StringAttr callee_name) {
  OpBuilder builder(call_op);
  auto cluster = builder.create<tf_device::ClusterOp>(
      call_op->getLoc(), call_op->getResultTypes());
  cluster.getBody().push_back(new Block);
  call_op->replaceAllUsesWith(cluster.getResults());
  call_op->moveBefore(&cluster.GetBody(), cluster.GetBody().end());
  builder.setInsertionPointToEnd(&cluster.GetBody());
  builder.create<tf_device::ReturnOp>(call_op->getLoc(), call_op->getResults());
  // Propagate necessary attributes to the cluster so that when it's outlined,
  // the function will have correct attributes.
  TF::CopyDeviceAndUnderscoredAttributes(call_op, cluster);
  cluster->setAttr(TF::kClusterOutlinedFunctionNameAttr, callee_name);
  cluster->setAttr(kAllowSoftPlacementAttr, builder.getBoolAttr(true));
}

// Encapsulate the first partitioned call that can be reached from
// `func` and is with compilation markers in a device cluster. For nested calls,
// if the outermost one has the markers, encapsulates the outermost call and
// returns. Otherwise, we'll keep going through inner calls until we found one.
LogicalResult EncapsulateFirstXlaCompilablePartitionedCalls(
    func::FuncOp func, SymbolTableCollection &symbol_table_collection,
    SymbolTable &symtab) {
  auto has_no_compile_device_type = [](SymbolUserOpInterface op) {
    return !op->hasAttr(TF::kCompileDeviceTypeAttr);
  };

  mlir::OpBuilder builder(func.getContext());
  auto noinline_attr_name = absl::StrCat("tf.", tensorflow::kNoInlineAttr);
  llvm::SmallVector<SymbolUserOpInterface> noinline_pcall_ops,
      outermost_pcall_ops;
  if (failed(GetOpsOfTypeUntilMiss<TF::StatefulPartitionedCallOp,
                                   TF::PartitionedCallOp>(
          func, symtab, /*predicate*/ has_no_compile_device_type,
          /*hits*/ noinline_pcall_ops,
          /*first_misses*/ outermost_pcall_ops))) {
    return failure();
  }
  // Cluster outermost partitioned calls with _xla_compile_device_type
  // attribute.
  for (auto &pcall_op : outermost_pcall_ops) {
    auto call = llvm::cast<CallOpInterface>(pcall_op.getOperation());
    CallInterfaceCallable callable = call.getCallableForCallee();
    auto sym = callable.get<SymbolRefAttr>();
    EncapsulatePartitionedCall(pcall_op, sym.getRootReference());
  }
  // Partitioned calls are executed asynchronous. The calls outside of
  // device clusters therefore should not be inlined to perserve run-time
  // performance.
  for (auto &pcall_op : noinline_pcall_ops) {
    auto call = llvm::cast<CallOpInterface>(pcall_op.getOperation());
    auto callee = llvm::cast<func::FuncOp>(
        call.resolveCallable(&symbol_table_collection));
    callee->setAttr(noinline_attr_name, builder.getBoolAttr(true));
  }
  return success();
}

void XlaClusterFormationPass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTableCollection symbol_table_collection;
  SymbolTable symtab = symbol_table_collection.getSymbolTable(module);
  llvm::SmallVector<func::FuncOp> entry_funcs = GetEntryFunctions(module);
  for (auto &entry_func : entry_funcs) {
    if (entry_func->hasAttr(TF::kCompileDeviceTypeAttr)) {
      if (EncapsulateEntryFunctionBody(entry_func).failed()) {
        return signalPassFailure();
      }
    } else if (EncapsulateFirstXlaCompilablePartitionedCalls(
                   entry_func, symbol_table_collection, symtab)
                   .failed()) {
      return signalPassFailure();
    }
  }
}

}  // namespace

namespace TFDevice {
std::unique_ptr<OperationPass<ModuleOp>> CreateXlaClusterFormationPass() {
  return std::make_unique<XlaClusterFormationPass>();
}
}  // namespace TFDevice

}  // namespace mlir
