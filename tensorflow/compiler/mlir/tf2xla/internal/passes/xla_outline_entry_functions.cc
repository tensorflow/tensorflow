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
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/call_graph_util.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

using mlir::ModuleOp;
using mlir::Operation;
using mlir::SymbolTable;

#define GEN_PASS_DEF_XLAOUTLINEENTRYFUNCTIONSPASS
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/clustering_passes.h.inc"

inline constexpr char kOutlinedFuncSuffix[] = "_outlined";

// Outlines the body of an entry function with `_xla_compile_device_type`
// attribute and calls the outlined function with a
// `tf.StatefulPartitionedCall`.
struct XlaOutlineEntryFunctionsPass
    : public impl::XlaOutlineEntryFunctionsPassBase<
          XlaOutlineEntryFunctionsPass> {
  void runOnOperation() override;
};

void RenameFunction(mlir::func::FuncOp func, const std::string &new_func_name,
                    SymbolTable &symtab) {
  symtab.remove(func);
  symtab.setSymbolName(func, new_func_name);
  // Name conflicts are resolved automatically by SymbolTable class by attaching
  // a unique counter value to the names.
  symtab.insert(func);
}

// Propagate compilation markers from the source to the destination.
void PropagateCompilationMarkers(Operation *src, Operation *dest) {
  mlir::TF::CopyUnderscoredAttributes(src, dest);
  if (src->hasAttr(mlir::TF::kAllowSoftPlacementAttr)) {
    dest->setAttr(mlir::TF::kAllowSoftPlacementAttr,
                  src->getAttr(mlir::TF::kAllowSoftPlacementAttr));
  }
}

mlir::func::FuncOp CreateWrapperFunction(mlir::func::FuncOp func,
                                         const std::string &caller_name,
                                         const std::string &callee_name) {
  mlir::OpBuilder builder(func);
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::FunctionType func_type = func.getFunctionType();
  mlir::Location loc = func.getLoc();
  auto wrapper_func = mlir::func::FuncOp::create(loc, caller_name, func_type);
  mlir::Block *block = builder.createBlock(&wrapper_func.getBody());
  block->addArguments(
      wrapper_func.getArgumentTypes(),
      llvm::SmallVector<mlir::Location>(wrapper_func.getNumArguments(), loc));
  auto pcall_op = builder.create<mlir::TF::StatefulPartitionedCallOp>(
      loc, func_type.getResults(), wrapper_func.getArguments(),
      mlir::SymbolRefAttr::get(builder.getContext(), callee_name),
      builder.getStringAttr(""), builder.getStringAttr(""),
      builder.getStringAttr(""));
  builder.create<mlir::func::ReturnOp>(loc, pcall_op.getResults());
  PropagateCompilationMarkers(func, pcall_op);
  // Mark the original function private so it can be inlined.
  func.setVisibility(mlir::func::FuncOp::Visibility::Private);
  return wrapper_func;
}

void ReplaceEntryFunction(mlir::func::FuncOp original_func,
                          mlir::func::FuncOp new_func) {
  auto move_attr = [&](auto attr, Operation *src, Operation *dest) {
    if (src->hasAttr(attr)) {
      dest->setAttr(attr, src->getAttr(attr));
      src->removeAttr(attr);
    }
  };

  for (const auto &attr : mlir::GetEntryFunctionAttributeNames()) {
    move_attr(attr, original_func, new_func);
  }
  mlir::TF::CopyDeviceAndUnderscoredAttributes(original_func, new_func);
}

mlir::func::FuncOp RewriteEntryFunctionWithCompilationMarkers(
    mlir::func::FuncOp entry_func, SymbolTable &symtab) {
  const std::string entry_func_name = entry_func.getSymName().str(),
                    outlined_entry_func_name =
                        entry_func_name + kOutlinedFuncSuffix;
  RenameFunction(entry_func, outlined_entry_func_name, symtab);
  auto new_entry_func = CreateWrapperFunction(entry_func, entry_func_name,
                                              outlined_entry_func_name);
  ReplaceEntryFunction(entry_func, new_entry_func);
  symtab.insert(new_entry_func);
  return new_entry_func;
}

void XlaOutlineEntryFunctionsPass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable symtab(module);

  llvm::SmallVector<mlir::func::FuncOp> entry_funcs = GetEntryFunctions(module);

  for (auto &entry_func : entry_funcs) {
    if (entry_func->hasAttr(mlir::TF::kCompileDeviceTypeAttr)) {
      RewriteEntryFunctionWithCompilationMarkers(entry_func, symtab);
    }
  }
}

std::unique_ptr<mlir::OperationPass<ModuleOp>>
CreateXlaOutlineEntryFunctionsPass() {
  return std::make_unique<XlaOutlineEntryFunctionsPass>();
}

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
