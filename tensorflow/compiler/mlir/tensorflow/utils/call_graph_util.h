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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_CALL_GRAPH_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_CALL_GRAPH_UTIL_H_

#include <functional>
#include <stack>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {

// Return a list of attribute names that indicates an entry function.
std::vector<llvm::StringRef> GetEntryFunctionAttributeNames();

// Check if a function is an entry in an MLIR module.
bool IsEntryFunction(func::FuncOp func);

// Get all the entry functions in an MLIR module.
llvm::SmallVector<func::FuncOp> GetEntryFunctions(ModuleOp module);

// Get all the functions referenced in a symber user op and save them in
// `callees`.
LogicalResult GetCallees(SymbolUserOpInterface op, SymbolTable &symtab,
                         llvm::SmallVector<func::FuncOp> &callees);

// Find the first op with any of the specified types on each path rooted at the
// `root` node in a tree. Additional checks can be applied via `predicate`. The
// results are stored in `ops`.
template <typename T, typename... Types>
LogicalResult GetFirstOpsOfType(
    func::FuncOp root, SymbolTable &symtab,
    const std::function<bool(SymbolUserOpInterface)> &predicate,
    llvm::SmallVector<SymbolUserOpInterface> &ops) {
  std::stack<func::FuncOp> worklist;
  worklist.push(root);
  while (!worklist.empty()) {
    func::FuncOp u = worklist.top();
    worklist.pop();
    auto result = u.walk([&](SymbolUserOpInterface op) {
      if (llvm::isa<T, Types...>(op) && (!predicate || predicate(op))) {
        ops.push_back(op);
        return WalkResult::advance();
      }
      llvm::SmallVector<func::FuncOp> callees;
      if (GetCallees(op, symtab, callees).failed()) {
        return WalkResult::interrupt();
      }
      for (auto callee : callees) {
        worklist.push(callee);
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) return failure();
  }
  return success();
}

// Find the nodes with any of the specified types on the tree rooted at `root`
// node. Additional checks can be applied via `predicate`. The search skips
// the current path if a node with the specified types fails the check, and
// continues on the next path. The passing ops are stored in `hits`, while the
// first failing on on each path is stored in `first_misses`.
template <typename T, typename... Types>
LogicalResult GetOpsOfTypeUntilMiss(
    func::FuncOp root, SymbolTable &symtab,
    const std::function<bool(SymbolUserOpInterface)> &predicate,
    llvm::SmallVector<SymbolUserOpInterface> &hits,
    llvm::SmallVector<SymbolUserOpInterface> &first_misses) {
  std::stack<func::FuncOp> worklist;
  worklist.push(root);
  while (!worklist.empty()) {
    func::FuncOp u = worklist.top();
    worklist.pop();
    auto result = u.walk([&](SymbolUserOpInterface op) {
      if (llvm::isa<T, Types...>(op)) {
        if (!predicate || predicate(op)) {
          hits.push_back(op);
        } else {
          first_misses.push_back(op);
          return WalkResult::advance();
        }
      }
      llvm::SmallVector<func::FuncOp> callees;
      if (GetCallees(op, symtab, callees).failed()) {
        return WalkResult::interrupt();
      }
      for (auto callee : callees) {
        worklist.push(callee);
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) return failure();
  }
  return success();
}

// Check if a function has one region and one block only.
bool HasSingleBlock(func::FuncOp func);

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_CALL_GRAPH_UTIL_H_
