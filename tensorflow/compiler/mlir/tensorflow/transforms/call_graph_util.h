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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_CALL_GRAPH_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_CALL_GRAPH_UTIL_H_

#include <functional>
#include <stack>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {

// Find the outermost ops with any of specified types starting from the tree
// rooted at `root` parameter. The results are stored in `ops`. Addtional
// filters can be specified by providing `predicate` parameter.
template <typename T, typename... Types>
LogicalResult GetOutermostOpsOfType(
    func::FuncOp root, SymbolTable &symtab, llvm::SmallVector<Operation *> &ops,
    const std::function<bool(Operation *)> &predicate = {}) {
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
      for (auto attr : op->getAttrs()) {
        auto sym = attr.getValue().dyn_cast<SymbolRefAttr>();
        if (!sym) continue;
        auto v = symtab.lookup<func::FuncOp>(sym.getRootReference());
        if (!v) {
          // This is not expected to happen in practice.
          v.emitError() << "Cannot find function " << sym.getRootReference();
          return WalkResult::interrupt();
        }
        worklist.push(v);
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) return failure();
  }
  return success();
}

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_CALL_GRAPH_UTIL_H_
