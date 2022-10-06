/* Copyright 2022 Google Inc. All Rights Reserved.

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

#include <algorithm>
#include <memory>
#include <queue>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TF {
namespace {

std::vector<Operation*> groupOperationsByDialect(Block& block);

#define GEN_PASS_DEF_ORDERBYDIALECTPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

// Reorder operations so that consecutive ops stay in the same dialect, as far
// as possible. This is to optimize the op order for the group-by-dialect pass,
// which factors consecutive same-dialect ops into functions.
// TODO(kramm): This pass needs to become aware of side-effects between ops
// of different dialects.
class OrderByDialectPass
    : public impl::OrderByDialectPassBase<OrderByDialectPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OrderByDialectPass)

  void runOnOperation() override {
    getOperation().walk([](Operation* function) {
      for (Region& region : function->getRegions()) {
        for (Block& block : region.getBlocks()) {
          if (block.empty()) continue;
          auto ops = groupOperationsByDialect(block);
          // Replace the block with the reordered block.
          for (Operation* op : ops) {
            op->remove();
            block.push_back(op);
          }
        }
      }
    });
  }
};

// Similar to MLIR's topological sort (lib/Transforms/TopologicalSort.cpp)
// but has an explicit scoring function to determine the next op to emit.
// Note that this doesn't explicitly handle TF side effects. However,
// it typically leaves the order of operations within a given dialect the
// same, and different dialects tend to not access the same resources.
std::vector<Operation*> groupOperationsByDialect(Block& block) {
  llvm::DenseMap<Operation*, int> remaining_incoming_edges;
  llvm::DenseMap<Operation*, int> position;
  llvm::DenseMap<Operation*, Operation*> ancestor;
  SmallVector<Operation*> ready;

  int i = 0;
  for (Operation& op : block.getOperations()) {
    int incoming_edges = 0;
    op.walk([&](Operation* child) {
      ancestor[child] = &op;
      for (Value v : child->getOperands()) {
        if (v.getParentBlock() == &block) {
          incoming_edges++;
        }
      }
    });
    remaining_incoming_edges[&op] = incoming_edges;
    if (incoming_edges == 0) {
      ready.push_back(&op);
    }
    position[&op] = i++;
  }

  std::queue<Value> todo;
  for (Value value : block.getArguments()) {
    todo.push(value);
  }

  StringRef current_dialect = "<none>";

  std::vector<Operation*> result;
  while (!todo.empty() || !ready.empty()) {
    while (!todo.empty()) {
      Value value = todo.front();
      todo.pop();
      // All operations that have all their inputs available are good to go.
      // Uses, not Users, in case getUsers ever dedups.
      for (OpOperand& operand : value.getUses()) {
        Operation* user = ancestor[operand.getOwner()];
        if (--remaining_incoming_edges[user] == 0) {
          ready.push_back(user);
        }
      }
    }

    // Find the "best" operation to emit. We
    // (a) stay in the same dialect as far as possible.
    // (b) preserve order within the ops of one dialect.
    // (c) emit the terminator last.
    auto better = [&](Operation* a, Operation* b) {
      if (a->hasTrait<OpTrait::IsTerminator>() !=
          b->hasTrait<OpTrait::IsTerminator>()) {
        return b->hasTrait<OpTrait::IsTerminator>();
      }
      bool a_current = a->getName().getDialectNamespace() == current_dialect;
      bool b_current = b->getName().getDialectNamespace() == current_dialect;
      if (a_current != b_current) {
        return a_current;
      }
      return position[a] < position[b];  // preserve order
    };

    Operation* best = nullptr;
    for (Operation* op : ready) {
      if (best == nullptr || better(op, best)) {
        best = op;
      }
    }

    if (!best) {
      assert(ready.empty());
      return result;  // happens for unused results for ops in the todo list
    }

    // Consider this operation emitted, and make its results available.
    ready.erase(std::find(ready.begin(), ready.end(), best));
    current_dialect = best->getName().getDialectNamespace();
    for (Value result : best->getResults()) {
      todo.push(result);
    }
    result.push_back(best);
  }
  return result;
}

}  // namespace

std::unique_ptr<Pass> CreateOrderByDialectPass() {
  return std::make_unique<OrderByDialectPass>();
}

void RegisterOrderByDialectPass() { registerPass(CreateOrderByDialectPass); }

}  // namespace TF
}  // namespace mlir
