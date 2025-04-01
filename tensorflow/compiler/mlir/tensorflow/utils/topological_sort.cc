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

#include "tensorflow/compiler/mlir/tensorflow/utils/topological_sort.h"

#include <algorithm>
#include <queue>
#include <utility>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace TF {

ExtraDependenciesFunction no_extra_dependencies = nullptr;

std::vector<Operation*> SortBlockTopologically(
    Block& block, PriorityFunction priorityFunction,
    ExtraDependenciesFunction extraDependencies) {
  llvm::DenseMap<Operation*, int> remaining_incoming_data_edges;
  llvm::DenseMap<Operation*, int> remaining_incoming_ctrl_edges;
  llvm::DenseMap<Operation*, int> position;
  llvm::DenseMap<Operation*, Operation*> ancestor;
  SmallVector<Operation*> ready;

  llvm::SmallVector<mlir::Operation*, 4> empty_op_set;
  auto ctrlPredecessors =
      [&](Operation* op) -> llvm::SmallVector<mlir::Operation*, 4> const& {
    if (extraDependencies) {
      return extraDependencies(op, /*incoming=*/true);
    } else {
      return empty_op_set;
    }
  };
  auto ctrlSuccessors =
      [&](Operation* op) -> llvm::SmallVector<mlir::Operation*, 4> const& {
    if (extraDependencies) {
      return extraDependencies(op, /*incoming=*/false);
    } else {
      return empty_op_set;
    }
  };

  int i = 0;
  for (Operation& op : block.getOperations()) {
    int incoming_ctrl_edges = 0;
    int incoming_data_edges = 0;
    op.walk([&](Operation* child) {
      ancestor[child] = &op;
      for (Operation* predecessor : ctrlPredecessors(child)) {
        if (predecessor->getBlock() == &block) {
          incoming_ctrl_edges++;
        }
      }
      for (Value v : child->getOperands()) {
        if (v.getParentBlock() == &block) {
          incoming_data_edges++;
        }
      }
    });
    remaining_incoming_data_edges[&op] = incoming_data_edges;
    remaining_incoming_ctrl_edges[&op] = incoming_ctrl_edges;
    if (incoming_data_edges == 0 && incoming_ctrl_edges == 0) {
      ready.push_back(&op);
    }
    position[&op] = i++;
  }

  std::queue<Value> todo;
  for (Value value : block.getArguments()) {
    todo.push(value);
  }

  std::vector<Operation*> result;
  Operation* previous_op = nullptr;
  while (!todo.empty() || !ready.empty()) {
    while (!todo.empty()) {
      Value value = todo.front();
      todo.pop();
      // All operations that have all their inputs available are good to go.
      // Uses, not Users, in case getUsers ever dedups.
      for (OpOperand& operand : value.getUses()) {
        Operation* user = ancestor[operand.getOwner()];
        remaining_incoming_data_edges[user]--;
        if (remaining_incoming_data_edges[user] == 0 &&
            remaining_incoming_ctrl_edges[user] == 0) {
          ready.push_back(user);
        }
      }
    }

    // Find the "best" operation to emit. We
    // (a) emit the terminator last.
    // (b) honor the priority function (as far as possible).
    // (c) preserve order within the ops of one dialect.
    auto better = [&](Operation* a, Operation* b) {
      if (a->hasTrait<OpTrait::IsTerminator>() !=
          b->hasTrait<OpTrait::IsTerminator>()) {
        return b->hasTrait<OpTrait::IsTerminator>();
      }
      int a_priority = priorityFunction(previous_op, a);
      int b_priority = priorityFunction(previous_op, b);
      if (a_priority != b_priority) {
        return a_priority > b_priority;
      } else {
        return position[a] < position[b];  // preserve order
      }
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
    previous_op = best;
    for (Value result : best->getResults()) {
      todo.push(result);
    }
    for (Operation* successor : ctrlSuccessors(best)) {
      if (ancestor.find(successor) != ancestor.end()) {
        successor = ancestor[successor];
        remaining_incoming_ctrl_edges[successor]--;
        if (remaining_incoming_ctrl_edges[successor] == 0 &&
            remaining_incoming_data_edges[successor] == 0) {
          ready.push_back(successor);
        }
      }
    }
    result.push_back(best);
  }
  return result;
}

}  // namespace TF
}  // namespace mlir
