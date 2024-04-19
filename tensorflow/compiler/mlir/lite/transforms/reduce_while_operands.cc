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

// This is a pass to reduce operands without changing the outcome.

#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_REDUCEWHILEOPERANDSPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

struct ReduceWhileOperandsPass
    : public impl::ReduceWhileOperandsPassBase<ReduceWhileOperandsPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReduceWhileOperandsPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TFL::TensorFlowLiteDialect, TF::TensorFlowDialect>();
  }
  void runOnOperation() override;
};

LogicalResult FindImplicityProducers(
    const std::vector<uint64_t> &explicitly_consumed_ids,
    std::vector<bool> &is_consumed_id,
    const std::vector<std::vector<uint64_t>> &dependency_graph) {
  std::vector<uint64_t> queue;
  queue.reserve(is_consumed_id.size());
  for (auto id : explicitly_consumed_ids) {
    is_consumed_id[id] = true;
    queue.push_back(id);
  }
  while (!queue.empty()) {
    auto i = queue.back();
    queue.pop_back();

    // If there is a consumer which cannot be found in dependency graph, return
    // false.
    if (i >= dependency_graph.size()) {
      return failure();
    }

    for (auto j : dependency_graph.at(i)) {
      if (is_consumed_id[j]) continue;
      queue.push_back(j);
      is_consumed_id[j] = true;
    }
  }

  return success();
}

void FindProducers(Value start_node, std::vector<uint64_t> &neighbors) {
  llvm::DenseSet<Value> visited;
  std::vector<Value> queue;
  queue.push_back(start_node);
  visited.insert(start_node);
  while (!queue.empty()) {
    auto node = queue.back();
    queue.pop_back();
    if (auto arg = node.dyn_cast_or_null<BlockArgument>()) {
      neighbors.push_back(arg.getArgNumber());
      continue;
    }
    if (!node.getDefiningOp()) continue;
    for (Value operand : node.getDefiningOp()->getOperands()) {
      if (visited.contains(operand)) continue;
      queue.push_back(operand);
      visited.insert(operand);
    }
  }
}

void FindConsumedOp(Operation *start_op,
                    llvm::DenseSet<Operation *> &consumed_ops) {
  if (consumed_ops.contains(start_op)) return;
  std::vector<Operation *> queue;
  queue.push_back(start_op);
  consumed_ops.insert(start_op);
  while (!queue.empty()) {
    auto op = queue.back();
    queue.pop_back();
    for (Value operand : op->getOperands()) {
      if (!operand.getDefiningOp()) continue;
      auto def_op = operand.getDefiningOp();
      if (consumed_ops.contains(def_op)) continue;
      queue.push_back(def_op);
      consumed_ops.insert(def_op);
    }
  }
}

inline bool IsConstant(Operation *op) { return matchPattern(op, m_Constant()); }

bool AllOperationSafe(Block &block) {
  auto walk_result = block.walk([&](Operation *op) {
    // op has SideEffect.
    if (!isa_and_nonnull<TFL::WhileOp>(op) &&
        !op->hasTrait<OpTrait::IsTerminator>() && !isMemoryEffectFree(op)) {
      return WalkResult::interrupt();
    }
    // op has implict arguments not listed in operands.
    // Fact: if every op's operands are defined in the same block as op,
    //       then no operation has implicit arugments (constant doesn't count).
    for (auto operand : op->getOperands()) {
      if (operand.dyn_cast_or_null<BlockArgument>()) continue;
      auto operand_op = operand.getDefiningOp();
      if (IsConstant(operand_op)) continue;
      if (operand_op->getBlock() != op->getBlock()) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return !walk_result.wasInterrupted();
}

// It reduces the following pattern:
//
// S = (0, 0, 0)
// while S[2] < 3:
//  a0 = S[0] * 2
//  a1 = a0 + S[1]
//  a2 = S[2] + 1
//  S = (a0, a1, a2)
// return S[0]
//
// the 2nd operand (i = 1) as well as its related op (a1 = a0 + S[1])
// can be removed since only S[0] is returned.
// It cannot be removed by loop-invariant-code-motion pass since every value
// is used and changed in the while loop.

// Moreover, we require
// 1. no implicit argument: For every operation in whileOp, all dependent values
//    (except for constant) are explicitly passed in.
// 2. no side effect: Every operation inside whileOp can be safely
//    remove when it is useEmpty().
// 3. no call func inside while.
bool ReduceWhileOperands(TFL::WhileOp while_op) {
  std::vector<uint64_t> explicitly_consumed_ids;
  Block &cond = while_op.getCond().front();
  Block &body = while_op.getBody().front();

  auto n = while_op.getNumOperands();
  if (!AllOperationSafe(cond) || !AllOperationSafe(body)) return false;

  // Find all Consumed indices.
  // i is consumed element if result(i) is used outside whileOp or
  // argument(i) is used in whileOp.getCond().
  for (auto i = 0; i < n; ++i) {
    if (!while_op.getResult(i).use_empty() ||
        !cond.getArgument(i).use_empty()) {
      explicitly_consumed_ids.push_back(i);
    }
  }
  // Empty consumed_element_ids implies none of results is used.
  if (explicitly_consumed_ids.empty()) {
    while_op.erase();
    return true;
  }
  // If every element is consumed, one can't reduce any operand.
  if (explicitly_consumed_ids.size() == n) {
    return false;
  }

  // Build the dependency graph.
  // If result(i) is depend on argument(j) in While.body(), then we put
  // directed edge (i->j) into the graph.
  std::vector<std::vector<uint64_t>> dependency_graph;
  dependency_graph.reserve(n);

  Operation &yield_op = body.back();
  auto results = yield_op.getOperands();
  for (auto i = 0; i < n; ++i) {
    std::vector<uint64_t> neighbors;
    neighbors.reserve(n);
    FindProducers(results[i], neighbors);
    dependency_graph.push_back(neighbors);
  }

  std::vector<bool> is_consumed_id(n, false);
  if (failed(FindImplicityProducers(explicitly_consumed_ids, is_consumed_id,
                                    dependency_graph))) {
    return false;
  }

  // Find all consumed operations in while body.
  llvm::DenseSet<Operation *> consumed_ops;
  // We'll pass in the erase_indices to erase several operands simultaneously.
  llvm::BitVector erase_indices(n);
  consumed_ops.insert(&yield_op);
  for (auto i = 0; i < n; ++i) {
    if (!is_consumed_id[i]) {
      erase_indices.set(i);
    } else if (results[i].getDefiningOp()) {
      FindConsumedOp(results[i].getDefiningOp(), consumed_ops);
    }
  }
  // Remove elements and operations in while_body that are not indispensable.
  yield_op.eraseOperands(erase_indices);
  // Remove ops from bottom to top.
  for (Operation &op :
       llvm::make_early_inc_range(reverse(body.getOperations())))
    // Constant will not be removed in case it is implicitly used.
    if (!consumed_ops.contains(&op) && !IsConstant(&op)) {
      op.erase();
    }
  body.eraseArguments(erase_indices);
  cond.eraseArguments(erase_indices);

  llvm::SmallVector<Value> new_operands;
  llvm::SmallVector<Type> new_result_types;
  new_operands.reserve(n - erase_indices.size());
  new_result_types.reserve(n - erase_indices.size());
  // After reducing, the number of results is decreased. The i-th result of old
  // WhileOp becomes the j-th (j<=i) result of new WhileOp. This information is
  // stored in id_map (id_map[i] = j).
  std::vector<uint64_t> id_map(n, 0);
  uint64_t j = 0;
  for (auto i = 0; i < n; ++i) {
    if (is_consumed_id[i]) {
      id_map[i] = j++;
      new_operands.push_back(while_op.getOperand(i));
      new_result_types.push_back(while_op.getResultTypes()[i]);
    }
  }

  auto new_while_op = OpBuilder(while_op).create<WhileOp>(
      while_op.getLoc(), new_result_types, new_operands, while_op->getAttrs());
  new_while_op.getCond().takeBody(while_op.getCond());
  new_while_op.getBody().takeBody(while_op.getBody());

  for (auto i = 0; i < n; ++i) {
    if (!while_op.getResult(i).use_empty()) {
      auto j = id_map[i];
      while_op.getResult(i).replaceAllUsesWith(new_while_op.getResult(j));
    }
  }
  while_op.erase();
  return erase_indices.any();
}

void ReduceWhileOperandsPass::runOnOperation() {
  auto fn = getOperation();
  fn.walk([&](TFL::WhileOp while_op) { ReduceWhileOperands(while_op); });
}

static PassRegistration<ReduceWhileOperandsPass> pass;
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateReduceWhileOperandsPass() {
  return std::make_unique<ReduceWhileOperandsPass>();
}

}  // namespace TFL
}  // namespace mlir
