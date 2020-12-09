/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <utility>

#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

namespace mlir {
namespace TFDevice {

namespace {

// This pass merges IfRegion ops together if they have the same predicate and it
// is safe to do so (there are no intermediate dependencies, they are in the
// same block, etc).
//
// A simple example:
//    "tf.IfRegion"(%0) ( {
//      %2 = "tf.A"() : () -> (tensor<f32>)
//      "tf.Yield"() : () -> ()
//      }, {
//      "tf.Yield"() : () -> ()
//     }) { is_stateless = true } : (tensor<i1>) -> ()
//    "tf.IfRegion"(%0) ( {
//      %2 = "tf.B"() : () -> (tensor<f32>)
//      "tf.Yield"() : () -> ()
//      }, {
//      "tf.Yield"() : () -> ()
//     }) { is_stateless = true } : (tensor<i1>) -> ()
// Would become:
//    "tf.IfRegion"(%0) ( {
//      %2 = "tf.A"() : () -> (tensor<f32>)
//      %3 = "tf.B"() : () -> (tensor<f32>)
//      "tf.Yield"() : () -> ()
//      }, {
//      "tf.Yield"() : () -> ()
//     }) { is_stateless = true } : (tensor<i1>) -> ()

struct MergeControlFlow : public TF::PerFunctionAggregateAnalysisConsumerPass<
                              MergeControlFlow, TF::SideEffectAnalysis> {
  void runOnFunction(FuncOp func,
                     const TF::SideEffectAnalysis::Info& side_effect_analysis);
};

// Returns whether it is safe to merge `source` IfRegion into `destination`
// IfRegion. `source` must come after `destination`.
bool SafeToMerge(TF::IfRegionOp source, TF::IfRegionOp destination,
                 const TF::SideEffectAnalysis::Info& side_effect_analysis) {
  // IfRegion ops must be in the same block.
  if (source.getOperation()->getBlock() !=
      destination.getOperation()->getBlock())
    return false;
  assert(destination.getOperation()->isBeforeInBlock(source.getOperation()));

  llvm::SmallSetVector<Operation*, 4> source_ops;
  source_ops.insert(source);
  for (Operation& op : source.then_branch().front()) {
    source_ops.insert(&op);
  }
  for (Operation& op : source.else_branch().front()) {
    source_ops.insert(&op);
  }

  // If there is an intermediate data or side effect dependency between the
  // ops in destination and the ops in the source, it's not safe to merge
  // them.
  llvm::SmallSetVector<Operation*, 4> op_stack;
  for (auto* user : destination.getOperation()->getUsers()) {
    if (!source_ops.contains(user)) op_stack.insert(user);
  }
  for (Operation& op : destination.then_branch().front()) {
    for (auto* successor : side_effect_analysis.DirectControlSuccessors(&op)) {
      if (!source_ops.contains(successor)) op_stack.insert(successor);
    }
  }
  for (Operation& op : destination.else_branch().front()) {
    for (auto* successor : side_effect_analysis.DirectControlSuccessors(&op)) {
      if (!source_ops.contains(successor)) op_stack.insert(successor);
    }
  }

  bool safe_to_merge = true;

  while (!op_stack.empty()) {
    auto* next_op = op_stack.pop_back_val();
    for (auto* user : next_op->getUsers()) {
      if (source_ops.contains(user)) {
        safe_to_merge = false;
        break;
      } else {
        op_stack.insert(user);
      }
    }
    for (auto* successor :
         side_effect_analysis.DirectControlSuccessors(next_op)) {
      if (source_ops.contains(successor)) {
        safe_to_merge = false;
        break;
      } else {
        op_stack.insert(successor);
      }
    }
    if (!safe_to_merge) break;
  }
  return safe_to_merge;
}

// Move the body excluding the terminators of else and then regions from
// 'source' to 'destination'.
void MoveBranches(TF::IfRegionOp source, TF::IfRegionOp destination) {
  Block& destination_then_block = destination.then_branch().front();
  auto& source_then_body = source.then_branch().front().getOperations();
  destination_then_block.getOperations().splice(
      destination_then_block.without_terminator().end(), source_then_body,
      source_then_body.begin(), std::prev(source_then_body.end()));

  Block& destination_else_block = destination.else_branch().front();
  auto& source_else_body = source.else_branch().front().getOperations();
  destination_else_block.getOperations().splice(
      destination_else_block.without_terminator().end(), source_else_body,
      source_else_body.begin(), std::prev(source_else_body.end()));
}

Operation* GetIfInsertionPoint(TF::IfRegionOp source,
                               TF::IfRegionOp destination) {
  // TODO(b/173422484): Pick this insertion point better.
  return source.getOperation();
}

TF::IfRegionOp CreateMergedIf(TF::IfRegionOp source,
                              TF::IfRegionOp destination) {
  llvm::SmallVector<Type, 4> merged_return_types;

  OpBuilder builder(destination);
  // Create new IfRegion with correct merged results.
  builder.setInsertionPoint(GetIfInsertionPoint(source, destination));
  auto new_if_op = builder.create<TF::IfRegionOp>(
      destination.getLoc(), merged_return_types, destination.cond(),
      destination.is_stateless() && source.is_stateless());
  new_if_op.then_branch().push_back(new Block);
  new_if_op.else_branch().push_back(new Block);
  llvm::SmallVector<Value, 4> merged_then_yield_values;
  builder.setInsertionPointToEnd(&new_if_op.then_branch().front());
  builder.create<TF::YieldOp>(
      destination.then_branch().front().getTerminator()->getLoc(),
      /*operands=*/merged_then_yield_values);

  llvm::SmallVector<Value, 4> merged_else_yield_values;
  builder.setInsertionPointToEnd(&new_if_op.else_branch().front());
  builder.create<TF::YieldOp>(
      destination.else_branch().front().getTerminator()->getLoc(),
      /*operands=*/merged_else_yield_values);

  // Merge the two branch regions from both IfRegionOps into new IfRegionOp.
  MoveBranches(/*source=*/destination, /*destination=*/new_if_op);
  destination.erase();
  MoveBranches(/*source=*/source, /*destination=*/new_if_op);
  source.erase();
  return new_if_op;
}

// Groups if regions by common predicate and attemps to merge them.
void OptimizeIfRegions(
    Block* block, const TF::SideEffectAnalysis::Info& side_effect_analysis) {
  // Determine IfRegions with the same predicate.
  llvm::SmallDenseMap<Value, llvm::SmallVector<TF::IfRegionOp, 8>, 8>
      grouped_if_ops;
  block->walk([&](TF::IfRegionOp if_op) {
    auto it = grouped_if_ops.try_emplace(if_op.cond());
    it.first->getSecond().push_back(if_op);
  });

  for (const auto& entry : grouped_if_ops) {
    llvm::ArrayRef<TF::IfRegionOp> if_ops = entry.second;
    TF::IfRegionOp first_if_op = if_ops[0];
    for (int i = 1; i < if_ops.size(); ++i) {
      TF::IfRegionOp if_op = if_ops[i];
      if (!SafeToMerge(if_op, first_if_op, side_effect_analysis)) break;

      auto new_if_op = CreateMergedIf(if_op, first_if_op);

      first_if_op = new_if_op;
    }
  }
}

void MergeControlFlow::runOnFunction(
    FuncOp func, const TF::SideEffectAnalysis::Info& side_effect_analysis) {
  auto result = func.walk([&](tf_device::ClusterOp cluster) {
    OptimizeIfRegions(&cluster.GetBody(), side_effect_analysis);
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateMergeControlFlowPass() {
  return std::make_unique<MergeControlFlow>();
}

static PassRegistration<MergeControlFlow> pass(
    "tf-merge-control-flow", "Merges control flow with a common predicate.");
}  // namespace TFDevice
}  // namespace mlir
