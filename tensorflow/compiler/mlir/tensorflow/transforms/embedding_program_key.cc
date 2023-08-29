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
#include <queue>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/split_into_island_per_op_pass.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"

namespace mlir {
namespace TFDevice {
namespace {
constexpr char kDeviceAttr[] = "device";
constexpr char kMiniBatchSplitsAttr[] = "mini_batch_splits";
constexpr char kMiniBatchCsrAttr[] = "mini_batch_in_csr";

#define GEN_PASS_DEF_EMBEDDINGPROGRAMKEYPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

struct EmbeddingProgramKeyPass
    : public impl::EmbeddingProgramKeyPassBase<EmbeddingProgramKeyPass> {
  void runOnOperation() override;
};

// Checks if `op` is nested in `block`.
bool OpInBlock(Operation* op, Block* block) {
  Block* op_block = op->getBlock();
  while (op_block) {
    if (op_block == block) return true;
    if (auto* parent_op = op_block->getParentOp()) {
      op_block = parent_op->getBlock();
    } else {
      break;
    }
  }
  return false;
}

// Checks if there is a precedecessor TPUCOmpileMlir op of `preprocess_op` in
// `func_op`.  Assumes both ops are wrapped in a tf_device.launch.
bool HasTPUCompilePredecessor(func::FuncOp func_op, Operation* preprocess_op) {
  bool has_tpu_compile_predecessor = false;
  func_op.walk([&](TF::_TPUCompileMlirOp compile_op) {
    if (compile_op->getParentOp()->getBlock() ==
            preprocess_op->getParentOp()->getBlock() &&
        compile_op->getParentOp()->isBeforeInBlock(
            preprocess_op->getParentOp())) {
      has_tpu_compile_predecessor = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return has_tpu_compile_predecessor;
}

// Get all of the successor ops of `root_op` in the same block.
llvm::SmallSetVector<Operation*, 4> GetSuccessorOps(Operation* root_op) {
  std::queue<Operation*> queue;
  llvm::SmallSetVector<Operation*, 4> ops_to_move;

  queue.push(root_op);
  while (!queue.empty()) {
    Operation* op = queue.front();
    queue.pop();
    if (llvm::isa<tf_device::ReturnOp>(op)) continue;
    ops_to_move.insert(op);
    for (Operation* user : op->getUsers()) {
      if (!ops_to_move.contains(user)) {
        queue.push(user);
      }
    }
  }
  return ops_to_move;
}

tf_device::LaunchOp CreateLaunchForBlock(OpBuilder* builder,
                                         Operation* before_op,
                                         Block* launch_block,
                                         llvm::StringRef host_device) {
  // Find results and result types of ops in block that needs to returned.
  llvm::SmallVector<Value, 4> launch_results;
  llvm::SmallVector<Type, 4> launch_result_types;
  for (Operation& op : *launch_block) {
    for (Value result : op.getResults()) {
      bool has_external_uses = false;
      for (Operation* user : result.getUsers()) {
        if (OpInBlock(user, launch_block)) continue;
        has_external_uses = true;
        break;
      }
      if (has_external_uses) {
        launch_results.push_back(result);
        launch_result_types.push_back(result.getType());
      }
    }
  }

  builder->setInsertionPointAfter(before_op);
  auto launch = builder->create<tf_device::LaunchOp>(
      before_op->getLoc(), builder->getStringAttr(host_device),
      launch_result_types);
  launch.getBody().push_back(launch_block);

  builder->setInsertionPointToEnd(&launch.GetBody());
  builder->create<tf_device::ReturnOp>(before_op->getLoc(), launch_results);

  return launch;
}

// Creates a new_launch after `before_op` with `ops_to_move` from
// `original_launch_op` on `device`.
void CreateNewLaunchOp(OpBuilder* builder,
                       tf_device::LaunchOp original_launch_op,
                       Operation* preprocess_op,
                       llvm::SmallSetVector<Operation*, 4>& ops_to_move,
                       llvm::StringRef device, Operation* before_op) {
  llvm::SmallDenseMap<Value, int> original_return_operand_map;
  for (OpOperand& operand :
       original_launch_op.GetBody().getTerminator()->getOpOperands()) {
    original_return_operand_map.insert(
        {operand.get(), operand.getOperandNumber()});
  }
  Block* old_block = preprocess_op->getBlock();
  Block* launch_block = new Block;
  for (Operation& op : llvm::make_early_inc_range(old_block->getOperations())) {
    if (ops_to_move.contains(&op)) {
      op.moveBefore(launch_block, launch_block->end());
    }
  }

  // Create the new launch op after TPUCompileOp with the preprocessing ops.
  tf_device::LaunchOp moved_launch_op =
      CreateLaunchForBlock(builder, before_op, launch_block, device);

  // Replace usages of the original launch op with the new launch op for
  // operations that were moved.
  for (OpOperand& operand :
       moved_launch_op.GetBody().getTerminator()->getOpOperands()) {
    auto iter = original_return_operand_map.find(operand.get());
    if (iter != original_return_operand_map.end()) {
      original_launch_op.getResult(iter->second)
          .replaceAllUsesWith(
              moved_launch_op.getResult(operand.getOperandNumber()));
    }
  }
}

// Creates a smaller launch op on `device` from `original_launch_op` where some
// ops have been removed from `old_block`.
void CreateReducedLaunchOp(OpBuilder* builder, Block* old_block,
                           tf_device::LaunchOp original_launch_op,
                           llvm::StringRef device) {
  Block* reduced_launch_block = new Block;
  for (Operation& op : llvm::make_early_inc_range(old_block->getOperations())) {
    if (!llvm::isa<tf_device::ReturnOp>(op)) {
      op.moveBefore(reduced_launch_block, reduced_launch_block->end());
    }
  }

  tf_device::LaunchOp reduced_launch_op = CreateLaunchForBlock(
      builder, original_launch_op, reduced_launch_block, device);
  // Replace an still existing usages of original_launch_op with
  // reduced_launch_op results.
  llvm::SmallDenseMap<Value, int> reduced_return_operand_map;
  for (OpOperand& operand :
       reduced_launch_op.GetBody().getTerminator()->getOpOperands()) {
    reduced_return_operand_map.insert(
        {operand.get(), operand.getOperandNumber()});
  }

  for (int i = 0; i < original_launch_op->getNumResults(); i++) {
    Value operand = original_launch_op.GetBody().getTerminator()->getOperand(i);
    auto defining_op = operand.getDefiningOp();
    if (defining_op != nullptr &&
        defining_op->getParentOp() == reduced_launch_op.getOperation()) {
      auto iter = reduced_return_operand_map.find(operand);
      if (iter != reduced_return_operand_map.end()) {
        original_launch_op->getResult(i).replaceAllUsesWith(
            reduced_launch_op->getResult(iter->second));
      }
    }
  }

  // Replace the outside usage of ops still in reduced launch.
  auto operand_not_in_launch = [&](OpOperand& operand) {
    return !reduced_launch_op.getOperation()->isProperAncestor(
        operand.getOwner());
  };

  for (OpOperand& operand :
       reduced_launch_op.GetBody().getTerminator()->getOpOperands()) {
    operand.get().replaceUsesWithIf(
        reduced_launch_op.getResult(operand.getOperandNumber()),
        operand_not_in_launch);
  }

  // Handle pass through block arguments.
  for (OpOperand& operand :
       original_launch_op.GetBody().getTerminator()->getOpOperands()) {
    if (operand.get().isa<BlockArgument>()) {
      original_launch_op.getResult(operand.getOperandNumber())
          .replaceAllUsesWith(operand.get());
    }
  }
}

// Move `preprocess_op` after _TPUCompileMlir op if there are no _TPUCompileMlir
// ops before `preprocess_op`.  This actually creates a new launch op after
// _TPUCompileMlir and moves `preprocess_op` and its successors that are input
// to TPUExecute to it.
LogicalResult MovePreprocessingOps(OpBuilder* builder, func::FuncOp func_op,
                                   Operation* preprocess_op) {
  // If this is already a TPUCompile predecessor, no need to move the
  // preprocessing ops.
  if (HasTPUCompilePredecessor(func_op, preprocess_op)) return success();

  auto original_launch_op =
      llvm::dyn_cast<tf_device::LaunchOp>(preprocess_op->getParentOp());
  // Device of original launch looked up before moving the preprocessing ops
  // around.
  StringAttr device =
      original_launch_op->getAttrOfType<StringAttr>(kDeviceAttr);

  if (!device) {
    return original_launch_op->emitOpError()
           << "Launch op has an invalid device attribute.";
  }

  // Find the TPUCompile successor.
  Operation* tpu_compile_successor = nullptr;
  func_op.walk([&](TF::_TPUCompileMlirOp compile_op) {
    if (compile_op->getParentOp() == nullptr ||
        !isa<tf_device::LaunchOp>(compile_op->getParentOp()))
      return WalkResult::advance();
    if (compile_op->getParentOp()->getBlock() ==
            original_launch_op->getBlock() &&
        original_launch_op->isBeforeInBlock(compile_op->getParentOp())) {
      tpu_compile_successor = compile_op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  // Return early if can't find TPUCompile successor.
  if (tpu_compile_successor == nullptr) return success();

  // Get all of the successor ops of the preprocess_op that are in the same
  // block.
  llvm::SmallSetVector<Operation*, 4> ops_to_move =
      GetSuccessorOps(preprocess_op);

  Block* old_block = preprocess_op->getBlock();

  // Move the successor ops of the preprocess op to a new launch after the
  // successor TPUCompileOp.
  CreateNewLaunchOp(builder, original_launch_op, preprocess_op, ops_to_move,
                    device, tpu_compile_successor->getParentOp());
  // Rewrite the original launch op with smaller set of returns.
  CreateReducedLaunchOp(builder, old_block, original_launch_op, device);

  original_launch_op->erase();
  return success();
}

// Rewrites the program_key input of `preprocess_op` to use the output of
// _TPUCompileMlir.
void RewritePreprocessInputs(OpBuilder* builder, func::FuncOp func_op,
                             Operation* preprocess_op) {
  if (preprocess_op->getParentOp() == nullptr ||
      !isa<tf_device::LaunchOp>(preprocess_op->getParentOp()))
    return;

  // Find predecessor TPUCompile Op and rewrite the program key.
  func_op.walk([&](TF::_TPUCompileMlirOp compile_op) {
    if (compile_op->getParentOp() == nullptr ||
        !isa<tf_device::LaunchOp>(compile_op->getParentOp()))
      return WalkResult::advance();
    if (compile_op->getParentOp()->getBlock() ==
            preprocess_op->getParentOp()->getBlock() &&
        compile_op->getParentOp()->isBeforeInBlock(
            preprocess_op->getParentOp())) {
      // Find the corresponding result of the _TPUCompileMlirOp in launch return
      // op.
      for (OpOperand& operand : compile_op->getResult(1).getUses()) {
        if (llvm::isa<tf_device::ReturnOp>(operand.getOwner())) {
          preprocess_op->setOperand(0, compile_op->getParentOp()->getResult(
                                           operand.getOperandNumber()));
        }
      }
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
}

void EmbeddingProgramKeyPass::runOnOperation() {
  // Find all of the relevant post processing ops.
  llvm::SmallVector<Operation*, 6> preprocess_ops;

  // Handle ops with mini_batch_splits attribute first since all preprocessing
  // ops may need to be moved.
  getOperation().walk([&](Operation* op) {
    if (op->hasAttr(kMiniBatchSplitsAttr) &&
        isa<tf_device::LaunchOp>(op->getParentOp())) {
      preprocess_ops.push_back(op);
    }
  });

  OpBuilder builder(&getContext());

  for (Operation* preprocess_op : preprocess_ops) {
    if (failed(MovePreprocessingOps(&builder, getOperation(), preprocess_op)))
      return signalPassFailure();
    RewritePreprocessInputs(&builder, getOperation(), preprocess_op);
  }

  // Handle ops with mini_batch_in_csr attribute.
  preprocess_ops.clear();
  getOperation().walk([&](Operation* op) {
    if (op->hasAttr(kMiniBatchCsrAttr)) {
      preprocess_ops.push_back(op);
    }
  });

  for (Operation* preprocess_op : preprocess_ops) {
    RewritePreprocessInputs(&builder, getOperation(), preprocess_op);
  }
}

}  // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateEmbeddingProgramKeyPass() {
  return std::make_unique<EmbeddingProgramKeyPass>();
}

}  // namespace TFDevice
}  // namespace mlir
