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

// This file implements logic for lowering TensorFlow dialect's control flow to
// the XLA dialect.

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <tuple>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

using mlir::PassRegistration;

namespace mlir {
namespace mhlo {
namespace {

#define GEN_PASS_DEF_LEGALIZETFCONTROLFLOW
#include "tensorflow/compiler/mlir/xla/transforms/xla_legalize_tf_passes.h.inc"

class LegalizeTFControlFlow
    : public impl::LegalizeTFControlFlowBase<LegalizeTFControlFlow> {
 public:
  void runOnOperation() override;
};
}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLegalizeTFControlFlowPass() {
  return std::make_unique<LegalizeTFControlFlow>();
}

namespace {

// Replaces block terminator (tf.Yield) with `mhlo.return`. Additional results
// can be returned if `extra_results` is not empty.
void ReplaceTerminator(Block* block, OpBuilder* builder) {
  Operation* terminator = block->getTerminator();
  assert(isa<TF::YieldOp>(terminator));
  Location loc = terminator->getLoc();

  builder->setInsertionPoint(terminator);
  auto results = llvm::to_vector<4>(terminator->getOperands());
  builder->create<mhlo::ReturnOp>(loc, results);
  terminator->erase();
}

void LowerIfRegion(TF::IfRegionOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  builder.setInsertionPoint(op);
  ReplaceTerminator(&op.getThenBranch().front(), &builder);

  builder.setInsertionPoint(op);
  ReplaceTerminator(&op.getElseBranch().front(), &builder);

  // Create the new `mhlo.if` op and take ownership of regions from
  // `tf.IfRegion` op.
  builder.setInsertionPoint(op);
  auto if_op =
      builder.create<mhlo::IfOp>(loc, op.getResultTypes(), op.getCond());
  if_op.getTrueBranch().takeBody(op.getThenBranch());
  if_op.getFalseBranch().takeBody(op.getElseBranch());

  // Replace all uses of `op` results with that of `mhlo.IfOp`.
  op->replaceAllUsesWith(if_op);

  op.erase();
}

void LowerCaseRegion(TF::CaseRegionOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  for (Region& region : op.getBranches()) {
    builder.setInsertionPoint(op);
    ReplaceTerminator(&region.front(), &builder);
  }

  // Create the new `mhlo.case` op and take ownership of regions from
  // `tf.CaseRegion` op.
  builder.setInsertionPoint(op);
  auto case_op = builder.create<mhlo::CaseOp>(
      loc, op.getResultTypes(), op.getBranchIndex(), op.getBranches().size());
  for (auto region : llvm::zip(case_op.getBranches(), op.getBranches()))
    std::get<0>(region).takeBody(std::get<1>(region));

  // Replace all uses of `op` results with that of `mhlo.CaseOp`.
  op.replaceAllUsesWith(case_op);
  op.erase();
}

void LowerWhileRegion(TF::WhileRegionOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  builder.setInsertionPoint(op);

  // Create the new `mhlo.while` op with 'inputs'.
  auto while_result_types = llvm::to_vector<4>(op.getResultTypes());
  SmallVector<Value, 3> inputs(op.getInput());
  auto while_op =
      builder.create<mhlo::WhileOp>(loc, while_result_types, inputs);

  // Rewrite cond and associated block arguments and terminator. Ownership of
  // cond region is transfered over from `tf.WhileRegion` to `mhlo.while`.
  Region& cond = while_op.getCond();
  cond.takeBody(op.getCond());
  Block& cond_block = cond.front();
  builder.setInsertionPointToStart(&cond_block);

  // Cond always returns a single result of bool type.
  ReplaceTerminator(&cond_block, &builder);

  // Rewrite body and associated block arguments and terminator. Ownership of
  // body region is transfered over from `tf.WhileRegion` to `mhlo.while`.
  Region& body = while_op.getBody();
  body.takeBody(op.getBody());
  Block& body_block = body.front();
  builder.setInsertionPointToStart(&body_block);
  ReplaceTerminator(&body_block, &builder);

  // Replace all uses of `op` results with that of `mhlo.while`.
  builder.setInsertionPoint(op);
  if (while_op.getNumResults() > 1) {
    for (const auto& result_it : llvm::enumerate(op.getResults()))
      result_it.value().replaceAllUsesWith(
          while_op.getResult(result_it.index()));
  } else {
    op->replaceAllUsesWith(while_op);
  }
  op.erase();
}
}  // namespace

void LegalizeTFControlFlow::runOnOperation() {
  getOperation().walk([&](Operation* op) {
    if (auto while_region_op = dyn_cast<TF::WhileRegionOp>(op)) {
      LowerWhileRegion(while_region_op);
      return;
    }
    if (auto if_region_op = dyn_cast<TF::IfRegionOp>(op)) {
      LowerIfRegion(if_region_op);
      return;
    }
    if (auto case_region_op = dyn_cast<TF::CaseRegionOp>(op)) {
      LowerCaseRegion(case_region_op);
      return;
    }
  });
}
}  // namespace mhlo
}  // namespace mlir
