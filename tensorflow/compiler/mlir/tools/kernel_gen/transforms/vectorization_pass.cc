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

#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Analysis/BufferViewFlowAnalysis.h"  // from @llvm-project
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {
#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

// This function takes ForOps that contain AffineMinOps and possibly peels off
// the last iteration of the loop. This is done in cases where it is provable
// that the AffineMinOp is deterministic in all cases except the possible last
// iteration. Some additional cleanup is done to simplify the IR that is correct
// through knowledge of what this transformation is doing but would generally be
// unwieldy in a canonicalization-like pattern.
//
// This pass is only necessary due to inefficiencies in VectorTransferSplit that
// is unlikely to be fixed upstream. If that changes, this pass can be fully
// removed.
//
// Example:
// scf.for %i = %c0 to %c11 step %c2
//   %a = affine.min(%c2, %c11-%i)
//
// Becomes:
// scf.for %i = %c0 to %c10 step %c2
//   %a = %c2
// scf.if %one_more_iter
//   %a = affine.min(2, %c11-%i)
//
// This is possible because we can determine that the min will always be 2
// except for the last iteration.
void SplitSCFForOp(scf::ForOp scf_for) {
  // The set of following steps is:
  // 1. Validate that there are min_ops to be modified in this function.
  // 2. Create the boundary that decides whether the min_op evaluates to the
  // loop's step value or to the computed value based upon the iteration value.
  // 3. Create the primary loop that does all the work except for possibly the
  // last iteration of the loop, and replace all relevant min_ops with the step.
  // 4. Create the final iteration, remove the step from relevant min_ops, and
  // additionally modify related if/else ops to have a constant condition based
  // on what we know about this loop structure.

  // Match only when the lower bound is zero and the step is constant.
  // TODO(TPOPP): Requiring constant steps and lower bound simplifies things
  // but isn't necesarilly needed
  auto lower_bound_op =
      llvm::dyn_cast<arith::ConstantOp>(scf_for.lowerBound().getDefiningOp());
  if (!lower_bound_op) {
    return;
  }
  auto lower_bound_value = lower_bound_op.getValue().dyn_cast<IntegerAttr>();
  if (!lower_bound_value || lower_bound_value.getInt() != 0) {
    return;
  }

  auto step_bound_op =
      llvm::dyn_cast<arith::ConstantOp>(scf_for.step().getDefiningOp());
  if (!step_bound_op) {
    return;
  }
  auto step_bound_value = step_bound_op.getValue().dyn_cast<IntegerAttr>();
  if (!step_bound_value) {
    return;
  }

  auto loc = scf_for.getLoc();
  ImplicitLocOpBuilder b(loc, scf_for);

  // This function will determine if the min_op is an operation that can be
  // transformed after loop splitting. This relies on the function that the op
  // represents relative to the induction variable in its loop and the
  // bounds of the original for loop.
  auto is_op_of_interest = [&](AffineMinOp min_op, Value iv) {
    bool min_by_step = false;
    for (auto i : min_op.getAffineMap().getResults()) {
      if (i == b.getAffineConstantExpr(step_bound_value.getInt())) {
        min_by_step = true;
        continue;
      }
      if (i == b.getAffineSymbolExpr(0) - b.getAffineDimExpr(0) &&
          min_op.getDimOperands().front() == iv &&
          min_op.getSymbolOperands().front() == scf_for.upperBound())
        continue;
      if (i == b.getAffineDimExpr(0) - b.getAffineDimExpr(1) &&
          min_op.getDimOperands().drop_front().front() == iv &&
          min_op.getDimOperands().front() == scf_for.upperBound())
        continue;
      if (auto idx_op =
              scf_for.upperBound().getDefiningOp<arith::ConstantIndexOp>()) {
        auto val = idx_op.value();
        if (i == b.getAffineConstantExpr(val) - b.getAffineDimExpr(0) &&
            min_op.getDimOperands().front() == iv)
          continue;
      }
      return false;
    }
    return min_by_step;
  };

  // Determine if the loop should be split based on the existence of
  // AffineMinOps of an expected form.
  llvm::SmallVector<AffineMinOp, 1> min_ops;
  scf_for->walk([&](AffineMinOp min_op) {
    if (is_op_of_interest(min_op, scf_for.getInductionVar()))
      min_ops.push_back(min_op);
  });
  if (min_ops.empty()) {
    return;
  }

  // Split the loop just before a possible last iteration.
  b.setInsertionPoint(scf_for);
  Value split_point = b.create<arith::SubIOp>(
      scf_for.upperBound(),
      b.create<arith::RemUIOp>(
          b.create<arith::SubIOp>(scf_for.upperBound(), scf_for.lowerBound()),
          scf_for.step()));

  // New primary loop with relevant min ops replaced with their constant value
  BlockAndValueMapping mapper;
  auto new_loop = llvm::cast<scf::ForOp>(b.clone(*scf_for, mapper));
  new_loop.setUpperBound(split_point);

  new_loop->walk([&](AffineMinOp min_op) {
    if (is_op_of_interest(min_op, new_loop.getInductionVar()))
      min_op->replaceAllUsesWith(llvm::makeArrayRef(scf_for.step()));
  });

  // Peeled loop iteration (or nothing if perfectly aligned data and step sizes)
  BlockAndValueMapping tail_mapper;
  tail_mapper.map(scf_for.getRegionIterArgs(), new_loop.results());
  tail_mapper.map(scf_for.getInductionVar(), split_point);
  auto tail_if = b.create<scf::IfOp>(
      scf_for.getResultTypes(),
      b.create<arith::CmpIOp>(arith::CmpIPredicate::ult, split_point, scf_for.upperBound()),
      [&](OpBuilder &then_b, Location loc) {
        for (auto &op : *scf_for.getBody()) {
          then_b.clone(op, tail_mapper);
        }
      }, scf_for->getNumResults() ?
      [&](OpBuilder &else_b, Location loc) {
        else_b.clone(scf_for.getBody()->back(), tail_mapper);
      } : static_cast<function_ref<void(OpBuilder &, Location)>>(nullptr));

  tail_if->walk([&](AffineMinOp min_op) {
    SmallVector<AffineExpr> exprs;

    if (!is_op_of_interest(min_op, split_point)) return;

    ImplicitLocOpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(min_op);

    // This function is to be called on comparisons that use the min_ops of
    // interest in the last loop iteration. Through loop splitting, we know that
    // the min result is strictly less than the step value. Therefore, we can
    // take the predicate and a statement regarding the location of the min_op
    // (and the implied position of the step value) to evaluate the cmpi.
    auto is_true_cmp = [](arith::CmpIPredicate pred, bool min_is_op_0) {
      switch (pred) {
        // This loop splitting guarantees the step is not equal to the min on
        // the last iteration.
        case arith::CmpIPredicate::eq:
        case arith::CmpIPredicate::ne:
          return false;
        case arith::CmpIPredicate::sle:
        case arith::CmpIPredicate::slt:
        case arith::CmpIPredicate::ule:
        case arith::CmpIPredicate::ult:
          return min_is_op_0;
        case arith::CmpIPredicate::sge:
        case arith::CmpIPredicate::sgt:
        case arith::CmpIPredicate::uge:
        case arith::CmpIPredicate::ugt:
          return !min_is_op_0;
      }
    };

    for (auto user : min_op->getUsers()) {
      if (auto cmp = dyn_cast<arith::CmpIOp>(user)) {
        if (cmp.getOperand(0) == min_op.getResult() &&
            cmp.getOperand(1) == step_bound_op) {
          cmp.replaceAllUsesWith(b.create<arith::ConstantIntOp>(
                                      is_true_cmp(cmp.getPredicate(), true), 1)
                                     .getResult());
          cmp.erase();
        } else if (cmp.getOperand(0) == step_bound_op &&
                   cmp.getOperand(1) == min_op.getResult()) {
          cmp.replaceAllUsesWith(b.create<arith::ConstantIntOp>(
                                      is_true_cmp(cmp.getPredicate(), false), 1)
                                     .getResult());
        }
      }
    }

    // Replace the min_op with a simplified min_op that removes the constant
    // step option. This will be further simplified after affine ops are
    // lowered.
    auto map = min_op.getAffineMap();
    for (auto i : map.getResults()) {
      if (i != b.getAffineConstantExpr(step_bound_value.getInt()))
        exprs.push_back(i);
    }

    Value new_min = b.createOrFold<AffineMinOp>(
        AffineMap::get(map.getNumDims(), map.getNumSymbols(), exprs,
                       b.getContext()),
        min_op.operands());

    min_op->replaceAllUsesWith(llvm::makeArrayRef(new_min));
  });

  scf_for->replaceAllUsesWith(tail_if.results());
  scf_for.erase();
}

// A pass to remove memref::AllocOps and other ops interacting with the memrefs
// if it is provable that this will not change the results of the program. This
// is determined by confirming all consumers of all aliases are only creating an
// alias or writing data to an alias but never reading from or interacting with
// the memref in other ways.
void RemoveDeadMemrefCode(FuncOp func) {
  BufferViewFlowAnalysis baa(func);
  llvm::SmallSet<Operation *, 8> to_remove;

  // Gather all operations interacting with memrefs guaranteed to never be read
  // from.
  func->walk([&](memref::AllocaOp op) {
    llvm::SmallVector<Operation *> maybe_to_remove;
    for (auto &alias : baa.resolve(op.getResult())) {
      for (auto user : alias.getUsers()) {
        if (!(isa<ViewLikeOpInterface>(user) ||
              (isa<linalg::CopyOp>(user) &&
               alias == cast<linalg::CopyOp>(user).output()) ||
              (isa<linalg::FillOp>(user) &&
               alias == cast<linalg::FillOp>(user).output()))) {
          return;
        }
        maybe_to_remove.push_back(user);
      }
    }
    to_remove.insert(maybe_to_remove.begin(), maybe_to_remove.end());
    to_remove.insert(op);
  });

  // Erase after the walk to avoid corrupting data being traversed.
  for (auto *op : to_remove) {
    op->dropAllUses();
    op->erase();
  }
}

struct VectorizationPass : public VectorizationPassBase<VectorizationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnFunction() override {
    // This functions in 2 passes:
    // 1. Tile, promote, and vectorize to create elementwise operations on
    //    <(1x)*4xty> memrefs
    // 2. cast <(1x)*4xty> memrefs to <4xty>
    auto f = getFunction();

    // Stage 1: Vectorize to form static shaped computations
    auto tiling_options =
        linalg::LinalgTilingOptions().setTileSizeComputationFunction(
            [](OpBuilder b, Operation *op) {
              auto num_loops = llvm::cast<linalg::LinalgOp>(op).getNumLoops();
              SmallVector<Value> tiles(
                  num_loops, b.create<arith::ConstantIndexOp>(op->getLoc(), 1));
              if (!tiles.empty())
                tiles.back() =
                    b.create<arith::ConstantIndexOp>(op->getLoc(), 4);
              return tiles;
            });
    auto alignment = 16;

    mlir::linalg::CodegenStrategy strategy;
    strategy.tile(mlir::linalg::GenericOp::getOperationName(), tiling_options)
        .promote(mlir::linalg::GenericOp::getOperationName(),
                 mlir::linalg::LinalgPromotionOptions()
                     .setAlignment(alignment)
                     .setUseFullTileBuffersByDefault(true)
                     .setUseAlloca(true))
        .vectorize(mlir::linalg::GenericOp::getOperationName())
        .vectorLowering(
            mlir::linalg::LinalgVectorLoweringOptions()
                .enableTransferLowering(false)
                .enableTransferPartialRewrite()
                .setVectorTransformsOptions(
                    mlir::vector::VectorTransformsOptions()
                        .setVectorTransferSplit(
                            mlir::vector::VectorTransferSplit::VectorTransfer))
                .enableTransferToSCFConversion()
                .setVectorTransferToSCFOptions(
                    mlir::VectorTransferToSCFOptions().enableFullUnroll())
                .enableContractionLowering());

    // Created a nested OpPassManager, populate the strategy and run.
    OpPassManager dynamicPM("builtin.func");
    strategy.configurePassPipeline(dynamicPM, f.getContext());
    if (failed(runPipeline(dynamicPM, f))) return signalPassFailure();

    // Stage 2: Remove extent 1 dims to ensure correct 1-ranked vectorization
    auto ctx = f.getContext();
    OwningRewritePatternList patterns(ctx);
    mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<FunctionPass> CreateVectorizationPass() {
  return std::make_unique<VectorizationPass>();
}

struct VectorizationCleanupPass
    : public VectorizationCleanupPassBase<VectorizationCleanupPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnFunction() override {
    getFunction().walk([](scf::ForOp op) { SplitSCFForOp(op); });

    RemoveDeadMemrefCode(getFunction());
  }
};

std::unique_ptr<FunctionPass> CreateVectorizationCleanupPass() {
  return std::make_unique<VectorizationCleanupPass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
