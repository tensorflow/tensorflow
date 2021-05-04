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
#include "mlir/Analysis/BufferAliasAnalysis.h"  // from @llvm-project
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {
#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

// This pattern takes ForOps that contain AffineMinOps and possibly peels off
// the last iteration of the loop. This is done in cases where it is provable
// that the AffineMinOp is deterministic in all cases except the possible last
// iteration.
//
// Example:
// scf.for %i = %c0 to %c11 step %c2
//   %a = affine.min(%c2, %c11-%i)
//
// Becomes:
// scf.for %i = %c0 to %c10 step %c2
//   %a = %c2
// scf.for %i = %c10 to %c11 step %c2
//   %a = affine.min(2, %c11-%i)
//
// This is possible because we can determine that the min will always be 2
// except for the last iteration.
void SplitSCFForOp(scf::ForOp scf_for) {
  // Match only when the lower bound is zero and the step is constant.
  // TODO(TPOPP): Requiring constant steps and lower bound simplifies things
  // but isn't necesarilly needed
  auto lower_bound_op =
      llvm::dyn_cast<ConstantOp>(scf_for.lowerBound().getDefiningOp());
  if (!lower_bound_op) {
    return;
  }
  auto lower_bound_value = lower_bound_op.getValue().dyn_cast<IntegerAttr>();
  if (!lower_bound_value || lower_bound_value.getInt() != 0) {
    return;
  }

  auto step_bound_op =
      llvm::dyn_cast<ConstantOp>(scf_for.step().getDefiningOp());
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
    if (min_op.getDimOperands().size() + min_op.getSymbolOperands().size() != 2)
      return false;
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
  Value split_point = b.create<SubIOp>(
      scf_for.upperBound(),
      b.create<UnsignedRemIOp>(
          b.create<SubIOp>(scf_for.upperBound(), scf_for.lowerBound()),
          scf_for.step()));

  // New primary loop with relevant min ops replaced with their constant value
  BlockAndValueMapping mapper;
  auto new_loop = llvm::cast<scf::ForOp>(b.clone(*scf_for, mapper));
  new_loop.setUpperBound(split_point);

  new_loop->walk([&](AffineMinOp min_op) {
    if (is_op_of_interest(min_op, new_loop.getInductionVar()))
      min_op->replaceAllUsesWith(llvm::makeArrayRef(scf_for.step()));
  });

  // Tail loop
  // TODO(TPOPP): Remove AffineMinOps and if/else statements because this only
  // executes if the result is less than the step size. This is for simpler
  // code rather than appreciable performance improvements with any large
  // inputs.
  BlockAndValueMapping tail_mapper;
  tail_mapper.map(scf_for.initArgs(), new_loop.results());
  auto tail_loop = llvm::cast<scf::ForOp>(b.clone(*scf_for, tail_mapper));
  tail_loop.setLowerBound(split_point);

  scf_for->replaceAllUsesWith(tail_loop.results());
  scf_for.erase();
}

struct VectorizationPass : public VectorizationPassBase<VectorizationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnFunction() override {
    mlir::linalg::LinalgTilingOptions tiling_options;
    tiling_options =
        tiling_options.setTileSizes(llvm::makeArrayRef<int64_t>(4));
    auto alignment = 16;
    mlir::linalg::CodegenStrategy strategy;
    strategy.tile<mlir::linalg::GenericOp>(tiling_options)
        .promote<mlir::linalg::GenericOp>(
            mlir::linalg::LinalgPromotionOptions()
                .setAlignment(alignment)
                .setUseFullTileBuffersByDefault(true)
                .setUseAlloca(false))
        .vectorize<mlir::linalg::GenericOp>()
        .setVectorTransformsOptions(
            mlir::vector::VectorTransformsOptions().setVectorTransferSplit(
                mlir::vector::VectorTransferSplit::VectorTransfer))
        .setVectorTransferToSCFOptions(
            mlir::VectorTransferToSCFOptions().setUnroll(true));
    strategy.transform(getFunction());
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
  }
};

std::unique_ptr<FunctionPass> CreateVectorizationCleanupPass() {
  return std::make_unique<VectorizationCleanupPass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
