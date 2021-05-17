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
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"  // from @llvm-project
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

  // Peeled loop iteration (or nothing if perfectly aligned data and step sizes)
  BlockAndValueMapping tail_mapper;
  tail_mapper.map(scf_for.getRegionIterArgs(), new_loop.results());
  tail_mapper.map(scf_for.getInductionVar(), split_point);
  auto tail_if = b.create<scf::IfOp>(
      scf_for.getResultTypes(),
      b.create<CmpIOp>(CmpIPredicate::ult, split_point, scf_for.upperBound()),
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
    auto is_true_cmp = [](CmpIPredicate pred, bool min_is_op_0) {
      switch (pred) {
        // This loop splitting guarantees the step is not equal to the min on
        // the last iteration.
        case CmpIPredicate::eq:
        case CmpIPredicate::ne:
          return false;
        case CmpIPredicate::sle:
        case CmpIPredicate::slt:
        case CmpIPredicate::ule:
        case CmpIPredicate::ult:
          return min_is_op_0;
        case CmpIPredicate::sge:
        case CmpIPredicate::sgt:
        case CmpIPredicate::uge:
        case CmpIPredicate::ugt:
          return !min_is_op_0;
      }
    };

    for (auto user : min_op->getUsers()) {
      if (auto cmp = dyn_cast<CmpIOp>(user)) {
        if (cmp.getOperand(0) == min_op.getResult() &&
            cmp.getOperand(1) == step_bound_op) {
          cmp.replaceAllUsesWith(
              b.create<ConstantIntOp>(is_true_cmp(cmp.predicate(), true), 1)
                  .getResult());
          cmp.erase();
        } else if (cmp.getOperand(0) == step_bound_op &&
                   cmp.getOperand(1) == min_op.getResult()) {
          cmp.replaceAllUsesWith(
              b.create<ConstantIntOp>(is_true_cmp(cmp.predicate(), false), 1)
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
  func->walk([&](memref::AllocOp op) {
    llvm::SmallVector<Operation *> maybe_to_remove;
    for (auto &alias : baa.resolve(op.getResult())) {
      for (auto user : alias.getUsers()) {
        if (!(isa<ViewLikeOpInterface>(user) || isa<memref::DeallocOp>(user) ||
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

// A pattern to remove extent 1 dimensions from linalg.generic inputs. This is
// used to create vectorized operations of the correct rank.
//
// For example:
// linalg.generic {indexing_maps =
//   [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
//    affine_map<(d0, d1) -> (d0, d1)>],
//   iterator_types = ["parallel", "parallel"]}
//  ins(%lhs, %rhs
//   : memref<1x4xf64, affine_map<(d0, d1)[s0, s1] -> (d0 * s0 + d1 * s1)>>,
//   memref<1x4xf64, affine_map<(d0, d1)[s0, s1] -> (d0 * s0 + d1 * s1)>>)
//  outs(%out : memref<1x4xf64>) {
// ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):  // no predecessors
//   %65 = addf %arg2, %arg3 : f64
//   linalg.yield %65 : f64
// }
//
// Becomes:
// %newLhs = memref.reshape %lhs : 2d -> 1d
// %newRhs = memref.reshape %lhs : 2d -> 1d
// %newOut = memref.reshape %lhs : 2d -> 1d
// linalg.generic {indexing_maps =
//   [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
//    affine_map<(d0) -> (d0)>],
//   iterator_types = ["parallel", "parallel"]}
//  ins(%newLhs, %newRhs
//   : memref<4xf64, affine_map<(d0)[s0] -> (d0 * s0)>>,
//   memref<4xf64, affine_map<(d0)[s0] -> (d0 * s0)>>)
//  outs(%newOut : memref<4xf64>) {
// ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):  // no predecessors
//   %65 = addf %arg2, %arg3 : f64
//   linalg.yield %65 : f64
// }
struct RemoveExtent1DimsPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &b) const override {
    if (!op.hasBufferSemantics()) return failure();
    if (op.getNumOutputBuffers() != 1) return failure();
    if (!llvm::all_of(op.getIndexingMaps(),
                      [](auto map) { return map.isIdentity(); }))
      return failure();
    if (op.getNumReductionLoops()) return failure();

    auto trim_and_reshape_or_null = [&](Value buffer) -> Value {
      auto type = buffer.getType().dyn_cast<MemRefType>();
      if (!(type && type.hasRank() && type.getNumDynamicDims() == 0 &&
            type.getAffineMaps().empty()))
        return nullptr;

      SmallVector<int64_t> new_shape;
      llvm::copy_if(type.getShape(), std::back_inserter(new_shape),
                    [](auto el) { return el != 1; });

      // If nothing would be changed, don't execute the reinterpret_cast
      if (new_shape.size() == type.getShape().size()) {
        return buffer;
      }

      auto loc = op.getLoc();
      auto extent_memref_type = MemRefType::get(
          llvm::makeArrayRef(static_cast<int64_t>(new_shape.size())),
          b.getIndexType());
      Value shape_tensor =
          b.create<ConstantOp>(loc, b.getIndexTensorAttr(new_shape));
      Value constant_shape =
          b.create<memref::BufferCastOp>(loc, extent_memref_type, shape_tensor);
      auto new_type = MemRefType::get(new_shape, type.getElementType());
      return b.create<memref::ReshapeOp>(loc, new_type, buffer, constant_shape)
          .result();
    };

    // Gather the buffers for the new operations
    SmallVector<Value> new_inputs, new_outputs;
    for (auto &in : op.getInputBuffers()) {
      if (auto buffer = trim_and_reshape_or_null(in)) {
        new_inputs.push_back(buffer);
      } else {
        return failure();
      }
    }
    for (auto &out : op.getOutputBuffers()) {
      if (auto buffer = trim_and_reshape_or_null(out)) {
        new_outputs.push_back(buffer);
      } else {
        return failure();
      }
    }

    // Quit if nothing was changed
    auto changed_operand = [&](auto pair) {
      return std::get<0>(pair) != std::get<1>(pair);
    };
    if (llvm::none_of(llvm::zip(new_inputs, op.getInputBuffers()),
                      changed_operand) &&
        llvm::none_of(llvm::zip(new_outputs, op.getOutputBuffers()),
                      changed_operand))
      return failure();

    auto num_loops = new_outputs.front().getType().cast<MemRefType>().getRank();

    // Replace with the simplified op
    auto indexing_maps = llvm::SmallVector<AffineMap>(
        op->getNumOperands(),
        AffineMap::getMultiDimIdentityMap(num_loops, b.getContext()));
    auto iterator_types = llvm::SmallVector<StringRef>(num_loops, "parallel");
    auto new_op = b.create<linalg::GenericOp>(
        op.getLoc(), new_inputs, new_outputs, indexing_maps, iterator_types);
    b.inlineRegionBefore(op.getRegion(), new_op.getRegion(),
                         new_op.getRegion().end());
    b.replaceOp(op, new_op->getResults());

    return success();
  }
};

struct VectorizationPass : public VectorizationPassBase<VectorizationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnFunction() override {
    // This functions in 3 passes:
    // 1. Tile and promote to create generic operations on <1x1x*x4xty> memrefs
    // 2. cast <1x1x*x4xty> memrefs to <4xty>
    // 2. vectorize the new <4xty> generics to 1d vector operations.
    auto f = getFunction();
    auto ctx = f.getContext();

    // Stage 1: Tile and Promote to form static shaped computations
    auto tiling_options =
        linalg::LinalgTilingOptions().setTileSizeComputationFunction(
            [](OpBuilder b, Operation *op) {
              auto num_loops = llvm::cast<linalg::LinalgOp>(op).getNumLoops();
              SmallVector<Value> tiles(
                  num_loops, b.create<ConstantIndexOp>(op->getLoc(), 1));
              tiles.back() = b.create<ConstantIndexOp>(op->getLoc(), 4);
              return tiles;
            });
    auto alignment = 16;
    mlir::linalg::CodegenStrategy()
        .tile<mlir::linalg::GenericOp>(tiling_options)
        .promote<mlir::linalg::GenericOp>(
            mlir::linalg::LinalgPromotionOptions()
                .setAlignment(alignment)
                .setUseFullTileBuffersByDefault(true)
                .setUseAlloca(false))
        .transform(f);

    // Stage 2: Remove extent 1 dims to ensure correct 1-ranked vectorization
    OwningRewritePatternList patterns(ctx);
    patterns.insert<RemoveExtent1DimsPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(f, std::move(patterns));

    // Stage 3: Vectorize the simplified linalg.generic ops
    mlir::linalg::CodegenStrategy()
        .vectorize<mlir::linalg::GenericOp>()
        .setVectorTransformsOptions(
            mlir::vector::VectorTransformsOptions().setVectorTransferSplit(
                mlir::vector::VectorTransferSplit::VectorTransfer))
        .setVectorTransferToSCFOptions(
            mlir::VectorTransferToSCFOptions().setUnroll(true))
        .transform(f);
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
