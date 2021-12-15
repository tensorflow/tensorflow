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

// This file implements inline fusion.
//
#include <limits>

#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/lhlo/transforms/lhlo_elemental_utils.h"
#include "mlir-hlo/Dialect/lhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopFusionUtils.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

using mlir::memref::LoadOp;

namespace mlir {
namespace lmhlo {

#define GEN_PASS_CLASSES
#include "mlir-hlo/Dialect/lhlo/transforms/lmhlo_passes.h.inc"

namespace {

// TODO(disc): Maybe it worth explicitly adding the I/O buffers onto the
// outlining of lmhlo::FusionOp and then mark IsolatedFromAbove for
// lmhlo::FusionOp. By this way the fusion codegen passes can be OperationPass
// on lmhlo::FusionOp for better compilation overhead.
class InputInlineFusion : public InputInlineFusionPassBase<InputInlineFusion> {
  void runOnFunction() override;
};

}  // end anonymous namespace

std::unique_ptr<FunctionPass> createInputInlineFusionPass() {
  return std::make_unique<InputInlineFusion>();
}

namespace {

constexpr unsigned c_MAX_ITERATION = 4096;

// This pass works after LhloLegalizeRootsToParallelLoops pass for the
// XLA-style fusion codegen.
//
// It iteratively looks for the lmhlo op which is the direct producer of the
// nested loops, and then inline fuse it if the fusion will not form a cycle.
//
// The inline fusion action can be generalized as:
// step 1: replace the producer Lhlo op into associate std op inside the nested
// loops. step 2: remove the original Load ops inside the loops and insert new
// Load ops.
//
// If there are multiple LoadOps with the same indices, they will be replaced
// with the same op. This obtains the similar result as GeneratedValueCache.
//
// IR after LhloLegalizeRootsToParallelLoops:
//    "lmhlo.fusion"() ( {
//       lmhlo.aaa(%0, %1, %2)
//       lmhlo.bbb(%2, %3, %4)
//       scf.parallel (...) {
//          memref.load %4[...]
//          ...
//          memref.store ...
//       }
//    })
//
// IR after one round of InputInlineFusionPattern:
//    "lmhlo.fusion"() ( {
//       lmhlo.aaa(%0, %1, %2)
//       scf.parallel (...) {
//          memref.load %2[...]
//          ...
//          memref.store ...
//       }
//    })
//
// Final IR after this pass:
//    "lmhlo.fusion"() ( {
//       scf.parallel (...) {
//          memref.load ...
//          ...
//          memref.store ...
//       }
//    })
class InputInlineFusionPattern : public RewritePattern {
 public:
  explicit InputInlineFusionPattern(MLIRContext* context)
      : RewritePattern(FusionOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    // skip if not the most outter ParallelOp
    auto fusion = cast<FusionOp>(op);
    auto& parent_block = fusion.region().front();
    SmallVector<scf::ParallelOp, 4> parallel_ops;
    fusion.walk([&](scf::ParallelOp parallel_op) {
      parallel_ops.push_back(parallel_op);
    });
    assert(parallel_ops.size() == 1 &&
           "only one scf::ParallelOp is expected after "
           "LhloLegalizeRootsToParallelLoops");
    scf::ParallelOp parallel_op = parallel_ops.front();
    SmallVector<LoadOp, 4> load_ops;
    parallel_op->walk([&](LoadOp load_op) { load_ops.push_back(load_op); });
    for (auto load_op : load_ops) {
      auto lhlo_op = getFusibleOperation(load_op);
      if (!lhlo_op) continue;
      // 1, in case of:
      //      A = ...
      //      B = op(A)
      //      C = op(A, B)
      //    C should fuse B first before fusing A.
      //    This is the same logic as in instruction_fusion pass of XLA
      //
      // 2, When multiple loads consumes the same result of lhlo_op and
      //    the load indices are also identical, the ir should be
      //    emitted only once. Other LoadOps should used cached Value.

      // 'load_ops' that can consume the same cached value
      SmallVector<LoadOp> same_load_ops;
      bool can_remove_producer;
      if (!checkIfFusible(parallel_op, lhlo_op, load_op, can_remove_producer,
                          same_load_ops))
        continue;
      // 'load_op' is always the one that locates in the most
      // external code block among all the 'same_load_ops', because the walker
      // is in the post order sequence.
      if (failed(inlineFuseLhloOp(rewriter, parallel_op, lhlo_op, load_op,
                                  same_load_ops)))
        return failure();
      if (can_remove_producer) rewriter.eraseOp(lhlo_op);
      for (LoadOp to_be_removed : same_load_ops)
        rewriter.eraseOp(to_be_removed);

      // Clean all the ops that do not have LoadOps inside the nested
      // ParallelOps and is not the ancestor of any ops that have LoadOps
      // inside the nested ParallelOps.
      cleanUnusedLhloOps(&parent_block);

      return success();
    }
    return failure();
  }

 private:
  Operation* getFusibleOperation(LoadOp load_op) const;
  LogicalResult inlineFuseLhloOp(PatternRewriter& b, Operation* user,
                                 Operation* producer, LoadOp load_op,
                                 const SmallVector<LoadOp>& load_ops) const;
  bool checkIfFusible(scf::ParallelOp user, Operation* producer, LoadOp load_op,
                      bool& can_remove_producer,
                      SmallVector<LoadOp>& load_ops) const;
};

Operation* InputInlineFusionPattern::getFusibleOperation(LoadOp load_op) const {
  Operation* lhlo_op = nullptr;
  for (auto* user : load_op.getMemRef().getUsers()) {
    if (isa<LmhloOp>(user) && (cast<LmhloOp>(user).getResultBuffer() ==
                               load_op.getOperation()->getOperand(0))) {
      if (lhlo_op)
        llvm::report_fatal_error(
            "More than one lhlo_op write to one Memref within one fusion");
      lhlo_op = user;
    }
  }
  return lhlo_op;
}

// Check if there are no other consumers of the producer
// except the ParallelOp.
bool InputInlineFusionPattern::checkIfFusible(
    scf::ParallelOp user, Operation* producer, LoadOp load_op,
    bool& can_remove_producer, SmallVector<LoadOp>& load_ops) const {
  load_ops.clear();
  assert(isa<LmhloOp>(producer) && "Unexpected producer in checkIfFusible");
  auto producer_result_memref = cast<LmhloOp>(producer).getResultBuffer();
  can_remove_producer = true;
  auto lhlo_dialect = user->getContext()->getLoadedDialect("lmhlo");
  for (auto* memref_user : producer_result_memref.getUsers()) {
    if ((memref_user->getDialect() == lhlo_dialect) &&
        (memref_user != producer)) {
      return false;
    }
    LoadOp other = dyn_cast<LoadOp>(memref_user);
    if (!other) continue;
    if (other.getMemRef() == load_op.getMemRef() &&
        other.getIndices() == load_op.getIndices())
      load_ops.emplace_back(other);
    else
      can_remove_producer = false;
    // TODO(disc): check the memref_user is inside the loops
  }
  return true;
}

template <typename LHLO_OpTy>
bool elemwiseFuseHelper(PatternRewriter& rewriter, Operation* user,
                        Operation* producer, LoadOp load_op,
                        const SmallVector<LoadOp>& load_ops) {
  if (!isa<LHLO_OpTy>(producer) ||
      !LHLO_OpTy::template hasTrait<OpTrait::Elementwise>())
    return false;
  auto loc = user->getLoc();
  SmallVector<Value, 4> operand_values;
  unsigned num_operands = producer->getNumOperands();
  for (unsigned i = 0; i < num_operands - 1; ++i) {
    auto producer_operand = producer->getOperand(i);
    rewriter.setInsertionPoint(load_op);
    operand_values.push_back(
        rewriter.create<LoadOp>(loc, producer_operand, load_op.getIndices()));
  }
  auto inlined_result =
      LhloOpToStdScalarOp::map<LHLO_OpTy>(llvm::cast<LHLO_OpTy>(producer),
                                          cast<LmhloOp>(producer)
                                              .getResultBuffer()
                                              .getType()
                                              .cast<MemRefType>()
                                              .getElementType(),
                                          operand_values, &rewriter);

  for (LoadOp to_be_replaced : load_ops)
    to_be_replaced.replaceAllUsesWith(inlined_result);
  return true;
}

template <typename LHLO_OpTy>
bool miscFuseHelper(PatternRewriter& rewriter, Operation* user,
                    Operation* opaque_producer, LoadOp load_op,
                    const SmallVector<LoadOp>& load_ops) {
  LHLO_OpTy producer = dyn_cast<LHLO_OpTy>(opaque_producer);
  if (!producer) return false;
  auto loc = user->getLoc();
  rewriter.setInsertionPoint(load_op);
  auto inlined_result =
      elementalLower<LHLO_OpTy>(&rewriter, loc, producer, load_op.getIndices());
  for (LoadOp to_be_replaced : load_ops)
    to_be_replaced.replaceAllUsesWith(inlined_result);
  return true;
}

template <>
bool miscFuseHelper<ConstOp>(PatternRewriter& rewriter, Operation* user,
                             Operation* producer, LoadOp load_op,
                             const SmallVector<LoadOp>& load_ops) {
  if (!isa<ConstOp>(producer)) return false;
  auto memref_type =
      cast<LmhloOp>(producer).getResultBuffer().getType().cast<MemRefType>();
  assert(memref_type.getRank() == 0 && "only scalar ConstOp can be fused");
  auto loc = user->getLoc();
  rewriter.setInsertionPoint(load_op);
  Value inlined_result = rewriter.create<arith::ConstantOp>(
      loc, memref_type.getElementType(),
      cast<ConstOp>(producer).value().getValues<Attribute>()[0]);
  for (LoadOp to_be_replaced : load_ops)
    to_be_replaced.replaceAllUsesWith(inlined_result);
  return true;
}

template <typename First>
bool elemwiseFuseHelperOr(PatternRewriter& rewriter, Operation* user,
                          Operation* producer, LoadOp load_op,
                          const SmallVector<LoadOp>& load_ops) {
  return elemwiseFuseHelper<First>(rewriter, user, producer, load_op, load_ops);
}

template <typename First, typename Second, typename... Rest>
bool elemwiseFuseHelperOr(PatternRewriter& rewriter, Operation* user,
                          Operation* producer, LoadOp load_op,
                          const SmallVector<LoadOp>& load_ops) {
  return elemwiseFuseHelperOr<First>(rewriter, user, producer, load_op,
                                     load_ops) ||
         elemwiseFuseHelperOr<Second, Rest...>(rewriter, user, producer,
                                               load_op, load_ops);
}

// load_op is among the load_ops, whose locates in the most
// external code block
LogicalResult InputInlineFusionPattern::inlineFuseLhloOp(
    PatternRewriter& b, Operation* user, Operation* producer, LoadOp load_op,
    const SmallVector<LoadOp>& load_ops) const {
  if (elemwiseFuseHelperOr<
#define GET_SUPPORTED_OP_LIST
#include "mlir-hlo/utils/disc_supported_list.h.inc"
          >(b, user, producer, load_op, load_ops) ||
      // TODO(disc): Upstream is on the way for more Ops
      miscFuseHelper<RealDynamicSliceOp>(b, user, producer, load_op,
                                         load_ops) ||
      miscFuseHelper<DynamicBroadcastInDimOp>(b, user, producer, load_op,
                                              load_ops) ||
      miscFuseHelper<BroadcastInDimOp>(b, user, producer, load_op, load_ops) ||
      miscFuseHelper<ConstOp>(b, user, producer, load_op, load_ops)) {
    return success();
  }

  return failure();
}

void InputInlineFusion::runOnFunction() {
  auto func = getFunction();
  auto* context = &this->getContext();
  OwningRewritePatternList patterns(context);
  patterns.insert<InputInlineFusionPattern>(context);

  // Just apply the patterns greedily.
  // There should always be one scf.ParallelOp in the fusion.
  auto config = GreedyRewriteConfig();
  config.maxIterations = c_MAX_ITERATION;
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
    signalPassFailure();
  }

  // there should be no lmhlo ops after inline fusion,
  // except for the ConstOp of ColReduction, which for now cannot be
  // properly optimized by general DCE pass
  std::vector<Operation*> to_be_removed;
  func.walk([&](FusionOp fusion) {
    fusion.region().walk([&](LmhloOp op) {
      if (isa<TerminatorOp>(op)) {
        return;
      }
      if (isa<ConstOp>(op)) {
        // TODO(disc): Check the ConstOp is from ReduceOp
        to_be_removed.push_back(op);
        return;
      }
      op.emitError("unexpected remaining operation in a FusionOp");
      signalPassFailure();
    });
  });
  for (auto op : to_be_removed) {
    op->erase();
  }
}

}  // namespace

}  // namespace lmhlo
}  // namespace mlir
