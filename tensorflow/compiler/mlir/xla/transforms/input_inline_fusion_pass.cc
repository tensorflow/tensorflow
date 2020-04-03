//===- LoopFusion.cpp - Code to perform loop fusion -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop fusion.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopFusionUtils.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/Dialect/LoopOps/EDSC/Builders.h" 
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/map_xla_to_scalar_op.h"
#include <iostream>

namespace mlir {
namespace xla_lhlo {

namespace {

struct InputInlineFusion : public FunctionPass<InputInlineFusion> {
  void runOnFunction() override;
};

} // end anonymous namespace

std::unique_ptr<OpPassBase<FuncOp>> createInputInlineFusionPass() {
  return std::make_unique<InputInlineFusion>();
}

namespace {

class InputInlineFusionPattern: public RewritePattern {
 public:
  explicit InputInlineFusionPattern(MLIRContext *context)
      : RewritePattern(loop::ForOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // skip if not the most outter ForOp
    auto parent = op->getParentOp();
    if (!isa<FuncOp>(parent)) {
      return failure();
    }


    SmallVector<mlir::LoadOp, 4> load_ops;
    op->walk([&](mlir::LoadOp load_op) {
      load_ops.push_back(load_op);
    });

    for (auto load_op : load_ops) {
      auto lhlo_op = getFusibleOperation(load_op);
      if (lhlo_op != nullptr) {
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
        std::vector<Operation*> load_ops;
        bool can_remove_producer;
        if (!checkIfFusible(op, lhlo_op, load_op,
              can_remove_producer, load_ops)) {
          continue;
        }

        // 'load_op' is always the one that locates in the most 
        // external code block among all the 'load_ops', because the walker
        // is in the post order sequence.
        if (!failed(inlineFuseLhloOp(rewriter, op, lhlo_op,
                load_op, load_ops))) {
          if (can_remove_producer) {
            rewriter.eraseOp(lhlo_op);
          }
          for (auto* to_be_removed : load_ops) {
            rewriter.eraseOp(to_be_removed);
          }
          op->getParentOp()->dump();
          return success();
        } else {
          assert(false && "inlineFuseLhloOp failed!");
        }
      }
    }
    return failure();
  }

 private:
  Operation* getFusibleOperation(mlir::LoadOp load_op) const;
  LogicalResult inlineFuseLhloOp(
      PatternRewriter& b, Operation* user,
      Operation* producer, mlir::LoadOp load_op,
      std::vector<Operation*> load_ops) const;
  bool checkIfFusible(Operation* user, Operation* producer,
      mlir::LoadOp load_op, bool& can_remove_producer,
      std::vector<Operation*>& load_ops) const;

  FuncOp func_;
};

Operation* InputInlineFusionPattern::getFusibleOperation(
    mlir::LoadOp load_op) const {
  Operation* lhlo_op = nullptr;
  for (auto *user : load_op.getMemRef().getUsers()) {
    if ((user->getDialect()->getNamespace() == "xla_lhlo") &&
        (user->getOperand(user->getNumOperands() - 1) ==
         load_op.getOperation()->getOperand(0))) {
      if (lhlo_op == nullptr) {
        lhlo_op = user;
      } else {
        assert(false &&
            "More than one lhlo_op write to one Memref within one fusion");
      }
    }
  }
  return lhlo_op;
}

// Check if there are no other consumers of the producer
// except the ForOp.
bool InputInlineFusionPattern::checkIfFusible(
    Operation* user, Operation* producer, mlir::LoadOp load_op,
    bool& can_remove_producer,
    std::vector<Operation*>& load_ops) const {
  assert(isa<loop::ForOp>(user) &&
      "user is expected to be loop::ForOp");
  load_ops.clear();
  auto producer_result_memref =
      producer->getOperand(producer->getNumOperands() - 1); 
  can_remove_producer = true;
  auto lhlo_dialect = user->getContext()->getRegisteredDialect("xla_lhlo");
  for (auto* memref_user : producer_result_memref.getUsers()) {
    if ((memref_user->getDialect() == lhlo_dialect) &&
        (memref_user != producer)) {
      return false;
    } else if (isa<mlir::LoadOp>(memref_user)) {
      // Check if the loads the same indices
      if (memref_user->getOperands() ==
          load_op.getOperation()->getOperands()) { 
        load_ops.emplace_back(memref_user);
      } else {
        can_remove_producer = false;
      }
      //TODO:: check the memref_user is inside the loops
    }
  }
  return true;
}

template <typename LHLO_OpTy>
bool fuseHelper(
    PatternRewriter &rewriter, Operation* user, Operation* producer,
    mlir::LoadOp load_op, std::vector<Operation*>& load_ops) {
  if (!isa<LHLO_OpTy>(producer)) {
    return false;
  }
  auto loc = user->getLoc();
  auto lhlo_op = cast<LHLO_OpTy>(producer);
  SmallVector<Value, 4> operand_values;
  // the last operand is the output memref
  for (int i=0; i < producer->getNumOperands() - 1; ++i) {
    auto producer_operand = producer->getOperand(i); 
    // create an AffineMap and affine.apply it
    rewriter.setInsertionPoint(load_op);
    auto memref_type =
        producer->getOperand(producer->getNumOperands()-1)
        .getType().cast<MemRefType>();
    auto rank = memref_type.getRank();
    auto affine_map = rewriter.getMultiDimIdentityMap(rank);
    SmallVector<Value, 4> operand_indices;
    for (int dim=0; dim < rank; dim++) {
      auto affine_apply_op = rewriter.create<AffineApplyOp>(
          loc, affine_map.getSubMap({dim}), load_op.getIndices());
      operand_indices.push_back(affine_apply_op);
    }
    operand_values.push_back(rewriter.create<LoadOp>(loc,
        producer_operand, operand_indices));
  }
  auto inlined_result = xla_lhlo::XlaOpToStdScalarOp::map<LHLO_OpTy>(
      llvm::cast<LHLO_OpTy>(producer),
      producer->getOperand(2).getType().cast<MemRefType>().getElementType(),
      operand_values, &rewriter);
  for (auto* to_be_replaced : load_ops) {
    llvm::cast<mlir::LoadOp>(to_be_replaced).
        replaceAllUsesWith(inlined_result);
  }
  return true;
}

// load_op is among the load_ops, whose locates in the most
// external code block
LogicalResult InputInlineFusionPattern::inlineFuseLhloOp(
    PatternRewriter& b, Operation* user, Operation* producer,
    mlir::LoadOp load_op, std::vector<Operation*> load_ops) const {
  if (fuseHelper<xla_lhlo::AddOp>(b, user, producer, load_op, load_ops) ||
      fuseHelper<xla_lhlo::SubOp>(b, user, producer, load_op, load_ops) ||
      fuseHelper<xla_lhlo::MulOp>(b, user, producer, load_op, load_ops) ||
      fuseHelper<xla_lhlo::DivOp>(b, user, producer, load_op, load_ops)) {
    return success();
  } else {
    assert(false && "unsupported lhlo_op");
  }

  return success();
}

void InputInlineFusion::runOnFunction() {

  auto func = getFunction();
  if (func.getBlocks().size() == 0) {
    return;
  }
  // TODO(pifon): Remove assumption that the function has a single block.
  if (func.getBlocks().size() != 1) {
    emitError(func.getLoc(), "The function needs to have a single block.");
    signalPassFailure();
    return;
  }

  // TODO: how to tell that this func is a fusion func?

  auto *context = &this->getContext();
  OwningRewritePatternList patterns;
  patterns.insert<InputInlineFusionPattern>(context);

  // Just apply the patterns greedily. 
  // There should always be one loop.ForOp in the func.
  applyPatternsGreedily(func, patterns);

}

} // namespace

static PassRegistration<InputInlineFusion> pass("input-inline-fusion", "greedy inline fusion");

}  // namespace xla_lhlo
}  // namespace mlir
