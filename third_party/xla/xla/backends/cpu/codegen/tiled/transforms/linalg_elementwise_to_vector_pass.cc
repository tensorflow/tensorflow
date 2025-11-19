/* Copyright 2025 The OpenXLA Authors.

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

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"

namespace xla::cpu {

#define GEN_PASS_DECL_LINALGELEMENTWISETOVECTORPASS
#define GEN_PASS_DEF_LINALGELEMENTWISETOVECTORPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

class ElementwiseToVectorPattern
    : public mlir::OpInterfaceRewritePattern<mlir::linalg::LinalgOp> {
 public:
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::LinalgOp op,
      mlir::PatternRewriter& rewriter) const override {
    if (!mlir::linalg::isElementwise(op)) {
      return rewriter.notifyMatchFailure(op, "Op is not elementwise");
    }

    // Is this possible?
    if (op.getDpsInits().empty()) {
      return rewriter.notifyMatchFailure(op, "op has no outputs");
    }

    auto result_type =
        mlir::dyn_cast<mlir::ShapedType>(op.getDpsInits().front().getType());
    if (!result_type) {
      return rewriter.notifyMatchFailure(op, "could not convert result type");
    }
    if (!result_type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "only static shapes are supported");
    }

    if (result_type.getRank() == 0) {
      mlir::FailureOr<mlir::linalg::VectorizationResult> result =
          mlir::linalg::vectorize(rewriter, op);
      if (failed(result)) {
        return rewriter.notifyMatchFailure(op, "failed to vectorize");
      }
      rewriter.replaceOp(op, result->replacements);
      return mlir::success();
    }

    int64_t vector_size = 8;
    auto minor_dim_size = result_type.getShape().back();
    while (vector_size > minor_dim_size) {
      vector_size >>= 1;
    }
    auto get_vector_type = [vector_size](mlir::Type type) {
      return mlir::VectorType::get({vector_size}, type);
    };
    llvm::SmallVector<mlir::Value> lbs(
        result_type.getRank(),
        mlir::arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0));
    auto ubs = llvm::map_to_vector(
        result_type.getShape(), [&](int64_t size) -> mlir::Value {
          return mlir::arith::ConstantIndexOp::create(rewriter, op.getLoc(),
                                                      size);
        });
    llvm::SmallVector<mlir::Value> step(
        result_type.getRank(),
        mlir::arith::ConstantIndexOp::create(rewriter, op.getLoc(), 1));

    step.back() = mlir::arith::ConstantIndexOp::create(rewriter, op.getLoc(),
                                                       vector_size);
    llvm::SmallVector<mlir::OpOperand*> body_operands =
        op.getOpOperandsMatchingBBargs();

    mlir::IRMapping ir_mapping;
    llvm::SetVector<mlir::Value> valuesSet;
    mlir::getUsedValuesDefinedAbove(op->getRegion(0), valuesSet);
    ir_mapping.map(valuesSet.getArrayRef(), valuesSet.getArrayRef());

    auto make_in_bounds = [vector_size](mlir::Type type) {
      auto shaped_type = mlir::cast<mlir::ShapedType>(type);
      llvm::ArrayRef<int64_t> shape = shaped_type.getShape();
      llvm::SmallVector<bool> result(1, shape.back() % vector_size == 0);
      return result;
    };

    // TODO: Peel last inner loop iteration if not a multiple of vector size.
    mlir::Block* block = op.getBlock();
    mlir::scf::buildLoopNest(
        rewriter, op.getLoc(), lbs, ubs, step,
        [&](mlir::OpBuilder& builder, mlir::Location loc,
            mlir::ValueRange induction_vars) {
          for (mlir::OpOperand* op_operand : body_operands) {
            mlir::BlockArgument body_arg =
                op.getMatchingBlockArgument(op_operand);
            mlir::Value vector = mlir::vector::TransferReadOp::create(
                builder, loc, get_vector_type(body_arg.getType()),
                op_operand->get(), induction_vars, std::nullopt,
                make_in_bounds(op_operand->get().getType()));
            ir_mapping.map(body_arg, vector);
          }

          for (auto& body_op : block->without_terminator()) {
            for (mlir::Value op_operand : body_op.getOperands()) {
              mlir::Value mapped_operand = ir_mapping.lookup(op_operand);
              if (!mlir::isa<mlir::VectorType>(mapped_operand.getType())) {
                auto broadcast_operand = mlir::vector::BroadcastOp::create(
                    builder, loc, get_vector_type(mapped_operand.getType()),
                    mapped_operand);
                ir_mapping.map(op_operand, broadcast_operand);
              }
            }
            auto new_op = builder.clone(body_op, ir_mapping);
            for (auto result : new_op->getResults()) {
              result.setType(get_vector_type(result.getType()));
            }
          }

          for (auto [old_result, dps_init] : llvm::zip(
                   block->getTerminator()->getOperands(), op.getDpsInits())) {
            auto result_vector = ir_mapping.lookup(old_result);
            mlir::vector::TransferWriteOp::create(
                builder, loc, result_vector, dps_init, induction_vars,
                make_in_bounds(dps_init.getType()));
          }
        });

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct LinalgElementwiseToVectorPass
    : public impl::LinalgElementwiseToVectorPassBase<
          LinalgElementwiseToVectorPass> {
  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<ElementwiseToVectorPattern>(context);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateLinalgElementwiseToVectorPass() {
  return std::make_unique<LinalgElementwiseToVectorPass>();
}

}  // namespace xla::cpu
