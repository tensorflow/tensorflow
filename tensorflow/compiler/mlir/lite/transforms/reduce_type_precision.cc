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

// This transformation pass reduces the precision of some tensor types by first
// checking if all values within that tensor are within the range.
// This pass is added to aid conversion of models that involve types not
// available in TF such as INT4, and ideally should be removed in favor of
// stronger type propagation.

#include <cstdint>
#include <memory>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/lite/utils/utils.h"

//===----------------------------------------------------------------------===//
// The ReduceTypePrecision Pass.
//
namespace mlir {
namespace TFL {

namespace {

#define GEN_PASS_DEF_REDUCETYPEPRECISIONPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

// This pattern checks if an i8 arith::ConstantOp tensor has all values within
// the INT4 range, i.e. [-8,7] and converts it into i4 if so. This assumes that
// the input is sign-extended two's complement.
class CheckRangeAndConvertI8ToI4 : public OpRewritePattern<arith::ConstantOp> {
 public:
  using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto const_type = mlir::dyn_cast<ShapedType>(op.getType());
    if (!const_type || !const_type.getElementType().isSignlessInteger(8)) {
      return failure();
    }

    auto attr = mlir::cast<ElementsAttr>(op.getValue());
    for (mlir::APInt v : attr.getValues<mlir::APInt>()) {
      auto v_int = static_cast<int8_t>(*(v.getRawData()));
      if (v_int > 7 || v_int < -8) {
        return failure();
      }
    }

    Builder builder(op.getContext());
    auto shaped_type =
        mlir::RankedTensorType::get(const_type.getShape(), builder.getI4Type());
    auto newAttr = DenseElementsAttr::getFromRawBuffer(
        shaped_type, mlir::cast<DenseElementsAttr>(op.getValue()).getRawData());
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newAttr);

    return success();
  }
};

class SanitizeGatherOpOutputToI4 : public OpRewritePattern<TFL::GatherOp> {
 public:
  using OpRewritePattern<TFL::GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::GatherOp op,
                                PatternRewriter &rewriter) const override {
    auto const_type = mlir::dyn_cast<ShapedType>(op.getOperand(0).getType());
    auto result_type = mlir::dyn_cast<ShapedType>(op.getResult().getType());
    if (!const_type || !const_type.getElementType().isSignlessInteger(4) ||
        !result_type || !result_type.getElementType().isSignlessInteger(8)) {
      return failure();
    }

    auto input_op =
        dyn_cast<arith::ConstantOp>(op.getOperand(0).getDefiningOp());
    if (!input_op) {
      return failure();
    }

    Builder builder(op.getContext());
    auto new_gather_op = rewriter.create<TFL::GatherOp>(
        op.getLoc(),
        /*result=*/
        mlir::cast<TensorType>(op.getResult().getType())
            .clone(builder.getI4Type()),
        /*operand=*/op.getOperands(), op->getAttrs());
    rewriter.replaceAllUsesWith(op.getResult(), new_gather_op.getResult());

    return success();
  }
};

class ReduceTypePrecisionPass
    : public impl::ReduceTypePrecisionPassBase<ReduceTypePrecisionPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReduceTypePrecisionPass)
  void runOnOperation() override;
};

void ReduceTypePrecisionPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<CheckRangeAndConvertI8ToI4, SanitizeGatherOpOutputToI4>(
      &getContext());
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateReduceTypePrecisionPass() {
  return std::make_unique<ReduceTypePrecisionPass>();
}

}  // namespace TFL
}  // namespace mlir
