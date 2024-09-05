/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace {

#define DEBUG_TYPE "canonicalize-boundary-value"

#define GEN_PASS_DEF_CANONICALIZEBOUNDARYVALUEPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

class CanonicalizeBoundaryValuePass
    : public impl::CanonicalizeBoundaryValuePassBase<
          CanonicalizeBoundaryValuePass> {
  void runOnOperation() override;
};

// Clamp constant -Inf/Inf to MIN/MAX float value.
template <typename OpTy>
struct ClampInfToMinMaxFloat : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy const_op,
                                PatternRewriter& rewriter) const override {
    Attribute attr = const_op.getValueAttr();
    if (auto float_attr = llvm::dyn_cast<FloatAttr>(attr)) {
      if (float_attr.getValue().isInfinity()) {
        FloatType float_type = llvm::dyn_cast<FloatType>(const_op.getType());
        if (!float_type) return failure();
        rewriter.replaceOpWithNewOp<OpTy>(
            const_op, rewriter.getFloatAttr(
                          float_type, APFloat::getLargest(
                                          float_type.getFloatSemantics(),
                                          float_attr.getValue().isNegative())));
        return success();
      }
    }

    ElementsAttr tensor_attr = llvm::dyn_cast<ElementsAttr>(attr);
    if (!tensor_attr) return failure();

    Type type = tensor_attr.getType();
    ShapedType tensor_type = llvm::cast<ShapedType>(type);
    auto float_type = dyn_cast<FloatType>(tensor_type.getElementType());
    if (!float_type) return failure();

    auto vals_orig = tensor_attr.getValues<APFloat>();
    // If all values are finite, no need to rewrite.
    if (llvm::all_of(vals_orig, [&](APFloat val) { return !val.isInfinity(); }))
      return failure();

    SmallVector<APFloat> vals_new(llvm::map_range(vals_orig, [&](APFloat val) {
      return val.isInfinity()
                 ? APFloat::getLargest(float_type.getFloatSemantics(),
                                       val.isNegative())
                 : val;
    }));
    rewriter.replaceOpWithNewOp<OpTy>(
        const_op, DenseElementsAttr::get(tensor_type, vals_new));
    return success();
  }
};

void CanonicalizeBoundaryValuePass::runOnOperation() {
  auto* ctx = &getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<ClampInfToMinMaxFloat<stablehlo::ConstantOp>,
               ClampInfToMinMaxFloat<TF::ConstOp>,
               ClampInfToMinMaxFloat<arith::ConstantOp>>(ctx);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // end namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateCanonicalizeBoundaryValuePass() {
  return std::make_unique<CanonicalizeBoundaryValuePass>();
}

}  // end namespace TFL
}  // end namespace mlir
