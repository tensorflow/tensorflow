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

#include "tensorflow/compiler/mlir/lite/transforms/canonicalize_boundary_value_pass.h"

#include <cmath>
#include <cstdint>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/utils/utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace {

#define DEBUG_TYPE "canonicalize-boundary-value"

// Checks whether `dense_attr` contains any +/-Inf values. For splats and common
// float types (f32, f64, f16, bf16), inspects raw values directly to avoid
// constructing an APFloat for every element on large finite constant tensors,
// falling back to `getValues<APFloat>()` for other float types.
bool HasInfiniteElement(DenseElementsAttr dense_attr, FloatType float_type) {
  if (dense_attr.isSplat()) {
    return dense_attr.getSplatValue<APFloat>().isInfinity();
  }
  if (float_type.isF32()) {
    return llvm::any_of(dense_attr.getValues<float>(),
                        [](float val) { return std::isinf(val); });
  }
  if (float_type.isF64()) {
    return llvm::any_of(dense_attr.getValues<double>(),
                        [](double val) { return std::isinf(val); });
  }
  if (float_type.isF16() || float_type.isBF16()) {
    const llvm::fltSemantics& semantics = float_type.getFloatSemantics();
    const uint16_t pos_inf_bits = APFloat::getInf(semantics, /*Negative=*/false)
                                      .bitcastToAPInt()
                                      .getZExtValue();
    const uint16_t neg_inf_bits = APFloat::getInf(semantics, /*Negative=*/true)
                                      .bitcastToAPInt()
                                      .getZExtValue();
    ArrayRef<char> raw_data = dense_attr.getRawData();
    ArrayRef<uint16_t> words(reinterpret_cast<const uint16_t*>(raw_data.data()),
                             raw_data.size() / sizeof(uint16_t));
    return llvm::any_of(words, [pos_inf_bits, neg_inf_bits](uint16_t bits) {
      return bits == pos_inf_bits || bits == neg_inf_bits;
    });
  }
  return llvm::any_of(dense_attr.getValues<APFloat>(),
                      [](const APFloat& val) { return val.isInfinity(); });
}

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

    // If all values are finite, no need to rewrite. Use the DenseElementsAttr
    // fast path to avoid constructing an APFloat for every element when all
    // elements are finite.
    const bool has_inf =
        llvm::isa<DenseElementsAttr>(tensor_attr)
            ? HasInfiniteElement(llvm::cast<DenseElementsAttr>(tensor_attr),
                                 float_type)
            : llvm::any_of(tensor_attr.getValues<APFloat>(),
                           [](const APFloat& val) { return val.isInfinity(); });
    if (!has_inf) return failure();

    auto vals_orig = tensor_attr.getValues<APFloat>();

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
}  // end namespace

void CanonicalizeBoundaryValuePass::runOnOperation() {
  auto* ctx = &getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<ClampInfToMinMaxFloat<stablehlo::ConstantOp>,
               ClampInfToMinMaxFloat<TF::ConstOp>,
               ClampInfToMinMaxFloat<arith::ConstantOp>>(ctx);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // end namespace TFL
}  // end namespace mlir
