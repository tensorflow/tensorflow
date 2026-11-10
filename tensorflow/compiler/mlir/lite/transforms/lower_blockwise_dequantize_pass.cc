/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"

namespace mlir {
namespace TFL {
namespace {

#define GEN_PASS_DEF_LOWERBLOCKWISEDEQUANTIZEPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

// Returns the elements of `value` if it is a constant.
ElementsAttr GetConstantElements(Value value) {
  ElementsAttr attr;
  if (matchPattern(value, m_Constant(&attr))) return attr;
  if (auto const_op = value.getDefiningOp<TFL::ConstOp>()) {
    return const_op.getValue();
  }
  return nullptr;
}

// Rewrites
//
//   %dq = tfl.blockwise_dequantize(%q, %scales, %zero_points)
//
// where all operands are constants into
//
//   %q_typed = tfl.pseudo_qconst() {qtype = tensor<...x!tfl.blockwise_quant<
//                  ..., block_shape, scales, zero_points>>, value = %q}
//   %dq = tfl.dequantize(%q_typed)
//
// so that the quantization parameters live on the tensor and the model can be
// exported. Both compute `scales * (q - zero_points)` per block.
struct LowerBlockwiseDequantizePattern
    : public OpRewritePattern<BlockwiseDequantizeOp> {
  using OpRewritePattern<BlockwiseDequantizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BlockwiseDequantizeOp op,
                                PatternRewriter& rewriter) const override {
    // `tfl.dequantize` produces f32; a float16 / bfloat16 output casts it.
    auto output_type = mlir::dyn_cast<RankedTensorType>(op.getType());
    if (!output_type) {
      return rewriter.notifyMatchFailure(op, "expects a ranked output");
    }
    Type output_element_type = output_type.getElementType();
    if (!output_element_type.isF32() && !output_element_type.isF16() &&
        !output_element_type.isBF16()) {
      return rewriter.notifyMatchFailure(op,
                                         "expects an f32, f16 or bf16 output");
    }

    ElementsAttr values = GetConstantElements(op.getInput());
    if (!values) {
      return rewriter.notifyMatchFailure(op, "expects a constant input");
    }
    auto storage_type = mlir::dyn_cast<IntegerType>(values.getElementType());
    if (!storage_type) {
      return rewriter.notifyMatchFailure(op, "expects an integer input");
    }

    ElementsAttr scales = GetConstantElements(op.getScales());
    if (!scales) {
      return rewriter.notifyMatchFailure(op, "expects constant scales");
    }

    ElementsAttr zero_points;
    if (!mlir::isa<NoneType>(op.getZeroPoints().getType())) {
      zero_points = GetConstantElements(op.getZeroPoints());
      if (!zero_points) {
        return rewriter.notifyMatchFailure(op, "expects constant zero points");
      }
    }

    SmallVector<int64_t> block_shape;
    block_shape.reserve(op.getBlockShape().size());
    for (Attribute block_size : op.getBlockShape()) {
      block_shape.push_back(mlir::cast<IntegerAttr>(block_size).getInt());
    }

    // The verifier of `tfl.blockwise_dequantize` already guarantees what the
    // type requires: an integer input of at most 16 bits, floating point
    // scales, and parameter tensors shaped like the block grid.
    auto f32_type = output_type.clone(rewriter.getF32Type());
    auto blockwise_type = BlockwiseQuantType::get(
        storage_type, rewriter.getF32Type(), block_shape, scales, zero_points);

    auto qtype = RankedTensorType::get(values.getShapedType().getShape(),
                                       blockwise_type);
    auto qconst =
        rewriter.create<QConstOp>(op.getLoc(), TypeAttr::get(qtype), values);
    if (output_element_type.isF32()) {
      rewriter.replaceOpWithNewOp<DequantizeOp>(op, output_type,
                                                qconst.getOutput());
      return success();
    }
    // Cost: the DEQUANTIZE of a constant is computed once, but this CAST of
    // its float32 result runs on every invocation (e.g. for a weight-only
    // float16 filter).
    auto dequantize = rewriter.create<DequantizeOp>(op.getLoc(), f32_type,
                                                    qconst.getOutput());
    rewriter.replaceOpWithNewOp<CastOp>(op, output_type, dequantize);
    return success();
  }
};

struct LowerBlockwiseDequantizePass
    : public impl::LowerBlockwiseDequantizePassBase<
          LowerBlockwiseDequantizePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBlockwiseDequantizePass)

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext* context = func.getContext();

    RewritePatternSet patterns(context);
    patterns.add<LowerBlockwiseDequantizePattern>(context);

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateLowerBlockwiseDequantizePass() {
  return std::make_unique<LowerBlockwiseDequantizePass>();
}

}  // namespace TFL
}  // namespace mlir
