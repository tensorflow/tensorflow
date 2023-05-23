/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace xla {
namespace cpu {
namespace {

#define GEN_PASS_DEF_LEGALIZEXLAABIPASS
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h.inc"

using namespace mlir;  // NOLINT

class LegalizeXlaAbiPass
    : public impl::LegalizeXlaAbiPassBase<LegalizeXlaAbiPass> {
  void runOnOperation() override;
};

bool IsDefaultLayout(ArrayRef<int64_t> layout) {
  return llvm::equal(llvm::reverse(layout),
                     llvm::seq(size_t{0}, layout.size()));
}

Value NormalizeTensor(ImplicitLocOpBuilder& b, TypedValue<ShapedType> tensor,
                      ArrayRef<int64_t> layout, bool isInput) {
  int64_t rank = tensor.getType().getRank();
  SmallVector<int64_t> permutation{llvm::reverse(layout)};
  SmallVector<int64_t> physical_dim_sizes(rank);
  for (int64_t dim = 0; dim < rank; ++dim) {
    physical_dim_sizes[dim] = tensor.getType().getDimSize(permutation[dim]);
  }
  auto physical_shape = RankedTensorType::get(
      physical_dim_sizes, tensor.getType().getElementType());

  if (isInput) {
    SmallVector<int64_t> inverse_permutation(rank);
    for (int64_t dim = 0; dim < rank; ++dim) {
      inverse_permutation[permutation[dim]] = dim;
    }
    Value reshape = b.create<mhlo::ReshapeOp>(physical_shape, tensor);
    return b.create<mhlo::TransposeOp>(tensor.getType(), reshape,
                                       b.getI64VectorAttr(inverse_permutation));
  }

  Value transpose = b.create<mhlo::TransposeOp>(
      physical_shape, tensor, b.getI64VectorAttr(permutation));
  return b.create<mhlo::ReshapeOp>(tensor.getType(), transpose);
}

void NormalizeInputInPlace(ImplicitLocOpBuilder& b, Value tensor,
                           ArrayRef<int64_t> layout) {
  auto typedTensor = tensor.dyn_cast<TypedValue<ShapedType>>();
  if (!typedTensor || IsDefaultLayout(layout)) {
    return;
  }

  Value normalized = NormalizeTensor(b, typedTensor, layout, /*isInput=*/true);
  tensor.replaceAllUsesExcept(
      normalized, normalized.getDefiningOp()->getOperand(0).getDefiningOp());
}

SmallVector<SmallVector<int64_t>> FlattenLayoutAttribute(Attribute attr) {
  SmallVector<SmallVector<int64_t>> layouts;

  auto visit_attr = [&](mlir::Attribute attr) {
    if (attr.isa<DenseElementsAttr>()) {
      layouts.emplace_back(attr.cast<DenseElementsAttr>().getValues<int64_t>());
    }
  };

  if (auto array = attr.dyn_cast<ArrayAttr>()) {
    for (int64_t i = 0; i < array.size(); ++i) {
      visit_attr(array[i]);
    }
  } else {
    visit_attr(attr);
  }
  return layouts;
}

struct RewriteInputArgs : OpRewritePattern<func::FuncOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter& rewriter) const override {
    // Note: this attribute is only present if
    // a) this function is the entry computation and
    // b) there's at least one input with a custom layout.
    auto layouts = op->getAttr("xla_entry_computation_parameter_layouts");
    if (layouts == nullptr) {
      return failure();
    }

    // Flatten the layouts (we're assuming we run after expand-hlo-tuples).
    SmallVector<SmallVector<int64_t>> param_layouts =
        FlattenLayoutAttribute(layouts);
    assert(param_layouts.size() == op.getNumArguments() &&
           "Unexpected number of parameter layouts");

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    b.setInsertionPointToStart(&op.getBody().front());
    IRMapping bvm;
    for (const auto&& [param, layout] :
         llvm::zip(op.getArguments(), param_layouts)) {
      NormalizeInputInPlace(b, param, layout);
    }
    op->removeAttr("xla_entry_computation_parameter_layouts");

    return success();
  }
};

struct RewriteReturnArgs : OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter& rewriter) const override {
    auto func = op->getParentOfType<func::FuncOp>();
    assert(func && "ReturnOp's parent is always a FuncOp");
    auto layouts = func->getAttr("xla_entry_computation_result_layout");
    if (layouts == nullptr) {
      return failure();
    }

    SmallVector<SmallVector<int64_t>> result_layouts =
        FlattenLayoutAttribute(layouts);
    assert(result_layouts.size() == func.getNumResults() &&
           "Unexpected number of result layouts");

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    b.setInsertionPoint(op);
    SmallVector<Value> results;
    for (const auto&& [result, layout] :
         llvm::zip(op.getOperands(), result_layouts)) {
      results.push_back(
          IsDefaultLayout(layout)
              ? result
              : NormalizeTensor(b, result.cast<TypedValue<ShapedType>>(),
                                layout,
                                /*isInput=*/false));
    }

    func->removeAttr("xla_entry_computation_result_layout");
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, results);
    return success();
  }
};

bool IsI1Tensor(Type ty) {
  return ty.isa<ShapedType>() &&
         ty.cast<ShapedType>().getElementType().isInteger(1);
}

struct RewriteI1Results : OpRewritePattern<func::FuncOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter& rewriter) const override {
    if (!llvm::any_of(op.getResultTypes(), IsI1Tensor)) {
      return failure();
    }

    func::ReturnOp return_op =
        cast<func::ReturnOp>(op.getBody().front().getTerminator());
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    b.setInsertionPoint(return_op);

    for (const auto& [index, resultTy] :
         llvm::enumerate(return_op.getOperandTypes())) {
      if (IsI1Tensor(resultTy)) {
        return_op.setOperand(
            index, b.create<mhlo::ConvertOp>(
                       return_op.getOperand(index),
                       rewriter.getIntegerType(8, /*isSigned=*/false)));
      }
    }

    FunctionType fun_ty = op.getFunctionType();
    op.setFunctionType(FunctionType::get(
        fun_ty.getContext(), fun_ty.getInputs(), return_op->getOperandTypes()));

    return success();
  }
};

struct RewriteCustomCalls : OpRewritePattern<mhlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    b.setInsertionPoint(op);

    if (!op->hasAttr("operand_layouts") && !op->hasAttr("result_layouts") &&
        !llvm::any_of(op.getOperandTypes(), IsI1Tensor)) {
      return failure();
    }

    // Normalize any operands that require it.
    if (auto operand_layouts_attr = op.getOperandLayoutsAttr()) {
      SmallVector<SmallVector<int64_t>> operand_layouts =
          FlattenLayoutAttribute(operand_layouts_attr);

      for (const auto& [index, operand] : llvm::enumerate(op.getOperands())) {
        const auto& layout = operand_layouts[index];
        if (!IsDefaultLayout(layout)) {
          Value normalized = NormalizeTensor(
              b, op.getOperand(index).cast<TypedValue<ShapedType>>(), layout,
              /*isInput=*/false);
          op.setOperand(index, normalized);
        }
      }
      op.removeOperandLayoutsAttr();
    }

    // Rewrite i1 inputs to ui8.
    for (const auto& [index, operand] : llvm::enumerate(op.getOperands())) {
      if (IsI1Tensor(operand.getType())) {
        op.setOperand(index, b.create<mhlo::ConvertOp>(
                                 operand, rewriter.getIntegerType(
                                              8, /*isSigned=*/false)));
      }
    }

    // Normalize outputs.
    b.setInsertionPointAfter(op);
    if (auto result_layouts_attr = op.getResultLayoutsAttr()) {
      SmallVector<SmallVector<int64_t>> result_layouts =
          FlattenLayoutAttribute(result_layouts_attr);
      assert(result_layouts.size() == op.getNumResults() &&
             "Unexpected number of result layouts");
      for (const auto& [result, layout] :
           llvm::zip(op.getResults(), result_layouts)) {
        NormalizeInputInPlace(b, result, layout);
      }

      op.removeResultLayoutsAttr();
    }

    return success();
  }
};

template <typename Op>
struct RewriteResultLayout : OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter& rewriter) const override {
    auto layout_attr = op->getAttr("result_layout");
    if (!layout_attr) {
      return failure();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    b.setInsertionPointAfter(op);

    SmallVector<SmallVector<int64_t>> result_layouts =
        FlattenLayoutAttribute(layout_attr);

    assert(result_layouts.size() == op->getNumResults() &&
           "Unexpected number of result layouts");
    for (const auto& [result, layout] :
         llvm::zip(op->getResults(), result_layouts)) {
      NormalizeInputInPlace(b, result, layout);
    }

    op->removeAttr("result_layout");
    return success();
  }
};

void LegalizeXlaAbiPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext* ctx = module.getContext();

  // Convert lmhlo operations to XLA cpu runtime custom calls.
  RewritePatternSet patterns(ctx);
  patterns.insert<RewriteInputArgs, RewriteReturnArgs, RewriteI1Results,
                  RewriteCustomCalls, RewriteResultLayout<mhlo::ConstantOp>>(
      ctx);

  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createXlaAbiLegalizationPass() {
  return std::make_unique<LegalizeXlaAbiPass>();
}

}  // namespace cpu
}  // namespace xla
