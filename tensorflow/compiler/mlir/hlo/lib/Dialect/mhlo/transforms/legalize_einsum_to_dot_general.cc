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

#include <algorithm>
#include <utility>

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

DenseIntElementsAttr Make1DElementsAttr(OpBuilder &b,
                                        ArrayRef<int64_t> integers) {
  auto type = RankedTensorType::get({static_cast<int64_t>(integers.size())},
                                    b.getI64Type());
  return DenseIntElementsAttr::get(type, integers);
}

struct EinsumToDotGeneralPattern : public OpRewritePattern<EinsumOp> {
  using OpRewritePattern<EinsumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(EinsumOp einsum,
                                PatternRewriter &rewriter) const override {
    StringRef equation = einsum.einsum_config();
    SmallVector<char> lhs_tokens, rhs_tokens;
    SmallVector<char> result_tokens;
    size_t index = 0;
    enum EquationVariable { kIsLhs, kIsRhs, kIsResult };
    EquationVariable current_variable = kIsLhs;
    while (index < equation.size()) {
      if (std::isalpha(equation[index])) {
        if (current_variable == kIsLhs) {
          lhs_tokens.push_back(equation[index]);
        } else if (current_variable == kIsRhs) {
          rhs_tokens.push_back(equation[index]);
        } else {
          result_tokens.push_back(equation[index]);
        }
      } else if (equation.substr(index, 1).contains(",")) {
        current_variable = kIsRhs;
      } else if ((index < (equation.size() - 1)) &&
                 (equation.substr(index, 2).contains("->"))) {
        current_variable = kIsResult;
        index++;
      } else {
        return einsum.emitError("unexpected character ")
               << equation.substr(index, 1) << " encountered";
      }
      index++;
    }

    auto lhs_type = einsum.lhs().getType().cast<RankedTensorType>();
    auto rhs_type = einsum.rhs().getType().cast<RankedTensorType>();
    assert(static_cast<int64_t>(lhs_tokens.size()) == lhs_type.getRank());
    assert(static_cast<int64_t>(rhs_tokens.size()) == rhs_type.getRank());

    auto collect_operand_dims = [&](RankedTensorType operand_type,
                                    SmallVector<char> operand_tokens,
                                    SmallVector<char> others,
                                    SmallVectorImpl<int64_t> &contracting_dims,
                                    SmallVectorImpl<int64_t> &batching_dims,
                                    SmallVector<char> &dot_result_tokens,
                                    SmallVector<int64_t> &dot_result_shape) {
      llvm::SmallDenseSet<char> others_set(others.begin(), others.end());
      llvm::SmallDenseSet<char> result_tokens_set(result_tokens.begin(),
                                                  result_tokens.end());
      for (const auto &en : llvm::enumerate(operand_tokens)) {
        bool is_result_token = result_tokens_set.contains(en.value());
        bool is_other_token = others_set.contains(en.value());

        if (!is_result_token) {
          contracting_dims.push_back(en.index());
        } else if (is_other_token) {
          batching_dims.push_back(en.index());
        } else {
          dot_result_tokens.push_back(en.value());
          dot_result_shape.push_back(operand_type.getShape()[en.index()]);
        }
      }
    };
    // Indices of batch and contracting dims, relative to each operand's
    // dimensions.
    SmallVector<int64_t> lhs_contracting_dims, lhs_batching_dims,
        rhs_contracting_dims, rhs_batching_dims;
    // Tokens representing the natural order of the dot_general op (i.e.
    // the lhs non-contracting followed by rhs non-contracting tokens).
    SmallVector<char> dot_result_tokens;
    SmallVector<int64_t> dot_result_shape;

    collect_operand_dims(lhs_type, lhs_tokens, rhs_tokens, lhs_contracting_dims,
                         lhs_batching_dims, dot_result_tokens,
                         dot_result_shape);
    collect_operand_dims(rhs_type, rhs_tokens, lhs_tokens, rhs_contracting_dims,
                         rhs_batching_dims, dot_result_tokens,
                         dot_result_shape);

    // Prepend batch tokens.
    for (const auto &it : llvm::enumerate(lhs_batching_dims)) {
      char batching_token = lhs_tokens[it.value()];
      int64_t batching_shape_dim = lhs_type.getShape()[it.value()];
      dot_result_tokens.insert(dot_result_tokens.begin() + it.index(),
                               batching_token);
      dot_result_shape.insert(dot_result_shape.begin() + it.index(),
                              batching_shape_dim);
    }

    // Lowering to dot_general does not support a mismatch between the number
    // of result dims and the number of non-contracting dims.
    if (dot_result_tokens.size() != result_tokens.size()) {
      return rewriter.notifyMatchFailure(einsum,
                                         "rank reducing einsum not supported");
    }

    // Generate a permutation sequence based on result tokens.
    SmallVector<int64_t> result_perms;
    bool is_natural_order = true;
    for (char result_token : result_tokens) {
      auto *found_it = std::find(dot_result_tokens.begin(),
                                 dot_result_tokens.end(), result_token);
      if (found_it == dot_result_tokens.end()) {
        return rewriter.notifyMatchFailure(
            einsum, "result token not found in operands");
      }
      auto result_index = std::distance(dot_result_tokens.begin(), found_it);
      if (result_perms.empty()) {
        if (result_index != 0) {
          is_natural_order = false;
        }
      } else if (result_index != (result_perms.back() + 1)) {
        is_natural_order = false;
      }
      result_perms.push_back(result_index);
    }

    // Emit the dot_general, using its native result ordering.
    auto dot_general_result_type = RankedTensorType::get(
        ArrayRef<int64_t>(dot_result_shape), lhs_type.getElementType());
    auto dim_numbers = mhlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), lhs_batching_dims, rhs_batching_dims,
        lhs_contracting_dims, rhs_contracting_dims);
    auto dot_general_op =
        rewriter.create<DotGeneralOp>(einsum.getLoc(), dot_general_result_type,
                                      einsum.lhs(), einsum.rhs(), dim_numbers,
                                      /*precision_config=*/ArrayAttr{});

    if (is_natural_order) {
      // The dot_general is already in an appropriate result order.
      rewriter.replaceOp(einsum, ValueRange{dot_general_op});
    } else {
      // Generate a transpose.
      rewriter.replaceOpWithNewOp<TransposeOp>(
          einsum, dot_general_op, rewriter.getI64TensorAttr(result_perms));
    }
    return success();
  }
};

struct LegalizeEinsumToDotGeneralPass
    : public LegalizeEinsumToDotGeneralPassBase<
          LegalizeEinsumToDotGeneralPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    PopulateEinsumToDotGeneralPatterns(&getContext(), &patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

void PopulateEinsumToDotGeneralPatterns(mlir::MLIRContext *context,
                                        RewritePatternSet *patterns) {
  patterns->add<EinsumToDotGeneralPattern>(context);
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeEinsumToDotGeneralPass() {
  return std::make_unique<LegalizeEinsumToDotGeneralPass>();
}

}  // namespace mhlo
}  // namespace mlir
