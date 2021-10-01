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

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
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
    llvm::SmallDenseSet<char> result_tokens;
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
          result_tokens.insert(equation[index]);
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
    assert(lhs_tokens.size() ==
           einsum.lhs().getType().cast<RankedTensorType>().getRank());
    assert(rhs_tokens.size() ==
           einsum.rhs().getType().cast<RankedTensorType>().getRank());

    auto collect_contracting_batching_dims =
        [&](SmallVector<char> tokens, SmallVector<char> others,
            SmallVectorImpl<int64_t> &contracting_dims,
            SmallVectorImpl<int64_t> &batching_dims) {
          llvm::SmallDenseSet<char> others_set(others.begin(), others.end());
          for (auto en : llvm::enumerate(tokens)) {
            if (!result_tokens.contains(en.value())) {
              contracting_dims.emplace_back(en.index());
            }
            if (others_set.contains(en.value()) &&
                result_tokens.contains(en.value())) {
              batching_dims.emplace_back(en.index());
            }
          }
        };
    SmallVector<int64_t> lhs_contracting_dims, lhs_batching_dims,
        rhs_contracting_dims, rhs_batching_dims;
    collect_contracting_batching_dims(lhs_tokens, rhs_tokens,
                                      lhs_contracting_dims, lhs_batching_dims);
    collect_contracting_batching_dims(rhs_tokens, lhs_tokens,
                                      rhs_contracting_dims, rhs_batching_dims);

    auto dim_numbers = mhlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), lhs_batching_dims, rhs_batching_dims,
        lhs_contracting_dims, rhs_contracting_dims);
    rewriter.replaceOpWithNewOp<DotGeneralOp>(
        einsum, einsum.getType(), einsum.lhs(), einsum.rhs(), dim_numbers,
        /*precision_config=*/ArrayAttr{});
    return success();
  }
};

struct LegalizeEinsumToDotGeneralPass
    : public LegalizeEinsumToDotGeneralPassBase<
          LegalizeEinsumToDotGeneralPass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns(&getContext());
    PopulateEinsumToDotGeneralPatterns(&getContext(), &patterns);
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }
};
}  // namespace

void PopulateEinsumToDotGeneralPatterns(mlir::MLIRContext *context,
                                        OwningRewritePatternList *patterns) {
  patterns->insert<EinsumToDotGeneralPattern>(context);
}

std::unique_ptr<FunctionPass> createLegalizeEinsumToDotGeneralPass() {
  return std::make_unique<LegalizeEinsumToDotGeneralPass>();
}

}  // namespace mhlo
}  // namespace mlir
