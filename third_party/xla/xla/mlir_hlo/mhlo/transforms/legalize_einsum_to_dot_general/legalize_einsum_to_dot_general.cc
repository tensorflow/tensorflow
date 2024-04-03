/* Copyright 2021 The OpenXLA Authors.

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
#include <cctype>
#include <iterator>
#include <memory>
#include <utility>

#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_LEGALIZEEINSUMTODOTGENERALPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

struct EinsumToDotGeneralPattern : public OpRewritePattern<EinsumOp> {
  using OpRewritePattern<EinsumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(EinsumOp einsum,
                                PatternRewriter &rewriter) const override {
    StringRef equation = einsum.getEinsumConfig();
    SmallVector<char> lhsTokens, rhsTokens;
    SmallVector<char> resultTokens;
    size_t index = 0;
    enum EquationVariable { kIsLhs, kIsRhs, kIsResult };
    EquationVariable currentVariable = kIsLhs;
    while (index < equation.size()) {
      if (std::isalpha(equation[index])) {
        if (currentVariable == kIsLhs) {
          lhsTokens.push_back(equation[index]);
        } else if (currentVariable == kIsRhs) {
          rhsTokens.push_back(equation[index]);
        } else {
          resultTokens.push_back(equation[index]);
        }
      } else if (equation.substr(index, 1).contains(",")) {
        currentVariable = kIsRhs;
      } else if ((index < (equation.size() - 1)) &&
                 (equation.substr(index, 2).contains("->"))) {
        currentVariable = kIsResult;
        index++;
      } else {
        return einsum.emitError("unexpected character ")
               << equation.substr(index, 1) << " encountered";
      }
      index++;
    }

    auto lhsType = einsum.getLhs().getType().cast<RankedTensorType>();
    auto rhsType = einsum.getRhs().getType().cast<RankedTensorType>();
    assert(static_cast<int64_t>(lhsTokens.size()) == lhsType.getRank());
    assert(static_cast<int64_t>(rhsTokens.size()) == rhsType.getRank());

    auto collectOperandDims =
        [resultTokens](
            RankedTensorType operandType, SmallVector<char> operandTokens,
            SmallVector<char> others, SmallVectorImpl<int64_t> &contractingDims,
            SmallVectorImpl<int64_t> &batchingDims,
            SmallVector<char> &dotResultTokens,
            SmallVector<int64_t> &dotResultShape) {
          llvm::SmallDenseSet<char> othersSet(others.begin(), others.end());
          llvm::SmallDenseSet<char> resultTokensSet(resultTokens.begin(),
                                                    resultTokens.end());
          for (const auto &en : llvm::enumerate(operandTokens)) {
            bool isResultToken = resultTokensSet.contains(en.value());
            bool isOtherToken = othersSet.contains(en.value());

            if (!isResultToken) {
              contractingDims.push_back(en.index());
            } else if (isOtherToken) {
              batchingDims.push_back(en.index());
            } else {
              dotResultTokens.push_back(en.value());
              dotResultShape.push_back(operandType.getShape()[en.index()]);
            }
          }
        };
    // Indices of batch and contracting dims, relative to each operand's
    // dimensions.
    SmallVector<int64_t> lhsContractingDims, lhsBatchingDims,
        rhsContractingDims, rhsBatchingDims;
    // Tokens representing the natural order of the dot_general op (i.e.
    // the lhs non-contracting followed by rhs non-contracting tokens).
    SmallVector<char> dotResultTokens;
    SmallVector<int64_t> dotResultShape;

    collectOperandDims(lhsType, lhsTokens, rhsTokens, lhsContractingDims,
                       lhsBatchingDims, dotResultTokens, dotResultShape);
    collectOperandDims(rhsType, rhsTokens, lhsTokens, rhsContractingDims,
                       rhsBatchingDims, dotResultTokens, dotResultShape);

    // Prepend batch tokens.
    for (const auto &it : llvm::enumerate(lhsBatchingDims)) {
      char batchingToken = lhsTokens[it.value()];
      int64_t batchingShapeDim = lhsType.getShape()[it.value()];
      dotResultTokens.insert(dotResultTokens.begin() + it.index(),
                             batchingToken);
      dotResultShape.insert(dotResultShape.begin() + it.index(),
                            batchingShapeDim);
    }

    // Lowering to dot_general does not support a mismatch between the number
    // of result dims and the number of non-contracting dims.
    if (dotResultTokens.size() != resultTokens.size()) {
      return rewriter.notifyMatchFailure(einsum,
                                         "rank reducing einsum not supported");
    }

    // Generate a permutation sequence based on result tokens.
    SmallVector<int64_t> resultPerms;
    bool isNaturalOrder = true;
    for (char resultToken : resultTokens) {
      auto *foundIt = std::find(dotResultTokens.begin(), dotResultTokens.end(),
                                resultToken);
      if (foundIt == dotResultTokens.end()) {
        return rewriter.notifyMatchFailure(
            einsum, "result token not found in operands");
      }
      auto resultIndex = std::distance(dotResultTokens.begin(), foundIt);
      if (resultPerms.empty()) {
        if (resultIndex != 0) {
          isNaturalOrder = false;
        }
      } else if (resultIndex != (resultPerms.back() + 1)) {
        isNaturalOrder = false;
      }
      resultPerms.push_back(resultIndex);
    }

    // Emit the dot_general, using its native result ordering.
    auto dotGeneralResultType = RankedTensorType::get(
        ArrayRef<int64_t>(dotResultShape), lhsType.getElementType());
    auto dimNumbers = mhlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), lhsBatchingDims, rhsBatchingDims,
        lhsContractingDims, rhsContractingDims);
    auto dotGeneralOp = rewriter.create<DotGeneralOp>(
        einsum.getLoc(), dotGeneralResultType, einsum.getLhs(), einsum.getRhs(),
        dimNumbers,
        /*precision_config=*/ArrayAttr{});

    if (isNaturalOrder) {
      // The dot_general is already in an appropriate result order.
      rewriter.replaceOp(einsum, ValueRange{dotGeneralOp});
    } else {
      // Generate a transpose.
      rewriter.replaceOpWithNewOp<TransposeOp>(
          einsum, dotGeneralOp, rewriter.getI64TensorAttr(resultPerms));
    }
    return success();
  }
};

struct LegalizeEinsumToDotGeneralPass
    : public impl::LegalizeEinsumToDotGeneralPassBase<
          LegalizeEinsumToDotGeneralPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateEinsumToDotGeneralPatterns(&getContext(), &patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

void populateEinsumToDotGeneralPatterns(mlir::MLIRContext *context,
                                        RewritePatternSet *patterns) {
  patterns->add<EinsumToDotGeneralPattern>(context);
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeEinsumToDotGeneralPass() {
  return std::make_unique<LegalizeEinsumToDotGeneralPass>();
}

}  // namespace mhlo
}  // namespace mlir
