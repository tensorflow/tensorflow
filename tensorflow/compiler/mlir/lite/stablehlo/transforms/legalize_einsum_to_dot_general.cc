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

// StableHLO port of MHLO's `-mhlo-legalize-einsum-to-dot-general` pass
// (xla/mlir_hlo/mhlo/transforms/legalize_einsum_to_dot_general). Rewrites
// binary `stablehlo.einsum` ops into a `stablehlo.dot_general`, followed by a
// `stablehlo.transpose` when the einsum result order differs from the natural
// dot_general result order (batch dims, lhs free dims, rhs free dims).
//
// `stablehlo.unary_einsum` is intentionally not handled (the MHLO pass did not
// handle `mhlo.unary_einsum` either). Run `stablehlo-legalize-deprecated-ops`
// beforehand if unary einsums need to be upgraded to binary einsums.

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>

#include "absl/algorithm/container.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"

namespace mlir {
namespace odml {
namespace {

#define GEN_PASS_DEF_LEGALIZESTABLEHLOEINSUMTODOTGENERALPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h.inc"

struct EinsumToDotGeneralPattern
    : public OpRewritePattern<stablehlo::EinsumOp> {
  using OpRewritePattern<stablehlo::EinsumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::EinsumOp einsum,
                                PatternRewriter& rewriter) const override {
    StringRef equation = einsum.getEinsumConfig();
    SmallVector<char> lhs_tokens, rhs_tokens;
    SmallVector<char> result_tokens;
    size_t index = 0;
    enum EquationVariable { kIsLhs, kIsRhs, kIsResult };
    EquationVariable current_variable = kIsLhs;
    while (index < equation.size()) {
      if (llvm::isAlpha(equation[index])) {
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
        // E.g. ellipsis ("...") or whitespace; not supported by this lowering.
        return rewriter.notifyMatchFailure(
            einsum, "unexpected character in einsum_config");
      }
      index++;
    }

    auto lhs_type = mlir::dyn_cast<RankedTensorType>(einsum.getLhs().getType());
    auto rhs_type = mlir::dyn_cast<RankedTensorType>(einsum.getRhs().getType());
    if (!lhs_type || !rhs_type) {
      return rewriter.notifyMatchFailure(einsum, "expected ranked operands");
    }
    if (static_cast<int64_t>(lhs_tokens.size()) != lhs_type.getRank() ||
        static_cast<int64_t>(rhs_tokens.size()) != rhs_type.getRank()) {
      return rewriter.notifyMatchFailure(
          einsum, "einsum_config does not match operand ranks");
    }

    auto collect_operand_dims =
        [&result_tokens](RankedTensorType operand_type,
                         ArrayRef<char> operand_tokens, ArrayRef<char> others,
                         SmallVectorImpl<int64_t>& contracting_dims,
                         SmallVectorImpl<int64_t>& batching_dims,
                         SmallVectorImpl<char>& dot_result_tokens,
                         SmallVectorImpl<int64_t>& dot_result_shape) {
          llvm::SmallDenseSet<char> others_set(others.begin(), others.end());
          llvm::SmallDenseSet<char> result_tokens_set(result_tokens.begin(),
                                                      result_tokens.end());
          for (const auto& en : llvm::enumerate(operand_tokens)) {
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
    for (const auto& it : llvm::enumerate(lhs_batching_dims)) {
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
      auto* found_it = absl::c_find(dot_result_tokens, result_token);
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

    // Emit the dot_general, using its native result ordering. The element type
    // is taken from the einsum result (identical to the lhs element type in
    // the common case, but also correct for e.g. i8 x i8 -> i32 einsums).
    auto dot_general_result_type =
        RankedTensorType::get(ArrayRef<int64_t>(dot_result_shape),
                              getElementTypeOrSelf(einsum.getType()));
    auto dim_numbers = stablehlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), lhs_batching_dims, rhs_batching_dims,
        lhs_contracting_dims, rhs_contracting_dims);
    auto dot_general_op = stablehlo::DotGeneralOp::create(
        rewriter, einsum.getLoc(), dot_general_result_type, einsum.getLhs(),
        einsum.getRhs(), dim_numbers,
        /*precision_config=*/ArrayAttr{},
        /*algorithm=*/stablehlo::DotAlgorithmAttr{});

    if (is_natural_order) {
      // The dot_general is already in an appropriate result order.
      rewriter.replaceOp(einsum, ValueRange{dot_general_op});
    } else {
      // Generate a transpose.
      rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(
          einsum, einsum.getType(), dot_general_op,
          rewriter.getDenseI64ArrayAttr(result_perms));
    }
    return success();
  }
};

class LegalizeStablehloEinsumToDotGeneralPass
    : public impl::LegalizeStablehloEinsumToDotGeneralPassBase<
          LegalizeStablehloEinsumToDotGeneralPass> {
 public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<EinsumToDotGeneralPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateLegalizeStablehloEinsumToDotGeneralPass() {
  return std::make_unique<LegalizeStablehloEinsumToDotGeneralPass>();
}

static PassRegistration<LegalizeStablehloEinsumToDotGeneralPass> pass;

}  // namespace odml
}  // namespace mlir
