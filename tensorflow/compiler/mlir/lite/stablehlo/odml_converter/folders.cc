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
#include <optional>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo

namespace mlir::odml {

namespace {

// Helper class for parsing operands to a foldable operation.
class FoldAdaptor {
 public:
  // Returns std::nullopt if the operation cannot be folded.
  static std::optional<FoldAdaptor> Create(Operation* operation) {
    auto foldable_opr = [](Value val) -> bool {
      return !llvm::isa<BlockArgument>(val) &&
             llvm::isa<stablehlo::ConstantOp>(val.getDefiningOp());
    };
    if (!llvm::all_of(operation->getOperands(), foldable_opr)) {
      return std::nullopt;
    }
    return FoldAdaptor(operation);
  }

  // Gets a list of ElementsAttr behind each constant operand.
  llvm::SmallVector<ElementsAttr> OperandData() {
    llvm::SmallVector<ElementsAttr> res;
    res.reserve(operation_->getNumOperands());
    for (auto opr : operation_->getOperands()) {
      auto op = llvm::dyn_cast<stablehlo::ConstantOp>(opr.getDefiningOp());
      res.push_back(op.getValue());
    }
    return res;
  }

  // Gets a pointer to the operation to be folded.
  Operation* Op() { return operation_; }

 private:
  explicit FoldAdaptor(Operation* operation) : operation_(operation) {}
  Operation* const operation_;
};

// APSInt provides operators which APInt does not, so allow for converting
// to APSInt for computation. Only APInts can be directly read from
// element attributes.
static const APFloat& AddSign(const APFloat& v) { return v; }
static APSInt AddSign(const APInt& v) { return APSInt(v); }

template <typename ResultType>
static LogicalResult FoldDivOpInternal(stablehlo::DivOp op,
                                       PatternRewriter& rewriter) {
  auto adaptor = FoldAdaptor::Create(op);
  if (!adaptor.has_value()) {
    return failure();
  }
  auto const_oprs = adaptor.value().OperandData();

  const bool lhs_splat = const_oprs[0].isSplat();
  const bool rhs_splat = const_oprs[1].isSplat();

  auto lhs_vals = const_oprs[0].getValues<ResultType>();
  auto rhs_vals = const_oprs[1].getValues<ResultType>();
  const auto num_results = std::max(lhs_vals.size(), rhs_vals.size());
  std::vector<ResultType> res;
  res.reserve(num_results);

  auto lhs_start = lhs_vals.begin();
  auto rhs_start = rhs_vals.begin();

  for (int i = 0; i < num_results; ++i) {
    auto lhs_val = lhs_splat ? *lhs_start : *(lhs_start++);
    auto rhs_val = rhs_splat ? *rhs_start : *(rhs_start++);
    auto signed_lhs_val = AddSign(lhs_val);
    auto signed_rhs_val = AddSign(rhs_val);
    if (signed_rhs_val.isZero()) {
      return failure();
    }
    res.push_back(signed_lhs_val / signed_rhs_val);
  }

  auto res_attr = DenseElementsAttr::get(
      const_oprs[0].getType().cast<RankedTensorType>(), res);
  rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(adaptor.value().Op(),
                                                     res_attr);
  return success();
}

static LogicalResult FoldDivOp(stablehlo::DivOp op, PatternRewriter& rewriter) {
  auto etype = op.getType().getElementType();
  if (etype.isa<FloatType>()) {
    return FoldDivOpInternal<APFloat>(op, rewriter);
  }
  if (etype.isa<IntegerType>()) {
    return FoldDivOpInternal<APInt>(op, rewriter);
  }
  return failure();
}
}  // namespace

void PopulateFolderPatterns(RewritePatternSet& patternSet) {
  patternSet.add(FoldDivOp);
}

}  // namespace mlir::odml
