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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/sort.h"

#include <cstdint>

#include "llvm/ADT/ilist.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/hlo_matchers.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {
namespace {

using OpListType = llvm::iplist<Operation>;

template <typename ReturnOpType>
bool MatchTopKComparator(Region& comparator) {
  if (!comparator.hasOneBlock()) return false;
  Block& comparator_blk = comparator.front();

  OpListType& operations = comparator_blk.getOperations();
  if (operations.size() != 2) return false;

  auto compare_op =
      llvm::dyn_cast_or_null<mhlo::CompareOp>(&operations.front());
  auto return_op = llvm::dyn_cast_or_null<ReturnOpType>(&operations.back());
  if (!compare_op || !return_op) return false;

  if (compare_op.getComparisonDirection() != mhlo::ComparisonDirection::GT) {
    return false;
  }

  if (compare_op.getOperands()[0] != comparator_blk.getArgument(0) ||
      compare_op.getOperands()[1] != comparator_blk.getArgument(1)) {
    return false;
  }

  return return_op.getOperands().front() == compare_op.getResult();
}

bool IsSortOpNotTopK(mhlo::SortOp op) {
  if (op->getNumOperands() != 2) {
    return true;
  }

  auto keys_opr = op.getInputs().front();
  auto keys_type = llvm::cast<ShapedType>(keys_opr.getType());

  if (!keys_type.hasStaticShape() ||
      !keys_type.getElementType().isIntOrFloat()) {
    return true;
  }

  auto indices_opr = op.getInputs().back();
  auto indices_type = llvm::cast<ShapedType>(indices_opr.getType());

  if (!indices_type.hasStaticShape() ||
      !indices_type.getElementType().isInteger(32)) {
    return true;
  }

  const int64_t sort_dim = op.getDimension();
  const auto k = indices_type.getDimSize(sort_dim);
  const auto rank = keys_type.getRank();

  if (sort_dim != rank - 1 || k < 1) {
    return true;
  }

  OpBuilder b(op->getContext());
  if (!MatchIota(b.getI64TensorAttr({sort_dim}), indices_opr)) {
    return true;
  }

  if (!MatchTopKComparator<mhlo::ReturnOp>(op.getComparator())) {
    return true;
  }

  return false;
}

class LegalizeSortOp : public OpConversionPattern<mhlo::SortOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::SortOp sort_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};

LogicalResult LegalizeSortOp::matchAndRewrite(
    mhlo::SortOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  if (IsSortOpNotTopK(op)) {
    return failure();
  }

  auto keys = op.getInputs().front();
  auto indices = op.getInputs().back();
  auto indices_type = llvm::cast<ShapedType>(indices.getType());

  const int32_t k = indices_type.getShape().back();
  auto k_cst_attr = DenseIntElementsAttr::get(
      RankedTensorType::get({}, rewriter.getI32Type()), k);
  auto k_cst = rewriter.create<arith::ConstantOp>(op->getLoc(), k_cst_attr);

  rewriter.replaceOpWithNewOp<TFL::TopKV2Op>(op, keys.getType(),
                                             indices.getType(), keys, k_cst);

  return success();
}

}  // namespace

void PopulateSortPatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                          ConversionTarget& target) {
  patterns.add<LegalizeSortOp>(ctx);
  target.addDynamicallyLegalOp<mhlo::SortOp>(IsSortOpNotTopK);
}

}  // namespace mlir::odml
