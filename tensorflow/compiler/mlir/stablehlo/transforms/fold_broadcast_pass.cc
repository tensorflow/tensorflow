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

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/stablehlo/transforms/stablehlo_passes.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {

static const APFloat &addSign(const APFloat &v, Type) { return v; }
static APSInt addSign(const APInt &v, Type t) {
  // Add signedness information to the value, treating signless as signed,
  // unless it's i1.
  return APSInt(v, t.isUnsignedInteger() || t.isSignlessInteger(1));
}

// Helper method that given 'shape' and 'current_index' representing
// index in broadcasted tensor, get the index in the flat original tensor.
// 'shape' is computed from the original shape and the broadcast dimensions to
// match result shape.
int64_t GetElementIndex(llvm::SmallVectorImpl<int64_t> &shape,
                        llvm::SmallVectorImpl<int64_t> &current_index) {
  int64_t ind = 0;
  int64_t mul = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    ind += (current_index[i] % shape[i]) * mul;
    mul *= shape[i];
  }
  return ind;
}

// Helper method that increment index represented in 'current_index_ptr'
// in the shape of 'result_shape'.
void IncrementIndex(ArrayRef<int64_t> result_shape,
                    llvm::SmallVectorImpl<int64_t> &current_index) {
  for (int i = result_shape.size() - 1; i >= 0; --i) {
    current_index[i]++;
    if (current_index[i] == result_shape[i]) {
      current_index[i] = 0;
    } else {
      break;
    }
  }
}

template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType>
Attribute ConstFoldBroadcastInDim(ShapedType result_type,
                                  DenseElementsAttr operand,
                                  DenseIntElementsAttr bcast_dims) {
  auto dimensions = llvm::to_vector(bcast_dims.getValues<int64_t>());
  const auto result_shape = result_type.getShape();
  // Index for the broadcasted matrix.
  llvm::SmallVector<int64_t, 16> current_index(result_type.getRank(), 0);
  // Computes the new operand shape using the original shape and the broadcast
  // dimensions to match result shape.
  llvm::SmallVector<int64_t, 16> operand_new_shape(result_type.getRank(), 1);
  for (int i = 0; i < dimensions.size(); ++i) {
    operand_new_shape[dimensions[i]] = operand.getType().getDimSize(i);
  }

  llvm::SmallVector<ElementValueT, 16> new_values;
  auto num_elements = result_type.getNumElements();
  new_values.reserve(num_elements);
  auto operand_values = operand.getValues<ElementValueT>();
  for (int64_t i = 0; i < num_elements; ++i) {
    const int64_t operand_index =
        GetElementIndex(operand_new_shape, current_index);
    new_values.push_back(*(operand_values.begin() + operand_index));
    IncrementIndex(result_shape, current_index);
  }
  return DenseElementsAttr::get(result_type,
                                ArrayRef<ElementValueT>(new_values));
}

template <typename Op, typename ElementType = Type, typename ValType,
          typename Convert>
static Attribute BinaryFolder(Op *op) {
  auto lhs_op = op->getLhs().template getDefiningOp<mhlo::ConstantOp>();
  auto rhs_op = op->getRhs().template getDefiningOp<mhlo::ConstantOp>();
  if (!lhs_op || !lhs_op) return {};

  auto lhs = dyn_cast_or_null<DenseElementsAttr>(lhs_op.getValue());
  auto rhs = dyn_cast_or_null<DenseElementsAttr>(rhs_op.getValue());
  if (!lhs || !rhs) return {};

  ShapedType type = mlir::cast<ShapedType>(op->getType());
  if (!type.hasStaticShape()) {
    return {};
  }

  Type etype = type.getElementType();

  // Evaluate for element types.
  if (!mlir::isa<ElementType>(etype)) {
    return {};
  }

  // Special case for folding splats no matter how large.
  // Only covers the case of both attrs being splats; operation-specific cases
  // like adding a zero or multiplying by one are handled elsewhere.
  SplatElementsAttr splatLhs = mlir::dyn_cast<SplatElementsAttr>(lhs);
  SplatElementsAttr splatRhs = mlir::dyn_cast<SplatElementsAttr>(rhs);
  if (splatLhs && splatRhs) {
    auto signedLhs = addSign(splatLhs.getSplatValue<ValType>(), etype);
    auto signedRhs = addSign(splatRhs.getSplatValue<ValType>(), etype);
    FailureOr<decltype(signedLhs)> result(Convert()(signedLhs, signedRhs));
    return succeeded(result) ? SplatElementsAttr::get(type, *result)
                             : Attribute();
  }

  SmallVector<ValType, 6> values;
  values.reserve(lhs.getNumElements());
  for (const auto zip : llvm::zip(lhs.template getValues<ValType>(),
                                  rhs.template getValues<ValType>())) {
    auto signedLhs = addSign(std::get<0>(zip), etype);
    auto signedRhs = addSign(std::get<1>(zip), etype);
    FailureOr<decltype(signedLhs)> result(Convert()(signedLhs, signedRhs));
    if (failed(result)) {
      return {};
    }
    values.push_back(std::move(*result));
  }

  return DenseElementsAttr::get(type, values);
}

template <typename BinaryOpType>
class FoldBroadcastInDimBeforeBinaryElementwiseOp
    : public OpRewritePattern<BinaryOpType> {
 public:
  using OpRewritePattern<BinaryOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOpType binary_op,
                                PatternRewriter &rewriter) const override {
    auto lhs = binary_op.getLhs();
    auto rhs = binary_op.getRhs();
    auto lhs_bcast_op = lhs.template getDefiningOp<mhlo::BroadcastInDimOp>();
    auto rhs_bcast_op = rhs.template getDefiningOp<mhlo::BroadcastInDimOp>();
    if ((lhs_bcast_op && rhs_bcast_op) || (!lhs_bcast_op && !rhs_bcast_op)) {
      return rewriter.notifyMatchFailure(
          binary_op, "Operands should have exactly one BroadcastInDim op.");
    }
    // When the operand other than the broadcast op is not a const op, we
    // should not fold broadcast op.
    auto binary_op_const_operand =
        (lhs_bcast_op ? rhs : lhs).template getDefiningOp<mhlo::ConstantOp>();
    if (!binary_op_const_operand) return failure();
    auto bcast_op = lhs_bcast_op ? lhs_bcast_op : rhs_bcast_op;
    auto const_op =
        bcast_op.getOperand().template getDefiningOp<mhlo::ConstantOp>();
    if (!const_op) return failure();
    auto const_val = dyn_cast_or_null<DenseElementsAttr>(const_op.getValue());
    if (!const_val) return failure();

    auto result_type =
        dyn_cast_or_null<ShapedType>(bcast_op.getResult().getType());
    if (!result_type || !result_type.hasStaticShape())
      return rewriter.notifyMatchFailure(binary_op,
                                         "Result type must have static shape.");

    auto bcast_dims = bcast_op.getBroadcastDimensions();
    auto elem_type = const_val.getElementType();
    Attribute result;
    if (mlir::isa<FloatType>(elem_type)) {
      result = ConstFoldBroadcastInDim<FloatAttr>(result_type, const_val,
                                                  bcast_dims);
    } else if (mlir::isa<IntegerType>(elem_type)) {
      result = ConstFoldBroadcastInDim<IntegerAttr>(result_type, const_val,
                                                    bcast_dims);
    } else {
      return rewriter.notifyMatchFailure(bcast_op, "Unsupported element type.");
    }
    Value new_const_op =
        rewriter.create<mhlo::ConstantOp>(bcast_op.getLoc(), result);
    rewriter.replaceOp(bcast_op, {new_const_op});
    return success();
  }
};

using FoldBroadcastInDimBeforeMulOp =
    FoldBroadcastInDimBeforeBinaryElementwiseOp<mhlo::MulOp>;

// Constant folds mhlo.mul, this folder doesn't have an upper limit on how many
// elements can be folded.
LogicalResult ConstantFoldMul(mhlo::MulOp op, PatternRewriter &rewriter) {
  ShapedType type = mlir::dyn_cast<ShapedType>(op.getType());
  Type etype = type.getElementType();
  Attribute result = {};
  if (mlir::isa<FloatType>(etype)) {
    result =
        BinaryFolder<mhlo::MulOp, FloatType, APFloat, std::multiplies<APFloat>>(
            &op);
  } else if (mlir::isa<IntegerType>(etype)) {
    result =
        BinaryFolder<mhlo::MulOp, IntegerType, APInt, std::multiplies<APSInt>>(
            &op);
  }
  if (result == Attribute()) return failure();
  Value new_const_op = rewriter.create<mhlo::ConstantOp>(op.getLoc(), result);
  rewriter.replaceOp(op, {new_const_op});
  return success();
}

class FoldBroadcastPass
    : public PassWrapper<FoldBroadcastPass, OperationPass<func::FuncOp>> {
 public:
  StringRef getArgument() const final { return "constant-fold-broadcast-pass"; }
  StringRef getDescription() const final {
    return "Constant folds BroadcastInDimOp before binary elementwise ops";
  }
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {}

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldBroadcastInDimBeforeMulOp>(&getContext());
    patterns.add(ConstantFoldMul);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createFoldBroadcastPass() {
  return std::make_unique<FoldBroadcastPass>();
}

static PassRegistration<FoldBroadcastPass> pass;

}  // namespace odml
}  // namespace mlir
