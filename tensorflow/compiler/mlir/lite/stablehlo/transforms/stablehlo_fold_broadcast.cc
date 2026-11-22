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

// StableHLO port of `-constant-fold-broadcast-pass`
// (tensorflow/compiler/mlir/stablehlo/transforms/fold_broadcast_pass.cc), which
// only handles MHLO.

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"

namespace mlir {
namespace odml {
namespace {

#define GEN_PASS_DEF_STABLEHLOFOLDBROADCASTPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h.inc"

const APFloat& AddSign(const APFloat& v, Type) { return v; }
APSInt AddSign(const APInt& v, Type t) {
  // Add signedness information to the value, treating signless as signed,
  // unless it's i1.
  return APSInt(v, t.isUnsignedInteger() || t.isSignlessInteger(1));
}

// Given `shape`, the operand shape expanded to the result rank, and
// `current_index`, an index into the broadcasted tensor, returns the index into
// the flat operand tensor.
int64_t GetElementIndex(ArrayRef<int64_t> shape,
                        ArrayRef<int64_t> current_index) {
  int64_t ind = 0;
  int64_t mul = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    ind += (current_index[i] % shape[i]) * mul;
    mul *= shape[i];
  }
  return ind;
}

// Increments `current_index` (row-major) within `result_shape`.
void IncrementIndex(ArrayRef<int64_t> result_shape,
                    SmallVectorImpl<int64_t>& current_index) {
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
DenseElementsAttr ConstFoldBroadcastInDim(ShapedType result_type,
                                          DenseElementsAttr operand,
                                          ArrayRef<int64_t> bcast_dims) {
  if (operand.isSplat()) {
    return SplatElementsAttr::get(result_type,
                                  operand.getSplatValue<Attribute>());
  }
  const auto result_shape = result_type.getShape();
  // Index into the broadcasted tensor.
  SmallVector<int64_t, 16> current_index(result_type.getRank(), 0);
  // The operand shape expanded to the result rank according to the broadcast
  // dimensions.
  SmallVector<int64_t, 16> operand_new_shape(result_type.getRank(), 1);
  for (int i = 0; i < bcast_dims.size(); ++i) {
    operand_new_shape[bcast_dims[i]] = operand.getType().getDimSize(i);
  }

  SmallVector<ElementValueT, 16> new_values;
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

template <typename ElementType, typename ValType, typename Convert>
DenseElementsAttr BinaryFolder(stablehlo::MulOp op) {
  auto lhs_op = op.getLhs().getDefiningOp<stablehlo::ConstantOp>();
  auto rhs_op = op.getRhs().getDefiningOp<stablehlo::ConstantOp>();
  if (!lhs_op || !rhs_op) return {};

  auto lhs = dyn_cast<DenseElementsAttr>(lhs_op.getValue());
  auto rhs = dyn_cast<DenseElementsAttr>(rhs_op.getValue());
  if (!lhs || !rhs) return {};

  auto type = cast<ShapedType>(op.getType());
  if (!type.hasStaticShape()) return {};

  Type etype = type.getElementType();
  if (!isa<ElementType>(etype)) return {};

  // Special case for folding splats no matter how large.
  auto splat_lhs = dyn_cast<SplatElementsAttr>(lhs);
  auto splat_rhs = dyn_cast<SplatElementsAttr>(rhs);
  if (splat_lhs && splat_rhs) {
    auto signed_lhs = AddSign(splat_lhs.getSplatValue<ValType>(), etype);
    auto signed_rhs = AddSign(splat_rhs.getSplatValue<ValType>(), etype);
    FailureOr<decltype(signed_lhs)> result(Convert()(signed_lhs, signed_rhs));
    return succeeded(result) ? SplatElementsAttr::get(type, *result)
                             : DenseElementsAttr();
  }

  SmallVector<ValType, 6> values;
  values.reserve(lhs.getNumElements());
  for (const auto zip :
       llvm::zip(lhs.getValues<ValType>(), rhs.getValues<ValType>())) {
    auto signed_lhs = AddSign(std::get<0>(zip), etype);
    auto signed_rhs = AddSign(std::get<1>(zip), etype);
    FailureOr<decltype(signed_lhs)> result(Convert()(signed_lhs, signed_rhs));
    if (failed(result)) return {};
    values.push_back(std::move(*result));
  }
  return DenseElementsAttr::get(type, values);
}

// Folds broadcast_in_dim(constant) into a constant when it feeds a multiply
// whose other operand is a constant.
class FoldBroadcastInDimBeforeMulOp
    : public OpRewritePattern<stablehlo::MulOp> {
 public:
  using OpRewritePattern<stablehlo::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::MulOp mul_op,
                                PatternRewriter& rewriter) const override {
    Value lhs = mul_op.getLhs();
    Value rhs = mul_op.getRhs();
    auto lhs_bcast_op = lhs.getDefiningOp<stablehlo::BroadcastInDimOp>();
    auto rhs_bcast_op = rhs.getDefiningOp<stablehlo::BroadcastInDimOp>();
    if ((lhs_bcast_op && rhs_bcast_op) || (!lhs_bcast_op && !rhs_bcast_op)) {
      return rewriter.notifyMatchFailure(
          mul_op, "Operands should have exactly one BroadcastInDim op.");
    }
    // When the operand other than the broadcast op is not a dense constant
    // (which `ConstantFoldMul` can fold), the broadcast op should not be
    // folded.
    auto mul_const_operand =
        (lhs_bcast_op ? rhs : lhs).getDefiningOp<stablehlo::ConstantOp>();
    if (!mul_const_operand ||
        !isa<DenseElementsAttr>(mul_const_operand.getValue())) {
      return failure();
    }
    auto bcast_op = lhs_bcast_op ? lhs_bcast_op : rhs_bcast_op;
    auto const_op =
        bcast_op.getOperand().getDefiningOp<stablehlo::ConstantOp>();
    if (!const_op) return failure();
    auto const_val = dyn_cast<DenseElementsAttr>(const_op.getValue());
    if (!const_val) return failure();

    auto result_type = dyn_cast<ShapedType>(bcast_op.getResult().getType());
    if (!result_type || !result_type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(mul_op,
                                         "Result type must have static shape.");
    }

    ArrayRef<int64_t> bcast_dims = bcast_op.getBroadcastDimensions();
    Type elem_type = const_val.getElementType();
    DenseElementsAttr result;
    if (isa<FloatType>(elem_type)) {
      result = ConstFoldBroadcastInDim<FloatAttr>(result_type, const_val,
                                                  bcast_dims);
    } else if (isa<IntegerType>(elem_type)) {
      result = ConstFoldBroadcastInDim<IntegerAttr>(result_type, const_val,
                                                    bcast_dims);
    } else {
      return rewriter.notifyMatchFailure(bcast_op, "Unsupported element type.");
    }
    // Only replace this multiply's use of the broadcast: other users (e.g. a
    // multiply with a non-constant operand) keep the broadcast.
    Value new_const =
        stablehlo::ConstantOp::create(rewriter, bcast_op.getLoc(), result);
    rewriter.modifyOpInPlace(
        mul_op, [&] { mul_op->setOperand(lhs_bcast_op ? 0 : 1, new_const); });
    return success();
  }
};

// Constant folds stablehlo.multiply, without an upper limit on the number of
// elements.
LogicalResult ConstantFoldMul(stablehlo::MulOp op, PatternRewriter& rewriter) {
  Type etype = cast<ShapedType>(op.getType()).getElementType();
  DenseElementsAttr result;
  if (isa<FloatType>(etype)) {
    result = BinaryFolder<FloatType, APFloat, std::multiplies<APFloat>>(op);
  } else if (isa<IntegerType>(etype)) {
    result = BinaryFolder<IntegerType, APInt, std::multiplies<APSInt>>(op);
  }
  if (!result) return failure();
  rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, result);
  return success();
}

class StablehloFoldBroadcastPass
    : public impl::StablehloFoldBroadcastPassBase<StablehloFoldBroadcastPass> {
 public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldBroadcastInDimBeforeMulOp>(&getContext());
    patterns.add(ConstantFoldMul);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateStablehloFoldBroadcastPass() {
  return std::make_unique<StablehloFoldBroadcastPass>();
}

static PassRegistration<StablehloFoldBroadcastPass> pass;

}  // namespace odml
}  // namespace mlir
