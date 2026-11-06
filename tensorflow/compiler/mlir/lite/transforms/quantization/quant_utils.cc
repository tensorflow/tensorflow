/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/transforms/quantization/quant_utils.h"

#include <algorithm>
#include <optional>
#include <tuple>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_interface.h.inc"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_traits.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"  // IWYU pragma: keep

namespace mlir::TFL {

namespace {

struct PropagatedQuantType {
  quant::QuantizedType qtype;
  bool is_from_result;
  bool is_propagated;

  // For sorting. Higher priority comes first.
  // Priority 1: non-propagated > propagated
  // Priority 2: result > operand
  bool operator<(const PropagatedQuantType& other) const {
    return std::make_tuple(!is_propagated, is_from_result) >
           std::make_tuple(!other.is_propagated, other.is_from_result);
  }
};

// A struct to hold the quantization information for a value.
struct QuantInfo {
  quant::QuantizedType qtype;
  // Whether the quantize op has the "propagated" attribute.
  bool is_propagated;
};

// Returns the quantized type from a `tfl.quantize` op if it is the only user
// of `value`.
//
//   value -> tfl.quantize
//
std::optional<QuantInfo> GetDefiningQuantInfo(mlir::Value value) {
  // If `value` has multiple uses, the `tfl.quantize` op cannot be fused with
  // `value`'s defining op. In this case, quantization parameter propagation is
  // not beneficial, so this function returns `std::nullopt`.
  if (!value.hasOneUse()) {
    return std::nullopt;
  }
  auto q_op = dyn_cast_or_null<TFL::QuantizeOp>(value.use_begin().getUser());
  if (!q_op) {
    return std::nullopt;
  }
  auto qtype = cast<quant::QuantizedType>(q_op.getQtype().getElementType());
  bool is_propagated = q_op->hasAttr(kPropagatedQuantizeOpAttr);
  return {{qtype, is_propagated}};
}

// Returns the quantized type and whether it is propagated if the value is
// an output of a Dequantize op.
//
//  QuantizeOp -> DequantizeOp -> value
//
// This function returns the quantized type of the QuantizeOp and whether it is
// propagated.
std::optional<QuantInfo> GetProducingQuantInfo(mlir::Value value) {
  auto dq_op = dyn_cast_or_null<TFL::DequantizeOp>(value.getDefiningOp());
  if (!dq_op) {
    return std::nullopt;
  }
  auto q_op =
      dyn_cast_or_null<TFL::QuantizeOp>(dq_op.getInput().getDefiningOp());
  if (!q_op) {
    // It can be a constant with quantized type. This is not a propagated Q.
    if (auto qtype = dyn_cast<quant::QuantizedType>(
            getElementTypeOrSelf(dq_op.getInput().getType()))) {
      return {{qtype, /*is_propagated=*/false}};
    }
    return std::nullopt;
  }

  auto qtype = cast<quant::QuantizedType>(q_op.getQtype().getElementType());
  bool is_propagated = q_op->hasAttr(kPropagatedQuantizeOpAttr);
  return {{qtype, is_propagated}};
}

std::vector<PropagatedQuantType> GetPropagatedQuantTypes(
    SameScalesOpInterface same_scales_op) {
  std::vector<PropagatedQuantType> propagated_types;
  mlir::Operation* op = same_scales_op.getOperation();

  for (mlir::Value result : op->getResults()) {
    if (auto res = GetDefiningQuantInfo(result)) {
      propagated_types.push_back(
          {res->qtype, /*is_from_result=*/true, res->is_propagated});
    }
  }

  for (mlir::Value operand : op->getOperands()) {
    if (auto res = GetProducingQuantInfo(operand)) {
      propagated_types.push_back(
          {res->qtype, /*is_from_result=*/false, res->is_propagated});
    }
  }
  return propagated_types;
}

}  // namespace

std::optional<quant::QuantizedType> GetQTypeFromDefiningDequantize(
    mlir::Value value) {
  mlir::Operation* op = value.getDefiningOp();
  TFL::DequantizeOp dq_op = dyn_cast_or_null<TFL::DequantizeOp>(op);
  if (!dq_op) {
    return std::nullopt;
  }
  if (auto qtype = dyn_cast<quant::QuantizedType>(
          getElementTypeOrSelf(dq_op.getInput().getType()))) {
    return qtype;
  }
  return std::nullopt;
}

std::optional<quant::QuantizedType> GetQTypeFromConsumingQuantize(
    mlir::Value value) {
  if (!value.hasOneUse()) {
    return std::nullopt;
  }
  mlir::Operation* op = value.use_begin().getUser();
  TFL::QuantizeOp q_op = dyn_cast_or_null<TFL::QuantizeOp>(op);
  if (!q_op) {
    return std::nullopt;
  }
  // The element type of the result of the quantize op is the quantized type.
  return cast<quant::QuantizedType>(getElementTypeOrSelf(q_op.getType()));
}

std::optional<quant::QuantizedType> GetPropagatedType(
    SameScalesOpInterface same_scales_op) {
  auto propagated_types = GetPropagatedQuantTypes(same_scales_op);
  if (propagated_types.empty()) return std::nullopt;
  std::sort(propagated_types.begin(), propagated_types.end());
  return propagated_types.front().qtype;
}

// Sets the insertion point for the rewriter safely. If the value is an op
// result, it sets the insertion point after the defining op. If the value is a
// block argument, it sets the insertion point to the start of the block.
static LogicalResult SetInsertionPointAfterDefiningOp(
    mlir::Value value, PatternRewriter& rewriter) {
  mlir::Operation* defining_op = value.getDefiningOp();
  if (defining_op) {
    // It's an operation result, insert after the defining op.
    rewriter.setInsertionPointAfter(defining_op);
  } else if (auto block_arg = dyn_cast<BlockArgument>(value)) {
    // It's a block argument, insert at the start of its owner block.
    rewriter.setInsertionPointToStart(block_arg.getOwner());
  } else {
    // Handle other unexpected cases, maybe emit an error or return.
    emitError(value.getLoc(),
              "Value is neither an op result nor a block argument.");
    return failure();
  }
  return success();
}

LogicalResult InsertQDQ(mlir::Value value, quant::QuantizedType qtype,
                        PatternRewriter& rewriter, mlir::Operation* target_op) {
  if (failed(SetInsertionPointAfterDefiningOp(value, rewriter))) {
    return failure();
  }

  // The new RankedTensorType with the element type being the quantized type.
  auto shaped_type = dyn_cast<mlir::ShapedType>(value.getType());
  if (!shaped_type) {
    return failure();
  }
  RankedTensorType result_type =
      RankedTensorType::get(shaped_type.getShape(), qtype);

  auto quantize = TFL::QuantizeOp::create(rewriter, value.getLoc(), result_type,
                                          value, TypeAttr::get(result_type));
  // mark this quantize as a propagated Quantize.
  quantize->setAttr(kPropagatedQuantizeOpAttr, rewriter.getUnitAttr());

  auto dequantize = TFL::DequantizeOp::create(rewriter, value.getLoc(),
                                              value.getType(), quantize);

  rewriter.replaceUsesWithIf(value, dequantize, [&](OpOperand& use) {
    // we have value -> Q -> dequantize so Q is already a "use" which we
    // need to keep.
    if (use.getOwner() == quantize) {
      return false;
    }
    // If a target_op is set, only replace the uses on that target.
    // This is helpful in the following case:
    // const -> [value] -> [use1] -> op1
    //                  \
    //                   \- [use2] -> op2
    //
    // In this case, we need to to be able to insert a QDQ only on once of the
    // uses and not necessarily all.
    if (target_op && use.getOwner() != target_op) {
      return false;
    }
    return true;
  });
  return success();
}

bool IsInt8SrqBlockwiseWeightConsumer(mlir::Operation* op) {
  if (auto fc = llvm::dyn_cast<TFL::FullyConnectedOp>(op)) {
    if (fc.getWeightsFormat() != "DEFAULT") return false;
    // The fused activations the FULLY_CONNECTED kernel supports with a
    // blockwise quantized filter.
    static constexpr llvm::StringLiteral kSupportedActivations[] = {
        "NONE", "RELU", "RELU_N1_TO_1", "RELU6"};
    if (!llvm::is_contained(kSupportedActivations,
                            fc.getFusedActivationFunction())) {
      return false;
    }
  } else if (!llvm::isa<TFL::BatchMatMulOp>(op)) {
    return false;
  }
  auto is_signed_i8 = [](mlir::Type type) {
    auto qtype =
        mlir::dyn_cast<quant::UniformQuantizedType>(getElementTypeOrSelf(type));
    return qtype && qtype.isSigned() &&
           qtype.getStorageTypeIntegralWidth() == 8;
  };
  auto lhs_dq = op->getOperand(0).getDefiningOp<DequantizeOp>();
  if (!lhs_dq || !is_signed_i8(lhs_dq.getInput().getType())) return false;
  // Only the first result is the matmul output (FULLY_CONNECTED may have an
  // unused shuffled-input workspace result for other weight formats).
  mlir::Value result = op->getResult(0);
  if (result.use_empty()) return false;
  return llvm::all_of(result.getUsers(), [&](mlir::Operation* user) {
    auto q = llvm::dyn_cast<QuantizeOp>(user);
    return q && is_signed_i8(q.getType());
  });
}

ElementsAttr GetConstantElements(mlir::Value value) {
  ElementsAttr attr;
  if (matchPattern(value, m_Constant(&attr))) return attr;
  if (auto const_op = value.getDefiningOp<TFL::ConstOp>()) {
    return const_op.getValue();
  }
  return nullptr;
}

bool IsFoldableBlockwiseDequantize(TFL::BlockwiseDequantizeOp bw_deq) {
  auto output_type = mlir::dyn_cast<RankedTensorType>(bw_deq.getType());
  if (!output_type || !output_type.getElementType().isF32()) return false;
  ElementsAttr values = GetConstantElements(bw_deq.getInput());
  if (!values || !mlir::isa<IntegerType>(values.getElementType())) return false;
  if (!GetConstantElements(bw_deq.getScales())) return false;
  if (!mlir::isa<NoneType>(bw_deq.getZeroPoints().getType()) &&
      !GetConstantElements(bw_deq.getZeroPoints())) {
    return false;
  }
  return true;
}

bool CanFoldBlockwiseWeightOperand(mlir::Operation* op, unsigned operand_index,
                                   mlir::Value operand) {
  if (operand_index != 1 || !IsInt8SrqBlockwiseWeightConsumer(op)) {
    return false;
  }
  auto bw_deq = operand.getDefiningOp<TFL::BlockwiseDequantizeOp>();
  if (!bw_deq || !IsFoldableBlockwiseDequantize(bw_deq)) return false;
  // The FULLY_CONNECTED kernel only takes rank 2 filters.
  if (llvm::isa<TFL::FullyConnectedOp>(op) &&
      mlir::cast<RankedTensorType>(bw_deq.getType()).getRank() != 2) {
    return false;
  }
  return true;
}

}  // namespace mlir::TFL
