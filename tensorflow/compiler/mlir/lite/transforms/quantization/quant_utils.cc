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

#include <optional>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
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
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"  // IWYU pragma: keep

namespace mlir::TFL {

std::optional<quant::QuantizedType> GetQTypeFromDefiningDequantize(
    mlir::Value value) {
  mlir::Operation* op = value.getDefiningOp();
  TFL::DequantizeOp dq_op = dyn_cast_or_null<TFL::DequantizeOp>(op);
  if (!dq_op) {
    return std::nullopt;
  }
  return cast<quant::QuantizedType>(
      getElementTypeOrSelf(dq_op.getInput().getType()));
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

  auto quantize = rewriter.create<TFL::QuantizeOp>(
      value.getLoc(), result_type, value, TypeAttr::get(result_type));
  // mark this quantize as a propagated Quantize.
  quantize->setAttr(kPropagatedQuantizeOpAttr, rewriter.getUnitAttr());

  auto dequantize = rewriter.create<TFL::DequantizeOp>(
      value.getLoc(), value.getType(), quantize);

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
}  // namespace mlir::TFL
