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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_REDUCE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_REDUCE_H_

#include <cstdint>
#include <optional>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/hlo_matchers.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {

LogicalResult MatchReduceToArgMinMaxType1(mhlo::ReduceOp reduce_op,
                                          bool is_float, bool is_argmax);

LogicalResult MatchReduceToArgMinMaxType2(mhlo::ReduceOp reduce_op,
                                          bool is_argmax);

// Base class for converting mhlo::ReduceOp to TF/TFL ArgMax/ArgMin ops.
template <typename Reduce, typename ArgReduce, typename BooleanReduce,
          bool is_argmax>
class ConvertReduceOpToArgMinMax : public OpConversionPattern<mhlo::ReduceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ReduceOp reduce_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (reduce_op.getInputs().size() != 2) return failure();
    if (reduce_op.getDimensions().getNumElements() != 1) return failure();

    // Check that the operand init is the expected value.
    DenseElementsAttr operand_init;
    if (!matchPattern(reduce_op.getInitValues().front(),
                      m_Constant(&operand_init)))
      return failure();
    if (!IsValueInitValue(operand_init)) return failure();

    // Check that the iota init is zero.
    DenseElementsAttr iota_init;
    if (!matchPattern(reduce_op.getInitValues().back(), m_Constant(&iota_init)))
      return failure();
    if (iota_init.getValues<APInt>()[0] != 0) return failure();

    // Verify that the second argument is an Iota op along the same dimension
    // as the reduction.
    Value iota = reduce_op.getInputs().back();
    if (!MatchIota(reduce_op.getDimensions(), iota)) return failure();

    // Match the reduction computation.
    const bool is_float = operand_init.getElementType().isa<FloatType>();
    if (failed(MatchReduceToArgMinMaxType1(reduce_op, is_float, is_argmax)) &&
        failed(MatchReduceToArgMinMaxType2(reduce_op, is_argmax)))
      return rewriter.notifyMatchFailure(
          reduce_op, "Unsupported Reduce -> ArgMax/ArgMin pattern");

    Value operand = reduce_op.getInputs().front();
    int64_t axis = reduce_op.getDimensions().getValues<int64_t>()[0];

    auto dim_type = RankedTensorType::get({1}, rewriter.getI32Type());
    auto reduction_indices = rewriter.create<arith::ConstantOp>(
        reduce_op.getLoc(), dim_type,
        rewriter.getI32TensorAttr({static_cast<int32_t>(axis)}));

    // Generate a Max and an ArgMax of as the mhlo op returns both while in TF
    // we have separate ops for them. If only one of them is used then the other
    // one will be garbage collected later.
    if (!operand.getType().isa<ShapedType>()) return failure();
    auto operand_type = operand.getType().cast<ShapedType>();
    if (operand_type.getElementType().isInteger(1)) {
      // TF does not support min or max on boolean (int1) arguments.
      // Use AnyOp for MaxOp and AllOp for MinOp.
      auto tf_reduce_op = rewriter.create<BooleanReduce>(
          reduce_op.getLoc(), reduce_op->getResult(0).getType(), operand,
          reduction_indices,
          /*keep_dim=*/rewriter.getBoolAttr(false));
      auto tf_argreduce_op = rewriter.create<ArgReduce>(
          reduce_op.getLoc(), reduce_op->getResult(1).getType(), operand,
          reduction_indices);

      rewriter.replaceOp(reduce_op, {tf_reduce_op, tf_argreduce_op});
    } else {
      auto tf_reduce_op = rewriter.create<Reduce>(
          reduce_op.getLoc(), reduce_op->getResult(0).getType(), operand,
          reduction_indices,
          /*keep_dim=*/rewriter.getBoolAttr(false));

      auto tf_argreduce_op = rewriter.create<ArgReduce>(
          reduce_op.getLoc(), reduce_op->getResult(1).getType(), operand,
          reduction_indices);

      rewriter.replaceOp(reduce_op, {tf_reduce_op, tf_argreduce_op});
    }
    return success();
  }

  virtual bool IsValueInitValue(const DenseElementsAttr& attr) const = 0;
};

// Returns true if the given reduce op can be legalized to ArgMax/ArgMin ops.
std::optional<bool> IsReduceOpLegal(mhlo::ReduceOp reduce_op);

}  // namespace odml
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_REDUCE_H_
