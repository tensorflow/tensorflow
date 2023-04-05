/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This file contains legalizations common to mapping both TensorFlow and
// TensorFlow Lite to TOSA. It operates generically on ops and does not have
// a hard reference on either dialect.
//
// Conversion functions return std::nullopt on a legalization failure or a
// legalized value on success.  Callers must check for presence of an
// llvm::Optional value after each call.

#include "tensorflow/compiler/mlir/tosa/transforms/legalize_common.h"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/IR/TosaOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_utils.h"

namespace mlir {
namespace tosa {

static int64_t multiply_dims(int64_t a, int64_t b) {
  if (a == ShapedType::kDynamic || b == ShapedType::kDynamic) {
    return ShapedType::kDynamic;
  }
  return a * b;
}

static int64_t multiply_dims(llvm::ArrayRef<int64_t> dims, int64_t res = 1) {
  for (auto dim : dims) {
    if (ShapedType::isDynamic(dim)) {
      return ShapedType::kDynamic;
    }
    res = res * dim;
  }
  return res;
}

static int64_t count_dynamic_dims(llvm::ArrayRef<int64_t> dims) {
  int64_t count = 0;
  for (auto dim : dims)
    if (ShapedType::isDynamic(dim)) ++count;
  return count;
}

namespace {
// Given an axis that can be a positive or negative value and the tensor size,
// return the adjusted axis value wrapped around the tensor size.
int32_t adjust_axis(int32_t axis, int32_t tensor_size) {
  return axis >= 0 ? axis % tensor_size : (axis + tensor_size) % tensor_size;
}
};  // namespace

// Copied Nudge implementation from
// tensorflow/core/kernels/fake_quant_ops_functor.h.
// Suggested approach to avoid significant TensorFlow
// build dependency.
void tensorflow_nudge(const float min, const float max, const int quant_min,
                      const int quant_max, float* nudged_min, float* nudged_max,
                      float* scale) {
  const float quant_min_float = static_cast<float>(quant_min);
  const float quant_max_float = static_cast<float>(quant_max);
  *scale = (max - min) / (quant_max_float - quant_min_float);
  const float zero_point_from_min = quant_min_float - min / *scale;
  const uint16_t nudged_zero_point = [zero_point_from_min, quant_min,
                                      quant_min_float, quant_max,
                                      quant_max_float] {
    if (zero_point_from_min < quant_min_float) {
      return static_cast<uint16_t>(quant_min);
    }
    if (zero_point_from_min > quant_max_float) {
      return static_cast<uint16_t>(quant_max);
    }
    return static_cast<uint16_t>(std::round(zero_point_from_min));
  }();
  *nudged_min = (quant_min_float - nudged_zero_point) * (*scale);
  *nudged_max = (quant_max_float - nudged_zero_point) * (*scale);
}

// Lowers the Pack operator to TOSA.
std::optional<Value> convertPackOp(PatternRewriter& rewriter, Operation* op,
                                   Value result_value,
                                   SmallVectorImpl<Value>& inputs,
                                   int32_t axis) {
  //////////////////////////////////////////////////
  // Operator: output = Pack([values], axis) or output = Stack([values], axis)
  // Lowering:
  //
  // This operator is lowered into a series of pairwise tosa.concat()
  // operators and a reshape
  // Depending on the inputs, a tranpose operator is also generated:
  //
  // Step 1: concatenate the tensors
  // a1_concat = tosa.concat(input[0], input[1], axis)
  // for (i = 2; i < len(input); i++)
  //   a1_concat = tosa.concat(a1_concat, input[i], axis)
  //
  // Step 2: reshape to N+1 dimensions
  // a2_reshape = tosa.reshape(a1_concat, new_rank)
  //
  // Step 3: Transpose if a new dimension is being added:
  // if (axis == rank(values[0]):
  //   // perm will be [1, 2, 3, 0]
  //   a3_transpose = tosa.transpose(a2_reshape, perm)

  // Sanity check 1: make sure all input tensors have the same shape
  // if input[0] has shape [A, B, C], input[1] to input[N-1] should also have
  // shape[A, B, C]
  RankedTensorType result_type =
      result_value.getType().dyn_cast<RankedTensorType>();

  // Check for ranked tensor type.
  if (!result_type) {
    (void)rewriter.notifyMatchFailure(op, "result type not ranked tensor");
    return std::nullopt;
  }

  // Valid axis in TF is [-rank(input), rank(input))
  // Valid axis in TOSA is [0, rank(input))
  // Plus rank(input) once if axis is negative.
  RankedTensorType input_type =
      op->getOperand(0).getType().dyn_cast<RankedTensorType>();
  if (!input_type) {
    (void)rewriter.notifyMatchFailure(op, "input type not ranked tensor");
    return std::nullopt;
  }

  input_type = inputs[0].getType().dyn_cast<RankedTensorType>();
  if (!input_type) {
    (void)rewriter.notifyMatchFailure(op, "input 0 type not ranked tensor");
    return std::nullopt;
  }
  ArrayRef<int64_t> input0_tensor_shape = input_type.getShape();
  int input_tensor_rank = input0_tensor_shape.size();

  if (axis < 0 || axis > input_tensor_rank) {
    (void)rewriter.notifyMatchFailure(
        op, llvm::formatv("reduce axis {} is not in valid range "
                          "[-rank(input), rank(input)]",
                          axis));
    return std::nullopt;
  }

  for (int i = 1; i < inputs.size(); i++) {
    input_type = inputs[0].getType().dyn_cast<RankedTensorType>();
    if (!input_type) {
      (void)rewriter.notifyMatchFailure(
          op, llvm::formatv("input {} is not ranked)", i));
      return std::nullopt;
    }
    ArrayRef<int64_t> next_tensor_shape = input_type.getShape();
    if (next_tensor_shape.size() != input_tensor_rank) {
      (void)rewriter.notifyMatchFailure(op, "input tensor rank mismatch");
      return std::nullopt;
    }
    for (int d = 0; d < input0_tensor_shape.size(); d++) {
      if (input0_tensor_shape[d] != next_tensor_shape[d]) {
        (void)rewriter.notifyMatchFailure(op, "input tensor shape mismatch");
        return std::nullopt;
      }
    }
  }

  // If input tensors are rank 0, should reshape them to rank 1 size 1 before
  // performing concat.
  if (input_tensor_rank == 0) {
    SmallVector<int64_t, 1> reshape_rank1_size1_shape({1});
    RankedTensorType reshape_rank1_size1_type =
        tensorflow::GetTypeFromTFTensorShape(reshape_rank1_size1_shape,
                                             result_type.getElementType());
    DenseI64ArrayAttr shape_rank1_size1_attr = rewriter.getDenseI64ArrayAttr(
        tensorflow::ConvertMlirShapeToTF(reshape_rank1_size1_shape));
    for (int i = 0; i < inputs.size(); i++) {
      auto a0_reshape_op = CreateOpAndInfer<tosa::ReshapeOp>(
          rewriter, op->getLoc(), reshape_rank1_size1_type, inputs[i],
          shape_rank1_size1_attr);
      inputs[i] = a0_reshape_op.getResult();
    }
  }

  // Sanity check 2: axis can be from [0, rank(input)+1]
  // Where rank(input)+1 means create a new dimension
  // Negative values are also allowed up to -(rank(input)+1)
  // where the axis "wraps around".
  if (axis < 0) axis += input_tensor_rank;
  if ((axis < 0) || (axis > input_tensor_rank)) {
    (void)rewriter.notifyMatchFailure(op, "axis out of valid range");
    return std::nullopt;
  }

  // Sanity check 2: if input shape is [A, B, C], output shape should be [N,
  // A, B, C]
  // 2.a check output is rank(input) + 1
  SmallVector<int64_t> output_shape_vals(result_type.getShape().begin(),
                                         result_type.getShape().end());
  if (output_shape_vals.size() != (input_tensor_rank + 1)) {
    (void)rewriter.notifyMatchFailure(op, "output tensor rank mismatch");
    return std::nullopt;
  }
  // 2.b check output rank 0 is N
  if (output_shape_vals[axis] != inputs.size()) {
    (void)rewriter.notifyMatchFailure(op, "output tensor shape mismatch");
    return std::nullopt;
  }
  // Most of the cases when PackOp.axis() is within [0, rank(input) - 1].
  // We can directly concatenate along that axis and perform the reshape.
  // For example, stack N [A, B, C] input tensor ranks along axis = 1
  // after concatenation, output will be [A, N * B, C]
  // and then reshape it into [A, N, B, C]
  // a special case would be PackOp.axis() equal to rank(input), in which case
  // we can't directly concatenate along the PackOp.axis(), instead
  // we concat along axis=0, and reshape into [N, A, B, C]
  // and then we need an extra transpose to [A, B, C, N].
  int64_t concat_axis;
  SmallVector<int32_t> perm;
  SmallVector<int64_t> reshape_output_shape;
  if (axis == 0 && input_tensor_rank == 0) {
    concat_axis = 0;
  } else if (axis == input_tensor_rank) {
    concat_axis = 0;

    // A special case when stack axis is equal to input tensor rank:
    // Output shape is [A, B, C, N]
    // so reshape output will be [N, A, B, C]
    // and perm will be [1, 2, 3, 0].
    reshape_output_shape.push_back(output_shape_vals[axis]);
    for (int d = 0; d < input_tensor_rank; d++) {
      perm.push_back(d + 1);
      reshape_output_shape.push_back(output_shape_vals[d]);
    }
    perm.push_back(0);
  } else {
    // General case, doesn't need perm vector.
    concat_axis = axis;
    reshape_output_shape.assign(output_shape_vals.begin(),
                                output_shape_vals.end());
  }
  IntegerAttr concat_axis_attr = rewriter.getI64IntegerAttr(concat_axis);
  DenseI64ArrayAttr shape_attr = rewriter.getDenseI64ArrayAttr(
      tensorflow::ConvertMlirShapeToTF(reshape_output_shape));

  // Concat output shape will depend on concat_axis. E.g. [N * A, B, C]
  SmallVector<int64_t> concat_output_shape;
  if (input_tensor_rank == 0) {
    concat_output_shape.push_back(1);
  } else {
    for (int i = 0; i < input_tensor_rank; i++) {
      concat_output_shape.push_back(input0_tensor_shape[i]);
    }
  }

  concat_output_shape[concat_axis] =
      concat_output_shape[concat_axis] * inputs.size();
  RankedTensorType concat_type = tensorflow::GetTypeFromTFTensorShape(
      ArrayRef<int64_t>(concat_output_shape), result_type.getElementType());

  SmallVector<Value> inputs_0;
  for (int i = 0; i < inputs.size(); i++) {
    inputs_0.push_back(inputs[i]);
  }
  auto a1_concat_op = CreateOpAndInfer<tosa::ConcatOp>(
      rewriter, op->getLoc(), concat_type, inputs_0, concat_axis_attr);

  // Doesn't need reshape or transpose if input tensor is rank 0, since inputs
  // are reshaped beforehand.
  if (input_tensor_rank == 0) return a1_concat_op.getResult();

  // Reshape [N * A, B, C] to [N, A, B, C].
  RankedTensorType reshape_output_type = tensorflow::GetTypeFromTFTensorShape(
      reshape_output_shape, result_type.getElementType());

  auto a2_reshape_op = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(), reshape_output_type, a1_concat_op.getResult(),
      shape_attr);

  // If axis is equal to input tensor rank, then we need extra transpose
  // [N, A, B, C] to [A, B, C, N]
  if (axis == input_tensor_rank) {
    std::optional<Value> a3_transpose_perm = getConstTensor<int32_t>(
        rewriter, op, perm, {static_cast<int64_t>(perm.size())});

    if (!a3_transpose_perm) return std::nullopt;

    return CreateOpAndInfer<tosa::TransposeOp>(
               rewriter, op->getLoc(), result_type, a2_reshape_op.getResult(),
               a3_transpose_perm.value())
        .getResult();
  }

  return a2_reshape_op.getResult();
}

// Lowers the Unpack operator to TOSA
std::optional<SmallVector<Value>> convertUnpackOp(PatternRewriter& rewriter,
                                                  Operation* op,
                                                  Value input_value,
                                                  int32_t axis) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type) return std::nullopt;

  auto input_shape = input_type.getShape();
  int64_t input_rank = input_shape.size();

  SmallVector<Value> results_vec;

  // Negative axis allowed as long as it's within [-input_rank, input_rank).
  if (axis < 0) axis += input_rank;
  if ((axis < 0) || (axis >= input_rank)) {
    (void)rewriter.notifyMatchFailure(op, "axis out of valid range");
    return std::nullopt;
  }

  // Step 1: transpose 'axis' to leftmost dimension.
  Value transposed_input_value;
  if (axis != 0) {
    SmallVector<int32_t> perm;
    SmallVector<int64_t> a1_transpose_shape(input_rank);

    perm.push_back(axis);
    for (int i = 0; i < input_rank; i++) {
      if (i == axis) continue;
      perm.push_back(i);
    }

    std::optional<Value> a1_transpose_perm = getConstTensor<int32_t>(
        rewriter, op, perm, {static_cast<int64_t>(perm.size())});

    if (!a1_transpose_perm) return std::nullopt;

    for (int i = 0; i < input_rank; i++) {
      a1_transpose_shape[i] = input_shape[perm[i]];
    }

    auto a1_transpose_op = CreateOpAndInfer<tosa::TransposeOp>(
        rewriter, op->getLoc(),
        tensorflow::GetTypeFromTFTensorShape(a1_transpose_shape,
                                             input_type.getElementType()),
        input_value, a1_transpose_perm.value());

    transposed_input_value = a1_transpose_op.getResult();
  } else {
    // Do nothing if axis is already at leftmost dimension.
    transposed_input_value = input_value;
  }

  // Step 2: slice [N, A, B, C] into N [A, B, C].
  RankedTensorType transposed_input_type =
      transposed_input_value.getType().dyn_cast<RankedTensorType>();
  if (!transposed_input_type) return std::nullopt;

  auto transposed_input_shape = transposed_input_type.getShape();
  int64_t transposed_input_rank = transposed_input_shape.size();

  for (int i = 0; i < transposed_input_shape[0]; i++) {
    SmallVector<int64_t> begin_vals, size_vals, shape_vals;

    for (int j = 0; j < transposed_input_rank; j++) {
      if (j == 0) {
        begin_vals.push_back(i);
        size_vals.push_back(1);
      } else {
        begin_vals.push_back(0);
        size_vals.push_back(transposed_input_shape[j]);
        shape_vals.push_back(transposed_input_shape[j]);
      }
    }

    DenseI64ArrayAttr begin = rewriter.getDenseI64ArrayAttr(begin_vals);
    DenseI64ArrayAttr size = rewriter.getDenseI64ArrayAttr(
        tensorflow::ConvertMlirShapeToTF(size_vals));

    auto a2_slice_op = CreateOpAndInfer<tosa::SliceOp>(
        rewriter, op->getLoc(),
        tensorflow::GetTypeFromTFTensorShape(
            size_vals, transposed_input_type.getElementType()),
        transposed_input_value, begin, size);

    auto a3_reshape_op = CreateOpAndInfer<tosa::ReshapeOp>(
        rewriter, op->getLoc(),
        tensorflow::GetTypeFromTFTensorShape(
            shape_vals, transposed_input_type.getElementType()),
        a2_slice_op.getResult(),
        rewriter.getDenseI64ArrayAttr(
            tensorflow::ConvertMlirShapeToTF(shape_vals)));

    results_vec.push_back(a3_reshape_op.getResult());
  }

  return results_vec;
}

// Lowers the Select operator to TOSA.
std::optional<Value> convertSelectOp(PatternRewriter& rewriter, Operation* op,
                                     Value result_value, Value condition_value,
                                     Value x_value, Value y_value) {
  RankedTensorType result_type =
      result_value.getType().dyn_cast<RankedTensorType>();
  RankedTensorType condition_type =
      condition_value.getType().dyn_cast<RankedTensorType>();
  RankedTensorType x_type = x_value.getType().dyn_cast<RankedTensorType>();
  RankedTensorType y_type = y_value.getType().dyn_cast<RankedTensorType>();

  if (!x_type || !y_type || !condition_type) {
    (void)rewriter.notifyMatchFailure(op, "failed ranked tensor type check");
    return std::nullopt;
  }

  // First check whether we need to reshape the condition to match
  // the same rank as the then/else clauses.
  if (x_type == y_type && x_type.getShape() == condition_type.getShape()) {
    // Nothing to reshape.
    return CreateOpAndInfer<tosa::SelectOp>(rewriter, op->getLoc(), x_type,
                                            condition_value, x_value, y_value)
        .getResult();
  }

  if (!result_type) {
    (void)rewriter.notifyMatchFailure(op, "failed ranked tensor type check");
    return std::nullopt;
  }

  // Need to reshape the condition.
  SmallVector<int64_t> new_cond_dims(
      result_type.getRank() - condition_type.getRank(), 1);

  for (int i = 0; i < condition_type.getRank(); i++) {
    new_cond_dims.push_back(condition_type.getShape()[i]);
  }

  auto reshape_op = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(new_cond_dims,
                                           condition_type.getElementType()),
      condition_value,
      rewriter.getDenseI64ArrayAttr(
          tensorflow::ConvertMlirShapeToTF(new_cond_dims)));

  return CreateOpAndInfer<tosa::SelectOp>(rewriter, op->getLoc(), result_type,
                                          reshape_op, x_value, y_value)
      .getResult();
}

// Lowers the ZerosLike operator to TOSA by creating a constant
// of the desired type and shape.
std::optional<Value> convertZerosLikeOp(PatternRewriter& rewriter,
                                        Operation* op, Value result,
                                        Value input) {
  RankedTensorType result_type = result.getType().dyn_cast<RankedTensorType>();
  if (!result_type) {
    (void)rewriter.notifyMatchFailure(op, "result not ranked tensor type");
    return std::nullopt;
  }

  RankedTensorType input_type = input.getType().dyn_cast<RankedTensorType>();
  if (!input_type) {
    (void)rewriter.notifyMatchFailure(op, "input not ranked tensor type");
    return std::nullopt;
  }

  auto input_shape = input_type.getShape();

  ShapedType zero_type = tensorflow::GetTypeFromTFTensorShape(
      input_shape, input_type.getElementType());
  Attribute zero_attr = rewriter.getZeroAttr(zero_type);

  return CreateOpAndInfer<tosa::ConstOp>(rewriter, op->getLoc(), zero_type,
                                         zero_attr.cast<ElementsAttr>())
      .getResult();
}

// Lowers the Mul operator to TOSA.  For quantized types, this requires
// inserting rescale operators before and after the operation.
std::optional<Value> convertMultiplyOp(PatternRewriter& rewriter, Operation* op,
                                       Value output_val, Value input_lhs_val,
                                       Value input_rhs_val) {
  ShapedType input_lhs_type = input_lhs_val.getType().dyn_cast<ShapedType>();
  ShapedType input_rhs_type = input_rhs_val.getType().dyn_cast<ShapedType>();
  ShapedType output_type = output_val.getType().dyn_cast<ShapedType>();
  // Not a shaped tensor output
  if (!input_lhs_type || !input_rhs_type || !output_type) return std::nullopt;

  bool input_lhs_is_qtype =
      input_lhs_type.getElementType().isa<mlir::quant::UniformQuantizedType>();
  bool input_rhs_is_qtype =
      input_rhs_type.getElementType().isa<mlir::quant::UniformQuantizedType>();
  bool output_is_qtype =
      output_type.getElementType().isa<mlir::quant::UniformQuantizedType>();

  if (input_lhs_is_qtype != output_is_qtype ||
      input_rhs_is_qtype != output_is_qtype) {
    (void)rewriter.notifyMatchFailure(
        op,
        "input/output tensor should be all quantized or all floating-point");
    return std::nullopt;
  }

  if (output_is_qtype) {
    ShapedType rescale_type = output_type.clone(rewriter.getI32Type());
    auto input_lhs_qtype = input_lhs_type.getElementType()
                               .cast<mlir::quant::UniformQuantizedType>();
    auto input_rhs_qtype = input_rhs_type.getElementType()
                               .cast<mlir::quant::UniformQuantizedType>();
    auto output_qtype =
        output_type.getElementType().cast<mlir::quant::UniformQuantizedType>();

    // MLIR store scale as double, but TFLite store scale as float
    // Downcasting from double to float to match TFLite behavior
    float in_lhs_scale = input_lhs_qtype.getScale();
    float in_rhs_scale = input_rhs_qtype.getScale();
    float output_scale = output_qtype.getScale();

    double output_rescale_scale = in_lhs_scale * in_rhs_scale / output_scale;

    // 16bits x 16bits -> 32bits
    // 32bits can be rescaled with 32bits quantize multiplier back to 16bits
    bool scale32 = true;

    Value op1_rescale_lhs = removeZeroPointAndCastToInt32(
        rewriter, op, input_lhs_val, input_lhs_qtype.getZeroPoint());
    Value op2_rescale_rhs = removeZeroPointAndCastToInt32(
        rewriter, op, input_rhs_val, input_rhs_qtype.getZeroPoint());

    auto op3_mul_op1_op2 =
        CreateOpAndInfer<tosa::MulOp>(rewriter, op->getLoc(), rescale_type,
                                      op1_rescale_lhs, op2_rescale_rhs, 0);
    return buildRescale(rewriter, op, output_type, op3_mul_op1_op2.getResult(),
                        output_rescale_scale, 0, output_qtype.getZeroPoint(),
                        true, scale32);
  }

  return CreateOpAndInfer<tosa::MulOp>(rewriter, op->getLoc(), output_type,
                                       input_lhs_val, input_rhs_val, 0)
      .getResult();
}

// Lowers the SquaredDifference operator to TOSA.
std::optional<Value> convertSquaredDifferenceOp(PatternRewriter& rewriter,
                                                Operation* op, Value result,
                                                Value x, Value y) {
  // Squared-difference is (x-y)*(x-y).
  // This lowering calculates the difference and multiplies.
  ShapedType result_type = result.getType().dyn_cast<ShapedType>();
  if (!result_type) {
    (void)rewriter.notifyMatchFailure(op, "result not ranked tensor type");
    return std::nullopt;
  }

  ShapedType x_type = x.getType().dyn_cast<ShapedType>();
  ShapedType y_type = y.getType().dyn_cast<ShapedType>();
  if (!x_type || !y_type) {
    (void)rewriter.notifyMatchFailure(op, "inputs not ranked tensor type");
    return std::nullopt;
  }

  auto sub_op =
      CreateOpAndInfer<tosa::SubOp>(rewriter, op->getLoc(), result_type, x, y);
  return CreateOpAndInfer<tosa::MulOp>(rewriter, op->getLoc(), result_type,
                                       sub_op.getResult(), sub_op.getResult(),
                                       0)
      .getResult();
}

// Lowers the Round operator to TOSA.
std::optional<Value> convertRoundOp(PatternRewriter& rewriter, Operation* op,
                                    Value result, Value input) {
  // Implements banker's rounding by calculating floor(input + 0.5).
  ShapedType result_type = result.getType().dyn_cast<ShapedType>();
  if (!result_type) {
    (void)rewriter.notifyMatchFailure(op, "result not shaped tensor type");
    return std::nullopt;
  }

  ShapedType input_type = input.getType().dyn_cast<ShapedType>();
  if (!input_type) {
    (void)rewriter.notifyMatchFailure(op, "input not shaped tensor type");
    return std::nullopt;
  }

  auto add_op = CreateOpAndInfer<tosa::AddOp>(
      rewriter, op->getLoc(), result_type, input,
      getTosaConstTensorSingleF32(rewriter, op, 0.5));

  return CreateOpAndInfer<tosa::FloorOp>(rewriter, op->getLoc(), result_type,
                                         add_op.getResult())
      .getResult();
}

// Lowers ConcatV2 to TOSA Concat.
std::optional<Value> convertConcatV2Op(PatternRewriter& rewriter, Operation* op,
                                       ShapedType result_type,
                                       SmallVectorImpl<Value>& values,
                                       int32_t axis) {
  // Check all inputs are RankedTensorType
  for (auto v : values) {
    if (!v.getType().dyn_cast<RankedTensorType>()) {
      (void)rewriter.notifyMatchFailure(op, "value type not ranked tensor");
      return std::nullopt;
    }
  }

  mlir::quant::UniformQuantizedType result_quant_type =
      result_type.getElementType()
          .dyn_cast_or_null<mlir::quant::UniformQuantizedType>();

  SmallVector<Value> values_rescaled;

  for (auto v : values) {
    RankedTensorType operand_type = v.getType().dyn_cast<RankedTensorType>();
    mlir::quant::UniformQuantizedType operand_quant_type =
        operand_type.getElementType()
            .dyn_cast_or_null<mlir::quant::UniformQuantizedType>();

    // tfl.concat currently allows different scales for each input tensor, which
    // TFlite team will fix in:
    // https://github.com/tensorflow/tensorflow/issues/39658
    // For backward compatibility, we still need to support this artifact by
    // scaling inputs to let them have the same scales.
    if (result_quant_type && operand_quant_type) {
      double operand_scale = static_cast<double>(operand_quant_type.getScale());
      int32_t operand_zeropoint = operand_quant_type.getZeroPoint();

      double result_scale = static_cast<double>(result_quant_type.getScale());
      int32_t result_zeropoint = result_quant_type.getZeroPoint();

      // Rescale input if scale is not equal to output tensor scale.
      if (operand_scale != result_scale) {
        RankedTensorType rescale_type = tensorflow::GetTypeFromTFTensorShape(
            operand_type.getShape(), result_quant_type);
        Value rescale_op = buildRescale(
            rewriter, op, rescale_type, v, operand_scale / result_scale,
            operand_zeropoint, result_zeropoint, false, true);
        values_rescaled.push_back(rescale_op);
      } else {
        values_rescaled.push_back(v);
      }
    } else {
      values_rescaled.push_back(v);
    }
  }

  int32_t tensor_rank = values[0].getType().cast<RankedTensorType>().getRank();

  if (axis < 0) axis += tensor_rank;
  if ((axis < 0) || (axis > tensor_rank)) {
    (void)rewriter.notifyMatchFailure(op, "axis out of valid range");
    return std::nullopt;
  }

  auto concat_op = CreateOpAndInfer<tosa::ConcatOp>(
      rewriter, op->getLoc(), result_type, values_rescaled,
      rewriter.getI64IntegerAttr(axis));

  return concat_op.getResult();
}

// Lowers SpaceToBatchND to TOSA.
std::optional<Value> convertSpaceToBatchNDOp(PatternRewriter& rewriter,
                                             Operation* op, Value result_value,
                                             Value input_value,
                                             Value block_shape_value,
                                             Value paddings_value) {
  /////////////////////////////////////////////////
  // Operator: output = SpaceToBatchND(input, block_shape, paddings)
  // Lowering:
  //
  // SpaceToBatch input tensors are broken into three pieces:
  //   (a) batch dimension (N in NHWC)
  //   (b) input being transformed to batch dimension (typically H, W in NHWC)
  //   (c) remainder of input (typically C in NHWC)
  //
  // Step 0. Generate padding constant for the first reshape.
  //   No padding on the batch dimension
  //   The input paddings array is addressed as [input_rank][2]
  //   No padding on the remaining dimensions
  //
  //  a0_pad_const = tosa.const(input=Tensor<input_rank, 2>)
  //
  // Step 1. Pad the input tensor
  //
  //  a1_pad_input_op = tosa.pad(input=input, shape=a0_pad_const_op)
  //
  // Step 2. Reshape the padded structure of shape padded_shape to
  // [batch + padded_shape[1] / block_shape[0], block_shape[0], ...
  //    padded_shape[M] / block_shape[M-1], block_shape[M-1]] +
  //    remaining_shape
  //
  // block_rank = M (number of elements in block_shape)
  // New rank: input_rank + block_rank
  //
  //  a2_reshape_a1_op = tosa.reshape(input=a1_pad_input_op, shape=a2_shape)
  //
  // Step 3. Transpose dimensions to:
  //  block-shape +
  //  [batch] +
  //  [padded_shape[1] / block_shape[0],
  // ...
  //  [padded_shape[M] / block_shape[M-1]] +
  //  remaining_shape
  //
  // a3_transpose_a2_op = tosa.tranpose(input=a2_reshape_a1_op,
  // perms=a3_perm)
  //
  // Step 4. Reshape the transposed tensor to flatten block_shape stuff
  // into the batch dimension with the following shape:
  // [ batch * prod(block_shape)] +
  // [ padded_shape[1] / block_shape[0],
  //   ...,
  // padded_shape[M] / block_shape[M-1]] +
  // remaining_shape
  //
  //  a4_reshape_a3_op = tosa.reshape(input=a3_tranpose_a2_op,
  //  shape=a3_shape)
  //

  RankedTensorType result_type =
      result_value.getType().dyn_cast<RankedTensorType>();
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  RankedTensorType block_shape_type =
      block_shape_value.getType().dyn_cast<RankedTensorType>();
  RankedTensorType paddings_type =
      paddings_value.getType().dyn_cast<RankedTensorType>();

  // Not a ranked tensor output.
  if (!result_type) {
    (void)rewriter.notifyMatchFailure(op, "result type not ranked tensor");
    return std::nullopt;
  }
  if (!input_type) {
    (void)rewriter.notifyMatchFailure(op, "input type not ranked tensor");
    return std::nullopt;
  }
  if (!block_shape_type) {
    (void)rewriter.notifyMatchFailure(op, "block shape type not ranked tensor");
    return std::nullopt;
  }
  if (!paddings_type) {
    (void)rewriter.notifyMatchFailure(op, "paddings type not ranked tensor");
    return std::nullopt;
  }

  // Follow implementation in
  // tensorflow/compiler/tf2xla/kernels/spacetobatch_op.cc

  // So, to figure out the spatial_shape, remove the batch dimension and
  // then use the next block_rank dimensions.  The remaining dimensions are
  // remaining_shape.

  auto block_shape = block_shape_type.getShape();
  auto input_shape = input_type.getShape();

  int block_rank = block_shape[0];
  int64_t batch_size = input_shape[0];
  int input_rank = input_type.getRank();
  int remaining_shape_rank = input_rank - block_rank - 1;
  int64_t block_num_elems = 1;
  int64_t padding_sum = 0;

  ElementsAttr block_shape_elems;
  ElementsAttr paddings_elems;

  if (!matchPattern(block_shape_value, m_Constant(&block_shape_elems)))
    return std::nullopt;

  if (!matchPattern(paddings_value, m_Constant(&paddings_elems)))
    return std::nullopt;

  SmallVector<int32_t> a0_pad_const(2 * (input_rank));
  SmallVector<int64_t> padded_shape(input_rank);

  // 1. Pad based on paddings operand.  No padding on the batch dimension.
  // The a0_pad_const array is addressed as [input_rank][2], but
  // it is flattened to a 1D array because LLVM appears to only accept 1D.
  //
  // padded_shape[] is the shape of the padded output of step a1.
  // The name is retained for consistency with the TF reference code.
  padded_shape[0] = input_shape[0];

  // Batch dimension padding
  a0_pad_const[0] = 0;
  a0_pad_const[1] = 0;

  // This iterator seems to be the only reliable way to get
  // int values out of a multi-dimensional ElementsAttr.
  int idx = 0;

  for (auto i : paddings_elems.getValues<IntegerAttr>()) {
    a0_pad_const[idx + 2] = i.getInt();
    padding_sum += i.getInt();
    idx++;
  }

  // Insert padding on the spatial shape dimensions
  for (int i = 0; i < block_rank; i++) {
    padded_shape[i + 1] = input_shape[i + 1];
    if (!ShapedType::isDynamic(padded_shape[i + 1])) {
      int32_t lo_pad = a0_pad_const[2 * (i + 1) + 0];
      int32_t hi_pad = a0_pad_const[2 * (i + 1) + 1];
      padded_shape[i + 1] += lo_pad + hi_pad;
    }
  }

  // No padding on the remaining_shape dimensions
  for (int i = 0; i < remaining_shape_rank; i++) {
    a0_pad_const[2 * (i + block_rank + 1) + 0] = 0;
    a0_pad_const[2 * (i + block_rank + 1) + 1] = 0;
    padded_shape[i + block_rank + 1] = input_shape[i + block_rank + 1];
  }

  RankedTensorType a0_pad_const_attr_type =
      tensorflow::GetTypeFromTFTensorShape({(input_rank), 2},
                                           rewriter.getIntegerType(32));

  // Create a const op to generate the tensor type for the input padding array
  auto a0_pad_const_op = rewriter.create<tosa::ConstOp>(
      op->getLoc(), a0_pad_const_attr_type,
      DenseElementsAttr::get(a0_pad_const_attr_type,
                             llvm::ArrayRef(a0_pad_const)));

  auto a1_pad_input_op = CreateOpAndInfer<tosa::PadOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(padded_shape,
                                           result_type.getElementType()),
      input_value, a0_pad_const_op.getResult());

  // 2. Reshape the padded structure of shape padded_shape to
  // [batch + padded_shape[1] / block_shape[0], block_shape[0], ...
  //    padded_shape[M] / block_shape[M-1], block_shape[M-1]] +
  //    remaining_shape

  // block_rank = M (number of elements in block_shape)
  // New rank: input_rank + block_rank
  SmallVector<int64_t> a2_shape(1 + block_rank * 2 + remaining_shape_rank);

  // First dimension is batch.
  a2_shape[0] = input_type.getDimSize(0);
  for (int i = 0; i < block_rank; i++) {
    int32_t block_shape_val =
        block_shape_elems.getValues<IntegerAttr>()[i].getInt();
    a2_shape[1 + i * 2 + 0] = padded_shape[1 + i];
    if (a2_shape[1 + i * 2 + 0] != ShapedType::kDynamic) {
      a2_shape[1 + i * 2 + 0] /= block_shape_val;
    }

    a2_shape[1 + i * 2 + 1] = block_shape_val;
    block_num_elems *= block_shape_val;
  }

  // Copy in the remaining block shape.
  for (int i = 0; i < remaining_shape_rank; i++) {
    a2_shape[1 + block_rank * 2 + i] = input_shape[1 + block_rank + i];
  }

  auto a2_reshape_a1_op = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      RankedTensorType::get(a2_shape, result_type.getElementType()),
      a1_pad_input_op.getResult(),
      rewriter.getDenseI64ArrayAttr(
          tensorflow::ConvertMlirShapeToTF((a2_shape))));

  // 3. Transpose dimensions to:
  //  block-shape +
  //  [batch] +
  //  [padded_shape[1] / block_shape[0],
  // ...
  //  [padded_shape[M] / block_shape[M-1]] +
  //  remaining_shape
  int32_t a2_reshape_a1_rank =
      a2_reshape_a1_op.getResult().getType().cast<RankedTensorType>().getRank();
  SmallVector<int32_t> a3_perm(a2_reshape_a1_rank);
  SmallVector<int64_t> a3_transpose_shape(a2_reshape_a1_rank);

  for (int i = 0; i < block_rank; i++) {
    a3_perm[i] = 1 + 2 * i + 1;
    a3_perm[block_rank + 1 + i] = 1 + 2 * i;
  }
  a3_perm[block_rank] = 0;
  for (int i = 1 + block_rank * 2; i < a2_reshape_a1_rank; i++) {
    a3_perm[i] = i;
  }

  for (int i = 0; i < a3_transpose_shape.size(); i++) {
    a3_transpose_shape[i] = a2_shape[a3_perm[i]];
  }

  std::optional<Value> a3_transpose_const = getConstTensor<int32_t>(
      rewriter, op, a3_perm, {static_cast<int64_t>(a3_perm.size())});

  if (!a3_transpose_const) return std::nullopt;

  auto a3_transpose_a2_op = CreateOpAndInfer<tosa::TransposeOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(a3_transpose_shape,
                                           result_type.getElementType()),
      a2_reshape_a1_op.getResult(), a3_transpose_const.value());

  // 4. Reshape the transposed tensor to flatten block_shape
  // into the batch dimension with the following shape:
  // [ batch * prod(block_shape)] +
  // [ padded_shape[1] / block_shape[0],
  //   ...,
  // padded_shape[M] / block_shape[M-1]] +
  // remaining_shape
  SmallVector<int64_t> a4_reshape_shape(input_rank);

  // Batch
  a4_reshape_shape[0] = multiply_dims(batch_size, block_num_elems);

  // padded shape / block_shape.
  for (int i = 0; i < block_rank; i++) {
    int32_t block_shape_val =
        block_shape_elems.getValues<IntegerAttr>()[i].getInt();
    a4_reshape_shape[i + 1] = padded_shape[i + 1];
    if (a4_reshape_shape[i + 1] != ShapedType::kDynamic) {
      a4_reshape_shape[i + 1] /= block_shape_val;
    }
  }

  // Copy in remainder shape.
  for (int i = 0; i < remaining_shape_rank; i++) {
    a4_reshape_shape[1 + block_rank + i] = input_shape[1 + block_rank + i];
  }

  return CreateOpAndInfer<tosa::ReshapeOp>(
             rewriter, op->getLoc(), result_type,
             a3_transpose_a2_op.getResult(),
             rewriter.getDenseI64ArrayAttr(
                 tensorflow::ConvertMlirShapeToTF(a4_reshape_shape)))
      .getResult();
}

// Lowers BatchToSpaceND to TOSA.
std::optional<Value> convertBatchToSpaceNDOp(PatternRewriter& rewriter,
                                             Operation* op, Value result_value,
                                             Value input_value,
                                             Value block_shape_value,
                                             Value crops_value) {
  /////////////////////////////////////////////////
  // Operator: output = BatchToSpaceND(input, block_shape, clips)
  // Lowering:
  //
  // BatchToSpace input tensors are broken into three pieces:
  //   (a) batch dimension (N in NHWC)
  //   (b) input being transformed from batch dimension (typically H, W in
  //   NHWC)
  //   (c) remainder of input (typically C in NHWC)
  //
  // Step 1. Reshape input to:
  // [block_shape[0],
  // ...
  // [block_shape[M-1],
  // [batch / prod(block_shape)]
  // [input_shape[1],
  // ...
  // [input_shape[N-1]
  //
  // a1_reshape_input_op = tosa.reshape(input=input, shape=a1_shape)
  //
  // Step 2. Permute to shape
  // [ batch / prod(block_shape) ],
  // [ input_shape[1] ], [ block_shape[1] ]
  //  ...
  // [ input_shape[M] ], [ block_shape[M-1]
  // + remaining_input_shapes input_shape[M .. N-1]
  //
  // a2_transpose_a1 = tosa.transpose(input=a1_reshape_input_op,
  // shape=a2_shape)
  //
  // Step 3. Reshape to:
  // [ batch / prod(block_shape) ],
  // [input_shape[1] * block_shape[0] ],
  //    ..
  // [input_shape[M * block_shape[M-1],
  // + remaining input shapes [input_shape[M+1.. N-1]]
  //
  // a3_reshape_a2 = tosa.reshape(input=a2_transpose_a1, shape=a3_shape)
  //
  // Step 4. Crop the start/end dimensions according to crops of the
  // a3_reshape_a2 shape
  //
  // a4_slice_a3 = tosa.slice(input=a3_reshape_a2, start=a4_start,
  // size=a4_size)

  RankedTensorType result_type =
      result_value.getType().dyn_cast<RankedTensorType>();
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  RankedTensorType block_shape_type =
      block_shape_value.getType().dyn_cast<RankedTensorType>();
  RankedTensorType crops_type =
      crops_value.getType().dyn_cast<RankedTensorType>();

  if (!result_type) {
    (void)rewriter.notifyMatchFailure(op, "result type not ranked tensor");
    return std::nullopt;
  }
  if (!input_type) {
    (void)rewriter.notifyMatchFailure(op, "input type not ranked tensor");
    return std::nullopt;
  }
  if (!block_shape_type) {
    (void)rewriter.notifyMatchFailure(op, "block shape type not ranked tensor");
    return std::nullopt;
  }
  if (!crops_type) {
    (void)rewriter.notifyMatchFailure(op, "crops type not ranked tensor");
    return std::nullopt;
  }

  // Another 4-step process
  int block_rank = block_shape_type.getShape()[0];
  int input_rank = input_type.getRank();
  int64_t crops_dims = crops_type.getShape()[0];
  int remaining_shape_rank = input_rank - block_rank - 1;
  auto input_shape = input_type.getShape();

  ElementsAttr block_shape_elems;
  ElementsAttr crops_elems;

  if (!matchPattern(block_shape_value, m_Constant(&block_shape_elems))) {
    (void)rewriter.notifyMatchFailure(op, "block_shape not a constant");
    return std::nullopt;
  }

  if (!matchPattern(crops_value, m_Constant(&crops_elems))) {
    (void)rewriter.notifyMatchFailure(op, "crops not a constant");
    return std::nullopt;
  }

  SmallVector<int64_t> block_shape(block_rank);
  SmallVector<std::pair<int64_t, int64_t>> crops(crops_dims);

  // Extract values for block_shape and crops now.
  int64_t block_num_elems = 1;
  for (int i = 0; i < block_rank; i++) {
    int64_t block_shape_val =
        rewriter
            .getI32IntegerAttr(
                block_shape_elems.getValues<IntegerAttr>()[i].getInt())
            .getInt();
    block_num_elems *= block_shape_val;
    block_shape[i] = block_shape_val;
  }

  // This iterator seems to be the only reliable way to get
  // int values out of a multi-dimensional ElementsAttr
  SmallVector<int32_t> crops_const(2 * (crops_dims));
  int idx = 0;
  for (auto i : crops_elems.getValues<IntegerAttr>()) {
    crops_const[idx++] = i.getInt();
  }

  for (int i = 0; i < crops_dims; i++) {
    int crops_lo = crops_const[i * crops_dims + 0];
    int crops_hi = crops_const[i * crops_dims + 1];
    crops[i] = std::make_pair(crops_lo, crops_hi);
  }

  // Step 1. Reshape input to:
  // [block_shape[0],
  // ...
  // [block_shape[M-1],
  // [batch / prod(block_shape)]
  // [input_shape[1],
  // ...
  // [input_shape[N-1]
  SmallVector<int64_t> a1_shape(block_rank + input_rank);

  for (int i = 0; i < block_rank; i++) a1_shape[i] = block_shape[i];

  a1_shape[block_rank] = (input_shape[0] == ShapedType::kDynamic)
                             ? ShapedType::kDynamic
                             : input_shape[0] / block_num_elems;

  for (int i = 0; i < input_rank - 1; i++)
    a1_shape[i + block_rank + 1] = input_shape[i + 1];

  auto a1_reshape_input_op = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(a1_shape,
                                           result_type.getElementType()),
      input_value,
      rewriter.getDenseI64ArrayAttr(
          tensorflow::ConvertMlirShapeToTF(a1_shape)));

  // 2. Permute to shape
  // [ batch / prod(block_shape) ],
  // [ input_shape[1] ], [ block_shape[0] ]
  //  ...
  // [ input_shape[M] ], [ block_shape[M-1]
  // + remaining_input_shapes input_shape[M+1 .. N-1]

  // 2a. calculate the permutation
  SmallVector<int32_t> a2_perm(block_rank + input_rank);
  SmallVector<int64_t> a2_transpose_shape(block_rank + input_rank);

  a2_perm[0] = block_rank;
  for (int i = 0; i < block_rank; i++) {
    a2_perm[1 + i * 2 + 0] = block_rank + 1 + i;
    a2_perm[1 + i * 2 + 1] = i;
  }

  for (int i = 0; i < remaining_shape_rank; i++) {
    a2_perm[1 + 2 * block_rank + i] = 1 + 2 * block_rank + i;
  }

  // 2b. calculate the a2_permuted shape
  for (int i = 0; i < (block_rank + input_rank); i++) {
    a2_transpose_shape[i] = a1_shape[a2_perm[i]];
  }

  std::optional<Value> a2_transpose_perm = getConstTensor<int32_t>(
      rewriter, op, a2_perm, {static_cast<int64_t>(a2_perm.size())});

  if (!a2_transpose_perm) return std::nullopt;

  auto a2_transpose_a1_op = CreateOpAndInfer<tosa::TransposeOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(a2_transpose_shape,
                                           result_type.getElementType()),
      a1_reshape_input_op.getResult(), a2_transpose_perm.value());

  // Step 3. Reshape to:
  // [ batch / prod(block_shape) ],
  // [input_shape[1] * block_shape[0] ],
  //    ..
  // [input_shape[M * block_shape[M-1],
  // + remaining input shapes [input_shape[M+1.. N-1]]
  SmallVector<int64_t> a4_shape(input_rank);

  a4_shape[0] = input_shape[0];
  if (a4_shape[0] != ShapedType::kDynamic) {
    a4_shape[0] /= block_num_elems;
  }
  for (int i = 0; i < block_rank; i++) {
    a4_shape[1 + i] = multiply_dims(input_shape[i + 1], block_shape[i]);
  }
  for (int i = 0; i < remaining_shape_rank; i++) {
    a4_shape[1 + block_rank + i] = input_shape[block_rank + 1 + i];
  }

  auto a3_reshape_a2 = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(a4_shape,
                                           result_type.getElementType()),
      a2_transpose_a1_op.getResult(),
      rewriter.getDenseI64ArrayAttr(
          tensorflow::ConvertMlirShapeToTF(a4_shape)));

  // 4. Crop the start/end dimensions on 'spatial dimension' according to
  // crops
  // Use a slice operator to do the cropping.
  //
  // Calculate a beginning point and a size:
  // - Begin is the origin, offset by the lo crop amount in each dimension
  // - Size is the reshaped tensor size, minus the quantity (lo + hi) for each
  // dimension
  SmallVector<int64_t> a4_begin_vals(input_rank), a4_size_vals(input_rank);

  for (int i = 0; i < input_rank; i++) {
    // Batch dimension and remaining dimensions.
    if (i == 0 || i > crops_dims) {
      a4_begin_vals[i] = 0;
      a4_size_vals[i] = result_type.getShape()[i];
    } else {
      // Spatial dimension.
      assert(i - 1 >= 0 && i - 1 < crops_dims);
      a4_begin_vals[i] = crops[i - 1].first;
      a4_size_vals[i] = a4_shape[i] - crops[i - 1].first - crops[i - 1].second;
    }
  }

  return CreateOpAndInfer<tosa::SliceOp>(
             rewriter, op->getLoc(),
             tensorflow::GetTypeFromTFTensorShape(a4_size_vals,
                                                  result_type.getElementType()),
             a3_reshape_a2.getResult(),
             rewriter.getDenseI64ArrayAttr(a4_begin_vals),
             rewriter.getDenseI64ArrayAttr(
                 tensorflow::ConvertMlirShapeToTF(a4_size_vals)))
      .getResult();
}

// Lowers ExpandDims to TOSA.
std::optional<Value> convertExpandDimsOp(PatternRewriter& rewriter,
                                         Operation* op, Value result_value,
                                         Value input_value, Value dim_value) {
  // Lowers to a reshape op with 1's inserted in the appropriate dimensions.
  RankedTensorType output_type =
      result_value.getType().dyn_cast<RankedTensorType>();
  // Not a ranked tensor output
  if (!output_type) {
    (void)rewriter.notifyMatchFailure(op, "output type not ranked tensor");
    return std::nullopt;
  }

  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type) {
    (void)rewriter.notifyMatchFailure(op, "input type not ranked tensor");
    return std::nullopt;
  }

  auto input_shape = input_type.getShape();

  ElementsAttr dim_elem;
  if (!matchPattern(dim_value, m_Constant(&dim_elem))) return std::nullopt;

  if (dim_elem.getNumElements() > 1) {
    (void)rewriter.notifyMatchFailure(op,
                                      "expected single dimension to expand");
    return std::nullopt;
  }
  int32_t dim = dim_elem.getValues<IntegerAttr>()[0].getInt();
  int32_t input_size = input_shape.size();
  SmallVector<int64_t> reshape_dims;
  if (dim >= input_size) {  // add dim at end of tensor
    dim = input_size;
    for (int i = 0; i < input_shape.size(); i++) {
      reshape_dims.emplace_back(input_shape[i]);
    }
    reshape_dims.emplace_back(1);
  } else {
    if (dim < 0) {
      dim += input_size;
      if (dim < 0) {
        (void)rewriter.notifyMatchFailure(
            op, "dimension to expand + size of input shape < 0");
        return std::nullopt;
      }
    }
    for (int i = 0; i < input_size; i++) {
      if (i == dim) {
        reshape_dims.emplace_back(1);
      }
      reshape_dims.emplace_back(input_shape[i]);
    }
  }

  DenseI64ArrayAttr shape_attr = rewriter.getDenseI64ArrayAttr(
      tensorflow::ConvertMlirShapeToTF(reshape_dims));

  return CreateOpAndInfer<tosa::ReshapeOp>(rewriter, op->getLoc(), output_type,
                                           input_value, shape_attr)
      .getResult();
}

// Lowers Squeeze to TOSA.
std::optional<Value> convertSqueezeOp(PatternRewriter& rewriter, Operation* op,
                                      Value result_value, Value input_value,
                                      SmallVectorImpl<int32_t>& squeeze_dims) {
  // Lowers to a reshape op where dimensions in squeeze_dims with size=1
  // are removed.
  RankedTensorType output_type =
      result_value.getType().dyn_cast<RankedTensorType>();
  // Not a ranked tensor output
  if (!output_type) {
    (void)rewriter.notifyMatchFailure(op, "output type not ranked tensor");
    return std::nullopt;
  }

  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type) {
    (void)rewriter.notifyMatchFailure(op, "input type not ranked tensor");
    return std::nullopt;
  }

  auto input_shape = input_type.getShape();

  SmallVector<int64_t> reshape_dims;

  if (squeeze_dims.empty()) {  // remove all 1-dims
    for (int i = 0; i < input_shape.size(); i++) {
      if (input_shape[i] != 1) {
        reshape_dims.emplace_back(input_shape[i]);
      }
    }
  } else {
    for (auto& dim : squeeze_dims) {
      dim = dim < 0 ? dim + input_shape.size() : dim;
    }

    // Remove only specified dims.
    // First sort the array so they can be picked off in sequence.
    std::sort(squeeze_dims.begin(), squeeze_dims.end(),
              [](const int32_t a, const int32_t b) { return a < b; });

    int pos = 0;
    auto dim = squeeze_dims[pos];
    for (int i = 0; i < input_shape.size(); i++) {
      if (i == dim) {
        pos = pos + 1;
        if (pos < squeeze_dims.size())
          dim = squeeze_dims[pos];
        else
          dim = -1;  // Invalid
      } else {
        reshape_dims.emplace_back(input_shape[i]);
      }
    }
  }

  DenseI64ArrayAttr shape_attr = rewriter.getDenseI64ArrayAttr(
      tensorflow::ConvertMlirShapeToTF(reshape_dims));

  return CreateOpAndInfer<tosa::ReshapeOp>(rewriter, op->getLoc(), output_type,
                                           input_value, shape_attr)
      .getResult();
}

// Lowers ELU to a sequence of TOSA ops.
std::optional<Value> convertEluOp(PatternRewriter& rewriter, Operation* op,
                                  Value result_value, Value features_value) {
  // Lowers Elu using the following formula:
  // elu(x) = x < 0 ? (exp(x) - 1) : x
  // one = const({1});
  // zero = const({0});
  // one_bcast = reshape(one, [1, ..., rank(x) - 1])
  // zero_bcast = reshape(zero, [1, ..., rank(x) - 1])
  // a1 = exp(x);
  // a2 = sub(a1, one_bcast)
  // a3 = ge(x, zero_bcast)
  // a4 = select(a3, x, a2)
  RankedTensorType output_type =
      result_value.getType().dyn_cast<RankedTensorType>();
  // Not a ranked tensor output
  if (!output_type) {
    (void)rewriter.notifyMatchFailure(op, "output type not ranked tensor");
    return std::nullopt;
  }

  int32_t input_rank = output_type.getShape().size();
  SmallVector<int64_t> bcast_shape(input_rank, 1);

  // Can't directly create size=1, rank=rank(input) tensor because
  // it will be optimized out.  Instead, create rank0 tensor and reshape later.
  Value one_const_op = getTosaConstTensorSingleF32(rewriter, op, 1.0);

  Value zero_const_op = getTosaConstTensorSingleF32(rewriter, op, 0.0);

  auto a1_exp_in_op = CreateOpAndInfer<tosa::ExpOp>(
      rewriter, op->getLoc(), output_type, features_value);

  auto a2_sub_a1_one_op =
      CreateOpAndInfer<tosa::SubOp>(rewriter, op->getLoc(), output_type,
                                    a1_exp_in_op.getResult(), one_const_op);

  auto a3_ge_in_zero_op = CreateOpAndInfer<tosa::GreaterEqualOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(output_type.getShape(),
                                           rewriter.getIntegerType(1)),
      features_value, zero_const_op);

  return CreateOpAndInfer<tosa::SelectOp>(
             rewriter, op->getLoc(), output_type, a3_ge_in_zero_op.getResult(),
             features_value, a2_sub_a1_one_op.getResult())
      .getResult();
}

// Lowers Softmax to a sequence of TOSA ops.
std::optional<Value> convertSoftmaxOp(PatternRewriter& rewriter, Operation* op,
                                      Value result_value, Value logits_value,
                                      double beta) {
  // softmax = exp(logits) / reduce_sum(exp(logits), -1)
  //
  // or equivalently multiply exp(-max(logits)) to both numerator and
  // denominator we get:
  //
  // softmax = exp(logits - max(logits)) / reduce_sum(exp(logits -
  // max(logits)), -1)
  //
  // Second equation is used for both quantized and fp lowering.
  // For quantized case, we can restrict input to exp() be negative,
  // and thus LUT can always be within [0.0, 1.0].
  // For fp case, the normalization in the equation is required to prevent
  // float overflow in softmax's intermediate calculations.
  RankedTensorType output_type =
      result_value.getType().dyn_cast<RankedTensorType>();
  RankedTensorType input_type =
      logits_value.getType().dyn_cast<RankedTensorType>();

  // Not a ranked tensor input/output
  if (!output_type || !input_type) {
    (void)rewriter.notifyMatchFailure(op,
                                      "input and result not ranked tensors");
    return std::nullopt;
  }

  // reduce_sum on last dimension
  int32_t input_rank = input_type.getShape().size();
  ArrayRef<int64_t> logits_shape = output_type.getShape();

  if (input_type.getElementType().isa<mlir::quant::QuantizedType>() &&
      output_type.getElementType().isa<mlir::quant::QuantizedType>()) {
    SmallVector<int64_t> rsum_shape_v(input_type.getShape().begin(),
                                      input_type.getShape().end() - 1);
    rsum_shape_v.push_back(1);
    ArrayRef<int64_t> rsum_shape(rsum_shape_v);
    // The if condition already checks if these are UQTs
    mlir::quant::UniformQuantizedType in_quant_type =
        input_type.getElementType().cast<mlir::quant::UniformQuantizedType>();
    mlir::quant::UniformQuantizedType out_quant_type =
        output_type.getElementType().cast<mlir::quant::UniformQuantizedType>();

    auto int16_element_qtype = mlir::quant::UniformQuantizedType::get(
        true, rewriter.getIntegerType(16), rewriter.getF32Type(), 1.0f, 0,
        -32768, 32767);
    RankedTensorType int16_logits_type =
        tensorflow::GetTypeFromTFTensorShape(logits_shape, int16_element_qtype);
    RankedTensorType int32_logits_type = tensorflow::GetTypeFromTFTensorShape(
        logits_shape, rewriter.getIntegerType(32));
    RankedTensorType int16_rsum_type =
        tensorflow::GetTypeFromTFTensorShape(rsum_shape, int16_element_qtype);
    RankedTensorType int32_rsum_type = tensorflow::GetTypeFromTFTensorShape(
        rsum_shape, rewriter.getIntegerType(32));

    if (in_quant_type.getStorageTypeIntegralWidth() == 8) {
      // Step 1. get x - max(x)
      Value op1_rescale_in =
          buildRescale(rewriter, op, int32_logits_type, logits_value, 1.0f,
                       in_quant_type.getZeroPoint(), 0, false, true);

      auto op2_reducemax_op1 = CreateOpAndInfer<tosa::ReduceMaxOp>(
          rewriter, op->getLoc(), int32_rsum_type, op1_rescale_in,
          rewriter.getI64IntegerAttr(input_rank - 1));

      auto op3_sub_op1_op2 = CreateOpAndInfer<tosa::SubOp>(
          rewriter, op->getLoc(), int32_logits_type, op1_rescale_in,
          op2_reducemax_op1.getResult());

      // Step 2. get exp() result
      // Implemented with 4 16-bit table lookup
      // We use a 16-bit table lookup, as the result of x - max(x) for
      // 8-bit values is a 9-bit value. We only use the bottom 8 bits of each
      // table to avoid having the slope between two 16-bit table entries be
      // greater than 16 bits, causing potential interpolation errors
      auto exp_func = [](double x) -> double { return std::exp(x); };

      Value exp_table_const_01, exp_table_const_02, exp_table_const_03,
          exp_table_const_04;
      getTosaConst32bitTable(rewriter, op, beta * in_quant_type.getScale(), 0,
                             exp_func, exp_table_const_01, exp_table_const_02,
                             exp_table_const_03, exp_table_const_04);

      Value op4_rescale_op3 =
          buildRescale(rewriter, op, int16_logits_type,
                       op3_sub_op1_op2.getResult(), 128.0, 0, 0, false, true);

      // Input is 9.7, where lower 7 bits are all zeros.
      // Output is 23 bits, where lower 7 bits should be all zeros as well,
      // since there's no interpolation here.
      auto op5_table_op4_bits_31_24 = CreateOpAndInfer<tosa::TableOp>(
          rewriter, op->getLoc(), int32_logits_type, op4_rescale_op3,
          exp_table_const_01);

      auto op6_table_op4_bits_23_16 = CreateOpAndInfer<tosa::TableOp>(
          rewriter, op->getLoc(), int32_logits_type, op4_rescale_op3,
          exp_table_const_02);

      auto op7_table_op4_bits_15_8 = CreateOpAndInfer<tosa::TableOp>(
          rewriter, op->getLoc(), int32_logits_type, op4_rescale_op3,
          exp_table_const_03);

      auto op8_table_op4_bits_7_0 = CreateOpAndInfer<tosa::TableOp>(
          rewriter, op->getLoc(), int32_logits_type, op4_rescale_op3,
          exp_table_const_04);

      // To get 16 bits upper/lower value, we need to right shift 7 bits
      // And then we reconstruct 32-bit value we need (upper << 24) + lower
      // So effectively we left shift the 1st group of 8-bit with 17 bits
      auto op7_lshift_op5 = CreateOpAndInfer<tosa::LogicalLeftShiftOp>(
          rewriter, op->getLoc(), int32_logits_type,
          op5_table_op4_bits_31_24.getResult(),
          getTosaConstTensorSingleI32(rewriter, op, 17));

      // For the 2nd group of 8-bit, we need to >> 7 AND << 16 ==> << 9
      auto op8_lshift_op6 = CreateOpAndInfer<tosa::LogicalLeftShiftOp>(
          rewriter, op->getLoc(), int32_logits_type,
          op6_table_op4_bits_23_16.getResult(),
          getTosaConstTensorSingleI32(rewriter, op, 9));

      // For the 3rd 8-bit, we need to >> 7 AND << 8 ==> << 1
      auto op9_lshift_op7 = CreateOpAndInfer<tosa::LogicalLeftShiftOp>(
          rewriter, op->getLoc(), int32_logits_type,
          op7_table_op4_bits_15_8.getResult(),
          getTosaConstTensorSingleI32(rewriter, op, 1));

      // For the last 8-bit, we only need to >> 7
      auto op10_rshift_op8 = CreateOpAndInfer<tosa::ArithmeticRightShiftOp>(
          rewriter, op->getLoc(), int32_logits_type,
          op8_table_op4_bits_7_0.getResult(),
          getTosaConstTensorSingleI32(rewriter, op, 7), true);

      // Add together all the 8-bit groups
      // Add [1+2]
      auto op11_add_op7_op8 = CreateOpAndInfer<tosa::AddOp>(
          rewriter, op->getLoc(), int32_logits_type, op7_lshift_op5.getResult(),
          op8_lshift_op6.getResult());

      // Add [1+2+3]
      auto op12_add_op11_op9 = CreateOpAndInfer<tosa::AddOp>(
          rewriter, op->getLoc(), int32_logits_type,
          op11_add_op7_op8.getResult(), op9_lshift_op7.getResult());

      // Add [1+2+3+4]
      auto op13_add_op12_op10 = CreateOpAndInfer<tosa::AddOp>(
          rewriter, op->getLoc(), int32_logits_type,
          op12_add_op11_op9.getResult(), op10_rshift_op8.getResult());

      // Step 3. get sum(exp()). output 12.19
      auto op14_rshift_op13_12 = CreateOpAndInfer<tosa::ArithmeticRightShiftOp>(
          rewriter, op->getLoc(), int32_logits_type,
          op13_add_op12_op10.getResult(),
          getTosaConstTensorSingleI32(rewriter, op, 12), true);

      auto op15_reducesum_op14 = CreateOpAndInfer<tosa::ReduceSumOp>(
          rewriter, op->getLoc(), int32_rsum_type,
          op14_rshift_op13_12.getResult(),
          rewriter.getI64IntegerAttr(input_rank - 1));

      // Step 4. calculate reciprocal(sum(exp()))
      // CLZ returns the number of leading zeros which equals to headroom + 1
      auto op16_clz_op15 =
          CreateOpAndInfer<tosa::ClzOp>(rewriter, op->getLoc(), int32_rsum_type,
                                        op15_reducesum_op14.getResult());

      // minus one to get headroom
      auto op17_sub_op16 = CreateOpAndInfer<tosa::SubOp>(
          rewriter, op->getLoc(), int32_rsum_type, op16_clz_op15.getResult(),
          getTosaConstTensorSingleI32(rewriter, op, 1));

      // Left shift to get s1.30 format
      auto op18_lshift_op15_op17 = CreateOpAndInfer<tosa::LogicalLeftShiftOp>(
          rewriter, op->getLoc(), int32_rsum_type,
          op15_reducesum_op14.getResult(), op17_sub_op16.getResult());

      // Step 5. Calculate one_over_one_plus_x() with Newton-Raphson division
      // with 3 iterations.
      // Need two magic constants 48/17 and -32/17 from Newton-Raphson algorithm
      // We need to operate in s2.29 since 48/17 is > 2.0
      // Reference: gemmlowp/fixedpoint/fixedpoint.h
      Value half_denominator = op18_lshift_op15_op17.getResult();
      Value four = getTosaConstTensorSingleI32(rewriter, op, 4);
      Value F2_one = getTosaConstTensorSingleI32(rewriter, op, (1U << 29));
      Value constant_48_over_17 =
          getTosaConstTensorSingleI32(rewriter, op, 1515870810);
      Value constant_neg_32_over_17 =
          getTosaConstTensorSingleI32(rewriter, op, -1010580540);

      // F2 x = constant_48_over_17 + half_denominator *
      // constant_neg_32_over_17;
      auto op19_mul_half_denominator = CreateOpAndInfer<tosa::MulOp>(
          rewriter, op->getLoc(), int32_rsum_type, half_denominator,
          constant_neg_32_over_17, 31);

      auto op20_add_op19 = CreateOpAndInfer<tosa::AddOp>(
          rewriter, op->getLoc(), int32_rsum_type,
          op19_mul_half_denominator.getResult(), constant_48_over_17);

      // Newton-Raphson 3x iteration
      Value nr_x = op20_add_op19.getResult();
      for (int i = 0; i < 3; i++) {
        // half_denominator_times_x =
        // SaturatingRoundingDoublingHighMul(half_denominator, x)
        auto op21_mul_x_half_denominator = CreateOpAndInfer<tosa::MulOp>(
            rewriter, op->getLoc(), int32_rsum_type, nr_x, half_denominator,
            31);

        // F2 one_minus_half_denominator_times_x = F2::One() -
        // half_denominator_times_x
        auto op22_sub_one_op21 = CreateOpAndInfer<tosa::SubOp>(
            rewriter, op->getLoc(), int32_rsum_type, F2_one,
            op21_mul_x_half_denominator.getResult());

        // SaturatingRoundingDoublingHighMul(x,
        // one_minus_half_denominator_times_x)
        auto op23_mul_x_op22 = CreateOpAndInfer<tosa::MulOp>(
            rewriter, op->getLoc(), int32_rsum_type, nr_x,
            op22_sub_one_op21.getResult(), 31);

        // x + Rescale<2>(x * one_minus_half_denominator_times_x)
        auto op24_mul_op23_four = CreateOpAndInfer<tosa::MulOp>(
            rewriter, op->getLoc(), int32_rsum_type,
            op23_mul_x_op22.getResult(), four, 0);

        auto op25_add_x_op24 = CreateOpAndInfer<tosa::AddOp>(
            rewriter, op->getLoc(), int32_rsum_type, nr_x,
            op24_mul_op23_four.getResult());

        nr_x = op25_add_x_op24.getResult();
      }

      // Step 6. multiply exp(x) with 1 / sum(exp(x))
      // combined with Rescale<0>(ExactMulByPot<-1>(x))
      // so shift 30 instead of 31
      auto op26_mul_op13_x = CreateOpAndInfer<tosa::MulOp>(
          rewriter, op->getLoc(), int32_logits_type,
          op13_add_op12_op10.getResult(), nr_x, 31 - 1);

      // Right shift amount is
      // num_bits_over_unit + 31 - (sizeof(OutputT) * 8 =
      // (12 - headroom_plus_one) + 31 - 8 =
      // (12 + 31 - 8) - headroom_plus_one
      auto op27_sub_op16 = CreateOpAndInfer<tosa::SubOp>(
          rewriter, op->getLoc(), int32_rsum_type,
          getTosaConstTensorSingleI32(rewriter, op, 12 + 31 - 8),
          op16_clz_op15.getResult());

      auto op28_rshift_op26_op27 =
          CreateOpAndInfer<tosa::ArithmeticRightShiftOp>(
              rewriter, op->getLoc(), int32_logits_type,
              op26_mul_op13_x.getResult(), op27_sub_op16.getResult(), true);

      return buildRescale(rewriter, op, output_type,
                          op28_rshift_op26_op27.getResult(), 1.0, 0,
                          out_quant_type.getZeroPoint(), false, true);

    } else if (in_quant_type.getStorageTypeIntegralWidth() == 16) {
      // Step 1. get x - max(x)
      Value op1_rescale_in =
          buildRescale(rewriter, op, int32_logits_type, logits_value, 1.0f,
                       in_quant_type.getZeroPoint(), 0, false, true);

      auto op2_reducemax_op1 = CreateOpAndInfer<tosa::ReduceMaxOp>(
          rewriter, op->getLoc(), int32_rsum_type, op1_rescale_in,
          rewriter.getI64IntegerAttr(input_rank - 1));

      // output range is [-65535, 0]
      auto op3_sub_op1_op2 = CreateOpAndInfer<tosa::SubOp>(
          rewriter, op->getLoc(), int32_logits_type, op1_rescale_in,
          op2_reducemax_op1.getResult());

      auto exp_func = [](double x) -> double { return std::exp(x); };

      // Follow TFLite reference: tensorflow/lite/kernels/activations.cc
      Value exp_table_const =
          getTosaConst16bitTable(rewriter, op, exp_func, -10.0, 0);

      double input_diff_scale = in_quant_type.getScale() / (10.0 / 65535.0);

      // Step 2. rescale input from [-65535, 0] to [-32768, 32767] for LUT input
      Value op4_rescale_op3 = buildRescale(
          rewriter, op, int32_logits_type, op3_sub_op1_op2.getResult(),
          /*scale=*/input_diff_scale, /*input_zp=*/0, /*output_zp=*/0,
          /*double_round=*/true, /*scale32=*/true);
      auto op5_add_op4 = CreateOpAndInfer<tosa::AddOp>(
          rewriter, op->getLoc(), int32_logits_type, op4_rescale_op3,
          getTosaConstTensorSingleI32(rewriter, op, 32767));

      auto op6_cast_op5 = CreateOpAndInfer<tosa::CastOp>(
          rewriter, op->getLoc(), int16_logits_type, op5_add_op4.getResult());

      // Step 3. get exp() result
      // Output is 15.7.
      // In 8-bit case, no interpolation here, since input should be right on
      // table entry.
      auto op7_table_op6 = CreateOpAndInfer<tosa::TableOp>(
          rewriter, op->getLoc(), int32_logits_type, op6_cast_op5,
          exp_table_const);

      // Right shift 7 bits. output 15. Shouldn't lose any precision since last
      // 7 bits should be all 0.
      auto op8_rshift_op7 = CreateOpAndInfer<tosa::ArithmeticRightShiftOp>(
          rewriter, op->getLoc(), int32_logits_type, op7_table_op6.getResult(),
          getTosaConstTensorSingleI32(rewriter, op, 7), true);

      // Step 4. get sum(exp()). output 16.15
      auto op9_reducesum_op8 = CreateOpAndInfer<tosa::ReduceSumOp>(
          rewriter, op->getLoc(), int32_rsum_type, op8_rshift_op7.getResult(),
          rewriter.getI64IntegerAttr(input_rank - 1));

      // Step 5. calculate reciprocal(sum(exp()))
      // CLZ returns 32 - first non zero bit
      auto op10_clz_op9 =
          CreateOpAndInfer<tosa::ClzOp>(rewriter, op->getLoc(), int32_rsum_type,
                                        op9_reducesum_op8.getResult());

      auto op11_sub_op10 = CreateOpAndInfer<tosa::SubOp>(
          rewriter, op->getLoc(), int32_rsum_type, op10_clz_op9.getResult(),
          getTosaConstTensorSingleI32(rewriter, op, 1));

      // Left shift to get  1.30 format
      auto op12_lshift_op9_op11 = CreateOpAndInfer<tosa::LogicalLeftShiftOp>(
          rewriter, op->getLoc(), int32_rsum_type,
          op9_reducesum_op8.getResult(), op11_sub_op10.getResult());

      // Subtract (1 << 30) to make 0 <= x <= 1 under 0.30 format
      auto op13_sub_op12 = CreateOpAndInfer<tosa::SubOp>(
          rewriter, op->getLoc(), int32_rsum_type,
          op12_lshift_op9_op11.getResult(),
          getTosaConstTensorSingleI32(rewriter, op, (1u << 30)));

      // Right shift 14 bits to get output range [0, 65535]
      auto op14_rshift_op13 = CreateOpAndInfer<tosa::ArithmeticRightShiftOp>(
          rewriter, op->getLoc(), int32_rsum_type, op13_sub_op12.getResult(),
          getTosaConstTensorSingleI32(rewriter, op, 14), true);

      // Remap input to [-32768, 32767] for LUT input
      auto op15_add_op14 = CreateOpAndInfer<tosa::SubOp>(
          rewriter, op->getLoc(), int32_rsum_type, op14_rshift_op13.getResult(),
          getTosaConstTensorSingleI32(rewriter, op, 32768));
      auto op16_cast_op15 = CreateOpAndInfer<tosa::CastOp>(
          rewriter, op->getLoc(), int16_rsum_type, op15_add_op14.getResult());

      // Generate table for 1 / (1 + x), for 0 <= x <= 1
      auto one_over_one_plus_x_func = [](double x) -> double {
        return 1.0 / (1.0 + x);
      };

      Value one_over_one_plus_x_table_const = getTosaConst16bitTable(
          rewriter, op, one_over_one_plus_x_func, 0.0, 1.0);

      // Get (1 / sum(exp(x))) result as 23 bits (including sign bit)
      auto op17_table_op16 = CreateOpAndInfer<tosa::TableOp>(
          rewriter, op->getLoc(), int32_rsum_type, op16_cast_op15,
          one_over_one_plus_x_table_const);

      // Right shift 7 bits back to 0.15
      auto op18_rshift_op17 = CreateOpAndInfer<tosa::ArithmeticRightShiftOp>(
          rewriter, op->getLoc(), int32_rsum_type, op17_table_op16.getResult(),
          getTosaConstTensorSingleI32(rewriter, op, 7), true);

      // Step 6. multiply exp(max-x) with 1 / sum(exp(max-x))
      // lhs: 0.15, rhs: 0.15, output: 0.30
      auto op19_mul_op18_op8 = CreateOpAndInfer<tosa::MulOp>(
          rewriter, op->getLoc(), int32_logits_type, op18_rshift_op17,
          op8_rshift_op7, 0);

      auto op20_sub_op10 = CreateOpAndInfer<tosa::SubOp>(
          rewriter, op->getLoc(), int32_rsum_type,
          getTosaConstTensorSingleI32(rewriter, op, 31),
          op10_clz_op9.getResult());

      // Apply the clz back, we get 0.15 output
      // [0, 32767] corresponding to [0.0, 1.0]
      auto op21_rshift_op19_op20 =
          CreateOpAndInfer<tosa::ArithmeticRightShiftOp>(
              rewriter, op->getLoc(), int32_logits_type,
              op19_mul_op18_op8.getResult(), op20_sub_op10.getResult(), true);

      return buildRescale(rewriter, op, output_type,
                          op21_rshift_op19_op20.getResult(),
                          (1.0 / out_quant_type.getScale()) * (1.0 / 32768.0),
                          0, out_quant_type.getZeroPoint(), false, true);
    } else {
      (void)rewriter.notifyMatchFailure(op, "unknown quantization bitwidth");
      return std::nullopt;
    }
  } else {
    SmallVector<int64_t> rsum_shape_v(input_type.getShape().begin(),
                                      input_type.getShape().end());
    rsum_shape_v[input_rank - 1] = 1;
    ArrayRef<int64_t> rsum_shape(rsum_shape_v);

    // Floating-point lowering is more direct:
    //
    // op1 = reducemax(logits)
    // op2 = sub(logits, op1)
    // op3 = exp(op2)
    // op4 = reduce_sum(op3, -1)
    // op5 = reciprocal(op4)
    // op6 = mul(op3, op5)
    RankedTensorType rsum_type = tensorflow::GetTypeFromTFTensorShape(
        rsum_shape, output_type.getElementType());
    RankedTensorType logits_type = tensorflow::GetTypeFromTFTensorShape(
        logits_shape, output_type.getElementType());

    // Step 1. get x - max(x)
    auto max_logits = CreateOpAndInfer<tosa::ReduceMaxOp>(
        rewriter, op->getLoc(), rsum_type, logits_value,
        rewriter.getI64IntegerAttr(input_rank - 1));
    auto normalized_logits =
        CreateOpAndInfer<tosa::SubOp>(rewriter, op->getLoc(), logits_type,
                                      logits_value, max_logits.getResult());

    // Step 2. get exp(x - max(x))
    auto exp_norm_logits = CreateOpAndInfer<tosa::ExpOp>(
        rewriter, op->getLoc(), output_type, normalized_logits);

    // Step 3. reuse softmax numerator to obtain denominator
    // Keep dims so we don't need to reshape later
    auto reducesum = CreateOpAndInfer<tosa::ReduceSumOp>(
        rewriter, op->getLoc(), rsum_type, exp_norm_logits.getResult(),
        rewriter.getI64IntegerAttr(input_rank - 1));
    auto denominator = CreateOpAndInfer<tosa::ReciprocalOp>(
        rewriter, op->getLoc(), reducesum.getType(), reducesum.getResult());

    return CreateOpAndInfer<tosa::MulOp>(rewriter, op->getLoc(), output_type,
                                         exp_norm_logits.getResult(),
                                         denominator.getResult(), 0)
        .getResult();
  }
}

// Lowers LogSoftmax to a sequence of TOSA ops.
std::optional<Value> convertLogSoftmaxOp(PatternRewriter& rewriter,
                                         Operation* op, Value result_value,
                                         Value logits_value) {
  // log_softmax = log(exp(logits) / reduce_sum(exp(logits), -1))
  // op1 = exp(logits)
  // op2 = reduce_sum(op1, -1)
  // op3 = reciprocal(op2)
  // op4 = mul(op1, op3)
  // op5 = log(op4)

  TensorType output_type = result_value.getType().dyn_cast<TensorType>();
  // Not a tensor output
  if (!output_type) {
    (void)rewriter.notifyMatchFailure(op, "output type not tensor");
    return std::nullopt;
  }

  RankedTensorType input_type =
      op->getOperand(0).getType().dyn_cast<RankedTensorType>();
  if (!input_type) {
    (void)rewriter.notifyMatchFailure(op, "input type not ranked tensor");
    return std::nullopt;
  }

  mlir::quant::UniformQuantizedType in_quant_type =
      input_type.getElementType()
          .dyn_cast_or_null<mlir::quant::UniformQuantizedType>();
  mlir::quant::UniformQuantizedType out_quant_type =
      output_type.getElementType()
          .dyn_cast_or_null<mlir::quant::UniformQuantizedType>();
  if (in_quant_type || out_quant_type) {
    (void)rewriter.notifyMatchFailure(
        op, "quantized log_softmax lowering not implemented yet");
    return std::nullopt;
  }

  auto op1_exp_in = CreateOpAndInfer<tosa::ExpOp>(rewriter, op->getLoc(),
                                                  output_type, logits_value);

  // reduce_sum on last dimension
  int32_t input_rank = input_type.getShape().size();
  // Keep dims so we don't need to reshape later
  auto op2_reducesum_op1 = CreateOpAndInfer<tosa::ReduceSumOp>(
      rewriter, op->getLoc(),
      UnrankedTensorType::get(output_type.getElementType()),
      op1_exp_in.getResult(), rewriter.getI64IntegerAttr(input_rank - 1));
  auto op3_reciprocal_op2 = CreateOpAndInfer<tosa::ReciprocalOp>(
      rewriter, op->getLoc(), op2_reducesum_op1.getType(),
      op2_reducesum_op1.getResult());

  auto op4_mul_op1_op3 = CreateOpAndInfer<tosa::MulOp>(
      rewriter, op->getLoc(), output_type, op1_exp_in.getResult(),
      op3_reciprocal_op2.getResult(), 0);

  return CreateOpAndInfer<tosa::LogOp>(rewriter, op->getLoc(), output_type,
                                       op4_mul_op1_op3.getResult())
      .getResult();
}

// Lowers SpaceToDepth to a sequence of TOSA ops.  Supports NHWC.
std::optional<Value> convertSpaceToDepthOp(PatternRewriter& rewriter,
                                           Operation* op, Value result_value,
                                           Value input_value,
                                           IntegerAttr block_size_attr,
                                           StringAttr data_format) {
  // NHWC lowering version:
  // a2 = tf.reshape(a, [orig_shape[0], orig_shape[1]//b, b, orig_shape[2]//b,
  // b, orig_shape[3]])
  // a3 = tf.transpose(a2, [0, 1, 3, 2, 4, 5])
  // a4 = tf.reshape(a3, [orig_shape[0], orig_shape[1]//b, orig_shape[2]//b,
  // orig_shape[3]*b*b])
  // return a4
  RankedTensorType output_type =
      result_value.getType().dyn_cast<RankedTensorType>();

  // Not a ranked tensor output.
  if (!output_type) {
    (void)rewriter.notifyMatchFailure(op, "output type not ranked tensor");
    return std::nullopt;
  }

  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type) {
    (void)rewriter.notifyMatchFailure(op, "input type not ranked tensor");
    return std::nullopt;
  }

  if (input_type.getRank() != 4) {
    (void)rewriter.notifyMatchFailure(op, "input rank not 4");
    return std::nullopt;
  }

  auto input_shape = input_type.getShape();

  if (!block_size_attr) {  // This is a required parameter
    (void)rewriter.notifyMatchFailure(op, "block size attribute not set");
    return std::nullopt;
  }

  SmallVector<int64_t, 2> block_size;
  block_size.assign(2, block_size_attr.getInt());

  if (!data_format) data_format = rewriter.getStringAttr("NHWC");

  if (data_format.getValue().str() != "NHWC") {
    (void)rewriter.notifyMatchFailure(op, "data format not NHWC");
    return std::nullopt;
  }

  assert(block_size[0] * block_size[1] != 0);

  SmallVector<int64_t, 6> a_reshape_dims;
  a_reshape_dims.push_back(input_shape[0]);
  a_reshape_dims.push_back(input_shape[1] / block_size[0]);
  a_reshape_dims.push_back(block_size[0]);
  a_reshape_dims.push_back(input_shape[2] / block_size[1]);
  a_reshape_dims.push_back(block_size[1]);
  a_reshape_dims.push_back(input_shape[3]);

  RankedTensorType a_reshape_output_type = tensorflow::GetTypeFromTFTensorShape(
      a_reshape_dims, output_type.getElementType());
  auto a2_reshape_a_op = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(), a_reshape_output_type, input_value,
      rewriter.getDenseI64ArrayAttr(
          tensorflow::ConvertMlirShapeToTF(a_reshape_dims)));

  std::optional<Value> a3_transpose_perm = getConstTensor<int32_t>(
      rewriter, op, /*vec=*/{0, 1, 3, 2, 4, 5}, /*shape=*/{6});

  if (!a3_transpose_perm) return std::nullopt;

  auto a3_transpose_a2_op = CreateOpAndInfer<tosa::TransposeOp>(
      rewriter, op->getLoc(), a_reshape_output_type,
      a2_reshape_a_op.getResult(), a3_transpose_perm.value());

  SmallVector<int64_t, 4> a3_reshape_dims;
  a3_reshape_dims.push_back(input_shape[0]);
  a3_reshape_dims.push_back(input_shape[1] / block_size[0]);
  a3_reshape_dims.push_back(input_shape[2] / block_size[1]);
  a3_reshape_dims.push_back(input_shape[3] * block_size[0] * block_size[1]);

  RankedTensorType a3_reshape_output_type =
      tensorflow::GetTypeFromTFTensorShape(a3_reshape_dims,
                                           output_type.getElementType());
  return CreateOpAndInfer<tosa::ReshapeOp>(
             rewriter, op->getLoc(), a3_reshape_output_type,
             a3_transpose_a2_op.getResult(),
             rewriter.getDenseI64ArrayAttr(
                 tensorflow::ConvertMlirShapeToTF(a3_reshape_dims)))
      .getResult();
}

// Lowers DepthToSpace to a sequence of TOSA ops.  Supports NHWC.
std::optional<Value> convertDepthToSpaceOp(PatternRewriter& rewriter,
                                           Operation* op, Value result_value,
                                           Value input_value,
                                           IntegerAttr block_size_attr,
                                           StringAttr data_format) {
  // NHWC version
  // a2 = tf.reshape(a, [orig_shape[0], orig_shape[1], orig_shape[2], b, b,
  // orig_shape[3] // (b*b)])
  // a3 = tf.transpose(a2, [0, 1, 3, 2, 4, 5])
  // a4 = tf.reshape(a3, [orig_shape[0], orig_shape[1] * b, orig_shape[2] * b,
  // orig_shape[3] // (b*b)])
  // return a4

  RankedTensorType output_type =
      result_value.getType().dyn_cast<RankedTensorType>();

  // Not a ranked tensor output
  if (!output_type) {
    (void)rewriter.notifyMatchFailure(op, "output type not ranked tensor");
    return std::nullopt;
  }

  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type) {
    (void)rewriter.notifyMatchFailure(op, "input type not ranked tensor");
    return std::nullopt;
  }

  if (input_type.getRank() != 4) return std::nullopt;
  auto input_shape = input_type.getShape();

  if (!block_size_attr) {  // This is a required parameter
    (void)rewriter.notifyMatchFailure(op, "block size attribute not set");
    return std::nullopt;
  }

  SmallVector<int64_t, 2> block_size;
  block_size.assign(2, block_size_attr.getInt());

  if (!data_format) data_format = rewriter.getStringAttr("NHWC");
  if (data_format.getValue().str() != "NHWC") {
    (void)rewriter.notifyMatchFailure(op, "data format not NHWC");
    return std::nullopt;
  }

  assert(block_size[0] * block_size[1] != 0);

  SmallVector<int64_t, 6> a_reshape_dims;
  a_reshape_dims.push_back(input_shape[0]);
  a_reshape_dims.push_back(input_shape[1]);
  a_reshape_dims.push_back(input_shape[2]);
  a_reshape_dims.push_back(block_size[0]);
  a_reshape_dims.push_back(block_size[1]);
  a_reshape_dims.push_back(input_shape[3] / (block_size[0] * block_size[1]));

  RankedTensorType a_reshape_output_type = tensorflow::GetTypeFromTFTensorShape(
      a_reshape_dims, output_type.getElementType());
  auto a2_reshape_a_op = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(), a_reshape_output_type, input_value,
      rewriter.getDenseI64ArrayAttr(
          tensorflow::ConvertMlirShapeToTF(a_reshape_dims)));

  std::optional<Value> a3_transpose_perm = getConstTensor<int32_t>(
      rewriter, op, /*vec=*/{0, 1, 3, 2, 4, 5}, /*shape=*/{6});

  if (!a3_transpose_perm) return std::nullopt;

  auto a3_transpose_a2_op = CreateOpAndInfer<tosa::TransposeOp>(
      rewriter, op->getLoc(), a_reshape_output_type,
      a2_reshape_a_op.getResult(), a3_transpose_perm.value());

  SmallVector<int64_t, 4> a3_reshape_dims;
  a3_reshape_dims.push_back(input_shape[0]);
  a3_reshape_dims.push_back(input_shape[1] * block_size[0]);
  a3_reshape_dims.push_back(input_shape[2] * block_size[1]);
  a3_reshape_dims.push_back(input_shape[3] / (block_size[0] * block_size[1]));

  RankedTensorType a3_reshape_output_type =
      tensorflow::GetTypeFromTFTensorShape(a3_reshape_dims,
                                           output_type.getElementType());
  return CreateOpAndInfer<tosa::ReshapeOp>(
             rewriter, op->getLoc(), a3_reshape_output_type,
             a3_transpose_a2_op.getResult(),
             rewriter.getDenseI64ArrayAttr(
                 tensorflow::ConvertMlirShapeToTF(a3_reshape_dims)))
      .getResult();
}

// Lowers Split to a sequence of TOSA ops.
std::optional<SmallVector<Value>> convertSplitOp(
    PatternRewriter& rewriter, Operation* op, Value result_value,
    Value input_value, int32_t num_split, int32_t axis) {
  // This lowering creates num_split slice ops and ties them together
  // with IdentityN to get from an array of Operations to a single Operation
  // with a list of result tensors.
  RankedTensorType result_type =
      result_value.getType().dyn_cast<RankedTensorType>();
  // Not a ranked tensor output
  if (!result_type) {
    (void)rewriter.notifyMatchFailure(op, "output type not ranked tensor");
    return std::nullopt;
  }

  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type) {
    (void)rewriter.notifyMatchFailure(op, "input type not ranked tensor");
    return std::nullopt;
  }

  Type etype = input_type.getElementType();

  if (axis < 0) axis += input_type.getRank();
  assert(num_split > 0);
  assert(axis >= 0 && axis < input_type.getRank());

  Value slice_value = input_value;
  bool is_dyn_split = input_type.isDynamicDim(axis);
  if (is_dyn_split) {
    SmallVector<int64_t> new_shape;
    for (int i = 0, s = input_type.getRank(); i < s; i++) {
      if (i != axis) {
        new_shape.push_back(input_type.getDimSize(i));
        continue;
      }

      new_shape.push_back(num_split);
      new_shape.push_back(-1);
    }
    slice_value = CreateOpAndInfer<tosa::ReshapeOp>(
        rewriter, op->getLoc(),
        tensorflow::GetTypeFromTFTensorShape(new_shape, etype), input_value,
        rewriter.getDenseI64ArrayAttr(
            tensorflow::ConvertMlirShapeToTF(new_shape)));
  }

  RankedTensorType slice_type = slice_value.getType().cast<RankedTensorType>();
  assert((slice_type.getDimSize(axis) % num_split) == 0);

  // Each slice has a different beginning point.
  // The slice size is actually the same each op.
  SmallVector<int64_t> begin_vals, size_vals;
  for (int j = 0, s = slice_type.getRank(); j < s; j++) {
    begin_vals.push_back(0);
    int64_t dim = slice_type.getDimSize(j);
    size_vals.push_back(ShapedType::isDynamic(dim) ? tensorflow::kTFDynamicSize
                                                   : dim);
  }
  size_vals[axis] = size_vals[axis] / num_split;

  SmallVector<Value> results_vec;
  for (int i = 0; i < num_split; i++) {
    begin_vals[axis] = i * size_vals[axis];
    DenseI64ArrayAttr begin = rewriter.getDenseI64ArrayAttr(begin_vals);
    DenseI64ArrayAttr size = rewriter.getDenseI64ArrayAttr(
        tensorflow::ConvertMlirShapeToTF(size_vals));

    Value result = CreateOpAndInfer<tosa::SliceOp>(
        rewriter, op->getLoc(),
        tensorflow::GetTypeFromTFTensorShape(size_vals,
                                             result_type.getElementType()),
        slice_value, begin, size);

    if (is_dyn_split) {
      SmallVector<int64_t> out_reshape_shape;
      for (int i = 0, s = size_vals.size(); i < s; i++)
        if (i != axis) out_reshape_shape.push_back(size_vals[i]);

      result =
          CreateOpAndInfer<tosa::ReshapeOp>(
              rewriter, op->getLoc(),
              tensorflow::GetTypeFromTFTensorShape(out_reshape_shape, etype),
              result,
              rewriter.getDenseI64ArrayAttr(
                  tensorflow::ConvertMlirShapeToTF(out_reshape_shape)))
              .getResult();
    }

    results_vec.push_back(result);
  }

  return results_vec;
}

// Lowers SplitV to a sequence of TOSA ops.
std::optional<SmallVector<Value>> convertSplitVOp(
    PatternRewriter& rewriter, Operation* op, Value result_value,
    Value input_value, SmallVectorImpl<int32_t>& size_split, int32_t axis) {
  // This lowering creates num_split slice ops and ties them together
  // with IdentityN to get from an array of Operations to a single Operation
  // with a list of result tensors.
  RankedTensorType result_type =
      result_value.getType().dyn_cast<RankedTensorType>();
  // Not a ranked tensor output
  if (!result_type) {
    (void)rewriter.notifyMatchFailure(op, "output type not ranked tensor");
    return std::nullopt;
  }

  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type) {
    (void)rewriter.notifyMatchFailure(op, "input type not ranked tensor");
    return std::nullopt;
  }

  auto input_shape = input_type.getShape();

  SmallVector<Value> results_vec;

  if ((axis >= 0 && axis >= input_shape.size()) ||
      (axis < 0 && axis < -input_shape.size())) {
    (void)rewriter.notifyMatchFailure(op, "invalid axis value");
    return std::nullopt;
  }
  axis = adjust_axis(axis, input_shape.size());
  int32_t size_split_sum = 0;
  for (int i = 0; i < size_split.size(); i++) {
    size_split_sum += size_split[i];
  }

  // The split sizes must sum up to the size of the axis being split
  assert(size_split_sum == input_shape[axis]);

  int32_t curr_split_start = 0;
  for (int i = 0; i < size_split.size(); i++) {
    // Each slice has a different beginning point.
    // The slice size is different for each op.
    SmallVector<int64_t> begin_vals, size_vals;

    for (int j = 0; j < input_shape.size(); j++) {
      if (j == axis) {
        begin_vals.push_back(curr_split_start);
        size_vals.push_back(size_split[i]);
      } else {
        begin_vals.push_back(0);
        size_vals.push_back(input_shape[j]);
      }
    }

    DenseI64ArrayAttr begin = rewriter.getDenseI64ArrayAttr(begin_vals);
    DenseI64ArrayAttr size = rewriter.getDenseI64ArrayAttr(
        tensorflow::ConvertMlirShapeToTF(size_vals));

    auto slice_op = CreateOpAndInfer<tosa::SliceOp>(
        rewriter, op->getLoc(),
        tensorflow::GetTypeFromTFTensorShape(size_vals,
                                             result_type.getElementType()),
        input_value, begin, size);

    results_vec.push_back(slice_op.getResult());

    // Next start position
    curr_split_start += size_split[i];
  }

  return results_vec;
}

// Helper function to reverse negative striding. Only checks for -1 as that is
// the only legal negative stride.
static Value reverseNegativeStride(PatternRewriter& rewriter, Operation* op,
                                   Value input, ArrayRef<int32_t> strides) {
  for (const auto& it : llvm::enumerate(strides)) {
    auto axis = it.index();
    auto stride = it.value();
    if (stride != -1) continue;

    input = CreateOpAndInfer<tosa::ReverseOp>(rewriter, op->getLoc(),
                                              input.getType(), input,
                                              rewriter.getI64IntegerAttr(axis))
                .getResult();
  }

  return input;
}

// Lowers StridedSlice to a sequence of TOSA ops.
std::optional<Value> convertStridedSliceOp(
    PatternRewriter& rewriter, Operation* op, Value result_value,
    Value input_value, Value begin_value, Value end_value, Value strides_value,
    int32_t begin_mask, int32_t end_mask, int32_t ellipsis_mask,
    int32_t new_axis_mask, int32_t shrink_axis_mask) {
  // The mask arguments are bitmasks where bit [i] applies to
  // dimension [i] of the input tensor.
  //
  // The rough algorithm for lowering strided slice is as follows:
  //
  // 0. Process begin/end masks, since they are basically syntactic sugar
  // on top of the begin_value/end_value arrays
  //
  // 1. Slice1: Ignoring stride, slice the interesting range from the input
  // tensor
  //
  // 2. Reshape2: Reshape the tensor from (1) such that each dimension with
  // stride is split into two dimensions of size_i/stride_i, stride_i. A naive
  // implementation doubles the input tensor rank, but only dimensions being
  // strided actually need to be doubled.
  //
  // 3. Slice3: Slice the tensor from (2) such that we select index [0] from
  // each of the stride_i dimensions in (2)
  //
  // 4. Reshape4: Reshape the tensor to eliminate the stride_i dimensions, add
  // any dimensions in new_axis_mask and remove any dimensions in the
  // shrink_axis_mask

  // Limitations:
  // * This implementation only supports ellipsis_mask=0 for now
  auto input_type = input_value.getType().dyn_cast<RankedTensorType>();
  ShapedType result_type = result_value.getType().cast<ShapedType>();

  if (ellipsis_mask != 0) {
    (void)rewriter.notifyMatchFailure(op, "ellipses mask not supported yet");
    return std::nullopt;
  }

  if (!input_type) {
    (void)rewriter.notifyMatchFailure(op, "input type has unknown rank");
    return std::nullopt;
  }

  int32_t input_rank = input_type.getRank();
  Type element_type = input_type.getElementType();

  // Conditionally extract begin/end values if requied.
  SmallVector<int32_t> begin, end, strides;
  if (failed(getVectorFromValue32(strides_value, strides))) {
    (void)rewriter.notifyMatchFailure(op, "strides isn't a constant");
    return std::nullopt;
  }

  // Current configuration does not support negative strides greater than 1.
  // Bail out for now (fix if this proves to be legal).
  for (auto stride : strides)
    if (stride < -1) return std::nullopt;

  bool all_strides_one = true;
  int32_t strides_size = strides.size();
  for (auto stride : strides) all_strides_one &= abs(stride) == 1;


  // If all of the masks are set we can just bypass the entire thing.
  const int32_t all_masks_one = (1 << strides_size) - 1;

  if (failed(getVectorFromValue32(begin_value, begin)) &&
      begin_mask != all_masks_one) {
    (void)rewriter.notifyMatchFailure(op, "begin isn't a constant");
    return std::nullopt;
  }

  if (end_mask != all_masks_one &&
      failed(getVectorFromValue32(end_value, end))) {
    (void)rewriter.notifyMatchFailure(op, "end isn't a constant");
    return std::nullopt;
  }

  if (llvm::any_of(strides, [](auto i) { return i < -1; })) {
    (void)rewriter.notifyMatchFailure(op, "stride < -1 unsupported");
    return std::nullopt;
  }

  // Set begin mask values if possible.
  for (const auto& val : llvm::enumerate(begin))
    begin_mask |= (val.value() == 0) << val.index();

  // If all begin/end masks are set and striding is one we can just return
  // the matrix with reversed dims (for negative strides).
  if (all_strides_one && begin_mask == all_masks_one &&
      end_mask == all_masks_one) {
    return reverseNegativeStride(rewriter, op, input_value, strides);
  }

  // Set the bits true for the remaining dimensions.
  int32_t new_mask_bits = (1 << input_rank) - all_masks_one - 1;
  begin_mask |= new_mask_bits;
  end_mask |= new_mask_bits;

  // Negative values are exclusive from the opposite end while TOSA is
  // inclusive, we offset to adjust for this.
  for (auto& b : begin)
    if (b < 0) b = b - 1;
  for (auto& e : end)
    if (e < 0) e = e - 1;

  // Fill the remaining stride and begin/end with default values.
  strides.resize(input_rank, 1);
  begin.resize(input_rank, 0);
  end.resize(input_rank, -1);

  // Set masking-bit overrides.
  for (int i = 0; i < input_rank; ++i) {
    if (begin_mask & (1 << i)) begin[i] = 0;
    if (end_mask & (1 << i)) end[i] = -1;
  }

  // If we know the static end we can adjust by it.
  for (int i = 0; i < input_rank; ++i) {
    if (input_type.isDynamicDim(i)) continue;
    if (begin[i] < 0) begin[i] += input_type.getDimSize(i) + 1;
    if (end[i] < 0) end[i] += input_type.getDimSize(i) + 1;
  }

  // Perform some final validation on the begin/end values.
  for (int i = 0; i < input_rank; ++i) {
    if (begin[i] < 0 && input_type.isDynamicDim(i)) {
      (void)rewriter.notifyMatchFailure(
          op, "begin offset is negative on dynamic size");
      return std::nullopt;
    }

    if (end[i] < -1 && input_type.isDynamicDim(i)) {
      (void)rewriter.notifyMatchFailure(
          op, "end is exclusive of final entry on dynamic size");
      return std::nullopt;
    }
  }

  // Step 0: Process the begin/end masks and build the begin/sizes for the
  // first slice
  SmallVector<int64_t> a1_begin(input_rank), a1_size(input_rank);
  for (int i = 0; i < input_rank; ++i) {
    // Wrap around index if begin and end is negative
    a1_begin[i] = begin[i];

    if (end[i] == -1 && input_type.isDynamicDim(i)) {
      // Slice using -1 as TOSA's sentinal value.
      a1_size[i] = -1;
    } else if (end[i] < 0 && input_type.isDynamicDim(i)) {
      // Other dynamic cases cannot be handled.
      (void)rewriter.notifyMatchFailure(
          op, "input dim is dynamic and slice end depends on the length");
      return std::nullopt;
    } else {
      a1_size[i] = end[i] - a1_begin[i];
    }

    // Shrink axis mask means we know the size and stride are 1.
    if (shrink_axis_mask & (1 << i)) {
      a1_size[i] = 1;
      strides[i] = 1;
    }
  }

  // Step 1: Slice the input array
  auto a1_slice_op = CreateOpAndInfer<tosa::SliceOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(a1_size, element_type), input_value,
      rewriter.getDenseI64ArrayAttr(a1_begin),
      rewriter.getDenseI64ArrayAttr(tensorflow::ConvertMlirShapeToTF(a1_size)));

  // If unary striding is used we can reverse, reshape, and return the result.
  if (all_strides_one) {
    auto reversed =
        reverseNegativeStride(rewriter, op, a1_slice_op.getResult(), strides);
    auto shape = reversed.getType().cast<RankedTensorType>().getShape();

    SmallVector<int64_t> new_shape;
    for (int i = 0; i < input_rank; ++i) {
      if (!(shrink_axis_mask & (1 << i))) {
        if (new_axis_mask & (1 << i)) new_shape.push_back(1);
        new_shape.push_back((shape[i]));
      }
    }

    return CreateOpAndInfer<tosa::ReshapeOp>(
               rewriter, op->getLoc(), result_type, reversed,
               rewriter.getDenseI64ArrayAttr(
                   tensorflow::ConvertMlirShapeToTF(new_shape)))
        .getResult();
  }

  // Step 2: reshape the sliced array
  SmallVector<int64_t> a2_shape(input_rank * 2);
  for (int i = 0; i < input_rank; ++i) {
    a2_shape[i * 2 + 0] = a1_size[i] == -1 ? -1 : a1_size[i] / abs(strides[i]);
    a2_shape[i * 2 + 1] = abs(strides[i]);
  }

  auto a2_reshape_op = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(a2_shape, element_type),
      a1_slice_op.getResult(),
      rewriter.getDenseI64ArrayAttr(
          tensorflow::ConvertMlirShapeToTF(a2_shape)));

  // Step 3: take a slice along the strides
  SmallVector<int64_t> a3_begin(input_rank * 2), a3_size(input_rank * 2);
  for (int i = 0; i < input_rank; ++i) {
    a3_begin[i * 2 + 0] = 0;
    a3_begin[i * 2 + 1] = 0;

    if (shrink_axis_mask & (1 << i)) {
      a3_size[i * 2 + 0] = 1;
    } else {
      a3_size[i * 2 + 0] =
          (a1_size[i] == -1) ? -1 : (a1_size[i] / abs(strides[i]));
    }
    a3_size[i * 2 + 1] = 1;
  }

  auto a3_slice_op = CreateOpAndInfer<tosa::SliceOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(a3_size, element_type),
      a2_reshape_op.getResult(), rewriter.getDenseI64ArrayAttr(a3_begin),
      rewriter.getDenseI64ArrayAttr(tensorflow::ConvertMlirShapeToTF(a3_size)));

  // Step 4: reshape the now-strided tensor
  SmallVector<int64_t> a4_shape;
  for (int i = 0; i < input_rank; ++i) {
    if (!(shrink_axis_mask & (1 << i))) {
      if (new_axis_mask & (1 << i)) a4_shape.push_back(1);
      a4_shape.push_back(
          ((a1_size[i] == -1) ? -1 : (a1_size[i] / abs(strides[i]))));
    }
  }

  auto a4_reshape_op =
      CreateOpAndInfer<tosa::ReshapeOp>(
          rewriter, op->getLoc(), result_type, a3_slice_op.getResult(),
          rewriter.getDenseI64ArrayAttr(
              tensorflow::ConvertMlirShapeToTF(a4_shape)))
          .getResult();

  return reverseNegativeStride(rewriter, op, a4_reshape_op, strides);
}

// Lowers FloorDiv to a sequence of TOSA operators.
std::optional<Value> convertFloorDivOp(PatternRewriter& rewriter, Operation* op,
                                       Value result_value, Value lhs_value,
                                       Value rhs_value) {
  // FloorDiv lowering:
  // floor(1/rhs * lhs)
  //
  // a1 = reciprocal(rhs);
  // a2 = mul(lhs, a1);
  // a3 = floor(a2);
  // return a3;
  ShapedType output_type = result_value.getType().dyn_cast<ShapedType>();
  // Not a shaped tensor output
  if (!output_type) return std::nullopt;

  Type element_type = output_type.getElementType();

  if (element_type.isa<IntegerType>()) {
    return CreateOpAndInfer<tosa::DivOp>(rewriter, op->getLoc(), output_type,
                                         lhs_value, rhs_value)
        .getResult();
  }

  auto a1_reciprocal_rhs_op = CreateOpAndInfer<tosa::ReciprocalOp>(
      rewriter, op->getLoc(), rhs_value.getType(), rhs_value);
  auto a2_mul_lhs_a1_op = CreateOpAndInfer<tosa::MulOp>(
      rewriter, op->getLoc(), output_type, lhs_value,
      a1_reciprocal_rhs_op.getResult(), 0);
  return CreateOpAndInfer<tosa::FloorOp>(rewriter, op->getLoc(), output_type,
                                         a2_mul_lhs_a1_op.getResult())
      .getResult();
}

// Lowers FloorMod to a sequence of TOSA operators.
std::optional<Value> convertFloorModOp(PatternRewriter& rewriter, Operation* op,
                                       Value result_value, Value lhs_value,
                                       Value rhs_value) {
  // FloorMod lowering:
  // (1/rhs * lhs) - floor(1/rhs * lhs)
  // a1 = reciprocal(rhs);
  // a2 = mul(lhs, a1);
  // a3 = floor(a2);
  // a4 = sub(a2, a3);
  // return a4;

  RankedTensorType output_type =
      result_value.getType().dyn_cast<RankedTensorType>();
  // Not a ranked tensor output
  if (!output_type) return std::nullopt;

  auto a1_reciprocal_rhs_op = CreateOpAndInfer<tosa::ReciprocalOp>(
      rewriter, op->getLoc(), rhs_value.getType(), rhs_value);
  auto a2_mul_lhs_a1_op = CreateOpAndInfer<tosa::MulOp>(
      rewriter, op->getLoc(), output_type, lhs_value,
      a1_reciprocal_rhs_op.getResult(), 0);
  auto a3_floor_a2_op = CreateOpAndInfer<tosa::FloorOp>(
      rewriter, op->getLoc(), output_type, a2_mul_lhs_a1_op.getResult());
  return CreateOpAndInfer<tosa::SubOp>(rewriter, op->getLoc(), output_type,
                                       a2_mul_lhs_a1_op.getResult(),
                                       a3_floor_a2_op.getResult())
      .getResult();
}

// Lowers FusedActivation to a sequence of TOSA ops.
std::optional<Value> convertFusedActivation(PatternRewriter& rewriter,
                                            Operation* op, Value input_value,
                                            StringAttr fused_activation_fn) {
  ShapedType input_type = input_value.getType().dyn_cast<ShapedType>();
  if (!input_type) return std::nullopt;

  bool input_is_qtype =
      input_type.getElementType().isa<mlir::quant::UniformQuantizedType>();

  if (input_is_qtype) {
    // We can always make output/input tensor's scale/zp always be the same
    // when legalizing fused_activation_function, as it's generated during
    // legalization.
    auto input_qtype =
        input_type.getElementType().cast<mlir::quant::UniformQuantizedType>();

    if (fused_activation_fn.getValue() == "NONE") {
      return input_value;
    } else if (fused_activation_fn.getValue() == "RELU") {
      int32_t quantized_0 = input_qtype.getZeroPoint();
      int32_t quantized_max = input_qtype.getStorageTypeMax();

      auto clamp_op = CreateOpAndInfer<tosa::ClampOp>(
          rewriter, op->getLoc(), input_type, input_value,
          rewriter.getI64IntegerAttr(quantized_0),
          rewriter.getI64IntegerAttr(quantized_max),
          rewriter.getF32FloatAttr(0), rewriter.getF32FloatAttr(0));

      return clamp_op.getResult();
    } else if (fused_activation_fn.getValue() == "RELU6") {
      int64_t quantized_0 = input_qtype.getZeroPoint();
      int64_t quantized_6 = std::llround((6.0f / input_qtype.getScale()) +
                                         input_qtype.getZeroPoint());
      int64_t quantized_min = input_qtype.getStorageTypeMin();
      int64_t quantized_max = input_qtype.getStorageTypeMax();

      int64_t clamp_min = std::max(quantized_0, quantized_min);
      int64_t clamp_max = std::min(quantized_6, quantized_max);

      auto clamp_op = CreateOpAndInfer<tosa::ClampOp>(
          rewriter, op->getLoc(), input_type, input_value,
          rewriter.getI64IntegerAttr(clamp_min),
          rewriter.getI64IntegerAttr(clamp_max), rewriter.getF32FloatAttr(0),
          rewriter.getF32FloatAttr(0));

      return clamp_op.getResult();
    } else if (fused_activation_fn.getValue() == "RELU_N1_TO_1") {
      int64_t quantized_max = input_qtype.getStorageTypeMax();
      int64_t quantized_min = input_qtype.getStorageTypeMin();
      int64_t quantized_n1 = std::llround((-1.0f / input_qtype.getScale()) +
                                          input_qtype.getZeroPoint());
      int64_t quantized_1 = std::llround((1.0f / input_qtype.getScale()) +
                                         input_qtype.getZeroPoint());

      int64_t clamp_min = std::max(quantized_n1, quantized_min);
      int64_t clamp_max = std::min(quantized_1, quantized_max);

      auto clamp_op = CreateOpAndInfer<tosa::ClampOp>(
          rewriter, op->getLoc(), input_type, input_value,
          rewriter.getI64IntegerAttr(clamp_min),
          rewriter.getI64IntegerAttr(clamp_max), rewriter.getF32FloatAttr(0),
          rewriter.getF32FloatAttr(0));

      return clamp_op.getResult();
    } else {
      op->emitWarning("convertFusedActivation: Not implemented yet");
      return std::nullopt;
    }
  } else {
    if (fused_activation_fn.getValue() == "NONE") {
      return input_value;
    } else {
      // For non-quantized type, only support F32.
      if (!input_type.getElementType().isF32()) {
        (void)rewriter.notifyMatchFailure(op, "only support F32");
        return std::nullopt;
      }

      if (fused_activation_fn.getValue() == "RELU") {
        return CreateOpAndInfer<tosa::ClampOp>(
                   rewriter, op->getLoc(), input_type, input_value,
                   rewriter.getI64IntegerAttr(0),
                   rewriter.getI64IntegerAttr(
                       std::numeric_limits<int32_t>::max()),
                   rewriter.getF32FloatAttr(0.0f),
                   rewriter.getF32FloatAttr(std::numeric_limits<float>::max()))
            .getResult();
      } else if (fused_activation_fn.getValue() == "RELU6") {
        return CreateOpAndInfer<tosa::ClampOp>(
                   rewriter, op->getLoc(), input_type, input_value,
                   rewriter.getI64IntegerAttr(0), rewriter.getI64IntegerAttr(6),
                   rewriter.getF32FloatAttr(0.0f),
                   rewriter.getF32FloatAttr(6.0f))
            .getResult();
      } else if (fused_activation_fn.getValue() == "RELU_N1_TO_1") {
        return CreateOpAndInfer<tosa::ClampOp>(rewriter, op->getLoc(),
                                               input_type, input_value,
                                               rewriter.getI64IntegerAttr(-1),
                                               rewriter.getI64IntegerAttr(1),
                                               rewriter.getF32FloatAttr(-1.0),
                                               rewriter.getF32FloatAttr(1.0))
            .getResult();
      } else if (fused_activation_fn.getValue() == "TANH") {
        return CreateOpAndInfer<tosa::TanhOp>(rewriter, op->getLoc(),
                                              input_type, input_value)
            .getResult();
      } else {
        // Unsupported activation type. Bail out.
        return std::nullopt;
      }
    }
  }

  return std::nullopt;
}

template <typename T>
static Value convertGenericReduceOp(PatternRewriter& rewriter, Operation* op,
                                    Value input, Type input_etype,
                                    Type reduce_etype,
                                    ArrayRef<int64_t> input_shape,
                                    ArrayRef<int64_t> axes) {
  Location loc = op->getLoc();
  int64_t input_rank = input_shape.size();
  llvm::SmallVector<int32_t> perms;
  llvm::SmallVector<int64_t> reshape_shape;
  perms.reserve(input_rank);
  reshape_shape.resize(2, 1);

  // First insert all non-reduction axes.
  for (int i = 0; i < input_rank; i++) {
    auto it = std::find(axes.begin(), axes.end(), i);
    if (it == axes.end()) {
      perms.push_back(i);
      reshape_shape[0] *= input_shape[i];
    }
  }

  // Then insert all reduction matrices.
  for (auto axis : axes) {
    perms.push_back(axis);
    reshape_shape[1] *= input_shape[axis];
  }

  Value perms_value =
      getConstTensor<int32_t>(rewriter, op, perms,
                              {static_cast<int64_t>(perms.size())})
          .value();

  auto transpose_op = CreateOpAndInfer<tosa::TransposeOp>(
      rewriter, loc, UnrankedTensorType::get(input_etype), input, perms_value);

  auto reshape_op = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, loc,
      tensorflow::GetTypeFromTFTensorShape(reshape_shape, input_etype),
      transpose_op,
      rewriter.getDenseI64ArrayAttr(
          tensorflow::ConvertMlirShapeToTF(reshape_shape)));

  return CreateOpAndInfer<T>(rewriter, loc,
                             UnrankedTensorType::get(reduce_etype), reshape_op,
                             rewriter.getI64IntegerAttr(1));
}

// Common function for lowering reduce operations to TOSA ops.
template <typename T>
std::optional<Value> convertReduceOpCommon(
    PatternRewriter& rewriter, Operation* op, RankedTensorType output_type,
    Value input_value, ElementsAttr axes_elems, Type reduce_element_type,
    bool is_quantized, double input_scale, int64_t input_zp,
    double output_scale, int64_t output_zp) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type) return std::nullopt;

  ArrayRef<int64_t> input_shape = input_type.getShape();
  ArrayRef<int64_t> output_shape = output_type.getShape();
  auto input_rank = input_shape.size();
  Location loc = op->getLoc();

  if (axes_elems.getNumElements() == 0) {
    // No axes means return the original tensor.
    auto identity_op = CreateOpAndInfer<tosa::IdentityOp>(
        rewriter, loc, output_type, input_value);
    return identity_op.getResult();
  }

  // Handle negative axis and guarantee in increasing order.
  llvm::SmallVector<int64_t> axes;
  for (auto axis : axes_elems.getValues<IntegerAttr>()) {
    auto axis_val = axis.getInt();
    if (axis_val < 0) axis_val += input_rank;
    if (axis_val < 0 || axis_val >= input_rank) {
      (void)rewriter.notifyMatchFailure(
          op, "axis values not within range of input shape");

      return std::nullopt;
    }
    axes.push_back(axis_val);
  }

  // Reduction operations are limited to 4D tensors, restructure to obey this
  // restriction. We transpose all reduction axis to the RHS, then reshape
  // such that all reduction and non-reduction values are grouped. This forms a
  // 2D matrix in all cases.
  Value val = input_value;
  if (input_rank > 4) {
    val = convertGenericReduceOp<T>(rewriter, op, val,
                                    input_type.getElementType(),
                                    reduce_element_type, input_shape, axes);
  } else {
    // Reduce along each axis
    SmallVector<int64_t> shape_vec(input_shape.begin(), input_shape.end());

    if (is_quantized) {
      val = buildRescaleToInt32(rewriter, op, val, input_scale, input_zp);
    }

    for (auto axis_val : axes) {
      auto axis_attr = rewriter.getI64IntegerAttr(axis_val);

      shape_vec[axis_val] = 1;
      RankedTensorType reduce_type =
          tensorflow::GetTypeFromTFTensorShape(shape_vec, reduce_element_type);

      auto reduce_op = CreateOpAndInfer<T>(rewriter, op->getLoc(), reduce_type,
                                           val, axis_attr);

      val = reduce_op.getResult();
    }
  }

  if (is_quantized) {
    UnrankedTensorType output_rescale_type =
        UnrankedTensorType::get(output_type.getElementType());
    val = buildRescale(rewriter, op, output_rescale_type, val, output_scale, 0,
                       output_zp, false, true);
  }

  // Squeeze out the reduced axes.
  auto reshape_op = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(), output_type, val,
      rewriter.getDenseI64ArrayAttr(
          tensorflow::ConvertMlirShapeToTF(output_shape)));
  return reshape_op.getResult();
}

// Lowers ReduceAll to a sequence of TOSA ops.
std::optional<Value> convertReduceAllOp(PatternRewriter& rewriter,
                                        Operation* op,
                                        RankedTensorType output_type,
                                        Value input_value,
                                        ElementsAttr axes_elems) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type) return std::nullopt;

  return convertReduceOpCommon<tosa::ReduceAllOp>(
      rewriter, op, output_type, input_value, axes_elems,
      output_type.getElementType(), false, 1.0f, 0, 1.0f, 0);
}

// Lowers ReduceAny to a sequence of TOSA ops.
std::optional<Value> convertReduceAnyOp(PatternRewriter& rewriter,
                                        Operation* op,
                                        RankedTensorType output_type,
                                        Value input_value,
                                        ElementsAttr axes_elems) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type) return std::nullopt;

  return convertReduceOpCommon<tosa::ReduceAnyOp>(
      rewriter, op, output_type, input_value, axes_elems,
      output_type.getElementType(), false, 1.0f, 0, 1.0f, 0);
}

// Lowers ReduceMin to a sequence of TOSA ops.
std::optional<Value> convertReduceMinOp(PatternRewriter& rewriter,
                                        Operation* op,
                                        RankedTensorType output_type,
                                        Value input_value,
                                        ElementsAttr axes_elems) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type) return std::nullopt;

  return convertReduceOpCommon<tosa::ReduceMinOp>(
      rewriter, op, output_type, input_value, axes_elems,
      output_type.getElementType(), false, 1.0f, 0, 1.0f, 0);
}

// Lowers ReduceMax to a sequence of TOSA ops.
std::optional<Value> convertReduceMaxOp(PatternRewriter& rewriter,
                                        Operation* op,
                                        RankedTensorType output_type,
                                        Value input_value,
                                        ElementsAttr axes_elems) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type) return std::nullopt;

  return convertReduceOpCommon<tosa::ReduceMaxOp>(
      rewriter, op, output_type, input_value, axes_elems,
      output_type.getElementType(), false, 1.0f, 0, 1.0f, 0);
}

// Lowers ReduceProd to a sequence of TOSA ops.
std::optional<Value> convertReduceProdOp(PatternRewriter& rewriter,
                                         Operation* op,
                                         RankedTensorType output_type,
                                         Value input_value,
                                         ElementsAttr axes_elems) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type) return std::nullopt;

  bool input_is_qtype =
      input_type.getElementType().isa<mlir::quant::UniformQuantizedType>();
  bool output_is_qtype =
      output_type.getElementType().isa<mlir::quant::UniformQuantizedType>();

  if (input_is_qtype || output_is_qtype) {
    (void)rewriter.notifyMatchFailure(
        op, "input/output tensor should be all floating-point");
    return std::nullopt;
  }

  return convertReduceOpCommon<tosa::ReduceProdOp>(
      rewriter, op, output_type, input_value, axes_elems,
      output_type.getElementType(), false, 1.0f, 0, 1.0f, 0);
}

// Lowers ReduceSum to a sequence of TOSA ops.
std::optional<Value> convertReduceSumOp(PatternRewriter& rewriter,
                                        Operation* op,
                                        RankedTensorType output_type,
                                        Value input_value,
                                        ElementsAttr axes_elems) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type) return std::nullopt;

  bool input_is_qtype =
      input_type.getElementType().isa<mlir::quant::UniformQuantizedType>();
  bool output_is_qtype =
      output_type.getElementType().isa<mlir::quant::UniformQuantizedType>();

  if (input_is_qtype != output_is_qtype) {
    (void)rewriter.notifyMatchFailure(
        op,
        "input/output tensor should be all quantized or all floating-point");
    return std::nullopt;
  }

  double input_scale = 1.0f;
  double output_scale = 1.0f;
  int64_t input_zp = 0;
  int64_t output_zp = 0;
  Type reduce_element_type = input_type.getElementType();

  if (input_is_qtype) {
    auto input_qtype =
        input_type.getElementType().cast<mlir::quant::UniformQuantizedType>();
    auto output_qtype =
        output_type.getElementType().cast<mlir::quant::UniformQuantizedType>();

    int32_t input_shift = 20;

    input_scale =
        static_cast<double>(1 << input_shift) * input_qtype.getScale();
    output_scale =
        1.0 / (output_qtype.getScale() * static_cast<double>(1 << input_shift));

    input_zp = input_qtype.getZeroPoint();
    output_zp = output_qtype.getZeroPoint();
    reduce_element_type = rewriter.getI32Type();
  }

  return convertReduceOpCommon<tosa::ReduceSumOp>(
      rewriter, op, output_type, input_value, axes_elems, reduce_element_type,
      input_is_qtype, input_scale, input_zp, output_scale, output_zp);
}

// Lowers ReduceMean to a sequence of TOSA ops.
std::optional<Value> convertReduceMeanOp(PatternRewriter& rewriter,
                                         Operation* op,
                                         RankedTensorType output_type,
                                         Value input_value,
                                         ElementsAttr axes_elems) {
  // reduce_mean is lowered as followed:
  // op1 = reduce_sum(input)
  // op2 = mul(op1, 1.0 / num_elements_on_reduced_axis)

  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type) return std::nullopt;

  bool input_is_qtype =
      input_type.getElementType().isa<mlir::quant::UniformQuantizedType>();
  bool output_is_qtype =
      output_type.getElementType().isa<mlir::quant::UniformQuantizedType>();

  if (input_is_qtype != output_is_qtype) {
    (void)rewriter.notifyMatchFailure(
        op,
        "input/output tensor should be all quantized or all floating-point");
    return std::nullopt;
  }

  // Only supports float type mean() if it's non-quantized
  if (!input_is_qtype && !output_type.getElementType().isa<mlir::FloatType>()) {
    op->emitWarning("input unquantized type but output element not FloatType");
    return std::nullopt;
  }

  int64_t input_rank = input_type.getRank();
  int64_t num_elems_on_reduced_axis = 1;
  for (int i = 0; i < axes_elems.getNumElements(); i++) {
    int64_t axis_val = axes_elems.getValues<IntegerAttr>()[i].getInt();
    if (axis_val < 0) axis_val += input_rank;
    if (axis_val < 0 || axis_val >= input_rank) {
      (void)rewriter.notifyMatchFailure(
          op, "axis values not within range of input shape");

      return std::nullopt;
    }
    num_elems_on_reduced_axis *= input_type.getShape()[axis_val];
  }
  double div_scale = 1.0 / static_cast<double>(num_elems_on_reduced_axis);

  double input_scale = 1.0f;
  double output_scale = 1.0f;
  int64_t input_zp = 0;
  int64_t output_zp = 0;
  Type reduce_element_type = input_type.getElementType();

  if (input_is_qtype) {
    auto input_qtype =
        input_type.getElementType().cast<mlir::quant::UniformQuantizedType>();
    auto output_qtype =
        output_type.getElementType().cast<mlir::quant::UniformQuantizedType>();

    // Combine 'div_scale' as part of output rescale
    output_scale = div_scale * input_qtype.getScale() / output_qtype.getScale();

    input_zp = input_qtype.getZeroPoint();
    output_zp = output_qtype.getZeroPoint();
    reduce_element_type = rewriter.getI32Type();
  }

  auto val = convertReduceOpCommon<tosa::ReduceSumOp>(
      rewriter, op, output_type, input_value, axes_elems, reduce_element_type,
      input_is_qtype, input_scale, input_zp, output_scale, output_zp);

  if (!val.has_value()) return std::nullopt;

  if (!input_is_qtype) {
    Value div_const = getTosaConstTensorSingleF32(rewriter, op, div_scale);
    return CreateOpAndInfer<tosa::MulOp>(rewriter, op->getLoc(), output_type,
                                         val.value(), div_const, 0)
        .getResult();
  }

  return val;
}

// Lowers ResizeBilinear and ResizeNearestNeighbor to TOSA resize.
std::optional<Value> convertResizeOp(PatternRewriter& rewriter, Operation* op,
                                     RankedTensorType output_type,
                                     Value input_value, StringRef mode,
                                     bool align_corners,
                                     bool half_pixel_centers) {
  const bool is_bilinear = mode == "BILINEAR";
  const bool is_nearest = mode == "NEAREST_NEIGHBOR";
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type) return std::nullopt;

  if (input_type.getRank() != 4 || output_type.getRank() != 4) {
    (void)rewriter.notifyMatchFailure(op, "input/output must be rank 4");
    return std::nullopt;
  }

  bool input_is_qtype =
      input_type.getElementType().isa<mlir::quant::UniformQuantizedType>();
  bool output_is_qtype =
      output_type.getElementType().isa<mlir::quant::UniformQuantizedType>();

  if (input_is_qtype != output_is_qtype) {
    (void)rewriter.notifyMatchFailure(
        op,
        "input/output tensor should be all quantized or all floating-point");
    return std::nullopt;
  }

  if (!input_is_qtype) {
    if (!input_type.getElementType().isa<mlir::FloatType>()) {
      (void)rewriter.notifyMatchFailure(
          op, "only quantized or float types supported");
      return std::nullopt;
    }
  }

  auto input_shape = input_type.getShape();
  auto output_shape = output_type.getShape();

  if (input_type.isDynamicDim(1) || input_type.isDynamicDim(2)) {
    (void)rewriter.notifyMatchFailure(op, "resize dynamic input not supported");
    return std::nullopt;
  }

  if (output_type.isDynamicDim(1) || output_type.isDynamicDim(2)) {
    (void)rewriter.notifyMatchFailure(op,
                                      "resize dynamic output not supported");
    return std::nullopt;
  }

  size_t input_height = input_shape[1];
  size_t input_width = input_shape[2];
  size_t output_height = output_shape[1];
  size_t output_width = output_shape[2];

  // The ratio below is a non-zero positive value if this is a power-of-two
  // upscaling.
  int height_ratio = 0;
  if (output_height % input_height == 0) {
    int quotient = output_height / input_height;
    if (llvm::isPowerOf2_64(quotient)) {
      height_ratio = quotient;
    }
  }

  int width_ratio = 0;
  if (output_width % input_width == 0) {
    int quotient = output_width / input_width;
    if (llvm::isPowerOf2_64(quotient)) {
      width_ratio = quotient;
    }
  }

  // Align corners sets the scaling ratio to (OH - 1)/(IH - 1)
  // rather than OH / IH. Similarly for width.
  auto normalize = [&](int input, int output, int& n, int& d, int& offset,
                       int& border) {
    // Dimension is length 1, we are just sampling from one value.
    if (input == 1) {
      n = output;
      d = 1;
      offset = 0;
      border = output - 1;
      return;
    }

    // Apply if aligned and capable to be aligned.
    bool apply_aligned = align_corners && (output > 1);
    n = apply_aligned ? (output - 1) : output;
    d = apply_aligned ? (input - 1) : input;

    // Simplify the scalers, make sure they are even values.
    int gcd = std::gcd(n, d);
    n = 2 * n / gcd;
    d = 2 * d / gcd;

    // If half pixel centers we need to sample half a pixel inward.
    offset = half_pixel_centers ? d / 2 : 0;

    // If nearest neighbours we need to guarantee we round up.
    if (is_nearest && align_corners) {
      offset += n / 2;
    }

    if (is_bilinear && half_pixel_centers) {
      offset -= n / 2;
    }

    // We can compute this directly based on previous values.
    border = d * (output - 1) - n * (input - 1) + offset;
  };

  int scale_y_n, scale_y_d, offset_y, border_y;
  int scale_x_n, scale_x_d, offset_x, border_x;
  normalize(input_height, output_height, scale_y_n, scale_y_d, offset_y,
            border_y);
  normalize(input_width, output_width, scale_x_n, scale_x_d, offset_x,
            border_x);

  DenseI64ArrayAttr scale = rewriter.getDenseI64ArrayAttr(
      {scale_y_n, scale_y_d, scale_x_n, scale_x_d});
  DenseI64ArrayAttr offset =
      rewriter.getDenseI64ArrayAttr({offset_y, offset_x});
  DenseI64ArrayAttr border =
      rewriter.getDenseI64ArrayAttr({border_y, border_x});

  StringAttr resize_mode = rewriter.getStringAttr(mode);

  auto isInt16Range = [](int x) {
    return (x <= std::numeric_limits<int16_t>::max()) &&
           (x >= std::numeric_limits<int16_t>::min());
  };

  if (input_is_qtype) {
    // It isn't commonly seen these numbers aren't fit within 16 bits, and won't
    // match TFLite reference.
    if (!isInt16Range(scale_y_n) || !isInt16Range(scale_y_d) ||
        !isInt16Range(scale_x_n) || !isInt16Range(scale_x_d) ||
        !isInt16Range(offset_y) || !isInt16Range(offset_x) ||
        !isInt16Range(border_y) || !isInt16Range(border_x)) {
      return (void)rewriter.notifyMatchFailure(
                 op, "stride or offset out of 16 bit range"),
             std::nullopt;
    }

    // If quantized bilinear mode, need to lower to RESIZE + RESCALE pair.
    if (is_bilinear) {
      RankedTensorType output_acc_type;
      auto input_element_qtype =
          input_type.getElementType().cast<mlir::quant::UniformQuantizedType>();

      bool is_scale32;

      // TOSA RESIZE: 16 bit input -> 48 bit output, or 8 bit input -> 32 bit
      // output.
      if (input_element_qtype.getStorageTypeIntegralWidth() == 16) {
        is_scale32 = false;
        output_acc_type = tensorflow::GetTypeFromTFTensorShape(
            output_type.getShape(), rewriter.getIntegerType(48));
      } else if (input_element_qtype.getStorageTypeIntegralWidth() == 8) {
        is_scale32 = true;
        output_acc_type = tensorflow::GetTypeFromTFTensorShape(
            output_type.getShape(), rewriter.getI32Type());
      } else {
        (void)rewriter.notifyMatchFailure(
            op, "support 16-bit and 8-bit quantized input");
        return std::nullopt;
      }

      auto resize_op = CreateOpAndInfer<tosa::ResizeOp>(
          rewriter, op->getLoc(), output_acc_type, input_value, scale, offset,
          border, resize_mode);

#ifdef RESIZE_BILINEAR_LOWER_SYMMETRIC_ROUNDING
      // TFLite resize_bilinear always assume input and output tensors have
      // same scale That means we only need to arithmetic right shift with
      // (2 * shift)
      // TODO(suderman): Align TFLite rounding behavior
      // TFLite also uses symmetric rounding by doing 'x / (1 << 20)'
      // TOSA arithmetic right shift is doing standard rounding.
      // Right now it's legalized using GreaterEqualOp + SelectOp to conform
      // to TFLite reference. But this eventually should be fixed in TFLite
      // reference
      Value cst_zero = getTosaConstTensorSingleI32(rewriter, op, 0);
      Value cst_twenty = getTosaConstTensorSingleI32(rewriter, op, 20);

      auto ge_op = CreateOpAndInfer<tosa::GreaterEqualOp>(
          rewriter, op->getLoc(), output_bool_type, resize_op.getResult(),
          cst_zero);

      auto abs_op = CreateOpAndInfer<tosa::AbsOp>(
          rewriter, op->getLoc(), output_acc_type, resize_op.getResult());

      auto rshift_op = CreateOpAndInfer<tosa::ArithmeticRightShiftOp>(
          rewriter, op->getLoc(), output_acc_type, abs_op.getResult(),
          cst_twenty, true);

      auto negate_op = CreateOpAndInfer<tosa::NegateOp>(
          rewriter, op->getLoc(), output_acc_type, rshift_op.getResult());

      auto select_op = CreateOpAndInfer<tosa::SelectOp>(
          rewriter, op->getLoc(), output_acc_type, ge_op.getResult(),
          rshift_op.getResult(), negate_op.getResult());

      auto cast_op = CreateOpAndInfer<tosa::CastOp>(
          rewriter, op->getLoc(), output_type, select_op.getResult());

      return cast_op.getResult();
#else
      // This should be the expected lowering, but is +-1 within compared to
      // TFLite reference.
      return buildRescale(rewriter, op, output_type, resize_op.getResult(),
                          1.0 / (scale_y_n * scale_x_n), 0, 0, false,
                          is_scale32);
#endif

    } else if (is_nearest) {
      auto resize_op = CreateOpAndInfer<tosa::ResizeOp>(
          rewriter, op->getLoc(), output_type, input_value, scale, offset,
          border, resize_mode);
      return resize_op.getResult();
    } else {
      (void)rewriter.notifyMatchFailure(
          op, "only support BILINEAR or NEAREST_NEIGHBOR mode");
      return std::nullopt;
    }
  } else {
    auto resize_op = CreateOpAndInfer<tosa::ResizeOp>(
        rewriter, op->getLoc(), output_type, input_value, scale, offset, border,
        resize_mode);

    return resize_op.getResult();
  }
}

// Lowers Quantize to a sequence of TOSA quantization ops.
std::optional<Value> convertQuantizeOp(PatternRewriter& rewriter, Operation* op,
                                       ShapedType output_type,
                                       Value input_value, double scale,
                                       int64_t zeropoint) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type) return std::nullopt;

  auto output_element_type = output_type.getElementType();

  // output element type could only be quantized integer
  if (!output_element_type.isa<mlir::quant::QuantizedType>()) {
    (void)rewriter.notifyMatchFailure(
        op, "lowering quantizeOp but output element type not quantized");
    return std::nullopt;
  }

  ShapedType output_fp_type = output_type.clone(rewriter.getF32Type());

  Value zp_val =
      getTosaConstTensorSingleF32(rewriter, op, static_cast<float>(zeropoint));

  auto op1_mul_in = CreateOpAndInfer<tosa::MulOp>(
      rewriter, op->getLoc(), output_fp_type, input_value,
      getTosaConstTensorSingleF32(rewriter, op, static_cast<float>(scale)), 0);

  auto op2_add_op1 = CreateOpAndInfer<tosa::AddOp>(
      rewriter, op->getLoc(), output_fp_type, op1_mul_in.getResult(), zp_val);

  auto op3_cast_op2 = CreateOpAndInfer<tosa::CastOp>(
      rewriter, op->getLoc(), output_type, op2_add_op1.getResult());

  return op3_cast_op2.getResult();
}

// Lowers Dequantize to a sequence of TOSA dequantization ops.
std::optional<Value> convertDequantizeOp(PatternRewriter& rewriter,
                                         Operation* op, ShapedType output_type,
                                         Value input_value,
                                         ArrayRef<float> scale,
                                         ArrayRef<float> zeropoint,
                                         int64_t dim) {
  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type) return std::nullopt;

  // input element type could only be quantized integer
  if (!input_type.getElementType().isa<mlir::quant::QuantizedType>())
    return std::nullopt;

  std::optional<Value> zp_val;
  if (zeropoint.size() == 1) {
    zp_val = getTosaConstTensorSingleF32(rewriter, op,
                                         static_cast<float>(zeropoint[0]));
  } else {
    SmallVector<int64_t> shape;
    shape.resize(input_type.getRank(), 1);
    shape[dim] = zeropoint.size();
    zp_val = getConstTensor(rewriter, op, zeropoint, shape);
  }

  std::optional<Value> scale_val;
  if (scale.size() == 1) {
    scale_val =
        getTosaConstTensorSingleF32(rewriter, op, static_cast<float>(scale[0]));
  } else {
    SmallVector<int64_t> shape;
    shape.resize(input_type.getRank(), 1);
    shape[dim] = scale.size();
    scale_val = getConstTensor(rewriter, op, scale, shape);
  }

  if (!zp_val || !scale_val) return std::nullopt;

  auto op1_cast_in = CreateOpAndInfer<tosa::CastOp>(rewriter, op->getLoc(),
                                                    output_type, input_value);

  auto op2_sub_op1 =
      CreateOpAndInfer<tosa::SubOp>(rewriter, op->getLoc(), output_type,
                                    op1_cast_in.getResult(), zp_val.value());

  return CreateOpAndInfer<tosa::MulOp>(rewriter, op->getLoc(), output_type,
                                       op2_sub_op1.getResult(),
                                       scale_val.value(), 0)
      .getResult();
}

// Lowers FakeQuant to a sequence of TOSA quantization ops.
std::optional<Value> convertFakeQuantOp(PatternRewriter& rewriter,
                                        Operation* op, ShapedType output_type,
                                        Value input_value, double min,
                                        double max, int64_t num_bits,
                                        bool narrow_range) {
  // FakeQuant is lowered as follow:
  // op1 = quantize(input)
  // op2 = dequantize(op1)

  RankedTensorType input_type =
      input_value.getType().dyn_cast<RankedTensorType>();
  if (!input_type) return std::nullopt;

  // quantized as INT<num_bits>, where num_bits can only be 8, 16
  if (num_bits != 8 && num_bits != 16) {
    op->emitWarning("FakeQuantOp lowering handles only 8 and 16 for num_bits!");
    return std::nullopt;
  }

  // This code originates from
  // tensorflow/core/kernels/fake_quant_ops_functor.h.
  int32_t qmax = (1 << (num_bits)) - 1;
  int32_t qmin = narrow_range ? 1 : 0;

  float nudged_min, nudged_max, nudged_scale;
  tensorflow_nudge(min, max, qmin, qmax, &nudged_min, &nudged_max,
                   &nudged_scale);

  Value cst_min = getTosaConstTensorSingleF32(rewriter, op, nudged_min);
  Value cst_max = getTosaConstTensorSingleF32(rewriter, op, nudged_max);
  Value cst_scale = getTosaConstTensorSingleF32(rewriter, op, nudged_scale);
  Value cst_inv_scale =
      getTosaConstTensorSingleF32(rewriter, op, 1.0f / nudged_scale);
  Value cst_half = getTosaConstTensorSingleF32(rewriter, op, 0.5f);

  // This code originates from
  // tensorflow/core/kernels/fake_quant_ops_functor.h.
  auto op1_min_in = CreateOpAndInfer<tosa::MinimumOp>(
      rewriter, op->getLoc(), output_type, input_value, cst_max);

  auto op2_max_op1 = CreateOpAndInfer<tosa::MaximumOp>(
      rewriter, op->getLoc(), output_type, op1_min_in.getResult(), cst_min);

  auto op3_sub_op2 = CreateOpAndInfer<tosa::SubOp>(
      rewriter, op->getLoc(), output_type, op2_max_op1.getResult(), cst_min);

  auto op4_mul_op3 =
      CreateOpAndInfer<tosa::MulOp>(rewriter, op->getLoc(), output_type,
                                    op3_sub_op2.getResult(), cst_inv_scale, 0);

  auto op5_add_op4 = CreateOpAndInfer<tosa::AddOp>(
      rewriter, op->getLoc(), output_type, op4_mul_op3.getResult(), cst_half);

  auto op6_floor_op5 = CreateOpAndInfer<tosa::FloorOp>(
      rewriter, op->getLoc(), output_type, op5_add_op4.getResult());

  auto op7_mul_op6 =
      CreateOpAndInfer<tosa::MulOp>(rewriter, op->getLoc(), output_type,
                                    op6_floor_op5.getResult(), cst_scale, 0);

  return CreateOpAndInfer<tosa::AddOp>(rewriter, op->getLoc(), output_type,
                                       op7_mul_op6.getResult(), cst_min)
      .getResult();
}

std::optional<Value> convertMirrorPadCommon(PatternRewriter& rewriter,
                                            Operation* op,
                                            RankedTensorType output_type,
                                            Value input, Value pad,
                                            TFTFLMirrorPaddingType mode) {
  RankedTensorType input_type = input.getType().dyn_cast<RankedTensorType>();
  if (!input_type) {
    (void)rewriter.notifyMatchFailure(op, "input type isn't a ranked tensor");
    return std::nullopt;
  }

  if (mode != TFTFLMirrorPaddingType::SYMMETRIC &&
      mode != TFTFLMirrorPaddingType::REFLECT) {
    return std::nullopt;
  }

  ElementsAttr pad_elems;
  if (!matchPattern(pad, m_Constant(&pad_elems))) {
    (void)rewriter.notifyMatchFailure(op, "fail to retrieve padding values");
    return std::nullopt;
  }

  // Split N-rank mirrorpad into N sequences of "slice, reverse, concat"
  // operation. Take an example below. Perform mirrorpad on a 2D input tensor
  // with paddings[] = ([[1, 1,], [2, 2]]) in `SYMMETRIC` mode. Original input
  // tensor:
  //            | 1 2 3 |
  //            | 4 5 6 |
  //
  // First, fill in the padding value on the top and bottom.
  // pad before | 1 2 3 |
  //            | 1 2 3 |
  //            | 4 5 6 |
  // pad after  | 4 5 6 |
  //
  // Second, fill in the padding value on the left and right.
  // pad before          pad after
  //       2 1 | 1 2 3 | 3 2
  //       2 1 | 1 2 3 | 3 2
  //       5 4 | 4 5 6 | 6 5
  //       5 4 | 4 5 6 | 6 5

  // Indicate the value and shape of the tensor being processed in that
  // dimension.
  Value current_tensor(input);
  SmallVector<int64_t> current_dim_size = to_vector(input_type.getShape());

  // Note that this also represents the intermediate padded tensors for each
  // dimension.
  std::optional<Value> result;

  const int rank = input_type.getRank();
  const int offset = (mode == TFTFLMirrorPaddingType::SYMMETRIC) ? 0 : 1;

  for (int axis = 0; axis < rank; ++axis) {
    int pad_before = pad_elems.getValues<IntegerAttr>()[2 * axis].getInt();
    int pad_after = pad_elems.getValues<IntegerAttr>()[2 * axis + 1].getInt();

    SmallVector<int64_t> slice_before_begin, slice_before_size;
    SmallVector<int64_t> slice_after_begin, slice_after_size;

    for (int i = 0; i < rank; ++i) {
      if (axis == i) {
        // Calculate the padding area of slices that is going to be added to
        // the core tensor in that axis.
        slice_before_begin.push_back(offset);
        slice_before_size.push_back(pad_before);
        slice_after_begin.push_back(current_dim_size[i] - pad_after - offset);
        slice_after_size.push_back(pad_after);
      } else {
        // Keep the whole length of other axes.
        slice_before_begin.push_back(0);
        slice_before_size.push_back(current_dim_size[i]);
        slice_after_begin.push_back(0);
        slice_after_size.push_back(current_dim_size[i]);
      }
    }

    SmallVector<Value, 3> slices;

    if (pad_before != 0) {
      // Construct a padding slice for adding values before the contents of
      // tensor.
      auto slice_before_op = CreateOpAndInfer<tosa::SliceOp>(
          rewriter, op->getLoc(),
          RankedTensorType::get(slice_before_size,
                                output_type.getElementType()),
          current_tensor, rewriter.getDenseI64ArrayAttr(slice_before_begin),
          rewriter.getDenseI64ArrayAttr(slice_before_size));

      // Reverse op is superfluous when the padding value is 1.
      if (pad_before == 1) {
        slices.push_back(slice_before_op);
      } else {
        auto reverse_before_op = CreateOpAndInfer<tosa::ReverseOp>(
            rewriter, op->getLoc(), slice_before_op.getType(), slice_before_op,
            rewriter.getI64IntegerAttr(axis));
        slices.push_back(reverse_before_op);
      }
    }

    // Copy the core tensor
    slices.push_back(current_tensor);

    if (pad_after != 0) {
      // Construct a padding slice for adding values after the contents of
      // tensor.
      auto slice_after_op = CreateOpAndInfer<tosa::SliceOp>(
          rewriter, op->getLoc(),
          RankedTensorType::get(slice_after_size, output_type.getElementType()),
          current_tensor, rewriter.getDenseI64ArrayAttr(slice_after_begin),
          rewriter.getDenseI64ArrayAttr(slice_after_size));

      if (pad_after == 1) {
        slices.push_back(slice_after_op);
      } else {
        auto reverse_after_op = CreateOpAndInfer<tosa::ReverseOp>(
            rewriter, op->getLoc(), slice_after_op.getType(), slice_after_op,
            rewriter.getI64IntegerAttr(axis));
        slices.push_back(reverse_after_op);
      }
    }

    // The padded size of each dimension D of the output is:
    // paddings[D, 0] + tensor.dim_size(D) + paddings[D, 1]
    current_dim_size[axis] =
        pad_before + input_type.getDimSize(axis) + pad_after;

    // Create the expected output shape and type, and initialize it with zero.
    ShapedType result_type =
        UnrankedTensorType::get(output_type.getElementType());

    // Concatenate the old tensor with padding areas.
    result = convertConcatV2Op(rewriter, op, result_type, slices, axis);

    if (!result) return std::nullopt;

    // Update to the padded tensor
    current_tensor = result.value();
  }

  return result;
}

std::optional<Value> convertConv3DCommon(PatternRewriter& rewriter,
                                         Operation* op, ShapedType output_type,
                                         Value input, Value filter, Value bias,
                                         DenseI64ArrayAttr pads,
                                         DenseI64ArrayAttr strides,
                                         DenseI64ArrayAttr dilations,
                                         StringRef data_format_ref) {
  if (data_format_ref.str() != "NDHWC") {
    (void)rewriter.notifyMatchFailure(op, "currently only supports NDHWC");
    return std::nullopt;
  }

  RankedTensorType filter_type = filter.getType().cast<RankedTensorType>();

  // Note that the kernel shape of tfl.conv_3d isn't [O, D, H, W, I] but
  // [D, H, W, I, O] which is the same as in TF.
  // Transpose filter shape from [D, H, W, I, O] to [O, D, H, W, C]
  auto filter_shape = filter_type.getShape();
  SmallVector<int64_t, 5> a1_transpose_dims;
  a1_transpose_dims.push_back(filter_shape[4]);
  a1_transpose_dims.push_back(filter_shape[0]);
  a1_transpose_dims.push_back(filter_shape[1]);
  a1_transpose_dims.push_back(filter_shape[2]);
  a1_transpose_dims.push_back(filter_shape[3]);
  std::optional<Value> a1_filter_transpose_perm = getConstTensor<int32_t>(
      rewriter, op, /*vec=*/{4, 0, 1, 2, 3}, /*shape=*/{5});

  if (!a1_filter_transpose_perm) return std::nullopt;

  auto a1_filter_transpose_op = CreateOpAndInfer<tosa::TransposeOp>(
      rewriter, op->getLoc(),
      RankedTensorType::get(a1_transpose_dims, filter_type.getElementType()),
      filter, a1_filter_transpose_perm.value());

  return CreateOpAndInfer<tosa::Conv3DOp>(
             rewriter, op->getLoc(), output_type, input,
             a1_filter_transpose_op.getResult(), bias, pads, strides, dilations)
      .getResult();
}

// Lowers Gather operators to a sequence of TOSA ops.
std::optional<Value> convertGatherOp(PatternRewriter& rewriter, Operation* op,
                                     Value result_value, Value params_value,
                                     Value indices_value, int32_t batch_dims,
                                     int32_t axis) {
  auto result_type = result_value.getType().dyn_cast<ShapedType>();
  auto params_type = params_value.getType().dyn_cast<RankedTensorType>();
  auto indices_type = indices_value.getType().dyn_cast<RankedTensorType>();

  if (!result_type || !params_type || !indices_type) return std::nullopt;

  // batch_dims indicates the number of batch dimensions in params and
  // indices axis indicates the axis at which the gather indexing is
  // applied.  axis must be >= batch_dims.  When axis is equal to
  // batch_dims, the right-most batch dimension disappears.
  //
  // N: number of batches
  // Computed as product of params.shape[0:batch_dims-1]
  //
  // W: number of indices in each batch
  // Computed as product of indices.shape[batch_dims:]
  //
  // K: range of each index
  // Computed as  params.shape[axis:axis+rank(indices)-1]
  //
  // C: number of channels for each index
  // Computed as:  LeftChannels * RightChannels:
  // product(params.shape[batch_dims:axis]) * product(params.shape[axis+1:])
  //
  // The params tensor needs to be transposed, then reshaped to move the
  // dimensions into [N, K, C] order.
  //
  // The dimensions of the input params[] tensor are grouped in the following
  // order to begin with:
  //
  //  [Batch, LeftChannels, Indices, RightChannels]
  //  |-----||------------||-------||-------------|
  //     N         C_l         K          C_r
  //
  // Where Batch (N), Indices (K) can be one or more dimensions in size,
  // while LeftChannels and RightChannels represent the group of data channels
  // (C) to the left and right (C_l, C_r) of the indices; the sum of these two
  // is one or more dimensions in size, but either one may be zero depending
  // on how axis was specified by the caller.
  //
  // The resulting tensor will look like:
  //
  //  [Batch, Indices, LeftChannels, RightChannels]
  //  |-----||-------||---------------------------|
  //     N       K                 C
  //
  // The indices tensor simply needs a reshape to flatten all of the
  // batch dimensions (N) together and flatten all of the indices (W)
  // together.
  //
  // Then do the tosa.GATHER
  //
  // output[N,W,C] = tosa.GATHER(values[N,K,C], indices[N,W])
  //
  // Finally, the resulting tensor will have shape [N, W, C], where C is a
  // flattened version of [LeftChannels, RightChannels].  We need to reshape
  // to unflatten to:
  //
  //  [N, W, LeftChannels, RightChannels]
  //
  // and finally transpose back to the output shape
  //
  //  [Batch, LeftChannels, Non-Batch-Indices, RightChannels]

  int params_rank = params_type.getRank();
  int indices_rank = indices_type.getRank();

  if (axis < 0 || axis >= params_rank) {
    (void)rewriter.notifyMatchFailure(
        op, llvm::formatv("axis {} must be within range of params rank", axis));
    return std::nullopt;
  }

  if (!(batch_dims <= indices_rank)) {
    (void)rewriter.notifyMatchFailure(
        op, "batch_dims must be <= indices_rank for a valid gather op");
    return std::nullopt;
  }

  if (!(axis >= batch_dims)) {
    (void)rewriter.notifyMatchFailure(
        op, "axis must be >= batch_dims for a valid gather op");
    return std::nullopt;
  }

  // tf/tfl allow i64 indices, but tosa does not.
  if (indices_type.getElementType().isInteger(64)) {
    indices_type =
        indices_type.clone(rewriter.getI32Type()).dyn_cast<RankedTensorType>();
    indices_value = CreateOpAndInfer<tosa::CastOp>(rewriter, op->getLoc(),
                                                   indices_type, indices_value)
                        .getResult();
  }

  // Sizes for each of these fields.
  SmallVector<int64_t> params_batch, params_indices, params_left_channels,
      params_right_channels;

  // Dimension indices for each of these fields.
  SmallVector<int64_t> params_idx_batch, params_idx_indices,
      params_idx_left_channels, params_idx_right_channels;

  // Read through the params tensor dimensions left-to-right and extract the
  // different fields.
  for (int i = 0; i < params_rank; i++) {
    // When batch_dims == axis, the batch dimension gets replaced.
    if (i < batch_dims && i < axis) {
      params_batch.push_back(params_type.getShape()[i]);
      params_idx_batch.push_back(i);
    } else if (i < axis) {
      params_left_channels.push_back(params_type.getShape()[i]);
      params_idx_left_channels.push_back(i);
    } else if (i < (axis + 1)) {
      params_indices.push_back(params_type.getShape()[i]);
      params_idx_indices.push_back(i);
    } else {
      params_right_channels.push_back(params_type.getShape()[i]);
      params_idx_right_channels.push_back(i);
    }
  }

  // Calculate N, K, W, C
  int64_t N = multiply_dims(params_type.getShape().take_front(batch_dims));
  int64_t W = multiply_dims(
      indices_type.getShape().slice(batch_dims, indices_rank - batch_dims));
  int64_t K = params_type.getShape()[axis];

  int64_t C = multiply_dims(
      params_type.getShape().slice(batch_dims, axis - batch_dims));
  C = multiply_dims(
      params_type.getShape().slice(axis + 1, params_rank - axis - 1), C);

  /////////////////////////////////////////////
  // Build up the params transpose operator
  SmallVector<int32_t> params_transpose_perm;
  SmallVector<int64_t> params_transpose_shape;

  // Batch
  for (int i = 0; i < params_batch.size(); i++) {
    params_transpose_perm.push_back(params_idx_batch[i]);
    params_transpose_shape.push_back(params_batch[i]);
  }

  // Indices
  for (int i = 0; i < params_indices.size(); i++) {
    params_transpose_perm.push_back(params_idx_indices[i]);
    params_transpose_shape.push_back(params_indices[i]);
  }

  // LeftChannels
  for (int i = 0; i < params_left_channels.size(); i++) {
    params_transpose_perm.push_back(params_idx_left_channels[i]);
    params_transpose_shape.push_back(params_left_channels[i]);
  }

  // RightChannels
  for (int i = 0; i < params_right_channels.size(); i++) {
    params_transpose_perm.push_back(params_idx_right_channels[i]);
    params_transpose_shape.push_back(params_right_channels[i]);
  }

  /////////////////////////////////////////////
  // Build up the result reshape, in prepration for transpose
  // [N, W, C] -> [ Batch, Indices, LeftChannels, RightChannels ]
  SmallVector<int64_t> result_reshape_shape;

  // Indices
  for (int i = 0; i < indices_type.getShape().size(); i++) {
    result_reshape_shape.push_back(indices_type.getShape()[i]);
  }

  // Left channels
  for (int i = 0; i < params_left_channels.size(); i++) {
    result_reshape_shape.push_back(params_left_channels[i]);
  }

  // Right channels.  But remove the axis dimension.
  for (int i = 0; i < params_right_channels.size(); i++) {
    result_reshape_shape.push_back(params_right_channels[i]);
  }

  /////////////////////////////////////////////
  // Build up the result transpose operator.
  SmallVector<int32_t> result_transpose_perm;

  // Batch dimensions
  for (int i = 0; i < batch_dims; i++) {
    result_transpose_perm.push_back(i);
  }

  // LeftChannels
  for (int i = 0; i < params_left_channels.size(); i++) {
    result_transpose_perm.push_back(i + indices_type.getShape().size());
  }

  // Indices (remainder of dimensions after batch).
  for (int i = batch_dims; i < (indices_type.getShape().size()); i++) {
    result_transpose_perm.push_back(i);
  }

  // RightChannels, coming from after both the Indices and LeftChannels.
  for (int i = 0; i < params_right_channels.size(); i++) {
    result_transpose_perm.push_back(i + indices_type.getShape().size() +
                                    params_left_channels.size());
  }

  SmallVector<int64_t> tosa_values_shape = {N, K, C};
  SmallVector<int64_t> tosa_indices_shape = {N, W};
  SmallVector<int64_t> tosa_gather_result_shape = {N, W, C};

  std::optional<Value> params_transpose_perm_val = getConstTensor<int32_t>(
      rewriter, op, params_transpose_perm,
      {static_cast<int64_t>(params_transpose_perm.size())});

  std::optional<Value> result_transpose_perm_val = getConstTensor<int32_t>(
      rewriter, op, result_transpose_perm,
      {static_cast<int64_t>(result_transpose_perm.size())});

  if (!params_transpose_perm_val || !result_transpose_perm_val)
    return std::nullopt;

  auto params_transpose_op = CreateOpAndInfer<tosa::TransposeOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(params_transpose_shape,
                                           params_type.getElementType()),
      params_value, params_transpose_perm_val.value());

  if (count_dynamic_dims(tosa_values_shape) > 1) {
    return (void)rewriter.notifyMatchFailure(
               op,
               "multiply dynamic shapes when reshaping values down to "
               "tosa.gather"),
           std::nullopt;
  }

  auto tosa_values_reshape_op = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(tosa_values_shape,
                                           params_type.getElementType()),
      params_transpose_op.getResult(),
      rewriter.getDenseI64ArrayAttr(
          tensorflow::ConvertMlirShapeToTF(tosa_values_shape)));

  if (count_dynamic_dims(tosa_indices_shape) > 1) {
    return (void)rewriter.notifyMatchFailure(
               op,
               "multiply dynamic shapes when reshaping indices down to "
               "tosa.gather"),
           std::nullopt;
  }

  auto tosa_indices_reshape_op = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(tosa_indices_shape,
                                           indices_type.getElementType()),
      indices_value,
      rewriter.getDenseI64ArrayAttr(
          tensorflow::ConvertMlirShapeToTF(tosa_indices_shape)));

  auto tosa_gather_op = CreateOpAndInfer<tosa::GatherOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(tosa_gather_result_shape,
                                           result_type.getElementType()),
      tosa_values_reshape_op.getResult(), tosa_indices_reshape_op.getResult());

  if (count_dynamic_dims(result_reshape_shape) > 1) {
    return (void)rewriter.notifyMatchFailure(
               op,
               "multiply dynamic shapes when reshaping tosa.gather result up"),
           std::nullopt;
  }

  auto tosa_result_reshape_op = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(result_reshape_shape,
                                           params_type.getElementType()),
      tosa_gather_op.getResult(),
      rewriter.getDenseI64ArrayAttr(
          tensorflow::ConvertMlirShapeToTF(result_reshape_shape)));

  return CreateOpAndInfer<tosa::TransposeOp>(rewriter, op->getLoc(),
                                             result_type,
                                             tosa_result_reshape_op.getResult(),
                                             result_transpose_perm_val.value())
      .getResult();
}

// Lowers Gather operators to a sequence of TOSA ops.
std::optional<Value> convertGatherNdOp(PatternRewriter& rewriter, Operation* op,
                                       Value result_value, Value params_value,
                                       Value indices_value)

{
  auto result_type = result_value.getType().dyn_cast<RankedTensorType>();
  auto params_type = params_value.getType().dyn_cast<RankedTensorType>();
  auto indices_type = indices_value.getType().dyn_cast<RankedTensorType>();

  if (!result_type || !params_type || !indices_type) return std::nullopt;

  // N: number of batches
  // Always 1 for GatherND
  //
  // Because TOSA's GATHER operator already uses the symbol 'N' for
  // the number of batches, we will use the symbol 'ND' to specify the
  // number of dimensions that are sliced from params instead of'N' in
  // the TF MLIR documentation.
  //
  // ND: indices.shape[-1]
  //
  // W: number of indices in each batch
  // Computed as:
  // product(indices.shape[0:-1]) (all but the last dimension)
  //
  // K: range of each index
  // Computed as:
  // product(params.shape[0:ND-1])
  //
  // C: number of channels for each index
  // Computed as:
  // product(params.shape[ND:])
  //
  // The params tensor needs to be reshaped, but not transposed, to move the
  // dimensions into [N, K, C] order.
  //
  // The dimensions of the input params[] tensor are grouped in the following
  // order to begin with:
  //
  //  [ParamIndices, ParamChannels]
  //  |------------||-------------|
  //         K              C
  //
  // The reshape simply flattens the params tensor into a 2D [K, C] shape.
  //
  // Indices needs to be put in the form of [N, W], but a simple flattening
  // will not suffice, because the indices need to index into a [W]-shape
  // vector instead of the params.shape[0:ND-1] tensor that we had before.
  //
  // To flatten the coordinates, first reshape indices to a [W, ND] matrix,
  // where the matrix now represents W ND-dimensional coordinates into the
  // params tensor.
  //
  // From here, we take each of the ND dimensions and multiply it with
  // the size of the next params dimension (or 1 for the last
  // dimension), then sum all these together with a reduce_sum
  // operator.  This is exactly the same mathematics as one would use
  // flatten the indices of an N-dimensional row-major array into a
  // 1-D array in C.
  //
  // More precisely, do an element-wise multiply with [params.shape[1
  // .. ND], 1] in axis 1, then reduce_sum in axis 1 to flatten to a
  // [W]-shaped tensor, then trivially reshape to [N=1, W] to be
  // compatible with the GATHER operator's shape.
  //
  // Then perform the tosa.GATHER() operation.
  //
  // Now we have result = [N, K, C].
  //
  // Reshape with a single, simple reshape to the final output shape of:
  //  [Indices, ParamChannels]
  //
  // Where, Indices is indices.shape[0:ND-1]

  int N = 1, W = 1, K = 1, C = 1, ND = 1;

  int params_rank = params_type.getShape().size();
  int indices_rank = indices_type.getShape().size();

  ND = indices_type.getShape()[indices_rank - 1];

  if (ND > params_rank) {
    (void)rewriter.notifyMatchFailure(
        op, "size of last dimension of indices must be <= params rank");
    return std::nullopt;
  }

  // Calculate N, K, W, C.  (N is always 1)
  for (int i = 0; i < (indices_rank - 1); i++) {
    W *= indices_type.getShape()[i];
  }

  for (int i = 0; i < ND; i++) {
    K *= params_type.getShape()[i];
  }

  for (int i = ND; i < params_rank; i++) {
    C *= params_type.getShape()[i];
  }

  SmallVector<int64_t, 3> tosa_values_shape({N, K, C});
  SmallVector<int64_t, 2> tosa_indices_shape({N, W});
  SmallVector<int64_t, 2> indices_matrix_shape({W, ND});
  SmallVector<int64_t, 3> tosa_gather_result_shape({N, W, C});

  auto tosa_values_reshape_op = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(tosa_values_shape,
                                           params_type.getElementType()),
      params_value,
      rewriter.getDenseI64ArrayAttr(
          tensorflow::ConvertMlirShapeToTF(tosa_values_shape)));

  // Flatten the input indices tensor to an [W, ND] matrix.
  auto indices_matrix_reshape_op = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(indices_matrix_shape,
                                           indices_type.getElementType()),
      indices_value,
      rewriter.getDenseI64ArrayAttr(
          tensorflow::ConvertMlirShapeToTF(indices_matrix_shape)));

  SmallVector<int32_t> flattened_coeff_vec;
  for (int i = 1; i < ND; i++) {
    flattened_coeff_vec.push_back(params_type.getShape()[i]);
  }
  flattened_coeff_vec.push_back(1);

  for (int i = ND - 1; i > 0; i--) {
    flattened_coeff_vec[i - 1] *= flattened_coeff_vec[i];
  }

  std::optional<Value> flattened_coeff_value = getConstTensor<int32_t>(
      rewriter, op, flattened_coeff_vec,
      {static_cast<int64_t>(flattened_coeff_vec.size())});

  if (!flattened_coeff_value) return std::nullopt;

  // Multiply the coefficients by the coordinates
  auto flattened_indices_mul_op = CreateOpAndInfer<tosa::MulOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(indices_matrix_shape,
                                           indices_type.getElementType()),
      indices_matrix_reshape_op.getResult(), flattened_coeff_value.value(), 0);

  // Sum up the products of the coefficients and coordinates
  auto flattened_indices_reduce_op = CreateOpAndInfer<tosa::ReduceSumOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(tosa_indices_shape,
                                           indices_type.getElementType()),
      flattened_indices_mul_op.getResult(), rewriter.getI64IntegerAttr(1));

  // And reshape to [N, W]
  auto tosa_indices_reshape_op = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(tosa_indices_shape,
                                           indices_type.getElementType()),
      flattened_indices_reduce_op.getResult(),
      rewriter.getDenseI64ArrayAttr(
          tensorflow::ConvertMlirShapeToTF(tosa_indices_shape)));

  // Now the gather op itself
  auto tosa_gather_op = CreateOpAndInfer<tosa::GatherOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(tosa_gather_result_shape,
                                           result_type.getElementType()),
      tosa_values_reshape_op.getResult(), tosa_indices_reshape_op.getResult());

  // Finally, reshape back to the original output shape of [Indices,
  // ParamChannels].
  return CreateOpAndInfer<tosa::ReshapeOp>(
             rewriter, op->getLoc(), result_type, tosa_gather_op.getResult(),
             rewriter.getDenseI64ArrayAttr(
                 tensorflow::ConvertMlirShapeToTF(result_type.getShape())))
      .getResult();
}

// Lowers OneHot operator to a sequence of TOSA ops.
std::optional<Value> convertOneHotOp(PatternRewriter& rewriter, Operation* op,
                                     Value result_value, Value indices_value,
                                     Value on_value, Value off_value,
                                     int32_t depth, int32_t axis) {
  auto result_type = result_value.getType().dyn_cast<RankedTensorType>();
  auto indices_type = indices_value.getType().dyn_cast<RankedTensorType>();
  auto on_value_type = on_value.getType().dyn_cast<RankedTensorType>();
  auto off_value_type = off_value.getType().dyn_cast<RankedTensorType>();

  if (!result_type || !indices_type || !on_value_type || !off_value_type)
    return std::nullopt;

  // OneHot operator creates a new tensor with shape indices.shape[:axis] +
  // [depth] + indices.shape[axis:] For each index in 'indices', it needs to
  // be within range of [0, depth - 1] and the [..., k, ...] = on_value (if k
  // = index), or [..., k, ...] = off_value (if k != index)
  //
  // The lowering below assumes depth is always known at compile time.
  // TBD for depth resolved in run time.
  //
  // OneHot can be lowered as TOSA Scatter, where off_value being mapped to
  // 'values_in', on_value being mapped to 'input', and indices naturally
  // mapped to 'indices'. Also the dimensions of TOSA scatter (N, W, K, C)
  // need to be picked.
  //
  // N: number of elements of input indices
  // Computed as:
  // product(indices.shape[:])
  //
  // K: newly added dimension
  // K = depth
  //
  // W, C: dummy dimension now
  // W = C = 1
  //
  // High level description of lowering looks like:
  // 1. off_value is reshaped/tiled into [N, K, C]
  // 2. on_value is reshaped/tiled into [N, W, C]
  // 3. indices is reshaped into [N, W]
  // 4. scatter into [N, K, C]
  // 5. reshaped into [LeftDims, RightDims, K]
  // 6. transpose into [LeftDims, K, RightDims]
  // 7. reshaped to result.shape

  if (on_value_type.getRank() != 0 || off_value_type.getRank() != 0) {
    (void)rewriter.notifyMatchFailure(op,
                                      "on_value/off_value needs to be scalar");
    return std::nullopt;
  }

  if (axis < -1 || axis > indices_type.getRank()) {
    (void)rewriter.notifyMatchFailure(
        op, "axis out of valid range [-1, indices.rank]");
    return std::nullopt;
  }

  // axis = -1 is equivalent to axis = indices.rank
  if (axis == -1) {
    axis = indices_type.getRank();
  }

  int N = 1, W = 1, C = 1;
  int K = depth;
  int left_dim = 1, right_dim = 1;

  for (int32_t i = 0; i < indices_type.getRank(); i++) {
    int32_t dim = indices_type.getShape()[i];
    N *= dim;
    if (i >= axis) {
      right_dim *= dim;
    } else {
      left_dim *= dim;
    }
  }

  // Reshape on_value to [1, 1, 1]
  auto op1_reshape_on_value = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape({1, 1, 1},
                                           on_value_type.getElementType()),
      on_value, rewriter.getDenseI64ArrayAttr({1, 1, 1}));

  // And tile to [N, W, C]
  auto op2_tile_op1 = CreateOpAndInfer<tosa::TileOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape({N, W, C},
                                           on_value_type.getElementType()),
      op1_reshape_on_value.getResult(),
      rewriter.getDenseI64ArrayAttr({N, W, C}));

  // Reshape off_value to [1, 1, 1]
  auto op3_reshape_off_value = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape({1, 1, 1},
                                           off_value_type.getElementType()),
      off_value, rewriter.getDenseI64ArrayAttr({1, 1, 1}));

  // And tile to [N, K, C]
  auto op4_tile_op3 = CreateOpAndInfer<tosa::TileOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape({N, K, C},
                                           on_value_type.getElementType()),
      op3_reshape_off_value.getResult(),
      rewriter.getDenseI64ArrayAttr({N, K, C}));

  // Reshape indices to [N, W]
  auto op5_reshape_indices = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape({N, W},
                                           indices_type.getElementType()),
      indices_value,
      rewriter.getDenseI64ArrayAttr(tensorflow::ConvertMlirShapeToTF({N, W})));

  // Scatter to [N, K, C]
  auto op6_scatter_op4_op5_op2 = CreateOpAndInfer<tosa::ScatterOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape({N, K, C},
                                           result_type.getElementType()),
      op4_tile_op3.getResult(), op5_reshape_indices.getResult(),
      op2_tile_op1.getResult());

  // Reshaped to [LeftDims, RightDims, K]. C being squeezed out since it's 1.
  auto op7_reshape_op6 = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape({left_dim, right_dim, K},
                                           result_type.getElementType()),
      op6_scatter_op4_op5_op2.getResult(),
      rewriter.getDenseI64ArrayAttr(
          tensorflow::ConvertMlirShapeToTF({left_dim, right_dim, K})));

  // Transposed to [LeftDims, K, RightDims].
  std::optional<Value> perm_const =
      getConstTensor<int32_t>(rewriter, op, /*vec=*/{0, 2, 1}, /*shape=*/{3});

  if (!perm_const) return std::nullopt;

  auto op8_transpose_op7 = CreateOpAndInfer<tosa::TransposeOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape({left_dim, K, right_dim},
                                           result_type.getElementType()),
      op7_reshape_op6.getResult(), perm_const.value());

  // Reshaped to result.shape.
  return CreateOpAndInfer<tosa::ReshapeOp>(
             rewriter, op->getLoc(), result_type, op8_transpose_op7.getResult(),
             rewriter.getDenseI64ArrayAttr(
                 tensorflow::ConvertMlirShapeToTF(result_type.getShape())))
      .getResult();
}

// Lowers Sin operator to a sequence of TOSA ops.
std::optional<Value> convertSinOp(PatternRewriter& rewriter, Operation* op,
                                  Value input, ShapedType output_type) {
  RankedTensorType input_type = input.getType().dyn_cast<RankedTensorType>();
  Location loc = op->getLoc();

  Type input_ety = input_type.getElementType();
  Type output_ety = output_type.getElementType();

  if (!input) return std::nullopt;

  if (input_ety != output_ety) {
    (void)rewriter.notifyMatchFailure(op,
                                      "input/output element type must match");
    return std::nullopt;
  }

  bool input_is_fp = input_ety.isF32();
  bool output_is_fp = output_ety.isF32();

  if (!input_is_fp || !output_is_fp) {
    (void)rewriter.notifyMatchFailure(op, "input/result must be fp32");
    return std::nullopt;
  }

  // To perform a sin operation we remap the sin domain to be over a single
  // period of the function, remapping to the domain of the table function.
  // We then remap the range of the table function to map to the range of the
  // sin operation.

  // 1. Normalize the period of the domain from [0, 2π) to [0, 1).
  auto fp_scalar_ty = RankedTensorType::get({}, rewriter.getF32Type());
  Value fp_scale = rewriter.create<tosa::ConstOp>(
      loc, fp_scalar_ty,
      DenseElementsAttr::get(fp_scalar_ty, {static_cast<float>(0.5 / M_PI)}));

  // 2. Remap the periodic behavior of the domain to line up within [0, 1).
  Value fp_scaled =
      CreateOpAndInfer<tosa::MulOp>(rewriter, loc, input_type, input, fp_scale,
                                    rewriter.getI32IntegerAttr(0));
  auto floored =
      CreateOpAndInfer<tosa::FloorOp>(rewriter, loc, input_type, fp_scaled);
  auto repeated = CreateOpAndInfer<tosa::SubOp>(rewriter, loc, input_type,
                                                fp_scaled, floored);

  // 3. Scale and translate the normalized domain to the table domain. This
  // includes a translating and scaling to [-int16_max, int16_max] and casting
  // to an i16.
  Value one = rewriter.create<tosa::ConstOp>(
      loc, fp_scalar_ty, DenseElementsAttr::get(fp_scalar_ty, {1.0f}));

  Value two = rewriter.create<tosa::ConstOp>(
      loc, fp_scalar_ty, DenseElementsAttr::get(fp_scalar_ty, {2.0f}));
  auto scale_up = CreateOpAndInfer<tosa::MulOp>(
      rewriter, loc, input_type, repeated, two, rewriter.getI32IntegerAttr(0));
  auto translate =
      CreateOpAndInfer<tosa::SubOp>(rewriter, loc, input_type, scale_up, one);

  Value int_limit = rewriter.create<tosa::ConstOp>(
      loc, fp_scalar_ty,
      DenseElementsAttr::get(
          fp_scalar_ty,
          {static_cast<float>(std::numeric_limits<int16_t>::max())}));
  auto int_scaled =
      CreateOpAndInfer<tosa::MulOp>(rewriter, loc, input_type, translate,
                                    int_limit, rewriter.getI32IntegerAttr(0));

  auto int16_ty = input_type.clone(rewriter.getIntegerType(16));
  auto casted =
      CreateOpAndInfer<tosa::CastOp>(rewriter, loc, int16_ty, int_scaled);

  // 4. Compute the lookup table using the range of [-255, 255] for sin.
  llvm::SmallVector<int16_t> values;
  const int num_values = 513;
  values.resize(num_values, 0);
  // First and last values should be 0;
  for (int i = 1; i < num_values - 1; ++i)
    values[i] = std::numeric_limits<int16_t>::max() *
                sin(static_cast<float>(i) * 2.0 * M_PI / (num_values - 1.0));

  auto table_ty =
      RankedTensorType::get({num_values}, rewriter.getIntegerType(16));
  Value table = rewriter.create<tosa::ConstOp>(
      loc, table_ty, DenseElementsAttr::get(table_ty, llvm::ArrayRef(values)));

  auto table_result_ty = input_type.clone(rewriter.getIntegerType(32));
  auto table_result = CreateOpAndInfer<tosa::TableOp>(
      rewriter, loc, table_result_ty, casted, table);

  // 5. The range of table is a 23-bit two's compliment value. Normalize the
  // range by casting to an fp32 and dividing by 2^22.
  auto table_result_fp =
      CreateOpAndInfer<CastOp>(rewriter, loc, input_type, table_result);
  auto output_scale = rewriter.create<ConstOp>(
      loc, fp_scalar_ty,
      DenseElementsAttr::get(
          fp_scalar_ty,
          {static_cast<float>(1.0 / static_cast<float>(1 << 22))}));

  return CreateOpAndInfer<MulOp>(rewriter, loc, output_type, table_result_fp,
                                 output_scale, rewriter.getI32IntegerAttr(0))
      .getResult();
}

// Lowers Sign operator to a sequence of TOSA ops.
llvm::Optional<Value> convertSignOp(PatternRewriter& rewriter, Operation* op,
                                    Value input, RankedTensorType output_type) {
  auto output_elem_type = output_type.getElementType();
  if (output_elem_type.isa<mlir::quant::QuantizedType>()) {
    (void)rewriter.notifyMatchFailure(op, "tfl quantization not yet supported");
    return llvm::None;
  }

  // TOSA greater and select can both broadcast, so simply create a tensor with
  // one element.
  Value pos_one, neg_one, zero;
  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  if (output_elem_type.isa<FloatType>()) {
    pos_one = getTosaConstTensorSingleF32(rewriter, op, 1.0f);
    neg_one = getTosaConstTensorSingleF32(rewriter, op, -1.0f);
    zero = getTosaConstTensorSingleF32(rewriter, op, 0.0f);
  } else {
    pos_one = getTosaConstTensorScalarInt(builder, output_elem_type, 1);
    neg_one = getTosaConstTensorScalarInt(builder, output_elem_type, -1);
    zero = getTosaConstTensorScalarInt(builder, output_elem_type, 0);
  }

  ShapedType const_type = output_type.clone(rewriter.getIntegerType(1));

  auto gt_zero_op =
      CreateOpAndInfer<tosa::GreaterOp>(builder, const_type, input, zero);

  auto lt_zero_op =
      CreateOpAndInfer<tosa::GreaterOp>(builder, const_type, zero, input);

  auto select_neg_op = CreateOpAndInfer<tosa::SelectOp>(
      builder, output_type, lt_zero_op, neg_one, zero);

  // Select positive one based on the condition tensor.
  return CreateOpAndInfer<tosa::SelectOp>(builder, output_type, gt_zero_op,
                                          pos_one, select_neg_op)
      .getResult();
}

};  // namespace tosa
};  // namespace mlir
