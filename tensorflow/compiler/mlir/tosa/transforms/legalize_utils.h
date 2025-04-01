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

#ifndef TENSORFLOW_COMPILER_MLIR_TOSA_TRANSFORMS_LEGALIZE_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TOSA_TRANSFORMS_LEGALIZE_UTILS_H_

#include <cfloat>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <optional>

#include "mlir/Dialect/Func/IR/FuncOps.h"             // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"            // from @llvm-project
#include "mlir/Dialect/Tosa/IR/TosaOps.h"             // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"       // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"                // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                     // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"             // from @llvm-project
#include "mlir/IR/PatternMatch.h"                     // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"     // from @llvm-project
#include "mlir/Rewrite/FrozenRewritePatternSet.h"     // from @llvm-project
#include "mlir/Support/LLVM.h"                        // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/kernels/conv_grad_shape_utils.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace mlir {
namespace tosa {

// returns acc_type attribute for Conv ops with specified input/output element
// types
mlir::TypeAttr getConvAccTypeAttr(PatternRewriter& rewriter,
                                  mlir::Type input_etype,
                                  mlir::Type output_etype);

LogicalResult getDynamicDims(PatternRewriter& rewriter, Value value,
                             llvm::SmallVector<Value>& dims);

std::optional<Value> buildReshapeWithDynamicDims(PatternRewriter& rewriter,
                                                 Operation* op,
                                                 Value input_value,
                                                 ShapedType output_type,
                                                 llvm::ArrayRef<Value> dims);

// Create a TOSA rescale op from TFLite scaling multiplier, scaling shift, zero
// points and rounding mode
Value buildRescale(PatternRewriter& rewriter, Operation* op,
                   ShapedType output_type, Value input_val,
                   int32_t scale_multiplier, int32_t scale_shit,
                   int64_t input_zp, int64_t output_zp, StringRef rounding_mode,
                   bool scale32);

// Create a TOSA rescale op from TFLite scaling, zero points and rounding mode
Value buildRescale(PatternRewriter& rewriter, Operation* op,
                   ShapedType output_type, Value input_val, double scale,
                   int64_t input_zp, int64_t output_zp, StringRef rounding_mode,
                   bool scale32);

// Removes the zero point and cast to int32, no need to handle roundings modes
Value removeZeroPointAndCastToInt32(PatternRewriter& rewriter, Operation* op,
                                    Value input_val, int64_t input_zp);

// Creates TOSA rescale op with int32 output
Value buildRescaleToInt32(PatternRewriter& rewriter, Operation* op,
                          Value input_val, int32_t input_scale_multiplier,
                          int32_t input_scale_shift, int64_t input_zp);

// Creates TOSA rescale op with int32 output
Value buildRescaleToInt32(PatternRewriter& rewriter, Operation* op,
                          Value input_val, double input_scale,
                          int64_t input_zp);

// Creates TOSA rescale op with int32 input
Value buildRescaleFromInt32(PatternRewriter& rewriter, Operation* op,
                            ShapedType output_type, Value input_val,
                            double output_scale, int64_t output_zp);

// Creates a TOSA rescale op based on conv2d parameters.
Value buildRescaleOpConvOutput(PatternRewriter& rewriter, Operation* op,
                               Value conv_val, ShapedType input_type,
                               ShapedType weight_type, ShapedType output_type);

// Create a 8-bit TOSA TABLE constant tensor
Value getTosaConst8bitTable(PatternRewriter& rewriter, Operation* op,
                            double input_scale, int32_t input_zp,
                            double output_scale, int32_t output_zp,
                            std::function<double(double)> func);

// Create a 16-bit TOSA TABLE constant tensor
Value getTosaConst16bitTable(PatternRewriter& rewriter, Operation* op,
                             std::function<double(double)> func, double min,
                             double max);

// Create a 32-bit TOSA TABLE for Softmax Exp
void getTosaConst32bitSoftmaxExpTable(PatternRewriter& rewriter, Operation* op,
                                      double beta, double input_scale,
                                      Value& first_const, Value& second_const,
                                      Value& third_const, Value& fourth_const);

// Create 8 bit TOSA TABLE constant tensor for the RSqrt operator
Value getTosaConstRsqrt8bitTable(PatternRewriter& rewriter, Operation* op,
                                 float input_scale, int32_t input_zp,
                                 float output_scale, int32_t output_zp);

// Create a 32-bit float constant operator from a float
Value getTosaConstTensorSingleF32(PatternRewriter& rewriter, Operation* op,
                                  float val, int rank);

// Create a 32-bit integer constant operator from an int of specified rank
Value getTosaConstTensorSingleI32(PatternRewriter& rewriter, Operation* op,
                                  int32_t val, int rank);

// Create an expected bitwidth integer constant operator based on the type
// parameter, of specified rank
Value getTosaConstTensorScalarInt(ImplicitLocOpBuilder& builder, Type type,
                                  int64_t val, int rank);

// Create a tosa::ConstShape based on the specified values
Value getTosaConstShape(PatternRewriter& rewriter, Operation* op,
                        llvm::ArrayRef<int64_t> values);


// Populate a int32_t vector from a val tensor
// return failure if val is not a constant value
// return success otherwise
LogicalResult getVectorFromValue32(Value val, SmallVectorImpl<int32_t>& vec);

// Populate a int64_t vector from a val tensor
// return failure if val is not a constant value
// return success otherwise
LogicalResult getVectorFromValue64(Value val, SmallVectorImpl<int64_t>& vec);

// Calculates the TOSA padding values based on TF operators padded with
// SAME/VALID.
bool getPaddingValuesFromPadType(tensorflow::Padding tf_pad,
                                 tensorflow::TensorFormat data_format_tf,
                                 uint32_t first_filter_spatial_dim,
                                 ShapedType input_type, ShapedType filter_type,
                                 DenseI64ArrayAttr strides,
                                 DenseI64ArrayAttr dilations,
                                 PatternRewriter& rewriter,
                                 DenseI64ArrayAttr& explicit_pad);

// Calculates the TOSA padding values for explicit-padded TF operators.
DenseI64ArrayAttr getPaddingValuesFromExplicitPadAttr(
    ArrayAttr explicit_pad, tensorflow::TensorFormat data_format_tf,
    PatternRewriter& rewriter);

// Calculates the TOSA padding values for transposeConv2d
bool getTransposeConv2dPaddingValues(
    tensorflow::Padding tf_pad, tensorflow::TensorFormat data_format_tf,
    uint32_t first_filter_spatial_dim, ShapedType input_type,
    ShapedType filter_type, ShapedType output_type, DenseI64ArrayAttr strides,
    PatternRewriter& rewriter, DenseI64ArrayAttr& explicit_pad);

// Templated function to create a constant op for given type and shape.
// T: storage C type.
// Default template creates a constant tensor in T.
// To create INT48 TOSA constant, need to pass in llvm::APInt instead.
template <typename T>
std::optional<Value> getConstTensor(PatternRewriter& rewriter, Operation* op,
                                    ArrayRef<T> vec, ArrayRef<int64_t> shape);

// For each spatial dimension, return the remainder of the output size
// calculation: (I - 1 + pad - (K - 1) * dilation) % stride.
llvm::SmallVector<int64_t> getOutputSpatialSizeRemainder(
    tensorflow::TensorFormat data_format_tf, ShapedType input_type,
    DenseI64ArrayAttr kernel_size, DenseI64ArrayAttr pads,
    DenseI64ArrayAttr strides, DenseI64ArrayAttr dilations);

// The TOSA specification requires the full size of the input to be used during
// the convolution (the output size remainder calculation must be 0). If input
// slicing is necessary to satisfy the condition, return a tosa::SliceOp,
// otherwise return input_val.
Value getInputSlicedToItsUsedSize(PatternRewriter& rewriter, Operation* op,
                                  tensorflow::TensorFormat data_format_tf,
                                  ShapedType input_type, Value input_val,
                                  DenseI64ArrayAttr kernel_size,
                                  DenseI64ArrayAttr pads,
                                  DenseI64ArrayAttr strides,
                                  DenseI64ArrayAttr dilations);

// Check if scale32 mode is used for given output_element_type
bool isScale32(mlir::quant::UniformQuantizedType output_element_type);

// Applies a set of patterns greedily to the specified function, then applies
// a cleanup to guarantee the function contract and constants are valid. This
// means patterns can performed shape inference while not altering immutable
// types.
LogicalResult ApplyPatternsWithShapeResolution(
    func::FuncOp func, const FrozenRewritePatternSet& patterns);

// Creates a TOSA operation and performs shape inference on the individual
// op. This allows shape inference during the TFLite to TOSA lowering.
template <typename TosaOp, typename... Args>
TosaOp CreateOpAndInfer(ImplicitLocOpBuilder& builder, Type result_ty,
                        Args&&... args) {
  return CreateOpAndInferShape<TosaOp>(builder, result_ty, args...);
}

template <typename TosaOp, typename... Args>
TosaOp CreateOpAndInfer(PatternRewriter& rewriter, Location loc, Type result_ty,
                        Args&&... args) {
  ImplicitLocOpBuilder builder(loc, rewriter);
  return CreateOpAndInfer<TosaOp>(builder, result_ty, args...);
}

template <typename TosaOp, typename... Args>
void CreateReplaceOpAndInfer(PatternRewriter& rewriter, Operation* op,
                             Type result_ty, Args&&... args) {
  auto result =
      CreateOpAndInfer<TosaOp>(rewriter, op->getLoc(), result_ty, args...);
  rewriter.replaceOp(op, result->getResults());
}

// Nan propagation mode is only applied to maximum and mininum.
template <typename TOSA_OP>
LogicalResult ConvertBinaryOp(Operation* op, PatternRewriter& rewriter,
                              StringRef nan_mode = "") {
  TensorType output_type = dyn_cast<TensorType>(op->getResults()[0].getType());
  if (!output_type) return failure();

  Value x = op->getOperands()[0];
  Value y = op->getOperands()[1];

  RankedTensorType x_type = dyn_cast<RankedTensorType>(x.getType());
  RankedTensorType y_type = dyn_cast<RankedTensorType>(y.getType());
  if (!x_type || !y_type) return failure();

  if constexpr (std::is_same_v<tosa::ReduceMaxOp, TOSA_OP> ||
                std::is_same_v<tosa::ReduceMinOp, TOSA_OP>) {
    if (nan_mode != "PROPAGATE" && nan_mode != "IGNORE") {
      (void)rewriter.notifyMatchFailure(op, "invalid NaN mode: must be either 'PROPAGATE' or 'IGNORE'");
      return failure();
    }
    CreateReplaceOpAndInfer<TOSA_OP>(rewriter, op, output_type, x, y, nan_mode);
  } else
    CreateReplaceOpAndInfer<TOSA_OP>(rewriter, op, output_type, x, y);

  return success();
}

// Create TOSA mul ops and infer the shape of the operation. During the
// creation, fill in the shift value if applied.
tosa::MulOp CreateMulOpAndInfer(PatternRewriter& rewriter, Operation* op,
                                Type result_ty, Value input1, Value input2,
                                int8_t shift = 0);

void TrimQuantizedIntegerRangeMin(mlir::quant::UniformQuantizedType dtype,
                                  int64_t& val_min);

void TrimQuantizedIntegerRangeMax(mlir::quant::UniformQuantizedType dtype,
                                  int64_t& val_max);

void TrimQuantizedIntegerRange(mlir::quant::UniformQuantizedType dtype,
                               int64_t& val_min, int64_t& val_max);

inline bool IsTFLDoubleRoundingMode() {
#if TFLITE_SINGLE_ROUNDING
  return false;
#else
  return true;
#endif  // TFLITE_SINGLE_ROUNDING
}

Value reshapeScalarTo1D(PatternRewriter& rewriter, Location loc, Value value);

LogicalResult broadcastLowRankTensor(PatternRewriter &rewriter, Operation* op,
                                     Value &input1, Value &input2);

}  // namespace tosa
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TOSA_TRANSFORMS_LEGALIZE_UTILS_H_
