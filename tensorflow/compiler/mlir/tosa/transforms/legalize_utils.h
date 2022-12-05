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

#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <optional>

#include "mlir/Dialect/Func/IR/FuncOps.h"          // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"         // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"    // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"             // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"          // from @llvm-project
#include "mlir/IR/PatternMatch.h"                  // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/Rewrite/FrozenRewritePatternSet.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"                     // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/kernels/conv_grad_shape_utils.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace mlir {
namespace tosa {

LogicalResult getDynamicDims(PatternRewriter& rewriter, Value value,
                             llvm::SmallVector<Value>& dims);

std::optional<Value> buildReshapeWithDynamicDims(PatternRewriter& rewriter,
                                                 Operation* op,
                                                 Value input_value,
                                                 ShapedType output_type,
                                                 llvm::ArrayRef<Value> dims);

// Create a TOSA rescale op from TFLite scaling, zero points and rounding mode
Value buildRescale(PatternRewriter& rewriter, Operation* op,
                   ShapedType output_type, Value input_val, double scale,
                   int64_t input_zp, int64_t output_zp, bool double_round,
                   bool scale32);

// Removes the zero point and cast to int32, no need to handle roundings modes
Value removeZeroPointAndCastToInt32(PatternRewriter& rewriter, Operation* op,
                                    Value input_val, int64_t input_zp);

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

// Create a 32-bit TOSA TABLE constant tensor
// Output is restricted to [-1.0, 1.0] as s0.31 format
void getTosaConst32bitTable(PatternRewriter& rewriter, Operation* op,
                            double input_scale, int32_t input_zp,
                            std::function<double(double)> func,
                            Value& first_const, Value& second_const,
                            Value& third_const, Value& fourth_const);

// Create a 32-bit float constant operator from a float
Value getTosaConstTensorSingleF32(PatternRewriter& rewriter, Operation* op,
                                  float val);

// Create a 32-bit integer constant operator from an int
Value getTosaConstTensorSingleI32(PatternRewriter& rewriter, Operation* op,
                                  int32_t val);

// Create an expected bitwidth integer constant operator based on the type
// parameter.
Value getTosaConstTensorScalarInt(ImplicitLocOpBuilder& builder, Type type,
                                  int32_t val);

// Create a vector from a 32-bit value tensor.  Returns vector size on success
// or -1 on error.
LogicalResult getVectorFromValue32(Value val, SmallVectorImpl<int32_t>& vec);

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
  auto op = builder.create<TosaOp>(result_ty, args...);

  InferShapedTypeOpInterface shapeInterface =
      dyn_cast<InferShapedTypeOpInterface>(op.getOperation());
  if (!shapeInterface) return op;

  SmallVector<ShapedTypeComponents> returnedShapes;
  if (shapeInterface
          .inferReturnTypeComponents(op.getContext(), builder.getLoc(),
                                     op->getOperands(), op->getAttrDictionary(),
                                     op->getRegions(), returnedShapes)
          .failed())
    return op;

  // We need to use the element type of the existing result type to generate
  // the new result shaped type. This is because rescale can include a cast to
  // different bit-width types and does not have a TypeAttr to define the
  // target type.
  auto result = op->getResult(0);
  auto predictedShape = returnedShapes[0];
  auto currentKnowledge = ValueKnowledge::getKnowledgeFromType(result_ty);

  // Compute the knowledge based on the inferred type.
  auto inferredKnowledge = ValueKnowledge::getPessimisticValueState();
  inferredKnowledge.dtype = result_ty.cast<ShapedType>().getElementType();
  inferredKnowledge.hasRank = predictedShape.hasRank();
  if (predictedShape.hasRank()) {
    for (auto dim : predictedShape.getDims()) {
      inferredKnowledge.sizes.push_back(dim);
    }
  }

  // Compute the new type based on the joined version.
  auto newKnowledge = ValueKnowledge::join(currentKnowledge, inferredKnowledge);
  Type new_ty =
      newKnowledge.hasRank
          ? Type{tensorflow::GetTypeFromTFTensorShape(
                llvm::ArrayRef(newKnowledge.sizes), newKnowledge.dtype)}
          : Type{mlir::UnrankedTensorType::get(newKnowledge.dtype)};
  result.setType(new_ty);
  return op;
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

void TrimQuantizedIntegerRangeMin(mlir::quant::UniformQuantizedType dtype,
                                  int64_t& val_min);

void TrimQuantizedIntegerRangeMax(mlir::quant::UniformQuantizedType dtype,
                                  int64_t& val_max);

void TrimQuantizedIntegerRange(mlir::quant::UniformQuantizedType dtype,
                               int64_t& val_min, int64_t& val_max);

}  // namespace tosa
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TOSA_TRANSFORMS_LEGALIZE_UTILS_H_
