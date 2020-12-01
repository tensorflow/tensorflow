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

#ifndef TENSORFLOW_COMPILER_MLIR_TOSA_TRANSFORMS_LEGALIZE_COMMON_H
#define TENSORFLOW_COMPILER_MLIR_TOSA_TRANSFORMS_LEGALIZE_COMMON_H

// This file contains legalizations common to mapping both TensorFlow and
// TensorFlow Lite to TOSA.
//
// Conversion functions return nullptr on a lowerization failure or a lowered
// operator on success.   Callers must check and return a LogicalResult failure
// on nullptr.
//
// For these functions, the framework-specific operands/attributes/defaults
// are already extracted and placed in a common form for lowering.
#include "mlir/Dialect/Quant/FakeQuantSupport.h"
#include "mlir/Dialect/Quant/UniformSupport.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
namespace tosa {

// Lowers the Pack operator to TOSA.
Operation* convertPackOp(PatternRewriter& rewriter, Operation* op,
                         Value result_value, SmallVector<Value, 8>& inputs,
                         int32_t axis);

// Lowers the Unpack operator to TOSA.
Operation* convertUnpackOp(PatternRewriter& rewriter, Operation* op,
                           Value input_value, int32_t axis);

// Lowers the Select operator to TOSA.
Operation* convertSelectOp(PatternRewriter& rewriter, Operation* op,
                           Value result_value, Value condition_value,
                           Value x_value, Value y_value);

// Lowers the ZerosLike operator to TOSA by creating a constant
// of the desired type and shape.
Operation* convertZerosLikeOp(PatternRewriter& rewriter, Operation* op,
                              Value result, Value input);

// Lowers the Mul operator to TOSA.  For quantized types, this requires
// inserting rescale operators before and after the operation.
Operation* convertMultiplyOp(PatternRewriter& rewriter, Operation* op,
                             Value output_val, Value input_lhs_val,
                             Value input_rhs_val);

// Lowers the SquaredDifference operator to TOSA.
Operation* convertSquaredDifferenceOp(PatternRewriter& rewriter, Operation* op,
                                      Value result, Value x, Value y);

// Lowers the Round operator to TOSA.
Operation* convertRoundOp(PatternRewriter& rewriter, Operation* op,
                          Value result, Value input);

// Lowers ConcatV2 to TOSA.
Operation* convertConcatV2Op(PatternRewriter& rewriter, Operation* op,
                             Value result_value, SmallVector<Value, 8>& values,
                             int32_t axis);

// Lowers SpaceToBatchND to TOSA.
Operation* convertSpaceToBatchNDOp(PatternRewriter& rewriter, Operation* op,
                                   Value result_value, Value input_value,
                                   Value block_shape_value,
                                   Value paddings_value);

// Lowers BatchToSpaceND to TOSA.
Operation* convertBatchToSpaceNDOp(PatternRewriter& rewriter, Operation* op,
                                   Value result_value, Value input_value,
                                   Value block_shape_value, Value crops_value);

// Lowers ExpandDims to TOSA.
Operation* convertExpandDimsOp(PatternRewriter& rewriter, Operation* op,
                               Value result_value, Value input_value,
                               Value dim_value);

// Lowers Squeeze to TOSA.
Operation* convertSqueezeOp(PatternRewriter& rewriter, Operation* op,
                            Value result_value, Value input_value,
                            SmallVector<int32_t, 8>& squeeze_dims);

// Lowers ELU to a sequence of TOSA ops.
Operation* convertEluOp(PatternRewriter& rewriter, Operation* op,
                        Value result_value, Value features_value);

// Lowers Softmax to a sequence of TOSA ops.
Operation* convertSoftmaxOp(PatternRewriter& rewriter, Operation* op,
                            Value result_value, Value logits_value);

// Lowers LogSoftmax to a sequence of TOSA ops.
Operation* convertLogSoftmaxOp(PatternRewriter& rewriter, Operation* op,
                               Value result_value, Value logits_value);

// Lowers SpaceToDepth to a sequence of TOSA ops.  Supports NHWC.
Operation* convertSpaceToDepthOp(PatternRewriter& rewriter, Operation* op,
                                 Value result_value, Value input_value,
                                 IntegerAttr block_size_attr,
                                 StringAttr data_format);

// Lowers DepthToSpace to a sequence of TOSA ops.  Supports NHWC.
Operation* convertDepthToSpaceOp(PatternRewriter& rewriter, Operation* op,
                                 Value result_value, Value input_value,
                                 IntegerAttr block_size_attr,
                                 StringAttr data_format);

// Lowers Split to a sequence of TOSA ops.
Operation* convertSplitOp(PatternRewriter& rewriter, Operation* op,
                          Value result_value, Value input_value,
                          int32_t num_split, int32_t axis);

// Lowers SplitV to a sequence of TOSA ops.
Operation* convertSplitVOp(PatternRewriter& rewriter, Operation* op,
                           Value result_value, Value input_value,
                           SmallVector<int32_t, 4>& size_split, int32_t axis);

// Lowers StridedSlice to a sequence of TOSA ops.
Operation* convertStridedSliceOp(PatternRewriter& rewriter, Operation* op,
                                 Value result_value, Value input_value,
                                 Value begin_value, Value end_value,
                                 Value strides_value, int32_t begin_mask,
                                 int32_t end_mask, int32_t ellipsis_mask,
                                 int32_t new_axis_mask,
                                 int32_t shrink_axis_mask);

// Lowers FloorDiv to a sequence of TOSA operators.
Operation* convertFloorDivOp(PatternRewriter& rewriter, Operation* op,
                             Value result_value, Value lhs_value,
                             Value rhs_value);

// Lowers FloorMod to a sequence of TOSA operators.
Operation* convertFloorModOp(PatternRewriter& rewriter, Operation* op,
                             Value result_value, Value lhs_value,
                             Value rhs_value);

// Lowers FusedActivation to a sequence of TOSA ops.
Operation* convertFusedActivation(PatternRewriter& rewriter, Operation* op,
                                  Value input_value,
                                  StringAttr fused_activation_fn);

// Helper function for implementing quantized divide by power-of-two in TOSA
// ops.
Operation* convertRoundingDivideByPOT(PatternRewriter& rewriter, Operation* op,
                                      Value input_value, Value rshift_value);

// Lowers ReduceAll to a sequence of TOSA ops.
Operation* convertReduceAllOp(PatternRewriter& rewriter, Operation* op,
                              RankedTensorType output_type, Value input_value,
                              ElementsAttr axes_elems, bool keep_dims);

// Lowers ReduceAny to a sequence of TOSA ops.
Operation* convertReduceAnyOp(PatternRewriter& rewriter, Operation* op,
                              RankedTensorType output_type, Value input_value,
                              ElementsAttr axes_elems, bool keep_dims);

// Lowers ReduceMin to a sequence of TOSA ops.
Operation* convertReduceMinOp(PatternRewriter& rewriter, Operation* op,
                              RankedTensorType output_type, Value input_value,
                              ElementsAttr axes_elems, bool keep_dims);

// Lowers ReduceMax to a sequence of TOSA ops.
Operation* convertReduceMaxOp(PatternRewriter& rewriter, Operation* op,
                              RankedTensorType output_type, Value input_value,
                              ElementsAttr axes_elems, bool keep_dims);

// Lowers ReduceProd to a sequence of TOSA ops.
Operation* convertReduceProdOp(PatternRewriter& rewriter, Operation* op,
                               RankedTensorType output_type, Value input_value,
                               ElementsAttr axes_elems, bool keep_dims);

// Lowers ReduceSum to a sequence of TOSA ops.
Operation* convertReduceSumOp(PatternRewriter& rewriter, Operation* op,
                              RankedTensorType output_type, Value input_value,
                              ElementsAttr axes_elems, bool keep_dims);

// Lowers ReduceMean to a sequence of TOSA ops.
Operation* convertReduceMeanOp(PatternRewriter& rewriter, Operation* op,
                               RankedTensorType output_type, Value input_value,
                               ElementsAttr axes_elems, bool keep_dims);

// Lowers ResizeBilinear and ResizeNearestNeighbor to TOSA resize.
Operation* convertResizeOp(PatternRewriter& rewriter, Operation* op,
                           RankedTensorType output_type, Value input_value,
                           StringRef mode);

// Lowers Quantize to a sequence of TOSA quantization ops.
Operation* convertQuantizeOp(PatternRewriter& rewriter, Operation* op,
                             RankedTensorType output_type, Value input_value,
                             double scale, int64_t zeropoint);

// Lowers Dequantize to a sequence of TOSA dequantization ops.
Operation* convertDequantizeOp(PatternRewriter& rewriter, Operation* op,
                               RankedTensorType output_type, Value input_value,
                               double scale, int64_t zeropoint);

// Lowers FakeQuant to a sequence of TOSA quantization ops.
Operation* convertFakeQuantOp(PatternRewriter& rewriter, Operation* op,
                              RankedTensorType output_type, Value input_value,
                              double min, double max, int64_t num_bits,
                              bool narrow_range);
Operation* convertTFConv2DCommon(
    PatternRewriter& rewriter, Operation* op, RankedTensorType output_type,
    Value input, Value filter, Value bias, ArrayAttr strides_attr,
    ArrayAttr dilations_attr, ArrayAttr explicit_padding_attr,
    StringRef padding_ref, StringRef data_format_ref);

};  // namespace tosa
};  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TOSA_TRANSFORMS_LEGALIZE_COMMON_H
