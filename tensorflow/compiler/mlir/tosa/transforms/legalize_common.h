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

#ifndef TENSORFLOW_COMPILER_MLIR_TOSA_TRANSFORMS_LEGALIZE_COMMON_H_
#define TENSORFLOW_COMPILER_MLIR_TOSA_TRANSFORMS_LEGALIZE_COMMON_H_

#include <optional>

#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

// This file contains legalizations common to mapping both TensorFlow and
// TensorFlow Lite to TOSA.
//
// Conversion functions return None on a failure or result value on success.
// Callers must check and return a LogicalResult failure on nullptr.
//
// For these functions, the framework-specific operands/attributes/defaults
// are already extracted and placed in a common form for lowering.

namespace mlir {
namespace tosa {

// Lowers the Pack operator to TOSA.
std::optional<Value> convertPackOp(PatternRewriter& rewriter, Operation* op,
                                   Value result_value,
                                   SmallVectorImpl<Value>& inputs,
                                   int32_t axis);

// Lowers the Unpack operator to TOSA.
std::optional<SmallVector<Value>> convertUnpackOp(PatternRewriter& rewriter,
                                                  Operation* op,
                                                  Value input_value,
                                                  int32_t axis);

// Lowers the Select operator to TOSA.
std::optional<Value> convertSelectOp(PatternRewriter& rewriter, Operation* op,
                                     Value result_value, Value condition_value,
                                     Value x_value, Value y_value);

// Lowers the ZerosLike operator to TOSA by creating a constant
// of the desired type and shape.
std::optional<Value> convertZerosLikeOp(PatternRewriter& rewriter,
                                        Operation* op, Value result,
                                        Value input);

// Lowers the Mul operator to TOSA.  For quantized types, this requires
// inserting rescale operators before and after the operation.
std::optional<Value> convertMultiplyOp(PatternRewriter& rewriter, Operation* op,
                                       Value output_val, Value input_lhs_val,
                                       Value input_rhs_val);

// Lowers the SquaredDifference operator to TOSA.
std::optional<Value> convertSquaredDifferenceOp(PatternRewriter& rewriter,
                                                Operation* op, Value result,
                                                Value x, Value y);

// Lowers the Round operator to TOSA.
std::optional<Value> convertRoundOp(PatternRewriter& rewriter, Operation* op,
                                    Value result, Value input);

// Lowers ConcatV2 to TOSA.
std::optional<Value> convertConcatV2Op(PatternRewriter& rewriter, Operation* op,
                                       ShapedType result_type,
                                       SmallVectorImpl<Value>& values,
                                       int32_t axis);

// Lowers SpaceToBatchND to TOSA.
std::optional<Value> convertSpaceToBatchNDOp(PatternRewriter& rewriter,
                                             Operation* op, Value result_value,
                                             Value input_value,
                                             Value block_shape_value,
                                             Value paddings_value);

// Lowers BatchToSpaceND to TOSA.
std::optional<Value> convertBatchToSpaceNDOp(PatternRewriter& rewriter,
                                             Operation* op, Value result_value,
                                             Value input_value,
                                             Value block_shape_value,
                                             Value crops_value);

// Lowers ExpandDims to TOSA.
std::optional<Value> convertExpandDimsOp(PatternRewriter& rewriter,
                                         Operation* op, Value result_value,
                                         Value input_value, Value dim_value);

// Lowers Squeeze to TOSA.
std::optional<Value> convertSqueezeOp(PatternRewriter& rewriter, Operation* op,
                                      Value result_value, Value input_value,
                                      SmallVectorImpl<int32_t>& squeeze_dims);

// Lowers ELU to a sequence of TOSA ops.
std::optional<Value> convertEluOp(PatternRewriter& rewriter, Operation* op,
                                  Value result_value, Value features_value);

// Lowers Softmax to a sequence of TOSA ops.
std::optional<Value> convertSoftmaxOp(PatternRewriter& rewriter, Operation* op,
                                      Value result_value, Value logits_value,
                                      double beta);

// Lowers LogSoftmax to a sequence of TOSA ops.
std::optional<Value> convertLogSoftmaxOp(PatternRewriter& rewriter,
                                         Operation* op, Value result_value,
                                         Value logits_value);

// Lowers SpaceToDepth to a sequence of TOSA ops.  Supports NHWC.
std::optional<Value> convertSpaceToDepthOp(PatternRewriter& rewriter,
                                           Operation* op, Value result_value,
                                           Value input_value,
                                           IntegerAttr block_size_attr,
                                           StringAttr data_format);

// Lowers DepthToSpace to a sequence of TOSA ops.  Supports NHWC.
std::optional<Value> convertDepthToSpaceOp(PatternRewriter& rewriter,
                                           Operation* op, Value result_value,
                                           Value input_value,
                                           IntegerAttr block_size_attr,
                                           StringAttr data_format);

// Lowers Split to a sequence of TOSA ops.
std::optional<SmallVector<Value>> convertSplitOp(
    PatternRewriter& rewriter, Operation* op, Value result_value,
    Value input_value, int32_t num_split, int32_t axis);

// Lowers SplitV to a sequence of TOSA ops.
std::optional<SmallVector<Value>> convertSplitVOp(
    PatternRewriter& rewriter, Operation* op, Value result_value,
    Value input_value, SmallVectorImpl<int32_t>& size_split, int32_t axis);

// Lowers StridedSlice to a sequence of TOSA ops.
std::optional<Value> convertStridedSliceOp(
    PatternRewriter& rewriter, Operation* op, Value result_value,
    Value input_value, Value begin_value, Value end_value, Value strides_value,
    int32_t begin_mask, int32_t end_mask, int32_t ellipsis_mask,
    int32_t new_axis_mask, int32_t shrink_axis_mask);

// Lowers FloorDiv to a sequence of TOSA operators.
std::optional<Value> convertFloorDivOp(PatternRewriter& rewriter, Operation* op,
                                       Value result_value, Value lhs_value,
                                       Value rhs_value);

// Lowers FloorMod to a sequence of TOSA operators.
std::optional<Value> convertFloorModOp(PatternRewriter& rewriter, Operation* op,
                                       Value result_value, Value lhs_value,
                                       Value rhs_value);

// Lowers FusedActivation to a sequence of TOSA ops.
std::optional<Value> convertFusedActivation(PatternRewriter& rewriter,
                                            Operation* op, Value input_value,
                                            StringAttr fused_activation_fn);

// Helper function for implementing quantized divide by power-of-two in TOSA
// ops.
std::optional<Value> convertRoundingDivideByPOT(PatternRewriter& rewriter,
                                                Operation* op,
                                                Value input_value,
                                                Value rshift_value);

// Lowers ReduceAll to a sequence of TOSA ops.
std::optional<Value> convertReduceAllOp(PatternRewriter& rewriter,
                                        Operation* op,
                                        RankedTensorType output_type,
                                        Value input_value,
                                        ElementsAttr axes_elems);

// Lowers ReduceAny to a sequence of TOSA ops.
std::optional<Value> convertReduceAnyOp(PatternRewriter& rewriter,
                                        Operation* op,
                                        RankedTensorType output_type,
                                        Value input_value,
                                        ElementsAttr axes_elems);

// Lowers ReduceMin to a sequence of TOSA ops.
std::optional<Value> convertReduceMinOp(PatternRewriter& rewriter,
                                        Operation* op,
                                        RankedTensorType output_type,
                                        Value input_value,
                                        ElementsAttr axes_elems);

// Lowers ReduceMax to a sequence of TOSA ops.
std::optional<Value> convertReduceMaxOp(PatternRewriter& rewriter,
                                        Operation* op,
                                        RankedTensorType output_type,
                                        Value input_value,
                                        ElementsAttr axes_elems);

// Lowers ReduceProd to a sequence of TOSA ops.
std::optional<Value> convertReduceProdOp(PatternRewriter& rewriter,
                                         Operation* op,
                                         RankedTensorType output_type,
                                         Value input_value,
                                         ElementsAttr axes_elems);

// Lowers ReduceSum to a sequence of TOSA ops.
std::optional<Value> convertReduceSumOp(PatternRewriter& rewriter,
                                        Operation* op,
                                        RankedTensorType output_type,
                                        Value input_value,
                                        ElementsAttr axes_elems);

// Lowers ReduceMean to a sequence of TOSA ops.
std::optional<Value> convertReduceMeanOp(PatternRewriter& rewriter,
                                         Operation* op,
                                         RankedTensorType output_type,
                                         Value input_value,
                                         ElementsAttr axes_elem);

// Lowers ResizeBilinear and ResizeNearestNeighbor to TOSA resize.
std::optional<Value> convertResizeOp(PatternRewriter& rewriter, Operation* op,
                                     RankedTensorType output_type,
                                     Value input_value, StringRef mode,
                                     bool align_corners,
                                     bool half_pixel_centers);

// Lowers Quantize to a sequence of TOSA quantization ops.
std::optional<Value> convertQuantizeOp(PatternRewriter& rewriter, Operation* op,
                                       ShapedType output_type,
                                       Value input_value, double scale,
                                       int64_t zeropoint);

// Lowers Dequantize to a sequence of TOSA dequantization ops.
std::optional<Value> convertDequantizeOp(PatternRewriter& rewriter,
                                         Operation* op, ShapedType output_type,
                                         Value input_value,
                                         ArrayRef<float> scale,
                                         ArrayRef<float> zeropoint,
                                         int64_t dim);

// Lowers FakeQuant to a sequence of TOSA quantization ops.
std::optional<Value> convertFakeQuantOp(PatternRewriter& rewriter,
                                        Operation* op, ShapedType output_type,
                                        Value input_value, double min,
                                        double max, int64_t num_bits,
                                        bool narrow_range);

// Align to TF_MirrorPadOp::mode and TFL_MirrorPadOp::mode
enum class TFTFLMirrorPaddingType : uint32_t {
  REFLECT = 0,
  SYMMETRIC = 1,
};

std::optional<Value> convertMirrorPadCommon(PatternRewriter& rewriter,
                                            Operation* op,
                                            RankedTensorType output_type,
                                            Value input, Value pad,
                                            TFTFLMirrorPaddingType mode);

// Lowers TensorFlow and TensorFlow Lite Conv3D to a sequence of TOSA
// quantization ops.
std::optional<Value> convertConv3DCommon(PatternRewriter& rewriter,
                                         Operation* op, ShapedType output_type,
                                         Value input, Value filter, Value bias,
                                         DenseI64ArrayAttr pads,
                                         DenseI64ArrayAttr strides,
                                         DenseI64ArrayAttr dilations,
                                         StringRef data_format_ref);

// Lowers Gather operator to a sequence of TOSA ops.
std::optional<Value> convertGatherOp(PatternRewriter& rewriter, Operation* op,
                                     Value result_value, Value params_value,
                                     Value indices_value, int32_t batch_dims,
                                     int32_t axis);

// Lowers GatherNd operator to a sequence of TOSA ops.
std::optional<Value> convertGatherNdOp(PatternRewriter& rewriter, Operation* op,
                                       Value result_value, Value params_value,
                                       Value indices_value);

// Lowers OneHot operator to a sequence of TOSA ops.
std::optional<Value> convertOneHotOp(PatternRewriter& rewriter, Operation* op,
                                     Value result_value, Value indices_value,
                                     Value on_value, Value off_value,
                                     int32_t depth, int32_t axis);

// Lowers 32-bit floating sin operator to a sequence of TOSA ops.
std::optional<Value> convertSinOp(PatternRewriter& rewriter, Operation* op,
                                  Value input, ShapedType output_type);

// Lowers Sign operator to a sequence of TOSA ops.
std::optional<Value> convertSignOp(PatternRewriter& rewriter, Operation* op,
                                   Value input, RankedTensorType output_type);

};  // namespace tosa
};  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TOSA_TRANSFORMS_LEGALIZE_COMMON_H_
