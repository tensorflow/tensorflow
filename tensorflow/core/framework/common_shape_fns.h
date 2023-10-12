/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_FRAMEWORK_COMMON_SHAPE_FNS_H_
#define TENSORFLOW_CORE_FRAMEWORK_COMMON_SHAPE_FNS_H_

#include <array>

#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

namespace shape_inference {

// Like GetWindowedOutputSize, but deals with DimensionHandles. Does not support
// EXPLICIT padding.
Status GetWindowedOutputSizeFromDims(InferenceContext* c,
                                     DimensionHandle input_size,
                                     DimensionOrConstant filter_size,
                                     int64_t stride, Padding padding_type,
                                     DimensionHandle* output_size);

// The V2 version computes the same outputs with arbitrary dilation_rate, and
// supports EXPLICIT padding. For detailed equations, refer to the comments
// for GetWindowedOutputSize(). The 'padding_before' and 'padding_after'
// parameters are only used if padding_type == EXPLICIT.
Status GetWindowedOutputSizeFromDimsV2(
    InferenceContext* c, DimensionHandle input_size,
    DimensionOrConstant filter_size, int64_t dilation_rate, int64_t stride,
    Padding padding_type, int64_t padding_before, int64_t padding_after,
    DimensionHandle* output_size);

// Transfers shape of input(0) to output(0).
Status UnchangedShape(shape_inference::InferenceContext* c);

// Transfers shape of input(0) to output(0), after asserting its rank is <rank>.
inline Status UnchangedShapeWithRank(shape_inference::InferenceContext* c,
                                     int32_t rank) {
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), rank, &out));
  c->set_output(0, out);
  return OkStatus();
}

// Transfers shape of input(0) to output(0), after asserting its rank >= <rank>.
inline Status UnchangedShapeWithRankAtLeast(
    shape_inference::InferenceContext* c, int32_t rank) {
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), rank, &out));
  c->set_output(0, out);
  return OkStatus();
}

// Transfers shape of input(0) to output(0), after asserting its rank <= <rank>.
inline Status UnchangedShapeWithRankAtMost(shape_inference::InferenceContext* c,
                                           int32_t rank) {
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), rank, &out));
  c->set_output(0, out);
  return OkStatus();
}

// Shape function for use with ops no outputs.
inline Status NoOutputs(shape_inference::InferenceContext* c) {
  return OkStatus();
}

// Shape function for ops that output a single scalar value.
inline Status ScalarShape(shape_inference::InferenceContext* c) {
  c->set_output(0, c->Scalar());
  return OkStatus();
}

// Shape function for binary ops where both inputs and the output match.
inline Status MergeBothInputsShapeFn(InferenceContext* c) {
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(1), &out));
  c->set_output(0, out);
  return OkStatus();
}

// Shape function for dataset iterators.
Status DatasetIteratorShape(shape_inference::InferenceContext* c);

// Returns a new shape with the specified dims arranged in the specified
// format. The returned value is owned by this context.
// Note: if format = "FORMAT_NCHW_VECT_C" then C represents the outer_depth.
Status MakeShapeFromFormat(TensorFormat format, DimensionOrConstant N,
                           const std::vector<DimensionOrConstant>& spatial,
                           DimensionOrConstant C, ShapeHandle* out,
                           shape_inference::InferenceContext* context);

// Shape function for MatMul-like operations.
Status MatMulShape(shape_inference::InferenceContext* c);

// Shape function for Batched MatMul-like operations with broadcasting across
// batch dimensions.
Status BatchMatMulV2Shape(shape_inference::InferenceContext* c);

// Shape function for BatchMatMul-like operations
Status BatchMatMulShape(shape_inference::InferenceContext* c);

// Shape function for Einsum.
Status EinsumShape(shape_inference::InferenceContext* c);

// Shape function for BiasAdd-like operations.
Status BiasAddShape(shape_inference::InferenceContext* c);

// Shape function for BiasAddGrad-like operations.
Status BiasAddGradShape(shape_inference::InferenceContext* c);

// Shape function for general Convolution operation
Status ConvShape(shape_inference::InferenceContext* c);

// Shape function for Conv2D-like operations that support explicit padding.
Status Conv2DShapeWithExplicitPadding(shape_inference::InferenceContext* c);

// Shape function for Conv2D-like operations that do not support explicit
// padding.
Status Conv2DShape(shape_inference::InferenceContext* c);

// Shape function for Conv3D-like operations.
Status Conv3DShape(shape_inference::InferenceContext* c);

// Shape function for DepthwiseConv2D-like operations that support explicit
// padding.
Status DepthwiseConv2DNativeShapeWithExplicitPadding(
    shape_inference::InferenceContext* c);

// Shape function for DepthwiseConv2D-like operations that do not support
// explicit padding.
Status DepthwiseConv2DNativeShape(shape_inference::InferenceContext* c);

// Shape function for Conv2DBackpropInput.
Status Conv2DBackpropInputShape(shape_inference::InferenceContext* c);

// Shape function for Conv2DBackpropFilterWithBias.
Status Conv2DBackpropFilterWithBiasShape(shape_inference::InferenceContext* c);

// Shape function for AvgPool-like operations.
Status AvgPoolShape(shape_inference::InferenceContext* c);

// Shape function for AvgPoolGrad-like operations.
Status AvgPoolGradShape(shape_inference::InferenceContext* c);

// Shape function for FusedBatchNorm and FusedBatchNormV2 operations.
Status FusedBatchNormShape(shape_inference::InferenceContext* c);

// Shape function for FusedBatchNormV3 operations.
Status FusedBatchNormV3Shape(shape_inference::InferenceContext* c);

// Shape function for _FusedBatchNormEx operations.
Status FusedBatchNormExShape(shape_inference::InferenceContext* c);

// Shape function for FusedBatchNormGrad and FusedBatchNormGradV2 operations.
Status FusedBatchNormGradShape(shape_inference::InferenceContext* c);

// Shape function for _FusedBatchNormGradEx operations.
Status FusedBatchNormGradExShape(shape_inference::InferenceContext* c);

// Shape function for MatrixDiagPartV2 and MatrixDiagPartV3 operations.
Status MatrixDiagPartV2Shape(shape_inference::InferenceContext* c);

// Shape function for MatrixDiagV2 and MatrixDiagV3 operations.
Status MatrixDiagV2Shape(shape_inference::InferenceContext* c);

// Shape function for MatrixSetDiagV2 and MatrixSetDiagV3 operations.
Status MatrixSetDiagV2Shape(shape_inference::InferenceContext* c);

// Shape function for MaxPool-like operations that support explicit padding.
Status MaxPoolShapeWithExplicitPadding(shape_inference::InferenceContext* c);

// Shape function for MaxPool-like operations that do not support explicit
// padding.
Status MaxPoolShape(shape_inference::InferenceContext* c);

// Shape function for MaxPoolV2-like operations.
Status MaxPoolV2Shape(shape_inference::InferenceContext* c, int num_inputs);

// Shape function for MaxPoolGrad-like operations.
Status MaxPoolGradShape(shape_inference::InferenceContext* c);

// Shape function for 3D Pooling operations.
Status Pool3DShape(shape_inference::InferenceContext* c);

// Shape function for MaxPool3DGrad-like operations.
Status MaxPool3DGradShape(shape_inference::InferenceContext* c);

// Shape function for AvgPool3DGrad-like operations.
Status AvgPool3DGradShape(shape_inference::InferenceContext* c);

// Shape function for use with ops whose output shapes are unknown.
Status UnknownShape(shape_inference::InferenceContext* c);

// Shape function for reduction operations.
Status ReductionShape(shape_inference::InferenceContext* c);

// Shape function for unsorted segment operations.
Status SegmentReductionWithNumSegmentsShapeFn(InferenceContext* c);

// Shape function for concat operations.
// <num_inputs_to_concat> is the number of inputs to concatenate and are taken
// from inputs
// [1,num_inputs_to_concat] of the op.  Input 0 is the concat_dim input.
Status ConcatShape(shape_inference::InferenceContext* c,
                   int num_inputs_to_concat);

// Shape function for concat operations.
Status ConcatV2Shape(shape_inference::InferenceContext* c);

Status QuantizedConcatV2Shape(InferenceContext* c, int num_inputs_to_concat);

// Shape function for binary operators that broadcast their inputs
// and with output to output_index.
// Note: out cannot be NULL.
Status BroadcastBinaryOpOutputShapeFnHelper(InferenceContext* c,
                                            ShapeHandle shape_x,
                                            ShapeHandle shape_y,
                                            bool incompatible_shape_error,
                                            ShapeHandle* out);

// Shape function for binary operators that broadcast their inputs
// and with output to output_index.
inline Status BroadcastBinaryOpOutputShapeFn(InferenceContext* c,
                                             int output_index) {
  ShapeHandle out;
  TF_RETURN_IF_ERROR(BroadcastBinaryOpOutputShapeFnHelper(
      c, c->input(0), c->input(1), true, &out));
  c->set_output(output_index, out);
  return OkStatus();
}

// Shape function for binary operators that broadcast their inputs.
// Tested by ops/math_ops_test.cc.
inline Status BroadcastBinaryOpShapeFn(InferenceContext* c) {
  return BroadcastBinaryOpOutputShapeFn(c, 0);
}

// Shape function for random operations.
Status RandomShape(shape_inference::InferenceContext* c);

// Shape function for Slice operations.
Status SliceShape(shape_inference::InferenceContext* c);

// Validates the 3 component tensors of a sparse tensor have the proper
// shapes. This mimics SparseTensor.__init__ in python/framework/ops.py.
Status ValidateSparseTensor(InferenceContext* c, ShapeHandle indices_shape,
                            ShapeHandle values_shape, ShapeHandle shape_shape);

Status ValidateVariableResourceHandle(
    InferenceContext* c, std::vector<ShapeAndType>* shape_and_type);

// Shape function for GatherNd operations.
Status GatherNdShape(InferenceContext* c);

// Helper shape function for ScatterNd.../TensorScatter... operations.
Status ScatterNdShapeHelper(InferenceContext* c, ShapeHandle indices_shape,
                            ShapeHandle updates_shape, ShapeHandle input_shape);

// Shape function for ops with an explicit "shape" attribute.
Status ExplicitShape(InferenceContext* c);

// Shape function for multiple-output ops with an explicit "shapes" attribute.
Status ExplicitShapes(InferenceContext* c);

// Shape function for SparseReduceMax and SparseReduceSum.
Status SparseReduceShapeFn(InferenceContext* c);

// Shape function for QuantizedConv2D op.
Status QuantizedConv2DShape(InferenceContext* c);

// Shape function for _QuantizedConv2D op/fusion.
Status FusedQuantizedConv2DShape(InferenceContext* c);

// Shape function for _QuantizedDepthwiseConv2D op/fusion.
Status FusedQuantizedDepthwiseConv2D(InferenceContext* c);

// Shape function for QuantizedAvgPool op
Status QuantizedAvgPoolShape(InferenceContext* c);

// Shape function for QuantizeV2 op
Status QuantizeV2Shape(InferenceContext* c);

// Shape function for ReduceScatter ops
Status ReduceScatterShape(shape_inference::InferenceContext* c);

}  // namespace shape_inference

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_COMMON_SHAPE_FNS_H_
