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
#ifndef THIRD_PARTY_TENSORFLOW_CORE_OPS_COMMON_SHAPE_FNS_H_
#define THIRD_PARTY_TENSORFLOW_CORE_OPS_COMMON_SHAPE_FNS_H_

#include <array>

#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/padding.h"

namespace tensorflow {

// GetWindowedOutputSize(): Given an input tensor, kernel, stride and padding
// type, the function computes the output and padding dimensions.
//
// For example, ignoring batches or multiple features, a 1D convolution
// takes as input a 1D tensor of shape (H), and convolves it with a filter of
// shape (K).
//
// It also takes in a few additional parameters:
//
// Stride (S): the stride with which we apply the filters. This is the offset
// between locations where we apply the filters. A larger stride
// means that the output will be spatially smaller.
//
// Padding (P): the padding we apply to the input tensor along each
// dimension. This is usually used to make sure that the spatial dimensions
// do not shrink when we progress with convolutions. Two types of padding are
// often used:
//   SAME: the pad value is computed so that the output will have size H/S.
//   VALID: no padding is carried out.
// The padded area is zero-filled.
//
// The output dimensions for convolution and many other operations, when given
// all the parameters above, are as follows:
// - When Padding = SAME: the output size is (H'), where
//     H' = ceil(float(H) / float(S))
//   where ceil is the ceiling function. The number of padded cells
//   is computed as:
//     Pc = ((H' - 1) * S + K - H) / 2
//   When the stride is 1, the expression simplifies to
//     H' = H, Pc = (K-1)/2.
//   This is where SAME comes from - the output has the same size as the input
//   has.
//
// - When Padding = VALID: the output size is computed as
//     H' = ceil(float(H - K + 1) / float(S))
//   and the number of padded cells is always zero.
//   When the stride is 1, the expression simplifies to
//     H' = H-K+1.
//
// For convolution, mathematically, the output value at location (r')
// is the inner product of two vectors: the chunk of input at
//    ((r'*S-Pr) : (r'*S-Pr+K)),
// and the filter.
//
// For 2D and 3D convolutions, the spatial dimensions are orthogonal, so the
// size and padding of each spatial dimension can be computed by calling
// GetWindowedOutputSize separately for each dimension.
//
Status GetWindowedOutputSize(int64 input_size, int64 filter_size, int64 stride,
                             Padding padding_type, int64* output_size,
                             int64* padding_size);

// Returns the same output dimensions as in GetWindowedOutputSize, but returns
// verbose padding dimensions (before/after). Any excess padding
// (caused by an odd padding size value) is added to the 'padding_after'
// dimension.
Status GetWindowedOutputSizeVerbose(int64 input_size, int64 filter_size,
                                    int64 stride, Padding padding_type,
                                    int64* output_size, int64* padding_before,
                                    int64* padding_after);

// Given an input tensor, kernel, stride and padding type, populates the 3D size
// of the output tensor and padding to be applied to the input tensor at the
// lower end of every dimension. Use for 3D convolutions, where the input data
// is padded with zeros, as well as for 3D avg/max pooling, where the input data
// is padded with invalid values that are not considered for pooling.
Status Get3dOutputSize(const std::array<int64, 3>& input,
                       const std::array<int64, 3>& window,
                       const std::array<int64, 3>& strides,
                       Padding padding_type, std::array<int64, 3>* output,
                       std::array<int64, 3>* padding);

namespace shape_inference {

// Like GetWindowedOutputSize, but deals with DimensionHandles.
Status GetWindowedOutputSizeFromDims(InferenceContext* c,
                                     DimensionHandle input_size,
                                     DimensionOrConstant filter_size,
                                     int64 stride, Padding padding_type,
                                     DimensionHandle* output_size);

// Transfers shape of input(0) to output(0).
Status UnchangedShape(shape_inference::InferenceContext* c);

// Transfers shape of input(0) to output(0), after asserting its rank is <rank>.
inline Status UnchangedShapeWithRank(shape_inference::InferenceContext* c,
                                     int32 rank) {
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), rank, &out));
  c->set_output(0, out);
  return Status::OK();
}

// Transfers shape of input(0) to output(0), after asserting its rank >= <rank>.
inline Status UnchangedShapeWithRankAtLeast(
    shape_inference::InferenceContext* c, int32 rank) {
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), rank, &out));
  c->set_output(0, out);
  return Status::OK();
}

// Transfers shape of input(0) to output(0), after asserting its rank <= <rank>.
inline Status UnchangedShapeWithRankAtMost(shape_inference::InferenceContext* c,
                                           int32 rank) {
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), rank, &out));
  c->set_output(0, out);
  return Status::OK();
}

// Shape function for use with ops no outputs.
inline Status NoOutputs(shape_inference::InferenceContext* c) {
  return Status::OK();
}

// Shape function for ops that output a single scalar value.
inline Status ScalarShape(shape_inference::InferenceContext* c) {
  c->set_output(0, c->Scalar());
  return Status::OK();
}

// Shape function for binary ops where both inputs and the output match.
inline Status MergeBothInputsShapeFn(InferenceContext* c) {
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(1), &out));
  c->set_output(0, out);
  return Status::OK();
}

// Shape function for MatMul-like operations.
Status MatMulShape(shape_inference::InferenceContext* c);

// Shape function for BiasAdd-like operations.
Status BiasAddShape(shape_inference::InferenceContext* c);

// Shape function for BiasAddGrad-like operations.
Status BiasAddGradShape(shape_inference::InferenceContext* c);

// Shape function for Conv2D-like operations.
Status Conv2DShape(shape_inference::InferenceContext* c);

// Shape function for Conv3D-like operations.
Status Conv3DShape(shape_inference::InferenceContext* c);

// Shape function for DepthwiseConv2D-like operations.
Status DepthwiseConv2DNativeShape(shape_inference::InferenceContext* c);

// Shape function for AvgPool-like operations.
Status AvgPoolShape(shape_inference::InferenceContext* c);

// Shape function for MaxPool-like operations.
Status MaxPoolShape(shape_inference::InferenceContext* c);

// Shape function for 3D Pooling operations.
Status Pool3DShape(shape_inference::InferenceContext* c);

// Shape function for use with ops whose output shapes are unknown.
Status UnknownShape(shape_inference::InferenceContext* c);

// Shape function for reduction operations.
Status ReductionShape(shape_inference::InferenceContext* c);

// Shape function for concat operations.
// <num_inputs_to_concat> is the number of inputs to concatenate and are taken
// from inputs
// [1,num_inputs_to_concat] of the op.  Input 0 is the concat_dim input.
Status ConcatShape(shape_inference::InferenceContext* c,
                   int num_inputs_to_concat);

// Shape function for concat operations.
Status ConcatV2Shape(shape_inference::InferenceContext* c);

// Shape function for binary operators that broadcast their inputs.
// Tested by ops/math_ops_test.cc.
Status BroadcastBinaryOpShapeFn(InferenceContext* c);

// Shape function for random operations.
Status RandomShape(shape_inference::InferenceContext* c);

// Validates the 3 component tensors of a sparse tensor have the proper
// shapes. This mimics SparseTensor.__init__ in python/framework/ops.py.
Status ValidateSparseTensor(InferenceContext* c, ShapeHandle indices_shape,
                            ShapeHandle values_shape, ShapeHandle shape_shape);

}  // namespace shape_inference

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_OPS_COMMON_SHAPE_FNS_H_
