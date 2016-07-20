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

#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace shape_inference {

// Transfers shape of input(0) to output(0).
Status UnchangedShape(shape_inference::InferenceContext* c);

// Transfers shape of input(0) to output(0), after asserting its rank is <rank>.
inline Status UnchangedShapeWithRank(shape_inference::InferenceContext* c,
                                     int32 rank) {
  const Shape* out;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), rank, &out));
  c->set_output(0, out);
  return Status::OK();
}

// Transfers shape of input(0) to output(0), after asserting its rank >= <rank>.
inline Status UnchangedShapeWithRankAtLeast(
    shape_inference::InferenceContext* c, int32 rank) {
  const Shape* out;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), rank, &out));
  c->set_output(0, out);
  return Status::OK();
}

// Transfers shape of input(0) to output(0), after asserting its rank <= <rank>.
inline Status UnchangedShapeWithRankAtMost(shape_inference::InferenceContext* c,
                                           int32 rank) {
  const Shape* out;
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
  const Shape* out;
  TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(1), &out));
  c->set_output(0, out);
  return Status::OK();
}

inline Status MatMulShape(shape_inference::InferenceContext* c) {
  const Shape* a;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a));

  const Shape* b;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &b));

  bool transpose_a, transpose_b;
  TF_RETURN_IF_ERROR(c->GetAttr("transpose_a", &transpose_a));
  TF_RETURN_IF_ERROR(c->GetAttr("transpose_b", &transpose_b));
  const Dimension* output_rows = transpose_a ? c->Dim(a, 1) : c->Dim(a, 0);
  const Dimension* output_cols = transpose_b ? c->Dim(b, 0) : c->Dim(b, 1);

  // Validate that the inner shapes are compatible.
  const Dimension* inner_a = transpose_a ? c->Dim(a, 0) : c->Dim(a, 1);
  const Dimension* inner_b = transpose_b ? c->Dim(b, 1) : c->Dim(b, 0);
  const Dimension* merged;
  TF_RETURN_IF_ERROR(c->Merge(inner_a, inner_b, &merged));

  c->set_output(0, c->Matrix(output_rows, output_cols));
  return Status::OK();
}

inline Status BiasAddShape(shape_inference::InferenceContext* c) {
  const Shape* input_shape;

  // Fetch the data_format attribute, which may not exist.
  string data_format;
  Status s = c->GetAttr("data_format", &data_format);

  if (s.ok() && data_format == "NCHW") {
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 4, &input_shape));
  } else {
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input_shape));
  }

  const Shape* bias_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &bias_shape));
  const Dimension* bias_dim = c->Dim(bias_shape, 0);

  // If rank unknown, return unknown shape.
  if (!c->RankKnown(input_shape)) {
    c->set_output(0, c->UnknownShape());
    return Status::OK();
  }

  const int32 rank = c->Rank(input_shape);

  // Output has the same shape as the input, and matches the length of
  // the bias in its bias dimension.
  const Shape* output_shape;
  if (s.ok() && data_format == "NCHW") {
    // Merge the length of bias_shape into the third to last dimension
    const Shape* first;
    TF_RETURN_IF_ERROR(c->Subshape(input_shape, 0, -3, &first));

    const Shape* last;
    TF_RETURN_IF_ERROR(c->Subshape(input_shape, -2, &last));

    const Dimension* input_bias_dim = c->Dim(input_shape, -3);
    const Dimension* merged_bias_dim;
    TF_RETURN_IF_ERROR(c->Merge(input_bias_dim, bias_dim, &merged_bias_dim));
    const Shape* merged_bias = c->Vector(merged_bias_dim);

    const Shape* temp;
    TF_RETURN_IF_ERROR(c->Concatenate(first, merged_bias, &temp));
    TF_RETURN_IF_ERROR(c->Concatenate(temp, last, &output_shape));
  } else {
    const Shape* all_but_bias;
    TF_RETURN_IF_ERROR(c->Subshape(input_shape, 0, -1, &all_but_bias));

    const Dimension* input_bias_dim = c->Dim(input_shape, -1);
    const Dimension* merged_bias_dim;
    TF_RETURN_IF_ERROR(c->Merge(input_bias_dim, bias_dim, &merged_bias_dim));

    const Shape* merged_bias = c->Vector(merged_bias_dim);
    TF_RETURN_IF_ERROR(
        c->Concatenate(all_but_bias, merged_bias, &output_shape));
  }

  c->set_output(0, output_shape);
  return Status::OK();
}

}  // namespace shape_inference
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_OPS_COMMON_SHAPE_FNS_H_
