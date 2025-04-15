/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/rng_alg.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

absl::Status StatefulRandomShape(shape_inference::InferenceContext* c) {
  using shape_inference::ShapeHandle;
  // Check algorithm shape
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
  // Set output shape
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &out));
  c->set_output(0, out);
  return absl::OkStatus();
}

#define REGISTER_STATEFUL_OP(name, default_dtype) \
  REGISTER_OP(name)                               \
      .Input("resource: resource")                \
      .Input("algorithm: int64")                  \
      .Input("shape: shape_dtype")                \
      .Output("output: dtype")                    \
      .Attr("dtype : type = " #default_dtype)     \
      .Attr("shape_dtype : type = DT_INT64")      \
      .SetShapeFn(StatefulRandomShape);

REGISTER_STATEFUL_OP("StatefulUniform", DT_FLOAT);
REGISTER_STATEFUL_OP("StatefulUniformFullInt", DT_UINT64);
REGISTER_STATEFUL_OP("StatefulStandardNormalV2", DT_FLOAT);
REGISTER_STATEFUL_OP("StatefulTruncatedNormal", DT_FLOAT);

REGISTER_OP("StatefulUniformInt")
    .Input("resource: resource")
    .Input("algorithm: int64")
    .Input("shape: shape_dtype")
    .Input("minval: dtype")
    .Input("maxval: dtype")
    .Output("output: dtype")
    .Attr("dtype : type = DT_INT64")
    .Attr("shape_dtype : type = DT_INT64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      using shape_inference::ShapeHandle;
      // Check inputs
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      absl::Status s = c->WithRank(c->input(3), 0, &unused);
      if (!s.ok()) {
        return errors::InvalidArgument(
            "minval must be a scalar; got a tensor of shape ",
            c->DebugString(c->input(3)));
      }
      s = c->WithRank(c->input(4), 0, &unused);
      if (!s.ok()) {
        return errors::InvalidArgument(
            "maxval must be a scalar; got a tensor of shape ",
            c->DebugString(c->input(4)));
      }
      // Set output
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &out));
      c->set_output(0, out);
      return absl::OkStatus();
    });

REGISTER_OP("RngSkip")
    .Input("resource: resource")
    .Input("algorithm: int64")
    .Input("delta: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      return absl::OkStatus();
    });

REGISTER_OP("RngReadAndSkip")
    .Input("resource: resource")
    .Input("alg: int32")
    .Input("delta: uint64")
    .Output("value: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(0, c->MakeShape({RNG_MAX_COUNTER_SIZE + RNG_KEY_SIZE}));
      return absl::OkStatus();
    });

REGISTER_OP("NonDeterministicInts")
    .Input("shape: shape_dtype")
    .SetIsStateful()
    .Output("output: dtype")
    .Attr("dtype : type = DT_INT64")
    .Attr("shape_dtype : type = DT_INT64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      using shape_inference::ShapeHandle;
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
      c->set_output(0, out);
      return absl::OkStatus();
    });

REGISTER_OP("StatefulRandomBinomial")
    .Input("resource: resource")
    .Input("algorithm: int64")
    .Input("shape: S")
    .Input("counts: T")
    .Input("probs: T")
    .Output("output: dtype")
    .Attr("S: {int32, int64}")
    .Attr("T: {half, float, double, int32, int64} = DT_DOUBLE")
    .Attr("dtype: {half, float, double, int32, int64} = DT_INT64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      using shape_inference::ShapeHandle;

      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &out));
      c->set_output(0, out);
      return absl::OkStatus();
    });

// Register the deprecated 'StatefulStandardNormal' op. This op is a short-lived
// version where the 'resource' variable also contains the algorithm tag.
// It is deprecated in favor of 'StatefulStandardNormalV2'.
REGISTER_OP("StatefulStandardNormal")
    .Deprecated(29, "Use StatefulStandardNormalV2 instead")
    .Input("resource: resource")
    .Input("shape: shape_dtype")
    .Output("output: dtype")
    .Attr("dtype : type = DT_FLOAT")
    .Attr("shape_dtype : type = DT_INT64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      using shape_inference::ShapeHandle;
      // Set output shape
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &out));
      c->set_output(0, out);
      return absl::OkStatus();
    });

}  // namespace tensorflow
