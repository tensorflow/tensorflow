/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

static absl::Status StatelessShape(InferenceContext* c) {
  // Check seed shape
  ShapeHandle seed;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &seed));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(seed, 0), 2, &unused));

  // Set output shape
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
  c->set_output(0, out);
  return absl::OkStatus();
}

#define REGISTER_STATELESS_OP(name)                           \
  REGISTER_OP(name)                                           \
      .Input("shape: T")                                      \
      .Input("seed: Tseed")                                   \
      .Output("output: dtype")                                \
      .Attr("dtype: {half,bfloat16,float,double} = DT_FLOAT") \
      .Attr("T: {int32, int64} = DT_INT32")                   \
      .Attr("Tseed: {int32, int64} = DT_INT64")               \
      .SetShapeFn(StatelessShape)

REGISTER_STATELESS_OP("StatelessRandomUniform");
REGISTER_STATELESS_OP("StatelessRandomNormal");
REGISTER_STATELESS_OP("StatelessTruncatedNormal");

#undef REGISTER_STATELESS_OP

REGISTER_OP("StatelessRandomUniformInt")
    .Input("shape: T")
    .Input("seed: Tseed")
    .Input("minval: dtype")
    .Input("maxval: dtype")
    .Output("output: dtype")
    .Attr("dtype: {int32, int64}")
    .Attr("T: {int32, int64}")
    .Attr("Tseed: {int32, int64} = DT_INT64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      absl::Status s = c->WithRank(c->input(2), 0, &unused);
      if (!s.ok()) {
        return errors::InvalidArgument(
            "minval must be a scalar; got a tensor of shape ",
            c->DebugString(c->input(2)));
      }
      s = c->WithRank(c->input(3), 0, &unused);
      if (!s.ok()) {
        return errors::InvalidArgument(
            "maxval must be a scalar; got a tensor of shape ",
            c->DebugString(c->input(3)));
      }
      return StatelessShape(c);
    });

REGISTER_OP("StatelessRandomUniformFullInt")
    .Input("shape: T")
    .Input("seed: Tseed")
    .Output("output: dtype")
    .Attr("dtype: {int32, int64, uint32, uint64} = DT_UINT64")
    .Attr("T: {int32, int64} = DT_INT32")
    .Attr("Tseed: {int32, int64, uint32, uint64} = DT_INT64")
    .SetShapeFn(StatelessShape);

REGISTER_OP("StatelessMultinomial")
    .Input("logits: T")
    .Input("num_samples: int32")
    .Input("seed: Tseed")
    .Output("output: output_dtype")
    .Attr("T: realnumbertype")
    .Attr("Tseed: {int32, int64} = DT_INT64")
    .Attr("output_dtype: {int32, int64} = DT_INT64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // Check seed shape
      ShapeHandle seed;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &seed));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(seed, 0), 2, &unused_dim));

      ShapeHandle logits_shape;
      ShapeHandle unused;
      DimensionHandle num_samples;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &logits_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(1, &num_samples));
      c->set_output(0, c->Matrix(c->Dim(logits_shape, 0), num_samples));
      return absl::OkStatus();
    });

REGISTER_OP("StatelessRandomBinomial")
    .Input("shape: S")
    .Input("seed: Tseed")
    .Input("counts: T")
    .Input("probs: T")
    .Output("output: dtype")
    .Attr("S: {int32, int64}")
    .Attr("Tseed: {int32, int64} = DT_INT64")
    .Attr("T: {half, float, double, int32, int64} = DT_DOUBLE")
    .Attr("dtype: {half, float, double, int32, int64} = DT_INT64")
    .SetShapeFn(StatelessShape);

REGISTER_OP("StatelessParameterizedTruncatedNormal")
    .Input("shape: S")
    .Input("seed: Tseed")
    .Input("means: dtype")
    .Input("stddevs: dtype")
    .Input("minvals: dtype")
    .Input("maxvals: dtype")
    .Output("output: dtype")
    .Attr("S: {int32, int64}")
    .Attr("Tseed: {int32, int64} = DT_INT64")
    .Attr("dtype: {float16, float32, float64}")
    .SetShapeFn([](InferenceContext* c) {
      // Check seed shape
      ShapeHandle seed;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &seed));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(seed, 0), 2, &unused_dim));

      ShapeHandle bcast_means_stddevs;
      ShapeHandle bcast_except_maxvals;
      ShapeHandle bcast_all;
      TF_RETURN_IF_ERROR(BroadcastBinaryOpOutputShapeFnHelper(
          c, c->input(2), c->input(3), true, &bcast_means_stddevs));
      TF_RETURN_IF_ERROR(BroadcastBinaryOpOutputShapeFnHelper(
          c, c->input(4), bcast_means_stddevs, true, &bcast_except_maxvals));
      TF_RETURN_IF_ERROR(BroadcastBinaryOpOutputShapeFnHelper(
          c, c->input(5), bcast_except_maxvals, true, &bcast_all));

      // Set output shape
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
      c->set_output(0, out);
      return absl::OkStatus();
    });

REGISTER_OP("StatelessRandomPoisson")
    .Input("shape: T")
    .Input("seed: Tseed")
    .Input("lam: Rtype")
    .Output("output: dtype")
    .Attr("Rtype: {float16, float32, float64, int32, int64}")
    .Attr("dtype: {float16, float32, float64, int32, int64}")
    .Attr("T: {int32, int64}")
    .Attr("Tseed: {int32, int64} = DT_INT64")
    .SetShapeFn(StatelessShape);

REGISTER_OP("StatelessRandomGammaV2")
    .Input("shape: T")
    .Input("seed: Tseed")
    .Input("alpha: dtype")
    .Output("output: dtype")
    .Attr("dtype: {float16, float32, float64}")
    .Attr("T: {int32, int64}")
    .Attr("Tseed: {int32, int64} = DT_INT64")
    .SetShapeFn(StatelessShape);

}  // namespace tensorflow
