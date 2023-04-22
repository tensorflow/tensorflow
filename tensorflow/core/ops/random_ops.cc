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
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("RandomUniform")
    .Input("shape: T")
    .SetIsStateful()
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("dtype: {half,bfloat16,float,double}")
    .Attr("T: {int32, int64}")
    .SetShapeFn(shape_inference::RandomShape);

REGISTER_OP("RandomUniformInt")
    .Input("shape: T")
    .Input("minval: Tout")
    .Input("maxval: Tout")
    .SetIsStateful()
    .Output("output: Tout")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("Tout: {int32, int64}")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      Status s = c->WithRank(c->input(1), 0, &unused);
      if (!s.ok()) {
        return errors::InvalidArgument(
            "minval must be a scalar; got a tensor of shape ",
            c->DebugString(c->input(1)));
      }
      s = c->WithRank(c->input(2), 0, &unused);
      if (!s.ok()) {
        return errors::InvalidArgument(
            "maxval must be a scalar; got a tensor of shape ",
            c->DebugString(c->input(2)));
      }
      return shape_inference::RandomShape(c);
    });

REGISTER_OP("RandomStandardNormal")
    .Input("shape: T")
    .SetIsStateful()
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("dtype: {half,bfloat16,float,double}")
    .Attr("T: {int32, int64}")
    .SetShapeFn(shape_inference::RandomShape);

REGISTER_OP("ParameterizedTruncatedNormal")
    .Input("shape: T")
    .Input("means: dtype")
    .Input("stdevs: dtype")
    .Input("minvals: dtype")
    .Input("maxvals: dtype")
    .SetIsStateful()
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("dtype: {half,bfloat16,float,double}")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      // Parameters must be 0-d or 1-d.
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(3), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(4), 1, &unused));
      return shape_inference::RandomShape(c);
    });

REGISTER_OP("TruncatedNormal")
    .Input("shape: T")
    .SetIsStateful()
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("dtype: {half,bfloat16,float,double}")
    .Attr("T: {int32, int64}")
    .SetShapeFn(shape_inference::RandomShape);

REGISTER_OP("RandomShuffle")
    .Input("value: T")
    .SetIsStateful()
    .Output("output: T")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Multinomial")
    .SetIsStateful()
    .Input("logits: T")
    .Input("num_samples: int32")
    .Output("output: output_dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("T: realnumbertype")
    .Attr("output_dtype: {int32, int64} = DT_INT64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle logits_shape;
      ShapeHandle unused;
      DimensionHandle num_samples;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &logits_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(1, &num_samples));
      c->set_output(0, c->Matrix(c->Dim(logits_shape, 0), num_samples));
      return Status::OK();
    });

REGISTER_OP("RandomGamma")
    .SetIsStateful()
    .Input("shape: S")
    .Input("alpha: T")
    .Output("output: T")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("S: {int32, int64}")
    .Attr("T: {half, float, double}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
      TF_RETURN_IF_ERROR(c->Concatenate(out, c->input(1), &out));
      c->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("RandomGammaGrad")
    .Input("alpha: T")
    .Input("sample: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("RandomPoisson")
    .SetIsStateful()
    .Input("shape: S")
    .Input("rate: dtype")
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("S: {int32, int64}")
    .Attr("dtype: {half, float, double}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
      TF_RETURN_IF_ERROR(c->Concatenate(out, c->input(1), &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Deprecated(25, "Replaced by RandomPoissonV2");

REGISTER_OP("RandomPoissonV2")
    .SetIsStateful()
    .Input("shape: S")
    .Input("rate: R")
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("S: {int32, int64}")
    .Attr("R: {half, float, double, int32, int64} = DT_DOUBLE")
    .Attr("dtype: {half, float, double, int32, int64} = DT_INT64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
      TF_RETURN_IF_ERROR(c->Concatenate(out, c->input(1), &out));
      c->set_output(0, out);
      return Status::OK();
    });

}  // namespace tensorflow
