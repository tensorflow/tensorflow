// Copyright 2017 The Sonnet Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("Resampler")
    .Input("data: T")
    .Input("warp: T")
    .Output("output: T")
    .Attr("T: {half, float, double}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle data;
      ShapeHandle warp;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &data));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &warp));

      ShapeHandle output;  // will be warp[:-1] + [data[-1]]
      TF_RETURN_IF_ERROR(c->Subshape(warp, 0, -1, &output));
      TF_RETURN_IF_ERROR(
          c->Concatenate(output, c->Vector(c->Dim(data, -1)), &output));

      c->set_output(0, output);
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(Resampler op.)doc");

REGISTER_OP("ResamplerGrad")
    .Input("data: T")
    .Input("warp: T")
    .Input("grad_output: T")
    .Output("grad_data: T")
    .Output("grad_warp: T")
    .Attr("T: {half, float, double}")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(Resampler Grad op.)doc");

}  // namespace tensorflow
