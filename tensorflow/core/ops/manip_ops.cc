/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// --------------------------------------------------------------------------
REGISTER_OP("Roll")
    .Input("input: T")
    .Input("shift: Tshift")
    .Input("axis: Taxis")
    .Output("output: T")
    .Attr("T: type")
    .Attr("Tshift: {int32,int64}")
    .Attr("Taxis: {int32,int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // The `input` must be 1-D or higher
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &unused));
      // The `shift` must be scalar or 1-D.
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(1), 1, &unused));
      // The `axis` must be scalar or 1-D.
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(2), 1, &unused));
      // Validate 'shift' is the same shape as axis'.
      TF_RETURN_IF_ERROR(c->Merge(c->input(1), c->input(2), &unused));
      return shape_inference::UnchangedShape(c);
    });

}  // namespace tensorflow
