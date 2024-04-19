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

#include <cstdint>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/platform/errors.h"

namespace tensorflow {

using shape_inference::ShapeHandle;

REGISTER_OP("KthOrderStatistic")
    .Input("input: float32")
    .Output("output: float32")
    .Attr("k: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));

      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->Subshape(input, 0, -1, &s));
      c->set_output(0, s);
      return absl::OkStatus();
    });

REGISTER_OP("TopKUnique")
    .Input("input: float32")
    .Output("topk: float32")
    .Output("topk_indices: int32")
    .Attr("k: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));

      int32_t k;
      TF_RETURN_IF_ERROR(c->GetAttr("k", &k));

      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->ReplaceDim(input, 1, c->MakeDim(k), &s));
      c->set_output(0, s);
      c->set_output(1, s);
      return absl::OkStatus();
    });

REGISTER_OP("MakeUnique")
    .Input("input: float32")
    .Output("output: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
      c->set_output(0, input);
      return absl::OkStatus();
    });

REGISTER_OP("TopKWithUnique")
    .Input("input: float32")
    .Output("topk: float32")
    .Output("topk_indices: int32")
    .Attr("k: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));

      int32_t k;
      TF_RETURN_IF_ERROR(c->GetAttr("k", &k));

      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->ReplaceDim(input, 1, c->MakeDim(k), &s));
      c->set_output(0, s);
      c->set_output(1, s);
      return absl::OkStatus();
    });
}  // namespace tensorflow
