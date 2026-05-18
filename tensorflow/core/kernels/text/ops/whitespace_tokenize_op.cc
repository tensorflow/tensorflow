// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

namespace shape_inference {
class InferenceContext;
}  // namespace shape_inference

namespace text {

using shape_inference::InferenceContext;

REGISTER_OP("WhitespaceTokenizeWithOffsets")
    .Input("input_values: int32")
    .Input("input_splits: Tsplits")
    .Output("output_values: int32")
    .Output("output_values_inner_splits: Tsplits")
    .Output("output_offset_starts: int64")
    .Output("output_offset_limits: int64")
    .Output("output_outer_splits: Tsplits")
    .Attr("Tsplits: {int32, int64} = DT_INT64")
    .SetShapeFn([](InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));

      c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(2, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(3, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(4, c->Vector(InferenceContext::kUnknownDim));
      return absl::OkStatus();
    });

}  // namespace text
}  // namespace tensorflow
