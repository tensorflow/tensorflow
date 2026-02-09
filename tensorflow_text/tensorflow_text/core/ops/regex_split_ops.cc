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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace text {

absl::Status RegexSplitOpShape(shape_inference::InferenceContext* c) {
  shape_inference::ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));

  for (int i = 0; i < c->num_outputs(); ++i) {
    c->set_output(i, c->UnknownShapeOfRank(1));
  }
  return absl::OkStatus();
}

REGISTER_OP("RegexSplitWithOffsets")
    .Input("input: string")
    .Input("delim_regex_pattern: string")
    .Input("keep_delim_regex_pattern: string")
    .Output("tokens: string")
    .Output("begin_offsets: int64")
    .Output("end_offsets: int64")
    .Output("row_splits: int64")
    .SetShapeFn(RegexSplitOpShape)
    .Doc(R"doc(
Split strings using a regex as the delimiter.

See https://github.com/google/re2/wiki/Syntax for the full list of supported
expressions.
)doc");

}  // namespace text
}  // namespace tensorflow
