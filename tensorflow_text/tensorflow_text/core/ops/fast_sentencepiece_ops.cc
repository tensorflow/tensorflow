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
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace text {

REGISTER_OP("TFText>FastSentencepieceTokenize")
    .Input("sp_model: uint8")
    .Input("input: string")
    .Input("nbest_size: int32")
    .Input("alpha: float")
    .Input("add_bos: bool")
    .Input("add_eos: bool")
    .Input("reverse: bool")
    .Attr("out_type: {int32, string} = DT_INT32")
    .Attr("Tsplits: {int32, int64} = DT_INT32")
    .Output("output_values: out_type")
    .Output("output_splits: Tsplits")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));

      c->set_output(
          0, c->Vector(
                 tensorflow::shape_inference::InferenceContext::kUnknownDim));

      tensorflow::shape_inference::DimensionHandle num_splits;
      TF_RETURN_IF_ERROR(c->Add(c->NumElements(c->input(1)), 1, &num_splits));
      c->set_output(1, c->Vector(num_splits));
      return absl::OkStatus();
    });

REGISTER_OP("TFText>FastSentencepieceDetokenize")
    .Input("sp_model: uint8")
    .Input("input_values: int32")
    .Input("input_splits: Tsplits")
    .Attr("Tsplits: {int32, int64} = DT_INT64")
    .Output("output: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));

      shape_inference::DimensionHandle dim;
      TF_RETURN_IF_ERROR(c->Subtract(c->NumElements(c->input(2)), 1, &dim));
      c->set_output(0, c->Vector(dim));
      return absl::OkStatus();
    });

}  // namespace text
}  // namespace tensorflow
