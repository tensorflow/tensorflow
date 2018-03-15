/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
REGISTER_OP("SkipGramGenerateCandidates")
    .Input("input_tensor: T")
    .Input("min_skips: int32")
    .Input("max_skips: int32")
    .Input("start: int32")
    .Input("limit: int32")
    .Input("emit_self_as_target: bool")
    .Output("tokens: T")
    .Output("labels: T")
    .Attr("T: type")
    // The seed attributes are needed by GuardedPhiloxRandom
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // input_tensor must be of rank-1.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      // All other args must be scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));

      // Due to possible randomness in selecting skips, we only know that the
      // outputs will be of rank-1, but not their sizes.
      c->set_output(0, c->Vector(c->UnknownDim()));
      c->set_output(1, c->Vector(c->UnknownDim()));
      return Status::OK();
    })
    .Doc(R"doc(
Generates skip-gram token and label paired Tensors from the input tensor.
See docs for the public-facing skip_gram_sample() Python op for more details.
)doc");
}  // namespace tensorflow
