/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

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

REGISTER_OP("Sparsemax")
  .Input("logits: T")
  .Output("sparsemax: T")
  .Attr("T: {half, float, double}")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    // TODO: use shape_inference::UnchangedShapeWithRank
    // This implements:
    //   return shape_inference::UnchangedShapeWithRank(c, 2);
    // which is defined in tensorflow/core/framework/common_shape_fns.h
    // but that is not yet in the 0.11 build.
    // Softmax uses UnchangedShapeWithRankAtLeast(c, 1) in tensorflow,
    // but UnchangedShapeWithRank(c, 2) in it's corresponding loss function,
    // which takes the same input. The strict version was chosen here.
    shape_inference::ShapeHandle input;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
    c->set_output(0, input);
    return Status::OK();
  })
  .Doc(R"doc(
Computes sparsemax activations [1].

For each batch `i` and class `j` we have

    sparsemax[i, j] = max(logits[i, j] - tau(logits[i, :]), 0)

[1]: https://arxiv.org/abs/1602.02068

)doc");

}  // namespace tensorflow
