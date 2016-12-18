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

REGISTER_OP("SparsemaxLoss")
  .Input("logits: T")
  .Input("sparsemax: T")
  .Input("labels: T")
  .Output("loss: T")
  .Attr("T: {half, float, double}")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    // Similar to SoftmaxCrossEntropyWithLogits
    // Ensure that input has rank 2, and they all have the same size
    shape_inference::ShapeHandle input;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
    TF_RETURN_IF_ERROR(c->Merge(input, c->input(1), &input));
    TF_RETURN_IF_ERROR(c->Merge(input, c->input(2), &input));
    // Output is a vector with the loss for each observation
    shape_inference::DimensionHandle batch_size = c->Dim(input, 0);
    c->set_output(0, c->Vector(batch_size));
    return Status::OK();
  })
  .Doc(R"doc(
Computes sparsemax loss function [1].

[1]: https://arxiv.org/abs/1602.02068

)doc");

}  // namespace tensorflow
