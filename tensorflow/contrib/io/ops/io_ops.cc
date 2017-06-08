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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

Status ScalarInputsAndOutputs(InferenceContext* c) {
  ShapeHandle unused;
  for (int i = 0; i < c->num_inputs(); ++i) {
    TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 0, &unused));
  }
  for (int i = 0; i < c->num_outputs(); ++i) {
    c->set_output(i, c->Scalar());
  }
  return Status::OK();
}

}  // namespace

REGISTER_OP("PollZmq")
    .Input("request: string")
    .Output("reply: string")
    .Attr("address: string")
    .Attr("timeout: int = -1")
    .SetIsStateful()
    .SetShapeFn(ScalarInputsAndOutputs)
    .Doc(R"doc(
Sends a message using ZeroMQ and returns the reply.

request: The request to send.
address: A string consisting of a 'transport'`://` followed by an 'address'. The
  'transport' specifies the underlying protocol to use (usually 'tcp'). The
  'address' specifies the transport-specific address to connect to.
timeout: The timeout for receiving messages in milliseconds. The default (-1)
  corresponds to an infinite timeout.
)doc");

}  // namespace tensorflow
