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

namespace tensorflow {

REGISTER_OP("XlaSend")
    .Input("tensor: T")
    .Attr("T: type")
    .Attr("tensor_name: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Sends the named tensor to another XLA computation. Wraps the XLA Send operator
documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#send .

tensor: The tensor to send.
tensor_name: A string key that identifies the channel.
)doc");

REGISTER_OP("XlaRecv")
    .Output("tensor: dtype")
    .Attr("dtype: type")
    .Attr("tensor_name: string")
    .Attr("shape: shape")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      TensorShape shape_attr;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &shape_attr));
      shape_inference::ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromTensorShape(shape_attr, &s));
      c->set_output(0, s);
      return Status::OK();
    })
    .Doc(R"doc(
Receives the named tensor from another XLA computation. Wraps the XLA Recv
operator documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#recv .

tensor: The tensor to receive.
dtype: The type of the tensor.
tensor_name: A string key that identifies the channel.
shape: The shape of the tensor.
)doc");

}  // namespace tensorflow
