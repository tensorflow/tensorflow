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

REGISTER_OP("_XLASend")
    .Input("tensor: T")
    .Attr("T: type")
    .Attr("tensor_name: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Sends the named tensor to another XLA computation.

tensor: The tensor to send.
tensor_name: The name of the tensor to send.
)doc");

REGISTER_OP("_XLARecv")
    .Output("tensor: T")
    .Attr("T: type")
    .Attr("tensor_name: string")
    .Attr("shape: shape")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Receives the named tensor from another XLA computation.

tensor: The tensor to receive.
tensor_name: The name of the tensor to receive.
shape: The shape of the input tensor.
)doc");

}  // namespace tensorflow
