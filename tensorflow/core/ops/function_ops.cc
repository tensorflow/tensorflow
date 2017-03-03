/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

REGISTER_OP("_Arg")
    .Output("output: T")
    .Attr("T: type")
    .Attr("index: int >= 0")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      context->set_output(0, context->UnknownShape());
      return Status::OK();
    })
    .Doc(R"doc(
A graph node which represents an argument to a function.

output: The argument.
index: This argument is the index-th argument of the function.
)doc");

REGISTER_OP("_Retval")
    .Input("input: T")
    .Attr("T: type")
    .Attr("index: int >= 0")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      return Status::OK();
    })
    .Doc(R"doc(
A graph node which represents a return value of a function.

input: The return value.
index: This return value is the index-th return value of the function.
)doc");

REGISTER_OP("_ListToArray")
    .Input("input: Tin")
    .Output("output: N * T")
    .Attr("Tin: list(type)")
    .Attr("T: type")
    .Attr("N: int >= 1")
    .Doc(R"doc(
Converts a list of tensors to an array of tensors.
)doc");

REGISTER_OP("_ArrayToList")
    .Input("input: N * T")
    .Output("output: out_types")
    .Attr("T: type")
    .Attr("N: int >= 1")
    .Attr("out_types: list(type)")
    .Doc(R"doc(
Converts an array of tensors to a list of tensors.
)doc");

}  // namespace tensorflow
