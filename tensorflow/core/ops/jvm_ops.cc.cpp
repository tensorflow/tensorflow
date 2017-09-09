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

REGISTER_OP("JVMCallback")
    .Input("input: Tin")
    .Output("output: Tout")
    .Attr("id: int")
    .Attr("jvm_pointer: string")
    .Attr("registry_class_name: string")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >=0")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Invokes a JVM callback function, `f` to compute `f(input)->output`.

This operation is considered stateful. For a stateless version, see
`JVMCallback`.

id: A unique ID representing a registered JVM callback function
  in this address space.
jvm_pointer: A pointer to an existing JVM instance represented as a
  string. This is the JVM that will be used when invoking this JVM
  callback.
registry_class_name: Name of the callbacks registry class.
input: List of tensors that will provide input to the op.
output: Output tensors from the op.
Tin: Data types of the inputs to the op.
Tout: Data types of the outputs from the op.
      The length of the list specifies the number of outputs.
)doc");

REGISTER_OP("JVMCallbackStateless")
    .Input("input: Tin")
    .Output("output: Tout")
    .Attr("id: int")
    .Attr("jvm_pointer: string")
    .Attr("registry_class_name: string")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
A stateless version of `JVMCallback`.
)doc");

}
