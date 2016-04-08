/* Copyright 2015 Google Inc. All Rights Reserved.

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

namespace tensorflow {

REGISTER_OP("PyFunc")
    .Input("input: Tin")
    .Output("output: Tout")
    .Attr("token: string")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type)")
    .Doc(R"doc(
Invokes a python function to compute func(input)->output.

token: A token representing a registered python function in this address space.
input: List of Tensors that will provide input to the Op.
output: The outputs from the Op.
Tin: Data types of the inputs to the op.
Tout: Data types of the outputs from the op.
      The length of the list specifies the number of outputs.
)doc");

}  // namespace tensorflow
