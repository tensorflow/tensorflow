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

REGISTER_OP("Assert")
    .Input("condition: bool")
    .Input("data: T")
    .Attr("T: list(type)")
    .Attr("summarize: int = 3")
    .Doc(R"doc(
Asserts that the given condition is true.

If `condition` evaluates to false, print the list of tensors in `data`.
`summarize` determines how many entries of the tensors to print.

condition: The condition to evaluate.
data: The tensors to print out when condition is false.
summarize: Print this many entries of each tensor.
)doc");

REGISTER_OP("Print")
    .Input("input: T")
    .Input("data: U")
    .Output("output: T")
    .Attr("T: type")
    .Attr("U: list(type)")
    .Attr("message: string = ''")
    .Attr("first_n: int = -1")
    .Attr("summarize: int = 3")
    .Doc(R"doc(
Prints a list of tensors.

Passes `input` through to `output` and prints `data` when evaluating.

input: The tensor passed to `output`
data: A list of tensors to print out when op is evaluated.
output:= The unmodified `input` tensor
message: A string, prefix of the error message.
first_n: Only log `first_n` number of times. -1 disables logging.
summarize: Only print this many entries of each tensor.
)doc");

}  // end namespace tensorflow
