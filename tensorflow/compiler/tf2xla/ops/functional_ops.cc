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

// TODO(b/37549631) setting the While Op to always be stateful is too
// conservative.
REGISTER_OP("XlaWhile")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: list(type) >= 0")
    .Attr("cond: func")
    .Attr("body: func")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
output = input; While (Cond(output)) { output = Body(output) }

input: A list of input tensors whose types are T.
output: A list of output tensors whose types are T.
cond: A function takes 'input' and returns a tensor.  If the tensor is
      a scalar of non-boolean, the scalar is converted to a boolean
      according to the following rule: if the scalar is a numerical
      value, non-zero means True and zero means False; if the scalar is
      a string, non-empty means True and empty means False. If the
      tensor is not a scalar, non-emptiness means True and False
      otherwise.
body: A function that takes a list of tensors and returns another
      list of tensors. Both lists have the same types as specified by T.
)doc");

// TODO(b/37549631) setting the If Op to always be stateful is too
// conservative.
REGISTER_OP("XlaIf")
    .Input("cond: Tcond")
    .Input("inputs: Tin")
    .Output("output: Tout")
    .Attr("Tcond: type")
    .Attr("then_branch: func")
    .Attr("else_branch: func")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
output = cond ? then_branch(inputs) : else_branch(inputs).

cond: A boolean scalar.
inputs: A list of input tensors.
output: A list of tensors returned by either then_branch(inputs) or
        else_branch(inputs). The input shapes of the then_branch and
        else_branch must match.
then_branch: A function takes 'inputs' and returns a list of tensors,
             whose types are the same as what else_branch returns.
else_branch: A function takes 'inputs' and returns a list of tensors.
             whose types are the same as what then_branch returns.
)doc");

}  // namespace tensorflow
