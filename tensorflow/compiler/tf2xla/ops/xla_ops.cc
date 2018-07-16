/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

REGISTER_OP("XlaDynamicUpdateSlice")
    .Input("input: T")
    .Input("update: T")
    .Input("indices: Tindices")
    .Output("output: T")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Wraps the XLA DynamicUpdateSlice operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#dynamicupdateslice
.

XlaDynamicUpdateSlice generates a result which is the value of the `input`
operand, with a slice update overwritten at `indices`. The shape of `update`
determines the shape of the sub-array of the result which is updated. The shape
of indices must be rank == 1, with dimension size equal to the rank of `input`.

Handling of out-of-bounds slice indices is implementation-defined.

input: A `Tensor` of type T.
indices: A vector of indices into `input`. Must have length equal to the rank of
  `input`.
update: A `Tensor` of type T. Same rank as `input`.
output: A `Tensor` of type T.
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

REGISTER_OP("XlaReduceWindow")
    .Input("input: T")
    .Input("init_value: T")
    .Attr("T: numbertype")
    .Attr("computation: func")
    .Attr("window_dimensions: list(int)")
    .Attr("window_strides: list(int)")
    .Attr("padding_low: list(int)")
    .Attr("padding_high: list(int)")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Wraps the XLA ReduceWindow operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#reducewindow .

input: the input tensor
init_value: a scalar representing the initial value for the reduction
computation: a reducer function to apply
window_dimensions: the shape of the window
window_strides: the inter-window strides
padding_low: the padding to apply at the start of each input dimensions
padding_high: the padding to apply at the end of each input dimension.
)doc");

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

REGISTER_OP("XlaSort")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Wraps the XLA Sort operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#sort
.

Sorts a tensor. Currently only rank 1 sorts in ascending order are supported.

input: A `Tensor` of type T.
output: A `Tensor` of type T.
)doc");

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

}  // namespace tensorflow
