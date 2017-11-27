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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::InferenceContext;

REGISTER_OP("SymbolicGradient")
    .Input("input: Tin")
    .Output("output: Tout")
    .Attr("Tin: list(type)")
    .Attr("Tout: list(type)")
    .Attr("f: func")
    .SetShapeFn([](InferenceContext* c) {
      if (c->num_inputs() < c->num_outputs()) {
        return errors::InvalidArgument("len(inputs) < len(outputs)");
      }
      // Say, (u, v) = f(x, y, z), _symbolic_gradient(f) is a function of
      // (x, y, z, du, dv) -> (dx, dy, dz). Therefore, shapes of its
      // outputs (dx, dy, dz) are the same as (x, y, z).
      for (int i = 0; i < c->num_outputs(); ++i) {
        c->set_output(i, c->input(i));
      }
      return Status::OK();
    })
    .Doc(R"doc(
Computes the gradient function for function f via backpropagation.

input: a list of input tensors of size N + M;
output: a list of output tensors of size N;
Tin: the type list for the input list.
Tout: the type list for the input list.
f: The function we want to compute the gradient for.

The function 'f' must be a numerical function which takes N inputs and
produces M outputs. Its gradient function 'g', which is computed by
this SymbolicGradient op is a function taking N + M inputs and
produces N outputs.

I.e. if we have
   (y1, y2, ..., y_M) = f(x1, x2, ..., x_N),
then, g is
   (dL/dx1, dL/dx2, ..., dL/dx_N) = g(x1, x2, ..., x_N,
                                     dL/dy1, dL/dy2, ..., dL/dy_M),

where L is a scalar-value function of (x1, x2, ..., xN) (e.g., the
loss function). dL/dx_i is the partial derivative of L with respect
to x_i.

(Needs some math expert to say the comment above better.)
)doc");

REGISTER_OP("RemoteCall")
    .Input("target: string")
    .Input("args: Tin")
    .Output("output: Tout")
    .Attr("Tin: list(type)")
    .Attr("Tout: list(type)")
    .Attr("f: func")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Runs function `f` on a remote device indicated by `target`.

target: A fully specified device name where we want to run the function.
args: A list of arguments for the function.
output: A list of return values.
Tin: The type list for the arguments.
Tout: The type list for the return values.
f: The function to run remotely.
)doc");
}  // end namespace tensorflow
