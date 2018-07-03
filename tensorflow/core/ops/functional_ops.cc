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
      std::vector<DataType> types;
      TF_RETURN_IF_ERROR(c->GetAttr("Tin", &types));
      // Say, (u, v) = f(x, y, z), _symbolic_gradient(f) is a function of
      // (x, y, z, du, dv) -> (dx, dy, dz). Therefore, shapes of its
      // outputs (dx, dy, dz) are the same as (x, y, z).
      for (int i = 0; i < c->num_outputs(); ++i) {
        if (types[i] == DT_RESOURCE) {
          const std::vector<shape_inference::ShapeAndType>* handle_type =
              c->input_handle_shapes_and_types(i);
          c->set_output(i, handle_type->at(0).shape);
        } else {
          c->set_output(i, c->input(i));
        }
      }
      return Status::OK();
    });

REGISTER_OP("RemoteCall")
    .Input("target: string")
    .Input("args: Tin")
    .Output("output: Tout")
    .Attr("Tin: list(type)")
    .Attr("Tout: list(type)")
    .Attr("f: func")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(drpng): remove this.
REGISTER_OP("_If")
    .Input("cond: Tcond")
    .Input("input: Tin")
    .Output("output: Tout")
    .Attr("Tcond: type")
    .Attr("Tin: list(type)")
    .Attr("Tout: list(type)")
    .Attr("then_branch: func")
    .Attr("else_branch: func")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
output = cond ? then_branch(input) : else_branch(input)

cond: A Tensor. If the tensor is a scalar of non-boolean type, the
    scalar is converted to a boolean according to the
    following rule: if the scalar is a numerical value, non-zero means
    True and zero means False; if the scalar is a string, non-empty
    means True and empty means False. If the tensor is not a scalar,
    being empty means False and being non-empty means True.
input: A list of input tensors.
then_branch: A function that takes 'inputs' and returns a list of
    tensors, whose types are the same as what else_branch returns.
else_branch: A function that takes 'inputs' and returns a list of
    tensors.  whose types are the same as what then_branch returns.
)doc");

REGISTER_OP("If")
    .Input("cond: Tcond")
    .Input("input: Tin")
    .Output("output: Tout")
    .Attr("Tcond: type")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type)")
    .Attr("then_branch: func")
    .Attr("else_branch: func")
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(drpng): remove this.
REGISTER_OP("_While")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: list(type) >= 0")
    .Attr("cond: func")
    .Attr("body: func")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      for (int i = 0; i < c->num_outputs(); ++i) {
        c->set_output(i, c->input(i));
      }
      return Status::OK();
    })
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
      list of tensors. Both lists have the same types as specified
      by T.
)doc");

// TODO(b/37549631) setting the While Op to always be stateful is too
// conservative.
REGISTER_OP("While")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: list(type) >= 0")
    .Attr("cond: func")
    .Attr("body: func")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      for (int i = 0; i < c->num_outputs(); ++i) {
        c->set_output(i, c->input(i));
      }
      return Status::OK();
    });

REGISTER_OP("For")
    .Input("start: int32")
    .Input("limit: int32")
    .Input("delta: int32")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: list(type) >= 0")
    .Attr("body: func")
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(b/73826847, b/37549631) Mark as stateful.
REGISTER_OP("PartitionedCall")
    .Input("args: Tin")
    .Output("output: Tout")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .Attr("f: func")
    .SetShapeFn(shape_inference::UnknownShape);

// This op is used as a placeholder in If branch functions. It doesn't provide a
// valid output when run, so must either be removed (e.g. replaced with a
// function input) or guaranteed not to be used (e.g. if mirroring an
// intermediate output needed for the gradient computation of the other branch).
REGISTER_OP("FakeParam")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .SetShapeFn([](InferenceContext* c) {
      PartialTensorShape shape;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &shape));
      shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(shape, &out));
      c->set_output(0, out);
      return Status::OK();
    });

}  // end namespace tensorflow
