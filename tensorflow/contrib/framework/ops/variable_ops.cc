/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

using shape_inference::InferenceContext;

REGISTER_OP("ZeroInitializer")
    .Input("ref: Ref(T)")
    .Output("output_ref: Ref(T)")
    .Attr("T: realnumbertype")
    .SetAllowsUninitializedInput()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Initialize 'ref' with all zeros. This op requires that the tensor is not
initialized. The tensor will first be allocated memory, then be filled with all
zeros. This op is intended to save memory during initialization,
if you use this op, you should not run initializer of the 'ref' tensor.

ref: Should be from a `Variable` node.
output_ref:= Same as "ref".
)doc");

REGISTER_OP("ZeroVarInitializer")
    .Input("var: resource")
    .Output("output_var: resource")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .SetAllowsUninitializedInput()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Scalar());
      DataType t;
      TF_RETURN_IF_ERROR(c->GetAttr("dtype", &t));
      PartialTensorShape p;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &p));
      shape_inference::ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(p, &s));
      c->set_output_handle_shapes_and_types(
          0, std::vector<shape_inference::ShapeAndType>{{s, t}});

      return Status::OK();
    })
    .Doc(R"doc(
Initialize 'var' with all zeros. This op requires that the resource var is not
initialized. The var will first be allocated memory, then be filled with all
zeros. This op is intended to save memory during initialization,
if you use this op, you should not run initializer of the var.

var: Should be a ResourceVariable.
output_var:= Same as "var".
)doc");

}  // namespace tensorflow
