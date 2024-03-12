/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/aot/aot_only_var_handle_op.h"

#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace {

// Implementation of varhandle that binds a VarHandleOp to an XlaResource of the
// same name. It is not safe to use this op in a JIT context.
class XlaAotOnlyVarHandleOp : public XlaOpKernel {
 public:
  explicit XlaAotOnlyVarHandleOp(OpKernelConstruction* c);
  void Compile(XlaOpKernelContext* context) override;

 private:
  string name_;
};

XlaAotOnlyVarHandleOp::XlaAotOnlyVarHandleOp(OpKernelConstruction* c)
    : XlaOpKernel(c) {
  OP_REQUIRES_OK(c, c->GetAttr("shared_name", &name_));
}

void XlaAotOnlyVarHandleOp::Compile(XlaOpKernelContext* context) {
  // Look for a resource of the same name. TF also keys that on the container
  // and type attributes, but that doesn't seem necessary.
  for (const auto& resource : context->xla_context()->resources()) {
    if (resource->kind() == XlaResource::kVariable &&
        resource->name() == name_) {
      context->SetResourceOutput(0, resource.get());
      return;
    }
  }
  context->SetStatus(
      errors::InvalidArgument("Variable: ", name_, " not configured"));
}
}  // namespace

REGISTER_OP(tfcompile::kXlaAotOnlyVarHandleOp)
    .Doc(R"doc(
Internal VarHandleOp registration used for XLA AOT compilation.
)doc")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("debug_name: string = ''")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .Output("resource: resource")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      DataType t;
      TF_RETURN_IF_ERROR(c->GetAttr("dtype", &t));
      PartialTensorShape p;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &p));
      shape_inference::ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(p, &s));
      c->set_output_handle_shapes_and_types(
          0, std::vector<shape_inference::ShapeAndType>{{s, t}});

      return absl::OkStatus();
    });

REGISTER_XLA_OP(Name(tfcompile::kXlaAotOnlyVarHandleOp).CompilationOnly(),
                XlaAotOnlyVarHandleOp);

}  // namespace tensorflow
