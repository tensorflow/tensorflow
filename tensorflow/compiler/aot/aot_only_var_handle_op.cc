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

#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

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

REGISTER_XLA_OP(Name("VarHandleOp").CompilationOnly(), XlaAotOnlyVarHandleOp);

}  // namespace tensorflow
