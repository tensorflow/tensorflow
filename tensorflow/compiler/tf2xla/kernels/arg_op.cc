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

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {
namespace {

// This OpKernel implements the _Arg Op for XLA JIT devices. It
// associates its output with one of the arguments to a
// subcomputation.
class ArgOp : public XlaOpKernel {
 public:
  explicit ArgOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("index", &index_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    // If 'frame' is non-null, this is a function call inside an outer JIT
    // compilation. Use the usual implementation of _Arg.
    auto frame = ctx->call_frame();
    if (frame != nullptr) {
      Tensor val;
      OP_REQUIRES_OK(ctx, frame->GetArg(index_, &val));
      OP_REQUIRES(ctx, val.dtype() == dtype_,
                  errors::InvalidArgument(
                      "Type mismatch: actual ", DataTypeString(val.dtype()),
                      " vs. expect ", DataTypeString(dtype_)));
      // Forwards the argument from the frame.
      ctx->op_kernel_context()->set_output(0, val);
      return;
    }

    XlaContext& xc = XlaContext::Get(ctx);
    const XlaContext::Argument& arg = xc.args()[index_];
    if (arg.is_resource) {
      XlaResource::Kind kind;
      switch (arg.kind) {
        case XlaCompiler::Argument::kVariable:
          kind = XlaResource::kVariable;
          break;
        case XlaCompiler::Argument::kTensorArray:
          kind = XlaResource::kTensorArray;
          break;
        case XlaCompiler::Argument::kStack:
          kind = XlaResource::kStack;
          break;
        default:
          CHECK(false);
      }

      // TODO(phawkins): this code assumes that variables do not alias.
      XlaResource* resource;
      OP_REQUIRES_OK(ctx,
                     xc.CreateResource(kind, index_, arg.name, arg.value.type,
                                       arg.value.handle, &resource));
      resource->tensor_array_size = arg.tensor_array_size;
      ctx->SetResourceOutput(0, resource);
    } else if (arg.value.is_constant) {
      ctx->SetConstantOutput(0, arg.value.constant_value);
    } else {
      ctx->SetOutput(0, arg.value.handle);
    }
  }

 private:
  int index_;
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(ArgOp);
};

REGISTER_XLA_OP(Name("_Arg").AllowResourceTypes(), ArgOp);

}  // namespace
}  // namespace tensorflow
