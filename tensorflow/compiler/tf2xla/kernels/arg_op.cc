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

    XlaContext& tc = XlaContext::Get(ctx);
    const XlaContext::Argument& arg = tc.args()[index_];
    if (arg.is_variable) {
      // We use the argument position of the variable input as a unique ID.
      // TODO(phawkins): this code assumes that variables do not alias.
      OP_REQUIRES_OK(ctx, tc.CreateVariable(index_, arg.name, arg.value.type,
                                            arg.value.handle));
      ctx->SetVariableOutput(0, index_);
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
