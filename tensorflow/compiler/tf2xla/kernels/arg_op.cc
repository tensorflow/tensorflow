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

#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

// This OpKernel implements the _Arg Op for XLA JIT devices. It
// associates its output with one of the arguments to a
// subcomputation.
class XlaArgOp : public XlaOpKernel {
 public:
  explicit XlaArgOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("index", &index_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    // If 'frame' is non-null, this is a function call inside an outer JIT
    // compilation. Use the usual implementation of _Arg.
    auto frame = ctx->call_frame();
    if (frame != nullptr) {
      const Tensor* val;
      OP_REQUIRES_OK(ctx, frame->GetArg(index_, &val));
      // Types that cannot be copied using memcpy (like DT_STRING) are wrapped
      // in a DT_UINT8 and hence the type mismatches. Skip the test in such
      // cases. See XlaOpKernelContext::SetOutputExpression for details.
      if (DataTypeCanUseMemcpy(dtype_)) {
        OP_REQUIRES(ctx, val->dtype() == dtype_,
                    errors::InvalidArgument(
                        "Type mismatch: actual ", DataTypeString(val->dtype()),
                        " vs. expect ", DataTypeString(dtype_)));
      }
      // Forwards the argument from the frame.
      ctx->op_kernel_context()->set_output(0, *val);
      return;
    }

    const XlaExpression& arg = ctx->xla_context()->args()[index_];
    OP_REQUIRES(ctx, arg.kind() != XlaExpression::Kind::kInvalid,
                errors::InvalidArgument("Invalid/missing argument expression"));
    if (ctx->expected_output_dtype(0) == DT_VARIANT) {
      ctx->SetTensorListOutput(0, arg.handle());
    } else if (arg.value_bound().has_value()) {
      // The argument has a bound attached to it, call SetBound op on the
      // argument.
      xla::XlaBuilder* builder = ctx->builder();
      auto input_op = arg.AsXlaOp(builder);

      // We pass two pieces of information to SetBound:
      // Bound - The upper-bounds of the argument's values.
      //
      // Dynamism - Whether or not each individual value is dynamic. If this
      // is false, it means value with same tensor index in the argument is
      // static, and it's upper-bound is same as lower-bound and also same as
      // the static value itself.
      //
      // E.g.,:
      // When we have an argument `arg` with shape s32[3], bound = [1, 2, 3] and
      // dynamism = [false, false, true]
      //
      // We know that:
      //  arg[0] is a static value, its value is 1
      //  arg[1] is a static value, its value is 2
      //  arg[2] is a dynamic value, its value is unknown at compile time, but
      //  its upper-bound is known to be 3.
      //
      // Note that `arg` is still considered dynamic as long as one element
      // inside is dynamic, therefore the argument node can't be constant folded
      // into a constant node.
      xla::Literal bound = HostTensorToLiteral(*arg.value_bound()).ValueOrDie();
      xla::Literal dynamism =
          HostTensorToLiteral(*arg.value_dynamism()).ValueOrDie();
      xla::Literal tuple = xla::LiteralUtil::MakeTupleOwned(
          std::move(bound), std::move(dynamism));
      ctx->SetOutput(
          0, xla::CustomCall(builder, "SetBound", {input_op},
                             builder->GetShape(input_op).ValueOrDie(), "",
                             false, {}, &tuple));
      return;
    } else {
      ctx->SetOutputExpression(0, arg);
    }
  }

 private:
  int index_;
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaArgOp);
};

REGISTER_XLA_OP(
    Name("_Arg").AllowResourceTypes().AllowVariantTypes().CompilationOnly(),
    XlaArgOp);

}  // namespace tensorflow
