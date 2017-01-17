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

// XLA-specific Ops for broadcasting used in gradient
// code.

#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {
namespace {

// Given shapes of two tensors, computes the reduction indices for the
// gradient computation.
//
// TODO(zhifengc):
//   1. Adds support for n-ary (n >= 2).
class BCastGradArgsOp : public XlaOpKernel {
 public:
  explicit BCastGradArgsOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(
        ctx, ctx->MatchSignature({DT_INT32, DT_INT32}, {DT_INT32, DT_INT32}));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES(
        ctx, ctx->num_inputs() == 2,
        errors::Unimplemented("Broadcast for n-ary operations (n > 2)"));

    gtl::InlinedVector<BCast::Vec, 4> shapes;
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      const TensorShape in_shape = ctx->InputShape(i);
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(in_shape),
                  errors::InvalidArgument("In[", i, "] must be a vector.",
                                          in_shape.DebugString()));
      xla::Literal literal;
      OP_REQUIRES_OK(ctx, ctx->ConstantInput(i, &literal));

      BCast::Vec vec;
      for (int64 i = 0; i < in_shape.num_elements(); ++i) {
        vec.push_back(xla::LiteralUtil::Get<int>(literal, {i}));
      }
      shapes.push_back(vec);
    }
    BCast bcast(shapes[0], shapes[1]);
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: [", str_util::Join(shapes[0], ","),
                    "] vs. [", str_util::Join(shapes[1], ","), "]"));
    Output(ctx, 0, bcast.grad_x_reduce_idx());
    Output(ctx, 1, bcast.grad_y_reduce_idx());
  }

 private:
  void Output(XlaOpKernelContext* ctx, int idx, const BCast::Vec& v) {
    const int64 len = v.size();
    Tensor constant(DT_INT32, TensorShape({len}));
    for (int64 i = 0; i < len; ++i) {
      constant.flat<int32>()(i) = static_cast<int32>(v[i]);
    }
    ctx->SetConstantOutput(idx, constant);
  }

  TF_DISALLOW_COPY_AND_ASSIGN(BCastGradArgsOp);
};

REGISTER_XLA_OP("BroadcastGradientArgs", BCastGradArgsOp);

}  // namespace
}  // namespace tensorflow
