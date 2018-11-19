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

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace tensorflow {
namespace {

class CrossOp : public XlaOpKernel {
 public:
  explicit CrossOp(OpKernelConstruction* context) : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape in0_shape = ctx->InputShape(0);
    TensorShape in1_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, in0_shape == in1_shape,
                errors::InvalidArgument("Both inputs must be of same shape: ",
                                        in0_shape.DebugString(), " vs. ",
                                        in1_shape.DebugString()));
    OP_REQUIRES(ctx, in0_shape.dims() >= 1,
                errors::InvalidArgument("Input must be at least 1D",
                                        in0_shape.DebugString()));

    auto inner_dim = in0_shape.dim_size(in0_shape.dims() - 1);
    OP_REQUIRES(ctx, inner_dim == 3,
                errors::FailedPrecondition(
                    "Cross-products are only defined for 3-element vectors."));

    // in0 is a [...,X,Y,Z,3]
    // in1 is the same shape as in0
    // So slice 0 is: in0[...,:,:,:,0:1]
    // So slice 1 is: in0[...,:,:,:,1:2]
    // So slice 2 is: in0[...,:,:,:,2:3]

    std::vector<int64> starts(in0_shape.dims(), 0);
    std::vector<int64> limits;
    for (auto dim_size : in0_shape.dim_sizes()) {
      limits.push_back(dim_size);
    }
    std::vector<int64> strides(in0_shape.dims(), 1);

    xla::XlaBuilder* b = ctx->builder();
    auto in0 = ctx->Input(0);
    auto in1 = ctx->Input(1);
    starts.back() = 0;
    limits.back() = 1;
    auto u1 = xla::Slice(in0, starts, limits, strides);
    auto v1 = xla::Slice(in1, starts, limits, strides);
    starts.back() = 1;
    limits.back() = 2;
    auto u2 = xla::Slice(in0, starts, limits, strides);
    auto v2 = xla::Slice(in1, starts, limits, strides);
    starts.back() = 2;
    limits.back() = 3;
    auto u3 = xla::Slice(in0, starts, limits, strides);
    auto v3 = xla::Slice(in1, starts, limits, strides);

    auto s1 = xla::Sub(xla::Mul(u2, v3), xla::Mul(u3, v2));
    auto s2 = xla::Sub(xla::Mul(u3, v1), xla::Mul(u1, v3));
    auto s3 = xla::Sub(xla::Mul(u1, v2), xla::Mul(u2, v1));
    auto output = xla::ConcatInDim(b, {s1, s2, s3}, in0_shape.dims() - 1);

    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CrossOp);
};

REGISTER_XLA_OP(Name("Cross"), CrossOp);

}  // namespace
}  // namespace tensorflow
