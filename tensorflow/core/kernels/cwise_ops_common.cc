/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {

BinaryOpShared::BinaryOpShared(OpKernelConstruction* ctx, DataType out,
                               DataType in)
    : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->MatchSignature({in, in}, {out}));
}

void BinaryOpShared::SetUnimplementedError(OpKernelContext* ctx) {
  ctx->SetStatus(errors::Unimplemented(
      "Broadcast between ", ctx->input(0).shape().DebugString(), " and ",
      ctx->input(1).shape().DebugString(), " is not supported yet."));
}

static BCast::Vec FromShape(const TensorShape& shape) {
  const int N = shape.dims();
  BCast::Vec ret(N);
  for (int i = 0; i < N; ++i) {
    ret[i] = shape.dim_size(i);
  }
  return ret;
}

static TensorShape ToShape(const BCast::Vec& vec) {
  TensorShape shape(vec);
  return shape;
}

BinaryOpShared::BinaryOpState::BinaryOpState(OpKernelContext* ctx)
    : bcast(FromShape(ctx->input(0).shape()),
            FromShape(ctx->input(1).shape())) {
  if (!bcast.IsValid()) {
    ctx->SetStatus(errors::InvalidArgument(
        "Incompatible shapes: ", ctx->input(0).shape().DebugString(), " vs. ",
        ctx->input(1).shape().DebugString()));
    return;
  }
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_output(0, ToShape(bcast.output_shape()), &out));
}

}  // namespace tensorflow
