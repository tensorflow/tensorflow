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

void BinaryOpShared::SetComputeError(OpKernelContext* ctx) {
  // For speed, errors during compute are caught only via boolean flag, with no
  // associated information.  This is sufficient for now, since the only binary
  // ops that have compute errors are integer division and mod, and the only
  // error they produce is zero division.
  const string& op = ctx->op_kernel().type_string();
  if ((op == "Div" || op == "Mod") &&
      DataTypeIsInteger(ctx->op_kernel().input_type(0))) {
    ctx->CtxFailure(errors::InvalidArgument("Integer division by zero"));
  } else {
    ctx->CtxFailure(
        errors::Internal("Unexpected error in binary operator "
                         "(only integer div and mod should have errors)"));
  }
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
    : in0(ctx->input(0)),
      in1(ctx->input(1)),
      bcast(BCast::FromShape(in0.shape()), BCast::FromShape(in1.shape())) {
  if (!bcast.IsValid()) {
    ctx->SetStatus(errors::InvalidArgument("Incompatible shapes: ",
                                           in0.shape().DebugString(), " vs. ",
                                           in1.shape().DebugString()));
    return;
  }
  OP_REQUIRES_OK(
      ctx, ctx->allocate_output(0, BCast::ToShape(bcast.output_shape()), &out));
  out_num_elements = out->NumElements();
  in0_num_elements = in0.NumElements();
  in1_num_elements = in1.NumElements();

  ndims = static_cast<int>(bcast.x_reshape().size());
}

}  // namespace tensorflow
