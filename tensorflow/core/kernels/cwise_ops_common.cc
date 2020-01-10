#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {

BinaryOpShared::BinaryOpShared(OpKernelConstruction* ctx, DataType out,
                               DataType in)
    : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->MatchSignature({in, in}, {out}));
}

void BinaryOpShared::SetUnimplementedError(OpKernelContext* ctx) {
  ctx->SetStatus(errors::Unimplemented(
      "Broadcast between ", ctx->input(0).shape().ShortDebugString(), " and ",
      ctx->input(1).shape().ShortDebugString(), " is not supported yet."));
}

static BCast::Vec FromShape(const TensorShape& shape) {
  BCast::Vec ret;
  for (int i = 0; i < shape.dims(); ++i) ret.push_back(shape.dim_size(i));
  return ret;
}

static TensorShape ToShape(const BCast::Vec& vec) {
  TensorShape shape;
  for (auto elem : vec) shape.AddDim(elem);
  return shape;
}

BinaryOpShared::BinaryOpState::BinaryOpState(OpKernelContext* ctx)
    : bcast(FromShape(ctx->input(0).shape()),
            FromShape(ctx->input(1).shape())) {
  if (!bcast.IsValid()) {
    ctx->SetStatus(errors::InvalidArgument(
        "Incompatible shapes: ", ctx->input(0).shape().ShortDebugString(),
        " vs. ", ctx->input(1).shape().ShortDebugString()));
    return;
  }
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_output(0, ToShape(bcast.output_shape()), &out));
}

}  // namespace tensorflow
