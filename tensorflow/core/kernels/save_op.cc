// See docs in ../ops/io_ops.cc
#include "tensorflow/core/kernels/io.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/util/tensor_slice_writer.h"

namespace tensorflow {

class SaveOp : public OpKernel {
 public:
  explicit SaveOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    SaveTensors(context, &checkpoint::CreateTableTensorSliceBuilder, false);
  }
};

REGISTER_KERNEL_BUILDER(Name("Save").Device(DEVICE_CPU), SaveOp);

class SaveSlicesOp : public OpKernel {
 public:
  explicit SaveSlicesOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    SaveTensors(context, &checkpoint::CreateTableTensorSliceBuilder, true);
  }
};

REGISTER_KERNEL_BUILDER(Name("SaveSlices").Device(DEVICE_CPU), SaveSlicesOp);

class ShardedFilenameOp : public OpKernel {
 public:
  explicit ShardedFilenameOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    static const char* input_names[3] = {"basename", "shard", "num_shards"};
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      OP_REQUIRES(ctx, TensorShapeUtils::IsLegacyScalar(ctx->input(i).shape()),
                  errors::InvalidArgument(
                      input_names[i], " must be a scalar, got shape ",
                      ctx->input(i).shape().ShortDebugString()));
    }
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
    out->scalar<string>()() = strings::Printf(
        "%s-%05d-of-%05d", ctx->input(0).scalar<string>()().c_str(),
        ctx->input(1).scalar<int32>()(), ctx->input(2).scalar<int32>()());
  }
};

REGISTER_KERNEL_BUILDER(Name("ShardedFilename").Device(DEVICE_CPU),
                        ShardedFilenameOp);

class ShardedFilespecOp : public OpKernel {
 public:
  explicit ShardedFilespecOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    static const char* input_names[2] = {"basename", "num_shards"};
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      OP_REQUIRES(ctx, TensorShapeUtils::IsLegacyScalar(ctx->input(i).shape()),
                  errors::InvalidArgument(
                      input_names[i], " must be a scalar, got shape ",
                      ctx->input(i).shape().ShortDebugString()));
    }
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
    out->scalar<string>()() = strings::Printf(
        "%s-\?\?\?\?\?-of-%05d", ctx->input(0).scalar<string>()().c_str(),
        ctx->input(1).scalar<int32>()());
  }
};
REGISTER_KERNEL_BUILDER(Name("ShardedFilespec").Device(DEVICE_CPU),
                        ShardedFilespecOp);

}  // namespace tensorflow
