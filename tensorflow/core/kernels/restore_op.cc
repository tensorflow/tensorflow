// See docs in ../ops/io_ops.cc.
#include "tensorflow/core/kernels/io.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/tensor_slice_reader.h"

namespace tensorflow {

class RestoreOp : public OpKernel {
 public:
  explicit RestoreOp(OpKernelConstruction* context) : OpKernel(context) {
    int preferred_shard;
    OP_REQUIRES_OK(context,
                   context->GetAttr("preferred_shard", &preferred_shard));
    if (preferred_shard == -1) {
      preferred_shard_ = checkpoint::TensorSliceReader::kLoadAllShards;
    } else {
      OP_REQUIRES(context, preferred_shard >= 0,
                  errors::InvalidArgument("Attribute 'preferred_shard' must be "
                                          "greater or equal to -1"));
      preferred_shard_ = preferred_shard;
    }
  }
  void Compute(OpKernelContext* context) override {
    RestoreTensor(context, &checkpoint::OpenTableTensorSliceReader,
                  preferred_shard_, false);
  }

 private:
  int preferred_shard_;
};

REGISTER_KERNEL_BUILDER(Name("Restore").Device(DEVICE_CPU), RestoreOp);

class RestoreSliceOp : public OpKernel {
 public:
  explicit RestoreSliceOp(OpKernelConstruction* context) : OpKernel(context) {
    int preferred_shard;
    OP_REQUIRES_OK(context,
                   context->GetAttr("preferred_shard", &preferred_shard));
    if (preferred_shard == -1) {
      preferred_shard_ = checkpoint::TensorSliceReader::kLoadAllShards;
    } else {
      OP_REQUIRES(context, preferred_shard >= 0,
                  errors::InvalidArgument("Attribute 'preferred_shard' must be "
                                          "greater or equal to -1"));
      preferred_shard_ = preferred_shard;
    }
  }
  void Compute(OpKernelContext* context) override {
    RestoreTensor(context, &checkpoint::OpenTableTensorSliceReader,
                  preferred_shard_, true);
  }

 private:
  int preferred_shard_;
};

REGISTER_KERNEL_BUILDER(Name("RestoreSlice").Device(DEVICE_CPU),
                        RestoreSliceOp);

}  // namespace tensorflow
