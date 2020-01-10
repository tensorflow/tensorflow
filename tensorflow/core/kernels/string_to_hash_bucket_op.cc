#include <string>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {

class StringToHashBucketOp : public OpKernel {
 public:
  explicit StringToHashBucketOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_buckets", &num_buckets_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("string_tensor", &input_tensor));
    const auto& input_flat = input_tensor->flat<string>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output", input_tensor->shape(),
                                            &output_tensor));
    auto output_flat = output_tensor->flat<int64>();

    for (int i = 0; i < input_flat.size(); ++i) {
      const uint64 input_hash = Hash64(input_flat(i));
      const uint64 bucket_id = input_hash % num_buckets_;
      // The number of buckets is always in the positive range of int64 so is
      // the resulting bucket_id. Casting the bucket_id from uint64 to int64 is
      // safe.
      output_flat(i) = static_cast<int64>(bucket_id);
    }
  }

 private:
  int64 num_buckets_;

  TF_DISALLOW_COPY_AND_ASSIGN(StringToHashBucketOp);
};

REGISTER_KERNEL_BUILDER(Name("StringToHashBucket").Device(DEVICE_CPU),
                        StringToHashBucketOp);

}  // namespace tensorflow
