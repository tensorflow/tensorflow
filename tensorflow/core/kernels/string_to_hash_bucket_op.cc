/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/string_to_hash_bucket_op.h"

#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/strong_hash.h"

namespace tensorflow {

// Deprecated class. It also uses `string_tensor` as Op argument instead of
// `input`.
class LegacyStringToHashBucketOp : public OpKernel {
 public:
  explicit LegacyStringToHashBucketOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_buckets", &num_buckets_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("string_tensor", &input_tensor));
    const auto& input_flat = input_tensor->flat<tstring>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output", input_tensor->shape(),
                                            &output_tensor));
    auto output_flat = output_tensor->flat<int64_t>();

    typedef decltype(input_flat.size()) Index;
    for (Index i = 0; i < input_flat.size(); ++i) {
      const uint64 input_hash = Hash64(input_flat(i));
      const uint64 bucket_id = input_hash % num_buckets_;
      // The number of buckets is always in the positive range of int64 so is
      // the resulting bucket_id. Casting the bucket_id from uint64 to int64 is
      // safe.
      output_flat(i) = static_cast<int64_t>(bucket_id);
    }
  }

 private:
  int64_t num_buckets_;

  LegacyStringToHashBucketOp(const LegacyStringToHashBucketOp&) = delete;
  void operator=(const LegacyStringToHashBucketOp&) = delete;
};

// StringToHashBucket is deprecated in favor of StringToHashBucketFast/Strong.
REGISTER_KERNEL_BUILDER(Name("StringToHashBucket").Device(DEVICE_CPU),
                        LegacyStringToHashBucketOp);

REGISTER_KERNEL_BUILDER(Name("StringToHashBucketStrong").Device(DEVICE_CPU),
                        StringToKeyedHashBucketOp<StrongKeyedHash>);

}  // namespace tensorflow
