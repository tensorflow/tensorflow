/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/record_yielder.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

class RecordInputOp : public OpKernel {
 public:
  explicit RecordInputOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
#define GETATTR(TYPE, FIELD) \
  TYPE FIELD;                \
  OP_REQUIRES_OK(ctx, ctx->GetAttr(#FIELD, &FIELD));

    GETATTR(string, file_pattern);
    GETATTR(int64, file_random_seed);
    GETATTR(float, file_shuffle_shift_ratio);
    GETATTR(int64, file_buffer_size);
    GETATTR(int64, file_parallelism);
    GETATTR(int64, batch_size);
    GETATTR(string, compression_type);
#undef GETATTR

    OP_REQUIRES_OK(ctx, ctx->GetAttr("compression_type", &compression_type));

    RecordYielder::Options yopts;
    yopts.file_pattern = file_pattern;
    yopts.seed = file_random_seed;
    yopts.bufsize = file_buffer_size;
    yopts.file_shuffle_shift_ratio = file_shuffle_shift_ratio;
    yopts.parallelism = file_parallelism;
    yopts.compression_type = compression_type;
    yielder_ = std::unique_ptr<RecordYielder>(new RecordYielder(ctx, yopts));

    batch_size_ = batch_size;
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor out(DT_STRING, {batch_size_});
    auto t_out = out.flat<tstring>();
    for (int i = 0; i < batch_size_; ++i) {
      OP_REQUIRES_OK(ctx, yielder_->YieldOne(&t_out(i)));
    }
    ctx->set_output(0, out);
  }

 private:
  int64 batch_size_;
  std::unique_ptr<RecordYielder> yielder_;
};

REGISTER_KERNEL_BUILDER(Name("RecordInput").Device(DEVICE_CPU), RecordInputOp);
}  // namespace tensorflow
