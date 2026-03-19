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

// See docs in ../ops/random_ops.cc.

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/shuffle_common.h"
#include "tensorflow/core/util/guarded_philox_random.h"

namespace tensorflow {

template <typename T>
class RandomShuffleOp : public OpKernel {
 public:
  explicit RandomShuffleOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, generator_.Init(context));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    auto get_rng = [this](int64_t num_samples) {
      return generator_.ReserveSamples32(num_samples);
    };
    OP_REQUIRES_OK(ctx,
                   RandomShuffle<T>(ctx, input, /*output_idx=*/0, get_rng));
  }

 private:
  GuardedPhiloxRandom generator_;
};

#define REGISTER(T)                                                    \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("RandomShuffle").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      RandomShuffleOp<T>);

TF_CALL_ALL_TYPES(REGISTER)

#undef REGISTER

}  // namespace tensorflow
