/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// Kernels of the StatelessShuffle op.

#include <tuple>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/random_ops_util.h"
#include "tensorflow/core/kernels/shuffle_common.h"
#include "tensorflow/core/kernels/stateless_random_ops_v2_util.h"

namespace tensorflow {

template <typename T>
class StatelessShuffleOp : public OpKernel {
 public:
  explicit StatelessShuffleOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    OP_REQUIRES_VALUE(auto key_counter_alg, ctx,
                      GetKeyCounterAlgFromInputs(ctx, 1, 2, 3));
    auto key_t = std::get<0>(key_counter_alg);
    auto counter_t = std::get<1>(key_counter_alg);
    auto alg = std::get<2>(key_counter_alg);

    if (alg != RNG_ALG_PHILOX) {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("Unsupported algorithm id: ", alg));
    }

    auto rng = GetPhiloxRandomFromCounterKeyMem(counter_t.flat<uint64>().data(),
                                                key_t.flat<uint64>().data());
    auto get_rng = [rng](auto num_samples) { return rng; };
    OP_REQUIRES_OK(ctx,
                   RandomShuffle<T>(ctx, input, /*output_idx=*/0, get_rng));
  }
};

#define REGISTER(T)                                      \
  REGISTER_KERNEL_BUILDER(Name("StatelessShuffle") \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<T>("T"),   \
                          StatelessShuffleOp<T>);

TF_CALL_ALL_TYPES(REGISTER)

#undef REGISTER

}  // namespace tensorflow
