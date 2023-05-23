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
#include "tensorflow/core/kernels/stochastic_cast_op.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/random_op.h"
#include "tensorflow/core/kernels/random_ops_util.h"
#include "tensorflow/core/kernels/stateless_random_ops_v2_util.h"
#include "tensorflow/core/lib/random/random_distributions.h"

namespace tensorflow {
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
using Eigen::half;

void internal::StochasticCastOpBase::Compute(OpKernelContext* ctx) {
  OP_REQUIRES_VALUE(auto key_counter_alg_tuple, ctx,
                    GetKeyCounterAlgFromInputs(ctx, 1, 2, 3));
  auto key_t = std::get<0>(key_counter_alg_tuple);
  auto counter_t = std::get<1>(key_counter_alg_tuple);
  auto alg = std::get<2>(key_counter_alg_tuple);

  Tensor* output;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, ctx->input(0).shape(), &output));
  RoundOff(ctx, alg, key_t, counter_t, output);
}

template <typename Device, typename FromType, typename ToType>
class StochasticCastToIntOp : public internal::StochasticCastOpBase {
 public:
  using StochasticCastOpBase::StochasticCastOpBase;

 protected:
  void RoundOff(OpKernelContext* ctx, Algorithm alg, const Tensor& key,
                const Tensor& counter, Tensor* output) override {
    if (alg == RNG_ALG_PHILOX) {
      random::PhiloxRandom gen(*counter.flat<uint64>().data(),
                               *key.flat<uint64>().data());
      output->flat<ToType>() =
          ctx->input(0)
              .flat<FromType>()
              .unaryExpr(Eigen::internal::StochasticRoundToIntOp<
                         FromType, ToType, random::PhiloxRandom>(&gen))
              .template cast<ToType>();
    } else {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("Unsupported algorithm id: ", alg));
    }
  }
};

#define REGISTER_CAST_TO_INT_KERNEL(DEVICE, FROM_TYPE, TO_TYPE) \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("StochasticCastToInt")                               \
          .Device(DEVICE_##DEVICE)                              \
          .HostMemory("alg")                                    \
          .TypeConstraint<FROM_TYPE>("Tin")                     \
          .TypeConstraint<TO_TYPE>("Tout"),                     \
      StochasticCastToIntOp<DEVICE##Device, FROM_TYPE, TO_TYPE>)

#define REGISTER_CAST_TO_INT_CPU_KERNEL(FROM_TYPE, TO_TYPE) \
  REGISTER_CAST_TO_INT_KERNEL(CPU, FROM_TYPE, TO_TYPE)
#define REGISTER_CAST_TO_INT_GPU_KERNEL(FROM_TYPE, TO_TYPE) \
  REGISTER_CAST_TO_INT_KERNEL(GPU, FROM_TYPE, TO_TYPE)
REGISTER_CAST_TO_INT_CPU_KERNEL(half, int8);
REGISTER_CAST_TO_INT_CPU_KERNEL(half, int16);
REGISTER_CAST_TO_INT_CPU_KERNEL(half, int32);

REGISTER_CAST_TO_INT_CPU_KERNEL(bfloat16, int8);
REGISTER_CAST_TO_INT_CPU_KERNEL(bfloat16, int16);
REGISTER_CAST_TO_INT_CPU_KERNEL(bfloat16, int32);

REGISTER_CAST_TO_INT_CPU_KERNEL(float, int8);
REGISTER_CAST_TO_INT_CPU_KERNEL(float, int16);
REGISTER_CAST_TO_INT_CPU_KERNEL(float, int32);

REGISTER_CAST_TO_INT_CPU_KERNEL(double, int8);
REGISTER_CAST_TO_INT_CPU_KERNEL(double, int16);
REGISTER_CAST_TO_INT_CPU_KERNEL(double, int32);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
REGISTER_CAST_TO_INT_CPU_KERNEL(half, int8);
REGISTER_CAST_TO_INT_CPU_KERNEL(half, int16);
REGISTER_CAST_TO_INT_CPU_KERNEL(half, int32);

REGISTER_CAST_TO_INT_GPU_KERNEL(bfloat16, int8);
REGISTER_CAST_TO_INT_GPU_KERNEL(bfloat16, int16);
REGISTER_CAST_TO_INT_GPU_KERNEL(bfloat16, int32);

REGISTER_CAST_TO_INT_GPU_KERNEL(float, int8);
REGISTER_CAST_TO_INT_GPU_KERNEL(float, int16);
REGISTER_CAST_TO_INT_GPU_KERNEL(float, int32);

REGISTER_CAST_TO_INT_GPU_KERNEL(double, int8);
REGISTER_CAST_TO_INT_GPU_KERNEL(double, int16);
REGISTER_CAST_TO_INT_GPU_KERNEL(double, int32);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER
#undef REGISTER_CPU
#undef REGISTER_GPU

}  // namespace tensorflow
