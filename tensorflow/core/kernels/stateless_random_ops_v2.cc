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

#include "tensorflow/core/kernels/stateless_random_ops_v2.h"

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/rng_alg.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/random_op.h"
#include "tensorflow/core/kernels/random_ops_util.h"
#include "tensorflow/core/kernels/random_poisson_op.h"
#include "tensorflow/core/kernels/stateless_random_ops.h"
#include "tensorflow/core/kernels/stateless_random_ops_v2_util.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/work_sharder.h"

#if EIGEN_COMP_GNUC && __cplusplus > 199711L
#define DISABLE_FLOAT_EQUALITY_WARNING \
  _Pragma("GCC diagnostic push")       \
      _Pragma("GCC diagnostic ignored \"-Wfloat-equal\"")
#define ENABLE_FLOAT_EQUALITY_WARNING _Pragma("GCC diagnostic pop")
#else
#define DISABLE_FLOAT_EQUALITY_WARNING
#define ENABLE_FLOAT_EQUALITY_WARNING
#endif

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

StatelessRandomOpBaseWithKeyCounter::StatelessRandomOpBaseWithKeyCounter(
    OpKernelConstruction* ctx)
    : OpKernel(ctx) {}

void StatelessRandomOpBaseWithKeyCounter::Compute(OpKernelContext* ctx) {
  OP_REQUIRES_VALUE(auto key_counter_alg, ctx,
                    GetKeyCounterAlgFromInputs(ctx, 1, 2, 3));
  auto key_t = std::get<0>(key_counter_alg);
  auto counter_t = std::get<1>(key_counter_alg);
  auto alg = std::get<2>(key_counter_alg);

  TensorShape shape;
  OP_REQUIRES_OK(ctx, tensor::MakeShape(ctx->input(0), &shape));

  // Allocate output
  Tensor* output;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
  if (shape.num_elements() == 0) {
    return;
  }

  // Fill in the random numbers
  Fill(ctx, alg, key_t, counter_t, output);
}

namespace {

template <typename Device, typename Distribution>
class StatelessRandomOp : public StatelessRandomOpBaseWithKeyCounter {
 public:
  using StatelessRandomOpBaseWithKeyCounter::
      StatelessRandomOpBaseWithKeyCounter;

 protected:
  void Fill(OpKernelContext* ctx, Algorithm alg, const Tensor& key,
            const Tensor& counter, Tensor* output) override {
    FillRandomTensor<Device, Distribution>(ctx, alg, key, counter,
                                           Distribution(), output);
  }
};

template <typename Device, typename IntType>
class StatelessRandomUniformIntOp : public StatelessRandomOpBaseWithKeyCounter {
 public:
  using StatelessRandomOpBaseWithKeyCounter::
      StatelessRandomOpBaseWithKeyCounter;

 protected:
  void Fill(OpKernelContext* ctx, Algorithm alg, const Tensor& key,
            const Tensor& counter, Tensor* output) override {
    const Tensor& minval = ctx->input(4);
    const Tensor& maxval = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(minval.shape()),
                errors::InvalidArgument("minval must be 0-D, got shape ",
                                        minval.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(maxval.shape()),
                errors::InvalidArgument("maxval must be 0-D, got shape ",
                                        maxval.shape().DebugString()));

    // Verify that minval < maxval.  Note that we'll never reach this point for
    // empty output.  Zero impossible things are fine.
    const auto lo = minval.scalar<IntType>()();
    const auto hi = maxval.scalar<IntType>()();
    OP_REQUIRES(
        ctx, lo < hi,
        errors::InvalidArgument("Need minval < maxval, got ", lo, " >= ", hi));

    // Build distribution
    typedef random::UniformDistribution<random::PhiloxRandom, IntType>
        Distribution;
    Distribution dist(lo, hi);
    FillRandomTensor<Device, Distribution>(ctx, alg, key, counter, dist,
                                           output);
  }
};

template <typename Device, typename IntType>
class StatelessRandomUniformFullIntOp
    : public StatelessRandomOpBaseWithKeyCounter {
 public:
  using StatelessRandomOpBaseWithKeyCounter::
      StatelessRandomOpBaseWithKeyCounter;

 protected:
  void Fill(OpKernelContext* ctx, Algorithm alg, const Tensor& key,
            const Tensor& counter, Tensor* output) override {
    typedef random::UniformFullIntDistribution<random::PhiloxRandom, IntType>
        Distribution;
    FillRandomTensor<Device, Distribution>(ctx, alg, key, counter,
                                           Distribution(), output);
  }
};

class GetKeyCounterAlgOp : public OpKernel {
 public:
  explicit GetKeyCounterAlgOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& seed_t = ctx->input(0);
    OP_REQUIRES(ctx, seed_t.dims() == 1 && seed_t.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_t.shape().DebugString()));
    // Allocate outputs
    Tensor* key_output;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({RNG_KEY_SIZE}), &key_output));
    Tensor* counter_output;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(1, TensorShape({RNG_MAX_COUNTER_SIZE}),
                                        &counter_output));
    Tensor* alg_output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({}), &alg_output));

    random::PhiloxRandom::Key key;
    random::PhiloxRandom::ResultType counter;
    OP_REQUIRES_OK(ctx, GenerateKey(seed_t, &key, &counter));
    WriteKeyToMem(key, key_output->flat<uint64>().data());
    WriteCounterToMem(counter, counter_output->flat<uint64>().data());
    alg_output->flat<int>()(0) = RNG_ALG_PHILOX;
  }
};

class GetKeyCounterOp : public OpKernel {
 public:
  explicit GetKeyCounterOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& seed_t = ctx->input(0);
    OP_REQUIRES(ctx, seed_t.dims() == 1 && seed_t.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_t.shape().DebugString()));
    // Allocate outputs
    Tensor* key_output;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({RNG_KEY_SIZE}), &key_output));
    Tensor* counter_output;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(1, TensorShape({RNG_MAX_COUNTER_SIZE}),
                                        &counter_output));

    random::PhiloxRandom::Key key;
    random::PhiloxRandom::ResultType counter;
    OP_REQUIRES_OK(ctx, GenerateKey(seed_t, &key, &counter));
    WriteKeyToMem(key, key_output->flat<uint64>().data());
    WriteCounterToMem(counter, counter_output->flat<uint64>().data());
  }
};

class GetAlgOp : public OpKernel {
 public:
  explicit GetAlgOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* alg_output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &alg_output));
    alg_output->flat<int>()(0) = RNG_ALG_PHILOX;
  }
};

#define REGISTER(DEVICE, TYPE)                                              \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("StatelessRandomUniformV2")                                      \
          .Device(DEVICE_##DEVICE)                                          \
          .HostMemory("shape")                                              \
          .HostMemory("alg")                                                \
          .TypeConstraint<TYPE>("dtype"),                                   \
      StatelessRandomOp<DEVICE##Device, random::UniformDistribution<        \
                                            random::PhiloxRandom, TYPE> >); \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("StatelessRandomNormalV2")                                       \
          .Device(DEVICE_##DEVICE)                                          \
          .HostMemory("shape")                                              \
          .HostMemory("alg")                                                \
          .TypeConstraint<TYPE>("dtype"),                                   \
      StatelessRandomOp<DEVICE##Device, random::NormalDistribution<         \
                                            random::PhiloxRandom, TYPE> >); \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("StatelessTruncatedNormalV2")                                    \
          .Device(DEVICE_##DEVICE)                                          \
          .HostMemory("shape")                                              \
          .HostMemory("alg")                                                \
          .TypeConstraint<TYPE>("dtype"),                                   \
      StatelessRandomOp<                                                    \
          DEVICE##Device,                                                   \
          random::TruncatedNormalDistribution<                              \
              random::SingleSampleAdapter<random::PhiloxRandom>, TYPE> >)

#define REGISTER_FULL_INT(DEVICE, TYPE)       \
  REGISTER_KERNEL_BUILDER(                    \
      Name("StatelessRandomUniformFullIntV2") \
          .Device(DEVICE_##DEVICE)            \
          .HostMemory("shape")                \
          .HostMemory("alg")                  \
          .TypeConstraint<TYPE>("dtype"),     \
      StatelessRandomUniformFullIntOp<DEVICE##Device, TYPE>)

#define REGISTER_INT(DEVICE, TYPE)                            \
  REGISTER_FULL_INT(DEVICE, TYPE);                            \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomUniformIntV2") \
                              .Device(DEVICE_##DEVICE)        \
                              .HostMemory("shape")            \
                              .HostMemory("alg")              \
                              .HostMemory("minval")           \
                              .HostMemory("maxval")           \
                              .TypeConstraint<TYPE>("dtype"), \
                          StatelessRandomUniformIntOp<DEVICE##Device, TYPE>)

#define REGISTER_CPU(TYPE) REGISTER(CPU, TYPE)
#define REGISTER_GPU(TYPE) REGISTER(GPU, TYPE)
#define REGISTER_INT_CPU(TYPE) REGISTER_INT(CPU, TYPE)
#define REGISTER_INT_GPU(TYPE) REGISTER_INT(GPU, TYPE)
#define REGISTER_FULL_INT_CPU(TYPE) REGISTER_FULL_INT(CPU, TYPE)
#define REGISTER_FULL_INT_GPU(TYPE) REGISTER_FULL_INT(GPU, TYPE)

TF_CALL_half(REGISTER_CPU);
TF_CALL_bfloat16(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);
TF_CALL_int32(REGISTER_INT_CPU);
TF_CALL_int64(REGISTER_INT_CPU);
TF_CALL_uint32(REGISTER_FULL_INT_CPU);
TF_CALL_uint64(REGISTER_FULL_INT_CPU);

#define REGISTER_GET_KCA(DEVICE)                                               \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomGetKeyCounterAlg")              \
                              .Device(DEVICE_##DEVICE)                         \
                              .HostMemory("seed")                              \
                              .HostMemory("key")                               \
                              .HostMemory("counter")                           \
                              .HostMemory("alg"),                              \
                          GetKeyCounterAlgOp)                                  \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomGetKeyCounter")                 \
                              .Device(DEVICE_##DEVICE)                         \
                              .HostMemory("seed")                              \
                              .HostMemory("key")                               \
                              .HostMemory("counter"),                          \
                          GetKeyCounterOp)                                     \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("StatelessRandomGetAlg").Device(DEVICE_##DEVICE).HostMemory("alg"), \
      GetAlgOp)

REGISTER_GET_KCA(CPU);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

TF_CALL_half(REGISTER_GPU);
TF_CALL_bfloat16(REGISTER_GPU);
TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);
TF_CALL_int32(REGISTER_INT_GPU);
TF_CALL_int64(REGISTER_INT_GPU);
TF_CALL_uint32(REGISTER_FULL_INT_GPU);
TF_CALL_uint64(REGISTER_FULL_INT_GPU);

REGISTER_GET_KCA(GPU);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER
#undef REGISTER_INT
#undef REGISTER_CPU
#undef REGISTER_GPU
#undef REGISTER_INT_CPU
#undef REGISTER_INT_GPU
#undef REGISTER_FULL_INT_CPU
#undef REGISTER_FULL_INT_GPU

#undef REGISTER_GET_KCA

}  // namespace

}  // namespace tensorflow
