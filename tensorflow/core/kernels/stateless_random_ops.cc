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

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/random_op.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

Status GenerateKey(Tensor seed, random::PhiloxRandom::Key* out_key,
                   random::PhiloxRandom::ResultType* out_counter) {
  // Grab the two seeds
  uint64 seed0;
  uint64 seed1;
  if (seed.dtype() == DT_INT32) {
    const auto seed_vals = seed.flat<int32>();
    seed0 = internal::SubtleMustCopy(seed_vals(0));
    seed1 = internal::SubtleMustCopy(seed_vals(1));
  } else if (seed.dtype() == DT_INT64) {
    const auto seed_vals = seed.flat<int64>();
    seed0 = internal::SubtleMustCopy(seed_vals(0));
    seed1 = internal::SubtleMustCopy(seed_vals(1));
  } else {
    return errors::InvalidArgument("Invalid seed type: ",
                                   DataTypeString(seed.dtype()));
  }

  // Scramble the seeds so that the user doesn't need to worry about which
  // part of the seed needs to be strong.
  (*out_key)[0] = 0x3ec8f720;
  (*out_key)[1] = 0x02461e29;
  (*out_counter)[0] = static_cast<uint32>(seed0);
  (*out_counter)[1] = static_cast<uint32>(seed0 >> 32);
  (*out_counter)[2] = static_cast<uint32>(seed1);
  (*out_counter)[3] = static_cast<uint32>(seed1 >> 32);
  const auto mix = random::PhiloxRandom(*out_counter, *out_key)();
  (*out_key)[0] = mix[0];
  (*out_key)[1] = mix[1];
  (*out_counter)[0] = (*out_counter)[1] = 0;
  (*out_counter)[2] = mix[2];
  (*out_counter)[3] = mix[3];
  return Status::OK();
}

namespace {

class StatelessRandomOpBase : public OpKernel {
 public:
  explicit StatelessRandomOpBase(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Sanitize input
    const Tensor& shape_t = context->input(0);
    const Tensor& seed_t = context->input(1);
    TensorShape shape;
    OP_REQUIRES_OK(context, MakeShape(shape_t, &shape));
    OP_REQUIRES(context, seed_t.dims() == 1 && seed_t.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_t.shape().DebugString()));

    // Allocate output
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output));
    if (shape.num_elements() == 0) return;

    random::PhiloxRandom::Key key;
    random::PhiloxRandom::ResultType counter;
    OP_REQUIRES_OK(context, GenerateKey(seed_t, &key, &counter));

    // Fill in the random numbers
    Fill(context, random::PhiloxRandom(counter, key), output);
  }

  // The part of Compute that depends on device, type, and distribution
  virtual void Fill(OpKernelContext* context, random::PhiloxRandom random,
                    Tensor* output) = 0;
};

template <typename Device, class Distribution>
class StatelessRandomOp : public StatelessRandomOpBase {
 public:
  using StatelessRandomOpBase::StatelessRandomOpBase;

  void Fill(OpKernelContext* context, random::PhiloxRandom random,
            Tensor* output) override {
    typedef typename Distribution::ResultElementType T;
    auto flat = output->flat<T>();
    // Reuse the compute kernels from the stateful random ops
    functor::FillPhiloxRandom<Device, Distribution>()(
        context, context->eigen_device<Device>(), random, flat.data(),
        flat.size(), Distribution());
  }
};

template <typename Device, typename IntType>
class StatelessRandomUniformIntOp : public StatelessRandomOpBase {
 public:
  using StatelessRandomOpBase::StatelessRandomOpBase;

  void Fill(OpKernelContext* context, random::PhiloxRandom random,
            Tensor* output) override {
    const Tensor& minval = context->input(2);
    const Tensor& maxval = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(minval.shape()),
                errors::InvalidArgument("minval must be 0-D, got shape ",
                                        minval.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(maxval.shape()),
                errors::InvalidArgument("maxval must be 0-D, got shape ",
                                        maxval.shape().DebugString()));

    // Verify that minval < maxval.  Note that we'll never reach this point for
    // empty output.  Zero impossible things are fine.
    const auto lo = minval.scalar<IntType>()();
    const auto hi = maxval.scalar<IntType>()();
    OP_REQUIRES(
        context, lo < hi,
        errors::InvalidArgument("Need minval < maxval, got ", lo, " >= ", hi));

    // Build distribution
    typedef random::UniformDistribution<random::PhiloxRandom, IntType>
        Distribution;
    Distribution dist(lo, hi);

    auto flat = output->flat<IntType>();
    // Reuse the compute kernels from the stateful random ops
    functor::FillPhiloxRandom<Device, Distribution>()(
        context, context->eigen_device<Device>(), random, flat.data(),
        flat.size(), dist);
  }
};

#define REGISTER(DEVICE, TYPE)                                              \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("StatelessRandomUniform")                                        \
          .Device(DEVICE_##DEVICE)                                          \
          .HostMemory("shape")                                              \
          .HostMemory("seed")                                               \
          .TypeConstraint<TYPE>("dtype"),                                   \
      StatelessRandomOp<DEVICE##Device, random::UniformDistribution<        \
                                            random::PhiloxRandom, TYPE> >); \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("StatelessRandomNormal")                                         \
          .Device(DEVICE_##DEVICE)                                          \
          .HostMemory("shape")                                              \
          .HostMemory("seed")                                               \
          .TypeConstraint<TYPE>("dtype"),                                   \
      StatelessRandomOp<DEVICE##Device, random::NormalDistribution<         \
                                            random::PhiloxRandom, TYPE> >); \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("StatelessTruncatedNormal")                                      \
          .Device(DEVICE_##DEVICE)                                          \
          .HostMemory("shape")                                              \
          .HostMemory("seed")                                               \
          .TypeConstraint<TYPE>("dtype"),                                   \
      StatelessRandomOp<                                                    \
          DEVICE##Device,                                                   \
          random::TruncatedNormalDistribution<                              \
              random::SingleSampleAdapter<random::PhiloxRandom>, TYPE> >);

#define REGISTER_INT(DEVICE, TYPE)                            \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomUniformInt")   \
                              .Device(DEVICE_##DEVICE)        \
                              .HostMemory("shape")            \
                              .HostMemory("seed")             \
                              .HostMemory("minval")           \
                              .HostMemory("maxval")           \
                              .TypeConstraint<TYPE>("dtype"), \
                          StatelessRandomUniformIntOp<DEVICE##Device, TYPE>);

#define REGISTER_CPU(TYPE) REGISTER(CPU, TYPE)
#define REGISTER_GPU(TYPE) REGISTER(GPU, TYPE)
#define REGISTER_INT_CPU(TYPE) REGISTER_INT(CPU, TYPE)
#define REGISTER_INT_GPU(TYPE) REGISTER_INT(GPU, TYPE)

TF_CALL_half(REGISTER_CPU);
TF_CALL_bfloat16(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);
TF_CALL_int32(REGISTER_INT_CPU);
TF_CALL_int64(REGISTER_INT_CPU);

#if GOOGLE_CUDA

TF_CALL_half(REGISTER_GPU);
TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);
TF_CALL_int32(REGISTER_INT_GPU);
TF_CALL_int64(REGISTER_INT_GPU);

#endif  // GOOGLE_CUDA

#undef REGISTER
#undef REGISTER_INT
#undef REGISTER_CPU
#undef REGISTER_GPU
#undef REGISTER_INT_CPU
#undef REGISTER_INT_GPU

}  // namespace

}  // namespace tensorflow
