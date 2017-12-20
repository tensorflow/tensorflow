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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/random_op.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

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

    // Grab the two seeds
    uint64 seed0;
    uint64 seed1;
    if (context->input_dtype(1) == DT_INT32) {
      const auto seed = seed_t.flat<int32>();
      seed0 = internal::SubtleMustCopy(seed(0));
      seed1 = internal::SubtleMustCopy(seed(1));
    } else {
      CHECK_EQ(DT_INT64, context->input_dtype(1));
      const auto seed = seed_t.flat<int64>();
      seed0 = internal::SubtleMustCopy(seed(0));
      seed1 = internal::SubtleMustCopy(seed(1));
    }

    // Scramble the seeds so that the user doesn't need to worry about which
    // part of the seed needs to be strong.
    random::PhiloxRandom::Key key;
    random::PhiloxRandom::ResultType counter;
    key[0] = 0x3ec8f720;
    key[1] = 0x02461e29;
    counter[0] = static_cast<uint32>(seed0);
    counter[1] = static_cast<uint32>(seed0 >> 32);
    counter[2] = static_cast<uint32>(seed1);
    counter[3] = static_cast<uint32>(seed1 >> 32);
    const auto mix = random::PhiloxRandom(counter, key)();
    key[0] = mix[0];
    key[1] = mix[1];
    counter[0] = counter[1] = 0;
    counter[2] = mix[2];
    counter[3] = mix[3];

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

}  // namespace

#define REGISTER(TYPE)                                                 \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("StatelessRandomUniform")                                   \
          .Device(DEVICE_CPU)                                          \
          .HostMemory("shape")                                         \
          .TypeConstraint<TYPE>("dtype"),                              \
      StatelessRandomOp<CPUDevice, random::UniformDistribution<        \
                                       random::PhiloxRandom, TYPE> >); \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("StatelessRandomNormal")                                    \
          .Device(DEVICE_CPU)                                          \
          .HostMemory("shape")                                         \
          .TypeConstraint<TYPE>("dtype"),                              \
      StatelessRandomOp<CPUDevice, random::NormalDistribution<         \
                                       random::PhiloxRandom, TYPE> >); \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("StatelessTruncatedNormal")                                 \
          .Device(DEVICE_CPU)                                          \
          .HostMemory("shape")                                         \
          .TypeConstraint<TYPE>("dtype"),                              \
      StatelessRandomOp<                                               \
          CPUDevice,                                                   \
          random::TruncatedNormalDistribution<                         \
              random::SingleSampleAdapter<random::PhiloxRandom>, TYPE> >);

TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#if GOOGLE_CUDA

#define REGISTER(TYPE)                                                 \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("StatelessRandomUniform")                                   \
          .Device(DEVICE_GPU)                                          \
          .HostMemory("shape")                                         \
          .HostMemory("seed")                                          \
          .TypeConstraint<TYPE>("dtype"),                              \
      StatelessRandomOp<GPUDevice, random::UniformDistribution<        \
                                       random::PhiloxRandom, TYPE> >); \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("StatelessRandomNormal")                                    \
          .Device(DEVICE_GPU)                                          \
          .HostMemory("shape")                                         \
          .HostMemory("seed")                                          \
          .TypeConstraint<TYPE>("dtype"),                              \
      StatelessRandomOp<GPUDevice, random::NormalDistribution<         \
                                       random::PhiloxRandom, TYPE> >); \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("StatelessTruncatedNormal")                                 \
          .Device(DEVICE_GPU)                                          \
          .HostMemory("shape")                                         \
          .HostMemory("seed")                                          \
          .TypeConstraint<TYPE>("dtype"),                              \
      StatelessRandomOp<                                               \
          GPUDevice,                                                   \
          random::TruncatedNormalDistribution<                         \
              random::SingleSampleAdapter<random::PhiloxRandom>, TYPE> >);

TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
