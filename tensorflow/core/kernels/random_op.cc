/* Copyright 2015 Google Inc. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/random_op.h"

#include <algorithm>
#include <memory>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// The default implementation of the functor, which should never be invoked
// But we still need to provide implementation for now for the linker to work,
// since we do not support all the distributions yet.
template <typename Device, class Distribution>
struct FillPhiloxRandom {
  typedef typename Distribution::ResultElementType T;
  void operator()(OpKernelContext*, const Device&, random::PhiloxRandom gen,
                  T* data, int64 size, Distribution dist) {
    LOG(FATAL) << "Default FillPhiloxRandom should not be executed.";
  }
};

#if GOOGLE_CUDA
// Declaration for the partial specialization with GPU
template <class Distribution>
struct FillPhiloxRandom<GPUDevice, Distribution> {
  typedef typename Distribution::ResultElementType T;
  void operator()(OpKernelContext* ctx, const GPUDevice&,
                  random::PhiloxRandom gen, T* data, int64 size,
                  Distribution dist);
};

#endif

// A class to fill a specified range of random groups
template <class Distribution, bool VariableSamplesPerOutput>
struct FillPhiloxRandomTask;

// Specialization for distribution that takes a fixed number of samples for
// each output.
template <class Distribution>
struct FillPhiloxRandomTask<Distribution, false> {
  typedef typename Distribution::ResultElementType T;
  static void Run(random::PhiloxRandom gen, T* data, int64 size,
                  int64 start_group, int64 limit_group, Distribution dist) {
    const int kGroupSize = Distribution::kResultElementCount;

    gen.Skip(start_group);
    int64 offset = start_group * kGroupSize;

    // First fill all the full-size groups
    int64 limit_group_full = std::min(limit_group, size / kGroupSize);
    for (int64 index = start_group; index < limit_group_full; ++index) {
      auto samples = dist(&gen);
      std::copy(&samples[0], &samples[0] + kGroupSize, data + offset);
      offset += kGroupSize;
    }

    // If there are any remaining elements that need to be filled, process them
    if (limit_group_full < limit_group) {
      int remaining_size = size - limit_group_full * kGroupSize;
      auto samples = dist(&gen);
      std::copy(&samples[0], &samples[0] + remaining_size, data + offset);
    }
  }
};

// Specialization for distribution that takes a variable number of samples for
// each output. This will be slower due to the generality.
template <class Distribution>
struct FillPhiloxRandomTask<Distribution, true> {
  typedef typename Distribution::ResultElementType T;
  static const int64 kReservedSamplesPerOutput = 256;

  static void Run(random::PhiloxRandom base_gen, T* data, int64 size,
                  int64 start_group, int64 limit_group, Distribution dist) {
    using random::PhiloxRandom;
    using random::SingleSampleAdapter;

    const int kGroupSize = Distribution::kResultElementCount;

    static const int kGeneratorSkipPerOutputGroup =
        kGroupSize * kReservedSamplesPerOutput /
        PhiloxRandom::kResultElementCount;

    int64 offset = start_group * kGroupSize;

    // First fill all the full-size groups
    int64 limit_group_full = std::min(limit_group, size / kGroupSize);
    int64 group_index;
    for (group_index = start_group; group_index < limit_group_full;
         ++group_index) {
      // Reset the generator to the beginning of the output group region
      // This is necessary if we want the results to be independent of order
      // of work
      PhiloxRandom gen = base_gen;
      gen.Skip(group_index * kGeneratorSkipPerOutputGroup);
      SingleSampleAdapter<PhiloxRandom> single_samples(&gen);

      auto samples = dist(&single_samples);
      std::copy(&samples[0], &samples[0] + kGroupSize, data + offset);
      offset += kGroupSize;
    }

    // If there are any remaining elements that need to be filled, process them
    if (limit_group_full < limit_group) {
      PhiloxRandom gen = base_gen;
      gen.Skip(group_index * kGeneratorSkipPerOutputGroup);
      SingleSampleAdapter<PhiloxRandom> single_samples(&gen);

      int remaining_size = size - limit_group_full * kGroupSize;
      auto samples = dist(&single_samples);
      std::copy(&samples[0], &samples[0] + remaining_size, data + offset);
    }
  }
};

// Partial specialization for CPU to fill the entire region with randoms
// It splits the work into several tasks and run them in parallel
template <class Distribution>
struct FillPhiloxRandom<CPUDevice, Distribution> {
  typedef typename Distribution::ResultElementType T;
  void operator()(OpKernelContext* context, const CPUDevice&,
                  random::PhiloxRandom gen, T* data, int64 size,
                  Distribution dist) {
    const int kGroupSize = Distribution::kResultElementCount;

    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());

    int64 total_group_count = (size + kGroupSize - 1) / kGroupSize;

    // Limit to maximum six threads for now. The performance scaling is very
    // sub-linear. Too many threads causes a much worse overall performance.
    int num_workers = 6;
    Shard(num_workers, worker_threads.workers, total_group_count, kGroupSize,
          [&gen, data, size, dist](int64 start_group, int64 limit_group) {
            FillPhiloxRandomTask<
                Distribution,
                Distribution::kVariableSamplesPerOutput>::Run(gen, data, size,
                                                              start_group,
                                                              limit_group,
                                                              dist);
          });
  }
};
}  // namespace functor

namespace {

static Status AllocateOutputWithShape(OpKernelContext* ctx, const Tensor& shape,
                                      int index, Tensor** output) {
  if (!ctx->op_kernel().IsLegacyVector(shape.shape())) {
    return errors::InvalidArgument(
        "shape must be a vector of {int32,int64}, got shape ",
        shape.shape().DebugString());
  }
  if (shape.dtype() == DataType::DT_INT32) {
    auto vec = shape.flat<int32>();
    TensorShape tensor_shape;
    TF_RETURN_IF_ERROR(
        TensorShapeUtils::MakeShape(vec.data(), vec.size(), &tensor_shape));
    TF_RETURN_IF_ERROR(ctx->allocate_output(index, tensor_shape, output));
  } else if (shape.dtype() == DataType::DT_INT64) {
    auto vec = shape.flat<int64>();
    TensorShape tensor_shape;
    TF_RETURN_IF_ERROR(
        TensorShapeUtils::MakeShape(vec.data(), vec.size(), &tensor_shape));
    TF_RETURN_IF_ERROR(ctx->allocate_output(index, tensor_shape, output));
  } else {
    return errors::InvalidArgument("shape must be a vector of {int32,int64}.");
  }
  return Status::OK();
}

// Reserve enough random samples in the generator for the given output count.
// Note that the 256 multiplier is repeated above; do not change it just here.
static random::PhiloxRandom ReserveRandomOutputs(GuardedPhiloxRandom& generator,
                                                 int64 output_count) {
  int64 conservative_sample_count = output_count << 8;
  return generator.ReserveSamples128(conservative_sample_count);
}

// For now, use the same interface as RandomOp, so we can choose either one
// at the run-time.
template <typename Device, class Distribution>
class PhiloxRandomOp : public OpKernel {
 public:
  typedef typename Distribution::ResultElementType T;
  explicit PhiloxRandomOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, generator_.Init(ctx));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape = ctx->input(0);
    Tensor* output;
    OP_REQUIRES_OK(ctx, AllocateOutputWithShape(ctx, shape, 0, &output));
    auto output_flat = output->flat<T>();
    functor::FillPhiloxRandom<Device, Distribution>()(
        ctx, ctx->eigen_device<Device>(),
        ReserveRandomOutputs(generator_, output_flat.size()),
        output_flat.data(), output_flat.size(), Distribution());
  }

 private:
  GuardedPhiloxRandom generator_;
};

template <typename Device, class IntType>
class RandomUniformIntOp : public OpKernel {
 public:
  explicit RandomUniformIntOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, generator_.Init(ctx));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape = ctx->input(0);
    const Tensor& minval = ctx->input(1);
    const Tensor& maxval = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(minval.shape()),
                errors::InvalidArgument("minval must be 0-D, got shape ",
                                        minval.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(maxval.shape()),
                errors::InvalidArgument("maxval must be 0-D, got shape ",
                                        maxval.shape().DebugString()));

    // Verify that minval < maxval
    IntType lo = minval.scalar<IntType>()();
    IntType hi = maxval.scalar<IntType>()();
    OP_REQUIRES(
        ctx, lo < hi,
        errors::InvalidArgument("Need minval < maxval, got ", lo, " >= ", hi));

    // Build distribution
    typedef random::UniformDistribution<random::PhiloxRandom, IntType>
        Distribution;
    Distribution dist(lo, hi);

    Tensor* output;
    OP_REQUIRES_OK(ctx, AllocateOutputWithShape(ctx, shape, 0, &output));
    auto output_flat = output->flat<IntType>();
    functor::FillPhiloxRandom<Device, Distribution>()(
        ctx, ctx->eigen_device<Device>(),
        ReserveRandomOutputs(generator_, output_flat.size()),
        output_flat.data(), output_flat.size(), dist);
  }

 private:
  GuardedPhiloxRandom generator_;
};

}  // namespace

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("RandomUniform")                                         \
          .Device(DEVICE_CPU)                                       \
          .HostMemory("shape")                                      \
          .TypeConstraint<TYPE>("dtype"),                           \
      PhiloxRandomOp<CPUDevice, random::UniformDistribution<        \
                                    random::PhiloxRandom, TYPE> >); \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("RandomStandardNormal")                                  \
          .Device(DEVICE_CPU)                                       \
          .HostMemory("shape")                                      \
          .TypeConstraint<TYPE>("dtype"),                           \
      PhiloxRandomOp<CPUDevice, random::NormalDistribution<         \
                                    random::PhiloxRandom, TYPE> >); \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("TruncatedNormal")                                       \
          .Device(DEVICE_CPU)                                       \
          .HostMemory("shape")                                      \
          .TypeConstraint<TYPE>("dtype"),                           \
      PhiloxRandomOp<                                               \
          CPUDevice,                                                \
          random::TruncatedNormalDistribution<                      \
              random::SingleSampleAdapter<random::PhiloxRandom>, TYPE> >)

#define REGISTER_INT(IntType)                                   \
  REGISTER_KERNEL_BUILDER(Name("RandomUniformInt")              \
                              .Device(DEVICE_CPU)               \
                              .HostMemory("shape")              \
                              .HostMemory("minval")             \
                              .HostMemory("maxval")             \
                              .TypeConstraint<IntType>("Tout"), \
                          RandomUniformIntOp<CPUDevice, IntType>);

REGISTER(Eigen::half);
REGISTER(float);
REGISTER(double);
REGISTER_INT(int32);
REGISTER_INT(int64);

#undef REGISTER
#undef REGISTER_INT

#if GOOGLE_CUDA

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("RandomUniform")                                         \
          .Device(DEVICE_GPU)                                       \
          .HostMemory("shape")                                      \
          .TypeConstraint<int32>("T")                               \
          .TypeConstraint<TYPE>("dtype"),                           \
      PhiloxRandomOp<GPUDevice, random::UniformDistribution<        \
                                    random::PhiloxRandom, TYPE> >); \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("RandomStandardNormal")                                  \
          .Device(DEVICE_GPU)                                       \
          .HostMemory("shape")                                      \
          .TypeConstraint<int32>("T")                               \
          .TypeConstraint<TYPE>("dtype"),                           \
      PhiloxRandomOp<GPUDevice, random::NormalDistribution<         \
                                    random::PhiloxRandom, TYPE> >); \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("TruncatedNormal")                                       \
          .Device(DEVICE_GPU)                                       \
          .HostMemory("shape")                                      \
          .TypeConstraint<int32>("T")                               \
          .TypeConstraint<TYPE>("dtype"),                           \
      PhiloxRandomOp<                                               \
          GPUDevice,                                                \
          random::TruncatedNormalDistribution<                      \
              random::SingleSampleAdapter<random::PhiloxRandom>, TYPE> >)

#define REGISTER_INT(IntType)                                   \
  REGISTER_KERNEL_BUILDER(Name("RandomUniformInt")              \
                              .Device(DEVICE_GPU)               \
                              .HostMemory("shape")              \
                              .HostMemory("minval")             \
                              .HostMemory("maxval")             \
                              .TypeConstraint<int32>("T")       \
                              .TypeConstraint<IntType>("Tout"), \
                          RandomUniformIntOp<GPUDevice, IntType>);

REGISTER(Eigen::half);
REGISTER(float);
REGISTER(double);
REGISTER_INT(int32);
REGISTER_INT(int64);

#undef REGISTER
#undef REGISTER_INT

#endif  // GOOGLE_CUDA

}  // end namespace tensorflow
