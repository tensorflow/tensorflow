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

// See docs in ../ops/random_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/random_op.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
using random::PhiloxRandom;
using random::SingleSampleAdapter;

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

template <typename Device, typename T>
struct MultinomialFunctor {
  void operator()(OpKernelContext* ctx, const Device& d,
                  typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<float>::Flat noises,
                  typename TTypes<float>::Flat scores,
                  typename TTypes<float>::Flat scratch, int batch_size,
                  int num_classes, int num_samples,
                  const random::PhiloxRandom& gen,
                  typename TTypes<int64>::Matrix output);
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
      int64 remaining_size = size - limit_group_full * kGroupSize;
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

      int64 remaining_size = size - limit_group_full * kGroupSize;
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

    const int kGroupCost =
        random::PhiloxRandom::kResultElementCount *
        (random::PhiloxRandom::kElementCost + Distribution::kElementCost);
    Shard(worker_threads.num_threads, worker_threads.workers, total_group_count,
          kGroupCost,
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

template <typename T>
struct MultinomialFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* ctx, const CPUDevice& d,
                  typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<float>::Flat /* noises */,
                  typename TTypes<float>::Flat /* scores */,
                  typename TTypes<float>::Flat /* scratch */, int batch_size,
                  int num_classes, int num_samples,
                  const random::PhiloxRandom& gen,
                  typename TTypes<int64>::Matrix output) {
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());

    // The implementation only parallelizes by batch.
    //
    // This takes O(BatchSize * NumSamples * log(NumClasses) + NumClasses) CPU
    // time.
    auto DoWork = [num_samples, num_classes, &gen, &output, &logits](
        int64 start_row, int64 limit_row) {
      // Capturing "gen" by-value would only make a copy for the _shared_
      // lambda.  Since we want to let each worker have its own copy, we pass
      // "gen" by reference and explicitly do a copy assignment here.
      random::PhiloxRandom gen_copy = gen;
      // Skip takes units of 128 bytes.  +3 is so rounding doesn't lead to
      // us using the same state in different batches.
      gen_copy.Skip(start_row * (num_samples + 3) / 4);
      random::SimplePhilox simple_philox(&gen_copy);

      std::vector<float> cdf(num_classes);

      for (int64 b = start_row; b < limit_row; ++b) {
        const auto* logits_row = &logits(b, 0);

        // Precompute cumulative probability distribution across classes.
        // Note: This isn't normalized.
        float running_total = 0;
        for (int64 j = 0; j < num_classes; ++j) {
          if (std::isfinite(static_cast<float>(logits_row[j]))) {
            running_total += std::exp(static_cast<float>(logits_row[j]));
          }
          cdf[j] = running_total;
        }
        // Generate each sample.
        for (int64 j = 0; j < num_samples; ++j) {
          float to_find = simple_philox.RandFloat() * running_total;
          auto found_iter = std::upper_bound(cdf.begin(), cdf.end(), to_find);
          output(b, j) = std::distance(cdf.begin(), found_iter);
        }
      }
    };
    // Incredibly rough estimate of clock cycles for DoWork();
    const int64 cost =
        50 * (num_samples * std::log(num_classes) / std::log(2) + num_classes);
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size, cost,
          DoWork);
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
        // Multiplier 256 is the same as in FillPhiloxRandomTask; do not change
        // it just here.
        generator_.ReserveRandomOutputs(output_flat.size(), 256),
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
        // Multiplier 256 is the same as in FillPhiloxRandomTask; do not change
        // it just here.
        generator_.ReserveRandomOutputs(output_flat.size(), 256),
        output_flat.data(), output_flat.size(), dist);
  }

 private:
  GuardedPhiloxRandom generator_;
};

// Samples from a multinomial distribution.
template <typename Device, typename T>
class MultinomialOp : public OpKernel {
 public:
  explicit MultinomialOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, generator_.Init(context));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& logits_t = ctx->input(0);
    const Tensor& num_samples_t = ctx->input(1);

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsMatrix(logits_t.shape()),
        errors::InvalidArgument("Input logits should be a matrix, got shape: ",
                                logits_t.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(num_samples_t.shape()),
                errors::InvalidArgument(
                    "Input num_samples should be a scalar, got shape: ",
                    num_samples_t.shape().DebugString()));

    const int num_samples = num_samples_t.scalar<int>()();
    OP_REQUIRES(ctx, num_samples > 0,
                errors::InvalidArgument(
                    "Input num_samples should be a positive integer, got: ",
                    num_samples));

    const int batch_size = static_cast<int>(logits_t.dim_size(0));
    const int num_classes = static_cast<int>(logits_t.dim_size(1));

    Tensor* samples_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({batch_size, num_samples}),
                                  &samples_t));
    Tensor noises, scores, scratch;  // Scratch space only used for GPU.
    if (std::is_same<Device, GPUDevice>::value) {
      OP_REQUIRES_OK(
          ctx,
          ctx->allocate_temp(
              DT_FLOAT, TensorShape({batch_size, num_samples, num_classes}),
              &noises));
      OP_REQUIRES_OK(
          ctx,
          ctx->allocate_temp(
              DT_FLOAT, TensorShape({batch_size, num_samples, num_classes}),
              &scores));
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(
                   DT_FLOAT, TensorShape({batch_size, num_samples}), &scratch));
    }

    const int num_samples_ceil_4 = (num_samples + 3) / 4 * 4;
    auto rng =
        generator_.ReserveRandomOutputs(batch_size * num_samples_ceil_4, 256);
    functor::MultinomialFunctor<Device, T>()(
        ctx, ctx->eigen_device<Device>(), logits_t.matrix<T>(),
        noises.flat<float>(), scores.flat<float>(), scratch.flat<float>(),
        batch_size, num_classes, num_samples, rng, samples_t->matrix<int64>());
  }

 private:
  GuardedPhiloxRandom generator_;

  TF_DISALLOW_COPY_AND_ASSIGN(MultinomialOp);
};

}  // namespace

#define REGISTER(TYPE)                                                     \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("RandomUniform")                                                \
          .Device(DEVICE_CPU)                                              \
          .HostMemory("shape")                                             \
          .TypeConstraint<TYPE>("dtype"),                                  \
      PhiloxRandomOp<CPUDevice, random::UniformDistribution<               \
                                    random::PhiloxRandom, TYPE> >);        \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("RandomStandardNormal")                                         \
          .Device(DEVICE_CPU)                                              \
          .HostMemory("shape")                                             \
          .TypeConstraint<TYPE>("dtype"),                                  \
      PhiloxRandomOp<CPUDevice, random::NormalDistribution<                \
                                    random::PhiloxRandom, TYPE> >);        \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("TruncatedNormal")                                              \
          .Device(DEVICE_CPU)                                              \
          .HostMemory("shape")                                             \
          .TypeConstraint<TYPE>("dtype"),                                  \
      PhiloxRandomOp<                                                      \
          CPUDevice,                                                       \
          random::TruncatedNormalDistribution<                             \
              random::SingleSampleAdapter<random::PhiloxRandom>, TYPE> >); \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Multinomial").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),    \
      MultinomialOp<CPUDevice, TYPE>)

#define REGISTER_INT(IntType)                                   \
  REGISTER_KERNEL_BUILDER(Name("RandomUniformInt")              \
                              .Device(DEVICE_CPU)               \
                              .HostMemory("shape")              \
                              .HostMemory("minval")             \
                              .HostMemory("maxval")             \
                              .TypeConstraint<IntType>("Tout"), \
                          RandomUniformIntOp<CPUDevice, IntType>);

TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
TF_CALL_int32(REGISTER_INT);
TF_CALL_int64(REGISTER_INT);

#undef REGISTER
#undef REGISTER_INT

#if GOOGLE_CUDA

#define REGISTER(TYPE)                                                    \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("RandomUniform")                                               \
          .Device(DEVICE_GPU)                                             \
          .HostMemory("shape")                                            \
          .TypeConstraint<int32>("T")                                     \
          .TypeConstraint<TYPE>("dtype"),                                 \
      PhiloxRandomOp<GPUDevice, random::UniformDistribution<              \
                                    random::PhiloxRandom, TYPE> >);       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("RandomStandardNormal")                                        \
          .Device(DEVICE_GPU)                                             \
          .HostMemory("shape")                                            \
          .TypeConstraint<int32>("T")                                     \
          .TypeConstraint<TYPE>("dtype"),                                 \
      PhiloxRandomOp<GPUDevice, random::NormalDistribution<               \
                                    random::PhiloxRandom, TYPE> >);       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("TruncatedNormal")                                             \
          .Device(DEVICE_GPU)                                             \
          .HostMemory("shape")                                            \
          .TypeConstraint<int32>("T")                                     \
          .TypeConstraint<TYPE>("dtype"),                                 \
      PhiloxRandomOp<                                                     \
          GPUDevice,                                                      \
          random::TruncatedNormalDistribution<                            \
              random::SingleSampleAdapter<random::PhiloxRandom>, TYPE> >) \
  REGISTER_KERNEL_BUILDER(Name("Multinomial")                             \
                              .Device(DEVICE_GPU)                         \
                              .HostMemory("num_samples")                  \
                              .TypeConstraint<TYPE>("T"),                 \
                          MultinomialOp<GPUDevice, TYPE>)

#define REGISTER_INT(IntType)                                   \
  REGISTER_KERNEL_BUILDER(Name("RandomUniformInt")              \
                              .Device(DEVICE_GPU)               \
                              .HostMemory("shape")              \
                              .HostMemory("minval")             \
                              .HostMemory("maxval")             \
                              .TypeConstraint<int32>("T")       \
                              .TypeConstraint<IntType>("Tout"), \
                          RandomUniformIntOp<GPUDevice, IntType>);

TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
TF_CALL_int32(REGISTER_INT);
TF_CALL_int64(REGISTER_INT);

#undef REGISTER
#undef REGISTER_INT

#endif  // GOOGLE_CUDA

}  // end namespace tensorflow
