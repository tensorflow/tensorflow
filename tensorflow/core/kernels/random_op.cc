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

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

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
void FillPhiloxRandom<CPUDevice, Distribution>::operator()(
    OpKernelContext* context, const CPUDevice&, random::PhiloxRandom gen,
    typename Distribution::ResultElementType* data, int64 size,
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
                                                            limit_group, dist);
        });
}

}  // namespace functor

namespace {

static Status AllocateOutputWithShape(OpKernelContext* ctx, const Tensor& shape,
                                      int index, Tensor** output) {
  TensorShape tensor_shape;
  TF_RETURN_IF_ERROR(ctx->op_kernel().MakeShape(shape, &tensor_shape));
  return ctx->allocate_output(index, tensor_shape, output);
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

    // Allocate output, and exit early if possible
    Tensor* output;
    OP_REQUIRES_OK(ctx, AllocateOutputWithShape(ctx, shape, 0, &output));
    if (output->NumElements() == 0) return;

    // Verify that minval < maxval.  This check intentionally happens after the
    // early exit for empty output.  Zero impossible things are fine.
    IntType lo = minval.scalar<IntType>()();
    IntType hi = maxval.scalar<IntType>()();
    OP_REQUIRES(
        ctx, lo < hi,
        errors::InvalidArgument("Need minval < maxval, got ", lo, " >= ", hi));

    // Build distribution
    typedef random::UniformDistribution<random::PhiloxRandom, IntType>
        Distribution;
    Distribution dist(lo, hi);

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

// Samples from one or more gamma distributions. All internal computations are
// done with double precision for numerical stability.
template <typename T>
class RandomGammaOp : public OpKernel {
 public:
  explicit RandomGammaOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, generator_.Init(context));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_t = ctx->input(0);
    const Tensor& alpha_t = ctx->input(1);

    OP_REQUIRES(ctx,
                TensorShapeUtils::IsVector(shape_t.shape()) &&
                    (shape_t.dtype() == DataType::DT_INT32 ||
                     shape_t.dtype() == DataType::DT_INT64),
                errors::InvalidArgument(
                    "shape must be a vector of {int32,int64}, got shape: ",
                    shape_t.DebugString()));
    TensorShape samples_shape;
    if (shape_t.dtype() == DataType::DT_INT32) {
      auto vec = shape_t.flat<int32>();
      OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(vec.data(), vec.size(),
                                                      &samples_shape));
    } else if (shape_t.dtype() == DataType::DT_INT64) {
      auto vec = shape_t.flat<int64>();
      OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(vec.data(), vec.size(),
                                                      &samples_shape));
    }
    const int64 num_samples = samples_shape.num_elements();

    samples_shape.AppendShape(alpha_t.shape());
    // Allocate output samples.
    Tensor* samples_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, samples_shape, &samples_t));

    if (num_samples == 0) return;

    using random::PhiloxRandom;

    typedef random::NormalDistribution<PhiloxRandom, double> Normal;
    typedef random::UniformDistribution<PhiloxRandom, double> Uniform;
#define UNIFORM(X)                                    \
  if (uniform_remaining == 0) {                       \
    uniform_remaining = Uniform::kResultElementCount; \
    uniform_result = uniform(&gen);                   \
  }                                                   \
  uniform_remaining--;                                \
  double X = uniform_result[uniform_remaining]

    // Each attempt is 95+% successful, and requires 1-2 normal + 1 uniform
    static constexpr int kReservedSamplesPerOutput = 256;

    const auto alpha_flat = alpha_t.flat<T>().data();
    const int64 num_alphas = alpha_t.NumElements();
    OP_REQUIRES(ctx, num_alphas > 0,
                errors::InvalidArgument(
                    "Input alpha should have non-zero element count, got: ",
                    num_alphas));
    auto samples_flat = samples_t->flat<T>().data();
    PhiloxRandom rng = generator_.ReserveRandomOutputs(
        num_samples * num_alphas, kReservedSamplesPerOutput);

    // We partition work first across alphas then across samples-per-alpha to
    // avoid a couple flops which can be done on a per-alpha basis.

    auto DoWork = [num_samples, num_alphas, &rng, samples_flat, alpha_flat](
                      int start_output, int limit_output) {
      using Eigen::numext::exp;
      using Eigen::numext::log;
      using Eigen::numext::pow;

      // Capturing "rng" by-value would only make a copy for the _shared_
      // lambda.  Since we want to let each worker have its own copy, we pass
      // "rng" by reference and explicitly do a copy assignment.

      Normal normal;
      Uniform uniform;
      typename Normal::ResultType norm_result;
      typename Uniform::ResultType uniform_result;
      for (int64 output_idx = start_output; output_idx < limit_output;
           /* output_idx incremented within inner loop below */) {
        int64 alpha_idx = output_idx / num_samples;

        // Instead of +alpha_idx for each sample, we offset the pointer once.
        T* const samples_alpha_offset = samples_flat + alpha_idx;

        // Several calculations can be done on a per-alpha basis.
        const double alpha = static_cast<double>(alpha_flat[alpha_idx]);

        DISABLE_FLOAT_EQUALITY_WARNING
        if (alpha == double(1.0)) {
          ENABLE_FLOAT_EQUALITY_WARNING
          // Sample from an exponential distribution.
          for (int64 sample_idx = output_idx % num_samples;
               sample_idx < num_samples && output_idx < limit_output;
               sample_idx++, output_idx++) {
            // As we want data stable regardless of sharding
            // (including eventually on GPU), we skip on a per-sample basis.
            PhiloxRandom gen = rng;
            gen.Skip(kReservedSamplesPerOutput * output_idx);
            short uniform_remaining = 0;
            UNIFORM(u);
            const double res = -log(1.0 - u);
            samples_alpha_offset[sample_idx * num_alphas] = static_cast<T>(res);
          }       // for (sample_idx)
        } else {  // if alpha != 1.0
          // Transformation-rejection from pairs of uniform and normal random
          // variables. http://dl.acm.org/citation.cfm?id=358414
          //
          // The algorithm has an acceptance rate of ~95% for small alpha (~1),
          // and higher accept rates for higher alpha, so runtime is
          // O(NumAlphas * NumSamples * k) with k ~ 1 / 0.95.
          //
          // For alpha<1, we add one to d=alpha-1/3, and multiply the final
          // result by uniform()^(1/alpha)
          const bool alpha_less_than_one = alpha < 1;
          const double d = alpha + (alpha_less_than_one ? 2.0 / 3 : -1.0 / 3);
          const double c = 1.0 / 3 / sqrt(d);

          // Compute the rest of the samples for the current alpha value.
          for (int64 sample_idx = output_idx % num_samples;
               sample_idx < num_samples && output_idx < limit_output;
               sample_idx++, output_idx++) {
            // Since each sample may use a variable number of normal/uniform
            // samples, and we want data stable regardless of sharding
            // (including eventually on GPU), we skip on a per-sample basis.
            PhiloxRandom gen = rng;
            gen.Skip(kReservedSamplesPerOutput * output_idx);
            short norm_remaining = 0;
            short uniform_remaining = 0;

            // Keep trying until we don't reject a sample. In practice, we will
            // only reject ~5% at worst, for low alpha near 1.
            while (true) {
              if (norm_remaining == 0) {
                norm_remaining = Normal::kResultElementCount;
                norm_result = normal(&gen);
              }
              norm_remaining--;
              const double x = norm_result[norm_remaining];
              double v = 1 + c * x;
              if (v <= 0) {
                continue;
              }
              v = v * v * v;
              UNIFORM(u);
              // The first option in the if is a "squeeze" short-circuit to
              // dodge the two logs. Magic constant sourced from the paper
              // linked above. Upward of .91 of the area covered by the log
              // inequality is covered by the squeeze as well (larger coverage
              // for smaller values of alpha).
              if ((u < 1 - 0.0331 * (x * x) * (x * x)) ||
                  (log(u) < 0.5 * x * x + d * (1 - v + log(v)))) {
                double res = d * v;
                if (alpha_less_than_one) {
                  UNIFORM(b);
                  res *= pow(b, 1 / alpha);
                }
                samples_alpha_offset[sample_idx * num_alphas] =
                    static_cast<T>(res);
                break;
              }
            }  // while: true
          }    // for: sample_idx
        }      // if (alpha == 1.0)
      }        // for: output_idx
    };         // DoWork
#undef UNIFORM
    // Two calls to log only occur for ~10% of samples reaching the log line.
    //   2 x 100 (64-bit cycles per log) x 0.10 = ~20.
    // Other ops: sqrt, +, *, /, %... something like 15 of these, at 3-6 cycles
    // each = ~60.
    // All of this /0.95 due to the rejection possibility = ~85.
    static const int kElementCost = 85 + 2 * Normal::kElementCost +
                                    Uniform::kElementCost +
                                    3 * PhiloxRandom::kElementCost;
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers,
          num_alphas * num_samples, kElementCost, DoWork);
  }

 private:
  GuardedPhiloxRandom generator_;

  TF_DISALLOW_COPY_AND_ASSIGN(RandomGammaOp);
};

}  // namespace

#define REGISTER(TYPE)                                                         \
  template struct functor::FillPhiloxRandom<                                   \
      CPUDevice, random::UniformDistribution<random::PhiloxRandom, TYPE>>;     \
  template struct functor::FillPhiloxRandom<                                   \
      CPUDevice, random::NormalDistribution<random::PhiloxRandom, TYPE>>;      \
  template struct functor::FillPhiloxRandom<                                   \
      CPUDevice,                                                               \
      random::TruncatedNormalDistribution<                                     \
          random::SingleSampleAdapter<random::PhiloxRandom>, TYPE>>;           \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("RandomUniform")                                                    \
          .Device(DEVICE_CPU)                                                  \
          .HostMemory("shape")                                                 \
          .TypeConstraint<TYPE>("dtype"),                                      \
      PhiloxRandomOp<CPUDevice, random::UniformDistribution<                   \
                                    random::PhiloxRandom, TYPE>>);             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("RandomStandardNormal")                                             \
          .Device(DEVICE_CPU)                                                  \
          .HostMemory("shape")                                                 \
          .TypeConstraint<TYPE>("dtype"),                                      \
      PhiloxRandomOp<CPUDevice,                                                \
                     random::NormalDistribution<random::PhiloxRandom, TYPE>>); \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TruncatedNormal")                                                  \
          .Device(DEVICE_CPU)                                                  \
          .HostMemory("shape")                                                 \
          .TypeConstraint<TYPE>("dtype"),                                      \
      PhiloxRandomOp<                                                          \
          CPUDevice,                                                           \
          random::TruncatedNormalDistribution<                                 \
              random::SingleSampleAdapter<random::PhiloxRandom>, TYPE>>);      \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("RandomGamma").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),        \
      RandomGammaOp<TYPE>)

#define REGISTER_INT(IntType)                                   \
  REGISTER_KERNEL_BUILDER(Name("RandomUniformInt")              \
                              .Device(DEVICE_CPU)               \
                              .HostMemory("shape")              \
                              .HostMemory("minval")             \
                              .HostMemory("maxval")             \
                              .TypeConstraint<IntType>("Tout"), \
                          RandomUniformIntOp<CPUDevice, IntType>);

TF_CALL_half(REGISTER);
TF_CALL_bfloat16(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
TF_CALL_int32(REGISTER_INT);
TF_CALL_int64(REGISTER_INT);

#undef REGISTER
#undef REGISTER_INT

#if GOOGLE_CUDA

#define REGISTER(TYPE)                                                         \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("RandomUniform")                                                    \
          .Device(DEVICE_GPU)                                                  \
          .HostMemory("shape")                                                 \
          .TypeConstraint<int32>("T")                                          \
          .TypeConstraint<TYPE>("dtype"),                                      \
      PhiloxRandomOp<GPUDevice, random::UniformDistribution<                   \
                                    random::PhiloxRandom, TYPE>>);             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("RandomStandardNormal")                                             \
          .Device(DEVICE_GPU)                                                  \
          .HostMemory("shape")                                                 \
          .TypeConstraint<int32>("T")                                          \
          .TypeConstraint<TYPE>("dtype"),                                      \
      PhiloxRandomOp<GPUDevice,                                                \
                     random::NormalDistribution<random::PhiloxRandom, TYPE>>); \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TruncatedNormal")                                                  \
          .Device(DEVICE_GPU)                                                  \
          .HostMemory("shape")                                                 \
          .TypeConstraint<int32>("T")                                          \
          .TypeConstraint<TYPE>("dtype"),                                      \
      PhiloxRandomOp<                                                          \
          GPUDevice,                                                           \
          random::TruncatedNormalDistribution<                                 \
              random::SingleSampleAdapter<random::PhiloxRandom>, TYPE>>);

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

#ifdef TENSORFLOW_USE_SYCL

namespace functor {

using namespace cl;

template <class Distribution, bool VariableSamplesPerOutput>
struct FillPhiloxRandomKernel;

template <class Distribution>
struct FillPhiloxRandomKernel<Distribution, false> {
  typedef typename Distribution::ResultElementType T;
  using write_accessor = sycl::accessor<uint8_t, 1, sycl::access::mode::write,
                                        sycl::access::target::global_buffer>;

  FillPhiloxRandomKernel(write_accessor& data, random::PhiloxRandom& gen,
                         Distribution& dist)
      : data_(data), gen_(gen), dist_(dist) {}

  void operator()(sycl::nd_item<1> item) {
    const size_t kGroupSize = Distribution::kResultElementCount;

    const size_t item_id = item.get_global(0);
    const size_t total_item_count = item.get_global_range();
    size_t offset = item_id * kGroupSize;
    gen_.Skip(item_id);

    const size_t size = data_.get_size() / sizeof(T);
    T* data = ConvertToActualTypeSycl(T, data_);

    while (offset + kGroupSize <= size) {
      const typename Distribution::ResultType samples = dist_(&gen_);
      for (size_t i = 0; i < kGroupSize; ++i) {
        data[offset + i] = samples[i];
      }

      offset += (total_item_count - 1) * kGroupSize;
      gen_.Skip(total_item_count - 1);
    }

    const typename Distribution::ResultType samples = dist_(&gen_);
    for (size_t i = 0; i < kGroupSize; ++i) {
      if (offset >= size) {
        return;
      }
      data[offset] = samples[i];
      ++offset;
    }
  }

 private:
  write_accessor data_;
  random::PhiloxRandom gen_;
  Distribution dist_;
};

template <class Distribution>
struct FillPhiloxRandomKernel<Distribution, true> {
  typedef typename Distribution::ResultElementType T;
  using write_accessor = sycl::accessor<uint8_t, 1, sycl::access::mode::write,
                                        sycl::access::target::global_buffer>;

  FillPhiloxRandomKernel(write_accessor& data, random::PhiloxRandom& gen,
                         Distribution& dist)
      : data_(data), gen_(gen), dist_(dist) {}

  void operator()(sycl::nd_item<1> item) {
    using random::PhiloxRandom;
    using random::SingleSampleAdapter;

    const size_t kReservedSamplesPerOutput = 256;
    const size_t kGroupSize = Distribution::kResultElementCount;
    const size_t kGeneratorSkipPerOutputGroup =
        kGroupSize * kReservedSamplesPerOutput /
        PhiloxRandom::kResultElementCount;

    const size_t item_id = item.get_global(0);
    const size_t total_item_count = item.get_global_range();
    size_t group_index = item_id;
    size_t offset = group_index * kGroupSize;

    T* data = ConvertToActualTypeSycl(T, data_);
    const size_t size = data_.get_size() / sizeof(T);

    while (offset < size) {
      // Since each output takes a variable number of samples, we need to
      // realign the generator to the beginning for the current output group
      PhiloxRandom gen = gen_;
      gen.Skip(group_index * kGeneratorSkipPerOutputGroup);
      SingleSampleAdapter<PhiloxRandom> single_samples(&gen);

      const typename Distribution::ResultType samples = dist_(&single_samples);

      for (size_t i = 0; i < kGroupSize; ++i) {
        if (offset >= size) {
          return;
        }
        data[offset] = samples[i];
        ++offset;
      }

      offset += (total_item_count - 1) * kGroupSize;
      group_index += total_item_count;
    }
  }

 private:
  write_accessor data_;
  random::PhiloxRandom gen_;
  Distribution dist_;
};

template <typename T>
class FillRandomKernel;
// Partial specialization for SYCL to fill the entire region with randoms
// It splits the work into several tasks and run them in parallel
template <class Distribution>
void FillPhiloxRandom<SYCLDevice, Distribution>::operator()(
    OpKernelContext* context, const SYCLDevice& device,
    random::PhiloxRandom gen, typename Distribution::ResultElementType* data,
    int64 size, Distribution dist) {
  const size_t group_size = device.maxSyclThreadsPerBlock();
  const size_t group_count = (size + group_size - 1) / group_size;

  auto buffer = device.get_sycl_buffer(data);

  device.sycl_queue().submit([&](sycl::handler& cgh) {
    auto access = buffer.template get_access<sycl::access::mode::write>(cgh);

    FillPhiloxRandomKernel<Distribution,
                           Distribution::kVariableSamplesPerOutput>
        task(access, gen, dist);
    cgh.parallel_for<class FillRandomKernel<Distribution>>(
        sycl::nd_range<1>(sycl::range<1>(group_count * group_size),
                          sycl::range<1>(group_size)),
        task);
  });
}

}  // namespace functor

#define REGISTER(TYPE)                                                         \
  template struct functor::FillPhiloxRandom<                                   \
      SYCLDevice, random::UniformDistribution<random::PhiloxRandom, TYPE>>;    \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("RandomUniform")                                                    \
          .Device(DEVICE_SYCL)                                                 \
          .HostMemory("shape")                                                 \
          .TypeConstraint<TYPE>("dtype"),                                      \
      PhiloxRandomOp<SYCLDevice, random::UniformDistribution<                  \
                                     random::PhiloxRandom, TYPE>>);            \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("RandomStandardNormal")                                             \
          .Device(DEVICE_SYCL)                                                 \
          .HostMemory("shape")                                                 \
          .TypeConstraint<TYPE>("dtype"),                                      \
      PhiloxRandomOp<SYCLDevice,                                               \
                     random::NormalDistribution<random::PhiloxRandom, TYPE>>); \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TruncatedNormal")                                                  \
          .Device(DEVICE_SYCL)                                                 \
          .HostMemory("shape")                                                 \
          .TypeConstraint<TYPE>("dtype"),                                      \
      PhiloxRandomOp<                                                          \
          SYCLDevice,                                                          \
          random::TruncatedNormalDistribution<                                 \
              random::SingleSampleAdapter<random::PhiloxRandom>, TYPE>>);

#define REGISTER_INT(IntType)                                   \
  REGISTER_KERNEL_BUILDER(Name("RandomUniformInt")              \
                              .Device(DEVICE_SYCL)              \
                              .HostMemory("shape")              \
                              .HostMemory("minval")             \
                              .HostMemory("maxval")             \
                              .TypeConstraint<IntType>("Tout"), \
                          RandomUniformIntOp<SYCLDevice, IntType>);

TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
TF_CALL_int32(REGISTER_INT);
TF_CALL_int64(REGISTER_INT);

#undef REGISTER
#undef REGISTER_INT

#endif  // TENSORFLOW_USE_SYCL

}  // end namespace tensorflow
