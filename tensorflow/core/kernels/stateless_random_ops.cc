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
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/random_op.h"
#include "tensorflow/core/kernels/random_poisson_op.h"
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
    OP_REQUIRES_OK(context, tensor::MakeShape(shape_t, &shape));
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

template <typename Device, typename IntType>
class StatelessRandomUniformFullIntOp : public StatelessRandomOpBase {
 public:
  using StatelessRandomOpBase::StatelessRandomOpBase;

  void Fill(OpKernelContext* context, random::PhiloxRandom random,
            Tensor* output) override {
    // Build distribution
    typedef random::UniformFullIntDistribution<random::PhiloxRandom, IntType>
        Distribution;
    Distribution dist;

    auto flat = output->flat<IntType>();
    // Reuse the compute kernels from the stateful random ops
    functor::FillPhiloxRandom<Device, Distribution>()(
        context, context->eigen_device<Device>(), random, flat.data(),
        flat.size(), dist);
  }
};

// Samples from one or more Poisson distributions.
template <typename T, typename U>
class StatelessRandomPoissonOp : public StatelessRandomOpBase {
 public:
  using StatelessRandomOpBase::StatelessRandomOpBase;

  void Fill(OpKernelContext* ctx, random::PhiloxRandom random,
            Tensor* output) override {
    const Tensor& rate_t = ctx->input(2);

    TensorShape samples_shape = output->shape();
    OP_REQUIRES(ctx, TensorShapeUtils::EndsWith(samples_shape, rate_t.shape()),
                errors::InvalidArgument(
                    "Shape passed in must end with broadcasted shape."));

    const int64 num_rate = rate_t.NumElements();
    const int64 samples_per_rate = samples_shape.num_elements() / num_rate;
    const auto rate_flat = rate_t.flat<T>().data();
    auto samples_flat = output->flat<U>().data();

    functor::PoissonFunctor<CPUDevice, T, U>()(
        ctx, ctx->eigen_device<CPUDevice>(), rate_flat, num_rate,
        samples_per_rate, random, samples_flat);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(StatelessRandomPoissonOp);
};

template <typename Device, typename T>
class StatelessRandomGammaOp : public StatelessRandomOpBase {
 public:
  using StatelessRandomOpBase::StatelessRandomOpBase;

  void Fill(OpKernelContext* ctx, random::PhiloxRandom random,
            Tensor* output) override {
    const Tensor& alpha_t = ctx->input(2);

    TensorShape samples_shape = output->shape();
    OP_REQUIRES(ctx, TensorShapeUtils::EndsWith(samples_shape, alpha_t.shape()),
                errors::InvalidArgument(
                    "Shape passed in must end with broadcasted shape."));

    typedef random::NormalDistribution<random::PhiloxRandom, double> Normal;
    typedef random::UniformDistribution<random::PhiloxRandom, double> Uniform;
#define UNIFORM(X)                                    \
  if (uniform_remaining == 0) {                       \
    uniform_remaining = Uniform::kResultElementCount; \
    uniform_result = uniform(&gen);                   \
  }                                                   \
  uniform_remaining--;                                \
  double X = uniform_result[uniform_remaining]

    // Each attempt is 95+% successful, and requires 1-2 normal + 1 uniform
    static constexpr int kReservedSamplesPerOutput = 256;

    const int64 num_alphas = alpha_t.NumElements();
    OP_REQUIRES(ctx, num_alphas > 0,
                errors::InvalidArgument(
                    "Input alpha should have non-zero element count, got: ",
                    num_alphas));
    const int64 samples_per_alpha = samples_shape.num_elements() / num_alphas;
    const auto alpha_flat = alpha_t.flat<T>().data();
    auto samples_flat = output->flat<T>().data();

    // We partition work first across alphas then across samples-per-alpha to
    // avoid a couple flops which can be done on a per-alpha basis.

    auto DoWork = [samples_per_alpha, num_alphas, &random, samples_flat,
                   alpha_flat](int start_output, int limit_output) {
      // Capturing "random" by-value would only make a copy for the _shared_
      // lambda.  Since we want to let each worker have its own copy, we pass
      // "random" by reference and explicitly do a copy assignment.

      using Eigen::numext::exp;
      using Eigen::numext::log;
      using Eigen::numext::pow;

      Normal normal;
      Uniform uniform;
      typename Normal::ResultType norm_result;
      typename Uniform::ResultType uniform_result;
      for (int64 output_idx = start_output; output_idx < limit_output;
           /* output_idx incremented within inner loop below */) {
        int64 alpha_idx = output_idx / samples_per_alpha;

        // Instead of +alpha_idx for each sample, we offset the pointer once.
        T* const samples_alpha_offset = samples_flat + alpha_idx;

        // Several calculations can be done on a per-alpha basis.
        const double alpha = static_cast<double>(alpha_flat[alpha_idx]);

        DISABLE_FLOAT_EQUALITY_WARNING
        if (alpha == static_cast<double>(1.0)) {
          ENABLE_FLOAT_EQUALITY_WARNING
          // Sample from an exponential distribution.
          for (int64 sample_idx = output_idx % samples_per_alpha;
               sample_idx < samples_per_alpha && output_idx < limit_output;
               sample_idx++, output_idx++) {
            // As we want data stable regardless of sharding
            // (including eventually on GPU), we skip on a per-sample basis.
            random::PhiloxRandom gen = random;
            gen.Skip(kReservedSamplesPerOutput * output_idx);
            int16 uniform_remaining = 0;
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
          for (int64 sample_idx = output_idx % samples_per_alpha;
               sample_idx < samples_per_alpha && output_idx < limit_output;
               sample_idx++, output_idx++) {
            // Since each sample may use a variable number of normal/uniform
            // samples, and we want data stable regardless of sharding
            // (including eventually on GPU), we skip on a per-sample basis.
            random::PhiloxRandom gen = random;
            gen.Skip(kReservedSamplesPerOutput * output_idx);
            int16 norm_remaining = 0;
            int16 uniform_remaining = 0;

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
                                    3 * random::PhiloxRandom::kElementCost;
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers,
          num_alphas * samples_per_alpha, kElementCost, DoWork);
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
              random::SingleSampleAdapter<random::PhiloxRandom>, TYPE> >)

#define REGISTER_FULL_INT(DEVICE, TYPE)     \
  REGISTER_KERNEL_BUILDER(                  \
      Name("StatelessRandomUniformFullInt") \
          .Device(DEVICE_##DEVICE)          \
          .HostMemory("shape")              \
          .HostMemory("seed")               \
          .TypeConstraint<TYPE>("dtype"),   \
      StatelessRandomUniformFullIntOp<DEVICE##Device, TYPE>)

#define REGISTER_INT(DEVICE, TYPE)                            \
  REGISTER_FULL_INT(DEVICE, TYPE);                            \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomUniformInt")   \
                              .Device(DEVICE_##DEVICE)        \
                              .HostMemory("shape")            \
                              .HostMemory("seed")             \
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

#define REGISTER_POISSON(RATE_TYPE, OUT_TYPE)                     \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomPoisson")          \
                              .Device(DEVICE_CPU)                 \
                              .HostMemory("shape")                \
                              .HostMemory("seed")                 \
                              .HostMemory("lam")                  \
                              .TypeConstraint<RATE_TYPE>("Rtype") \
                              .TypeConstraint<OUT_TYPE>("dtype"), \
                          StatelessRandomPoissonOp<RATE_TYPE, OUT_TYPE>)

#define REGISTER_ALL_POISSON(RATE_TYPE)     \
  REGISTER_POISSON(RATE_TYPE, Eigen::half); \
  REGISTER_POISSON(RATE_TYPE, float);       \
  REGISTER_POISSON(RATE_TYPE, double);      \
  REGISTER_POISSON(RATE_TYPE, int32);       \
  REGISTER_POISSON(RATE_TYPE, int64)

TF_CALL_half(REGISTER_ALL_POISSON);
TF_CALL_float(REGISTER_ALL_POISSON);
TF_CALL_double(REGISTER_ALL_POISSON);
TF_CALL_int32(REGISTER_ALL_POISSON);
TF_CALL_int64(REGISTER_ALL_POISSON);

#undef REGISTER_ALL_POISSON
#undef REGISTER_POISSON

#define REGISTER_GAMMA(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomGammaV2")      \
                              .Device(DEVICE_CPU)             \
                              .HostMemory("shape")            \
                              .HostMemory("seed")             \
                              .HostMemory("alpha")            \
                              .TypeConstraint<TYPE>("dtype"), \
                          StatelessRandomGammaOp<CPUDevice, TYPE>)

TF_CALL_half(REGISTER_GAMMA);
TF_CALL_bfloat16(REGISTER_GAMMA);
TF_CALL_float(REGISTER_GAMMA);
TF_CALL_double(REGISTER_GAMMA);

#undef REGISTER_GAMMA

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

TF_CALL_half(REGISTER_GPU);
TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);
TF_CALL_int32(REGISTER_INT_GPU);
TF_CALL_int64(REGISTER_INT_GPU);
TF_CALL_uint32(REGISTER_FULL_INT_GPU);
TF_CALL_uint64(REGISTER_FULL_INT_GPU);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER
#undef REGISTER_INT
#undef REGISTER_CPU
#undef REGISTER_GPU
#undef REGISTER_INT_CPU
#undef REGISTER_INT_GPU
#undef REGISTER_FULL_INT_CPU
#undef REGISTER_FULL_INT_GPU

}  // namespace

}  // namespace tensorflow
