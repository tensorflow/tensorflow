/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/stateless_random_gamma_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/stateless_random_ops.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/errors.h"
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

namespace {

// Each attempt to generate a new draw from the Gamma distribution is 95+%
// successful, and requires 1-2 normal + 1 uniform sample.
static constexpr int kReservedSamplesPerOutput = 256;

typedef Eigen::ThreadPoolDevice CPUDevice;

// Buffer that holds multiple samples. Operator()(random::PhiloxRandom*) returns
// a single sample from this buffer. If the buffer is empty, it first generates
// new samples using the provided distribution.
//
// If the call to Distribution::operator() returns samples[0...N-1], then this
// class returns samples in the following order:
//
//   samples[N-1], samples[N-2],..., samples[1], samples[0]
//
// For comparison, random::SingleSampleAdapter returns samples in
// the following order:
//
//   samples[0], samples[1],...,samples[N-2], samples[N-1].
//
template <class Distribution>
class SampleBuffer {
 public:
  typedef typename Distribution::ResultElementType ResultElementType;

  PHILOX_DEVICE_INLINE
  explicit SampleBuffer(Distribution* distribution)
      : distribution_(distribution), remaining_numbers_(0) {}

  PHILOX_DEVICE_INLINE
  ResultElementType operator()(random::PhiloxRandom* random) {
    if (remaining_numbers_ == 0) {
      results_ = (*distribution_)(random);
      remaining_numbers_ = Distribution::kResultElementCount;
    }

    remaining_numbers_--;
    return results_[remaining_numbers_];
  }

  // Mark this buffer as empty. The next call to operator() will fill it
  // with new random numbers.
  PHILOX_DEVICE_INLINE
  void Clear() { remaining_numbers_ = 0; }

 private:
  typedef typename Distribution::ResultType ResultType;

  Distribution* distribution_;
  ResultType results_;
  int remaining_numbers_;
};

};  // namespace

namespace functor {

template <typename T>
struct StatelessRandomGammaFunctor<CPUDevice, T> {
  static Status Fill(OpKernelContext* ctx, const T* alpha_flat,
                     int64 num_alphas, int64 samples_per_alpha,
                     const random::PhiloxRandom& random, T* samples_flat) {
    typedef random::NormalDistribution<random::PhiloxRandom, double> Normal;
    typedef random::UniformDistribution<random::PhiloxRandom, double> Uniform;

    // We partition work first across alphas then across samples-per-alpha to
    // avoid a couple flops which can be done on a per-alpha basis.

    auto DoWork = [samples_per_alpha, num_alphas, &random, samples_flat,
                   alpha_flat](int64 start_output, int64 limit_output) {
      // Capturing "random" by-value would only make a copy for the _shared_
      // lambda.  Since we want to let each worker have its own copy, we pass
      // "random" by reference and explicitly do a copy assignment.

      using Eigen::numext::exp;
      using Eigen::numext::log;
      using Eigen::numext::log1p;
      using Eigen::numext::pow;

      Normal normal;
      Uniform uniform;

      SampleBuffer<Normal> normal_buffer(&normal);
      SampleBuffer<Uniform> uniform_buffer(&uniform);

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
            double u = uniform(&gen)[Uniform::kResultElementCount - 1];
            const double res = -log1p(-u);
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

            // To prevent overwriting SampleBuffer's underlying array with
            // zeros (in tensorflow::random::Array constructor), we just mark
            // the buffer as empty instead of initializing a new SampleBuffer
            // object here. The next call to operator() will fill the buffer
            // with new numbers.
            normal_buffer.Clear();
            uniform_buffer.Clear();

            // Keep trying until we don't reject a sample. In practice, we will
            // only reject ~5% at worst, for low alpha near 1.
            while (true) {
              const double x = normal_buffer(&gen);
              double v = 1 + c * x;
              if (v <= 0) {
                continue;
              }
              v = v * v * v;
              double u = uniform_buffer(&gen);
              // The first option in the if is a "squeeze" short-circuit to
              // dodge the two logs. Magic constant sourced from the paper
              // linked above. Upward of .91 of the area covered by the log
              // inequality is covered by the squeeze as well (larger coverage
              // for smaller values of alpha).
              if ((u < 1 - 0.0331 * (x * x) * (x * x)) ||
                  (log(u) < 0.5 * x * x + d * (1 - v + log(v)))) {
                double res = d * v;
                if (alpha_less_than_one) {
                  double b = uniform_buffer(&gen);
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

    // Two calls to log only occur for ~10% of samples reaching the log line.
    //   2 x 100 (64-bit cycles per log) x 0.10 = ~20.
    // Other ops: sqrt, +, *, /, %... something like 15 of these, at 3-6 cycles
    // each = ~60.
    // All of this /0.95 (expected value of geometric distribution is 1/p) due
    // to the rejection possibility = ~85.
    static const int kElementCost = 85 + 2 * Normal::kElementCost +
                                    Uniform::kElementCost +
                                    3 * random::PhiloxRandom::kElementCost;
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers,
          num_alphas * samples_per_alpha, kElementCost, DoWork);
    return Status::OK();
  }
};

}  // namespace functor

namespace {

template <typename Device, typename T>
class StatelessRandomGammaOp : public OpKernel {
 public:
  explicit StatelessRandomGammaOp(OpKernelConstruction* context)
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

 private:
  void Fill(OpKernelContext* ctx, random::PhiloxRandom random, Tensor* output) {
    const Tensor& alpha_t = ctx->input(2);

    TensorShape samples_shape = output->shape();
    OP_REQUIRES(ctx, TensorShapeUtils::EndsWith(samples_shape, alpha_t.shape()),
                errors::InvalidArgument(
                    "Shape passed in must end with broadcasted shape."));

    const int64 num_alphas = alpha_t.NumElements();
    OP_REQUIRES(ctx, num_alphas > 0,
                errors::InvalidArgument(
                    "Input alpha should have non-zero element count, got: ",
                    num_alphas));

    const int64 samples_per_alpha = samples_shape.num_elements() / num_alphas;
    const auto alpha_flat = alpha_t.flat<T>().data();
    auto samples_flat = output->flat<T>().data();

    OP_REQUIRES_OK(ctx, functor::StatelessRandomGammaFunctor<Device, T>::Fill(
                            ctx, alpha_flat, num_alphas, samples_per_alpha,
                            random, samples_flat));
  }

  TF_DISALLOW_COPY_AND_ASSIGN(StatelessRandomGammaOp);
};

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

}  // namespace
}  // namespace tensorflow
