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

#include "tensorflow/core/kernels/random_poisson_op.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/lib/random/simple_philox.h"
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

#define UNIFORM(X)                                    \
  if (uniform_remaining == 0) {                       \
    uniform_remaining = Uniform::kResultElementCount; \
    uniform_result = uniform(&gen);                   \
  }                                                   \
  uniform_remaining--;                                \
  CT X = uniform_result[uniform_remaining]

namespace tensorflow {
namespace {

static constexpr int kReservedSamplesPerOutput = 256;

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename T>
struct PoissonComputeType {
  typedef double ComputeType;
};

}  // namespace

namespace functor {

template <typename Device, typename T, typename U>
struct PoissonFunctor {
  void operator()(OpKernelContext* ctx, const Device& d, const T* rate_flat,
                  int num_rate, int num_samples,
                  const random::PhiloxRandom& rng, U* samples_flat);
};

template <typename T, typename U>
struct PoissonFunctor<CPUDevice, T, U> {
  void operator()(OpKernelContext* ctx, const CPUDevice& d, const T* rate_flat,
                  int num_rate, int num_samples,
                  const random::PhiloxRandom& rng, U* samples_flat) {
    // Two different algorithms are employed, depending on the size of
    // rate.
    // If rate < 10, we use an algorithm attributed to Knuth:
    // Seminumerical Algorithms. Art of Computer Programming, Volume 2.
    //
    // This algorithm runs in O(rate) time, and will require O(rate)
    // uniform variates.
    //
    // If rate >= 10 we use a transformation-rejection algorithm from
    // pairs of uniform random variables due to Hormann.
    // http://www.sciencedirect.com/science/article/pii/0167668793909974
    //
    // The algorithm has an acceptance rate of ~89% for the smallest rate
    // (~10),
    // and higher accept rates for higher rate, so runtime is
    // O(NumRate * NumSamples * k) with k ~ 1 / 0.89.
    //
    // We partition work first across rates then across
    // samples-per-rate to
    // avoid a couple flops which can be done on a per-rate basis.

    typedef random::UniformDistribution<random::PhiloxRandom, CT> Uniform;

    auto DoWork = [num_samples, num_rate, &rng, samples_flat, rate_flat](
                      int start_output, int limit_output) {
      // Capturing "rng" by value would only make a copy for the _shared_
      // lambda.  Since we want to let each worker have its own copy, we pass
      // "rng" by reference and explicitly do a copy assignment.

      Uniform uniform;
      typename Uniform::ResultType uniform_result;
      for (int64 output_idx = start_output; output_idx < limit_output;
           /* output_idx incremented within inner loop below */) {
        const int64 rate_idx = output_idx / num_samples;

        // Several calculations can be done on a per-rate basis.
        const CT rate = CT(rate_flat[rate_idx]);

        auto samples_rate_output = samples_flat + rate_idx;

        if (rate < CT(10)) {
          // Knuth's algorithm for generating Poisson random variates.
          // Given a Poisson process, the time between events is exponentially
          // distributed. If we have a Poisson process with rate lambda, then,
          // the time between events is distributed Exp(lambda). If X ~
          // Uniform(0, 1), then Y ~ Exp(lambda), where Y = -log(X) / lambda.
          // Thus to simulate a Poisson draw, we can draw X_i ~ Exp(lambda),
          // and N ~ Poisson(lambda), where N is the least number such that
          // \sum_i^N X_i > 1.
          const CT exp_neg_rate = Eigen::numext::exp(-rate);

          // Compute the rest of the samples for the current rate value.
          for (int64 sample_idx = output_idx % num_samples;
               sample_idx < num_samples && output_idx < limit_output;
               sample_idx++, output_idx++) {
            random::PhiloxRandom gen = rng;
            gen.Skip(kReservedSamplesPerOutput * output_idx);
            int16 uniform_remaining = 0;

            CT prod = 1;
            CT x = 0;

            // Keep trying until we surpass e^(-rate). This will take
            // expected time proportional to rate.
            while (true) {
              UNIFORM(u);
              prod = prod * u;
              if (prod <= exp_neg_rate &&
                  x <= CT(Eigen::NumTraits<U>::highest())) {
                samples_rate_output[sample_idx * num_rate] = U(x);
                break;
              }
              x += 1;
            }
          }
          continue;
        }
        // Transformed rejection due to Hormann.
        //
        // Given a CDF F(x), and G(x), a dominating distribution chosen such
        // that it is close to the inverse CDF F^-1(x), compute the following
        // steps:
        //
        // 1) Generate U and V, two independent random variates. Set U = U - 0.5
        // (this step isn't strictly necessary, but is done to make some
        // calculations symmetric and convenient. Henceforth, G is defined on
        // [-0.5, 0.5]).
        //
        // 2) If V <= alpha * F'(G(U)) * G'(U), return floor(G(U)), else return
        // to step 1. alpha is the acceptance probability of the rejection
        // algorithm.
        //
        // For more details on transformed rejection, see:
        // http://citeseer.ist.psu.edu/viewdoc/citations;jsessionid=1BEB35946CC807879F55D42512E5490C?doi=10.1.1.48.3054.
        //
        // The dominating distribution in this case:
        //
        // G(u) = (2 * a / (2 - |u|) + b) * u + c

        using Eigen::numext::log;
        const CT log_rate = log(rate);

        // Constants used to define the dominating distribution. Names taken
        // from Hormann's paper. Constants were chosen to define the tightest
        // G(u) for the inverse Poisson CDF.
        const CT b = CT(0.931) + CT(2.53) * Eigen::numext::sqrt(rate);
        const CT a = CT(-0.059) + CT(0.02483) * b;

        // This is the inverse acceptance rate. At a minimum (when rate = 10),
        // this corresponds to ~75% acceptance. As the rate becomes larger, this
        // approaches ~89%.
        const CT inv_alpha = CT(1.1239) + CT(1.1328) / (b - CT(3.4));

        // Compute the rest of the samples for the current rate value.
        for (int64 sample_idx = output_idx % num_samples;
             sample_idx < num_samples && output_idx < limit_output;
             sample_idx++, output_idx++) {
          random::PhiloxRandom gen = rng;
          gen.Skip(kReservedSamplesPerOutput * output_idx);
          int16 uniform_remaining = 0;

          while (true) {
            UNIFORM(u);
            u -= CT(0.5);
            UNIFORM(v);

            CT u_shifted = CT(0.5) - Eigen::numext::abs(u);
            CT k = Eigen::numext::floor((CT(2) * a / u_shifted + b) * u + rate +
                                        CT(0.43));

            if (k > CT(Eigen::NumTraits<U>::highest())) {
              // retry in case of overflow.
              continue;
            }

            // When alpha * f(G(U)) * G'(U) is close to 1, it is possible to
            // find a rectangle (-u_r, u_r) x (0, v_r) under the curve, such
            // that if v <= v_r and |u| <= u_r, then we can accept.
            // Here v_r = 0.9227 - 3.6224 / (b - 2) and u_r = 0.43.
            if (u_shifted >= CT(0.07) &&
                v <= CT(0.9277) - CT(3.6224) / (b - CT(2))) {
              samples_rate_output[sample_idx * num_rate] = U(k);
              break;
            }

            if (k < 0 || (u_shifted < CT(0.013) && v > u_shifted)) {
              continue;
            }

            // The expression below is equivalent to the computation of step 2)
            // in transformed rejection (v <= alpha * F'(G(u)) * G'(u)).
            CT s = log(v * inv_alpha / (a / (u_shifted * u_shifted) + b));
            CT t = -rate + k * log_rate - Eigen::numext::lgamma(k + 1);
            if (s <= t) {
              samples_rate_output[sample_idx * num_rate] = U(k);
              break;
            }
          }
        }
      }
    };

    // This will depend on rate.
    // For rate < 10, on average, O(rate) calls to uniform are
    // needed, with that
    // many multiplies. ~10 uniform calls on average with ~25 cost op calls.
    //
    // Very roughly, for rate >= 10, the single call to log + call to
    // lgamma
    // occur for ~60 percent of samples.
    // 2 x 100 (64-bit cycles per log) * 0.62 = ~124
    // Additionally, there are ~10 other ops (+, *, /, ...) at 3-6 cycles each:
    // 40 * .62  = ~25.
    //
    // Finally, there are several other ops that are done every loop along with
    // 2 uniform generations along with 5 other ops at 3-6 cycles each.
    // ~15 / .89 = ~16
    //
    // In total this should be ~165 + 2 * Uniform::kElementCost.
    // We assume that half the tensor has rate < 10, so on average 6
    // uniform's
    // will be needed. We will upper bound the other op cost by the one for
    // rate > 10.
    static const int kElementCost = 165 + 6 * Uniform::kElementCost +
                                    6 * random::PhiloxRandom::kElementCost;
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers,
          num_rate * num_samples, kElementCost, DoWork);
  }

 private:
  typedef typename PoissonComputeType<T>::ComputeType CT;
};

}  // namespace functor

namespace {

// Samples from one or more Poisson distributions.
template <typename T, typename U>
class RandomPoissonOp : public OpKernel {
 public:
  explicit RandomPoissonOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, generator_.Init(context));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_t = ctx->input(0);
    const Tensor& rate_t = ctx->input(1);

    TensorShape samples_shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_t, &samples_shape));
    const int64 num_samples = samples_shape.num_elements();

    samples_shape.AppendShape(rate_t.shape());
    // Allocate output samples.
    Tensor* samples_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, samples_shape, &samples_t));
    if (num_samples == 0) return;

    const auto rate_flat = rate_t.flat<T>().data();
    const int64 num_rate = rate_t.NumElements();
    auto samples_flat = samples_t->flat<U>().data();
    random::PhiloxRandom rng = generator_.ReserveRandomOutputs(
        num_samples * num_rate, kReservedSamplesPerOutput);

    functor::PoissonFunctor<CPUDevice, T, U>()(
        ctx, ctx->eigen_device<CPUDevice>(), rate_flat, num_rate, num_samples,
        rng, samples_flat);
  }

 private:
  GuardedPhiloxRandom generator_;

  TF_DISALLOW_COPY_AND_ASSIGN(RandomPoissonOp);
};
}  // namespace

#undef UNIFORM

#define REGISTER(TYPE)                                                        \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("RandomPoisson").Device(DEVICE_CPU).TypeConstraint<TYPE>("dtype"), \
      RandomPoissonOp<TYPE, TYPE>);

TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#define REGISTER_V2(RTYPE, OTYPE)                              \
  REGISTER_KERNEL_BUILDER(Name("RandomPoissonV2")              \
                              .Device(DEVICE_CPU)              \
                              .TypeConstraint<RTYPE>("R")      \
                              .TypeConstraint<OTYPE>("dtype"), \
                          RandomPoissonOp<RTYPE, OTYPE>);

#define REGISTER_ALL(RTYPE)        \
  REGISTER_V2(RTYPE, Eigen::half); \
  REGISTER_V2(RTYPE, float);       \
  REGISTER_V2(RTYPE, double);      \
  REGISTER_V2(RTYPE, int32);       \
  REGISTER_V2(RTYPE, int64);

REGISTER_ALL(Eigen::half);
REGISTER_ALL(float);
REGISTER_ALL(double);
REGISTER_ALL(int32);
REGISTER_ALL(int64);

#undef REGISTER_ALL
#undef REGISTER_V2
#undef REGISTER

}  // end namespace tensorflow
