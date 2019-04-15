/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
// NOTE: If the algorithm is changed, please run the test
// .../python/kernel_tests/random:random_binomial_test
// commenting out the "tf.set_random_seed(seed)" lines, and using the
// "--runs-per-test=1000" flag. This tests the statistical correctness of the
// op results.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/random_binomial_op.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/stateful_random_ops_cpu_gpu.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/util/work_sharder.h"

#define UNIFORM(X)                                    \
  if (uniform_remaining == 0) {                       \
    uniform_remaining = Uniform::kResultElementCount; \
    uniform_result = uniform(gen);                    \
  }                                                   \
  uniform_remaining--;                                \
  double X = uniform_result[uniform_remaining]

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace {

typedef random::UniformDistribution<random::PhiloxRandom, double> Uniform;

// Binomial inversion. Given prob, sum geometric random variables until they
// exceed count. The number of random variables used is binomially distributed.
// This is also known as binomial inversion, as this is equivalent to inverting
// the Binomial CDF.
double binomial_inversion(double count, double prob,
                          random::PhiloxRandom* gen) {
  using Eigen::numext::ceil;
  using Eigen::numext::log;
  using Eigen::numext::log1p;

  double geom_sum = 0;
  int num_geom = 0;

  Uniform uniform;
  typename Uniform::ResultType uniform_result;
  int16 uniform_remaining = 0;

  while (true) {
    UNIFORM(u);
    double geom = ceil(log(u) / log1p(-prob));
    geom_sum += geom;
    if (geom_sum > count) {
      break;
    }
    ++num_geom;
  }
  return num_geom;
}

double stirling_approx_tail(double k) {
  static double kTailValues[] = {0.0810614667953272,  0.0413406959554092,
                                 0.0276779256849983,  0.02079067210376509,
                                 0.0166446911898211,  0.0138761288230707,
                                 0.0118967099458917,  0.0104112652619720,
                                 0.00925546218271273, 0.00833056343336287};
  if (k <= 9) {
    return kTailValues[static_cast<int>(k)];
  }
  double kp1sq = (k + 1) * (k + 1);
  return (1 / 12 - (1 / 360 + 1 / 1260 / kp1sq) / kp1sq) / (k + 1);
}

// We use a transformation-rejection algorithm from
// pairs of uniform random variables due to Hormann.
// https://www.tandfonline.com/doi/abs/10.1080/00949659308811496
double btrs(double count, double prob, random::PhiloxRandom* gen) {
  using Eigen::numext::abs;
  using Eigen::numext::floor;
  using Eigen::numext::log;
  using Eigen::numext::log1p;
  using Eigen::numext::sqrt;

  // This is spq in the paper.
  const double stddev = sqrt(count * prob * (1 - prob));

  // Other coefficients for Transformed Rejection sampling.
  const double b = 1.15 + 2.53 * stddev;
  const double a = -0.0873 + 0.0248 * b + 0.01 * prob;
  const double c = count * prob + 0.5;
  const double v_r = 0.92 - 4.2 / b;
  const double r = prob / (1 - prob);

  Uniform uniform;
  typename Uniform::ResultType uniform_result;
  int16 uniform_remaining = 0;

  while (true) {
    UNIFORM(u);
    UNIFORM(v);
    u = u - 0.5;
    double us = 0.5 - abs(u);
    double k = floor((2 * a / us + b) * u + c);

    // Region for which the box is tight, and we
    // can return our calculated value This should happen
    // 0.86 * v_r times. In the limit as n * p is large,
    // the acceptance rate converges to ~79% (and in the lower
    // regime it is ~24%).
    if (us >= 0.07 && v <= v_r) {
      return k;
    }
    // Reject non-sensical answers.
    if (k < 0 || k > count) {
      continue;
    }

    double alpha = (2.83 + 5.1 / b) * stddev;
    double m = floor((count + 1) * prob);
    // This deviates from Hormann's BRTS algorithm, as there is a log missing.
    // For all (u, v) pairs outside of the bounding box, this calculates the
    // transformed-reject ratio.
    v = log(v * alpha / (a / (us * us) + b));
    double upperbound =
        ((m + 0.5) * log((m + 1) / (r * (count - m + 1))) +
         (count + 1) * log((count - m + 1) / (count - k + 1)) +
         (k + 0.5) * log(r * (count - k + 1) / (k + 1)) +
         stirling_approx_tail(m) + stirling_approx_tail(count - m) -
         stirling_approx_tail(k) - stirling_approx_tail(count - k));
    if (v <= upperbound) {
      return k;
    }
  }
}

}  // namespace

namespace functor {

template <typename T, typename U>
struct RandomBinomialFunctor<CPUDevice, T, U> {
  void operator()(OpKernelContext* ctx, const CPUDevice& d, int64 num_batches,
                  int64 samples_per_batch, int64 num_elements,
                  typename TTypes<T>::ConstFlat counts,
                  typename TTypes<T>::ConstFlat probs,
                  const random::PhiloxRandom& gen,
                  typename TTypes<U>::Flat output) {
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());

    auto DoWork = [samples_per_batch, num_elements, &counts, &probs, &gen,
                   &output](int start_batch, int limit_batch) {
      // Capturing "gen" by-value would only make a copy for the _shared_
      // lambda.  Since we want to let each worker have its own copy, we pass
      // "gen" by reference and explicitly do a copy assignment here.
      random::PhiloxRandom gen_copy = gen;
      // Skip takes units of 128 bytes.  +3 is so rounding doesn't lead to
      // us using the same state in different batches.
      // The sample from each iteration uses 2 random numbers.
      gen_copy.Skip(start_batch * 2 * 3 * (samples_per_batch + 3) / 4);

      // Vectorized intermediate calculations for uniform rejection sampling.
      // We always generate at most 4 samples.
      Eigen::array<T, 4> z;
      Eigen::array<T, 4> g;

      for (int64 b = start_batch; b < limit_batch; ++b) {
        // We are passed a flat array for each of the parameter tensors.
        // The input is either a scalar broadcasted to all batches or a vector
        // with length num_batches, but the scalar becomes an array of length 1.
        T count = counts((counts.dimension(0) == 1) ? 0 : b);
        T prob = probs((probs.dimension(0) == 1) ? 0 : b);

        // The last batch can be short, if we adjusted num_batches and
        // samples_per_batch.
        const int64 limit_sample =
            std::min((b + 1) * samples_per_batch, num_elements);
        int64 sample = b * samples_per_batch;

        // Calculate normalized samples, then convert them.
        // Determine the method to use.
        double dcount = static_cast<double>(count);
        if (prob <= T(0.5)) {
          double dp = static_cast<double>(prob);
          if (count * prob >= T(10)) {
            while (sample < limit_sample) {
              output(sample) = static_cast<U>(btrs(dcount, dp, &gen_copy));
              sample++;
            }
          } else {
            while (sample < limit_sample) {
              output(sample) =
                  static_cast<U>(binomial_inversion(dcount, dp, &gen_copy));
              sample++;
            }
          }
        } else {
          T q = T(1) - prob;
          double dcount = static_cast<double>(count);
          double dq = static_cast<double>(q);
          if (count * q >= T(10)) {
            while (sample < limit_sample) {
              output(sample) =
                  static_cast<U>(dcount - btrs(dcount, dq, &gen_copy));
              sample++;
            }
          } else {
            while (sample < limit_sample) {
              output(sample) = static_cast<U>(
                  dcount - binomial_inversion(dcount, dq, &gen_copy));
              sample++;
            }
          }
        }
      }
    };

    const int64 batch_init_cost =
        // normMin, normMax
        (Eigen::TensorOpCost::AddCost<T>() +
         Eigen::TensorOpCost::MulCost<T>()) *
            2
        // sqrtFactor
        + Eigen::TensorOpCost::AddCost<T>() +
        Eigen::TensorOpCost::MulCost<T>() +
        Eigen::internal::functor_traits<
            Eigen::internal::scalar_sqrt_op<T>>::Cost
        // cutoff
        + Eigen::TensorOpCost::MulCost<T>() * 4 +
        Eigen::internal::functor_traits<Eigen::internal::scalar_exp_op<T>>::Cost
        // diff
        + Eigen::TensorOpCost::AddCost<T>();
    // This will depend on count * p (or count * q).
    // For n * p < 10, on average, O(n * p) calls to uniform are
    // needed, with that
    // many multiplies. ~10 uniform calls on average with ~200 cost op calls.
    //
    // Very roughly, for rate >= 10, the four calls to log
    // occur for ~72 percent of samples.
    // 4 x 100 (64-bit cycles per log) * 0.72 = ~288
    // Additionally, there are ~10 other ops (+, *, /, ...) at 3-6 cycles each:
    // 40 * .72  = ~25.
    //
    // Finally, there are several other ops that are done every loop along with
    // 2 uniform generations along with 5 other ops at 3-6 cycles each.
    // ~15 / .89 = ~16
    //
    // In total this should be ~529 + 2 * Uniform::kElementCost.
    // We assume that half the tensor has rate < 10, so on average 6
    // uniform's
    // will be needed. We will upper bound the other op cost by the one for
    // rate > 10.
    static const int kElementCost = 529 + 6 * Uniform::kElementCost +
                                    6 * random::PhiloxRandom::kElementCost;
    // Assume we use uniform sampling, and accept the 2nd sample on average.
    const int64 batch_cost = batch_init_cost + kElementCost * samples_per_batch;
    Shard(worker_threads.num_threads, worker_threads.workers, num_batches,
          batch_cost, DoWork);
  }
};

}  // namespace functor

namespace {

// Samples from a binomial distribution, using the given parameters.
template <typename Device, typename T, typename U>
class RandomBinomialOp : public OpKernel {
  // Reshape batches so each batch is this size if possible.
  static const int32 kDesiredBatchSize = 100;

 public:
  explicit RandomBinomialOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& alg_tensor = ctx->input(1);
    const Tensor& shape_tensor = ctx->input(2);
    const Tensor& counts_tensor = ctx->input(3);
    const Tensor& probs_tensor = ctx->input(4);

    OP_REQUIRES(ctx, alg_tensor.dims() == 0,
                errors::InvalidArgument("algorithm must be of shape [], not ",
                                        alg_tensor.shape().DebugString()));
    Algorithm alg = alg_tensor.flat<Algorithm>()(0);

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(shape_tensor.shape()),
        errors::InvalidArgument("Input shape should be a vector, got shape: ",
                                shape_tensor.shape().DebugString()));
    int32 num_batches = shape_tensor.flat<int32>()(0);

    int32 samples_per_batch = 1;
    const int32 num_dims = shape_tensor.dim_size(0);
    for (int32 i = 1; i < num_dims; i++) {
      samples_per_batch *= shape_tensor.flat<int32>()(i);
    }
    const int32 num_elements = num_batches * samples_per_batch;

    // Allocate the output before fudging num_batches and samples_per_batch.
    auto shape_vec = shape_tensor.flat<int32>();
    TensorShape tensor_shape;
    OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(
                            shape_vec.data(), shape_vec.size(), &tensor_shape));
    Tensor* samples_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, tensor_shape, &samples_tensor));

    // Parameters must be 0-d or 1-d.
    OP_REQUIRES(ctx, counts_tensor.dims() <= 1,
                errors::InvalidArgument(
                    "Input counts should be a scalar or vector, got shape: ",
                    counts_tensor.shape().DebugString()));
    OP_REQUIRES(ctx, probs_tensor.dims() <= 1,
                errors::InvalidArgument(
                    "Input probs should be a scalar or vector, got shape: ",
                    probs_tensor.shape().DebugString()));

    if ((counts_tensor.dims() == 0 || counts_tensor.dim_size(0) == 1) &&
        (probs_tensor.dims() == 0 || probs_tensor.dim_size(0) == 1)) {
      // All batches have the same parameters, so we can update the batch size
      // to a reasonable value to improve parallelism (ensure enough batches,
      // and no very small batches which have high overhead).
      int32 size = num_batches * samples_per_batch;
      int32 adjusted_samples = kDesiredBatchSize;
      // Ensure adjusted_batches * adjusted_samples >= size.
      int32 adjusted_batches = Eigen::divup(size, adjusted_samples);
      num_batches = adjusted_batches;
      samples_per_batch = adjusted_samples;
    } else {
      // Parameters must be broadcastable to the shape [num_batches].
      OP_REQUIRES(
          ctx,
          TensorShapeUtils::IsScalar(counts_tensor.shape()) ||
              counts_tensor.dim_size(0) == 1 ||
              counts_tensor.dim_size(0) == num_batches,
          errors::InvalidArgument(
              "Input counts should have length 1 or shape[0], got shape: ",
              counts_tensor.shape().DebugString()));
      OP_REQUIRES(
          ctx,
          TensorShapeUtils::IsScalar(probs_tensor.shape()) ||
              probs_tensor.dim_size(0) == 1 ||
              probs_tensor.dim_size(0) == num_batches,
          errors::InvalidArgument(
              "Input probs should have length 1 or shape[0], got shape: ",
              probs_tensor.shape().DebugString()));
    }
    Var* var = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &var));

    ScopedUnlockUnrefVar var_guard(var);
    Tensor* var_tensor = var->tensor();
    OP_REQUIRES(
        ctx, var_tensor->dtype() == STATE_ELEMENT_DTYPE,
        errors::InvalidArgument("dtype of RNG state variable must be ",
                                DataTypeString(STATE_ELEMENT_DTYPE), ", not ",
                                DataTypeString(var_tensor->dtype())));
    OP_REQUIRES(ctx, var_tensor->dims() == 1,
                errors::InvalidArgument(
                    "RNG state must have one and only one dimension, not ",
                    var_tensor->dims()));
    auto var_tensor_flat = var_tensor->flat<StateElementType>();
    OP_REQUIRES(ctx, alg == RNG_ALG_PHILOX,
                errors::InvalidArgument("Unsupported algorithm id: ", alg));
    static_assert(std::is_same<StateElementType, int64>::value,
                  "StateElementType must be int64");
    static_assert(std::is_same<PhiloxRandom::ResultElementType, uint32>::value,
                  "PhiloxRandom::ResultElementType must be uint32");
    OP_REQUIRES(ctx, var_tensor_flat.size() >= PHILOX_MIN_STATE_SIZE,
                errors::InvalidArgument(
                    "For Philox algorithm, the size of state must be at least ",
                    PHILOX_MIN_STATE_SIZE, "; got ", var_tensor_flat.size()));

    // Each worker has the fudge factor for samples_per_batch, so use it here.
    OP_REQUIRES_OK(ctx, PrepareToUpdateVariable<Device, StateElementType>(
                            ctx, var_tensor, var->copy_on_read_mode.load()));
    auto var_data = var_tensor_flat.data();
    auto philox = GetPhiloxRandomFromMem(var_data);
    UpdateMemWithPhiloxRandom(
        philox, num_batches * 2 * 100 * (samples_per_batch + 3) / 4, var_data);
    var_guard.Release();

    auto binomial_functor = functor::RandomBinomialFunctor<Device, T, U>();
    binomial_functor(ctx, ctx->eigen_device<Device>(), num_batches,
                     samples_per_batch, num_elements, counts_tensor.flat<T>(),
                     probs_tensor.flat<T>(), philox, samples_tensor->flat<U>());
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RandomBinomialOp);
};

}  // namespace

#define REGISTER(RTYPE, TYPE)                                 \
  REGISTER_KERNEL_BUILDER(Name("StatefulRandomBinomial")      \
                              .Device(DEVICE_CPU)             \
                              .HostMemory("resource")         \
                              .HostMemory("algorithm")        \
                              .HostMemory("shape")            \
                              .HostMemory("counts")           \
                              .HostMemory("probs")            \
                              .TypeConstraint<RTYPE>("dtype") \
                              .TypeConstraint<TYPE>("T"),     \
                          RandomBinomialOp<CPUDevice, TYPE, RTYPE>)

#define REGISTER_ALL(RTYPE)     \
  REGISTER(RTYPE, Eigen::half); \
  REGISTER(RTYPE, float);       \
  REGISTER(RTYPE, double);

REGISTER_ALL(Eigen::half);
REGISTER_ALL(float);
REGISTER_ALL(double);
REGISTER_ALL(int32);
REGISTER_ALL(int64);

#undef REGISTER
#undef REGISTER_ALL

}  // end namespace tensorflow
