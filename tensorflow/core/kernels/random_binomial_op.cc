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
#include "tensorflow/core/framework/rng_alg.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/random_ops_util.h"
#include "tensorflow/core/kernels/stateful_random_ops_cpu_gpu.h"
#include "tensorflow/core/kernels/stateless_random_ops.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/bcast.h"
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
  int16_t uniform_remaining = 0;

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

inline double stirling_approx_tail(double k) {
  static double kTailValues[] = {0.0810614667953272,  0.0413406959554092,
                                 0.0276779256849983,  0.02079067210376509,
                                 0.0166446911898211,  0.0138761288230707,
                                 0.0118967099458917,  0.0104112652619720,
                                 0.00925546218271273, 0.00833056343336287};
  if (k <= 9) {
    return kTailValues[static_cast<int>(k)];
  }
  double kp1sq = (k + 1) * (k + 1);
  return (1.0 / 12 - (1.0 / 360 - 1.0 / 1260 / kp1sq) / kp1sq) / (k + 1);
}

// We use a transformation-rejection algorithm from
// pairs of uniform random variables due to Hormann.
// https://www.tandfonline.com/doi/abs/10.1080/00949659308811496
inline double btrs(double count, double prob, random::PhiloxRandom* gen) {
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

  const double alpha = (2.83 + 5.1 / b) * stddev;
  const double m = floor((count + 1) * prob);

  Uniform uniform;
  typename Uniform::ResultType uniform_result;
  int16_t uniform_remaining = 0;

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
  void operator()(OpKernelContext* ctx, const CPUDevice& d, int64_t num_batches,
                  int64_t samples_per_batch, int64_t num_elements,
                  const BCast& bcast, typename TTypes<T>::ConstFlat counts,
                  typename TTypes<T>::ConstFlat probs,
                  const random::PhiloxRandom& gen,
                  typename TTypes<U>::Flat output) {
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());

    // The output layout is [B1, ... Bk, H1, ... Hm]. We have [B1, ... Bk] for
    // the sample shape and [H1, ... Hm] for the batch shape of the samples.
    // We have B1 * ... * Bk samples per batch member we need.
    auto DoWork = [num_batches, samples_per_batch, &bcast, &counts, &probs,
                   &gen, &output](int64_t start_output, int64_t limit_output) {
      // Vectorized intermediate calculations for uniform rejection sampling.
      // We always generate at most 4 samples.
      Eigen::array<T, 4> z;
      Eigen::array<T, 4> g;
      const bool should_bcast = bcast.IsBroadcastingRequired();
      const auto& counts_batch_indices = bcast.x_batch_indices();
      const auto& probs_batch_indices = bcast.y_batch_indices();
      auto output_flat = output.data();

      // We partition work across batches (count, prob) and then across samples
      // per batch member, to avoid extra work.
      for (int64_t output_idx = start_output; output_idx < limit_output;
           // output_idx is incremented with the inner loops below.
      ) {
        int64_t batch_idx = output_idx / samples_per_batch;
        U* const output_batch_offset = output_flat + batch_idx;
        // Generate batch counts from BCast, as it has the right indices to loop
        // over.
        T count, prob;
        if (should_bcast) {
          count = counts(counts_batch_indices[batch_idx]);
          prob = probs(probs_batch_indices[batch_idx]);
        } else {
          count = counts(batch_idx);
          prob = probs(batch_idx);
        }

        // Calculate normalized samples, then convert them.
        // Determine the method to use.
        double dcount = static_cast<double>(count);
        if (dcount <= 0.0 || prob <= T(0.0)) {
          for (int64_t sample_idx = output_idx % samples_per_batch;
               sample_idx < samples_per_batch && output_idx < limit_output;
               ++sample_idx, ++output_idx) {
            output_batch_offset[sample_idx * num_batches] = static_cast<U>(0.0);
          }
        } else if (prob >= T(1.0)) {
          for (int64_t sample_idx = output_idx % samples_per_batch;
               sample_idx < samples_per_batch && output_idx < limit_output;
               ++sample_idx, ++output_idx) {
            output_batch_offset[sample_idx * num_batches] =
                static_cast<U>(dcount);
          }
        } else if (prob <= T(0.5)) {
          double dp = static_cast<double>(prob);
          if (count * prob >= T(10)) {
            for (int64_t sample_idx = output_idx % samples_per_batch;
                 sample_idx < samples_per_batch && output_idx < limit_output;
                 ++sample_idx, ++output_idx) {
              random::PhiloxRandom gen_copy = gen;
              gen_copy.Skip(256 * output_idx);
              output_batch_offset[sample_idx * num_batches] =
                  static_cast<U>(btrs(dcount, dp, &gen_copy));
            }
          } else {
            for (int64_t sample_idx = output_idx % samples_per_batch;
                 sample_idx < samples_per_batch && output_idx < limit_output;
                 ++sample_idx, ++output_idx) {
              random::PhiloxRandom gen_copy = gen;
              // For binomial inversion, we have mean <= 10, variance <= 10.
              // This means on average we need at most 10 number of samples,
              // and for 10 standard deviations, we need 42 samples. We reserve
              // that much.
              gen_copy.Skip(42 * output_idx);
              output_batch_offset[sample_idx * num_batches] =
                  static_cast<U>(binomial_inversion(dcount, dp, &gen_copy));
            }
          }
        } else if (prob > T(0.5)) {
          T q = T(1) - prob;
          double dq = static_cast<double>(q);
          if (count * q >= T(10)) {
            for (int64_t sample_idx = output_idx % samples_per_batch;
                 sample_idx < samples_per_batch && output_idx < limit_output;
                 ++sample_idx, ++output_idx) {
              random::PhiloxRandom gen_copy = gen;
              gen_copy.Skip(256 * output_idx);
              output_batch_offset[sample_idx * num_batches] =
                  static_cast<U>(dcount - btrs(dcount, dq, &gen_copy));
            }
          } else {
            for (int64_t sample_idx = output_idx % samples_per_batch;
                 sample_idx < samples_per_batch && output_idx < limit_output;
                 ++sample_idx, ++output_idx) {
              random::PhiloxRandom gen_copy = gen;
              // For binomial inversion, we have mean <= 10, variance <= 10.
              // This means on average we need at most 10 number of samples,
              // and for 10 standard deviations, we need 42 samples. We reserve
              // that much.
              gen_copy.Skip(42 * output_idx);
              output_batch_offset[sample_idx * num_batches] = static_cast<U>(
                  dcount - binomial_inversion(dcount, dq, &gen_copy));
            }
          }
        } else {  // prob is NaN
          // TODO(srvasude): What should happen if prob is NaN but the output
          // type is an integer (which doesn't have a sentinel for NaN)?  Fail
          // the whole batch sample?  Return a specialized sentinel like -1?
          for (int64_t sample_idx = output_idx % samples_per_batch;
               sample_idx < samples_per_batch && output_idx < limit_output;
               ++sample_idx, ++output_idx) {
            output_batch_offset[sample_idx * num_batches] = static_cast<U>(NAN);
          }
        }
      }
    };

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
    // In total this (rate >= 10) should be ~329 + 2 * Uniform::kElementCost.
    // We assume that half the tensor has rate < 10, so on average 6
    // uniform's
    // will be needed. We will upper bound the other op cost by the one for
    // rate > 10.
    static const int kElementCost = 329 + 6 * Uniform::kElementCost +
                                    6 * random::PhiloxRandom::kElementCost;
    Shard(worker_threads.num_threads, worker_threads.workers, num_elements,
          kElementCost, DoWork);
  }
};

}  // namespace functor

namespace {

// Samples from a binomial distribution, using the given parameters.
template <typename Device, typename T, typename U>
class RandomBinomialOp : public OpKernel {
  // Reshape batches so each batch is this size if possible.
  static constexpr int32_t kDesiredBatchSize = 100;

 public:
  explicit RandomBinomialOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& alg_tensor = ctx->input(1);
    const Tensor& shape_tensor = ctx->input(2);
    const Tensor& counts_tensor = ctx->input(3);
    const Tensor& probs_tensor = ctx->input(4);

    tensorflow::BCast bcast(counts_tensor.shape().dim_sizes(),
                            probs_tensor.shape().dim_sizes(),
                            /*fewer_dims_optimization=*/false,
                            /*return_flattened_batch_indices=*/true);
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "counts and probs must have compatible batch dimensions: ",
                    counts_tensor.shape().DebugString(), " vs. ",
                    probs_tensor.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(shape_tensor.shape()),
        errors::InvalidArgument("Input shape should be a vector, got shape: ",
                                shape_tensor.shape().DebugString()));
    OP_REQUIRES(ctx,
                (shape_tensor.dtype() == DataType::DT_INT32 ||
                 shape_tensor.dtype() == DataType::DT_INT64),
                errors::InvalidArgument(
                    "Input shape should have dtype {int32, int64}."));

    // Let's check that the shape tensor dominates the broadcasted tensor.
    TensorShape bcast_shape = BCast::ToShape(bcast.output_shape());
    TensorShape output_shape;
    if (shape_tensor.dtype() == DataType::DT_INT32) {
      OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(shape_tensor.vec<int32>(),
                                                      &output_shape));
    } else {
      OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(
                              shape_tensor.vec<int64_t>(), &output_shape));
    }
    OP_REQUIRES(ctx, TensorShapeUtils::EndsWith(output_shape, bcast_shape),
                errors::InvalidArgument(
                    "Shape passed in must end with broadcasted shape."));
    // Now that we have a guarantee, we can get the additional dimensions added
    // by sampling.
    OP_REQUIRES(ctx, alg_tensor.dims() == 0,
                errors::InvalidArgument("algorithm must be of shape [], not ",
                                        alg_tensor.shape().DebugString()));
    Algorithm alg = Algorithm(alg_tensor.flat<int64_t>()(0));

    int64_t samples_per_batch = 1;
    const int64_t num_sample_dims =
        (shape_tensor.dim_size(0) - bcast.output_shape().size());
    for (int64_t i = 0; i < num_sample_dims; ++i) {
      samples_per_batch *= shape_tensor.flat<int32>()(i);
    }
    int64_t num_batches = 1;
    for (int64_t i = num_sample_dims; i < shape_tensor.dim_size(0); ++i) {
      num_batches *= shape_tensor.flat<int32>()(i);
    }
    const int64_t num_elements = num_batches * samples_per_batch;

    Tensor* samples_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &samples_tensor));

    core::RefCountPtr<Var> var;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &var));

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
    static_assert(std::is_same<StateElementType, int64_t>::value,
                  "StateElementType must be int64");
    static_assert(std::is_same<PhiloxRandom::ResultElementType, uint32>::value,
                  "PhiloxRandom::ResultElementType must be uint32");
    OP_REQUIRES(ctx, var_tensor_flat.size() >= PHILOX_MIN_STATE_SIZE,
                errors::InvalidArgument(
                    "For Philox algorithm, the size of state must be at least ",
                    PHILOX_MIN_STATE_SIZE, "; got ", var_tensor_flat.size()));

    OP_REQUIRES_OK(ctx, PrepareToUpdateVariable<Device, StateElementType>(
                            ctx, var_tensor, var->copy_on_read_mode.load()));
    auto var_data = var_tensor_flat.data();
    auto philox = GetPhiloxRandomFromMem(var_data);
    UpdateMemWithPhiloxRandom(
        philox, num_batches * 2 * 100 * (samples_per_batch + 3) / 4, var_data);

    auto binomial_functor = functor::RandomBinomialFunctor<Device, T, U>();
    binomial_functor(ctx, ctx->eigen_device<Device>(), num_batches,
                     samples_per_batch, num_elements, bcast,
                     counts_tensor.flat<T>(), probs_tensor.flat<T>(), philox,
                     samples_tensor->flat<U>());
  }

 private:
  RandomBinomialOp(const RandomBinomialOp&) = delete;
  void operator=(const RandomBinomialOp&) = delete;
};

// Samples from a binomial distribution, using the given parameters.
template <typename Device, typename T, typename U>
class StatelessRandomBinomialOp : public OpKernel {
  // Reshape batches so each batch is this size if possible.
  static constexpr int32_t kDesiredBatchSize = 100;

 public:
  explicit StatelessRandomBinomialOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_tensor = ctx->input(0);
    const Tensor& seed_tensor = ctx->input(1);
    const Tensor& counts_tensor = ctx->input(2);
    const Tensor& probs_tensor = ctx->input(3);

    OP_REQUIRES(ctx, seed_tensor.dims() == 1 && seed_tensor.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_tensor.shape().DebugString()));

    tensorflow::BCast bcast(counts_tensor.shape().dim_sizes(),
                            probs_tensor.shape().dim_sizes(),
                            /*fewer_dims_optimization=*/false,
                            /*return_flattened_batch_indices=*/true);
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "counts and probs must have compatible batch dimensions: ",
                    counts_tensor.shape().DebugString(), " vs. ",
                    probs_tensor.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(shape_tensor.shape()),
        errors::InvalidArgument("Input shape should be a vector, got shape: ",
                                shape_tensor.shape().DebugString()));
    OP_REQUIRES(ctx,
                (shape_tensor.dtype() == DataType::DT_INT32 ||
                 shape_tensor.dtype() == DataType::DT_INT64),
                errors::InvalidArgument(
                    "Input shape should have dtype {int32, int64}."));

    // Let's check that the shape tensor dominates the broadcasted tensor.
    TensorShape bcast_shape = BCast::ToShape(bcast.output_shape());
    TensorShape output_shape;
    if (shape_tensor.dtype() == DataType::DT_INT32) {
      OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(shape_tensor.vec<int32>(),
                                                      &output_shape));
    } else {
      OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(
                              shape_tensor.vec<int64_t>(), &output_shape));
    }
    OP_REQUIRES(ctx, TensorShapeUtils::EndsWith(output_shape, bcast_shape),
                errors::InvalidArgument(
                    "Shape passed in must end with broadcasted shape."));
    // Now that we have a guarantee, we can get the additional dimensions added
    // by sampling.
    int64_t samples_per_batch = 1;
    const int64_t num_sample_dims =
        (shape_tensor.dim_size(0) - bcast.output_shape().size());
    for (int64_t i = 0; i < num_sample_dims; ++i) {
      samples_per_batch *= shape_tensor.dtype() == DataType::DT_INT32
                               ? shape_tensor.flat<int32>()(i)
                               : shape_tensor.flat<int64>()(i);
    }
    int64_t num_batches = 1;
    for (int64_t i = num_sample_dims; i < shape_tensor.dim_size(0); ++i) {
      num_batches *= shape_tensor.dtype() == DataType::DT_INT32
                         ? shape_tensor.flat<int32>()(i)
                         : shape_tensor.flat<int64>()(i);
    }
    const int64_t num_elements = num_batches * samples_per_batch;

    Tensor* samples_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &samples_tensor));
    if (output_shape.num_elements() == 0) return;

    random::PhiloxRandom::Key key;
    random::PhiloxRandom::ResultType counter;
    OP_REQUIRES_OK(ctx, GenerateKey(seed_tensor, &key, &counter));

    auto philox = random::PhiloxRandom(counter, key);
    auto binomial_functor = functor::RandomBinomialFunctor<Device, T, U>();
    binomial_functor(ctx, ctx->eigen_device<Device>(), num_batches,
                     samples_per_batch, num_elements, bcast,
                     counts_tensor.flat<T>(), probs_tensor.flat<T>(), philox,
                     samples_tensor->flat<U>());
  }

 private:
  StatelessRandomBinomialOp(const StatelessRandomBinomialOp&) = delete;
  void operator=(const StatelessRandomBinomialOp&) = delete;
};

}  // namespace

#define REGISTER(RTYPE, TYPE)                                        \
  REGISTER_KERNEL_BUILDER(Name("StatefulRandomBinomial")             \
                              .Device(DEVICE_CPU)                    \
                              .HostMemory("resource")                \
                              .HostMemory("algorithm")               \
                              .HostMemory("shape")                   \
                              .HostMemory("counts")                  \
                              .HostMemory("probs")                   \
                              .TypeConstraint<RTYPE>("dtype")        \
                              .TypeConstraint<TYPE>("T"),            \
                          RandomBinomialOp<CPUDevice, TYPE, RTYPE>); \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomBinomial")            \
                              .Device(DEVICE_CPU)                    \
                              .HostMemory("shape")                   \
                              .HostMemory("seed")                    \
                              .HostMemory("counts")                  \
                              .HostMemory("probs")                   \
                              .TypeConstraint<RTYPE>("dtype")        \
                              .TypeConstraint<TYPE>("T"),            \
                          StatelessRandomBinomialOp<CPUDevice, TYPE, RTYPE>)

#define REGISTER_ALL(RTYPE)     \
  REGISTER(RTYPE, Eigen::half); \
  REGISTER(RTYPE, float);       \
  REGISTER(RTYPE, double);

REGISTER_ALL(Eigen::half);
REGISTER_ALL(float);
REGISTER_ALL(double);
REGISTER_ALL(int32);
REGISTER_ALL(int64_t);

#undef REGISTER
#undef REGISTER_ALL

}  // end namespace tensorflow
