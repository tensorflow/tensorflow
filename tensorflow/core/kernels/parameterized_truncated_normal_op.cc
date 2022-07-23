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
// NOTE: If the algorithm is changed, please run the test
// .../python/kernel_tests:parameterized_truncated_normal_op_test
// commenting out the "tf.set_random_seed(seed)" lines, and using the
// "--runs-per-test=1000" flag. This tests the statistical correctness of the
// op results.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/parameterized_truncated_normal_op.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/stateless_random_ops.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
using random::PhiloxRandom;

static constexpr int kMaxIterations = 1000;

template <typename T>
struct TruncatedNormalFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* ctx, const CPUDevice& d, int64_t num_batches,
                  int64_t samples_per_batch, int64_t num_elements,
                  typename TTypes<T>::ConstFlat means,
                  typename TTypes<T>::ConstFlat stddevs,
                  typename TTypes<T>::ConstFlat minvals,
                  typename TTypes<T>::ConstFlat maxvals,
                  const random::PhiloxRandom& gen,
                  typename TTypes<T>::Flat output) {
    // The randn rejection sampling is used when the mean and at least this many
    // standard deviations are inside the bounds.
    // The uniform proposal samplers become less efficient as the bounds are
    // further from the mean, the reverse is true for the randn sampler.
    // This number was chosen by empirical benchmarking. If modified, the
    // benchmarks in parameterized_truncated_normal_op_test should also be
    // changed.
    const T kStdDevsInsideBoundsToUseRandnSampler = T(1.3);
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());

    auto do_work = [samples_per_batch, num_elements, &ctx, &means, &stddevs,
                    &minvals, &maxvals, &gen, &output,
                    kStdDevsInsideBoundsToUseRandnSampler](
                       int64_t start_batch, int64_t limit_batch) {
      // Capturing "gen" by-value would only make a copy for the _shared_
      // lambda.  Since we want to let each worker have its own copy, we pass
      // "gen" by reference and explicitly do a copy assignment here.
      random::PhiloxRandom gen_copy = gen;
      // Skip takes units of 128 bytes.  +3 is so rounding doesn't lead to
      // us using the same state in different batches.
      // The sample from each iteration uses 2 random numbers.
      gen_copy.Skip(start_batch * 2 * kMaxIterations * (samples_per_batch + 3) /
                    4);
      using Uniform = random::UniformDistribution<random::PhiloxRandom, T>;
      Uniform dist;
      using Normal = random::NormalDistribution<random::PhiloxRandom, T>;
      Normal normal_dist;

      // Vectorized intermediate calculations for uniform rejection sampling.
      // We always generate at most 4 samples.
      Eigen::array<T, 4> z;
      Eigen::array<T, 4> g;

      for (int64_t b = start_batch; b < limit_batch; ++b) {
        // We are passed a flat array for each of the parameter tensors.
        // The input is either a scalar broadcasted to all batches or a vector
        // with length num_batches, but the scalar becomes an array of length 1.
        T mean = means((means.dimension(0) == 1) ? 0 : b);
        T stddev = stddevs((stddevs.dimension(0) == 1) ? 0 : b);
        T minval = minvals((minvals.dimension(0) == 1) ? 0 : b);
        T maxval = maxvals((maxvals.dimension(0) == 1) ? 0 : b);

        // The last batch can be short, if we adjusted num_batches and
        // samples_per_batch.
        const int64_t limit_sample =
            std::min((b + 1) * samples_per_batch, num_elements);
        int64_t sample = b * samples_per_batch;

        // On GPU, this check will just fill samples with NAN if it fails.
        OP_REQUIRES(ctx,
                    stddev > T(0) && minval < maxval &&
                        (Eigen::numext::isfinite(minval) ||
                         Eigen::numext::isfinite(maxval)),
                    errors::InvalidArgument("Invalid parameters"));

        int num_iterations = 0;

        // If possible, make one-sided bound be the lower bound, or make both
        // bounds positive. Otherwise, the bounds are on either side of the
        // mean.
        if ((Eigen::numext::isinf(minval) && minval < T(0)) || maxval < mean) {
          // Reverse all calculations. normMin and normMax will be flipped.
          std::swap(minval, maxval);
          stddev = -stddev;
        }

        // Calculate normalized samples, then convert them.
        const T normMin = (minval - mean) / stddev;
        const T normMax = (maxval - mean) / stddev;

        // Determine the method to use.
        const T sqrtFactor = Eigen::numext::sqrt((normMin * normMin) + T(4));
        const T cutoff =
            T(2) *
            Eigen::numext::exp(T(0.5) +
                               (normMin * (normMin - sqrtFactor)) / T(4)) /
            (normMin + sqrtFactor);
        const T diff = normMax - normMin;

        if (((normMin < -kStdDevsInsideBoundsToUseRandnSampler) &&
             (normMax >= T(0.))) ||
            ((normMax > kStdDevsInsideBoundsToUseRandnSampler) &&
             (normMin <= T(0.)))) {
          // If the bounds are a least 3 standard deviations from the mean
          // on at least one side then we rejection sample by sampling
          // from the normal distribution and rejecting samples outside
          // the bounds.
          // Under this condition the acceptance rate per iteration should
          // always be ~ 50%. This sampler is more efficient (and more
          // numerically stable when one or both bounds is far from the mean).

          while (sample < limit_sample) {
            const auto randn_sample = normal_dist(&gen_copy);
            const int size = randn_sample.size();

            for (int i = 0; i < size; i++) {
              if ((randn_sample[i] >= normMin) &&
                  (randn_sample[i] <= normMax)) {
                output(sample) = randn_sample[i] * stddev + mean;
                sample++;
                if (sample >= limit_sample) {
                  break;
                }
                num_iterations = 0;
              } else {
                num_iterations++;
                if (num_iterations > kMaxIterations) {
                  // This should never occur because this sampler should
                  // (by the selection criteria above) be used if at least 3
                  // standard deviations of one side of the distribution
                  // is within the limits (so acceptance probability per
                  // iterations >~ 1/2 per iteration).
                  LOG(ERROR) << "TruncatedNormal randn rejection sampler "
                             << "exceeded maximum iterations for "
                             << "normMin=" << normMin << " normMax=" << normMax
                             << " kMaxIterations=" << kMaxIterations;
                  ctx->SetStatus(errors::Internal(
                      "TruncatedNormal randn rejection sampler failed to accept"
                      " a sample."));
                  return;
                }
              }
            }
          }
        } else if (diff < cutoff) {
          // Sample from a uniform distribution on [normMin, normMax].

          const T plusFactor = (normMin < T(0)) ? T(0) : normMin * normMin;

          while (sample < limit_sample) {
            const auto rand = dist(&gen_copy);
            const int size = rand.size();
            // NOTE(ringwalt): These loops seem to only generate packed AVX
            // instructions for float32.
            for (int i = 0; i < size; i++) {
              z[i] = rand[i] * diff + normMin;
            }
            for (int i = 0; i < size; i++) {
              g[i] = (plusFactor - z[i] * z[i]) / T(2.0);
            }

            const auto u = dist(&gen_copy);
            for (int i = 0; i < size; i++) {
              auto accept = u[i] <= Eigen::numext::exp(g[i]);
              if (accept || num_iterations + 1 >= kMaxIterations) {
                // Accept the sample z.
                // If we run out of iterations, just use the current uniform
                // sample, but emit a warning.
                // TODO(jjhunt) For small entropies (relative to the bounds),
                // this sampler is poor and may take many iterations since
                // the proposal distribution is the uniform distribution
                // U(lower_bound, upper_bound).
                if (!accept) {
                  LOG(ERROR) << "TruncatedNormal uniform rejection sampler "
                             << "exceeded max iterations. Sample may contain "
                             << "outliers.";
                  ctx->SetStatus(errors::Internal(
                      "TruncatedNormal uniform rejection sampler failed to "
                      " accept a sample."));
                  return;
                }
                output(sample) = z[i] * stddev + mean;
                sample++;
                if (sample >= limit_sample) {
                  break;
                }
                num_iterations = 0;
              } else {
                num_iterations++;
              }
            }
          }
        } else {
          // Sample from an exponential distribution with alpha maximizing
          // acceptance probability, offset by normMin from the origin.
          // Accept only if less than normMax.
          const T alpha =
              (normMin + Eigen::numext::sqrt((normMin * normMin) + T(4))) /
              T(2);
          while (sample < limit_sample) {
            auto rand = dist(&gen_copy);
            const int size = rand.size();
            int i = 0;
            while (i < size) {
              const T z = -Eigen::numext::log(rand[i]) / alpha + normMin;
              i++;
              const T x = normMin < alpha ? alpha - z : normMin - alpha;
              const T g = Eigen::numext::exp(-x * x / T(2.0));
              const T u = rand[i];
              i++;
              auto accept = (u <= g && z < normMax);
              if (accept || num_iterations + 1 >= kMaxIterations) {
                if (!accept) {
                  LOG(ERROR) << "TruncatedNormal exponential distribution "
                             << "rejection sampler exceeds max iterations. "
                             << "Sample may contain outliers.";
                  ctx->SetStatus(errors::Internal(
                      "TruncatedNormal exponential distribution rejection"
                      " sampler failed to accept a sample."));
                  return;
                }
                output(sample) = z * stddev + mean;
                sample++;
                if (sample >= limit_sample) {
                  break;
                }
                num_iterations = 0;
              } else {
                num_iterations++;
              }
            }
          }
        }
      }
    };
    // The cost of the initial calculations for the batch.
    const int64_t batchInitCost =
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
    const int64_t uniformSampleCost =
        random::PhiloxRandom::kElementCost +
        random::UniformDistribution<random::PhiloxRandom, T>::kElementCost;
    // The cost of a single uniform sampling round.
    const int64_t uniformRejectionSamplingCost =
        uniformSampleCost + Eigen::TensorOpCost::MulCost<T>() +
        Eigen::TensorOpCost::AddCost<T>() +
        Eigen::TensorOpCost::MulCost<T>() * 2 +
        Eigen::TensorOpCost::AddCost<T>() + uniformSampleCost +
        Eigen::internal::functor_traits<
            Eigen::internal::scalar_exp_op<T>>::Cost +
        Eigen::TensorOpCost::MulCost<T>() + Eigen::TensorOpCost::AddCost<T>();
    // Estimate the cost for an entire batch.
    // Assume we use uniform sampling, and accept the 2nd sample on average.
    const int64_t batchCost =
        batchInitCost + uniformRejectionSamplingCost * 2 * samples_per_batch;
    Shard(worker_threads.num_threads, worker_threads.workers, num_batches,
          batchCost, do_work);
  }
};

template <typename T>
struct TruncatedNormalFunctorV2<CPUDevice, T> {
  void operator()(OpKernelContext* ctx, const CPUDevice& d, int64_t num_batches,
                  int64_t samples_per_batch, int64_t num_elements,
                  const BCastList<4>& bcast,
                  typename TTypes<T>::ConstFlat means,
                  typename TTypes<T>::ConstFlat stddevs,
                  typename TTypes<T>::ConstFlat minvals,
                  typename TTypes<T>::ConstFlat maxvals,
                  const random::PhiloxRandom& gen,
                  typename TTypes<T>::Flat output) {
    // The randn rejection sampling is used when the mean and at least this many
    // standard deviations are inside the bounds.
    // The uniform proposal samplers become less efficient as the bounds are
    // further from the mean, the reverse is true for the randn sampler.
    // This number was chosen by empirical benchmarking. If modified, the
    // benchmarks in parameterized_truncated_normal_op_test should also be
    // changed.
    const T kStdDevsInsideBoundsToUseRandnSampler = T(1.3);
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());

    auto do_work = [num_batches, samples_per_batch, &ctx, &bcast, &means,
                    &stddevs, &minvals, &maxvals, &gen, &output,
                    kStdDevsInsideBoundsToUseRandnSampler](
                       int64_t start_output, int64_t limit_output) {
      // Capturing "gen" by-value would only make a copy for the _shared_
      // lambda.  Since we want to let each worker have its own copy, we pass
      // "gen" by reference and explicitly do a copy assignment here.
      random::PhiloxRandom gen_copy = gen;
      using Uniform = random::UniformDistribution<random::PhiloxRandom, T>;
      Uniform dist;
      using Normal = random::NormalDistribution<random::PhiloxRandom, T>;
      Normal normal_dist;
      // Skip takes units of 128 bits. The Uniform::kResultElementCount - 1
      // is so rounding doesn't lead to
      // us using the same state in different workloads.
      // The sample from each iteration uses 2 random numbers.
      gen_copy.Skip((start_output * 2 * kMaxIterations +
                     Uniform::kResultElementCount - 1) /
                    Uniform::kResultElementCount);

      // Vectorized intermediate calculations for uniform rejection sampling.
      // We always generate at most 4 samples.
      Eigen::array<T, Uniform::kResultElementCount> z;
      Eigen::array<T, Uniform::kResultElementCount> g;

      const bool should_bcast = bcast.IsBroadcastingRequired();
      const auto& means_batch_indices = bcast.batch_indices(0);
      const auto& stddevs_batch_indices = bcast.batch_indices(1);
      const auto& minvals_batch_indices = bcast.batch_indices(2);
      const auto& maxvals_batch_indices = bcast.batch_indices(3);
      auto output_flat = output.data();

      // We partition work across batches and then across samples
      // per batch member, to avoid extra work.
      for (int64_t output_idx = start_output; output_idx < limit_output;
           // output_idx is incremented with the inner loops below.
      ) {
        int64_t batch_idx = output_idx / samples_per_batch;
        // The output layout is [samples_per_batch, num_batches]. Thus
        // the output address is sample_idx * num_batches + batch_idx.
        // Below, code will index at output_batch_offset[sample_idx *
        // num_batches] matching this.
        T* const output_batch_offset = output_flat + batch_idx;
        // Generate batch counts from BCast, as it has the right indices to loop
        // over.
        T mean, stddev, minval, maxval;
        if (should_bcast) {
          mean = means(means_batch_indices[batch_idx]);
          stddev = stddevs(stddevs_batch_indices[batch_idx]);
          minval = minvals(minvals_batch_indices[batch_idx]);
          maxval = maxvals(maxvals_batch_indices[batch_idx]);
        } else {
          mean = means(batch_idx);
          stddev = stddevs(batch_idx);
          minval = minvals(batch_idx);
          maxval = maxvals(batch_idx);
        }

        // On GPU, this check will just fill samples with NAN if it fails.
        OP_REQUIRES(ctx,
                    stddev > T(0) && minval < maxval &&
                        (Eigen::numext::isfinite(minval) ||
                         Eigen::numext::isfinite(maxval)),
                    errors::InvalidArgument("Invalid parameters"));

        int num_iterations = 0;

        // If possible, make one-sided bound be the lower bound, or make both
        // bounds positive. Otherwise, the bounds are on either side of the
        // mean.
        if ((Eigen::numext::isinf(minval) && minval < T(0)) || maxval < mean) {
          // Reverse all calculations. normMin and normMax will be flipped.
          std::swap(minval, maxval);
          stddev = -stddev;
        }

        // Calculate normalized samples, then convert them.
        const T normMin = (minval - mean) / stddev;
        const T normMax = (maxval - mean) / stddev;

        // Determine the method to use.
        const T sqrtFactor = Eigen::numext::sqrt((normMin * normMin) + T(4));
        const T cutoff =
            T(2) *
            Eigen::numext::exp(T(0.5) +
                               (normMin * (normMin - sqrtFactor)) / T(4)) /
            (normMin + sqrtFactor);
        const T diff = normMax - normMin;

        if (((normMin < -kStdDevsInsideBoundsToUseRandnSampler) &&
             (normMax >= T(0.))) ||
            ((normMax > kStdDevsInsideBoundsToUseRandnSampler) &&
             (normMin <= T(0.)))) {
          // If the bounds are a least 3 standard deviations from the mean
          // on at least one side then we rejection sample by sampling
          // from the normal distribution and rejecting samples outside
          // the bounds.
          // Under this condition the acceptance rate per iteration should
          // always be ~ 50%. This sampler is more efficient (and more
          // numerically stable when one or both bounds is far from the mean).
          for (int64_t sample_idx = output_idx % samples_per_batch;
               sample_idx < samples_per_batch && output_idx < limit_output;) {
            const auto randn_sample = normal_dist(&gen_copy);
            const int size = randn_sample.size();
            for (int i = 0; i < size; ++i) {
              if ((randn_sample[i] >= normMin) &&
                  (randn_sample[i] <= normMax)) {
                output_batch_offset[sample_idx * num_batches] =
                    randn_sample[i] * stddev + mean;
                ++sample_idx;
                ++output_idx;
                if (sample_idx >= samples_per_batch ||
                    output_idx >= limit_output) {
                  break;
                }
                num_iterations = 0;
              } else {
                ++num_iterations;
                if (num_iterations > kMaxIterations) {
                  // This should never occur because this sampler should
                  // (by the selection criteria above) be used if at least 3
                  // standard deviations of one side of the distribution
                  // is within the limits (so acceptance probability per
                  // iterations >~ 1/2 per iteration).
                  LOG(ERROR) << "TruncatedNormal randn rejection sampler "
                             << "exceeded maximum iterations for "
                             << "normMin=" << normMin << " normMax=" << normMax
                             << " kMaxIterations=" << kMaxIterations;
                  ctx->SetStatus(errors::Internal(
                      "TruncatedNormal randn rejection sampler failed to accept"
                      " a sample."));
                  return;
                }
              }
            }
          }
        } else if (diff < cutoff) {
          // Sample from a uniform distribution on [normMin, normMax].

          const T plusFactor = (normMin < T(0)) ? T(0) : normMin * normMin;

          for (int64_t sample_idx = output_idx % samples_per_batch;
               sample_idx < samples_per_batch && output_idx < limit_output;) {
            const auto rand = dist(&gen_copy);
            const int size = rand.size();
            // NOTE(ringwalt): These loops seem to only generate packed AVX
            // instructions for float32.
            for (int i = 0; i < size; i++) {
              z[i] = rand[i] * diff + normMin;
              g[i] = (plusFactor - z[i] * z[i]) / T(2.0);
            }

            const auto u = dist(&gen_copy);
            for (int i = 0; i < size; i++) {
              auto accept = u[i] <= Eigen::numext::exp(g[i]);
              if (accept || num_iterations + 1 >= kMaxIterations) {
                // Accept the sample z.
                // If we run out of iterations, just use the current uniform
                // sample, but emit a warning.
                // TODO(jjhunt) For small entropies (relative to the bounds),
                // this sampler is poor and may take many iterations since
                // the proposal distribution is the uniform distribution
                // U(lower_bound, upper_bound).
                if (!accept) {
                  LOG(ERROR) << "TruncatedNormal uniform rejection sampler "
                             << "exceeded max iterations. Sample may contain "
                             << "outliers.";
                  ctx->SetStatus(errors::Internal(
                      "TruncatedNormal uniform rejection sampler failed to "
                      " accept a sample."));
                  return;
                }
                output_batch_offset[sample_idx * num_batches] =
                    z[i] * stddev + mean;
                ++sample_idx;
                ++output_idx;
                if (sample_idx >= samples_per_batch ||
                    output_idx >= limit_output) {
                  break;
                }
                num_iterations = 0;
              } else {
                num_iterations++;
              }
            }
          }
        } else {
          // Sample from an exponential distribution with alpha maximizing
          // acceptance probability, offset by normMin from the origin.
          // Accept only if less than normMax.
          const T alpha =
              (normMin + Eigen::numext::sqrt((normMin * normMin) + T(4))) /
              T(2);
          for (int64_t sample_idx = output_idx % samples_per_batch;
               sample_idx < samples_per_batch && output_idx < limit_output;) {
            auto rand = dist(&gen_copy);
            const int size = rand.size();
            int i = 0;
            while (i < size) {
              const T z = -Eigen::numext::log(rand[i]) / alpha + normMin;
              i++;
              const T x = normMin < alpha ? alpha - z : normMin - alpha;
              const T g = Eigen::numext::exp(-x * x / T(2.0));
              const T u = rand[i];
              i++;
              auto accept = (u <= g && z < normMax);
              if (accept || num_iterations + 1 >= kMaxIterations) {
                if (!accept) {
                  LOG(ERROR) << "TruncatedNormal exponential distribution "
                             << "rejection sampler exceeds max iterations. "
                             << "Sample may contain outliers.";
                  ctx->SetStatus(errors::Internal(
                      "TruncatedNormal exponential distribution rejection"
                      " sampler failed to accept a sample."));
                  return;
                }
                output_batch_offset[sample_idx * num_batches] =
                    z * stddev + mean;
                ++sample_idx;
                ++output_idx;
                if (sample_idx >= samples_per_batch ||
                    output_idx >= limit_output) {
                  break;
                }
                num_iterations = 0;
              } else {
                num_iterations++;
              }
            }
          }
        }
      }
    };
    // The cost of the initial calculations for the batch.
    const int64_t batchInitCost =
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
    const int64_t uniformSampleCost =
        random::PhiloxRandom::kElementCost +
        random::UniformDistribution<random::PhiloxRandom, T>::kElementCost;
    // The cost of a single uniform sampling round.
    const int64_t uniformRejectionSamplingCost =
        uniformSampleCost + Eigen::TensorOpCost::MulCost<T>() +
        Eigen::TensorOpCost::AddCost<T>() +
        Eigen::TensorOpCost::MulCost<T>() * 2 +
        Eigen::TensorOpCost::AddCost<T>() + uniformSampleCost +
        Eigen::internal::functor_traits<
            Eigen::internal::scalar_exp_op<T>>::Cost +
        Eigen::TensorOpCost::MulCost<T>() + Eigen::TensorOpCost::AddCost<T>();
    // Estimate the cost for an entire batch.
    // Assume we use uniform sampling, and accept the 2nd sample on average.
    const int64_t batchCost = batchInitCost + uniformRejectionSamplingCost * 2;
    Shard(worker_threads.num_threads, worker_threads.workers, num_elements,
          batchCost, do_work);
  }
};

}  // namespace functor

namespace {

// Samples from a truncated normal distribution, using the given parameters.
template <typename Device, typename T>
class ParameterizedTruncatedNormalOp : public OpKernel {
  // Reshape batches so each batch is this size if possible.
  static constexpr int32_t kDesiredBatchSize = 100;

 public:
  explicit ParameterizedTruncatedNormalOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, generator_.Init(context));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_tensor = ctx->input(0);
    const Tensor& means_tensor = ctx->input(1);
    const Tensor& stddevs_tensor = ctx->input(2);
    const Tensor& minvals_tensor = ctx->input(3);
    const Tensor& maxvals_tensor = ctx->input(4);

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(shape_tensor.shape()),
        errors::InvalidArgument("Input shape should be a vector, got shape: ",
                                shape_tensor.shape().DebugString()));
    OP_REQUIRES(ctx, shape_tensor.NumElements() > 0,
                errors::InvalidArgument("Shape tensor must not be empty, got ",
                                        shape_tensor.DebugString()));
    TensorShape tensor_shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_tensor, &tensor_shape));

    int32_t num_batches = tensor_shape.dim_size(0);
    int32_t samples_per_batch = 1;
    const int32_t num_dims = tensor_shape.dims();
    for (int32_t i = 1; i < num_dims; i++) {
      samples_per_batch *= tensor_shape.dim_size(i);
    }
    const int32_t num_elements = num_batches * samples_per_batch;

    // Allocate the output before fudging num_batches and samples_per_batch.
    Tensor* samples_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, tensor_shape, &samples_tensor));

    // Parameters must be 0-d or 1-d.
    OP_REQUIRES(ctx, means_tensor.dims() <= 1,
                errors::InvalidArgument(
                    "Input means should be a scalar or vector, got shape: ",
                    means_tensor.shape().DebugString()));
    OP_REQUIRES(ctx, stddevs_tensor.dims() <= 1,
                errors::InvalidArgument(
                    "Input stddevs should be a scalar or vector, got shape: ",
                    stddevs_tensor.shape().DebugString()));
    OP_REQUIRES(ctx, minvals_tensor.dims() <= 1,
                errors::InvalidArgument(
                    "Input minvals should be a scalar or vector, got shape: ",
                    minvals_tensor.shape().DebugString()));
    OP_REQUIRES(ctx, maxvals_tensor.dims() <= 1,
                errors::InvalidArgument(
                    "Input maxvals should be a scalar or vector, got shape: ",
                    maxvals_tensor.shape().DebugString()));

    if ((means_tensor.dims() == 0 || means_tensor.dim_size(0) == 1) &&
        (stddevs_tensor.dims() == 0 || stddevs_tensor.dim_size(0) == 1) &&
        minvals_tensor.dims() == 0 && maxvals_tensor.dims() == 0) {
      // All batches have the same parameters, so we can update the batch size
      // to a reasonable value to improve parallelism (ensure enough batches,
      // and no very small batches which have high overhead).
      int32_t size = num_batches * samples_per_batch;
      int32_t adjusted_samples = kDesiredBatchSize;
      // Ensure adjusted_batches * adjusted_samples >= size.
      int32_t adjusted_batches = Eigen::divup(size, adjusted_samples);
      num_batches = adjusted_batches;
      samples_per_batch = adjusted_samples;
    } else {
      // Parameters must be broadcastable to the shape [num_batches].
      OP_REQUIRES(
          ctx,
          TensorShapeUtils::IsScalar(means_tensor.shape()) ||
              means_tensor.dim_size(0) == 1 ||
              means_tensor.dim_size(0) == num_batches,
          errors::InvalidArgument(
              "Input means should have length 1 or shape[0], got shape: ",
              means_tensor.shape().DebugString()));
      OP_REQUIRES(
          ctx,
          TensorShapeUtils::IsScalar(stddevs_tensor.shape()) ||
              stddevs_tensor.dim_size(0) == 1 ||
              stddevs_tensor.dim_size(0) == num_batches,
          errors::InvalidArgument(
              "Input stddevs should have length 1 or shape[0], got shape: ",
              stddevs_tensor.shape().DebugString()));
      OP_REQUIRES(
          ctx,
          TensorShapeUtils::IsScalar(minvals_tensor.shape()) ||
              minvals_tensor.dim_size(0) == 1 ||
              minvals_tensor.dim_size(0) == num_batches,
          errors::InvalidArgument(
              "Input minvals should have length 1 or shape[0], got shape: ",
              minvals_tensor.shape().DebugString()));
      OP_REQUIRES(
          ctx,
          TensorShapeUtils::IsScalar(maxvals_tensor.shape()) ||
              maxvals_tensor.dim_size(0) == 1 ||
              maxvals_tensor.dim_size(0) == num_batches,
          errors::InvalidArgument(
              "Input maxvals should have length 1 or shape[0], got shape: ",
              maxvals_tensor.shape().DebugString()));
    }

    auto truncFunctor = functor::TruncatedNormalFunctor<Device, T>();
    // Each worker has the fudge factor for samples_per_batch, so use it here.
    random::PhiloxRandom rng =
        generator_.ReserveSamples128(num_batches * 2 * functor::kMaxIterations *
                                     (samples_per_batch + 3) / 4);
    truncFunctor(ctx, ctx->eigen_device<Device>(), num_batches,
                 samples_per_batch, num_elements, means_tensor.flat<T>(),
                 stddevs_tensor.flat<T>(), minvals_tensor.flat<T>(),
                 maxvals_tensor.flat<T>(), rng, samples_tensor->flat<T>());
  }

 private:
  GuardedPhiloxRandom generator_;

  TF_DISALLOW_COPY_AND_ASSIGN(ParameterizedTruncatedNormalOp);
};

// Samples from a truncated normal distribution, using the given parameters.
template <typename Device, typename T>
class StatelessParameterizedTruncatedNormal : public OpKernel {
  // Reshape batches so each batch is this size if possible.
  static const int32_t kDesiredBatchSize = 100;

 public:
  explicit StatelessParameterizedTruncatedNormal(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_tensor = ctx->input(0);
    const Tensor& seed_tensor = ctx->input(1);
    const Tensor& means_tensor = ctx->input(2);
    const Tensor& stddevs_tensor = ctx->input(3);
    const Tensor& minvals_tensor = ctx->input(4);
    const Tensor& maxvals_tensor = ctx->input(5);

    OP_REQUIRES(ctx, seed_tensor.dims() == 1 && seed_tensor.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_tensor.shape().DebugString()));

    tensorflow::BCastList<4> bcast(
        {means_tensor.shape().dim_sizes(), stddevs_tensor.shape().dim_sizes(),
         minvals_tensor.shape().dim_sizes(),
         maxvals_tensor.shape().dim_sizes()},
        /*fewer_dims_optimization=*/false,
        /*return_flattened_batch_indices=*/true);

    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "means, stddevs, minvals, maxvals must have compatible "
                    "batch dimensions: ",
                    means_tensor.shape().DebugString(), " vs. ",
                    stddevs_tensor.shape().DebugString(), " vs. ",
                    minvals_tensor.shape().DebugString(), " vs. ",
                    maxvals_tensor.shape().DebugString()));

    // Let's check that the shape tensor dominates the broadcasted tensor.
    TensorShape bcast_shape = BCast::ToShape(bcast.output_shape());
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(shape_tensor.shape()),
        errors::InvalidArgument("Input shape should be a vector, got shape: ",
                                shape_tensor.shape().DebugString()));
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

    int64_t samples_per_batch = 1;
    const int64_t num_sample_dims =
        (shape_tensor.dim_size(0) - bcast.output_shape().size());
    for (int64_t i = 0; i < num_sample_dims; ++i) {
      samples_per_batch *= output_shape.dim_size(i);
    }
    int64_t num_batches = 1;
    for (int64_t i = num_sample_dims; i < shape_tensor.dim_size(0); ++i) {
      num_batches *= output_shape.dim_size(i);
    }
    const int64_t num_elements = num_batches * samples_per_batch;

    Tensor* samples_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &samples_tensor));

    auto truncFunctor = functor::TruncatedNormalFunctorV2<Device, T>();
    // Each worker has the same fudge factor, so use it here.
    random::PhiloxRandom::Key key;
    random::PhiloxRandom::ResultType counter;
    OP_REQUIRES_OK(ctx, GenerateKey(seed_tensor, &key, &counter));

    auto philox = random::PhiloxRandom(counter, key);

    truncFunctor(ctx, ctx->eigen_device<Device>(), num_batches,
                 samples_per_batch, num_elements, bcast, means_tensor.flat<T>(),
                 stddevs_tensor.flat<T>(), minvals_tensor.flat<T>(),
                 maxvals_tensor.flat<T>(), philox, samples_tensor->flat<T>());
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(StatelessParameterizedTruncatedNormal);
};

}  // namespace

#define REGISTER(TYPE)                                                     \
  REGISTER_KERNEL_BUILDER(Name("ParameterizedTruncatedNormal")             \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<TYPE>("dtype"),              \
                          ParameterizedTruncatedNormalOp<CPUDevice, TYPE>) \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("StatelessParameterizedTruncatedNormal")                        \
          .HostMemory("shape")                                             \
          .HostMemory("seed")                                              \
          .HostMemory("means")                                             \
          .HostMemory("stddevs")                                           \
          .HostMemory("minvals")                                           \
          .HostMemory("maxvals")                                           \
          .Device(DEVICE_CPU)                                              \
          .TypeConstraint<TYPE>("dtype"),                                  \
      StatelessParameterizedTruncatedNormal<CPUDevice, TYPE>)

TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER(TYPE)                                         \
  REGISTER_KERNEL_BUILDER(Name("ParameterizedTruncatedNormal") \
                              .Device(DEVICE_GPU)              \
                              .HostMemory("shape")             \
                              .TypeConstraint<TYPE>("dtype"),  \
                          ParameterizedTruncatedNormalOp<GPUDevice, TYPE>)

TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // end namespace tensorflow
