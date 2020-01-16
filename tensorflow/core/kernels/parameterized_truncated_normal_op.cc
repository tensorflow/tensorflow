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
  void operator()(OpKernelContext* ctx, const CPUDevice& d, int64 num_batches,
                  int64 samples_per_batch, int64 num_elements,
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

    auto DoWork = [samples_per_batch, num_elements, &ctx, &means, &stddevs,
                   &minvals, &maxvals, &gen, &output,
                   kStdDevsInsideBoundsToUseRandnSampler](int start_batch,
                                                          int limit_batch) {
      // Capturing "gen" by-value would only make a copy for the _shared_
      // lambda.  Since we want to let each worker have its own copy, we pass
      // "gen" by reference and explicitly do a copy assignment here.
      random::PhiloxRandom gen_copy = gen;
      // Skip takes units of 128 bytes.  +3 is so rounding doesn't lead to
      // us using the same state in different batches.
      // The sample from each iteration uses 2 random numbers.
      gen_copy.Skip(start_batch * 2 * kMaxIterations * (samples_per_batch + 3) /
                    4);
      typedef random::UniformDistribution<random::PhiloxRandom, T> Uniform;
      Uniform dist;
      typedef random::NormalDistribution<random::PhiloxRandom, T> Normal;
      Normal normal_dist;

      // Vectorized intermediate calculations for uniform rejection sampling.
      // We always generate at most 4 samples.
      Eigen::array<T, 4> z;
      Eigen::array<T, 4> g;

      for (int64 b = start_batch; b < limit_batch; ++b) {
        // We are passed a flat array for each of the parameter tensors.
        // The input is either a scalar broadcasted to all batches or a vector
        // with length num_batches, but the scalar becomes an array of length 1.
        T mean = means((means.dimension(0) == 1) ? 0 : b);
        T stddev = stddevs((stddevs.dimension(0) == 1) ? 0 : b);
        T minval = minvals((minvals.dimension(0) == 1) ? 0 : b);
        T maxval = maxvals((maxvals.dimension(0) == 1) ? 0 : b);

        // The last batch can be short, if we adjusted num_batches and
        // samples_per_batch.
        const int64 limit_sample =
            std::min((b + 1) * samples_per_batch, num_elements);
        int64 sample = b * samples_per_batch;

        // On GPU, this check will just fill samples with NAN if it fails.
        OP_REQUIRES(ctx,
                    stddev > T(0) && minval < maxval &&
                        (Eigen::numext::isfinite(minval) ||
                         Eigen::numext::isfinite(maxval)),
                    errors::InvalidArgument("Invalid parameters"));

        int numIterations = 0;

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
                numIterations = 0;
              } else {
                numIterations++;
                if (numIterations > kMaxIterations) {
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
              if (accept || numIterations + 1 >= kMaxIterations) {
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
                numIterations = 0;
              } else {
                numIterations++;
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
              if (accept || numIterations + 1 >= kMaxIterations) {
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
                numIterations = 0;
              } else {
                numIterations++;
              }
            }
          }
        }
      }
    };
    // The cost of the initial calculations for the batch.
    const int64 batchInitCost =
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
    const int64 uniformSampleCost =
        random::PhiloxRandom::kElementCost +
        random::UniformDistribution<random::PhiloxRandom, T>::kElementCost;
    // The cost of a single uniform sampling round.
    const int64 uniformRejectionSamplingCost =
        uniformSampleCost + Eigen::TensorOpCost::MulCost<T>() +
        Eigen::TensorOpCost::AddCost<T>() +
        Eigen::TensorOpCost::MulCost<T>() * 2 +
        Eigen::TensorOpCost::AddCost<T>() + uniformSampleCost +
        Eigen::internal::functor_traits<
            Eigen::internal::scalar_exp_op<T>>::Cost +
        Eigen::TensorOpCost::MulCost<T>() + Eigen::TensorOpCost::AddCost<T>();
    // Estimate the cost for an entire batch.
    // Assume we use uniform sampling, and accept the 2nd sample on average.
    const int64 batchCost =
        batchInitCost + uniformRejectionSamplingCost * 2 * samples_per_batch;
    Shard(worker_threads.num_threads, worker_threads.workers, num_batches,
          batchCost, DoWork);
  }
};

}  // namespace functor

namespace {

// Samples from a truncated normal distribution, using the given parameters.
template <typename Device, typename T>
class ParameterizedTruncatedNormalOp : public OpKernel {
  // Reshape batches so each batch is this size if possible.
  static const int32 kDesiredBatchSize = 100;

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

}  // namespace

#define REGISTER(TYPE)                                         \
  REGISTER_KERNEL_BUILDER(Name("ParameterizedTruncatedNormal") \
                              .Device(DEVICE_CPU)              \
                              .TypeConstraint<TYPE>("dtype"),  \
                          ParameterizedTruncatedNormalOp<CPUDevice, TYPE>)

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
