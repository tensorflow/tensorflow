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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/parameterized_truncated_normal_op.h"

#include <assert.h>
#include <stdio.h>
#include <cmath>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

#if defined(_MSC_VER) && !defined(__clang__)
// msvc does not support unroll. One could try the loop pragma but we need to
// take a closer look if this generates better code in this case. For now let
// the compiler take care of it.
#define UNROLL
#else
#define UNROLL _Pragma("unroll")
#endif

namespace tensorflow {

class OpKernelContext;

namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
__global__ void __launch_bounds__(1024)
    TruncatedNormalKernel(random::PhiloxRandom gen, T* data, int64 num_batches,
                          int64 samples_per_batch, int64 num_elements,
                          const T* means, bool single_mean, const T* stddevs,
                          bool single_stddev, const T* minvals,
                          bool single_minval, const T* maxvals,
                          bool single_maxval, int64 kMaxIterations) {
  const int32 max_samples_per_item = 2 * kMaxIterations;
  // Initial offset as given by CUDA_1D_KERNEL_LOOP.
  const int32 initial_offset = blockIdx.x * blockDim.x + threadIdx.x;
  gen.Skip(max_samples_per_item * initial_offset);
  typedef random::UniformDistribution<random::PhiloxRandom, T> Uniform;
  Uniform dist;
  const int kDistSize = Uniform::kResultElementCount;
  const T quietNaN = Eigen::NumTraits<T>::quiet_NaN();

  // We skip the total number of threads to get to the next element. To produce
  // deterministic results between devices, each element in the output array
  // skips max_samples_per_item in the generator. Then after generating this
  // item, we need to skip the samples for one element for every thread to get
  // to the next element that we actually process.
  const int32 samples_between_processed_elements =
      max_samples_per_item * (gridDim.x * blockDim.x);

  CUDA_1D_KERNEL_LOOP(offset, num_elements) {
    // Track how many more samples we need to skip before we process the next
    // element.
    int32 remaining_samples = samples_between_processed_elements;

    const int64 batch_id = offset / samples_per_batch;
    T mean = means[single_mean ? 0 : batch_id];
    const T input_stddev = stddevs[single_stddev ? 0 : batch_id];
    T minval = minvals[single_minval ? 0 : batch_id];
    T maxval = maxvals[single_maxval ? 0 : batch_id];

    // Flip the distribution if we can make the lower bound positive.
    T stddev;
    if (Eigen::numext::isinf(minval) || maxval < mean) {
      // Reverse all calculations. normMin and normMax will be flipped.
      // std::swap is a host function (not available in CUDA).
      T temp = minval;
      minval = maxval;
      maxval = temp;
      stddev = -input_stddev;
    } else {
      stddev = input_stddev;
    }

    // Calculate normalized samples, then scale them.
    const T normMin = (minval - mean) / stddev;
    const T normMax = (maxval - mean) / stddev;

    // Determine the method to use.
    const T sqrtFactor = Eigen::numext::sqrt((normMin * normMin) + T(4));
    const T cutoff =
        T(2) *
        Eigen::numext::exp(T(0.5) + (normMin * (normMin - sqrtFactor)) / T(4)) /
        (normMin + sqrtFactor);
    const T diff = normMax - normMin;
    const T two = T(2.0);

    // Validate the normalized min and max, because the originals may have been
    // flipped already.
    if (!(input_stddev > T(0) && normMin < normMax &&
          (Eigen::numext::isfinite(normMin) ||
           Eigen::numext::isfinite(normMax)))) {
      data[offset] = quietNaN;
    } else if (diff < cutoff) {
      // Sample from a uniform distribution on [normMin, normMax].

      // Vectorized intermediate calculations for uniform rejection sampling.
      // We always generate at most 4 samples.
      Eigen::array<T, 4> z;
      Eigen::array<T, 4> g;

      const T plusFactor = (normMin < T(0)) ? T(0) : normMin * normMin;

      int numIterations = 0;
      while (numIterations < kMaxIterations) {
        const auto rand = dist(&gen);
        remaining_samples -= gen.kResultElementCount;
        UNROLL for (int i = 0; i < kDistSize; i++) {
          z[i] = rand[i] * diff + normMin;
        }
        UNROLL for (int i = 0; i < kDistSize; i++) {
          g[i] = (plusFactor - z[i] * z[i]) / two;
        }

        const auto u = dist(&gen);
        remaining_samples -= gen.kResultElementCount;
        UNROLL for (int i = 0; i < kDistSize; i++) {
          if (u[i] <= Eigen::numext::exp(g[i]) ||
              numIterations + 1 >= kMaxIterations) {
            // Accept the sample z.
            // If we run out of iterations, just use the current uniform
            // sample. Emperically, the probability of accepting each sample
            // is at least 50% for typical inputs, so we will always accept
            // by 100 iterations.
            // This introduces a slight inaccuracy when at least one bound
            // is large, minval is negative and maxval is positive.
            data[offset] = z[i] * stddev + mean;
            // Break out of the nested loop by updating numIterations.
            numIterations = kMaxIterations;
            break;
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
          (normMin + Eigen::numext::sqrt((normMin * normMin) + T(4))) / T(2);
      int numIterations = 0;
      while (numIterations < kMaxIterations) {
        auto rand = dist(&gen);
        remaining_samples -= gen.kResultElementCount;
        UNROLL for (int i = 0; i < kDistSize; i += 2) {
          const T z = -Eigen::numext::log(rand[i]) / alpha + normMin;
          const T x = normMin < alpha ? alpha - z : normMin - alpha;
          const T g = Eigen::numext::exp(-x * x / two);
          const T u = rand[i + 1];
          if ((u <= g && z < normMax) || numIterations + 1 >= kMaxIterations) {
            data[offset] = z * stddev + mean;
            // Break out of the nested loop by updating numIterations.
            numIterations = kMaxIterations;
            break;
          } else {
            numIterations++;
          }
        }
      }
    }

    gen.Skip(remaining_samples);
  }
}

// Partial specialization for GPU
template <typename T>
struct TruncatedNormalFunctor<GPUDevice, T> {
  static const int kMaxIterations = 1000;

  void operator()(OpKernelContext* ctx, const GPUDevice& d, int64 num_batches,
                  int64 samples_per_batch, int64 num_elements,
                  typename TTypes<T>::ConstFlat means,
                  typename TTypes<T>::ConstFlat stddevs,
                  typename TTypes<T>::ConstFlat minvals,
                  typename TTypes<T>::ConstFlat maxvals,
                  const random::PhiloxRandom& gen,
                  typename TTypes<T>::Flat output) {
    const auto config = GetCudaLaunchConfig(num_elements, d);

    TruncatedNormalKernel<T>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            gen, output.data(), num_batches, samples_per_batch, num_elements,
            means.data(), means.dimension(0) == 1, stddevs.data(),
            stddevs.dimension(0) == 1, minvals.data(),
            minvals.dimension(0) == 1, maxvals.data(),
            maxvals.dimension(0) == 1, kMaxIterations);
  };
};

// Explicit instantiation of the GPU distributions functors
template struct TruncatedNormalFunctor<GPUDevice, Eigen::half>;
template struct TruncatedNormalFunctor<GPUDevice, float>;
template struct TruncatedNormalFunctor<GPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
