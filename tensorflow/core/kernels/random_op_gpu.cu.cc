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

#include "tensorflow/core/kernels/random_op.h"

#include <assert.h>
#include <stdio.h>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

class OpKernelContext;

namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template <class Distribution, bool VariableSamplesPerOutput>
struct FillPhiloxRandomKernel;

// A cuda kernel to fill the data with random numbers from the specified
// distribution. Each output takes a fixed number of samples.
template <class Distribution>
struct FillPhiloxRandomKernel<Distribution, false> {
  typedef typename Distribution::ResultElementType T;
  PHILOX_DEVICE_FUNC void Run(random::PhiloxRandom gen, T* data, int64 size,
                              Distribution dist) {
    const int kGroupSize = Distribution::kResultElementCount;

    const int32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int32 total_thread_count = gridDim.x * blockDim.x;
    int32 offset = thread_id * kGroupSize;
    gen.Skip(thread_id);

    while (offset < size) {
      typename Distribution::ResultType samples = dist(&gen);

      for (int i = 0; i < kGroupSize; ++i) {
        if (offset >= size) {
          return;
        }
        data[offset] = samples[i];
        ++offset;
      }

      offset += (total_thread_count - 1) * kGroupSize;
      gen.Skip(total_thread_count - 1);
    }
  }
};

// A cuda kernel to fill the data with random numbers from the specified
// distribution. Each output takes a variable number of samples.
template <class Distribution>
struct FillPhiloxRandomKernel<Distribution, true> {
  typedef typename Distribution::ResultElementType T;
  PHILOX_DEVICE_FUNC void Run(const random::PhiloxRandom& base_gen, T* data,
                              int64 size, Distribution dist) {
    using random::PhiloxRandom;
    using random::SingleSampleAdapter;

    const int kReservedSamplesPerOutput = 256;
    const int kGroupSize = Distribution::kResultElementCount;
    const int kGeneratorSkipPerOutputGroup = kGroupSize *
                                             kReservedSamplesPerOutput /
                                             PhiloxRandom::kResultElementCount;

    const int32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int32 total_thread_count = gridDim.x * blockDim.x;
    int64 group_index = thread_id;
    int64 offset = group_index * kGroupSize;

    while (offset < size) {
      // Since each output takes a variable number of samples, we need to
      // realign the generator to the beginning for the current output group
      PhiloxRandom gen = base_gen;
      gen.Skip(group_index * kGeneratorSkipPerOutputGroup);
      SingleSampleAdapter<PhiloxRandom> single_samples(&gen);

      typename Distribution::ResultType samples = dist(&single_samples);

      for (int i = 0; i < kGroupSize; ++i) {
        if (offset >= size) {
          return;
        }
        data[offset] = samples[i];
        ++offset;
      }

      offset += (total_thread_count - 1) * kGroupSize;
      group_index += total_thread_count;
    }
  }
};

// A simple launch pad to call the correct function templates to fill the data
template <class Distribution>
__global__ void __launch_bounds__(1024)
    FillPhiloxRandomKernelLaunch(random::PhiloxRandom base_gen,
                                 typename Distribution::ResultElementType* data,
                                 int64 size, Distribution dist) {
  FillPhiloxRandomKernel<Distribution,
                         Distribution::kVariableSamplesPerOutput>()
      .Run(base_gen, data, size, dist);
}

// Partial specialization for GPU
template <class Distribution>
struct FillPhiloxRandom<GPUDevice, Distribution> {
  typedef typename Distribution::ResultElementType T;
  typedef GPUDevice Device;
  void operator()(OpKernelContext*, const Device& d, random::PhiloxRandom gen,
                  T* data, int64 size, Distribution dist) {
    const int32 block_size = d.maxCudaThreadsPerBlock();
    const int32 num_blocks =
        (d.getNumCudaMultiProcessors() * d.maxCudaThreadsPerMultiProcessor()) /
        block_size;

    FillPhiloxRandomKernelLaunch<
        Distribution><<<num_blocks, block_size, 0, d.stream()>>>(gen, data,
                                                                 size, dist);
  }
};

// Kernel for Multinomial op.  Data is interpreted to have the following shapes:
//   scores: [B, S, C];  maxima: [B, S];  output: [B, S].
__global__ void MultinomialKernel(int32 nthreads, const int32 num_classes,
                                  const int32 num_samples, const float* scores,
                                  const float* maxima, int64* output) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int maxima_idx = index / num_classes;
    if (ldg(maxima + maxima_idx) == ldg(scores + index)) {
      CudaAtomicMax(reinterpret_cast<uint64*>(output + maxima_idx),
                    static_cast<uint64>(index % num_classes));
    }
  }
}

template <typename T>
struct MultinomialFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* ctx, const GPUDevice& d,
                  typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<float>::Flat noises,
                  typename TTypes<float>::Flat scores,
                  typename TTypes<float>::Flat maxima, int batch_size,
                  int num_classes, int num_samples,
                  const random::PhiloxRandom& gen,
                  typename TTypes<int64>::Matrix output) {
    // Uniform, [0, 1).
    typedef random::UniformDistribution<random::PhiloxRandom, float> Dist;
    functor::FillPhiloxRandom<GPUDevice, Dist>()(ctx, d, gen, noises.data(),
                                                 noises.size(), Dist());

#if defined(EIGEN_HAS_INDEX_LIST)
    Eigen::IndexList<Eigen::type2index<2>> kTwo;
    Eigen::IndexList<int, int, int> bsc;
    bsc.set(0, batch_size);
    bsc.set(1, num_samples);
    bsc.set(2, num_classes);

    Eigen::IndexList<int, Eigen::type2index<1>, int> boc;
    boc.set(0, batch_size);
    boc.set(2, num_classes);

    Eigen::IndexList<Eigen::type2index<1>, int, Eigen::type2index<1>> oso;
    oso.set(1, num_samples);
#else
    Eigen::array<int, 1> kTwo{2};
    Eigen::array<int, 3> bsc{batch_size, num_samples, num_classes};
    Eigen::array<int, 3> boc{batch_size, 1, num_classes};
    Eigen::array<int, 3> oso{1, num_samples, 1};
#endif

    // Calculates "scores = logits - log(-log(noises))"; B*C*S elements.
    // NOTE: we don't store back to "noises" because having it appear on both
    // sides is potentially unsafe (e.g. Eigen may use ldg() to load RHS data).
    To32Bit(scores).device(d) =
        To32Bit(logits).reshape(boc).broadcast(oso).template cast<float>() -
        ((-(To32Bit(noises).log())).log());

    // Max-reduce along classes for each (batch, sample).
    To32Bit(maxima).device(d) = To32Bit(scores).reshape(bsc).maximum(kTwo);

    // Necessary for atomicMax() inside the kernel.
    output.device(d) = output.constant(0LL);

    const int32 work_items = batch_size * num_samples * num_classes;
    CudaLaunchConfig config = GetCudaLaunchConfig(work_items, d);
    MultinomialKernel<<<config.block_count, config.thread_per_block, 0,
                        d.stream()>>>(config.virtual_thread_count, num_classes,
                                      num_samples, scores.data(), maxima.data(),
                                      output.data());
  }
};

// Explicit instantiation of the GPU distributions functors
// clang-format off
// NVCC cannot handle ">>" properly
template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, Eigen::half> >;
template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, float> >;
template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, double> >;
template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, int32> >;
template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, int64> >;
template struct FillPhiloxRandom<
    GPUDevice, random::NormalDistribution<random::PhiloxRandom, Eigen::half> >;
template struct FillPhiloxRandom<
    GPUDevice, random::NormalDistribution<random::PhiloxRandom, float> >;
template struct FillPhiloxRandom<
    GPUDevice, random::NormalDistribution<random::PhiloxRandom, double> >;
template struct FillPhiloxRandom<
    GPUDevice, random::TruncatedNormalDistribution<
        random::SingleSampleAdapter<random::PhiloxRandom>, Eigen::half> >;
template struct FillPhiloxRandom<
    GPUDevice, random::TruncatedNormalDistribution<
                   random::SingleSampleAdapter<random::PhiloxRandom>, float> >;
template struct FillPhiloxRandom<
    GPUDevice, random::TruncatedNormalDistribution<
                   random::SingleSampleAdapter<random::PhiloxRandom>, double> >;

template struct MultinomialFunctor<GPUDevice, Eigen::half>;
template struct MultinomialFunctor<GPUDevice, float>;
template struct MultinomialFunctor<GPUDevice, double>;
template struct MultinomialFunctor<GPUDevice, int32>;
template struct MultinomialFunctor<GPUDevice, int64>;
// clang-format on

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
