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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/multinomial_op.h"

#include <assert.h>
#include <stdio.h>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/random_op.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

namespace functor {

using GPUDevice = Eigen::GpuDevice;

// Kernel for Multinomial op.  Data is interpreted to have the following shapes:
//   scores: [B, S, C];  maxima: [B, S];  output: [B, S].
template <typename OutputType>
__global__ void MultinomialKernel(int32 nthreads, const int32 num_classes,
                                  const int32 num_samples, const float* scores,
                                  const float* maxima, OutputType* output) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int maxima_idx = index / num_classes;
    if (ldg(maxima + maxima_idx) == ldg(scores + index)) {
      using UnsignedOutputType = typename std::make_unsigned<OutputType>::type;
      CudaAtomicMax(reinterpret_cast<UnsignedOutputType*>(output + maxima_idx),
                    static_cast<UnsignedOutputType>(index % num_classes));
    }
  }
}

template <typename T, typename OutputType>
struct MultinomialFunctor<GPUDevice, T, OutputType> {
  void operator()(OpKernelContext* ctx, const GPUDevice& d,
                  typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<float>::Flat noises,
                  typename TTypes<float>::Flat scores,
                  typename TTypes<float>::Flat maxima, int batch_size,
                  int num_classes, int num_samples,
                  const random::PhiloxRandom& gen,
                  typename TTypes<OutputType>::Matrix output) {
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
    // 2e-30 is chosen so as to be small enough to only change 0 -> 2e-30 while
    // not affect any of the other numbers (smallest is ~1e-7), but not so small
    // that log(x) == -inf, which is why it needs to be larger than 0 in the
    // first place.
    To32Bit(scores).device(d) =
        To32Bit(logits).reshape(boc).broadcast(oso).template cast<float>() -
        ((-((To32Bit(noises) + 2e-30f).log())).log());

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

// Explicit instantiation of the GPU functors.
template struct MultinomialFunctor<GPUDevice, Eigen::half, int32>;
template struct MultinomialFunctor<GPUDevice, float, int32>;
template struct MultinomialFunctor<GPUDevice, double, int32>;
template struct MultinomialFunctor<GPUDevice, int32, int32>;
template struct MultinomialFunctor<GPUDevice, int64, int32>;

template struct MultinomialFunctor<GPUDevice, Eigen::half, int64>;
template struct MultinomialFunctor<GPUDevice, float, int64>;
template struct MultinomialFunctor<GPUDevice, double, int64>;
template struct MultinomialFunctor<GPUDevice, int32, int64>;
template struct MultinomialFunctor<GPUDevice, int64, int64>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
