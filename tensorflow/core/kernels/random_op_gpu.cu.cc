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
#include "tensorflow/core/kernels/random_op_gpu.h"

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
void FillPhiloxRandom<GPUDevice, Distribution>::operator()(
    OpKernelContext*, const GPUDevice& d, random::PhiloxRandom gen,
    typename Distribution::ResultElementType* data, int64 size,
    Distribution dist) {
  const int32 block_size = d.maxGpuThreadsPerBlock();
  const int32 num_blocks =
      (d.getNumGpuMultiProcessors() * d.maxGpuThreadsPerMultiProcessor()) /
      block_size;

  FillPhiloxRandomKernelLaunch<Distribution>
      <<<num_blocks, block_size, 0, d.stream()>>>(gen, data, size, dist);
}

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
// clang-format on

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
