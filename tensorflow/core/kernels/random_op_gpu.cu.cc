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

template <typename T, int ElementCount>
class SampleCopier {
 public:
  inline __device__ void operator()(
      T* buf, const tensorflow::random::Array<T, ElementCount>& array) const {
#pragma unroll
    for (int i = 0; i < ElementCount; i++) {
      buf[i] = array[i];
    }
  }
};

template <>
class SampleCopier<float, 4> {
 public:
  // Copies the elements from the array to buf. buf must be 128-bit aligned,
  // which is true for tensor data, and all offsets that are a multiple of the
  // vector size (because the vectors are 128 bits long).
  inline __device__ void operator()(
      float* buf, const tensorflow::random::Array<float, 4>& array) const {
    // NOTE(ringwalt): It's not safe to cast &array[0] to a float4, because they
    // have 32-bit alignment vs 128-bit alignment. There seems to be no
    // performance loss when assigning each element to a vector.
    float4 vec;
    vec.x = array[0];
    vec.y = array[1];
    vec.z = array[2];
    vec.w = array[3];
    float4* buf_vector = reinterpret_cast<float4*>(buf);
    *buf_vector = vec;
  }
};

template <>
class SampleCopier<int32, 4> {
 public:
  // Copies the elements from the array to buf. buf must be 128-bit aligned,
  // which is true for tensor data, and all offsets that are a multiple of the
  // vector size (because the vectors are 128 bits long).
  inline __device__ void operator()(
      int32* buf, const tensorflow::random::Array<int32, 4>& array) const {
    int4 vec;
    vec.x = array[0];
    vec.y = array[1];
    vec.z = array[2];
    vec.w = array[3];
    int4* buf_vector = reinterpret_cast<int4*>(buf);
    *buf_vector = vec;
  }
};

template <>
class SampleCopier<double, 2> {
 public:
  // Copies the elements from the array to buf. buf must be 128-bit aligned,
  // which is true for tensor data, and all offsets that are a multiple of the
  // vector size (because the vectors are 128 bits long).
  inline __device__ void operator()(
      double* buf, const tensorflow::random::Array<double, 2>& array) const {
    double2 vec;
    vec.x = array[0];
    vec.y = array[1];
    double2* buf_vector = reinterpret_cast<double2*>(buf);
    *buf_vector = vec;
  }
};

template <>
class SampleCopier<int64, 2> {
 public:
  // Copies the elements from the array to buf. buf must be 128-bit aligned,
  // which is true for tensor data, and all offsets that are a multiple of the
  // vector size (because the vectors are 128 bits long).
  inline __device__ void operator()(
      int64* buf, const tensorflow::random::Array<int64, 2>& array) const {
    longlong2 vec;
    vec.x = array[0];
    vec.y = array[1];
    longlong2* buf_vector = reinterpret_cast<longlong2*>(buf);
    *buf_vector = vec;
  }
};

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

    const SampleCopier<T, kGroupSize> copier;
    while (offset + kGroupSize <= size) {
      const typename Distribution::ResultType samples = dist(&gen);
      copier(&data[offset], samples);

      offset += total_thread_count * kGroupSize;
      gen.Skip(total_thread_count - 1);
    }

    typename Distribution::ResultType samples = dist(&gen);
    for (int i = 0; i < kGroupSize; ++i) {
      if (offset >= size) {
        return;
      }
      data[offset] = samples[i];
      ++offset;
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
// clang-format on

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
