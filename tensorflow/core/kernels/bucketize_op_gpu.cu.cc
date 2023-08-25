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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/bucketize_op.h"
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T, bool useSharedMem>
__global__ void BucketizeCustomKernel(
    const int32 size_in, const T* __restrict__ in, const int32 size_boundaries,
    GpuDeviceArrayStruct<float> boundaries_array, int32* __restrict__ out) {
  const float* boundaries = GetGpuDeviceArrayOnDevice(&boundaries_array);

  GPU_DYNAMIC_SHARED_MEM_DECL(sizeof(float), unsigned char, shared_mem);
  float* shared_mem_boundaries = reinterpret_cast<float*>(shared_mem);

  if (useSharedMem) {
    int32 lidx = threadIdx.y * blockDim.x + threadIdx.x;
    int32 blockSize = blockDim.x * blockDim.y;

    for (int32 i = lidx; i < size_boundaries; i += blockSize) {
      shared_mem_boundaries[i] = boundaries[i];
    }

    __syncthreads();

    boundaries = shared_mem_boundaries;
  }

  GPU_1D_KERNEL_LOOP(i, size_in) {
    T value = in[i];
    int32 bucket = 0;
    int32 count = size_boundaries;
    while (count > 0) {
      int32 l = bucket;
      int32 step = count / 2;
      l += step;
      if (!(value < static_cast<T>(boundaries[l]))) {
        bucket = ++l;
        count -= step + 1;
      } else {
        count = step;
      }
    }
    out[i] = bucket;
  }
}

namespace functor {

template <typename T>
struct BucketizeFunctor<GPUDevice, T> {
  // PRECONDITION: boundaries_vector must be sorted.
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<T, 1>::ConstTensor& input,
                        const std::vector<float>& boundaries_vector,
                        typename TTypes<int32, 1>::Tensor& output) {
    const GPUDevice& d = context->eigen_device<GPUDevice>();

    GpuDeviceArrayOnHost<float> boundaries_array(context,
                                                 boundaries_vector.size());
    TF_RETURN_IF_ERROR(boundaries_array.Init());
    for (int i = 0; i < boundaries_vector.size(); ++i) {
      boundaries_array.Set(i, boundaries_vector[i]);
    }
    TF_RETURN_IF_ERROR(boundaries_array.Finalize());

    GpuLaunchConfig config = GetGpuLaunchConfig(input.size(), d);
    int32 shared_mem_size = sizeof(float) * boundaries_vector.size();
    const int32 kMaxSharedMemBytes = 16384;
    if (shared_mem_size < d.sharedMemPerBlock() &&
        shared_mem_size < kMaxSharedMemBytes) {
      TF_CHECK_OK(GpuLaunchKernel(BucketizeCustomKernel<T, true>,
                                  config.block_count, config.thread_per_block,
                                  shared_mem_size, d.stream(), input.size(),
                                  input.data(), boundaries_vector.size(),
                                  boundaries_array.data(), output.data()));
    } else {
      TF_CHECK_OK(GpuLaunchKernel(
          BucketizeCustomKernel<T, false>, config.block_count,
          config.thread_per_block, 0, d.stream(), input.size(), input.data(),
          boundaries_vector.size(), boundaries_array.data(), output.data()));
    }
    return OkStatus();
  }
};
}  // namespace functor

#define REGISTER_GPU_SPEC(type) \
  template struct functor::BucketizeFunctor<GPUDevice, type>;

REGISTER_GPU_SPEC(int32);
REGISTER_GPU_SPEC(int64);
REGISTER_GPU_SPEC(float);
REGISTER_GPU_SPEC(double);
#undef REGISTER_GPU_SPEC

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
