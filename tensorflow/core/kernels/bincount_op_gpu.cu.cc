/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/bincount_op.h"
#include "external/cub_archive/cub/device/device_histogram.cuh"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
__global__ void BincountCustomKernel(const int32 size_in, const int32* key,
                                     const T* value, const int32 size_out,
                                     T* out) {
  CUDA_1D_KERNEL_LOOP(i, size_in) {
    const int32 k = key[i];
    if (k < 0 || k >= size_out) {
      continue;
    }
    CudaAtomicAdd(out + k, value[i]);
  }
}

namespace functor {

template <typename T>
struct BincountFunctor<GPUDevice, T> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<int32, 1>::ConstTensor& arr,
                        const typename TTypes<T, 1>::ConstTensor& weights,
                        typename TTypes<T, 1>::Tensor& output) {
    if (output.size() == 0) {
      return Status::OK();
    }
    if (weights.size() == 0) {
      // In case weight.size() == 0, use CUB
      size_t temp_storage_bytes = 0;
      const int32* d_samples = arr.data();
      T* d_histogram = output.data();
      int num_levels = output.size() + 1;
      int32 lower_level = 0;
      int32 upper_level = output.size();
      int num_samples = arr.size();
      const cudaStream_t& stream = GetCudaStream(context);

      // The first HistogramEven is to obtain the temp storage size required
      // with d_temp_storage = NULL passed to the call.
      auto err = cub::DeviceHistogram::HistogramEven(
          /* d_temp_storage */ NULL,
          /* temp_storage_bytes */ temp_storage_bytes,
          /* d_samples */ d_samples,
          /* d_histogram */ d_histogram,
          /* num_levels */ num_levels,
          /* lower_level */ lower_level,
          /* upper_level */ upper_level,
          /* num_samples */ num_samples,
          /* stream */ stream);
      if (err != cudaSuccess) {
        return errors::Internal(
            "Could not launch HistogramEven to get temp storage: ",
            cudaGetErrorString(err), ".");
      }
      Tensor temp_storage;
      TF_RETURN_IF_ERROR(context->allocate_temp(
          DataTypeToEnum<int8>::value,
          TensorShape({static_cast<int64>(temp_storage_bytes)}),
          &temp_storage));

      void* d_temp_storage = temp_storage.flat<int8>().data();
      // The second HistogramEven is to actual run with d_temp_storage
      // allocated with temp_storage_bytes.
      err = cub::DeviceHistogram::HistogramEven(
          /* d_temp_storage */ d_temp_storage,
          /* temp_storage_bytes */ temp_storage_bytes,
          /* d_samples */ d_samples,
          /* d_histogram */ d_histogram,
          /* num_levels */ num_levels,
          /* lower_level */ lower_level,
          /* upper_level */ upper_level,
          /* num_samples */ num_samples,
          /* stream */ stream);
      if (err != cudaSuccess) {
        return errors::Internal("Could not launch HistogramEven: ",
                                cudaGetErrorString(err), ".");
      }
      return Status::OK();
    }

    const GPUDevice& d = context->eigen_device<GPUDevice>();
    // Set 'output' to zeros.
    CudaLaunchConfig config = GetCudaLaunchConfig(output.size(), d);
    SetZero<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        output.size(), output.data());

    if (arr.size() == 0) {
      return Status::OK();
    }
    config = GetCudaLaunchConfig(arr.size(), d);
    BincountCustomKernel<
        T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        arr.size(), arr.data(), weights.data(), output.size(), output.data());
    return Status::OK();
  }
};

}  // end namespace functor

#define REGISTER_GPU_SPEC(type) \
  template struct functor::BincountFunctor<GPUDevice, type>;

TF_CALL_float(REGISTER_GPU_SPEC)
#undef REGISTER_GPU_SPEC

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
