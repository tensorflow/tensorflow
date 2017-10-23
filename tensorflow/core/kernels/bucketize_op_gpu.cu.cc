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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/bucketize_op.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
__global__ void BucketizeCustomKernel(const int32 size_in, const T* in,
                                      const int32 size_boundaries,
                                      const float* boundaries, int32* out) {
  CUDA_1D_KERNEL_LOOP(i, size_in) {
    T value = in[i];
    int32 bucket = 0;
    while (bucket < size_boundaries &&
           value >= static_cast<T>(boundaries[bucket])) {
      bucket++;
    }
    out[i] = bucket;
  }
}

namespace functor {

template <typename T>
struct BucketizeFunctor<GPUDevice, T> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<T, 1>::ConstTensor& input,
                        const std::vector<float>& boundaries_vector,
                        typename TTypes<int32, 1>::Tensor& output) {
    const GPUDevice& d = context->eigen_device<GPUDevice>();

    tensorflow::AllocatorAttributes pinned_allocator;
    pinned_allocator.set_on_host(true);
    pinned_allocator.set_gpu_compatible(true);

    Tensor boundaries_tensor;
    TF_RETURN_IF_ERROR(
        context->allocate_temp(DataTypeToEnum<float>::value,
                               TensorShape({(int64)boundaries_vector.size()}),
                               &boundaries_tensor, pinned_allocator));

    auto boundaries = boundaries_tensor.flat<float>();
    boundaries = typename TTypes<float, 1>::ConstTensor(
        boundaries_vector.data(), boundaries_vector.size());

    CudaLaunchConfig config = GetCudaLaunchConfig(input.size(), d);

    BucketizeCustomKernel<
        T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        input.size(), input.data(), boundaries.size(), boundaries.data(),
        output.data());

    return Status::OK();
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

#endif  // GOOGLE_CUDA
