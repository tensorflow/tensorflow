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

namespace functor {

template <typename T>
struct BincountFunctor<GPUDevice, T> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<int32, 1>::ConstTensor& arr,
                        const typename TTypes<T, 1>::ConstTensor& weights,
                        typename TTypes<T, 1>::Tensor& output) {
    const GPUDevice& d = context->eigen_device<GPUDevice>();

    // Set 'output' to zeros.
    CudaLaunchConfig config = GetCudaLaunchConfig(output.size(), d);
    SetZero<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        output.size(), output.data());

    return Status::OK();
  }
};

}  // end namespace functor

#define REGISTER_GPU_SPEC(type) \
  template struct functor::BincountFunctor<GPUDevice, type>;

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_SPEC);
#undef REGISTER_GPU_SPEC

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
