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

#ifndef TENSORFLOW_CORE_KERNELS_SCATTER_FUNCTOR_GPU_CU_H_
#define TENSORFLOW_CORE_KERNELS_SCATTER_FUNCTOR_GPU_CU_H_

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/scatter_functor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace scatter_op_gpu {

template <typename T, scatter_op::UpdateOp op>
struct ScatterOpKernelBody;

template <typename T>
struct ScatterOpKernelBody<T, scatter_op::UpdateOp::ASSIGN> {
  __device__ void operator()(T* dest, T src) const { *dest = src; }
};

template <typename T>
struct ScatterOpKernelBody<T, scatter_op::UpdateOp::ADD> {
  __device__ void operator()(T* dest, T src) const { CudaAtomicAdd(dest, src); }
};

template <typename T>
struct ScatterOpKernelBody<T, scatter_op::UpdateOp::SUB> {
  __device__ void operator()(T* dest, T src) const { CudaAtomicSub(dest, src); }
};

template <typename T>
struct ScatterOpKernelBody<T, scatter_op::UpdateOp::MUL> {
  __device__ void operator()(T* dest, T src) const { CudaAtomicMul(dest, src); }
};

template <typename T>
struct ScatterOpKernelBody<T, scatter_op::UpdateOp::DIV> {
  __device__ void operator()(T* dest, T src) const { CudaAtomicDiv(dest, src); }
};

template <typename T>
struct ScatterOpKernelBody<T, scatter_op::UpdateOp::MIN> {
  __device__ void operator()(T* dest, T src) const { CudaAtomicMin(dest, src); }
};

template <typename T>
struct ScatterOpKernelBody<T, scatter_op::UpdateOp::MAX> {
  __device__ void operator()(T* dest, T src) const { CudaAtomicMax(dest, src); }
};

template <typename T, typename Index, scatter_op::UpdateOp op>
__global__ void ScatterOpCustomKernel(T* params, const T* updates,
                                      const Index* indices,
                                      Index first_dim_size, Index updates_size,
                                      Index indices_size) {
  Index update_block = updates_size / indices_size;
  ScatterOpKernelBody<T, op> body;
  CUDA_1D_KERNEL_LOOP(i, updates_size) {
    int indices_i = i / update_block;
    int updates_i = i;
    int param_first_index = indices[indices_i];
    if (!(param_first_index >= 0 && param_first_index < first_dim_size)) {
      // Ignore indices that are out of range.
      continue;
    }
    int params_i = param_first_index * update_block + (i % update_block);
    body(&params[params_i], ldg(updates + updates_i));
  }
}

template <typename T, typename Index, scatter_op::UpdateOp op>
__global__ void ScatterScalarOpCustomKernel(T* params, const T* update,
                                            const Index* indices,
                                            Index first_dim_size,
                                            Index indices_size,
                                            Index synthesized_updates_size) {
  Index update_block = synthesized_updates_size / indices_size;
  ScatterOpKernelBody<T, op> body;
  CUDA_1D_KERNEL_LOOP(i, synthesized_updates_size) {
    int indices_i = i / update_block;
    int param_first_index = indices[indices_i];
    const T update_val = *update;
    if (!(param_first_index >= 0 && param_first_index < first_dim_size)) {
      // Ignore indices that are out of range.
      continue;
    }
    int params_i = param_first_index * update_block + (i % update_block);
    body(&params[params_i], update_val);
  }
}

}  // namespace scatter_op_gpu

namespace functor {
// Specialization for a GPU device.
template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterFunctor<GPUDevice, T, Index, op> {
  Index operator()(OpKernelContext* c, const GPUDevice& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices) {
    // TODO(b/31801742): Implement indices range check. The hardest part is
    // with returning a value after the range check, as we do not want to do
    // device to host memcpy during a stream.
    const Index first_dim_size = params.dimension(0);
    const Index indices_size = indices.size();
    const Index updates_size = updates.size();
    CudaLaunchConfig config = GetCudaLaunchConfig(updates_size, d);
    TF_CHECK_OK(CudaLaunchKernel(
        scatter_op_gpu::ScatterOpCustomKernel<T, Index, op>, config.block_count,
        config.thread_per_block, 0, d.stream(), params.data(), updates.data(),
        indices.data(), first_dim_size, updates_size, indices_size));
    return -1;
  }
};

template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterScalarFunctor<GPUDevice, T, Index, op> {
  Index operator()(OpKernelContext* c, const GPUDevice& d,
                   typename TTypes<T>::Matrix params,
                   const typename TTypes<T>::ConstScalar update,
                   typename TTypes<Index>::ConstFlat indices) {
    // TODO(b/31801742): Implement indices range check. The hardest part is
    // with returning a value after the range check, as we do not want to do
    // device to host memcpy during a stream.
    const Index first_dim_size = params.dimension(0);
    const Index indices_size = indices.size();
    const Index synthesized_updates_size = indices_size * params.dimension(1);
    CudaLaunchConfig config = GetCudaLaunchConfig(synthesized_updates_size, d);
    TF_CHECK_OK(CudaLaunchKernel(
        scatter_op_gpu::ScatterScalarOpCustomKernel<T, Index, op>,
        config.block_count, config.thread_per_block, 0, d.stream(),
        params.data(), update.data(), indices.data(), first_dim_size,
        indices_size, synthesized_updates_size));
    return -1;
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_KERNELS_SCATTER_FUNCTOR_GPU_CU_H_
