/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/kernels/scatter_op.h"

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T, typename Index, scatter_op::UpdateOp op>
__global__ void ScatterOpCustomKernel(
    T* params, const T* updates, const Index* indices,
    Index first_dim_size, Index updates_size, Index indices_size) {
  Index update_block = updates_size / indices_size;
  CUDA_1D_KERNEL_LOOP(i, updates_size) {
    int indices_i = i / update_block;
    int updates_i = i;
    int param_first_index = indices[indices_i];
    if (!(param_first_index >= 0 && param_first_index < first_dim_size)) {
      // Ignore indices that are out of range.
      continue;
    }
    int params_i = param_first_index * update_block + (i % update_block);
    switch (op) {
      case scatter_op::UpdateOp::ASSIGN: {
        params[params_i] = ldg(updates + updates_i);
        break;
      }
      case scatter_op::UpdateOp::ADD: {
        CudaAtomicAdd(params + params_i, ldg(updates + updates_i));
        break;
      }
      case scatter_op::UpdateOp::SUB: {
        CudaAtomicSub(params + params_i, ldg(updates + updates_i));
        break;
      }
    }
  }
}

namespace functor {
// Specialization for a GPU device.
template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterFunctor<GPUDevice, T, Index, op> {
  Index operator()(OpKernelContext* c, const GPUDevice& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices) {
    // TODO: Implement indices range check.  The hardest part is with returning
    // a value after the range check, as we do not want to do device to host
    // memcpy during a stream.
    const Index first_dim_size = params.dimension(0);
    const Index indices_size = indices.size();
    const Index updates_size = updates.size();
    CudaLaunchConfig config = GetCudaLaunchConfig(updates_size, d);
    ScatterOpCustomKernel<T,Index,op>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            params.data(), updates.data(), indices.data(),
            first_dim_size, updates_size, indices_size);
    return -1;
  }
};

}  // namespace functor

#define DEFINE_GPU_SPECS_OP(T, Index, op)                               \
  template struct functor::ScatterFunctor<GPUDevice, T, Index, op>;

#define DEFINE_GPU_SPECS_INDEX(T, Index)                        \
  DEFINE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::ASSIGN);  \
  DEFINE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::ADD);     \
  DEFINE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::SUB);

#define DEFINE_GPU_SPECS(T)                     \
  DEFINE_GPU_SPECS_INDEX(T, int32);             \
  DEFINE_GPU_SPECS_INDEX(T, int64);

DEFINE_GPU_SPECS(float);
DEFINE_GPU_SPECS(double);
// TODO: The following fails to compile.
// TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS
#undef DEFINE_GPU_SPECS_INDEX
#undef DEFINE_GPU_SPECS_OP

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
