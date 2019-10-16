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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/sparse_xent_op.h"

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/reduction_gpu_kernels.cu.h"
#include "tensorflow/core/kernels/reduction_ops_common.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA
namespace gpuprim = ::cub;
#elif TENSORFLOW_USE_ROCM
namespace gpuprim = ::hipcub;
#endif

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// Partial specialization for a GPUDevice, that uses the CUB implementation
// from reduction_gpu_kernels.cu.h.
template <typename T>
struct RowMaxReduction<GPUDevice, T> {
  // Computes the maximum across the rows of logits
  //
  // logits: batch_size, num_classes.
  // maximum: temporary tensor, dims: batch_size, 1
  static inline void Compute(OpKernelContext* ctx,
                             typename TTypes<T>::ConstMatrix logits,
                             typename TTypes<T>::Vec maximum) {
    const int kBatchDim = 0;
    const int kClassDim = 1;
    const int rows = logits.dimension(kBatchDim);
    const int cols = logits.dimension(kClassDim);

    typedef const Eigen::array<TTypes<float>::Tensor::Index, 1>& ReductionAxes;
    Constants<GPUDevice> constants;
    gpuprim::Max op;
    functor::ReduceImpl<T, gpuprim::Max, T*, const T*, ReductionAxes>(
        ctx, maximum.data(), logits.data(), 2, rows, cols, 1, 1, constants.kOne,
        op);
  }
};

// Partial specialization for a GPUDevice, that uses the Eigen implementation
// from XentEigenImpl.
template <typename T, typename Index>
struct SparseXentFunctor<GPUDevice, T, Index> {
  void operator()(OpKernelContext* ctx, typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<Index>::ConstVec labels,
                  typename TTypes<T>::Vec scratch, typename TTypes<T>::Vec loss,
                  typename TTypes<T>::Matrix backprop) {
    SparseXentEigenImpl<GPUDevice, T, Index>::Compute(ctx, logits, labels,
                                                      scratch, loss, backprop);
  }
};
}  // end namespace functor

// Instantiate the GPU implementation for float.
#define REGISTER(Index)                                                      \
  template struct functor::SparseXentFunctor<GPUDevice, float, Index>;       \
  template class generator::SparseXentGradGenerator<float, Index>;           \
  template struct functor::SparseXentFunctor<GPUDevice, Eigen::half, Index>; \
  template class generator::SparseXentGradGenerator<Eigen::half, Index>;
REGISTER(int32)
REGISTER(int64)
#undef REGISTER

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
