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

#include "tensorflow/core/kernels/sparse_xent_op.h"

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// Partial specialization for a GPUDevice, that uses the Eigen implementation
// from XentEigenImpl.
namespace functor {
template <typename T, typename Index>
struct SparseXentFunctor<GPUDevice, T, Index> {
  void operator()(const GPUDevice& d, typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<Index>::ConstVec labels,
                  typename TTypes<T>::Vec scratch, typename TTypes<T>::Vec loss,
                  typename TTypes<T>::Matrix backprop) {
    SparseXentEigenImpl<GPUDevice, T, Index>::Compute(d, logits, labels,
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

#endif  // GOOGLE_CUDA
