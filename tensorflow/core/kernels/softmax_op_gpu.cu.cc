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

#include "tensorflow/core/kernels/softmax_op_functor.h"

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// Partial specialization for a GPUDevice, that uses the Eigen implementation
// from SoftmaxEigenImpl.
namespace functor {
template <typename T>
struct SoftmaxFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<T>::Matrix softmax, const bool log) {
    SoftmaxEigenImpl<GPUDevice, T>::Compute(d, logits, softmax, log);
  }
};
}  // end namespace functor

// Instantiate the GPU implementation for float.
template struct functor::SoftmaxFunctor<GPUDevice, float>;

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
