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

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/aggregate_ops.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// Partial specialization for a GPUDevice, that uses the Eigen implementation.
namespace functor {
template <typename T>
struct Add2Functor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2) {
    Add2EigenImpl<GPUDevice, T>::Compute(d, out, in1, in2);
  }
};

template <typename T>
struct Add3Functor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3) {
    Add3EigenImpl<GPUDevice, T>::Compute(d, out, in1, in2, in3);
  }
};

template <typename T>
struct Add4Functor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4) {
    Add4EigenImpl<GPUDevice, T>::Compute(d, out, in1, in2, in3, in4);
  }
};

template <typename T>
struct Add5Functor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4,
                  typename TTypes<T>::ConstFlat in5) {
    Add5EigenImpl<GPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5);
  }
};

template <typename T>
struct Add6Functor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4,
                  typename TTypes<T>::ConstFlat in5,
                  typename TTypes<T>::ConstFlat in6) {
    Add6EigenImpl<GPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6);
  }
};

template <typename T>
struct Add7Functor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4,
                  typename TTypes<T>::ConstFlat in5,
                  typename TTypes<T>::ConstFlat in6,
                  typename TTypes<T>::ConstFlat in7) {
    Add7EigenImpl<GPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6,
                                         in7);
  }
};

template <typename T>
struct Add8Functor<GPUDevice, T> {
  void operator()(
      const GPUDevice& d, typename TTypes<T>::Flat out,
      typename TTypes<T>::ConstFlat in1, typename TTypes<T>::ConstFlat in2,
      typename TTypes<T>::ConstFlat in3, typename TTypes<T>::ConstFlat in4,
      typename TTypes<T>::ConstFlat in5, typename TTypes<T>::ConstFlat in6,
      typename TTypes<T>::ConstFlat in7, typename TTypes<T>::ConstFlat in8) {
    Add8EigenImpl<GPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6,
                                         in7, in8);
  }
};

template <typename T>
struct Add8pFunctor<GPUDevice, T> {
  void operator()(
      const GPUDevice& d, typename TTypes<T>::Flat out,
      typename TTypes<T>::ConstFlat in1, typename TTypes<T>::ConstFlat in2,
      typename TTypes<T>::ConstFlat in3, typename TTypes<T>::ConstFlat in4,
      typename TTypes<T>::ConstFlat in5, typename TTypes<T>::ConstFlat in6,
      typename TTypes<T>::ConstFlat in7, typename TTypes<T>::ConstFlat in8) {
    Add8pEigenImpl<GPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6,
                                          in7, in8);
  }
};

template <typename T>
struct Add9Functor<GPUDevice, T> {
  void operator()(
      const GPUDevice& d, typename TTypes<T>::Flat out,
      typename TTypes<T>::ConstFlat in1, typename TTypes<T>::ConstFlat in2,
      typename TTypes<T>::ConstFlat in3, typename TTypes<T>::ConstFlat in4,
      typename TTypes<T>::ConstFlat in5, typename TTypes<T>::ConstFlat in6,
      typename TTypes<T>::ConstFlat in7, typename TTypes<T>::ConstFlat in8,
      typename TTypes<T>::ConstFlat in9) {
    Add9EigenImpl<GPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6,
                                         in7, in8, in9);
  }
};

}  // end namespace functor

// Instantiate the GPU implementation for GPU number types.
#define REGISTER_FUNCTORS(type)                           \
  template struct functor::Add2Functor<GPUDevice, type>;  \
  template struct functor::Add3Functor<GPUDevice, type>;  \
  template struct functor::Add4Functor<GPUDevice, type>;  \
  template struct functor::Add5Functor<GPUDevice, type>;  \
  template struct functor::Add6Functor<GPUDevice, type>;  \
  template struct functor::Add7Functor<GPUDevice, type>;  \
  template struct functor::Add8Functor<GPUDevice, type>;  \
  template struct functor::Add8pFunctor<GPUDevice, type>; \
  template struct functor::Add9Functor<GPUDevice, type>;

TF_CALL_int64(REGISTER_FUNCTORS);
TF_CALL_uint32(REGISTER_FUNCTORS);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_FUNCTORS);
TF_CALL_COMPLEX_TYPES(REGISTER_FUNCTORS);

#undef REGISTER_FUNCTORS

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
