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

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/dense_update_functor.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T>
struct DenseUpdate<GPUDevice, T, ASSIGN> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) = update;
  }
};

template <typename T>
struct DenseUpdate<GPUDevice, T, ADD> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) += update;
  }
};

template <typename T>
struct DenseUpdate<GPUDevice, T, SUB> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) -= update;
  }
};

}  // namespace functor

#define DEFINE_GPU_KERNELS(T)                              \
  template struct functor::DenseUpdate<GPUDevice, T, ADD>; \
  template struct functor::DenseUpdate<GPUDevice, T, SUB>;
TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);
TF_CALL_INTEGRAL_TYPES(DEFINE_GPU_KERNELS);
TF_CALL_float8_e5m2(DEFINE_GPU_KERNELS);
TF_CALL_float8_e4m3fn(DEFINE_GPU_KERNELS);
#undef DEFINE_GPU_KERNELS

#define DEFINE_GPU_KERNELS(T) \
  template struct functor::DenseUpdate<GPUDevice, T, ASSIGN>;
TF_CALL_GPU_ALL_TYPES(DEFINE_GPU_KERNELS);
TF_CALL_INTEGRAL_TYPES(DEFINE_GPU_KERNELS);
TF_CALL_float8_e5m2(DEFINE_GPU_KERNELS);
TF_CALL_float8_e4m3fn(DEFINE_GPU_KERNELS);
TF_CALL_int4(DEFINE_GPU_KERNELS);
TF_CALL_uint4(DEFINE_GPU_KERNELS);
#undef DEFINE_GPU_KERNELS

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
