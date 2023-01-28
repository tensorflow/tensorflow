/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/fill_empty_rows_functor.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"


namespace tensorflow {

namespace functor {

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
#define DECLARE_GPU_SPEC(T, Tindex)                                            \
  template <>                                                                  \
  Status FillEmptyRows<GPUDevice, T, Tindex, false>::operator()(               \
      OpKernelContext* context, const Tensor& default_value_t,                 \
      const Tensor& indices_t, const Tensor& values_t,                         \
      const Tensor& dense_shape_t, typename AsyncOpKernel::DoneCallback done); \
  extern template struct FillEmptyRows<GPUDevice, T, Tindex, false>;           \
  template <>                                                                  \
  Status FillEmptyRows<GPUDevice, T, Tindex, true>::operator()(                \
      OpKernelContext* context, const Tensor& default_value_t,                 \
      const Tensor& indices_t, const Tensor& values_t,                         \
      const Tensor& dense_shape_t, typename AsyncOpKernel::DoneCallback done); \
  extern template struct FillEmptyRows<GPUDevice, T, Tindex,true>;
#define DECLARE_GPU_SPEC_INT64(T) DECLARE_GPU_SPEC(T, int64_t)
TF_CALL_POD_TYPES(DECLARE_GPU_SPEC_INT64)
#undef DECLARE_GPU_SPEC_INT64
#undef DECLARE_GPU_SPEC


// Forward declarations of the functor specializations for GPU.
#define DECLARE_GPU_SPEC(T, Tindex)                                            \
  template <>                                                                  \
  Status FillEmptyRowsGrad<GPUDevice, T, Tindex>::operator()(                  \
      OpKernelContext* context,                                                \
      typename TTypes<Tindex>::ConstVec reverse_index_map,                     \
      typename TTypes<T>::ConstVec grad_values,                                \
      typename TTypes<T>::Vec d_values,                                        \
      typename TTypes<T>::Scalar d_default_value);                             \
  extern template struct FillEmptyRowsGrad<GPUDevice, T, Tindex>;
#define DECLARE_GPU_SPEC_INT64(T) DECLARE_GPU_SPEC(T, int64_t)
TF_CALL_REAL_NUMBER_TYPES(DECLARE_GPU_SPEC_INT64);
#undef DECLARE_GPU_SPEC_INT64
#undef DECLARE_GPU_SPEC
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace functor
}  // namespace tensorflow
