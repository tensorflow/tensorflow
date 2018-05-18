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

#if GOOGLE_CUDA

#include "tensorflow/core/kernels/scatter_functor.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

class OpKernelContext;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// Forward declarations of the functor specializations for GPU.
#define DECLARE_GPU_SPECS_OP(T, Index, op)                         \
  template <>                                                      \
  Index ScatterFunctor<GPUDevice, T, Index, op>::operator()(       \
      OpKernelContext* c, const GPUDevice& d,                      \
      typename TTypes<T>::Matrix params,                           \
      typename TTypes<T>::ConstMatrix updates,                     \
      typename TTypes<Index>::ConstFlat indices);                  \
  extern template struct ScatterFunctor<GPUDevice, T, Index, op>;  \
  template <>                                                      \
  Index ScatterScalarFunctor<GPUDevice, T, Index, op>::operator()( \
      OpKernelContext* c, const GPUDevice& d,                      \
      typename TTypes<T>::Matrix params,                           \
      const typename TTypes<T>::ConstScalar update,                \
      typename TTypes<Index>::ConstFlat indices);                  \
  extern template struct ScatterScalarFunctor<GPUDevice, T, Index, op>;

#define DECLARE_GPU_SPECS_INDEX(T, Index)                       \
  DECLARE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::ASSIGN); \
  DECLARE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::ADD);    \
  DECLARE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::SUB);    \
  DECLARE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::MUL);    \
  DECLARE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::DIV);    \
  DECLARE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::MIN);    \
  DECLARE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::MAX);

#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, int32); \
  DECLARE_GPU_SPECS_INDEX(T, int64);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX
#undef DECLARE_GPU_SPECS_OP

}  // namespace functor
}  // namespace tensorflow

#else

#include "tensorflow/core/kernels/scatter_functor.h"

#endif  // GOOGLE_CUDA
