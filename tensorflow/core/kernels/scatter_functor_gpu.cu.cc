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

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/scatter_functor_gpu.cu.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_GPU_SPECS_OP(T, Index, op) \
  template struct functor::ScatterFunctor<GPUDevice, T, Index, op>;

#define DEFINE_GPU_SPECS_INDEX(T, Index)                       \
  DEFINE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::ASSIGN); \
  DEFINE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::ADD);    \
  DEFINE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::SUB);    \
  DEFINE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::MUL);    \
  DEFINE_GPU_SPECS_OP(T, Index, scatter_op::UpdateOp::DIV);

#define DEFINE_GPU_SPECS(T)         \
  DEFINE_GPU_SPECS_INDEX(T, int32); \
  DEFINE_GPU_SPECS_INDEX(T, int64);

DEFINE_GPU_SPECS(float);
DEFINE_GPU_SPECS(double);
// TODO(b/27222123): The following fails to compile due to lack of support for
// fp16.
// TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS
#undef DEFINE_GPU_SPECS_INDEX
#undef DEFINE_GPU_SPECS_OP

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
