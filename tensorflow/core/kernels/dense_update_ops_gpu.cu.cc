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

#include "tensorflow/core/kernels/dense_update_ops.h"

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_UPDATE(T, OP) \
  template struct functor::DenseUpdate<GPUDevice, T, DenseUpdateType::OP>;
#define DEFINE_ADD_SUB(T) \
  DEFINE_UPDATE(T, ADD)   \
  DEFINE_UPDATE(T, SUB)
#define DEFINE_ASSIGN(T) DEFINE_UPDATE(T, ASSIGN)
TF_CALL_GPU_NUMBER_TYPES(DEFINE_ADD_SUB);
TF_CALL_POD_TYPES(DEFINE_ASSIGN);
#undef DEFINE_UPDATE
#undef DEFINE_ADD_SUB
#undef DEFINE_ASSIGN

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
