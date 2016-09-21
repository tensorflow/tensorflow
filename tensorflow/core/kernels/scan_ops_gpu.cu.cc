/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/register_types.h"

#include "tensorflow/core/kernels/scan_ops.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::Index Index;

#define DEFINE(REDUCER, T) template struct functor::Scan<GPUDevice, REDUCER, T>;

#define DEFINE_FOR_ALL_REDUCERS(T)           \
  DEFINE(Eigen::internal::SumReducer<T>, T); \
  DEFINE(Eigen::internal::ProdReducer<T>, T);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_FOR_ALL_REDUCERS);
#undef DEFINE_FOR_ALL_REDUCERS
#undef DEFINE

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
