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

#include "tensorflow/core/kernels/adjust_contrast_op.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// this is for v2
template struct functor::AdjustContrastv2<GPUDevice>;

// these are for v1
template struct functor::AdjustContrast<GPUDevice, uint8>;
template struct functor::AdjustContrast<GPUDevice, int8>;
template struct functor::AdjustContrast<GPUDevice, int16>;
template struct functor::AdjustContrast<GPUDevice, int32>;
template struct functor::AdjustContrast<GPUDevice, int64>;
template struct functor::AdjustContrast<GPUDevice, float>;
template struct functor::AdjustContrast<GPUDevice, double>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
