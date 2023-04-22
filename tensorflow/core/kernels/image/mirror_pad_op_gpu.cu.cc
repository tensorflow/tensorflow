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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/image/mirror_pad_op.h"

namespace tensorflow {

using GpuDevice = Eigen::GpuDevice;

#define DEFINE_GPU_SPECS(T)                                       \
  template struct functor::MirrorPad<GpuDevice, T, int32, 1>;     \
  template struct functor::MirrorPad<GpuDevice, T, int32, 2>;     \
  template struct functor::MirrorPad<GpuDevice, T, int32, 3>;     \
  template struct functor::MirrorPad<GpuDevice, T, int32, 4>;     \
  template struct functor::MirrorPad<GpuDevice, T, int32, 5>;     \
  template struct functor::MirrorPad<GpuDevice, T, int64, 1>;     \
  template struct functor::MirrorPad<GpuDevice, T, int64, 2>;     \
  template struct functor::MirrorPad<GpuDevice, T, int64, 3>;     \
  template struct functor::MirrorPad<GpuDevice, T, int64, 4>;     \
  template struct functor::MirrorPad<GpuDevice, T, int64, 5>;     \
  template struct functor::MirrorPadGrad<GpuDevice, T, int32, 1>; \
  template struct functor::MirrorPadGrad<GpuDevice, T, int32, 2>; \
  template struct functor::MirrorPadGrad<GpuDevice, T, int32, 3>; \
  template struct functor::MirrorPadGrad<GpuDevice, T, int32, 4>; \
  template struct functor::MirrorPadGrad<GpuDevice, T, int32, 5>; \
  template struct functor::MirrorPadGrad<GpuDevice, T, int64, 1>; \
  template struct functor::MirrorPadGrad<GpuDevice, T, int64, 2>; \
  template struct functor::MirrorPadGrad<GpuDevice, T, int64, 3>; \
  template struct functor::MirrorPadGrad<GpuDevice, T, int64, 4>; \
  template struct functor::MirrorPadGrad<GpuDevice, T, int64, 5>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);
#undef DEFINE_GPU_SPECS

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
