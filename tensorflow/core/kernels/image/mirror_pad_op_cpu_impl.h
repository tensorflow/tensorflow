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

#ifndef TENSORFLOW_CORE_KERNELS_IMAGE_MIRROR_PAD_OP_CPU_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_IMAGE_MIRROR_PAD_OP_CPU_IMPL_H_

#if CPU_PROVIDED_IXDIM
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/image/mirror_pad_op.h"

namespace tensorflow {

using CpuDevice = Eigen::ThreadPoolDevice;

#define DEFINE_CPU_SPECS(T)                                                    \
  template struct functor::MirrorPad<CpuDevice, T, int32, CPU_PROVIDED_IXDIM>; \
  template struct functor::MirrorPad<CpuDevice, T, int64_t, CPU_PROVIDED_IXDIM>;
TF_CALL_POD_TYPES(DEFINE_CPU_SPECS);
TF_CALL_QUANTIZED_TYPES(DEFINE_CPU_SPECS);
TF_CALL_tstring(DEFINE_CPU_SPECS);
#undef DEFINE_CPU_SPECS

#define DEFINE_CPU_SPECS(T)                                     \
  template struct functor::MirrorPadGrad<CpuDevice, T, int32,   \
                                         CPU_PROVIDED_IXDIM>;   \
  template struct functor::MirrorPadGrad<CpuDevice, T, int64_t, \
                                         CPU_PROVIDED_IXDIM>;
TF_CALL_NUMBER_TYPES(DEFINE_CPU_SPECS);
#undef DEFINE_CPU_SPECS
}  // namespace tensorflow

#endif  // CPU_PROVIDED_IXDIM
#endif  // TENSORFLOW_CORE_KERNELS_IMAGE_MIRROR_PAD_OP_CPU_IMPL_H_
