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

#include "tensorflow/core/kernels/strided_slice_op.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_GPU_KERNELS(T)                                   \
  template struct functor::StridedSlice<GPUDevice, T, 1>;       \
  template struct functor::StridedSlice<GPUDevice, T, 2>;       \
  template struct functor::StridedSlice<GPUDevice, T, 3>;       \
  template struct functor::StridedSlice<GPUDevice, T, 4>;       \
  template struct functor::StridedSlice<GPUDevice, T, 5>;       \
  template struct functor::StridedSlice<GPUDevice, T, 6>;       \
  template struct functor::StridedSlice<GPUDevice, T, 7>;       \
  template struct functor::StridedSliceGrad<GPUDevice, T, 1>;   \
  template struct functor::StridedSliceGrad<GPUDevice, T, 2>;   \
  template struct functor::StridedSliceGrad<GPUDevice, T, 3>;   \
  template struct functor::StridedSliceGrad<GPUDevice, T, 4>;   \
  template struct functor::StridedSliceGrad<GPUDevice, T, 5>;   \
  template struct functor::StridedSliceGrad<GPUDevice, T, 6>;   \
  template struct functor::StridedSliceGrad<GPUDevice, T, 7>;   \
  template struct functor::StridedSliceAssign<GPUDevice, T, 1>; \
  template struct functor::StridedSliceAssign<GPUDevice, T, 2>; \
  template struct functor::StridedSliceAssign<GPUDevice, T, 3>; \
  template struct functor::StridedSliceAssign<GPUDevice, T, 4>; \
  template struct functor::StridedSliceAssign<GPUDevice, T, 5>; \
  template struct functor::StridedSliceAssign<GPUDevice, T, 6>; \
  template struct functor::StridedSliceAssign<GPUDevice, T, 7>; \
  template struct functor::StridedSliceAssignScalar<GPUDevice, T>;
TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);
TF_CALL_complex64(DEFINE_GPU_KERNELS);
TF_CALL_complex128(DEFINE_GPU_KERNELS);
DEFINE_GPU_KERNELS(int32);

#undef DEFINE_GPU_KERNELS

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
