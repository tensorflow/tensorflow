/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_REDUCTION_OPS_COMMON_GPU_H_
#define TENSORFLOW_CORE_KERNELS_REDUCTION_OPS_COMMON_GPU_H_

#if !GOOGLE_CUDA && !TENSORFLOW_USE_ROCM
#error This file must only be included when building with GPU support
#endif

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

template <typename Reducer>
struct ReduceFunctor<Eigen::GpuDevice, Reducer> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Reducer& reducer);

  template <typename OUT_T>
  static void FillIdentity(const Eigen::GpuDevice& d, OUT_T out,
                           const Reducer& reducer);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_REDUCTION_OPS_COMMON_GPU_H_
