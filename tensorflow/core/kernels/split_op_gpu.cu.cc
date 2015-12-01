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

#include <stdio.h>

#include "tensorflow/core/kernels/split_op.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
void Split<Device, T>::operator()(
    const Device& d, typename TTypes<T, 3>::Tensor output,
    typename TTypes<T, 3>::ConstTensor input,
    const Eigen::DSizes<Eigen::DenseIndex, 3>& slice_indices,
    const Eigen::DSizes<Eigen::DenseIndex, 3>& slice_sizes) {
  To32Bit(output).device(d) = To32Bit(input).slice(slice_indices, slice_sizes);
}

#define DEFINE_GPU_KERNELS(T) template struct Split<Eigen::GpuDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
