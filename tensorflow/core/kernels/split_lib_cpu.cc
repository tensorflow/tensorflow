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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/split_lib.h"

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

template <typename T>
void Split<Eigen::ThreadPoolDevice, T>::operator()(
    const Eigen::ThreadPoolDevice& d, typename TTypes<T, 3>::Tensor output,
    typename TTypes<T, 3>::ConstTensor input,
    const Eigen::DSizes<Eigen::DenseIndex, 3>& slice_indices,
    const Eigen::DSizes<Eigen::DenseIndex, 3>& slice_sizes) {
  if (output.size() < 131072) {
    output = input.slice(slice_indices, slice_sizes);
  } else {
    output.device(d) = input.slice(slice_indices, slice_sizes);
  }
}

#define DEFINE_CPU_KERNELS(T) template struct Split<Eigen::ThreadPoolDevice, T>;

TF_CALL_ALL_TYPES(DEFINE_CPU_KERNELS)
DEFINE_CPU_KERNELS(quint8)
DEFINE_CPU_KERNELS(bfloat16)

}  // namespace functor
}  // namespace tensorflow
