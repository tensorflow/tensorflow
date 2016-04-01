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

#ifndef TENSORFLOW_KERNELS_SLICE_OP_H_
#define TENSORFLOW_KERNELS_SLICE_OP_H_

// Functor definition for SliceOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T, int NDIMS>
struct Slice {
  void operator()(const Device& d, typename TTypes<T, NDIMS>::Tensor output,
                  typename TTypes<T, NDIMS>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& slice_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& slice_sizes) {
    if (Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
      Eigen::DSizes<int, NDIMS> indices;
      for (int i = 0; i < NDIMS; ++i) {
        indices[i] = slice_indices[i];
      }
      Eigen::DSizes<int, NDIMS> sizes;
      for (int i = 0; i < NDIMS; ++i) {
        sizes[i] = slice_sizes[i];
      }
      To32Bit(output).device(d) = To32Bit(input).slice(indices, sizes);
    } else {
      output.device(d) = input.slice(slice_indices, slice_sizes);
    }
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_SLICE_OP_H_
