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

#ifndef TENSORFLOW_KERNELS_REVERSE_OP_H_
#define TENSORFLOW_KERNELS_REVERSE_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

// Functor used by MirrorOp to do the computations.
template <typename Device, typename T, int Dims>
struct Reverse {
  void operator()(const Device& d, typename TTypes<T, Dims>::ConstTensor input,
                  typename TTypes<bool, 1>::ConstTensor dims,
                  typename TTypes<T, Dims>::Tensor output) {
    // mirror is in host memory
    Eigen::array<bool, Dims> reverse_dims;
    for (int i = 0; i < Dims; ++i) {
      reverse_dims[i] = dims(i);
    }
    output.device(d) = input.reverse(reverse_dims);
  }
};

template <typename Device, typename T>
struct Reverse<Device, T, 0> {
  void operator()(const Device& d, typename TTypes<T, 0>::ConstTensor input,
                  typename TTypes<bool, 1>::ConstTensor,
                  typename TTypes<T, 0>::Tensor output) {
    // Reversing a scalar is copying it.
    output.device(d) = input;
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_MIRROR_OP_H_
