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

#ifndef TENSORFLOW_CORE_KERNELS_BIAS_OP_H_
#define TENSORFLOW_CORE_KERNELS_BIAS_OP_H_
// Functor definition for BiasOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

// Functor used by BiasOp to do the computations.
template <typename Device, typename T>
struct Bias {
  // Add "bias" to "input", repeating "bias".
  void operator()(const Device& d, typename TTypes<T>::ConstFlat input,
                  typename TTypes<T>::ConstVec bias,
                  typename TTypes<T>::Flat output) {
    const Eigen::Index rest_size = input.size() / bias.dimension(0);
    Eigen::DSizes<Eigen::Index, 1> bcast(rest_size);
    MaybeWith32BitIndexing<Device>(
        [&](auto input32, auto bias32, auto output32, const auto& bcast32) {
          output32.device(d) = input32 + bias32.broadcast(bcast32);
        },
        input, bias, output, bcast);
  }

  // NCHW layout, repeating on the first dimension, broadcasting on the last
  // dimension.
  void operator()(const Device& d, typename TTypes<T>::ConstMatrix input,
                  typename TTypes<T>::ConstMatrix bias1,  // shape [C, 1].
                  typename TTypes<T>::Matrix output) {
    const Eigen::Index rest_size = input.dimension(0) / bias1.dimension(0);
    Eigen::DSizes<Eigen::Index, 2> bcast(rest_size, input.dimension(1));
    MaybeWith32BitIndexing<Device>(
        [&](auto input32, auto bias32, auto output32, const auto& bcast32) {
          output32.device(d) = input32 + bias32.broadcast(bcast32);
        },
        input, bias1, output, bcast);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BIAS_OP_H_
