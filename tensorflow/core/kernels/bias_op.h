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

#ifndef TENSORFLOW_KERNELS_BIAS_OP_H_
#define TENSORFLOW_KERNELS_BIAS_OP_H_
// Functor definition for BiasOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

// Functor used by BiasOp to do the computations.
template <typename Device, typename T, int Dims>
struct Bias {
  // Add "bias" to "input", broadcasting it on all dimensions but the last one.
  void operator()(const Device& d, typename TTypes<T, Dims>::ConstTensor input,
                  typename TTypes<T>::ConstVec bias,
                  typename TTypes<T, Dims>::Tensor output) {
    if (input.size() >= INT_MAX) {
      const int64_t bias_size = bias.dimension(0);
      const int64_t rest_size = input.size() / bias_size;
      Eigen::DSizes<int64_t, 1> one_d(input.size());
      Eigen::DSizes<int64_t, 1> bcast(rest_size);
      output.reshape(one_d).device(d) =
          input.reshape(one_d) + bias.broadcast(bcast).reshape(one_d);
    } else {
      const int bias_size = bias.dimension(0);
      const int rest_size = input.size() / bias_size;
      Eigen::DSizes<int, 1> one_d(input.size());
      Eigen::DSizes<int, 1> bcast(rest_size);
      To32Bit(output).reshape(one_d).device(d) =
          To32Bit(input).reshape(one_d) +
          To32Bit(bias).broadcast(bcast).reshape(one_d);
    }
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_BIAS_OP_H_
