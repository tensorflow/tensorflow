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

#ifndef TENSORFLOW_KERNELS_L2LOSS_OP_H_
#define TENSORFLOW_KERNELS_L2LOSS_OP_H_
// Functor definition for L2LossOp, must be compilable by nvcc.
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

// Functor used by L2LossOp to do the computations.
template <typename Device, typename T>
struct L2Loss {
  void operator()(const Device& d, typename TTypes<T>::ConstTensor input,
                  typename TTypes<T>::Scalar output) {
    // We flatten the input tensor and reduce on dimension 0, producing
    // a single number which is Mul(Sum(x^2), 0.5).
    output.device(d) = input.square().sum() * static_cast<T>(0.5);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_L2LOSS_OP_H_
