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

#ifndef TENSORFLOW_KERNELS_SOFTPLUS_OP_H_
#define TENSORFLOW_KERNELS_SOFTPLUS_OP_H_
// Functor definition for SoftplusOp and SoftplusGradOp, must be compilable by
// nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

// Functor used by SoftplusOp to do the computations.
template <typename Device, typename T>
struct Softplus {
  // Computes Softplus activation.
  //
  // features: any shape.
  // activations: same shape as "features".
  void operator()(const Device& d, typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor activations) {
    activations.device(d) =
        (features > features.constant(T(30)))
            .select(features, (features.exp() + features.constant(T(1))).log());
  }
};

// Functor used by SoftplusGradOp to do the computations.
template <typename Device, typename T>
struct SoftplusGrad {
  // Computes SoftplusGrad backprops.
  //
  // gradients: gradients backpropagated to the Softplus op.
  // features: inputs that where passed to the Softplus op.
  // backprops: gradients to backpropagate to the Softplus inputs.
  void operator()(const Device& d, typename TTypes<T>::ConstTensor gradients,
                  typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor backprops) {
    backprops.device(d) =
        gradients / ((-features).exp() + features.constant(T(1)));
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_SOFTPLUS_OP_H_
