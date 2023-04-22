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

#ifndef TENSORFLOW_CORE_KERNELS_SOFTPLUS_OP_H_
#define TENSORFLOW_CORE_KERNELS_SOFTPLUS_OP_H_
// Functor definition for SoftplusOp and SoftplusGradOp, must be compilable by
// nvcc.

// clang-format off
#include "tensorflow/core/platform/bfloat16.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
// clang-format on
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
    // Choose a threshold on x below which exp(x) may underflow
    // when added to 1, but for which exp(x) is always within epsilon of the
    // true softplus(x).  Offset of 2 from machine epsilon checked
    // experimentally for float16, float32, float64.  Checked against
    // softplus implemented with numpy's log1p and numpy's logaddexp.
    static const T threshold =
        Eigen::numext::log(Eigen::NumTraits<T>::epsilon()) + T(2);
    // Value above which exp(x) may overflow, but softplus(x) == x
    // is within machine epsilon.
    auto too_large = features > features.constant(-threshold);
    // Value below which exp(x) may underflow, but softplus(x) == exp(x)
    // is within machine epsilon.
    auto too_small = features < features.constant(threshold);
    auto features_exp = features.exp();
    activations.device(d) = too_large.select(
        features,                       // softplus(x) ~= x for x large
        too_small.select(features_exp,  // softplus(x) ~= exp(x) for x small
                         features_exp.log1p()));
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

#endif  // TENSORFLOW_CORE_KERNELS_SOFTPLUS_OP_H_
