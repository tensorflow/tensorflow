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

#ifndef TENSORFLOW_CORE_KERNELS_RELU_OP_FUNCTOR_H_
#define TENSORFLOW_CORE_KERNELS_RELU_OP_FUNCTOR_H_
// Functor definition for ReluOp and ReluGradOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

// Functor used by ReluOp to do the computations.
template <typename Device, typename T>
struct Relu {
  // Computes Relu activation.
  //
  // features: any shape.
  // activations: same shape as "features".
  void operator()(const Device& d, typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor activations) {
    activations.device(d) = features.cwiseMax(static_cast<T>(0));
  }
};

// Functor used by ReluGradOp to do the computations.
template <typename Device, typename T>
struct ReluGrad {
  // Computes ReluGrad backprops.
  //
  // gradients: gradients backpropagated to the Relu op.
  // features: either the inputs that were passed to the Relu or, or its
  //           outputs (using either one yields the same result here).
  // backprops: gradients to backpropagate to the Relu inputs.
  void operator()(const Device& d, typename TTypes<T>::ConstTensor gradients,
                  typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor backprops) {
    // NOTE: When the activation is exactly zero, we do not propagate the
    // associated gradient value. This allows the output of the Relu to be used,
    // as well as its input.
    backprops.device(d) =
        gradients * (features > static_cast<T>(0)).template cast<T>();
  }
};

// Functor used by Relu6Op to do the computations.
template <typename Device, typename T>
struct Relu6 {
  // Computes Relu6 activation.
  //
  // features: any shape.
  // activations: same shape as "features".
  void operator()(const Device& d, typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor activations) {
    activations.device(d) =
        features.cwiseMax(static_cast<T>(0)).cwiseMin(static_cast<T>(6));
  }
};

// Functor used by ReluGradOp to do the computations.
template <typename Device, typename T>
struct Relu6Grad {
  // Computes Relu6Grad backprops.
  //
  // gradients: gradients backpropagated to the Relu6 op.
  // features: inputs that where passed to the Relu6 op, or its outputs.
  // backprops: gradients to backpropagate to the Relu6 inputs.
  void operator()(const Device& d, typename TTypes<T>::ConstTensor gradients,
                  typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor backprops) {
    // NOTE: When the activation is exactly zero or six, we
    // make sure not to propagate the associated gradient
    // value. This allows "features" to be either the input or the output of
    // the relu6.
    backprops.device(d) = gradients * ((features > static_cast<T>(0)) *
                                       (features < static_cast<T>(6)))
                                          .template cast<T>();
  }
};

// Functor used by LeakyReluOp to do the computations.
template <typename Device, typename T>
struct LeakyRelu {
  // Computes LeakyRelu activation.
  //
  // features: any shape.
  // activations: same shape as "features".

  // Need to bundle the args (to the LeakyRelu functor) within a struct
  // Not doing so leads to Eigen kernel args not getting populated
  // corretly for Eigen::half type (when building on the ROCM platform)
  struct LeakyReluArgs {
    const Device& d;
    typename TTypes<T>::ConstTensor features;
    T alpha;
    typename TTypes<T>::Tensor activations;
  };
  void operator()(LeakyReluArgs args) {
    // Note that alpha might be > 1 or < 0, so we don't use cwiseMax here.
    args.activations.device(args.d) =
        (args.features > static_cast<T>(0))
            .select(args.features, args.features * args.alpha);
  }
};

// Functor used by LeakyReluGradOp to do the computations.
template <typename Device, typename T>
struct LeakyReluGrad {
  // Computes LeakyReluGrad backprops.
  //
  // gradients: gradients backpropagated to the LeakyRelu op.
  // features: either the inputs that were passed to the LeakyRelu or, or its
  //           outputs (using either one yields the same result here).
  // backprops: gradients to backpropagate to the LeakyRelu inputs.
  void operator()(const Device& d, typename TTypes<T>::ConstTensor gradients,
                  typename TTypes<T>::ConstTensor features, T alpha,
                  typename TTypes<T>::Tensor backprops) {
    backprops.device(d) =
        (features > static_cast<T>(0)).select(gradients, gradients * alpha);
  }
};

// Functor used by EluOp to do the computations.
template <typename Device, typename T>
struct Elu {
  // Computes Elu activation.
  //
  // features: any shape.
  // activations: same shape as "features".
  void operator()(const Device& d, typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor activations) {
    // features.constant(?)
    activations.device(d) =
        (features < static_cast<T>(0))
            .select(features.exp() - features.constant(static_cast<T>(1)),
                    features);
  }
};

// Functor used by EluGradOp to do the computations.
template <typename Device, typename T>
struct EluGrad {
  // Computes EluGrad backprops.
  //
  // gradients: gradients backpropagated to the Elu op.
  // activations: outputs of the Elu op.
  // backprops: gradients to backpropagate to the Elu inputs.
  void operator()(const Device& d, typename TTypes<T>::ConstTensor gradients,
                  typename TTypes<T>::ConstTensor activations,
                  typename TTypes<T>::Tensor backprops) {
    backprops.device(d) =
        (activations < static_cast<T>(0))
            .select((activations + static_cast<T>(1)) * gradients, gradients);
  }
};

// Functor used by SeluOp to do the computations.
template <typename Device, typename T>
struct Selu {
  // Computes Selu activation.
  //
  // features: any shape.
  // activations: same shape as "features".
  void operator()(const Device& d, typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor activations) {
    // features.constant(?)
    const auto scale = static_cast<T>(1.0507009873554804934193349852946);
    const auto scale_alpha = static_cast<T>(1.7580993408473768599402175208123);
    const auto one = static_cast<T>(1);
    const auto zero = static_cast<T>(0);
    activations.device(d) =
        (features < zero)
            .select(scale_alpha * (features.exp() - features.constant(one)),
                    scale * features);
  }
};

// Functor used by SeluGradOp to do the computations.
template <typename Device, typename T>
struct SeluGrad {
  // Computes SeluGrad backprops.
  //
  // gradients: gradients backpropagated to the Selu op.
  // activations: outputs of the Selu op.
  // backprops: gradients to backpropagate to the Selu inputs.
  void operator()(const Device& d, typename TTypes<T>::ConstTensor gradients,
                  typename TTypes<T>::ConstTensor activations,
                  typename TTypes<T>::Tensor backprops) {
    const auto scale = static_cast<T>(1.0507009873554804934193349852946);
    const auto scale_alpha = static_cast<T>(1.7580993408473768599402175208123);
    backprops.device(d) =
        (activations < static_cast<T>(0))
            .select(gradients * (activations + scale_alpha), gradients * scale);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_RELU_OP_FUNCTOR_H_
