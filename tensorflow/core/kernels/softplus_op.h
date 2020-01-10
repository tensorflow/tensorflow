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
        (features > features.constant(30.f))
            .select(features, (features.exp() + features.constant(1.0f)).log());
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
        gradients / ((-features).exp() + features.constant(1.0f));
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_SOFTPLUS_OP_H_
