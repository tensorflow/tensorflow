#ifndef TENSORFLOW_KERNELS_L2LOSS_OP_H_
#define TENSORFLOW_KERNELS_L2LOSS_OP_H_
// Functor definition for L2LossOp, must be compilable by nvcc.
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

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
