#ifndef TENSORFLOW_KERNELS_REVERSE_OP_H_
#define TENSORFLOW_KERNELS_REVERSE_OP_H_

#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

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

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_MIRROR_OP_H_
