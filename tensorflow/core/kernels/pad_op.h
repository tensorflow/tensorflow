#ifndef TENSORFLOW_KERNELS_PAD_OP_H_
#define TENSORFLOW_KERNELS_PAD_OP_H_
// Functor definition for PadOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {
namespace functor {

// Functor used by PadOp to do the computations.
template <typename Device, typename T, int Dims>
struct Pad {
  // Pad "input" into "output", as specified by "paddings".  See pad_op.cc for
  // details.
  void operator()(const Device& d, typename TTypes<T, Dims>::Tensor output,
                  typename TTypes<T, Dims>::ConstTensor input,
                  Eigen::array<std::pair<int32, int32>, Dims> paddings) {
    output.device(d) = input.pad(paddings);
  }
};

template <typename Device, typename T>
struct Pad<Device, T, 0> {
  // In the scalar case we simply copy the input.
  void operator()(const Device& d, typename TTypes<T, 0>::Tensor output,
                  typename TTypes<T, 0>::ConstTensor input,
                  Eigen::array<std::pair<int32, int32>, 0>) {
    output.device(d) = input;
  }
};
}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_PAD_OP_H_
