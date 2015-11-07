#ifndef TENSORFLOW_KERNELS_PAD_OP_H_
#define TENSORFLOW_KERNELS_PAD_OP_H_
// Functor definition for PadOp, must be compilable by nvcc.

#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

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

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_PAD_OP_H_
