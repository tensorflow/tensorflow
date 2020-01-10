#ifndef TENSORFLOW_KERNELS_SLICE_OP_H_
#define TENSORFLOW_KERNELS_SLICE_OP_H_

// Functor definition for SliceOp, must be compilable by nvcc.

#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace functor {

template <typename Device, typename T, int NDIMS>
struct Slice {
  void operator()(const Device& d, typename TTypes<T, NDIMS>::Tensor output,
                  typename TTypes<T, NDIMS>::ConstTensor input,
                  const Eigen::DSizes<ptrdiff_t, NDIMS>& slice_indices,
                  const Eigen::DSizes<ptrdiff_t, NDIMS>& slice_sizes) {
    output.device(d) = input.slice(slice_indices, slice_sizes);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_SLICE_OP_H_
