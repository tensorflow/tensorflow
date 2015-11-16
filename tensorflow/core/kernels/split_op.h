#ifndef TENSORFLOW_KERNELS_SPLIT_OP_H_
#define TENSORFLOW_KERNELS_SPLIT_OP_H_
// Functor definition for SplitOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct Split {
  void operator()(const Device& d, typename TTypes<T, 3>::Tensor output,
                  typename TTypes<T, 3>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, 3>& slice_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, 3>& slice_sizes);
};

template <typename T>
struct Split<Eigen::ThreadPoolDevice, T> {
  void operator()(const Eigen::ThreadPoolDevice& d,
                  typename TTypes<T, 3>::Tensor output,
                  typename TTypes<T, 3>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, 3>& slice_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, 3>& slice_sizes);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_SPLIT_OP_H_
