#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>

#include "tensorflow/core/kernels/split_op.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
void Split<Device, T>::operator()(
    const Device& d, typename TTypes<T, 3>::Tensor output,
    typename TTypes<T, 3>::ConstTensor input,
    const Eigen::DSizes<ptrdiff_t, 3>& slice_indices,
    const Eigen::DSizes<ptrdiff_t, 3>& slice_sizes) {
  output.device(d) = input.slice(slice_indices, slice_sizes);
}

#define DEFINE_GPU_KERNELS(T) template struct Split<Eigen::GpuDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
