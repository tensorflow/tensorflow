#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/split_op.h"

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

template <typename T>
void Split<Eigen::ThreadPoolDevice, T>::operator()(
    const Eigen::ThreadPoolDevice& d, typename TTypes<T, 3>::Tensor output,
    typename TTypes<T, 3>::ConstTensor input,
    const Eigen::DSizes<ptrdiff_t, 3>& slice_indices,
    const Eigen::DSizes<ptrdiff_t, 3>& slice_sizes) {
  if (output.size() < 131072) {
    output = input.slice(slice_indices, slice_sizes);
  } else {
    output.device(d) = input.slice(slice_indices, slice_sizes);
  }
}

#define DEFINE_CPU_KERNELS(T) template struct Split<Eigen::ThreadPoolDevice, T>;

TF_CALL_ALL_TYPES(DEFINE_CPU_KERNELS)

}  // namespace functor
}  // namespace tensorflow
