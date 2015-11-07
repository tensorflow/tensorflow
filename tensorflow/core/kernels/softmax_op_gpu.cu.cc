#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/softmax_op.h"

#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// Partial specialization for a GPUDevice, that uses the Eigen implementation
// from SoftmaxEigenImpl.
namespace functor {
template <typename T>
struct SoftmaxFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<T>::Matrix softmax) {
    SoftmaxEigenImpl<GPUDevice, T>::Compute(d, logits, softmax);
  }
};
}  // end namespace functor

// Instantiate the GPU implementation for float.
template struct functor::SoftmaxFunctor<GPUDevice, float>;

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
