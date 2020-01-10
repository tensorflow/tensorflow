#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/matmul_op.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// Partial specialization MatMulTensorFunctor<Device=GPUDevice, T>
template <typename T>
struct MatMulFunctor<GPUDevice, T> {
  void operator()(
      const GPUDevice& d, typename MatMulTypes<T>::out_type out,
      typename MatMulTypes<T>::in_type in0,
      typename MatMulTypes<T>::in_type in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair) {
    MatMul<GPUDevice>(d, To32Bit(out), To32Bit(in0), To32Bit(in1), dim_pair);
  }
};

#define DEFINE(T) template struct MatMulFunctor<GPUDevice, T>;
DEFINE(float);
// DEFINE(double);  // Does not compile 1/2015.
#undef DEFINE

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
