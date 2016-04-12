#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/contrib/rnn/kernels/lstm_ops.h"

namespace tensorflow {

namespace functor {

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_GPU_SPECS(T)                                \
  template struct TensorMemZero<GPUDevice, T>;             \
  template struct TensorMemCopy<GPUDevice, T>;             \
  template struct LSTMCellBlockFprop<GPUDevice, T, true>;  \
  template struct LSTMCellBlockBprop<GPUDevice, T, true>;

DEFINE_GPU_SPECS(float);
DEFINE_GPU_SPECS(double);
#undef DEFINE_GPU_SPECS

}  // end namespace functor
}  // end namespace tensorflow
#endif  // GOOGLE_CUDA
