#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/reverse_sequence_op.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_GPU_SPEC(T, dims)                       \
  template class generator::ReverseGenerator<T, dims>; \
  template struct functor::ReverseSequence<GPUDevice, T, dims>;

#define DEFINE_GPU_SPECS(T) \
  DEFINE_GPU_SPEC(T, 2);    \
  DEFINE_GPU_SPEC(T, 3);    \
  DEFINE_GPU_SPEC(T, 4);    \
  DEFINE_GPU_SPEC(T, 5);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
