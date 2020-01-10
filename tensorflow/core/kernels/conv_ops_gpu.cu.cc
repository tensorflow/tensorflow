#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/conv_2d.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T>
struct SpatialConvolution<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::Tensor output,
                  typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 4>::ConstTensor filter, int stride,
                  const Eigen::PaddingType& padding) {
    // TODO(keveman): nvcc 6.5 crashes when 32 bit indexing is turned on. Enable
    // this when we move to cuda 7.0.
    // SpatialConvolutionFunc(d, To32Bit(output), To32Bit(input),
    // To32Bit(filter), stride, padding);

    SpatialConvolutionFunc(d, output, input, filter, stride, padding);
  }
};

template struct SpatialConvolution<GPUDevice, float>;

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
