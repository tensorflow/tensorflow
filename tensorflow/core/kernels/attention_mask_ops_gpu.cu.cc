#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/attention_mask_ops.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

__global__
void ComputeMedianKernel(
    const int batch_size, const int dist_size, const float* input,
    long long* median) {
  const int b = blockIdx.x;
  if (b < batch_size) {
    input += b * dist_size;

    float sum = 0.0f;
    int median_idx = 0;
    for (; median_idx < dist_size; ++median_idx) {
      sum += input[median_idx];
      if (sum >= 0.5f) {
        break;
      }
    }

    median[b] = median_idx;
  }
}

namespace functor {
template <>
void ComputeMedian<GPUDevice>::Compute(
    const GPUDevice& d, typename TTypes<float>::ConstMatrix input,
    typename TTypes<int64>::Vec median) {
  const int64 batch_size = input.dimensions()[0];
  const int64 dist_size = input.dimensions()[1];

  ComputeMedianKernel<<<batch_size, 1, 0, d.stream()>>>(
      batch_size, dist_size, input.data(), median.data());
};
}  // end namespace functor

template struct functor::AttentionMask<GPUDevice>;
template struct functor::AttentionMaskMedian<GPUDevice>;
template struct functor::ComputeMedian<GPUDevice>;

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
