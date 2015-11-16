#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_MAXPOOLING_OP_GPU_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_MAXPOOLING_OP_GPU_H_

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/NeuralNetworks"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

// Run the forward pass of max pooling, optionally writing the argmax indices to
// the mask array, if it is not nullptr. If mask is passed in as nullptr, the
// argmax indices are not written.
bool MaxPoolForwardWithOptionalArgmax(
    const float* bottom_data, const int batch, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_t, const int pad_l,
    float* top_data, int64* mask, const Eigen::GpuDevice& d);

bool MaxPoolBackwardWithArgmax(const int output_size, const int input_size,
                               const float* top_diff, const int64* mask,
                               const int top_offset, const int bottom_offset,
                               float* bottom_diff, const Eigen::GpuDevice& d);

bool MaxPoolBackwardNoMask(const float* bottom_data, const int batch,
                           const int height, const int width,
                           const int channels, const int pooled_height,
                           const int pooled_width, const int kernel_h,
                           const int kernel_w, const int stride_h,
                           const int stride_w, const int pad_t, const int pad_l,
                           const float* top_diff, float* bottom_diff,
                           const Eigen::GpuDevice& d);

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_MAXPOOLING_OP_GPU_H_
