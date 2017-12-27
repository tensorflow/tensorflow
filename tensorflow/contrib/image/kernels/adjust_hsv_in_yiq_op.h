/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_IMAGE_KERNELS_ADJUST_HSV_IN_YIQ_OP_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_IMAGE_KERNELS_ADJUST_HSV_IN_YIQ_OP_H_

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include <cmath>
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

static constexpr int kChannelSize = 3;

namespace internal {

template <int MATRIX_SIZE>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void compute_tranformation_matrix(
    const float delta_h, const float scale_s, const float scale_v,
    float* matrix) {
  static_assert(MATRIX_SIZE == kChannelSize * kChannelSize,
                "Size of matrix should be 9.");
  // Projection matrix from RGB to YIQ. Numbers from wikipedia
  // https://en.wikipedia.org/wiki/YIQ
  Eigen::Matrix3f yiq;
  /* clang-format off */
  yiq << 0.299, 0.587, 0.114,
         0.596, -0.274, -0.322,
         0.211, -0.523, 0.312;
  Eigen::Matrix3f yiq_inverse;
  yiq_inverse << 1, 0.95617069, 0.62143257,
                 1, -0.2726886, -0.64681324,
                 1, -1.103744, 1.70062309;
  /* clang-format on */
  // Construct hsv linear transformation matrix in YIQ space.
  // https://beesbuzz.biz/code/hsv_color_transforms.php
  float vsu = scale_v * scale_s * std::cos(delta_h);
  float vsw = scale_v * scale_s * std::sin(delta_h);
  Eigen::Matrix3f hsv_transform;
  /* clang-format off */
  hsv_transform << scale_v, 0, 0,
                   0, vsu, -vsw,
                   0, vsw, vsu;
  /* clang-format on */
  // Compute final transformation matrix = inverse_yiq * hsv_transform * yiq
  Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::ColMajor>> eigen_matrix(matrix);
  eigen_matrix = yiq_inverse * hsv_transform * yiq;
}
}  // namespace internal

#if GOOGLE_CUDA
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

struct AdjustHsvInYiqGPU {
  void operator()(OpKernelContext* ctx, int channel_count,
                  const Tensor* const input, const float* const delta_h,
                  const float* const scale_s, const float* const scale_v,
                  Tensor* const output);
};

}  // namespace functor

#endif  // GOOGLE_CUDA

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_IMAGE_KERNELS_ADJUST_HSV_IN_YIQ_OP_H_
