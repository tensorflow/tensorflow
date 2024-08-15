/* Copyright 2024 The OpenXLA Authors.

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
#define EIGEN_USE_THREADS

#include "xla/service/cpu/runtime/conv_impl.h"

namespace tensorflow::xla {

// Instantiate Conv2D template for all supported devices and data types.
#define CONV2D_INSTANTIATE_TEMPLATE(EigenDevice, ScalarType)               \
  template void EigenConv2DImpl<EigenDevice, ScalarType>(                  \
      const EigenDevice& device, ScalarType* out, ScalarType* lhs,         \
      ScalarType* rhs, Eigen::Index input_batch, Eigen::Index input_x,     \
      Eigen::Index input_y, Eigen::Index input_channels,                   \
      Eigen::Index kernel_x, Eigen::Index kernel_y,                        \
      Eigen::Index kernel_channels, Eigen::Index kernel_filters,           \
      Eigen::Index output_x, Eigen::Index output_y, Eigen::Index x_stride, \
      Eigen::Index y_stride, Eigen::Index padding_x_before,                \
      Eigen::Index padding_x_after, Eigen::Index padding_y_before,         \
      Eigen::Index padding_y_after, Eigen::Index lhs_x_dilation,           \
      Eigen::Index lhs_y_dilation, Eigen::Index rhs_x_dilation,            \
      Eigen::Index rhs_y_dilation, Eigen::Index feature_group_count,       \
      std::optional<std::function<void()>> done_callback)

CONV2D_INSTANTIATE_TEMPLATE(Eigen::DefaultDevice, Eigen::half);
CONV2D_INSTANTIATE_TEMPLATE(Eigen::DefaultDevice, float);
CONV2D_INSTANTIATE_TEMPLATE(Eigen::ThreadPoolDevice, Eigen::half);
CONV2D_INSTANTIATE_TEMPLATE(Eigen::ThreadPoolDevice, float);

#undef CONV2D_INSTANTIATE_TEMPLATE

// Instantiate Conv3D template for all supported devices and data types.
#define CONV3D_INSTANTIATE_TEMPLATE(EigenDevice, ScalarType)                   \
  template void EigenConv3DImpl<EigenDevice, ScalarType>(                      \
      const EigenDevice& device, ScalarType* out, ScalarType* lhs,             \
      ScalarType* rhs, Eigen::Index input_batch, Eigen::Index input_x,         \
      Eigen::Index input_y, Eigen::Index input_z, Eigen::Index input_channels, \
      Eigen::Index kernel_x, Eigen::Index kernel_y, Eigen::Index kernel_z,     \
      Eigen::Index kernel_channels, Eigen::Index kernel_filters,               \
      Eigen::Index output_x, Eigen::Index output_y, Eigen::Index output_z,     \
      Eigen::Index x_stride, Eigen::Index y_stride, Eigen::Index z_stride,     \
      Eigen::Index padding_x_before, Eigen::Index padding_x_after,             \
      Eigen::Index padding_y_before, Eigen::Index padding_y_after,             \
      Eigen::Index padding_z_before, Eigen::Index padding_z_after,             \
      Eigen::Index lhs_x_dilation, Eigen::Index lhs_y_dilation,                \
      Eigen::Index lhs_z_dilation, Eigen::Index rhs_x_dilation,                \
      Eigen::Index rhs_y_dilation, Eigen::Index rhs_z_dilation,                \
      Eigen::Index feature_group_count,                                        \
      std::optional<std::function<void()>> done_callback)

CONV3D_INSTANTIATE_TEMPLATE(Eigen::DefaultDevice, Eigen::half);
CONV3D_INSTANTIATE_TEMPLATE(Eigen::DefaultDevice, float);
CONV3D_INSTANTIATE_TEMPLATE(Eigen::ThreadPoolDevice, Eigen::half);
CONV3D_INSTANTIATE_TEMPLATE(Eigen::ThreadPoolDevice, float);

}  // namespace tensorflow::xla
