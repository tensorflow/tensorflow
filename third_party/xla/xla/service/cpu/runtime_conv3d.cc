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

#include "xla/service/cpu/runtime_conv3d.h"

#define EIGEN_USE_THREADS

#include "absl/base/dynamic_annotations.h"
#include "xla/executable_run_options.h"
#include "xla/service/cpu/runtime_conv_impl.h"
#include "xla/service/cpu/runtime_lightweight_check.h"

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_EigenConv3DF32(
    const void* run_options_ptr, float* out, float* lhs, float* rhs,
    int64_t input_batch, int64_t input_x, int64_t input_y, int64_t input_z,
    int64_t input_channels, int64_t kernel_x, int64_t kernel_y,
    int64_t kernel_z, int64_t kernel_channels, int64_t kernel_filters,
    int64_t output_x, int64_t output_y, int64_t output_z, int64_t x_stride,
    int64_t y_stride, int64_t z_stride, int64_t padding_x_before,
    int64_t padding_x_after, int64_t padding_y_before, int64_t padding_y_after,
    int64_t padding_z_before, int64_t padding_z_after, int64_t lhs_x_dilation,
    int64_t lhs_y_dilation, int64_t lhs_z_dilation, int64_t rhs_x_dilation,
    int64_t rhs_y_dilation, int64_t rhs_z_dilation,
    int64_t feature_group_count) {
  const xla::ExecutableRunOptions* run_options =
      static_cast<const xla::ExecutableRunOptions*>(run_options_ptr);
  XLA_LIGHTWEIGHT_CHECK(run_options->intra_op_thread_pool() != nullptr);
  tensorflow::xla::EigenConv3DImpl(
      *run_options->intra_op_thread_pool(), out, lhs, rhs, input_batch, input_x,
      input_y, input_z, input_channels, kernel_x, kernel_y, kernel_z,
      kernel_channels, kernel_filters, output_x, output_y, output_z, x_stride,
      y_stride, z_stride, padding_x_before, padding_x_after, padding_y_before,
      padding_y_after, padding_z_before, padding_z_after, lhs_x_dilation,
      lhs_y_dilation, lhs_z_dilation, rhs_x_dilation, rhs_y_dilation,
      rhs_z_dilation, feature_group_count);
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_EigenConv3DF16(
    const void* run_options_ptr, Eigen::half* out, Eigen::half* lhs,
    Eigen::half* rhs, int64_t input_batch, int64_t input_x, int64_t input_y,
    int64_t input_z, int64_t input_channels, int64_t kernel_x, int64_t kernel_y,
    int64_t kernel_z, int64_t kernel_channels, int64_t kernel_filters,
    int64_t output_x, int64_t output_y, int64_t output_z, int64_t x_stride,
    int64_t y_stride, int64_t z_stride, int64_t padding_x_before,
    int64_t padding_x_after, int64_t padding_y_before, int64_t padding_y_after,
    int64_t padding_z_before, int64_t padding_z_after, int64_t lhs_x_dilation,
    int64_t lhs_y_dilation, int64_t lhs_z_dilation, int64_t rhs_x_dilation,
    int64_t rhs_y_dilation, int64_t rhs_z_dilation,
    int64_t feature_group_count) {
  const xla::ExecutableRunOptions* run_options =
      static_cast<const xla::ExecutableRunOptions*>(run_options_ptr);
  XLA_LIGHTWEIGHT_CHECK(run_options->intra_op_thread_pool() != nullptr);
  tensorflow::xla::EigenConv3DImpl(
      *run_options->intra_op_thread_pool(), out, lhs, rhs, input_batch, input_x,
      input_y, input_z, input_channels, kernel_x, kernel_y, kernel_z,
      kernel_channels, kernel_filters, output_x, output_y, output_z, x_stride,
      y_stride, z_stride, padding_x_before, padding_x_after, padding_y_before,
      padding_y_after, padding_z_before, padding_z_after, lhs_x_dilation,
      lhs_y_dilation, lhs_z_dilation, rhs_x_dilation, rhs_y_dilation,
      rhs_z_dilation, feature_group_count);
}
