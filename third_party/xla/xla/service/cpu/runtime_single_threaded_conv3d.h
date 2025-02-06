/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_RUNTIME_SINGLE_THREADED_CONV3D_H_
#define XLA_SERVICE_CPU_RUNTIME_SINGLE_THREADED_CONV3D_H_

#include <stdint.h>

#include "Eigen/Core"

extern "C" {

extern void __xla_cpu_runtime_EigenSingleThreadedConv3DF16(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr,
    Eigen::half* out, Eigen::half* lhs, Eigen::half* rhs, int64_t input_batch,
    int64_t input_x, int64_t input_y, int64_t input_z, int64_t input_channels,
    int64_t kernel_x, int64_t kernel_y, int64_t kernel_z,
    int64_t kernel_channels, int64_t kernel_filters, int64_t output_x,
    int64_t output_y, int64_t output_z, int64_t x_stride, int64_t y_stride,
    int64_t z_stride, int64_t padding_x_before, int64_t padding_x_after,
    int64_t padding_y_before, int64_t padding_y_after, int64_t padding_z_before,
    int64_t padding_z_after, int64_t lhs_x_dilation, int64_t lhs_y_dilation,
    int64_t lhs_z_dilation, int64_t rhs_x_dilation, int64_t rhs_y_dilation,
    int64_t rhs_z_dilation, int64_t feature_group_count);

extern void __xla_cpu_runtime_EigenSingleThreadedConv3DF32(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr, float* out,
    float* lhs, float* rhs, int64_t input_batch, int64_t input_x,
    int64_t input_y, int64_t input_z, int64_t input_channels, int64_t kernel_x,
    int64_t kernel_y, int64_t kernel_z, int64_t kernel_channels,
    int64_t kernel_filters, int64_t output_x, int64_t output_y,
    int64_t output_z, int64_t x_stride, int64_t y_stride, int64_t z_stride,
    int64_t padding_x_before, int64_t padding_x_after, int64_t padding_y_before,
    int64_t padding_y_after, int64_t padding_z_before, int64_t padding_z_after,
    int64_t lhs_x_dilation, int64_t lhs_y_dilation, int64_t lhs_z_dilation,
    int64_t rhs_x_dilation, int64_t rhs_y_dilation, int64_t rhs_z_dilation,
    int64_t feature_group_count);

}  // extern "C"

#endif  // XLA_SERVICE_CPU_RUNTIME_SINGLE_THREADED_CONV3D_H_
