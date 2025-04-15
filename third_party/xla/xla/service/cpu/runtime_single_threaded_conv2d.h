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

#ifndef XLA_SERVICE_CPU_RUNTIME_SINGLE_THREADED_CONV2D_H_
#define XLA_SERVICE_CPU_RUNTIME_SINGLE_THREADED_CONV2D_H_

#include <stdint.h>

#include "Eigen/Core"

extern "C" {

extern void __xla_cpu_runtime_EigenSingleThreadedConv2DF16(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr,
    Eigen::half* out, Eigen::half* lhs, Eigen::half* rhs, int64_t input_batch,
    int64_t input_rows, int64_t input_cols, int64_t input_channels,
    int64_t kernel_rows, int64_t kernel_cols, int64_t kernel_channels,
    int64_t kernel_filters, int64_t output_rows, int64_t output_cols,
    int64_t row_stride, int64_t col_stride, int64_t padding_top,
    int64_t padding_bottom, int64_t padding_left, int64_t padding_right,
    int64_t lhs_row_dilation, int64_t lhs_col_dilation,
    int64_t rhs_row_dilation, int64_t rhs_col_dilation,
    int64_t feature_group_count);

extern void __xla_cpu_runtime_EigenSingleThreadedConv2DF32(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr, float* out,
    float* lhs, float* rhs, int64_t input_batch, int64_t input_rows,
    int64_t input_cols, int64_t input_channels, int64_t kernel_rows,
    int64_t kernel_cols, int64_t kernel_channels, int64_t kernel_filters,
    int64_t output_rows, int64_t output_cols, int64_t row_stride,
    int64_t col_stride, int64_t padding_top, int64_t padding_bottom,
    int64_t padding_left, int64_t padding_right, int64_t lhs_row_dilation,
    int64_t lhs_col_dilation, int64_t rhs_row_dilation,
    int64_t rhs_col_dilation, int64_t feature_group_count);

}  // extern "C"

#endif  // XLA_SERVICE_CPU_RUNTIME_SINGLE_THREADED_CONV2D_H_
