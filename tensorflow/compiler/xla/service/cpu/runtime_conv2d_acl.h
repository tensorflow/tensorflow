/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_CONV2D_ACL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_CONV2D_ACL_H_

#include "tensorflow/tsl/platform/types.h"

#ifdef XLA_CPU_USE_ACL
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "utils/Utils.h"

extern "C" {
struct acl_depthwise_conv_obj_t {
  arm_compute::NEDepthwiseConvolutionLayer depthwise_conv;
  arm_compute::NEArithmeticAddition add;
  arm_compute::NEActivationLayer act;
  arm_compute::Tensor input_tensor;
  arm_compute::Tensor kernel_tensor;
  arm_compute::Tensor bia_tensor;
  arm_compute::Tensor output_tensor;
  arm_compute::Tensor output_acc_tensor;
};

struct acl_gemm_conv_obj_t {
  arm_compute::NEGEMMConvolutionLayer gemm_conv;
  arm_compute::NEArithmeticAddition add;
  arm_compute::NEActivationLayer act;
  arm_compute::Tensor input_tensor;
  arm_compute::Tensor kernel_tensor;
  arm_compute::Tensor bia_tensor;
  arm_compute::Tensor output_tensor;
  arm_compute::Tensor output_acc_tensor;
};

struct acl_conv_conf_t {
  bool with_bias;
  bool is_int8;
  bool sum_with_eltwise;
  bool fast_math;
  arm_compute::TensorInfo input_info;
  arm_compute::TensorInfo kernel_info;
  arm_compute::TensorInfo bia_info;
  arm_compute::TensorInfo output_info;
  arm_compute::PadStrideInfo padstride_info;
  arm_compute::Size2D dilation_info;
  arm_compute::WeightsInfo kernel_wei_info;
  arm_compute::ActivationLayerInfo act_info;
};

extern void __xla_cpu_runtime_ACLConv2DF32(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr, float* out,
    float* lhs, float* rhs, int64_t input_batch, int64_t input_rows,
    int64_t input_cols, int64_t input_channels, int64_t kernel_rows,
    int64_t kernel_cols, int64_t kernel_channels, int64_t kernel_filters,
    int64_t output_rows, int64_t output_cols, int64_t row_stride,
    int64_t col_stride, int64_t padding_top, int64_t padding_bottom,
    int64_t padding_left, int64_t padding_right, int64_t lhs_row_dilation,
    int64_t lhs_col_dilation, int64_t rhs_row_dilation,
    int64_t rhs_col_dilation, int64_t feature_group_count);
}
#else
#include <iostream>

extern "C" {
inline extern void __xla_cpu_runtime_ACLConv2DF32(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr, float* out,
    float* lhs, float* rhs, int64_t input_batch, int64_t input_rows,
    int64_t input_cols, int64_t input_channels, int64_t kernel_rows,
    int64_t kernel_cols, int64_t kernel_channels, int64_t kernel_filters,
    int64_t output_rows, int64_t output_cols, int64_t row_stride,
    int64_t col_stride, int64_t padding_top, int64_t padding_bottom,
    int64_t padding_left, int64_t padding_right, int64_t lhs_row_dilation,
    int64_t lhs_col_dilation, int64_t rhs_row_dilation,
    int64_t rhs_col_dilation, int64_t feature_group_count) {
  std::cerr
      << "Attempt to call ACL Conv2D runtime library without defining "
         "XLA_CPU_USE_ACL. Add --define=build_with_acl=true to build with ACL.";
  exit(1);
}
}
#endif  // XLA_CPU_USE_ACL
#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_CONV2D_ACL_H_
