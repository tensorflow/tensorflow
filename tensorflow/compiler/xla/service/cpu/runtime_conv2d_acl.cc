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

#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tensorflow/core/platform/types.h"
#ifdef XLA_CPU_USE_ACL
#include "tensorflow/compiler/xla/service/cpu/runtime_conv2d.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_conv2d_acl.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_lightweight_check.h"
#include "tensorflow/core/platform/logging.h"
#include <mutex>
#define EIGEN_USE_THREADS
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace {
int32_t ACLConvImpl(const void* run_options_ptr, float* out, float* lhs,
                 float* rhs, int64_t input_batch, int64_t input_rows,
                 int64_t input_cols, int64_t input_channels,
                 int64_t kernel_rows, int64_t kernel_cols,
                 int64_t kernel_channels, int64_t kernel_filters,
                 int64_t output_rows, int64_t output_cols, int64_t row_stride,
                 int64_t col_stride, int64_t padding_top,
                 int64_t padding_bottom, int64_t padding_left,
                 int64_t padding_right, int64_t lhs_row_dilation,
                 int64_t lhs_col_dilation, int64_t rhs_row_dilation,
                 int64_t rhs_col_dilation) {
  /* TODO: optimize this object creation along with tensor init and
   * gemm configuration by caching the shapes, similar to onednn
   * primitive caching feature
   */
  struct acl_conv_obj_t acl_conv_obj;
  struct acl_conv_conf_t acl_conf;

  /* TODO: add TF_XLA_* flag for runtime control of fast math mode
   */
  acl_conf.fast_math = true;

  /* ir_emitter HandleConvolution ensures the below preconditions before dispatching it to ACL
   *  layout: NHWC
   *  format: FP32
   *  Number of feature groups: 1
   *  source and kernel dilation is: 1
   */
  acl_conf.dilation_info = arm_compute::Size2D(lhs_col_dilation, lhs_row_dilation);
  acl_conf.padstride_info = arm_compute::PadStrideInfo(col_stride, row_stride,
                                                       static_cast<unsigned int>(padding_left),
                                                       static_cast<unsigned int>(padding_right),
                                                       static_cast<unsigned int>(padding_top),
                                                       static_cast<unsigned int>(padding_bottom),
                                                       arm_compute::DimensionRoundingType::FLOOR);
  acl_conf.with_bias = false;

  acl_conf.input_info = arm_compute::TensorInfo(arm_compute::TensorShape(input_channels, input_cols,
                                                                         input_rows, input_batch),
                                                1, arm_compute::DataType::F32, arm_compute::DataLayout::NHWC);
  acl_conf.kernel_info = arm_compute::TensorInfo(arm_compute::TensorShape(input_channels, kernel_cols,
                                                                          kernel_rows, kernel_filters),
                                                 1, arm_compute::DataType::F32, arm_compute::DataLayout::NHWC);
  acl_conf.output_info = arm_compute::TensorInfo(arm_compute::TensorShape(kernel_filters, output_cols,
                                                                          output_rows, input_batch),
                                                 1, arm_compute::DataType::F32, arm_compute::DataLayout::NHWC);
  acl_conf.act_info = arm_compute::ActivationLayerInfo();

  // Validate convolution manually to check for return status
  auto acl_st = arm_compute::NEGEMMConvolutionLayer::validate(&acl_conf.input_info,
                                                              &acl_conf.kernel_info,
                                                              /*acp.with_bias */ nullptr,
                                                              &acl_conf.output_info,
                                                              acl_conf.padstride_info,
                                                              acl_conf.kernel_wei_info,
                                                              acl_conf.dilation_info,
                                                              acl_conf.act_info,
                                                              acl_conf.fast_math);
  if (acl_st.error_code() != arm_compute::ErrorCode::OK) {
     VLOG(1) << " Gemm conv validation failed";
     return -1;
  }

  static std::once_flag flag_once;
  const xla::ExecutableRunOptions* run_options = static_cast<const xla::ExecutableRunOptions*>(run_options_ptr);
  XLA_LIGHTWEIGHT_CHECK(run_options->intra_op_thread_pool() != nullptr);
  const Eigen::ThreadPoolDevice* tpd = (Eigen::ThreadPoolDevice*) (run_options->intra_op_thread_pool());
  // The threads in Compute Library are bound for the cores 0..max_threads-1
  const int max_threads = tpd->numThreads();

  // arm_compute::Scheduler does not support concurrent access thus a
  // workaround here restricts it to only one call
  std::call_once(flag_once, [&]() {
     arm_compute::Scheduler::get().set_num_threads(max_threads);
  });

  //configure the acl obj with the config
  acl_conv_obj.input_tensor.allocator()->init(acl_conf.input_info);
  acl_conv_obj.kernel_tensor.allocator()->init(acl_conf.kernel_info);
  acl_conv_obj.output_tensor.allocator()->init(acl_conf.output_info);

  // Configure GEMM
  acl_conv_obj.conv.configure(&acl_conv_obj.input_tensor,
                              &acl_conv_obj.kernel_tensor,
                              nullptr,
                              &acl_conv_obj.output_tensor,
                              acl_conf.padstride_info,
                              acl_conf.kernel_wei_info,
                              acl_conf.dilation_info,
                              acl_conf.act_info,
                              acl_conf.fast_math);

  /* import_memory() and free() methods do not allocate/free any additional
   * memory, only acquire/release pointers.
   */
  acl_conv_obj.input_tensor.allocator()->import_memory(lhs);
  acl_conv_obj.kernel_tensor.allocator()->import_memory(rhs);
  acl_conv_obj.output_tensor.allocator()->import_memory(out);

  acl_conv_obj.conv.run();

  acl_conv_obj.input_tensor.allocator()->free();
  acl_conv_obj.kernel_tensor.allocator()->free();
  acl_conv_obj.output_tensor.allocator()->free();

  return 0;
  }
}  // namespace

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_ACLConv2DF32(
    const void* run_options_ptr, float* out, float* lhs, float* rhs,
    int64_t input_batch, int64_t input_rows, int64_t input_cols,
    int64_t input_channels, int64_t kernel_rows, int64_t kernel_cols,
    int64_t kernel_channels, int64_t kernel_filters, int64_t output_rows,
    int64_t output_cols, int64_t row_stride, int64_t col_stride,
    int64_t padding_top, int64_t padding_bottom, int64_t padding_left,
    int64_t padding_right, int64_t lhs_row_dilation, int64_t lhs_col_dilation,
    int64_t rhs_row_dilation, int64_t rhs_col_dilation, int64_t feature_group_count) {
  if (lhs_row_dilation > 1 || lhs_col_dilation > 1 ||
      (ACLConvImpl(run_options_ptr, out, lhs, rhs, input_batch, input_rows, input_cols,
                  input_channels, kernel_rows, kernel_cols, kernel_channels,
                  kernel_filters, output_rows, output_cols, row_stride,
                  col_stride, padding_top, padding_bottom, padding_left,
                  padding_right, lhs_row_dilation, lhs_col_dilation,
                  rhs_row_dilation, rhs_col_dilation) < 0)) {
     VLOG(1) << "XLA conv2d not supported by ACL, fallback to Eigen runtime";
     __xla_cpu_runtime_EigenConv2DF32(run_options_ptr, out, lhs, rhs, input_batch,
                                    input_rows, input_cols, input_channels, kernel_rows,
                                    kernel_cols, kernel_channels,kernel_filters,
                                    output_rows, output_cols, row_stride, col_stride,
                                    padding_top, padding_bottom, padding_left, padding_right,
                                    lhs_row_dilation, lhs_col_dilation, rhs_row_dilation,
				    rhs_col_dilation, feature_group_count);
  }
}
#endif  // XLA_CPU_USE_ACL
