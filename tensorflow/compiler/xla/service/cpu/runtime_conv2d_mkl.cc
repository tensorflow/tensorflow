/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/service/cpu/runtime_conv2d_mkl.h"

#include <iostream>

#include "absl/base/dynamic_annotations.h"
#include "tensorflow/compiler/xla/executable_run_options.h"

#ifdef ENABLE_MKL
#include <omp.h>

#include "dnnl.hpp"
#include "tensorflow/compiler/xla/service/cpu/runtime_conv2d.h"

namespace {

// Downcast an int64_t to int and check if value is in range.
int ToInt(int64_t input) {
  int output = static_cast<int>(input);
  if (static_cast<int64_t>(output) != input) {
    std::cerr << "Error occurred in downcasting int64_t to int32_t: Value "
              << input << " is out-of-range for type int32_t. \n";
    exit(1);
  }
  return output;
}

using dnnl::convolution_direct;
using dnnl::convolution_forward;
using dnnl::engine;
using dnnl::memory;
using dnnl::padding_kind;
using dnnl::primitive;
using dnnl::prop_kind;
using dnnl::reorder;
using dnnl::stream;

template <typename EigenDevice, typename ScalarType>
void MKLConvImpl(const EigenDevice& device, ScalarType* out, ScalarType* lhs,
                 ScalarType* rhs, int64_t input_batch, int64_t input_rows,
                 int64_t input_cols, int64_t input_channels,
                 int64_t kernel_rows, int64_t kernel_cols,
                 int64_t kernel_channels, int64_t kernel_filters,
                 int64_t output_rows, int64_t output_cols, int64_t row_stride,
                 int64_t col_stride, int64_t padding_top,
                 int64_t padding_bottom, int64_t padding_left,
                 int64_t padding_right, int64_t lhs_row_dilation,
                 int64_t lhs_col_dilation, int64_t rhs_row_dilation,
                 int64_t rhs_col_dilation) {
  auto cpu_engine = engine(engine::cpu, 0);

  // Create a vector primitive to hold the network.
  std::vector<primitive> net;

  // Since memory::dims takes int for each dimension, we downcast the int64_t
  // values to int using the ToInt function defined above.
  memory::dims conv1_src_dim = {ToInt(input_batch), ToInt(input_channels),
                                ToInt(input_rows), ToInt(input_cols)};
  memory::dims conv1_weights_dim = {ToInt(kernel_filters),
                                    ToInt(kernel_channels), ToInt(kernel_rows),
                                    ToInt(kernel_cols)};
  memory::dims conv1_dst_dim = {ToInt(input_batch), ToInt(kernel_filters),
                                ToInt(output_rows), ToInt(output_cols)};
  memory::dims conv1_strides = {ToInt(row_stride), ToInt(col_stride)};
  // Note: In MKL_DNN dilation starts from 0.
  memory::dims conv1_dilates = {ToInt(rhs_row_dilation - 1),
                                ToInt(rhs_col_dilation - 1)};
  memory::dims conv1_padding_l = {ToInt(padding_top), ToInt(padding_left)};
  memory::dims conv1_padding_r = {ToInt(padding_bottom), ToInt(padding_right)};

  // Create memory for user data. Input and output data have format of NHWC and
  // kernel data has format of HWIO.
  // Note that as a convention in MKL-DNN, the dimensions of the data is always
  // described in NCHW/IOHW, regardless of the actual layout of the data.
  auto user_src_memory =
      memory({{{conv1_src_dim}, memory::data_type::f32, memory::format::nhwc},
              cpu_engine},
             lhs);
  auto user_weights_memory = memory(
      {{{conv1_weights_dim}, memory::data_type::f32, memory::format::hwio},
       cpu_engine},
      rhs);
  auto user_dst_memory =
      memory({{{conv1_dst_dim}, memory::data_type::f32, memory::format::nhwc},
              cpu_engine},
             out);

  // Create memory descriptors for convolution data with no specified format for
  // best performance.
  auto conv1_src_mem_desc = memory::desc(
      {conv1_src_dim}, memory::data_type::f32, memory::format::any);
  auto conv1_weights_mem_desc = memory::desc(
      {conv1_weights_dim}, memory::data_type::f32, memory::format::any);
  auto conv1_dst_mem_desc = memory::desc(
      {conv1_dst_dim}, memory::data_type::f32, memory::format::any);

  // Create a convolution.
  auto conv1_desc = convolution_forward::desc(
      prop_kind::forward_inference, convolution_direct, conv1_src_mem_desc,
      conv1_weights_mem_desc, conv1_dst_mem_desc, conv1_strides, conv1_dilates,
      conv1_padding_l, conv1_padding_r, padding_kind::zero);
  auto conv1_prim_desc =
      convolution_forward::primitive_desc(conv1_desc, cpu_engine);

  // Create reorders for data and weights if layout requested by convolution is
  // different from NCHW/OIHW.
  auto conv1_src_memory = user_src_memory;
  if (memory::primitive_desc(conv1_prim_desc.src_primitive_desc()) !=
      user_src_memory.get_primitive_desc()) {
    conv1_src_memory = memory(conv1_prim_desc.src_primitive_desc());
    net.push_back(reorder(user_src_memory, conv1_src_memory));
  }

  auto conv1_weights_memory = user_weights_memory;
  if (memory::primitive_desc(conv1_prim_desc.weights_primitive_desc()) !=
      user_weights_memory.get_primitive_desc()) {
    conv1_weights_memory = memory(conv1_prim_desc.weights_primitive_desc());
    net.push_back(reorder(user_weights_memory, conv1_weights_memory));
  }

  // Check if output need layout conversion. If yes, create memory for
  // intermediate layer of conv1_dst_memory.
  bool need_output_conversion =
      (memory::primitive_desc(conv1_prim_desc.dst_primitive_desc()) !=
       user_dst_memory.get_primitive_desc());
  auto conv1_dst_memory = need_output_conversion
                              ? memory(conv1_prim_desc.dst_primitive_desc())
                              : user_dst_memory;

  // Create convolution primitive and add it to net.
  net.push_back(convolution_forward(conv1_prim_desc, conv1_src_memory,
                                    conv1_weights_memory, conv1_dst_memory));
  if (need_output_conversion) {
    net.push_back(reorder(conv1_dst_memory, user_dst_memory));
  }
  stream(stream::kind::eager_nostore).submit(net).wait();
}
}  // namespace
#endif  // ENABLE_MKL

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_MKLConv2DF32(
    const void* run_options_ptr, float* out, float* lhs, float* rhs,
    int64_t input_batch, int64_t input_rows, int64_t input_cols,
    int64_t input_channels, int64_t kernel_rows, int64_t kernel_cols,
    int64_t kernel_channels, int64_t kernel_filters, int64_t output_rows,
    int64_t output_cols, int64_t row_stride, int64_t col_stride,
    int64_t padding_top, int64_t padding_bottom, int64_t padding_left,
    int64_t padding_right, int64_t lhs_row_dilation, int64_t lhs_col_dilation,
    int64_t rhs_row_dilation, int64_t rhs_col_dilation) {
#ifdef ENABLE_MKL
  // Since MKL_DNN cannot handle transposed convolution, this is handled by
  // Eigen.
  if (lhs_row_dilation > 1 || lhs_col_dilation > 1) {
    __xla_cpu_runtime_EigenConvF32(
        run_options_ptr, out, lhs, rhs, input_batch, input_rows, input_cols,
        input_channels, kernel_rows, kernel_cols, kernel_channels,
        kernel_filters, output_rows, output_cols, row_stride, col_stride,
        padding_top, padding_bottom, padding_left, padding_right,
        lhs_row_dilation, lhs_col_dilation, rhs_row_dilation, rhs_col_dilation);
  } else {
    MKLConvImpl(nullptr, out, lhs, rhs, input_batch, input_rows, input_cols,
                input_channels, kernel_rows, kernel_cols, kernel_channels,
                kernel_filters, output_rows, output_cols, row_stride,
                col_stride, padding_top, padding_bottom, padding_left,
                padding_right, lhs_row_dilation, lhs_col_dilation,
                rhs_row_dilation, rhs_col_dilation);
  }
#else
  std::cerr << "Attempt to call MKL Conv2D runtime library without defining "
               "ENABLE_MKL. Add --config=mkl to build with MKL.";
  exit(1);
#endif  // ENABLE_MKL
}
