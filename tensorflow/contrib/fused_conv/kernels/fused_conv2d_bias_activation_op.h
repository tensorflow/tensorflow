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

#ifndef TENSORFLOW_CONTRIB_FUSED_CONV_KERNELS_FUSED_CONV2D_BIAS_ACTIVATION_OP_H_
#define TENSORFLOW_CONTRIB_FUSED_CONV_KERNELS_FUSED_CONV2D_BIAS_ACTIVATION_OP_H_

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/util/activation_mode.h"
#include "tensorflow/core/util/tensor_format.h"

#if GOOGLE_CUDA
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/contrib/fused_conv/kernels/fused_conv_ops_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

// Forward declaration.
class OpKernelContext;

template <typename Device, typename T, typename BiasType, typename ScaleType>
class LaunchFusedConv2DBiasActivationOp {
 public:
  void launch(OpKernelContext* ctx, bool cudnn_use_autotune,
              const Tensor& conv_input, ScaleType conv_input_scale,
              const Tensor& filter, int32 row_stride, int32 col_stride,
              const Eigen::PaddingType& padding, const Tensor& side_input,
              ScaleType side_input_scale, const Tensor& bias,
              ActivationMode activation_mode, TensorFormat data_format,
              FilterTensorFormat filter_format, Tensor* output);
};

#ifdef GOOGLE_CUDA
template <typename T, typename BiasType, typename ScaleType>
class LaunchFusedConv2DBiasActivationOp<Eigen::GpuDevice, T, BiasType,
                                        ScaleType> {
 public:
  void launch(OpKernelContext* ctx, bool cudnn_use_autotune,
              const Tensor& conv_input, ScaleType conv_input_scale,
              const Tensor& filter, int32 row_stride, int32 col_stride,
              const Eigen::PaddingType& padding, const Tensor& side_input,
              ScaleType side_input_scale, const Tensor& bias,
              ActivationMode activation_mode, TensorFormat data_format,
              FilterTensorFormat filter_format, Tensor* output);
};
#endif  // GOOGLE_CUDA

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_FUSED_CONV_KERNELS_FUSED_CONV2D_BIAS_ACTIVATION_OP_H_
