/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_FUSED_BATCH_NORM_OP_H_
#define TENSORFLOW_CORE_KERNELS_FUSED_BATCH_NORM_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace functor {

// FusedBatchNormEx op supports side inputs and activations:
//   (1) batch_norm + activation
//   (2) batch norm + side input + activation
enum class FusedBatchNormActivationMode { kIdentity, kRelu };

string ToString(FusedBatchNormActivationMode activation_mode);

Status ParseActivationMode(OpKernelConstruction* context,
                           FusedBatchNormActivationMode* activation_mode);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// There is a behavior difference between cuDNN v4 and v5 with regard to the
// scaling factor for function cudnnBatchNormalizationForwardInference.
// This function corrects the scaling factor if cuDNN v4 is used, so that
// this behavior inconsistency is hidden from TensorFlow users.
// Details: in cuDNN v4, y = bnScale * (x - mean) * variance + bnBias;
// in v5, y = bnScale * (x - mean) / sqrt(variance + epsilon) + bnBias
// The template is instantiated with T as float in batch_norm_ops.cu.cc; for
// other types, the instantiation needs to be added accordingly.
template <class T>
struct VarianceToInvVariance {
  void operator()(const Eigen::GpuDevice& d, const T* variance, double epsilon,
                  int channels, T* inv_variance);
};

// This function converts the inverted variance of the cuDNN forward training
// output to variance for TensorFlow to calculate the running variance.
// The template is instantiated with T as float in batch_norm_ops.cu.cc; for
// other types, the instantiation needs to be added accordingly.
template <class T>
struct InvVarianceToVariance {
  void operator()(const Eigen::GpuDevice& d, double epsilon, int sample_size,
                  int channels, T* variance);
};

// This function sets a GPU tensor to NaNs.
template <class T>
struct SetNanFunctor {
  void operator()(const Eigen::GpuDevice& d, typename TTypes<T>::Flat out);
};

// This is a functor to launch custom CUDA kernel for FusedBatchNorm with side
// input and activation when 'is_training=False'. In training we rely on cuDNN.
template <typename Device, typename T, typename U>
struct FusedBatchNormInferenceFunctor {
  void operator()(OpKernelContext* context, TensorFormat tensor_format,
                  typename TTypes<T, 4>::ConstTensor in,
                  typename TTypes<U>::ConstVec scale,
                  typename TTypes<U>::ConstVec offset,
                  typename TTypes<U>::ConstVec estimated_mean,
                  typename TTypes<U>::ConstVec estimated_variance,
                  typename TTypes<T, 4>::ConstTensor side_input, U epsilon,
                  FusedBatchNormActivationMode activation_mode,
                  typename TTypes<T, 4>::Tensor out);
};

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Functor used by FusedBatchNormGradOp to do the computations when
// is_training=False.
template <typename Device, typename T, typename U>
struct FusedBatchNormFreezeGrad {
  void operator()(OpKernelContext* context, const Tensor& y_backprop_input,
                  const Tensor& x_input, const Tensor& scale_input,
                  const Tensor& pop_mean_input,
                  const Tensor& pop_variance_input, U epsilon,
                  Tensor* x_backprop_output, Tensor* scale_backprop_output,
                  Tensor* offset_backprop_output) {}
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_FUSED_BATCH_NORM_OP_H_
