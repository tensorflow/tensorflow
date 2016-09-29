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

#ifndef TENSORFLOW_KERNELS_FUSED_BATCH_NORM_OP_H_
#define TENSORFLOW_KERNELS_FUSED_BATCH_NORM_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace functor {
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
}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_FUSED_BATCH_NORM_OP_H_
