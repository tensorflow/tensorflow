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

#ifndef TENSORFLOW_CONTRIB_FUSED_CONV_KERNELS_FUSED_CONV_OPS_GPU_H_
#define TENSORFLOW_CONTRIB_FUSED_CONV_KERNELS_FUSED_CONV_OPS_GPU_H_

#if GOOGLE_CUDA

#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/util/activation_mode.h"

// TODO(pauldonnelly): Merge this file into core/kernels/conv_ops_gpu.h.

namespace tensorflow {

// Add additional parameters specific to fused convolutions.
class FusedConvParameters : public ConvParameters {
 public:
  FusedConvParameters(int64 batch, int64 in_depths, const SpatialArray& in,
                      TensorFormat data_format, int64 out_depths,
                      const SpatialArray& filter, const SpatialArray& dilation,
                      const SpatialArray& stride, const SpatialArray& padding,
                      DataType dtype, int device_id, bool has_side_input,
                      ActivationMode activation_mode)
      : ConvParameters(batch, in_depths, in, data_format, out_depths, filter,
                       dilation, stride, padding, dtype, device_id),
        activation_mode_(activation_mode),
        has_side_input_(has_side_input) {
    hash_code_ = Hash64Combine(hash_code_, has_side_input);
    hash_code_ = Hash64Combine(hash_code_, activation_mode);
  }

  bool operator==(const FusedConvParameters& other) const {
    return this->get_data_as_tuple() == other.get_data_as_tuple();
  }

  bool operator!=(const FusedConvParameters& other) const {
    return !(*this == other);
  }

  string ToString() const {
    return strings::StrCat(ConvParameters::ToString(), ", ", has_side_input_,
                           ", ", activation_mode_, ", ");
  }

 private:
  using ParameterDataType =
      std::tuple<ConvParameters::ParameterDataType, bool, ActivationMode>;

  ParameterDataType get_data_as_tuple() const {
    return std::make_tuple(ConvParameters::get_data_as_tuple(), has_side_input_,
                           activation_mode_);
  }

  ActivationMode activation_mode_;
  bool has_side_input_;
};

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CONTRIB_FUSED_CONV_KERNELS_FUSED_CONV_OPS_GPU_H_
