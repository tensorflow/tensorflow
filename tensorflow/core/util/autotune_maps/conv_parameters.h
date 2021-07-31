/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_AUTOTUNE_MAPS_CONV_PARAMETERS_H_
#define TENSORFLOW_CORE_UTIL_AUTOTUNE_MAPS_CONV_PARAMETERS_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/autotune_maps/conv_parameters.pb.h"

namespace tensorflow {
// Uniquely identifies a convolution operation that runs on a particular device
// model.
//
// This can serve as a hashtable key, where the value might be the autotuned
// algorithm we choose for the conv.
//
// All of the data in this class is stored in the ConvParametersProto, so it can
// be easily serialized (for the purposes of ahead-of-time autotuning).
//
// Note: The device_id (in the dense range 0, 1, ...) passed to the constructor
// is translated into a "device model" string that's stored in the proto.  This
// means that if GPUs X and Y are of the same model, two otherwise identical
// ConvParameters with different device_ids X and Y will compare equal.
class ConvParameters {
 public:
  ConvParameters(int64_t batch, int64_t in_depths, absl::Span<const int64_t> in,
                 int data_format, int64_t out_depths,
                 absl::Span<const int64_t> filter,
                 absl::Span<const int64_t> dilation,
                 absl::Span<const int64_t> stride,
                 absl::Span<const int64_t> padding, DataType dtype,
                 int device_id, int group_count = 1,
                 bool has_side_input = false,
                 stream_executor::dnn::ActivationMode activation_mode =
                     stream_executor::dnn::ActivationMode::kNone);

  explicit ConvParameters(const ConvParametersProto& proto);

  bool operator==(const ConvParameters& other) const;

  bool operator!=(const ConvParameters& other) const {
    return !(*this == other);
  }
  uint64 hash() const { return hash_code_; }

  string ToString() const;

  const ConvParametersProto& proto() const { return proto_; }

 private:
  uint64 hash_code_;
  ConvParametersProto proto_;
};
}  // namespace tensorflow
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_UTIL_AUTOTUNE_MAPS_CONV_PARAMETERS_H_
