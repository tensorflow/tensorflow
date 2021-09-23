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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/util/autotune_maps/conv_parameters.h"

#include <vector>

#include "absl/strings/str_format.h"
#include "tensorflow/core/platform/hash.h"
#include "tensorflow/core/util/autotune_maps/autotune_maps_utils.h"
#include "tensorflow/core/util/autotune_maps/conv_parameters.pb.h"

namespace tensorflow {

namespace {
using ::tensorflow::protobuf::util::MessageDifferencer;

uint64 ComputeHash(int device_id, const ConvParametersProto& proto) {
  return Hash64Combine(device_id, autotune_maps_utils::HashProto(proto));
}
}  // namespace

ConvParameters::ConvParameters(
    int64_t batch, int64_t in_depths, const absl::Span<const int64_t> in,
    int data_format, int64_t out_depths, const absl::Span<const int64_t> filter,
    const absl::Span<const int64_t> dilation,
    const absl::Span<const int64_t> stride,
    const absl::Span<const int64_t> padding, DataType dtype, int device_id,
    int group_count, absl::optional<ConvParameters::FusionInfo> fusion_info,
    int version)
    : device_id_(device_id) {
  proto_.set_batch(batch);
  proto_.set_in_depths(in_depths);
  *proto_.mutable_in() = {in.begin(), in.end()};
  proto_.set_data_format(static_cast<int>(data_format));
  proto_.set_out_depths(out_depths);
  *proto_.mutable_filter() = {filter.begin(), filter.end()};
  *proto_.mutable_dilation() = {dilation.begin(), dilation.end()};
  *proto_.mutable_stride() = {stride.begin(), stride.end()};
  *proto_.mutable_padding() = {padding.begin(), padding.end()};
  proto_.set_dtype(dtype);
  proto_.set_group_count(group_count);
  if (fusion_info.has_value()) {
    ConvParametersProto::Fusion fusion_proto;
    fusion_proto.set_conv_scale(fusion_info.value().conv_scale);
    fusion_proto.set_side_input_scale(fusion_info.value().side_input_scale);
    fusion_proto.set_activation_mode(fusion_info.value().activation_mode);
    fusion_proto.set_is_contrib(fusion_info.value().is_contrib);
    *proto_.mutable_fusion() = fusion_proto;
  }
  proto_.set_device_identifier(
      autotune_maps_utils::DeviceIdToIdentifier(device_id));
  proto_.set_version(version);
  hash_code_ = ComputeHash(device_id_, proto_);
}

ConvParameters::ConvParameters(int device_id, const ConvParametersProto& proto)
    : device_id_(device_id),
      proto_(proto),
      hash_code_(ComputeHash(device_id, proto_)) {}

bool ConvParameters::operator==(const ConvParameters& other) const {
  return device_id_ == other.device_id_ &&
         MessageDifferencer::Equals(this->proto_, other.proto_);
}

string ConvParameters::ToString() const { return proto_.DebugString(); }

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
