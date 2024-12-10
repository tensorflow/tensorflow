// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/python/ifrt_proxy/common/types.h"

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/types.pb.h"

namespace xla {
namespace ifrt {
namespace proxy {

absl::StatusOr<xla::PjRtValueType> FromVariantProto(
    const proto::Variant& variant_proto) {
  switch (variant_proto.value_case()) {
    case proto::Variant::kStringValue:
      return variant_proto.string_value();
    case proto::Variant::kInt64Value:
      return variant_proto.int64_value();
    case proto::Variant::kInt64List: {
      const auto& values = variant_proto.int64_list().values();
      return std::vector<int64_t>(values.begin(), values.end());
    }
    case proto::Variant::kFloatValue:
      return variant_proto.float_value();
    default:
      return absl::UnimplementedError(absl::StrCat(
          "Unknown xla.ifrt.proto.Variant case: ", variant_proto.value_case()));
  }
}

absl::StatusOr<proto::Variant> ToVariantProto(const xla::PjRtValueType& value) {
  proto::Variant variant;
  if (auto* s = std::get_if<std::string>(&value)) {
    variant.set_string_value(*s);
  } else if (auto* i = std::get_if<int64_t>(&value)) {
    variant.set_int64_value(*i);
  } else if (auto* is = std::get_if<std::vector<int64_t>>(&value)) {
    for (const int64_t i : *is) {
      variant.mutable_int64_list()->add_values(i);
    }
  } else if (auto* f = std::get_if<float>(&value)) {
    variant.set_float_value(*f);
  } else {
    return absl::UnimplementedError("Unknown xla::PjRtValueType type");
  }
  return variant;
}

proto::ArrayCopySemantics ToArrayCopySemanticsProto(ArrayCopySemantics s) {
  switch (s) {
    case ArrayCopySemantics::kAlwaysCopy:
      return proto::ARRAY_COPY_SEMANTICS_ALWAYS_COPY;
    case ArrayCopySemantics::kDonateInput:
      return proto::ARRAY_COPY_SEMANTICS_DONATE_INPUT;
    case ArrayCopySemantics::kReuseInput:
      return proto::ARRAY_COPY_SEMANTICS_REUSE_INPUT;
  }
}

absl::StatusOr<ArrayCopySemantics> FromArrayCopySemanticsProto(
    proto::ArrayCopySemantics s) {
  switch (s) {
    case proto::ARRAY_COPY_SEMANTICS_ALWAYS_COPY:
      return ArrayCopySemantics::kAlwaysCopy;
    case proto::ARRAY_COPY_SEMANTICS_DONATE_INPUT:
      return ArrayCopySemantics::kDonateInput;
    case proto::ARRAY_COPY_SEMANTICS_REUSE_INPUT:
      return ArrayCopySemantics::kReuseInput;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unhandled proto-enum value ", s, ":",
                       proto::ArrayCopySemantics_Name(s)));
  }
}

proto::SingleDeviceShardSemantics ToSingleDeviceShardSemanticsProto(
    SingleDeviceShardSemantics s) {
  switch (s) {
    case SingleDeviceShardSemantics::kAddressableShards:
      return proto::SINGLE_DEVICE_SHARD_SEMANTICS_ADDRESSABLE_SHARDS;
    case SingleDeviceShardSemantics::kAllShards:
      return proto::SINGLE_DEVICE_SHARD_SEMANTICS_ALL_SHARDS;
  }
}

absl::StatusOr<SingleDeviceShardSemantics> FromSingleDeviceShardSemanticsProto(
    proto::SingleDeviceShardSemantics s) {
  switch (s) {
    case proto::SINGLE_DEVICE_SHARD_SEMANTICS_ADDRESSABLE_SHARDS:
      return SingleDeviceShardSemantics::kAddressableShards;
    case proto::SINGLE_DEVICE_SHARD_SEMANTICS_ALL_SHARDS:
      return SingleDeviceShardSemantics::kAllShards;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unhandled proto-enum value ", s, ":",
                       proto::SingleDeviceShardSemantics_Name(s)));
  }
}

std::vector<int64_t> FromByteStridesProto(const proto::ByteStrides& strides) {
  std::vector<int64_t> result;
  result.reserve(strides.strides_size());
  for (auto x : strides.strides()) {
    result.push_back(x);
  }
  return result;
}

proto::ByteStrides ToByteStridesProto(const absl::Span<const int64_t> strides) {
  proto::ByteStrides result;
  for (auto x : strides) {
    result.add_strides(x);
  }
  return result;
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
