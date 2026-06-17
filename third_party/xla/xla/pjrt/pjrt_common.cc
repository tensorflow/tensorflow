/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/pjrt/pjrt_common.h"

#include <cstdint>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "xla/pjrt/proto/pjrt_value_type.pb.h"

template<typename T>
struct always_false : std::false_type {};

namespace xla {

xla::PjRtValueType PjRtValueTypeFromProto(
    const xla::PjRtValueTypeProto& value) {
  switch (value.value_case()) {
    case xla::PjRtValueTypeProto::kStringValue:
      return value.string_value();
    case xla::PjRtValueTypeProto::kBoolValue:
      return value.bool_value();
    case xla::PjRtValueTypeProto::kIntValue:
      return value.int_value();
    case xla::PjRtValueTypeProto::kIntVector:
      return std::vector<int64_t>(value.int_vector().values().begin(),
                                  value.int_vector().values().end());
    case xla::PjRtValueTypeProto::kFloatValue:
      return value.float_value();
    default:
      return std::string("");
  }
}

xla::PjRtValueTypeProto PjRtValueTypeToProto(const xla::PjRtValueType& value) {
  xla::PjRtValueTypeProto value_proto;
  std::visit(
      [&](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, float>) {
          value_proto.set_float_value(v);
        } else if constexpr (std::is_same_v<T, int64_t>) {
          value_proto.set_int_value(v);
        } else if constexpr (std::is_same_v<T, std::string>) {
          value_proto.set_string_value(v);
        } else if constexpr (std::is_same_v<T, bool>) {
          value_proto.set_bool_value(v);
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
          value_proto.mutable_int_vector()->mutable_values()->Reserve(v.size());
          value_proto.mutable_int_vector()->mutable_values()->Add(v.begin(),
                                                                  v.end());
        } else {
          // Note: code below should really be static_assert(false, ...), but
          // that is unfortunately not possible, as some compilers consider it
          // invalid code, see
          // https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2593r0.html.
          static_assert(always_false<T>::value,
                        "Unhandled type in PjRtValueType variant");
        }
      },
      value);
  return value_proto;
}

}  // namespace xla
