/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/python/ifrt/attribute_map.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/python/ifrt/attribute_map.pb.h"

namespace xla {
namespace ifrt {

absl::StatusOr<AttributeMap> AttributeMap::FromProto(
    const AttributeMapProto& proto) {
  AttributeMap::Map map;
  map.reserve(proto.attributes_size());
  for (const auto& [key, value] : proto.attributes()) {
    switch (value.value_case()) {
      case AttributeMapProto::Value::kStringValue:
        map.insert({key, StringValue(value.string_value())});
        break;
      case AttributeMapProto::Value::kBoolValue:
        map.insert({key, BoolValue(value.bool_value())});
        break;
      case AttributeMapProto::Value::kInt64Value:
        map.insert({key, Int64Value(value.int64_value())});
        break;
      case AttributeMapProto::Value::kInt64ListValue:
        map.insert({key, Int64ListValue(std::vector<int64_t>(
                             value.int64_list_value().elements().begin(),
                             value.int64_list_value().elements().end()))});
        break;
      case AttributeMapProto::Value::kFloatValue:
        map.insert({key, FloatValue(value.float_value())});
        break;
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported value type: ", value.value_case()));
    }
  }
  return AttributeMap(std::move(map));
}

AttributeMapProto AttributeMap::ToProto() const {
  AttributeMapProto proto;
  for (const auto& [key, value] : map_) {
    AttributeMapProto::Value value_proto;
    std::visit(
        [&](const auto& value) {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, StringValue>) {
            value_proto.set_string_value(value.value);
          } else if constexpr (std::is_same_v<T, BoolValue>) {
            value_proto.set_bool_value(value.value);
          } else if constexpr (std::is_same_v<T, Int64Value>) {
            value_proto.set_int64_value(value.value);
          } else if constexpr (std::is_same_v<T, Int64ListValue>) {
            auto* int64_list = value_proto.mutable_int64_list_value();
            int64_list->mutable_elements()->Reserve(value.value.size());
            for (const auto& element : value.value) {
              int64_list->add_elements(element);
            }
          } else if constexpr (std::is_same_v<T, FloatValue>) {
            value_proto.set_float_value(value.value);
          }
        },
        value);
    proto.mutable_attributes()->insert({key, std::move(value_proto)});
  }
  return proto;
}

std::string AttributeMap::DebugString(size_t max_string_length,
                                      size_t max_int64_list_size) const {
  auto formatter = [=](std::string* out,
                       const AttributeMap::Map::value_type& key_value) {
    absl::StrAppend(out, key_value.first, "=");
    std::visit(
        [&](const auto& value) {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, StringValue>) {
            if (value.value.size() > max_string_length) {
              absl::StrAppend(
                  out, "\"", value.value.substr(0, max_string_length), "...\"");
            } else {
              absl::StrAppend(out, "\"", value.value, "\"");
            }
          } else if constexpr (std::is_same_v<T, BoolValue>) {
            absl::StrAppend(out, value.value ? "true" : "false");
          } else if constexpr (std::is_same_v<T, Int64Value>) {
            absl::StrAppend(out, value.value);
          } else if constexpr (std::is_same_v<T, Int64ListValue>) {
            if (value.value.size() > max_int64_list_size) {
              absl::StrAppend(
                  out, "[",
                  absl::StrJoin(value.value.begin(),
                                value.value.begin() + max_int64_list_size,
                                ", "),
                  "...]");
            } else {
              absl::StrAppend(out, "[", absl::StrJoin(value.value, ", "), "]");
            }
          } else if constexpr (std::is_same_v<T, FloatValue>) {
            absl::StrAppend(out, value.value);
          }
        },
        key_value.second);
  };

  return absl::StrCat("AttributeMap([", absl::StrJoin(map_, ", ", formatter),
                      "])");
}

}  // namespace ifrt
}  // namespace xla
