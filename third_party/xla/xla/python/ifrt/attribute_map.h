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

#ifndef XLA_PYTHON_IFRT_ATTRIBUTE_MAP_H_
#define XLA_PYTHON_IFRT_ATTRIBUTE_MAP_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/python/ifrt/attribute_map.pb.h"
#include "xla/python/ifrt/serdes_default_version_accessor.h"
#include "xla/python/ifrt/serdes_version.h"

namespace xla {
namespace ifrt {

// Attribute map that contains UTF-8 keys and variant values.
class AttributeMap {
 public:
  // Supported value types for `AttributeMap`. Modeled after
  // `xla::PjRtValueType`, but they add a layer of structs that prevent implicit
  // conversion. This ensures that `Value` to be constructed with a correct
  // type. See
  // https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0608r3.html
  // construction of `Value` with a wrong type.
  struct StringValue {
    explicit StringValue(std::string value) : value(std::move(value)) {}
    std::string value;
    bool operator==(const StringValue& other) const {
      return value == other.value;
    }
  };
  struct BoolValue {
    explicit BoolValue(bool value) : value(value) {}
    bool operator==(const BoolValue& other) const {
      return value == other.value;
    }
    bool value;
  };
  struct Int64Value {
    explicit Int64Value(int64_t value) : value(value) {}
    bool operator==(const Int64Value& other) const {
      return value == other.value;
    }
    int64_t value;
  };
  struct Int64ListValue {
    explicit Int64ListValue(std::vector<int64_t> value)
        : value(std::move(value)) {}
    bool operator==(const Int64ListValue& other) const {
      return value == other.value;
    }
    std::vector<int64_t> value;
  };
  struct FloatValue {
    explicit FloatValue(float value) : value(value) {}
    bool operator==(const FloatValue& other) const {
      return value == other.value;
    }
    float value;
  };
  using Value = std::variant<StringValue, BoolValue, Int64Value, Int64ListValue,
                             FloatValue>;

  using Map = absl::flat_hash_map<std::string, Value>;

  explicit AttributeMap(Map map) : map_(std::move(map)) {}

  const Map& map() const { return map_; }

  // Deserializes `AttributeMapProto` into `AttributeMap`.
  static absl::StatusOr<AttributeMap> FromProto(const AttributeMapProto& proto);

  // Serializes `AttributeMap` into `AttributeMapProto`.
  AttributeMapProto ToProto(
      SerDesVersion version = SerDesDefaultVersionAccessor::Get()) const;

  std::string DebugString(size_t max_string_length = 64,
                          size_t max_int64_list_size = 16) const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const AttributeMap& attribute_map) {
    sink.Append(attribute_map.DebugString());
  }

 private:
  Map map_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_ATTRIBUTE_MAP_H_
