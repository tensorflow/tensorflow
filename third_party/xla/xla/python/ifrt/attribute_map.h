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
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/python/ifrt/attribute_map.pb.h"
#include "xla/python/ifrt/serdes_default_version_accessor.h"
#include "xla/python/ifrt/serdes_version.h"

namespace xla {
namespace ifrt {

// Attribute map that contains UTF-8 keys and variant values.
//
// This class is thread-safe.
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

  template <typename T>
  absl::StatusOr<T> Get(const std::string& key) const {
    absl::ReaderMutexLock lock(mu_);
    if constexpr (std::is_same_v<T, Value>) {
      auto it = map_.find(key);
      if (it == map_.end()) {
        return absl::NotFoundError(absl::StrCat("Key not found: ", key));
      }
      return it->second;
    } else if constexpr (std::is_same_v<T, std::string>) {
      return Get<T, StringValue>(key);
    } else if constexpr (std::is_same_v<T, bool>) {
      return Get<T, BoolValue>(key);
    } else if constexpr (std::is_same_v<T, int64_t>) {
      return Get<T, Int64Value>(key);
    } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
      return Get<T, Int64ListValue>(key);
    } else if constexpr (std::is_same_v<T, float>) {
      return Get<T, FloatValue>(key);
    } else {
      static_assert(false, "Unsupported type for AttributeMap::Get");
    }
  }

  template <typename T>
  absl::Status Set(const std::string& key, T value) {
    absl::MutexLock lock(mu_);
    using ValueType = std::decay_t<T>;
    if constexpr (std::is_same_v<ValueType, std::string> ||
                  std::is_same_v<ValueType, const char*> ||
                  std::is_convertible_v<ValueType, std::string>) {
      map_.insert_or_assign(key, StringValue(std::move(value)));
    } else if constexpr (std::is_same_v<ValueType, bool>) {
      map_.insert_or_assign(key, BoolValue(std::move(value)));
    } else if constexpr (std::is_same_v<ValueType, int64_t>) {
      map_.insert_or_assign(key, Int64Value(std::move(value)));
    } else if constexpr (std::is_same_v<ValueType, std::vector<int64_t>>) {
      map_.insert_or_assign(key, Int64ListValue(std::move(value)));
    } else if constexpr (std::is_same_v<ValueType, float>) {
      map_.insert_or_assign(key, FloatValue(std::move(value)));
    } else {
      static_assert(false, "Unsupported type for AttributeMap::Set");
    }
    return absl::OkStatus();
  }

  // Deserializes `AttributeMapProto` into `AttributeMap`.
  static absl::StatusOr<AttributeMap> FromProto(const AttributeMapProto& proto);

  // Converts the attribute map to a protobuf.
  void ToProto(
      AttributeMapProto& proto,
      SerDesVersion version = SerDesDefaultVersionAccessor::Get()) const;

  AttributeMapProto ToProto(
      SerDesVersion version = SerDesDefaultVersionAccessor::Get()) const {
    AttributeMapProto proto;
    ToProto(proto, version);
    return proto;
  }

  std::string DebugString(size_t max_string_length = 64,
                          size_t max_int64_list_size = 16) const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const AttributeMap& attribute_map) {
    sink.Append(attribute_map.DebugString());
  }

  bool IsEmpty() const {
    absl::ReaderMutexLock lock(mu_);
    return map_.empty();
  }

  // Invokes `f` for each key-value pair in the attribute map.
  void ForEach(
      absl::FunctionRef<void(const std::string&, const Value&)> f) const {
    absl::ReaderMutexLock lock(mu_);
    for (const auto& [key, value] : map_) {
      f(key, value);
    }
  }

  bool operator==(const AttributeMap& other) const {
    absl::ReaderMutexLock lock1(mu_);
    absl::ReaderMutexLock lock2(other.mu_);
    return map_ == other.map_;
  }

  size_t size() const {
    absl::ReaderMutexLock lock(mu_);
    return map_.size();
  }

  // Copyable and movable.
  AttributeMap(const AttributeMap& other) {
    absl::ReaderMutexLock lock(other.mu_);
    map_ = other.map_;
  }
  AttributeMap& operator=(const AttributeMap& other) {
    Map map;
    {
      absl::ReaderMutexLock lock(other.mu_);
      map = other.map_;
    }
    absl::MutexLock lock(mu_);
    map_ = std::move(map);
    return *this;
  }
  AttributeMap(AttributeMap&& other) {
    absl::MutexLock lock(other.mu_);
    map_ = std::move(other.map_);
  }
  AttributeMap& operator=(AttributeMap&& other) {
    Map map;
    {
      absl::MutexLock lock(other.mu_);
      map = std::move(other.map_);
    }
    absl::MutexLock lock(mu_);
    map_ = std::move(map);
    return *this;
  }

 private:
  template <typename T, typename V>
  absl::StatusOr<T> Get(const std::string& key) const
      ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return absl::NotFoundError(absl::StrCat("Key not found: ", key));
    }
    const V* value = std::get_if<V>(&it->second);
    if (value == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Value type mismatch for key: ", key));
    }
    return value->value;
  }

  mutable absl::Mutex mu_;
  Map map_ ABSL_GUARDED_BY(mu_);
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_ATTRIBUTE_MAP_H_
