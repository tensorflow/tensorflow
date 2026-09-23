/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_CUSTOM_OPTIONS_H_
#define XLA_CUSTOM_OPTIONS_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/util.h"

namespace xla {

// Custom options are per-execution key/value options passed by the caller to
// the XLA runtime. Unlike compile-time options, they can be different for every
// execution of the same executable.
class CustomOptions {
 public:
  // The order of the alternatives matches `xla::PjRtValueType` so that the two
  // maps are layout-compatible: `bool` comes before `std::string` to avoid
  // pybind conversion ambiguity (see `xla/pjrt/pjrt_common.h`).
  using Value =
      std::variant<std::string, bool, int64_t, std::vector<int64_t>, float>;
  using Map = absl::flat_hash_map<std::string, Value>;

  CustomOptions() = default;
  explicit CustomOptions(Map options) : options_(std::move(options)) {}

  // Returns the value of the option `name`, or an error if the option is not
  // found or has a different type. `T` must be one of the `Value` types.
  template <typename T>
  absl::StatusOr<T> Get(absl::string_view name) const;

  // Returns the value of the option `name` or nullptr if it is not found.
  const Value* Find(absl::string_view name) const {
    auto it = options_.find(name);
    return it == options_.end() ? nullptr : &it->second;
  }

  bool contains(absl::string_view name) const {
    return options_.contains(name);
  }

  size_t size() const { return options_.size(); }
  bool empty() const { return options_.empty(); }

  const Map& map() const { return options_; }

 private:
  Map options_;
};

template <typename T>
absl::StatusOr<T> CustomOptions::Get(absl::string_view name) const {
  const Value* value = Find(name);
  if (value == nullptr) {
    return NotFound("Custom option %s not found", name);
  }
  const T* typed = std::get_if<T>(value);
  if (typed == nullptr) {
    return InvalidArgument("Custom option %s has an unexpected type", name);
  }
  return *typed;
}

}  // namespace xla

#endif  // XLA_CUSTOM_OPTIONS_H_
