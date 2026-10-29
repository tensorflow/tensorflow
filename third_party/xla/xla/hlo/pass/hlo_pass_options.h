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

#ifndef XLA_HLO_PASS_HLO_PASS_OPTIONS_H_
#define XLA_HLO_PASS_HLO_PASS_OPTIONS_H_

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <type_traits>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/map.h"
#include "xla/hlo/pass/hlo_pipeline.pb.h"

namespace xla {

// Typed read access to a pass's options (`HloPipelineNodeProto::options`),
// tracking which options have been consumed so the pipeline driver can detect
// misspelled or unsupported option names after pass instantiation.
//
// Example (a pass factory):
//
//   absl::StatusOr<std::unique_ptr<HloPassInterface>> MakeHloDce(
//       const HloPassOptions& options) {
//     return std::make_unique<HloDCE>(
//         options.Get<bool>("remove_cross_partition_collective_ops")
//             .value_or(false));
//   }
//
// Example (building options in code):
//
//   HloPassOptions options = {{"is_layout_sensitive", true}, {"limit", 2}};
class HloPassOptions {
 public:
  using ProtoMap = google::protobuf::Map<std::string, HloPassOptionValue>;

  struct Option {
    template <typename T>
    Option(absl::string_view name, T val) : name(name) {
      if constexpr (std::is_same_v<T, bool>) {
        value.set_bool_val(val);
      } else if constexpr (std::is_integral_v<T>) {
        value.set_int_val(val);
      } else {
        static_assert(sizeof(T) == 0, "Unsupported HloPassOptions type");
      }
    }

    absl::string_view name;
    HloPassOptionValue value;
  };

  HloPassOptions() = default;
  explicit HloPassOptions(const ProtoMap& options)
      : options_(options.begin(), options.end()) {}
  // Option names must be unique.
  HloPassOptions(std::initializer_list<Option> options);

  // Non-copyable to avoid splitting `consumed_keys_`; can be enabled if needed.
  HloPassOptions(const HloPassOptions&) = delete;
  HloPassOptions& operator=(const HloPassOptions&) = delete;
  HloPassOptions(HloPassOptions&&) = default;
  HloPassOptions& operator=(HloPassOptions&&) = default;

  // Returns the option value for `key` and marks it as consumed. Returns
  // nullopt if `key` is absent or holds a different type; the latter leaves
  // `key` unconsumed, so `CheckAllConsumed()` reports it.
  template <typename T>
  std::optional<T> Get(absl::string_view key) const;

  // Returns InvalidArgumentError if any option has not been read via `Get()`,
  // either because no pass reads it or because it was read as another type.
  absl::Status CheckAllConsumed() const;

  // Returns all options. Does not mark any option as consumed.
  ProtoMap ToProto() const;

 private:
  absl::flat_hash_map<std::string, HloPassOptionValue> options_;
  mutable absl::flat_hash_set<std::string> consumed_keys_;
};

template <typename T>
std::optional<T> HloPassOptions::Get(absl::string_view key) const {
  auto it = options_.find(key);
  if (it == options_.end()) {
    return std::nullopt;
  }
  const HloPassOptionValue& value = it->second;
  std::optional<T> result;
  if constexpr (std::is_same_v<T, bool>) {
    if (value.has_bool_val()) {
      result = value.bool_val();
    }
  } else if constexpr (std::is_same_v<T, int64_t>) {
    if (value.has_int_val()) {
      result = value.int_val();
    }
  } else {
    static_assert(sizeof(T) == 0, "Unsupported HloPassOptions type");
  }
  if (result.has_value()) {
    consumed_keys_.insert(it->first);
  }
  return result;
}

}  // namespace xla

#endif  // XLA_HLO_PASS_HLO_PASS_OPTIONS_H_
