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

#ifndef XLA_HLO_PASS_HLO_PASS_FILTER_H_
#define XLA_HLO_PASS_HLO_PASS_FILTER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/repeated_ptr_field.h"

namespace xla {

// Filter for HLO passes, parsed from xla_disable_hlo_passes or
// xla_enable_hlo_passes_only flags.
class HloPassFilter {
 public:
  struct InvocationInfo {
    absl::string_view pass_name;
    // The name of the immediate parent pipeline of this pass invocation.
    absl::string_view pipeline_name;
    // The raw pass_id assigned by HloModuleMetadata (includes pipeline-start
    // pseudo-pass increments, so it lines up with GetHloPassPipelineTrace and
    // HLO dumps).
    int64_t pass_id;
    // 0-based index of this invocation among all invocations of `pass_name` in
    // the module so far.
    int64_t global_occurrence;
    // 0-based index of this invocation among invocations of `pass_name` within
    // the current pipeline instance so far.
    int64_t pipeline_occurrence;
  };

  // Parses a comma-separated flag string (e.g. from --xla_disable_hlo_passes).
  // Returns an InvalidArgument error if any entry in the string is malformed.
  static absl::StatusOr<HloPassFilter> FromFlag(
      absl::string_view comma_separated_values);

  // Validates a single filter entry string (e.g. "algsimp", "algsimp:2",
  // "scope/name", "@42"). Returns OK for valid syntax, or an InvalidArgument
  // error if the entry is malformed.
  static absl::Status ValidateEntry(absl::string_view entry);

  // Constructs a filter from a repeated protobuf string field (e.g. from
  // debug_options.xla_disable_hlo_passes()).
  static absl::StatusOr<HloPassFilter> FromRepeatedProtoField(
      const google::protobuf::RepeatedPtrField<std::string>& entries);

  bool empty() const;

  // Returns true iff this filter matches the runtime pass invocation.
  bool Matches(const InvocationInfo& invocation) const;

 private:
  struct FilterSpec {
    std::optional<int64_t> pass_id;
    std::optional<std::string> pipeline_scope;
    std::optional<std::string> pass_name;
    std::optional<int64_t> occurrence;
  };

  static absl::StatusOr<FilterSpec> ParseSpec(absl::string_view entry);

  std::vector<FilterSpec> specs_;
};

}  // namespace xla

#endif  // XLA_HLO_PASS_HLO_PASS_FILTER_H_
