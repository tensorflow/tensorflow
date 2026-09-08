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

#include "xla/hlo/pass/hlo_pass_filter.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/repeated_ptr_field.h"

namespace xla {

absl::StatusOr<HloPassFilter::FilterSpec> HloPassFilter::ParseSpec(
    absl::string_view entry) {
  FilterSpec spec;

  // "@N": match by raw pass_id.
  if (!entry.empty() && entry[0] == '@') {
    int64_t pass_id = 0;
    if (!absl::SimpleAtoi(entry.substr(1), &pass_id)) {
      return absl::InvalidArgumentError(
          absl::StrCat("invalid pass_id in HLO pass filter entry '", entry,
                       "': expected an integer after '@'"));
    }
    spec.pass_id = pass_id;
    return spec;
  }

  // Optional trailing ":N" occurrence index.
  const std::vector<absl::string_view> colon_parts =
      absl::StrSplit(entry, absl::MaxSplits(':', 1));
  if (colon_parts.size() == 2) {
    int64_t occurrence = 0;
    if (!absl::SimpleAtoi(colon_parts[1], &occurrence)) {
      return absl::InvalidArgumentError(
          absl::StrCat("invalid occurrence in HLO pass filter entry '", entry,
                       "': expected an integer after ':'"));
    }
    spec.occurrence = occurrence;
  }

  // Optional "scope/" immediate-parent pipeline prefix.
  const std::vector<absl::string_view> slash_parts =
      absl::StrSplit(colon_parts[0], absl::MaxSplits('/', 1));
  if (slash_parts.size() == 2) {
    spec.pipeline_scope = std::string(slash_parts[0]);
  }
  spec.pass_name = std::string(slash_parts[slash_parts.size() - 1]);

  return spec;
}

absl::StatusOr<HloPassFilter> HloPassFilter::FromFlag(
    absl::string_view comma_separated_values) {
  HloPassFilter filter;
  for (absl::string_view token : absl::StrSplit(comma_separated_values, ',')) {
    ABSL_ASSIGN_OR_RETURN(FilterSpec spec, ParseSpec(token));
    filter.specs_.push_back(std::move(spec));
  }
  return filter;
}

absl::Status HloPassFilter::ValidateEntry(absl::string_view entry) {
  return ParseSpec(entry).status();
}

absl::StatusOr<HloPassFilter> HloPassFilter::FromRepeatedProtoField(
    const google::protobuf::RepeatedPtrField<std::string>& entries) {
  HloPassFilter filter;
  for (const auto& entry : entries) {
    ABSL_ASSIGN_OR_RETURN(FilterSpec spec, ParseSpec(entry));
    filter.specs_.push_back(std::move(spec));
  }
  return filter;
}

bool HloPassFilter::empty() const { return specs_.empty(); }

bool HloPassFilter::Matches(const InvocationInfo& invocation) const {
  return absl::c_any_of(specs_, [&](const FilterSpec& spec) {
    // "@N" matches the raw pass_id and ignores all other context.
    if (spec.pass_id.has_value()) {
      return invocation.pass_id == *spec.pass_id;
    }
    // Plain name rules also match the parent pipeline name (e.g. disabling
    // "fusion" disables all passes inside a "fusion" pipeline).
    if (spec.pass_name.has_value() && *spec.pass_name != invocation.pass_name &&
        (spec.pipeline_scope.has_value() || spec.occurrence.has_value() ||
         *spec.pass_name != invocation.pipeline_name)) {
      return false;
    }
    if (spec.pipeline_scope.has_value() &&
        *spec.pipeline_scope != invocation.pipeline_name) {
      return false;
    }
    if (spec.occurrence.has_value()) {
      const int64_t occurrence = spec.pipeline_scope.has_value()
                                     ? invocation.pipeline_occurrence
                                     : invocation.global_occurrence;
      if (occurrence != *spec.occurrence) {
        return false;
      }
    }
    return true;
  });
}

}  // namespace xla
