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

#include "xla/pjrt/gpu/allocator_config.h"

#include <cmath>
#include <cstddef>
#include <string>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace xla {
namespace {

constexpr absl::string_view kGrammar =
    "expected START, START+, or START-CAP as fractions of total device memory, "
    "for example \"0.75\" (fixed at 75%), \"0.75+\" (preallocate 75%, grow to "
    "all memory), or \"0.75-0.85\" (grow to at most 85%)";

absl::Status InvalidSpec(absl::string_view spec, absl::string_view reason) {
  return absl::InvalidArgumentError(absl::StrCat(
      "Invalid memory fraction \"", spec, "\": ", reason, "; ", kGrammar));
}

absl::StatusOr<double> ParseFraction(absl::string_view text,
                                     absl::string_view spec) {
  double value = 0;
  if (!absl::SimpleAtod(absl::StripAsciiWhitespace(text), &value) ||
      !std::isfinite(value) || value <= 0) {
    return InvalidSpec(spec, "fractions must be positive numbers");
  }
  return value;
}

}  // namespace

absl::StatusOr<MemFraction> ParseMemFraction(absl::string_view spec) {
  absl::string_view text = absl::StripAsciiWhitespace(spec);
  const bool grow_to_full_memory = !text.empty() && text.back() == '+';
  size_t separator = absl::string_view::npos;
  if (grow_to_full_memory) {
    text.remove_suffix(1);
  } else {
    // A minus sign in a numeric exponent is not a range separator.
    separator = text.find('-');
    while (separator != absl::string_view::npos && separator > 0 &&
           (text[separator - 1] == 'e' || text[separator - 1] == 'E')) {
      separator = text.find('-', separator + 1);
    }
  }
  absl::StatusOr<double> start = ParseFraction(text.substr(0, separator), spec);
  if (!start.ok()) {
    return start.status();
  }
  if (!grow_to_full_memory && separator == absl::string_view::npos) {
    return MemFractionFromFraction(*start);
  }
  absl::StatusOr<double> cap =
      grow_to_full_memory ? absl::StatusOr<double>(1.0)
                          : ParseFraction(text.substr(separator + 1), spec);
  if (!cap.ok()) {
    return cap.status();
  }
  if (*cap < *start) {
    return InvalidSpec(spec, "the cap must be at least the start");
  }
  if (*cap == *start) {
    return MemFraction(FixedMemFraction{*start});
  }
  if (*cap > 1) {
    return InvalidSpec(spec,
                       "a growth cap cannot exceed 1 (all of device memory)");
  }
  return MemFraction(FlexMemFraction{*start, *cap});
}

MemFraction MemFractionFromFraction(double fraction) {
  return FixedMemFraction{fraction};
}

double MemFractionStart(const MemFraction& fraction) {
  if (const auto* fixed = std::get_if<FixedMemFraction>(&fraction)) {
    return fixed->fraction;
  }
  return std::get<FlexMemFraction>(fraction).start;
}

std::string MemFractionToString(const MemFraction& fraction) {
  if (const auto* fixed = std::get_if<FixedMemFraction>(&fraction)) {
    return absl::StrCat(fixed->fraction);
  }
  const auto& flex = std::get<FlexMemFraction>(fraction);
  if (flex.cap == 1) {
    return absl::StrCat(flex.start, "+");
  }
  return absl::StrCat(flex.start, "-", flex.cap);
}

MemFraction GpuAllocatorConfig::GetMemoryFraction() const {
  return memory_fraction_policy.value_or(
      MemFractionFromFraction(memory_fraction));
}

}  // namespace xla
