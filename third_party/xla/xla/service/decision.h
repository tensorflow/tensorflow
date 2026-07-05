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

#ifndef XLA_SERVICE_DECISION_H_
#define XLA_SERVICE_DECISION_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/base/attributes.h"
#include "absl/base/macros.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/platform.h"  // For PLATFORM_GOOGLE.

#if defined(PLATFORM_GOOGLE)
#include "absl/types/source_location.h"
#endif  // PLATFORM_GOOGLE

namespace xla {

// A holder for the source location. absl::SourceLocation is not available in
// open source, so we have a stub implementation to limit
// #if define(PLATFORM_GOOGLE).
class SourceLocationHolder {
 public:
#if defined(PLATFORM_GOOGLE)
  explicit constexpr SourceLocationHolder(
      absl::SourceLocation source_location = absl::SourceLocation::current())
      : source_location_(source_location) {}

  std::string ToString() const {
    return absl::StrCat(" at: ", source_location_.file_name(), ":",
                        source_location_.line());
  }

 private:
  absl::SourceLocation source_location_;
#else
  SourceLocationHolder() = default;
  std::string ToString() const { return ""; }
#endif  // PLATFORM_GOOGLE
};

// Generic decision class that wraps a boolean result and an optional
// explanation.
class Decision {
 public:
  static Decision Allow() { return Decision(); }

  static Decision Forbid(
      absl::string_view explanation,
      SourceLocationHolder source_location = SourceLocationHolder()) {
    return Decision(false, explanation, source_location);
  }

  Decision(const Decision& decision) = default;
  Decision(Decision&& decision) = default;
  Decision& operator=(const Decision& decision) = default;
  Decision& operator=(Decision&& decision) = default;

  // If condition is `true` means that we CAN make the decision. In that case,
  // explanation is discarded.
  Decision(bool condition, absl::string_view explanation,
           SourceLocationHolder source_location = SourceLocationHolder()) {
    if (!condition) {
      explanation_ = explanation;
      source_location_ = source_location;
    }
  }

  explicit Decision(absl::Status status, SourceLocationHolder source_location =
                                             SourceLocationHolder()) {
    if (!status.ok()) {
      explanation_ = status.message();
      source_location_ = source_location;
    }
  }

  // We can make the decision iff. the decision is `true`.
  Decision(  // NOLINT
      bool decision,
      SourceLocationHolder source_location = SourceLocationHolder())
      : Decision(decision, "Not allowed", source_location) {}

  // Returns whether the decision is positive.
  explicit operator bool() const { return IsAllowed(); }

  // Whether the decision is positive.
  bool IsAllowed() const { return !explanation_.has_value(); }

  // Whether the decision is negative.
  bool IsForbidden() const { return explanation_.has_value(); }

  ABSL_REFACTOR_INLINE ABSL_DEPRECATED(
      "Use IsAllowed() or IsForbidden() instead.") inline bool CanFuse() const {
    return IsAllowed();
  }

  // Connects two decisions with a disjunction.
  Decision Or(const Decision& decision) const {
    if (IsAllowed() || decision.IsAllowed()) {
      return Allow();
    }
    return Forbid(absl::StrCat(Explain(), " ; ", decision.Explain()));
  }

  // Connects two decisions with a conjunction.
  Decision And(const Decision& decision) const {
    if (IsAllowed()) {
      return decision;
    }
    return *this;
  }

  // Appends to explanation, or turns the decision negative.
  Decision operator<<(absl::string_view explanation) const {
    return Forbid(absl::StrCat(explanation_.value_or(""), explanation),
                  source_location_);
  }

  // Appends to explanation, or turns the decision negative.
  Decision operator<<(int64_t explanation) const {
    return Forbid(absl::StrCat(explanation_.value_or(""), explanation),
                  source_location_);
  }

  // Explains why the decision could not be made, or that it was.
  std::string Explain() const {
    if (IsForbidden()) {
      return absl::StrCat(explanation_.value(), source_location_.ToString());
    }
    return "Allowed";
  }

 protected:
  // Empty IFF decision is positive (explanation provided for negative cases).
  std::optional<std::string> explanation_;
  SourceLocationHolder source_location_;

  Decision() = default;
};

}  // namespace xla

#endif  // XLA_SERVICE_DECISION_H_
