/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/pjrt/layout_mode.h"

#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/layout.h"
#include "xla/service/hlo_parser.h"
#include "xla/statusor.h"

namespace xla {

LayoutMode::LayoutMode(Mode layout_mode, std::optional<Layout> layout)
    : mode(layout_mode), user_layout(std::move(layout)) {
  if (mode == Mode::kUserSpecified) {
    CHECK(user_layout) << "Must pass layout to LayoutMode constructor when "
                          "mode == kUserSpecified";
  } else {
    CHECK(!user_layout) << "Only pass layout to LayoutMode constructor "
                           "if mode == kUserSpecified";
  }
}

std::string LayoutMode::ToString() const {
  switch (mode) {
    case Mode::kDefault:
      return "default";
    case Mode::kUserSpecified:
      CHECK(user_layout);
      return user_layout->ToString();
    case Mode::kAuto:
      return "auto";
  }
}

absl::StatusOr<LayoutMode> LayoutMode::FromString(std::string s) {
  if (s == "default") {
    return LayoutMode(Mode::kDefault);
  }
  if (s == "auto") {
    return LayoutMode(Mode::kAuto);
  }
  // LayoutMode is user-specified; parse Layout string
  absl::StatusOr<Layout> layout = ParseLayout(s);
  if (!layout.ok()) {
    absl::Status new_status(
        layout.status().code(),
        absl::StrCat("Error parsing user-specified layout mode '", s,
                     "': ", layout.status().message()));
    return new_status;
  }
  return LayoutMode(*layout);
}

}  // namespace xla
