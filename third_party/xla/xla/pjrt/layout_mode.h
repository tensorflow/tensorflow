/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PJRT_LAYOUT_MODE_H_
#define XLA_PJRT_LAYOUT_MODE_H_

#include <string>

#include "absl/status/statusor.h"
#include "xla/layout.h"
#include "xla/shape.h"

namespace xla {

// Helper struct for specifying how to choose the layout for a value in a
// program to be compiled (e.g. a computation argument).
//
// The source of truth for this info is the "mhlo.layout_mode" string attribute
// of input MLIR modules. This struct can help manage the attribute. The
// ToString and FromString methods can be used to convert between this struct
// and the "mhlo.layout_mode" string attr.
struct LayoutMode {
  enum class Mode {
    // Use the default compact layout.
    kDefault = 0,
    // Use `layout`.
    kUserSpecified,
    // Let compiler choose layout.
    kAuto
  };
  Mode mode = Mode::kDefault;

  // Only set iff layout_mode == kUserSpecified. This is the layout of the
  // per-device data, i.e. if the computation is sharded, the caller must choose
  // both the sharding and layout for this value such that they're compatible.
  std::optional<Layout> user_layout;

  LayoutMode() = default;
  explicit LayoutMode(Mode layout_mode,
                      std::optional<Layout> layout = std::nullopt);
  explicit LayoutMode(const Layout& layout)
      : LayoutMode(Mode::kUserSpecified, layout) {}
  explicit LayoutMode(const Shape& shape_with_layout)
      : LayoutMode(Mode::kUserSpecified, shape_with_layout.layout()) {}

  // Produces a human-readable string representing this LayoutMode. Is also in
  // the correct format for the "mhlo.layout_mode" attribute.
  std::string ToString() const;
  // Parses a string produced by LayoutMode::ToString() or Layout::ToString().
  static absl::StatusOr<LayoutMode> FromString(std::string s);
};

}  // namespace xla

#endif  // XLA_PJRT_LAYOUT_MODE_H_
