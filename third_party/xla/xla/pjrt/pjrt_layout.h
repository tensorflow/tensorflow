/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_PJRT_PJRT_LAYOUT_H_
#define XLA_PJRT_PJRT_LAYOUT_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout.h"
#include "tsl/platform/statusor.h"

namespace xla {

// Represents the memory layout of a PjRtBuffer.
class PjRtLayout {
 public:
  explicit PjRtLayout(Layout layout) : xla_layout_(std::move(layout)) {
    // Strip memory space and set it to the default. PJRT tracks memory space
    // separately from layout.
    xla_layout_.set_memory_space(xla::Layout::kDefaultMemorySpace);
  }

  PjRtLayout(PjRtLayout& other) = delete;
  PjRtLayout& operator=(const PjRtLayout& other) = delete;

  static absl::StatusOr<std::shared_ptr<const PjRtLayout>> Deserialize(
      absl::string_view serialized) {
    TF_ASSIGN_OR_RETURN(Layout xla_layout, ParseLayout(serialized));
    return std::make_shared<PjRtLayout>(std::move(xla_layout));
  }

  const Layout& xla_layout() const { return xla_layout_; }

  // Returns the serialized layout as a string.
  std::string Serialize() const { return xla_layout_.ToString(); }

  // Human-readable string for error messages, user introspection, etc.
  std::string ToString() const { return xla_layout_.ToString(); }

  bool operator==(const PjRtLayout& other) const {
    return xla_layout_ == other.xla_layout_;
  }

  template <typename H>
  friend H AbslHashValue(H state, const PjRtLayout& layout) {
    return H::combine(std::move(state), layout.xla_layout_);
  }

 private:
  Layout xla_layout_;
};

}  // namespace xla

#endif  // XLA_PJRT_PJRT_LAYOUT_H_
