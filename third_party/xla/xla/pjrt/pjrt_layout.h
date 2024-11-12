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

#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/statusor.h"

namespace xla {

// Abstract class representing the memory layout of a PjRtBuffer.
class PjRtLayout {
 public:
  virtual ~PjRtLayout() = default;

  // Returns the serialized layout as a string.
  // TODO(b/328671718): add generic deserialize method to PjRtClient and/or
  // PjRtCompiler.
  virtual std::string Serialize() const = 0;

  // Human-readable string for error messages, user introspection, etc.
  virtual std::string ToString() const = 0;

  virtual bool operator==(const PjRtLayout& other) const = 0;

  template <typename H>
  friend H AbslHashValue(H state, const PjRtLayout& layout) {
    layout.Hash(absl::HashState::Create(&state));
    return std::move(state);
  }

 protected:
  virtual void Hash(absl::HashState state) const = 0;
};

// PjRtLayout backed by an xla::Layout. This is a convenience class for PJRT
// implementations that use XLA. PJRT users should use the PjRtLayout interface
// to be compatible with all implementations, e.g. PjRtCApiClient which doesn't
// have access to full xla::Layouts.
class PjRtXlaLayout : public PjRtLayout {
 public:
  explicit PjRtXlaLayout(Layout layout) : xla_layout_(std::move(layout)) {
    // Strip memory space and set it to the default. PJRT tracks memory space
    // separately from layout.
    xla_layout_.set_memory_space(xla::Layout::kDefaultMemorySpace);
  }

  std::string Serialize() const override { return xla_layout_.ToString(); }

  static absl::StatusOr<PjRtXlaLayout> Deserialize(
      absl::string_view serialized) {
    TF_ASSIGN_OR_RETURN(Layout xla_layout, ParseLayout(serialized));
    return PjRtXlaLayout(std::move(xla_layout));
  }

  std::string ToString() const override { return xla_layout_.ToString(); }

  bool operator==(const PjRtLayout& other) const override {
    auto xla_other = dynamic_cast<const PjRtXlaLayout*>(&other);
    if (xla_other == nullptr) {
      return false;
    }
    return xla_layout_ == xla_other->xla_layout_;
  };

  const Layout& xla_layout() const { return xla_layout_; }

 protected:
  void Hash(absl::HashState state) const override {
    absl::HashState::combine(std::move(state), xla_layout_);
  }

 private:
  Layout xla_layout_;
};

// TODO(b/327524065): make callers use PjRtLayout directly instead of assuming
// an xla::Layout and get rid of this function.
inline Layout GetXlaLayoutUnsafe(
    const std::unique_ptr<PjRtLayout>& pjrt_layout) {
  PjRtXlaLayout* xla_layout =
      tensorflow::down_cast<PjRtXlaLayout*>(pjrt_layout.get());
  CHECK(xla_layout != nullptr) << "Got unexpected layout type";
  return xla_layout->xla_layout();
}

}  // namespace xla

#endif  // XLA_PJRT_PJRT_LAYOUT_H_
