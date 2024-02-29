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

#include <string>
#include <utility>

#include "xla/layout.h"
#include "xla/service/hlo_parser.h"
#include "xla/statusor.h"

namespace xla {

// Abstract class representing the memory layout of a PjRtBuffer.
class PjRtLayout {
 public:
  virtual ~PjRtLayout() = default;

  // Returns the serialized layout as a string.
  // TODO(skyewm): add generic deserialize method to PjRtClient and/or
  // PjRtCompiler.
  virtual std::string Serialize() const = 0;

  // Human-readable string for error messages, user introspection, etc.
  virtual std::string ToString() const = 0;
};

// PjRtLayout backed by an xla::Layout. This is a convenience class for PJRT
// implementations that use XLA. PJRT users should use the PjRtLayout interface
// to be compatible with all implementations, e.g. PjRtCApiClient which doesn't
// have access to full xla::Layouts.
class PjRtXlaLayout : public PjRtLayout {
 public:
  explicit PjRtXlaLayout(Layout layout) : xla_layout_(std::move(layout)) {}

  std::string Serialize() const override { return xla_layout_.ToString(); }

  static StatusOr<PjRtXlaLayout> Deserialize(absl::string_view serialized) {
    TF_ASSIGN_OR_RETURN(Layout xla_layout, ParseLayout(serialized));
    return PjRtXlaLayout(std::move(xla_layout));
  }

  std::string ToString() const override { return xla_layout_.ToString(); }

  const Layout& xla_layout() const { return xla_layout_; }

 private:
  Layout xla_layout_;
};

}  // namespace xla

#endif  // XLA_PJRT_PJRT_LAYOUT_H_
