/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PJRT_HOST_MEMORY_SPACES_H_
#define XLA_PJRT_HOST_MEMORY_SPACES_H_

#include <string>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"

namespace xla {

// Represents the unpinned host memory accessible to a `PjRtDevice`.
// An "unpinned" host memory space accommodates ordinary host buffers that are
// not mapped to any virtual memory of the attached `PjRtDevice`.
class UnpinnedHostMemorySpace : public PjRtMemorySpace {
 public:
  static constexpr absl::string_view kKind = "unpinned_host";
  static const int kKindId;

  UnpinnedHostMemorySpace(int id, PjRtDevice* device);

  PjRtClient* client() const override { return device_->client(); }

  absl::Span<PjRtDevice* const> devices() const override {
    return absl::Span<PjRtDevice* const>(&device_, device_ != nullptr ? 1 : 0);
  }

  int id() const override { return id_; }

  absl::string_view kind() const override { return kKind; }

  int kind_id() const override { return kKindId; }

  absl::string_view DebugString() const override { return debug_string_; }

  absl::string_view ToString() const override { return to_string_; }

 private:
  int id_;
  PjRtDevice* device_ = nullptr;
  std::string debug_string_;
  std::string to_string_;
};

// Represents the pinned host memory accessible to a `PjRtDevice`.
// A "pinned" host memory space accommodates host buffers that are mapped to a
// virtual memory of the attached `PjRtDevice`. The `PjRtDevice` may have the
// capability to direct-memory-access (DMA) the buffers in this memory space.
class PinnedHostMemorySpace : public PjRtMemorySpace {
 public:
  static constexpr absl::string_view kKind = "pinned_host";
  static const int kKindId;

  PinnedHostMemorySpace(int id, PjRtDevice* device);

  PjRtClient* client() const override { return device_->client(); }

  absl::Span<PjRtDevice* const> devices() const override {
    return absl::Span<PjRtDevice* const>(&device_, device_ != nullptr ? 1 : 0);
  }

  int id() const override { return id_; }

  absl::string_view kind() const override { return kKind; }

  int kind_id() const override { return kKindId; }

  absl::string_view DebugString() const override { return debug_string_; }

  absl::string_view ToString() const override { return to_string_; }

 private:
  int id_;
  PjRtDevice* device_ = nullptr;
  std::string debug_string_;
  std::string to_string_;
};

}  // namespace xla

#endif  // XLA_PJRT_HOST_MEMORY_SPACES_H_
