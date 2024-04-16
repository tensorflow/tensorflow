// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/python/ifrt_proxy/client/device.h"

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"

namespace xla {
namespace ifrt {
namespace proxy {

std::unique_ptr<xla::ScopedAsyncTrackingEvent> Device::CreateAsyncTrackingEvent(
    absl::string_view description) const {
  return nullptr;
}

absl::Status Device::TransferToInfeed(const xla::LiteralSlice& literal) {
  return absl::UnimplementedError("Device does not support TransferToInfeed");
}

absl::Status Device::TransferFromOutfeed(xla::MutableBorrowingLiteral literal) {
  return absl::UnimplementedError(
      "Device does not support TransferFromOutfeed");
}

absl::Span<xla::PjRtMemorySpace* const> Device::memory_spaces() const {
  return memory_spaces_;
}

absl::StatusOr<xla::PjRtMemorySpace*> Device::default_memory_space() const {
  if (default_memory_space_ == nullptr) {
    return absl::UnimplementedError(
        "Device does not support default_memory_space");
  }
  return default_memory_space_;
}

char Device::ID = 0;  // NOLINT

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
