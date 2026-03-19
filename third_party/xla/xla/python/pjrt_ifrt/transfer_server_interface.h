/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_PJRT_IFRT_TRANSFER_SERVER_INTERFACE_H_
#define XLA_PYTHON_PJRT_IFRT_TRANSFER_SERVER_INTERFACE_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/memory.h"

namespace xla {
namespace ifrt {

class PjRtClient;

// Interface for cross-host transfers. Serves as a fallback in PjRt-IFRT when
// the backend does not support cross-host transfers natively.
class TransferServerInterface {
 public:
  static constexpr int kPjRtBufferInlineSize = 1;
  using PjRtBuffers =
      absl::InlinedVector<std::shared_ptr<PjRtBuffer>, kPjRtBufferInlineSize>;

  virtual ~TransferServerInterface() = default;

  virtual absl::StatusOr<std::vector<xla::ifrt::ArrayRef>>
  CopyArraysForCrossHost(xla::ifrt::PjRtClient* client,
                         absl::Span<ArrayRef> arrays, DeviceListRef src_devices,
                         DeviceListRef dst_devices,
                         std::optional<MemoryKind> memory_kind) = 0;

  // Awaits a pull from a remote process.
  virtual absl::Status CrossHostAwaitPull(
      int64_t uuid, absl::Span<xla::ifrt::ArrayRef> arrays,
      const std::vector<int>& buffer_idxs) = 0;

  // Pulls buffers from a remote process.
  virtual absl::Status CrossHostPull(
      int64_t uuid, absl::Span<xla::ifrt::ArrayRef> arrays,
      std::vector<int>& dst_device_idxs, xla::ifrt::DeviceListRef dst_devices,
      std::optional<MemoryKind> memory_kind, int remote_pid,
      absl::btree_map<int, PjRtBuffers>& buffer_list) = 0;
};

struct TransferServerInterfaceFactory {
  std::function<
      absl::StatusOr<std::unique_ptr<xla::ifrt::TransferServerInterface>>(
          std::shared_ptr<xla::PjRtClient>)>
      factory_fn;
};
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_TRANSFER_SERVER_INTERFACE_H_
