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

#ifndef XLA_PJRT_CPU_CPU_DEVICE_H_
#define XLA_PJRT_CPU_CPU_DEVICE_H_

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_device_description.h"
#include "xla/pjrt/semaphore.h"

namespace xla {

class TfrtCpuDevice final : public PjRtDevice {
 public:
  explicit TfrtCpuDevice(int process_id, int local_device_id,
                         int max_inflight_computations = 32);

  const CpuDeviceDescription& description() const override {
    return description_;
  }

  void SetClient(PjRtClient* client) {
    CHECK(client_ == nullptr);
    client_ = client;
  }

  PjRtClient* client() const override { return client_; }

  bool IsAddressable() const override {
    return process_index() == client()->process_index();
  }

  PjRtLocalDeviceId local_device_id() const override {
    return PjRtLocalDeviceId(local_hardware_id().value());
  }

  PjRtLocalHardwareId local_hardware_id() const override {
    return PjRtLocalHardwareId(description_.local_hardware_id());
  }

  absl::Status TransferToInfeed(const LiteralSlice& literal) override;

  absl::Status TransferFromOutfeed(MutableBorrowingLiteral literal) override;

  void AttachMemorySpace(PjRtMemorySpace* memory_space);

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

  absl::StatusOr<PjRtMemorySpace*> default_memory_space() const override;

  absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind(
      absl::string_view memory_space_kind) const override;

  absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind_id(int id) const;

  // Returns a semaphore for admission control on inflight computations.
  Semaphore& max_inflight_computations_semaphore() {
    return max_inflight_computations_semaphore_;
  }

  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
    return nullptr;
  }

 private:
  PjRtClient* client_ = nullptr;
  CpuDeviceDescription description_;
  absl::InlinedVector<PjRtMemorySpace*, 1> memory_spaces_;
  absl::flat_hash_map<int, PjRtMemorySpace*> memory_spaces_by_id_;

  // TODO(zhangqiaorjc): Optimize semaphore related overhead.
  // Semaphore used to limit how many programs can be enqueued by the host
  // ahead of the device.
  Semaphore max_inflight_computations_semaphore_;
};

}  // namespace xla

#endif  // XLA_PJRT_CPU_CPU_DEVICE_H_
