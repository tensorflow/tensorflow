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

#ifndef XLA_PJRT_GPU_TFRT_TFRT_GPU_CLIENT_H_
#define XLA_PJRT_GPU_TFRT_TFRT_GPU_CLIENT_H_

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "xla/client/local_client.h"
#include "xla/literal.h"
#include "xla/pjrt/gpu/gpu_topology.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/service/hlo.pb.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

class TfrtGpuDevice final : public PjRtDevice {
 public:
  struct Options {
    int id;
    PjRtLocalDeviceId local_device_id;
    PjRtLocalHardwareId local_hardware_id;
    se::StreamExecutor* executor;
    std::unique_ptr<tsl::Allocator> allocator;
    int stream_capacity;
    int max_inflight_computations;
    std::string platform_version;
  };

  explicit TfrtGpuDevice(Options&& options);

  PjRtClient* client() const override { return client_; }

  bool IsAddressable() const override {
    return process_index() == client()->process_index();
  }

  int id() const override { return id_; }

  int process_index() const override { return 0; }

  PjRtLocalDeviceId local_device_id() const override {
    return local_device_id_;
  }

  // Used as `device_ordinal`.
  PjRtLocalHardwareId local_hardware_id() const override {
    return local_hardware_id_;
  }

  absl::string_view DebugString() const override;

  absl::string_view ToString() const override;

  absl::Status TransferToInfeed(const LiteralSlice& literal) override;

  absl::Status TransferFromOutfeed(MutableBorrowingLiteral literal) override;

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

  absl::StatusOr<PjRtMemorySpace*> default_memory_space() const override;

  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
    return nullptr;
  }

 private:
  int id_;
  PjRtClient* client_ = nullptr;
  PjRtLocalDeviceId local_device_id_;
  PjRtLocalHardwareId local_hardware_id_;
};

class TfrtGpuClient final : public PjRtClient {
 public:
  struct Options {
    std::optional<std::set<int>> allowed_devices;
    std::optional<std::string> platform_name;
  };

  TfrtGpuClient(int process_index, xla::LocalClient* xla_client,
                std::vector<std::unique_ptr<TfrtGpuDevice>> devices,
                std::unique_ptr<tsl::Allocator> host_memory_allocator,
                std::shared_ptr<const GpuTopology> gpu_topology);

  int process_index() const override { return process_index_; }

  int device_count() const override { return devices_.size(); }

  int addressable_device_count() const override {
    return addressable_devices_.size();
  }

  absl::Span<PjRtDevice* const> devices() const override { return devices_; }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

  PjRtPlatformId platform_id() const override {
    // TODO(b/382117736): Add support for ROCM and SYCL.
    return tsl::Fingerprint64(xla::CudaName());
  }

  absl::string_view platform_name() const override { return xla::CudaName(); }

  absl::string_view platform_version() const override;

 private:
  int process_index_;

  se::Platform* platform_;
  xla::LocalClient* xla_client_;

  std::vector<PjRtDevice*> devices_;

  // Addressable devices indexed by core_id.
  std::vector<PjRtDevice*> addressable_devices_;
};

absl::StatusOr<std::unique_ptr<PjRtClient>> GetTfrtGpuClient(
    TfrtGpuClient::Options options);

}  // namespace xla

#endif  // XLA_PJRT_GPU_TFRT_TFRT_GPU_CLIENT_H_
