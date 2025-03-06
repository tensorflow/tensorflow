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

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
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
#include "xla/pjrt/pjrt_stream_executor_device_description.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/utils.h"
#include "xla/service/hlo.pb.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

class TfrtGpuMemorySpace : public PjRtMemorySpace {
 public:
  TfrtGpuMemorySpace(int id, PjRtDevice* device, absl::string_view kind,
                     int kind_id);

  PjRtClient* client() const override { return device_->client(); }

  absl::Span<PjRtDevice* const> devices() const override {
    return absl::Span<PjRtDevice* const>(&device_, device_ != nullptr ? 1 : 0);
  }

  int id() const override { return id_; }

  absl::string_view kind() const override { return kind_; }

  int kind_id() const override { return kind_id_; }

  absl::string_view DebugString() const override { return debug_string_; }

  absl::string_view ToString() const override { return to_string_; }

 private:
  int id_;
  PjRtDevice* device_ = nullptr;
  absl::string_view kind_;
  int kind_id_;
  std::string debug_string_;
  std::string to_string_;
};

class TfrtGpuDeviceMemorySpace : public TfrtGpuMemorySpace {
 public:
  static constexpr absl::string_view kKind = "device";
  static const int kKindId;

  TfrtGpuDeviceMemorySpace(int id, PjRtDevice* device);
};

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

  void SetClient(PjRtClient* client) {
    CHECK(client_ == nullptr);
    client_ = client;

    // We have to define debug_string_ and to_string_ here, because
    // platform_name() requires client_ to be set.
    std::string device_name =
        absl::StrCat(MakeAsciiTitlecase(client_->platform_name()), "Device");
    description_.SetDebugString(
        absl::StrCat(client_->platform_name(), ":", id()));
    description_.SetToString(absl::StrCat(device_name, "(id=", id(), ")"));
  }

  const PjRtStreamExecutorDeviceDescription& description() const override {
    return description_;
  }

  PjRtClient* client() const override { return client_; }

  bool IsAddressable() const override {
    return process_index() == client()->process_index();
  }

  int id() const override { return id_; }

  PjRtLocalDeviceId local_device_id() const override {
    return local_device_id_;
  }

  // Used as `device_ordinal`.
  PjRtLocalHardwareId local_hardware_id() const override {
    return local_hardware_id_;
  }

  absl::Status TransferToInfeed(const LiteralSlice& literal) override;

  absl::Status TransferFromOutfeed(MutableBorrowingLiteral literal) override;

  void AttachMemorySpace(PjRtMemorySpace* memory_space,
                         bool is_default = false);

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

  absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind_id(int id) const;

  absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind(
      absl::string_view kind) const override;

  absl::StatusOr<PjRtMemorySpace*> default_memory_space() const override;

  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
    return nullptr;
  }

 private:
  friend class TfrtGpuClient;

  int id_;
  PjRtClient* client_ = nullptr;
  PjRtLocalDeviceId local_device_id_;
  PjRtLocalHardwareId local_hardware_id_;

  absl::InlinedVector<PjRtMemorySpace*, 1> memory_spaces_;
  absl::flat_hash_map<int, PjRtMemorySpace*> memory_spaces_by_kind_id_;

  PjRtStreamExecutorDeviceDescription description_;
  PjRtMemorySpace* default_memory_space_ = nullptr;
};

class TfrtGpuClient final : public PjRtClient {
 public:
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

  absl::string_view platform_version() const override {
    return platform_version_;
  }

 private:
  int process_index_;

  xla::LocalClient* xla_client_;

  const std::string platform_version_;

  // Includes all devices, including non-local devices on multi-host platforms.
  std::vector<std::unique_ptr<TfrtGpuDevice>> owned_devices_;
  // Pointers to `owned_devices_`.
  std::vector<PjRtDevice*> devices_;
  // Maps Device::id() to the corresponding Device. Includes all devices.
  absl::flat_hash_map<PjRtGlobalDeviceId, TfrtGpuDevice*> id_to_device_;
  // Local devices indexed by local device ordinal.
  std::vector<PjRtDevice*> addressable_devices_;

  // Addressable memory spaces.
  std::vector<std::unique_ptr<PjRtMemorySpace>> owned_memory_spaces_;
  // Pointers to `owned_memory_spaces_`.
  std::vector<PjRtMemorySpace*> memory_spaces_;
};

absl::StatusOr<std::unique_ptr<PjRtClient>> GetTfrtGpuClient(
    const GpuClientOptions& options);

}  // namespace xla

#endif  // XLA_PJRT_GPU_TFRT_TFRT_GPU_CLIENT_H_
