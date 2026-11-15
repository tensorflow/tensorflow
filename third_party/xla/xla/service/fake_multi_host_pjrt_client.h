/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_FAKE_MULTI_HOST_PJRT_CLIENT_H_
#define XLA_SERVICE_FAKE_MULTI_HOST_PJRT_CLIENT_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/interpreter/interpreter_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/scoped_async_tracking_event.h"
#include "xla/runtime/chip_id.h"
#include "xla/runtime/device_id.h"
#include "xla/service/device_assignment.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

// Fake PjRtDevice with configurable addressability and execution counters.
class FakeMultiHostDevice : public PjRtDevice {
 public:
  FakeMultiHostDevice(PjRtClient* client, int id, bool is_addressable)
      : client_(client), id_(id), is_addressable_(is_addressable) {}

  PjRtClient* client() const override { return client_; }
  bool IsAddressable() const override { return is_addressable_; }
  GlobalDeviceId global_device_id() const override {
    return GlobalDeviceId(id_);
  }
  LocalChipId local_hardware_id() const override { return LocalChipId(id_); }
  absl::string_view ToString() const override { return "FakeMultiHostDevice"; }
  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
    return nullptr;
  }
  absl::Status TransferToInfeed(const LiteralSlice& literal) override {
    ++infeed_count_;
    return absl::OkStatus();
  }
  absl::Status TransferFromOutfeed(MutableBorrowingLiteral literal) override {
    ++outfeed_count_;
    return literal.CopyFrom(LiteralUtil::CreateR0<int32_t>(100 + id_));
  }
  absl::Span<PjRtMemorySpace* const> memory_spaces() const override {
    return is_addressable_ ? client_->memory_spaces()
                           : absl::Span<PjRtMemorySpace* const>{};
  }
  absl::StatusOr<PjRtMemorySpace*> default_memory_space() const override {
    if (!is_addressable_) {
      return absl::InternalError("No memory space on non-addressable device");
    }
    return client_->memory_spaces().front();
  }

  void RecordExecute() { ++execute_count_; }
  int execute_count() const { return execute_count_; }
  int infeed_count() const { return infeed_count_; }
  int outfeed_count() const { return outfeed_count_; }

 private:
  PjRtClient* client_;
  int id_;
  bool is_addressable_;
  int execute_count_ = 0;
  int infeed_count_ = 0;
  int outfeed_count_ = 0;
};

// Fake PjRtLoadedExecutable that records ExecuteSharded calls per device.
class FakeMultiHostLoadedExecutable : public PjRtLoadedExecutable {
 public:
  FakeMultiHostLoadedExecutable(PjRtClient* client,
                                std::shared_ptr<HloModule> module)
      : client_(client), module_(std::move(module)) {}

  PjRtClient* client() const override { return client_; }
  const DeviceAssignment& device_assignment() const override {
    return device_assignment_;
  }
  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return client_->addressable_devices();
  }
  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
    return std::vector<std::shared_ptr<HloModule>>{module_};
  }
  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options,
      std::optional<std::vector<Future<>>>& returned_futures) const override {
    return absl::UnimplementedError("Not implemented");
  }
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options, std::optional<Future<>>& returned_future,
      bool fill_future) const override {
    if (!device->IsAddressable() || argument_handles.size() != 1) {
      return absl::InternalError("Invalid ExecuteSharded call");
    }
    tensorflow::down_cast<FakeMultiHostDevice*>(device)->RecordExecute();
    if (fill_future) {
      returned_future = Future<>(absl::OkStatus());
    }
    std::vector<std::unique_ptr<PjRtBuffer>> outputs;
    outputs.push_back(std::make_unique<InterpreterLiteralWrapperBuffer>(
        client_, client_->memory_spaces().front(),
        LiteralUtil::CreateR0<int32_t>(10 +
                                       device->global_device_id().value())));
    return outputs;
  }
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options, std::optional<Future<>>& returned_future,
      bool fill_future) const override {
    return absl::UnimplementedError("Not implemented");
  }
  void Delete() override {}
  bool IsDeleted() const override { return false; }

 private:
  PjRtClient* client_;
  std::shared_ptr<HloModule> module_;
  DeviceAssignment device_assignment_{1, 1};
};

// Fake PjRtClient supporting a mix of addressable and non-addressable devices.
class FakeMultiHostClient : public PjRtClient {
 public:
  static constexpr absl::string_view kIdentityHlo =
      "ENTRY %id (x: s32[]) -> s32[] { ROOT %x = s32[] parameter(0) }";

  explicit FakeMultiHostClient(absl::Span<const bool> device_addressability) {
    for (int i = 0; i < device_addressability.size(); ++i) {
      owned_devices_.push_back(std::make_unique<FakeMultiHostDevice>(
          this, i, device_addressability[i]));
      devices_.push_back(owned_devices_.back().get());
      if (device_addressability[i]) {
        addressable_devices_.push_back(owned_devices_.back().get());
      }
    }
  }

  int process_index() const override { return 0; }
  int device_count() const override { return devices_.size(); }
  int addressable_device_count() const override {
    return addressable_devices_.size();
  }
  absl::Span<PjRtDevice* const> devices() const override { return devices_; }
  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }
  absl::Span<PjRtMemorySpace* const> memory_spaces() const override {
    return memory_spaces_;
  }
  PjRtPlatformId platform_id() const override { return platform_id_; }
  absl::string_view platform_name() const override { return platform_name_; }
  absl::string_view platform_version() const override { return "0.0.0"; }

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override {
    DeviceAssignment assignment(num_replicas, num_partitions);
    assignment.FillIota(0);
    return assignment;
  }

  absl::StatusOr<PjRtDevice*> LookupDevice(
      GlobalDeviceId global_device_id) const override {
    return devices_.at(global_device_id.value());
  }

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtMemorySpace* memory_space,
      const Layout* device_layout) override {
    return std::make_unique<InterpreterLiteralWrapperBuffer>(this, memory_space,
                                                             literal);
  }

  using PjRtClient::CompileAndLoad;
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      const XlaComputation& computation, CompileOptions options) override {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                     ParseAndReturnUnverifiedModule(kIdentityHlo));
    return std::make_unique<FakeMultiHostLoadedExecutable>(this,
                                                           std::move(module));
  }

  const FakeMultiHostDevice& device(int i) const { return *owned_devices_[i]; }

 private:
  std::string platform_name_ = "fake_multi_host";
  PjRtPlatformId platform_id_ = tsl::Fingerprint64(platform_name_);
  std::vector<std::unique_ptr<FakeMultiHostDevice>> owned_devices_;
  std::vector<PjRtDevice*> devices_;
  std::vector<PjRtDevice*> addressable_devices_;
  InterpreterMemorySpace memory_space_{this};
  std::vector<PjRtMemorySpace*> memory_spaces_{&memory_space_};
};

}  // namespace xla

#endif  // XLA_SERVICE_FAKE_MULTI_HOST_PJRT_CLIENT_H_
