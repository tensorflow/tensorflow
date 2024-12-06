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

#include "xla/pjrt/plugin/stablehlo_reference/client_cpp_pjrt.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/plugin/stablehlo_reference/buffer.h"
#include "xla/pjrt/plugin/stablehlo_reference/device.h"
#include "xla/pjrt/plugin/stablehlo_reference/executable.h"
#include "xla/pjrt/plugin/stablehlo_reference/logging.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/fingerprint.h"

#define UNIMPLEMENTED(name) \
  xla::Unimplemented("MlirPjrtBuffer::" #name " is not implemented")

namespace mlir::stablehlo {

const xla::PjRtPlatformId kStablehloReferenceBackendId =
    tsl::Fingerprint64(kStablehloReferenceBackendName);

class StablehloReferencePjrtClient : public xla::PjRtClient {
 public:
  StablehloReferencePjrtClient()
      : xla::PjRtClient(),
        owned_devices_(),
        devices_(),
        owned_memory_space_(),
        memory_space_(nullptr) {
    // Init device and memory space.
    TRACE_ME_MEMBER;
    owned_devices_.push_back(GetStablehloReferenceDevice(this));
    devices_.push_back(owned_devices_.back().get());
    owned_memory_space_ = std::make_unique<xla::UnpinnedHostMemorySpace>(
        /*id=*/0, devices_.front());
    memory_space_ = owned_memory_space_.get();
    AttachStablehloReferenceMemorySpace(devices_.front(), memory_space_);
  }

  ~StablehloReferencePjrtClient() override {};

  absl::string_view platform_name() const override {
    TRACE_ME_MEMBER;
    return kStablehloReferenceBackendName;
  }
  int process_index() const override {
    TRACE_ME_MEMBER;
    return 0;
  }

  int device_count() const override {
    TRACE_ME_MEMBER;
    return devices_.size();
  }

  int addressable_device_count() const override {
    TRACE_ME_MEMBER;
    return devices_.size();
  }

  absl::Span<xla::PjRtDevice* const> devices() const override {
    TRACE_ME_MEMBER;
    return devices_;
  }
  absl::Span<xla::PjRtDevice* const> addressable_devices() const override {
    TRACE_ME_MEMBER;
    return devices_;
  }

  absl::Span<xla::PjRtMemorySpace* const> memory_spaces() const override {
    TRACE_ME_MEMBER;
    return absl::MakeSpan(&memory_space_, 1);
  }

  // Return an ID that identifies the platform via tsl fingerprint.
  xla::PjRtPlatformId platform_id() const override {
    TRACE_ME_MEMBER;
    return kStablehloReferenceBackendId;
  }

  // Returns a string containing human-readable, platform-specific version
  // info (e.g. the CUDA version on GPU or libtpu version on Cloud TPU).
  absl::string_view platform_version() const override {
    TRACE_ME_MEMBER;
    return "StableHLO Reference v0.1";
  }

  /////////////
  // Device
  /////////////

  // Lookup any PjRtDevice for a given PjRtDevice::id().
  // TODO: Should this be a base class? I.e. why doesn't the base client have
  // a vector a device pointers?
  absl::StatusOr<xla::PjRtDevice*> LookupDevice(
      xla::PjRtGlobalDeviceId global_device_id) const override {
    TRACE_ME_MEMBER;
    for (auto device : devices_) {
      if (device->global_device_id() == global_device_id) {
        return device;
      }
    }
    // TODO: This error should be a base class method since its used in tests.
    return xla::InvalidArgument("No matching device found for device_id %d",
                                global_device_id.value());
  }

  absl::StatusOr<xla::PjRtDevice*> LookupAddressableDevice(
      xla::PjRtLocalDeviceId local_device_id) const override {
    TRACE_ME_MEMBER;

    for (auto* device : addressable_devices()) {
      if (local_device_id == device->local_device_id()) {
        return device;
      }
    }
    return xla::InvalidArgument(
        "No matching device found for local_device_id %d",
        local_device_id.value());
  }

  absl::StatusOr<xla::DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override {
    TRACE_ME_MEMBER;
    xla::DeviceAssignment assignment(num_replicas, num_partitions);
    for (int64_t i = 0; i < num_replicas; ++i) {
      for (int64_t j = 0; j < num_partitions; ++j) {
        auto idx = (i + (j * num_replicas)) % devices_.size();
        assignment(i, j) = devices_[idx]->global_device_id().value();
      }
    }
    return assignment;
  }

  /////////////////
  // Buffer methods
  /////////////////

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> CreateErrorBuffer(
      absl::Status error, const xla::Shape& shape,
      xla::PjRtDevice* device) override {
    // Prefer memory space implementation, device holding buffer is
    // deprecated.
    return CreateErrorBuffer(error, shape,
                             device->default_memory_space().value_or(nullptr));
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> CreateErrorBuffer(
      absl::Status error, const xla::Shape& shape,
      xla::PjRtMemorySpace* memory) override {
    return UNIMPLEMENTED(CreateErrorBuffer);
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> CreateUninitializedBuffer(
      const xla::Shape& shape, xla::PjRtDevice* device) override {
    TRACE_ME_MEMBER;
    return CreateMlirBufferUninitizlied(
        shape, device->default_memory_space().value_or(nullptr));
  }

  absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(absl::Span<const xla::Shape> shapes,
                                    xla::PjRtDevice* device) override {
    TRACE_ME_MEMBER;
    return CreateBuffersForAsyncHostToDevice(
        shapes, device->default_memory_space().value_or(nullptr));
  }

  absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(
      absl::Span<const xla::Shape> shapes,
      xla::PjRtMemorySpace* memory_space) override {
    TRACE_ME_MEMBER;
    return UNIMPLEMENTED(CreateBuffersForAsyncHostToDevice);
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> BufferFromHostBuffer(
      const void* data, xla::PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      xla::PjRtDevice* device) override {
    TRACE_ME_MEMBER;
    return BufferFromHostBuffer(
        data, type, dims, byte_strides, host_buffer_semantics,
        std::move(on_done_with_host_buffer),
        device->default_memory_space().value_or(nullptr), nullptr);
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> BufferFromHostBuffer(
      const void* data, xla::PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      xla::PjRtDevice* device, const xla::Layout* device_layout) override {
    TRACE_ME_MEMBER;
    return BufferFromHostBuffer(
        data, type, dims, byte_strides, host_buffer_semantics,
        std::move(on_done_with_host_buffer),
        device->default_memory_space().value_or(nullptr), device_layout);
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> BufferFromHostBuffer(
      const void* data, xla::PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      xla::PjRtMemorySpace* memory_space,
      const xla::Layout* device_layout) override {
    TRACE_ME_MEMBER;
    // Buffer to Literal
    auto shape = xla::ShapeUtil::MakeShape(type, dims);
    auto literal =
        xla::BorrowingLiteral(reinterpret_cast<const char*>(data), shape);
    auto buffer = CreateMlirBufferFromLiteral(literal, memory_space);
    if (on_done_with_host_buffer) {
      // If host is awaiting the result, must call this function.
      std::move(on_done_with_host_buffer)();
      on_done_with_host_buffer = nullptr;
    }
    return buffer;
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> BufferFromHostLiteral(
      const xla::LiteralSlice& literal, xla::PjRtDevice* device) override {
    TRACE_ME_MEMBER;
    return CreateMlirBufferFromLiteral(
        literal, device->default_memory_space().value_or(nullptr));
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> BufferFromHostLiteral(
      const xla::LiteralSlice& literal,
      xla::PjRtMemorySpace* memory_space) override {
    TRACE_ME_MEMBER;
    return CreateMlirBufferFromLiteral(literal, memory_space);
  }

  ///////////
  // Compile
  ///////////
  absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> Compile(
      mlir::ModuleOp module, xla::CompileOptions options) override {
    TRACE_ME_MEMBER;
    return mlir::stablehlo::StablehloReferenceCompile(
        module, GetDefaultDeviceAssignment(1, devices_.size()).value(), this);
  }

  // Compile `computation` with given `options`.
  absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> Compile(
      const xla::XlaComputation& computation,
      xla::CompileOptions options) override {
    TRACE_ME_MEMBER;
    return mlir::stablehlo::StablehloReferenceCompile(
        computation.proto(),
        GetDefaultDeviceAssignment(1, devices_.size()).value(), this);
  }

 private:
  std::vector<std::unique_ptr<xla::PjRtDevice>> owned_devices_;
  std::vector<xla::PjRtDevice*> devices_;
  std::unique_ptr<xla::PjRtMemorySpace> owned_memory_space_;
  xla::PjRtMemorySpace* memory_space_;
};  // end class

std::unique_ptr<xla::PjRtClient> CreateStablehloReferencePjrtClient() {
  SetupLogLevelFromEnv();
  return std::make_unique<StablehloReferencePjrtClient>();
}

}  // namespace mlir::stablehlo
