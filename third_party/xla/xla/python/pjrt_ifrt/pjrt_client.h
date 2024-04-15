/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_PJRT_IFRT_PJRT_CLIENT_H_
#define XLA_PYTHON_PJRT_IFRT_PJRT_CLIENT_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/pjrt_ifrt/pjrt_compiler.h"
#include "xla/xla_data.pb.h"
#include "tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

class PjRtCompatibleArray;

// PjRt-compatible `Client` interface.
class PjRtCompatibleClient
    : public llvm::RTTIExtends<PjRtCompatibleClient, Client> {
 public:
  static constexpr int kPjRtBufferInlineSize = 1;
  using PjRtBuffers =
      absl::InlinedVector<std::shared_ptr<PjRtBuffer>, kPjRtBufferInlineSize>;

  // APIs that allow direct access to `xla::PjRtClient` for PjRt-only
  // operations.
  virtual xla::PjRtClient* pjrt_client() = 0;
  virtual std::shared_ptr<xla::PjRtClient> shared_ptr_pjrt_client() = 0;
  virtual absl::StatusOr<tsl::RCReference<PjRtCompatibleArray>> CreatePjRtArray(
      std::shared_ptr<PjRtBuffer> pjrt_buffer) = 0;
  virtual absl::StatusOr<tsl::RCReference<PjRtCompatibleArray>> CreatePjRtArray(
      Shape shape, PjRtBuffers pjrt_buffers) = 0;

  static char ID;  // NOLINT
};

// `Client` implementation that wraps `xla::PjRtClient`.
class PjRtClient final
    : public llvm::RTTIExtends<PjRtClient, PjRtCompatibleClient> {
 public:
  // Creates a `Client` with a `PjRtClient`.
  static std::unique_ptr<PjRtClient> Create(
      std::shared_ptr<xla::PjRtClient> pjrt_client);

  // PjRtCompatibleClient implementation.

  xla::PjRtClient* pjrt_client() override { return pjrt_client_.get(); }
  std::shared_ptr<xla::PjRtClient> shared_ptr_pjrt_client() override {
    return pjrt_client_;
  }
  absl::StatusOr<tsl::RCReference<PjRtCompatibleArray>> CreatePjRtArray(
      std::shared_ptr<PjRtBuffer> pjrt_buffer) override;
  absl::StatusOr<tsl::RCReference<PjRtCompatibleArray>> CreatePjRtArray(
      Shape shape, PjRtBuffers pjrt_buffers) override;

  // Client implementation.

  ~PjRtClient() override = default;

  absl::StatusOr<tsl::RCReference<Array>> MakeArrayFromHostBuffer(
      const void* data, DType dtype, Shape shape,
      std::optional<absl::Span<const int64_t>> byte_strides,
      std::shared_ptr<const Sharding> sharding,
      Client::HostBufferSemantics semantics,
      std::function<void()> on_done_with_host_buffer) override;

  absl::StatusOr<tsl::RCReference<Array>> AssembleArrayFromSingleDeviceArrays(
      Shape shape, std::shared_ptr<const Sharding> sharding,
      absl::Span<tsl::RCReference<Array>> arrays,
      ArrayCopySemantics semantics) override;

  absl::StatusOr<tsl::RCReference<Tuple>> MakeTuple(
      absl::Span<tsl::RCReference<Value>> values) override;

  absl::string_view runtime_type() const override {
    DCHECK(this);
    return PjRtRuntimeTypeString(pjrt_client_->runtime_type());
  }

  absl::string_view platform_name() const override {
    DCHECK(this);
    return pjrt_client_->platform_name();
  }
  absl::string_view platform_version() const override {
    DCHECK(this);
    return pjrt_client_->platform_version();
  }
  PlatformId platform_id() const override {
    DCHECK(this);
    return pjrt_client_->platform_id();
  }

  absl::flat_hash_map<std::string, ClientAttribute> attributes() const override;

  int device_count() const override {
    DCHECK(this);
    return pjrt_client_->device_count();
  }
  int addressable_device_count() const override {
    DCHECK(this);
    return pjrt_client_->addressable_device_count();
  }
  absl::Span<Device* const> devices() const override {
    DCHECK(this);
    return pjrt_client_->devices();
  }
  absl::Span<Device* const> addressable_devices() const override {
    DCHECK(this);
    return pjrt_client_->addressable_devices();
  }
  int process_index() const override { return pjrt_client_->process_index(); }
  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override {
    DCHECK(this);
    return pjrt_client_->GetDefaultDeviceAssignment(num_replicas,
                                                    num_partitions);
  }
  absl::StatusOr<Device*> LookupDevice(int device_id) const override {
    DCHECK(this);
    return pjrt_client_->LookupDevice(device_id);
  }

  absl::StatusOr<Device*> LookupAddressableDevice(
      int local_hardware_id) const override {
    DCHECK(this);
    return pjrt_client_->LookupAddressableDevice(local_hardware_id);
  }

  Compiler* GetDefaultCompiler() override {
    DCHECK(this);
    return &default_compiler_;
  }

  absl::StatusOr<std::shared_ptr<const xla::PjRtTopologyDescription>>
  GetTopologyForDevices(const DeviceList& devices) const override;

  absl::StatusOr<std::unique_ptr<xla::PjRtLayout>> GetDefaultLayoutForDevice(
      DType dtype, absl::Span<const int64_t> dims,
      Device* device) const override;

  static char ID;  // NOLINT

 private:
  explicit PjRtClient(std::shared_ptr<xla::PjRtClient> pjrt_client)
      : pjrt_client_(std::move(pjrt_client)), default_compiler_(this) {}

  std::shared_ptr<xla::PjRtClient> pjrt_client_;
  PjRtCompiler default_compiler_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_PJRT_CLIENT_H_
