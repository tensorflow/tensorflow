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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/tuple.h"
#include "xla/python/ifrt/value.h"
#include "xla/python/pjrt_ifrt/pjrt_compiler.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace ifrt {

class PjRtCompatibleArray;
class PjRtCompatibleDevice;
class PjRtCompatibleMemory;
class PjRtDevice;
class PjRtMemory;

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
  virtual absl::StatusOr<PjRtCompatibleDevice*> LookupPjRtDevice(
      xla::PjRtDevice* pjrt_device) const = 0;
  virtual absl::StatusOr<PjRtCompatibleMemory*> LookupPjRtMemory(
      xla::PjRtMemorySpace* pjrt_memory) const = 0;

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

  ~PjRtClient() override;

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
    return devices_;
  }
  absl::Span<Device* const> addressable_devices() const override {
    DCHECK(this);
    return addressable_devices_;
  }
  int process_index() const override { return pjrt_client_->process_index(); }
  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override {
    DCHECK(this);
    return pjrt_client_->GetDefaultDeviceAssignment(num_replicas,
                                                    num_partitions);
  }
  absl::StatusOr<Device*> LookupDevice(DeviceId device_id) const override;

  absl::StatusOr<Device*> LookupAddressableDevice(
      int local_hardware_id) const override;

  Compiler* GetDefaultCompiler() override {
    DCHECK(this);
    return &default_compiler_;
  }

  absl::StatusOr<std::shared_ptr<const xla::PjRtTopologyDescription>>
  GetTopologyForDevices(const DeviceList& devices) const override;

  absl::StatusOr<std::unique_ptr<xla::PjRtLayout>> GetDefaultLayoutForDevice(
      DType dtype, absl::Span<const int64_t> dims,
      Device* device) const override;

  absl::StatusOr<PjRtCompatibleDevice*> LookupPjRtDevice(
      xla::PjRtDevice* pjrt_device) const override;
  absl::StatusOr<PjRtCompatibleMemory*> LookupPjRtMemory(
      xla::PjRtMemorySpace* pjrt_memory) const override;

  // Transfer the given literal to the infeed queue.
  absl::Status TransferToInfeed(PjRtDevice* device,
                                const LiteralSlice& literal);

  // Transfer and return a value of the given shape from the outfeed queue.
  absl::Status TransferFromOutfeed(PjRtDevice* device,
                                   MutableBorrowingLiteral literal);

  static char ID;  // NOLINT

 private:
  explicit PjRtClient(std::shared_ptr<xla::PjRtClient> pjrt_client);

  std::shared_ptr<xla::PjRtClient> pjrt_client_;
  PjRtCompiler default_compiler_;

  std::vector<Device*> devices_;
  std::vector<Device*> addressable_devices_;
  absl::flat_hash_map<xla::PjRtDevice*, std::unique_ptr<PjRtDevice>>
      device_map_;
  absl::flat_hash_map<xla::PjRtMemorySpace*, std::unique_ptr<PjRtMemory>>
      memory_map_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_PJRT_CLIENT_H_
