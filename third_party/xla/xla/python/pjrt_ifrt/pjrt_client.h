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
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/literal.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/ifrt/tuple.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt/value.h"
#include "xla/python/pjrt_ifrt/pjrt_compiler.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/logging.h"
#include "xla/xla_data.pb.h"

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
  struct CreateOptions {
    std::shared_ptr<xla::PjRtClient> pjrt_client;

    // KV store for sharing topology information. If present, PJRT-IFRT will do
    // its own topology exchange. If omitted, we will trust whatever topology
    // information the PJRT client reports.
    std::shared_ptr<xla::KeyValueStoreInterface> kv_store = nullptr;

    // Number of distributed processes. Ignored if kv_store is omitted.
    int num_processes = 1;

    // My process ID. Ignored if kv_store is omitted.
    int process_id = 0;

    absl::Duration get_local_topology_timeout = absl::Minutes(2);
    absl::Duration get_global_topology_timeout = absl::Minutes(5);

    // Device mapping to construct a global view consisting of both addressable
    // and non-addressable devices.
    //
    // If omitted, the PjRt client's device view will be used as-is.
    //
    // Currently supported only if `kv_store` is unspecified.
    struct GlobalDeviceMapping {
      // Device IDs to use for addressable devices exported by `pjrt_client`.
      // It must have the same number of addressable devices as
      // `pjrt_client`.
      absl::flat_hash_set<DeviceId> addressable_device_ids;

      // Mapping of device ID to process index for all processes. The local
      // process index is identified by the entry whose device ID matches one in
      // `addressable_device_ids`.
      absl::flat_hash_map<DeviceId, int> device_id_to_process_index;
    };
    std::optional<GlobalDeviceMapping> global_device_mapping;

    // Whether to sort devices by (process index, device ID). If false, sort
    // devices only by device ID.
    bool sort_devices_by_process_index = true;
  };

  static absl::StatusOr<std::unique_ptr<PjRtClient>> Create(
      CreateOptions options);

  // Creates a `Client` with a `PjRtClient`.
  // Dies if Create() fails.
  // Deprecated, use the overload that accepts `CreateOptions`.
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

  // For making Arrays with `dtype` as kString:
  //   (1) the `data` argument should point to an array of `absl::Cord`
  //   in major-to-minor order,
  //   (2) `byte_strides` are not supported, and non-`nullopt` values cause this
  //   function to fail.
  //   (3) only the `kImmutableDuringCall` semantics is supported currently.
  //   Fails for other values of `HostBufferSemantics`.
  absl::StatusOr<ArrayRef> MakeArrayFromHostBuffer(
      const void* data, DType dtype, Shape shape,
      std::optional<absl::Span<const int64_t>> byte_strides,
      ShardingRef sharding, HostBufferSemantics semantics,
      std::function<void()> on_done_with_host_buffer,
      tsl::RCReference<UserContext> user_context) override;

  absl::StatusOr<std::vector<ArrayRef>> MakeArraysFromHostBufferShards(
      absl::Span<MakeArraysFromHostBufferShardsSpec> specs,
      HostBufferSemantics semantics,
      tsl::RCReference<UserContext> user_context) override;

  absl::StatusOr<std::vector<ArrayRef>> MakeErrorArrays(
      const absl::Status& error, absl::Span<const ArraySpec> array_specs,
      tsl::RCReference<UserContext> user_context) override;

  absl::StatusOr<ArrayRef> AssembleArrayFromSingleDeviceArrays(
      DType dtype, Shape shape, ShardingRef sharding,
      absl::Span<ArrayRef> arrays, ArrayCopySemantics array_copy_semantics,
      SingleDeviceShardSemantics single_device_shard_semantics) override;

  absl::StatusOr<std::vector<ArrayRef>> CopyArrays(
      absl::Span<ArrayRef> arrays, std::optional<DeviceListRef> devices,
      std::optional<MemoryKind> memory_kind,
      ArrayCopySemantics semantics) override;

  absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> RemapArrays(
      const RemapPlan& plan, absl::Span<xla::ifrt::ArrayRef> arrays,
      ArrayCopySemantics semantics) override;

  Future<> GetReadyFuture(
      absl::Span<const tsl::RCReference<Value>> values) override;

  absl::StatusOr<tsl::RCReference<Tuple>> MakeTuple(
      absl::Span<tsl::RCReference<Value>> values) override;

  absl::string_view runtime_type() const override { return "pjrt_ifrt"; }

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

  const AttributeMap& Attributes() const override;

  int device_count() const override {
    DCHECK(this);
    return devices_.size();
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
  int process_index() const override { return my_process_index_; }

  absl::Span<Device* const> GetAllDevices() const override {
    DCHECK(this);
    return devices_;
  }

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override {
    DCHECK(this);
    return pjrt_client_->GetDefaultDeviceAssignment(num_replicas,
                                                    num_partitions);
  }
  absl::StatusOr<Device*> LookupDevice(DeviceId device_id) const override;

  absl::StatusOr<Device*> LookupAddressableDevice(
      int local_hardware_id) const override;

  DeviceListRef MakeDeviceList(
      absl::Span<Device* const> devices) const override;

  Compiler* GetDefaultCompiler() override {
    DCHECK(this);
    return &default_compiler_;
  }

  absl::StatusOr<std::shared_ptr<Topology>> GetTopologyForDevices(
      const DeviceListRef& devices) const override;

  absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> GetDefaultLayout(
      DType dtype, absl::Span<const int64_t> dims, Device* device,
      MemoryKind memory_kind) const override;

  absl::StatusOr<PjRtCompatibleDevice*> LookupPjRtDevice(
      xla::PjRtDevice* pjrt_device) const override;
  absl::StatusOr<PjRtCompatibleMemory*> LookupPjRtMemory(
      xla::PjRtMemorySpace* pjrt_memory) const override;

  // Returns the PjRt global device ID for the given IFRT device ID. This
  // succeeds only if the PjRt global device ID was available in `pjrt_client_`
  // or it has been discovered through topology exchange; in other words, it
  // also supports getting the PjRt global device ID for non-addressable IFRT
  // device IDs, unlike `xla::ifrt::PjRtDevice` that does not keep
  // `xla::PjRtDevice` around for non-addressable devices.
  //
  // Note that it does not yet support non-addressable IFRT device IDs created
  // by PjRt-IFRT with the global device mapping because there is no well-agreed
  // PjRt device ID allocation that PjRt-IFRT can assume.
  absl::StatusOr<xla::PjRtGlobalDeviceId> GetPjRtGlobalDeviceId(
      DeviceId device_id) const;

  // Transfer the given literal to the infeed queue.
  absl::Status TransferToInfeed(PjRtDevice* device,
                                const LiteralSlice& literal);

  // Transfer and return a value of the given shape from the outfeed queue.
  absl::Status TransferFromOutfeed(PjRtDevice* device,
                                   MutableBorrowingLiteral literal);

  tsl::RCReference<UserContext> CreateUserContext() override {
    return tsl::RCReference<UserContext>();
  }

  static char ID;  // NOLINT

 private:
  explicit PjRtClient(std::shared_ptr<xla::PjRtClient> pjrt_client);

  std::shared_ptr<xla::PjRtClient> pjrt_client_;
  PjRtCompiler default_compiler_;

  // My process ID used as an IFRT client.
  int my_process_index_;
  // Mapping from IFRT device ID to PjRt global device ID. Made for the devices
  // that are accessible via `pjrt_client_->devices()`.
  absl::flat_hash_map<DeviceId, xla::PjRtGlobalDeviceId>
      ifrt_device_id_to_pjrt_global_device_id_;

  AttributeMap attributes_;

  std::vector<std::unique_ptr<PjRtDevice>> owned_devices_;
  std::vector<std::unique_ptr<PjRtMemory>> owned_memories_;

  std::vector<Device*> devices_;
  std::vector<Device*> addressable_devices_;
  absl::flat_hash_map<xla::PjRtDevice*, PjRtDevice*> device_map_;
  absl::flat_hash_map<xla::PjRtMemorySpace*, PjRtMemory*> memory_map_;
  absl::flat_hash_map<DeviceId, PjRtDevice*> device_id_map_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_PJRT_CLIENT_H_
