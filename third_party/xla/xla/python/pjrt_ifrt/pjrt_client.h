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

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/literal.h"
#include "xla/pjrt/distributed/client.h"
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
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/ifrt/tuple.h"
#include "xla/python/ifrt/value.h"
#include "xla/python/pjrt_ifrt/pjrt_compiler.h"
#include "xla/python/pjrt_ifrt/transfer_server_interface.h"
#include "xla/runtime/device_id.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/env.h"
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

  // Creates an IFRT `PjRtCompatibleArray` from `PjRtBuffer`(s).
  //
  // Most array properties will be inferred from the input `PjRtBuffer`(s),
  // except for the layout's defaultness that is absent information at the PjRt
  // level.
  //
  // `has_custom_layout` indicates that the layout of the input `PjRtBuffer`(s)
  // is intended to be a user-chosen custom layout, and
  // `PjRtCompatibleArray::pjrt_layout()` should return a non-null value.
  // Treating a default layout as a custom layout is typically allowed in PjRt
  // if their concrete layouts match, but it may not pass a strict check that
  // unconditionally says a default layout != any non-default layout designed
  // for portability. Thus, it is useful for the caller to provide as accurate
  // information as possible.
  virtual absl::StatusOr<tsl::RCReference<PjRtCompatibleArray>> CreatePjRtArray(
      std::shared_ptr<PjRtBuffer> pjrt_buffer, bool has_custom_layout) = 0;
  virtual absl::StatusOr<tsl::RCReference<PjRtCompatibleArray>> CreatePjRtArray(
      Shape shape, PjRtBuffers pjrt_buffers, bool has_custom_layout) = 0;

  // Temporary overloads for API transition.
  absl::StatusOr<tsl::RCReference<PjRtCompatibleArray>> CreatePjRtArray(
      std::shared_ptr<PjRtBuffer> pjrt_buffer);
  absl::StatusOr<tsl::RCReference<PjRtCompatibleArray>> CreatePjRtArray(
      Shape shape, PjRtBuffers pjrt_buffers);

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
  static constexpr absl::string_view kRuntimeType = "pjrt_ifrt";

  struct CreateOptions {
    std::shared_ptr<xla::PjRtClient> pjrt_client;

    // TODO: mwhittaker - Remove kv_store; it is subsumed by distributed_client.
    std::shared_ptr<xla::DistributedRuntimeClient> distributed_client = nullptr;

    // KV store for coordinating cross-host device transfers and sharing
    // topology information. If present and `use_kv_store_for_topology_exchange`
    // is true, PJRT-IFRT will do its own topology exchange. If omitted or
    // `use_kv_store_for_topology_exchange` is false, we will trust whatever
    // topology information the PJRT client reports.
    std::shared_ptr<xla::KeyValueStoreInterface> kv_store = nullptr;

    // If true, use the KV store for topology exchange. Ignored if kv_store is
    // not provided.
    bool use_kv_store_for_topology_exchange = true;

    // Number of distributed processes. Ignored if kv_store is omitted.
    int num_processes = 1;

    // My process ID. Ignored if kv_store is omitted.
    int process_id = 0;

    absl::Duration get_local_topology_timeout = absl::Minutes(2);
    absl::Duration get_global_topology_timeout = absl::Minutes(5);
    absl::Duration cross_host_transfer_timeout = absl::Minutes(1);

    std::function<absl::StatusOr<std::unique_ptr<TransferServerInterface>>(
        std::shared_ptr<xla::PjRtClient>)>
        transfer_server_factory;

    // If true, force DCN-based cross-host transfers even when the PJRT plugin
    // supports cross-host transfers.
    bool force_dcn_cross_host_transfers = false;

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
      std::shared_ptr<PjRtBuffer> pjrt_buffer, bool has_custom_layout) override;
  absl::StatusOr<tsl::RCReference<PjRtCompatibleArray>> CreatePjRtArray(
      Shape shape, PjRtBuffers pjrt_buffers, bool has_custom_layout) override;

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
      ShardingRef sharding, LayoutRef layout, HostBufferSemantics semantics,
      std::function<void()> on_done_with_host_buffer) override;
  // Expose the base class's `MakeArrayFromHostBuffer` overloads.
  using xla::ifrt::Client::MakeArrayFromHostBuffer;

  absl::StatusOr<std::vector<ArrayRef>> MakeArraysFromHostBufferShards(
      absl::Span<MakeArraysFromHostBufferShardsSpec> specs,
      HostBufferSemantics semantics) override;

  absl::StatusOr<std::vector<ArrayRef>> MakeErrorArrays(
      const absl::Status& error,
      absl::Span<const ArraySpec> array_specs) override;

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

  absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> ReshardArrays(
      absl::Span<ArrayRef> arrays, absl::Span<const ArraySpec> specs,
      ArrayCopySemantics semantics) override;

  tsl::Future<> GetReadyFuture(absl::Span<const ValueRef> values) override;

  absl::StatusOr<tsl::RCReference<Tuple>> MakeTuple(
      absl::Span<ValueRef> values) override;

  void CancelExecution(
      xla::ifrt::LoadedExecutable::CancellationHandle cancellation_handle,
      absl::Status error) override {}

  absl::string_view runtime_type() const override { return kRuntimeType; }

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

  absl::StatusOr<DeviceListRef> MakeDeviceList(
      absl::Span<Device* const> devices) const override;

  Compiler* GetDefaultCompiler() override {
    DCHECK(this);
    return &default_compiler_;
  }

  absl::StatusOr<std::shared_ptr<Topology>> GetTopologyForDevices(
      const DeviceListRef& devices) const override;

  absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> GetDefaultPjRtLayout(
      DType dtype, absl::Span<const int64_t> dims, Device* device,
      MemoryKind memory_kind) const override;
  absl::StatusOr<CustomLayoutRef> GetDefaultLayout(
      DType dtype, const Shape& shape,
      const ShardingRef& sharding) const override;

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

  // Returns the latest set of incarnation ids for every task.
  absl::StatusOr<absl::flat_hash_map<int, IncarnationId>> Incarnations() const;

  absl::StatusOr<std::unique_ptr<xla::ifrt::DeviceAttributeSubscription>>
  SubscribeToAttributeChanges(
      absl::Span<xla::ifrt::Device* const> devices,
      std::optional<absl::Span<const std::string>> attribute_names,
      xla::ifrt::OnDeviceAttributeChangeCallback callback) override {
    return absl::UnimplementedError(
        "SubscribeToAttributeChanges is not implemented in PjRtClient.");
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

  // Cached concrete default layouts.
  mutable absl::Mutex default_layout_cache_mu_;
  mutable absl::flat_hash_map<
      std::tuple<DType, std::vector<int64_t>, MemoryKind>,
      absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>>>
      default_layout_cache_ ABSL_GUARDED_BY(default_layout_cache_mu_);

  // Copies arrays from source to destination devices when at least one of the
  // (source, destination) pairs is cross-host.
  absl::StatusOr<std::vector<ArrayRef>> CopyArraysForCrossHost(
      absl::Span<ArrayRef> arrays, DeviceListRef src_devices,
      DeviceListRef dst_devices, std::optional<MemoryKind> memory_kind,
      ArrayCopySemantics semantics);

  // Extracts receive descriptors from a key-value store and sends buffers to a
  // remote device. This is used when the backend does not implement the
  // CrossHostSendBuffers API.
  absl::Status CrossHostSendBuffers(
      std::vector<PjRtBuffer*> buffers,
      const std::vector<CrossHostTransferKey>& keys);

  // Populates a key-value store with receive descriptors and places buffers
  // from a cross-host send onto device. This is used when the backend does not
  // implement the CrossHostReceiveBuffers API.
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  CrossHostReceiveBuffers(absl::Span<const xla::Shape> shapes,
                          xla::PjRtDevice* device,
                          std::vector<CrossHostTransferKey> keys);

  // Copies arrays from source to destination devices when at least one of the
  // (source, destination) pairs is cross-host using an experimental DCN
  // transfer library. Called when the PjRt backend does not support
  // `CopyArraysForCrossHost`.
  absl::StatusOr<std::vector<ArrayRef>> CopyArraysForCrossHostFallback(
      absl::Span<ArrayRef> arrays, DeviceListRef src_devices,
      DeviceListRef dst_devices, std::optional<MemoryKind> memory_kind);

  // Creates a unique identifier for each cross-host transfer. Every process
  // must call it, regardless of whether it participates in the cross-host
  // transfer, so that the returned value must be the same in all processes.
  CrossHostTransferKey CreateNewTransferKey();

  // If true, the backend implements the cross-host transfer APIs.
  bool pjrt_supports_cross_host_transfers_ = false;

  // If true, force DCN-based cross-host transfers even when the backend
  // supports cross-host transfers.
  bool force_dcn_cross_host_transfers_ = false;

  absl::Status WatchGlobalProcessInfo(xla::CoordinationServiceAgent& agent);

  std::atomic<int64_t> next_transfer_key_ = 0;
  std::shared_ptr<xla::DistributedRuntimeClient> distributed_client_;
  std::shared_ptr<xla::KeyValueStoreInterface> kv_store_;
  absl::Duration cross_host_transfer_timeout_;
  absl::Mutex transfer_server_mu_;
  std::function<absl::StatusOr<std::unique_ptr<TransferServerInterface>>(
      std::shared_ptr<xla::PjRtClient>)>
      transfer_server_factory_;
  std::optional<std::unique_ptr<TransferServerInterface>> transfer_server_;
  absl::Status InitializeTransferServer()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(transfer_server_mu_);

  // Note that global_process_info_thread_'s destructor will block until the
  // thread has stopped. Because it is the last field, we know the thread won't
  // access any other fields that have already been destructed.
  absl::Mutex shutting_down_mu_;
  bool shutting_down_ ABSL_GUARDED_BY(shutting_down_mu_) = false;
  std::unique_ptr<tsl::Thread> global_process_info_thread_;

  friend class PjRtClientPeer;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_PJRT_CLIENT_H_
