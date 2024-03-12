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

#ifndef XLA_PJRT_PJRT_C_API_CLIENT_H_
#define XLA_PJRT_PJRT_C_API_CLIENT_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "xla/client/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/framework/allocator.h"

namespace xla {

class PjRtCApiClient;

class PjRtCApiDeviceDescription : public PjRtDeviceDescription {
 public:
  PjRtCApiDeviceDescription(const PJRT_Api* c_api,
                            PJRT_DeviceDescription* device_description);

  int id() const override;

  int process_index() const override;

  absl::string_view device_kind() const override;

  absl::string_view DebugString() const override;

  absl::string_view ToString() const override;

  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override;

 private:
  const PJRT_Api* c_api_;
  // `device_description_` is owned by the `PJRT_Client` wrapped by `client_`
  PJRT_DeviceDescription* device_description_;
  // Device specific attributes with corresponding values.
  absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute> attributes_;

  // Initializes device specific attributes.
  void InitAttributes();
};

class PjRtCApiMemorySpace : public PjRtMemorySpace {
 public:
  explicit PjRtCApiMemorySpace(PJRT_Memory* c_memory, PjRtCApiClient* client)
      : client_(client), c_memory_(c_memory) {}

  PjRtClient* client() const override;

  absl::Span<PjRtDevice* const> devices() const override { return devices_; }

  int id() const override;

  absl::string_view memory_space_kind() const override;

  absl::string_view DebugString() const override;

  absl::string_view ToString() const override;

  const PJRT_Api* pjrt_c_api() const;

  PJRT_Memory* c_memory() const { return c_memory_; }

 private:
  friend class PjRtCApiClient;

  PjRtCApiClient* client_;
  PJRT_Memory* c_memory_;
  std::vector<PjRtDevice*> devices_;
};

class PjRtCApiDevice : public PjRtDevice {
 public:
  explicit PjRtCApiDevice(PJRT_Device* device, PjRtCApiClient* client);

  PjRtClient* client() const override;

  bool IsAddressable() const override;

  int local_hardware_id() const override;
  PjRtLocalHardwareId local_hardware_id_typed() const override;

  Status TransferToInfeed(const LiteralSlice& literal) override {
    return Unimplemented("PJRT C API does not support TransferToInfeed");
  }

  Status TransferFromOutfeed(MutableBorrowingLiteral literal) override {
    return Unimplemented("PJRT C API does not support TransferFromOutfeed");
  }

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override {
    return memory_spaces_;
  }

  StatusOr<PjRtMemorySpace*> default_memory_space() const override;

  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
    LOG(FATAL) << "PJRT C API does not support CreateAsyncTrackingEvent";
    return nullptr;
  }

  PJRT_Device* c_device() const { return device_; }

  const PjRtCApiDeviceDescription& description() const override {
    return description_;
  }

  StatusOr<tsl::AllocatorStats> GetAllocatorStats() const override;

 private:
  friend class PjRtCApiClient;

  PjRtCApiClient* client_ = nullptr;
  // `device_` is owned by the `PJRT_Client` wrapped by `client_`
  PJRT_Device* device_;
  PjRtCApiDeviceDescription description_;
  std::vector<PjRtMemorySpace*> memory_spaces_;
};

class PjRtCApiCompiler : public PjRtCompiler {
 public:
  explicit PjRtCApiCompiler(const PJRT_Api* c_api) : c_api_(c_api) {}

  StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, const XlaComputation& computation,
      const PjRtTopologyDescription& topology, PjRtClient* client) override;

  StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, mlir::ModuleOp module,
      const PjRtTopologyDescription& topology, PjRtClient* client) override;

 private:
  const PJRT_Api* c_api_;
};

class PjRtCApiTopologyDescription : public PjRtTopologyDescription {
 public:
  // `owned` indicates whether this PjRtCApiTopologyDescription should take
  // ownership of `c_topology`, i.e., if owned is true,
  // PJRT_TopologyDescription_Destroy will be called on `c_topology` when this
  // PjRtCApiTopologyDescription is destroyed.
  PjRtCApiTopologyDescription(const PJRT_Api* c_api,
                              PJRT_TopologyDescription* c_topology, bool owned);

  PjRtPlatformId platform_id() const override {
    CHECK(false) << "PJRT C API does not support platform_id.";
  }

  absl::string_view platform_name() const override;

  absl::string_view platform_version() const override;

  std::optional<PjRtCompiler*> compiler() const override {
    return compiler_.get();
  }

  PJRT_TopologyDescription* c_topology() const { return c_topology_; }

  std::vector<std::unique_ptr<const PjRtDeviceDescription>> DeviceDescriptions()
      const override;

  absl::StatusOr<std::string> Serialize() const override;

  // Returns vendor specific attributes about the topology.
  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }

 private:
  std::unique_ptr<PjRtCApiCompiler> compiler_;
  const PJRT_Api* c_api_;
  // nullptr iff the PJRT_TopologyDescription isn't owned by this wrapper
  // (i.e. by the caller).
  std::unique_ptr<PJRT_TopologyDescription,
                  ::pjrt::PJRT_TopologyDescriptionDeleter>
      owned_c_topology_;
  PJRT_TopologyDescription* c_topology_;
  // Device specific attributes with corresponding values.
  absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute> attributes_;

  // Initializes device specific attributes.
  void InitAttributes();
};

class PjRtCApiClient : public PjRtClient {
 public:
  PjRtCApiClient(
      const PJRT_Api* c_api, PJRT_Client* c_client,
      std::unique_ptr<::pjrt::PJRT_KeyValueCallbackData> kv_callback_data);

  int process_index() const override;

  int device_count() const override;
  int addressable_device_count() const override;

  absl::Span<PjRtDevice* const> devices() const override;
  absl::Span<PjRtDevice* const> addressable_devices() const override;

  StatusOr<PjRtDevice*> LookupDevice(int device_id) const override;
  StatusOr<PjRtDevice*> LookupDevice(
      PjRtGlobalDeviceId global_device_id) const override;

  StatusOr<PjRtDevice*> LookupAddressableDevice(
      int local_hardware_id) const override;
  StatusOr<PjRtDevice*> LookupAddressableDevice(
      PjRtLocalDeviceId local_device_id) const override;

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

  PjRtPlatformId platform_id() const override { return platform_id_; }

  absl::string_view platform_name() const override { return platform_name_; };

  absl::string_view platform_version() const override;

  std::optional<PjRtPluginAttributes> plugin_attributes() const override {
    return PjRtPluginAttributes{c_api_->pjrt_api_version.major_version,
                                c_api_->pjrt_api_version.minor_version};
  }

  // TODO(b/244756954): Rethink this function altogether
  PjRtRuntimeType runtime_type() const override {
    return PjRtRuntimeType::kTfrt;
  }

  StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis()
      const override {
    return Unimplemented("PJRT C API does not support GetHloCostAnalysis");
  }

  StatusOr<Layout> GetDefaultLayout(PrimitiveType element_type,
                                    absl::Span<const int64_t> dims) override {
    // TODO(skyewm): implement
    return Unimplemented("PJRT C API does not support GetDefaultLayout");
  }

  StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override;

  StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      mlir::ModuleOp module, CompileOptions options) override;

  // `PjRtCApiClient::DeserializeExecutable()` ignores `CompileOptions` arg
  StatusOr<std::unique_ptr<PjRtLoadedExecutable>> DeserializeExecutable(
      absl::string_view serialized,
      std::optional<CompileOptions> options) override;

  StatusOr<std::unique_ptr<PjRtBuffer>> CreateUninitializedBuffer(
      const Shape& shape, PjRtDevice* device) override {
    return Unimplemented(
        "PJRT C API does not support CreateUninitializedBuffer");
  }

  StatusOr<const PjRtTopologyDescription*> GetTopologyDescription()
      const override;

  StatusOr<std::unique_ptr<AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(absl::Span<const Shape> shapes,
                                    PjRtDevice* device) override {
    return Unimplemented(
        "PJRT C API does not support CreateBuffersForAsyncHostToDevice");
  }

  absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(absl::Span<const Shape> shapes,
                                    PjRtMemorySpace* memory_space) override {
    return Unimplemented(
        "PJRT C API does not support CreateBuffersForAsyncHostToDevice");
  }

  StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtDevice* device) override;

  StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtDevice* device, const Layout* device_layout) override;

  StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtMemorySpace* memory_space, const Layout* device_layout) override;

  StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtDevice* device) override {
    return Unimplemented("PJRT C API does not support BufferFromHostLiteral");
  }

  StatusOr<std::unique_ptr<PjRtBuffer>> CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtDevice* device,
      std::function<void()> on_delete_callback,
      std::optional<std::intptr_t> stream) override;

  StatusOr<std::uintptr_t> UnsafeBufferPointer(PjRtBuffer* buffer) override;

  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffers(absl::Span<const Shape> shapes,
                              PjRtDevice* device,
                              PjRtCrossHostRecvNotifier notifier) override {
    return Unimplemented(
        "PJRT C API does not support MakeCrossHostReceiveBuffers");
  }

  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffersForGather(
      absl::Span<const Shape> shapes, std::vector<GatherDetails> gather_details,
      PjRtDevice* device, PjRtCrossHostRecvNotifier notifier) override {
    return Unimplemented(
        "PJRT C API does not support MakeCrossHostReceiveBuffers");
  }

  StatusOr<ChannelHandle> CreateChannelHandle() override {
    return Unimplemented("PJRT C API does not support CreateChannelHandle");
  }

  StatusOr<ChannelHandle> CreateDeviceToHostChannelHandle() override {
    return Unimplemented(
        "PJRT C API does not support CreateDeviceToHostChannelHandle");
  }

  StatusOr<ChannelHandle> CreateHostToDeviceChannelHandle() override {
    return Unimplemented(
        "PJRT C API does not support CreateHostToDeviceChannelHandle");
  }

  Status Defragment() override {
    return Unimplemented("PJRT C API does not support Defragment");
  }

  bool SupportsSendRecvCallbacks() const override { return true; }

  const PJRT_Api* pjrt_c_api() const;

  PJRT_Client* pjrt_c_client() { return c_client_.get(); }

  PjRtCApiDevice* GetCppDevice(PJRT_Device* c_device) const {
    auto it = c_to_cpp_device_map_.find(c_device);
    CHECK(it != c_to_cpp_device_map_.end());
    return it->second;
  }

  PjRtCApiMemorySpace* GetCppMemory(PJRT_Memory* c_memory) const {
    auto it = c_to_cpp_memory_map_.find(c_memory);
    CHECK(it != c_to_cpp_memory_map_.end());
    return it->second;
  }

  PjRtHostMemoryForDeviceManager* GetPjRtHostMemoryForDeviceManager()
      const override {
    return nullptr;
  }

 private:
  void InitDevicesAndMemorySpaces();

  StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBufferInternalImpl(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      std::variant<PjRtDevice*, PjRtMemorySpace*> device_or_memory,
      const Layout* device_layout);

  const PJRT_Api* c_api_;
  std::unique_ptr<PJRT_Client, ::pjrt::PJRT_ClientDeleter> c_client_;
  std::unique_ptr<::pjrt::PJRT_KeyValueCallbackData> kv_callback_data_;
  std::vector<std::unique_ptr<PjRtCApiDevice>> owned_devices_;
  std::vector<PjRtDevice*> devices_;
  std::vector<PjRtDevice*> addressable_devices_;
  absl::flat_hash_map<PJRT_Device*, PjRtCApiDevice*> c_to_cpp_device_map_;
  std::vector<std::unique_ptr<PjRtCApiMemorySpace>> owned_memory_spaces_;
  // TODO(yueshengys): Add a `memory_spaces_` member when global memories are
  // supported.
  std::vector<PjRtMemorySpace*> addressable_memory_spaces_;
  absl::flat_hash_map<PJRT_Memory*, PjRtCApiMemorySpace*> c_to_cpp_memory_map_;
  // There may be an error fetching the topology desc via the C API
  // (e.g. unimplemented). Save the error during client init so we can return it
  // from GetTopologyDescription().
  StatusOr<const PjRtCApiTopologyDescription> topo_desc_;

  const std::string platform_version_;
  const std::string platform_name_;
  const PjRtPlatformId platform_id_;
};

class PjRtCApiBuffer : public PjRtBuffer {
 public:
  PjRtCApiBuffer(PjRtCApiClient* client, PJRT_Buffer* buffer);

  PrimitiveType element_type() const override;

  absl::Span<const int64_t> dimensions() const override;

  std::unique_ptr<PjRtLayout> layout() const override;

  // PJRT C API doesn't support tuple buffers.
  bool IsTuple() const override { return false; }

  const Shape& on_device_shape() const override {
    LOG(FATAL) << "PjRtBuffer::on_device_shape() not implemented in PJRT C API";
  }

  bool has_dynamic_dimensions() const override;

  absl::Span<const bool> is_dynamic_dimension() const override;

  StatusOr<std::vector<int64_t>> logical_dimensions() override;

  StatusOr<Shape> logical_on_device_shape() override {
    LOG(FATAL) << "PjRtBuffer::on_logical_device_shape() not implemented in "
                  "PJRT C API";
  }

  PjRtMemorySpace* memory_space() const override;

  PjRtDevice* device() const override;

  PjRtClient* client() const override { return client_; }

  StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override;

  PjRtFuture<absl::Status> ToLiteral(MutableLiteralBase* literal) override;
  PjRtFuture<absl::Status> LazyToLiteral(
      absl::AnyInvocable<absl::StatusOr<MutableLiteralBase*>() &&> generator)
      override;

  StatusOr<size_t> GetOnDeviceSizeInBytes() const override;

  PjRtFuture<Status> CopyRawToHost(void* dst, int64_t offset,
                                   int64_t transfer_size) override {
    return PjRtFuture<Status>(
        Unimplemented("PJRT C API does not support CopyRawToHost"));
  }

  void Delete() override;

  StatusOr<std::unique_ptr<ExternalReference>> ReleaseDeviceMemoryOwnership(
      bool wait_for_operations_to_complete) override {
    return Unimplemented(
        "PJRT C API does not support ReleaseDeviceMemoryOwnership");
  }

  bool IsDeleted() override;

  StatusOr<std::unique_ptr<PjRtBuffer>> CopyToDevice(
      PjRtDevice* dst_device) override;

  StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      PjRtMemorySpace* dst_memory_space) override;

  void CopyToRemoteDevice(
      PjRtFuture<StatusOr<std::string>> serialized_descriptor,
      RemoteSendCallback on_done) override {
    LOG(ERROR) << "PJRT C API does not support CopyToRemoteDevice";
  }

  void CopyToRemoteDeviceScattered(
      PjRtFuture<StatusOr<std::vector<std::string>>> serialized_descriptors,
      std::vector<RemoteSendCallback> callbacks,
      const ScatterDetails& scatter_details) override {
    LOG(ERROR) << "PJRT C API does not support CopyToRemoteDeviceScattered";
  }

  PjRtFuture<Status> GetReadyFuture() override;

  bool IsOnCpu() const override;

  PJRT_Buffer* c_buffer() const { return buffer_.get(); }

  const PJRT_Api* pjrt_c_api() const { return client_->pjrt_c_api(); }

 private:
  // Gets the raw pointer to `readiness_event_`. If `readiness_event_` has not
  // yet been initialized, this function does so before returning the pointer.
  PJRT_Event* GetReadyEvent();

  // `MakePromiseTrackEvent` sets `readiness_promise_` up to track
  // `readiness_event_`. This is used to implement `GetReadyFuture()`.
  // `readiness_promise_` should be created before calling this function.
  void MakePromiseTrackEvent();

  PjRtCApiClient* client_;
  std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter> buffer_;
  std::unique_ptr<PJRT_Event, ::pjrt::PJRT_EventDeleter> readiness_event_;
  // This is a shared_ptr to keep the underlying future alive even if
  // `readiness_promise` is destroyed before `readiness_event`, and the callback
  // we set on `readiness_event` modifies `readiness_promise_`.
  std::shared_ptr<PjRtFuture<Status>::Promise> readiness_promise_;
  // Set and cached the first time layout() is called.
  mutable std::optional<PjRtXlaLayout> layout_;
  // Set and cached the first time is_dynamic_dimension() is called.
  mutable std::optional<absl::InlinedVector<bool, InlineRank()>>
      is_dynamic_dimension_;
  // Used to synchronize concurrent setting of cached values.
  mutable absl::Mutex mu_;
};

class PjRtCApiExternalReference : public PjRtBuffer::ExternalReference {
 public:
  PjRtCApiExternalReference(PjRtCApiClient* client, PjRtCApiBuffer* buffer,
                            void* data_ptr)
      : client_(client), buffer_(buffer) {
    data_ptr_ = data_ptr;
  }
  ~PjRtCApiExternalReference() override;

 private:
  PjRtCApiClient* client_;
  PjRtCApiBuffer* buffer_;
};

class PjRtCApiExecutable : public PjRtExecutable {
 public:
  PjRtCApiExecutable(const PJRT_Api* c_api, PJRT_Executable* executable);

  absl::string_view name() const override;
  int num_replicas() const override;
  int num_partitions() const override;

  int64_t SizeOfGeneratedCodeInBytes() const override;

  StatusOr<absl::flat_hash_map<std::string, PjRtValueType>> GetCostAnalysis()
      const override;

  StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override;

  StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override {
    return pjrt::GetCompiledMemoryStats(c_api_, executable_.get());
  }

  StatusOr<std::vector<Shape>> GetOutputShapes() const override {
    LOG(FATAL) << "PjRtExecutable::GetOutputShapes() not implemented in PJRT C "
                  "API. Please use PjRtExecutable::GetOutputElementTypes() or "
                  "PjRtExecutable::GetOutputDimensions().";
  }

  StatusOr<std::vector<std::vector<PrimitiveType>>> GetOutputElementTypes()
      const override;

  StatusOr<std::vector<std::vector<DimensionVector>>> GetOutputDimensions()
      const override;

  StatusOr<std::vector<std::vector<absl::string_view>>> GetOutputMemoryKinds()
      const override;

  const PJRT_Api* pjrt_c_api() const { return c_api_; }
  PJRT_Executable* c_executable() const { return executable_.get(); }

  StatusOr<std::string> SerializeExecutable() const override;

  StatusOr<std::string> FingerprintExecutable() const override;

 private:
  const PJRT_Api* c_api_;
  std::unique_ptr<PJRT_Executable, ::pjrt::PJRT_ExecutableDeleter> executable_;
};

class PjRtCApiLoadedExecutable : public PjRtLoadedExecutable {
 public:
  PjRtCApiLoadedExecutable(PjRtCApiClient* client,
                           PJRT_LoadedExecutable* executable);

  PjRtClient* client() const override { return client_; }
  absl::string_view name() const override { return executable_->name(); }
  int num_replicas() const override { return executable_->num_replicas(); }
  int num_partitions() const override { return executable_->num_partitions(); }

  int64_t SizeOfGeneratedCodeInBytes() const override {
    return executable_->SizeOfGeneratedCodeInBytes();
  }

  StatusOr<absl::flat_hash_map<std::string, PjRtValueType>> GetCostAnalysis()
      const override {
    return executable_->GetCostAnalysis();
  }

  const DeviceAssignment& device_assignment() const override {
    CHECK(false) << "PJRT C API does not support device_assignment";
  }

  absl::Span<const LogicalDeviceIds> addressable_device_logical_ids()
      const override {
    CHECK(false)
        << "PJRT C API does not support addressable_device_logical_ids";
  }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }

  StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
    return executable_->GetHloModules();
  }

  StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override {
    return executable_->GetCompiledMemoryStats();
  }

  StatusOr<std::vector<Shape>> GetOutputShapes() const override {
    LOG(FATAL)
        << "PjRtLoadedExecutable::GetOutputShapes() not implemented in PJRT C "
           "API. Please use PjRtLoadedExecutable::GetOutputElementTypes() or "
           "PjRtLoadedExecutable::GetOutputDimensions().";
  }

  StatusOr<std::vector<std::vector<PrimitiveType>>> GetOutputElementTypes()
      const override {
    return executable_->GetOutputElementTypes();
  }

  StatusOr<std::vector<std::vector<DimensionVector>>> GetOutputDimensions()
      const override {
    return executable_->GetOutputDimensions();
  }

  StatusOr<std::vector<std::vector<absl::string_view>>> GetOutputMemoryKinds()
      const override {
    return executable_->GetOutputMemoryKinds();
  }

  StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options,
      std::optional<std::vector<PjRtFuture<Status>>>& returned_futures)
      override;

  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<Status>>& returned_future,
      bool fill_future) override;

  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<Status>>& returned_future,
      bool fill_future) override;

  void Delete() override;
  bool IsDeleted() override;

  StatusOr<std::string> SerializeExecutable() const override {
    return executable_->SerializeExecutable();
  }

  const PJRT_Api* pjrt_c_api() const { return client_->pjrt_c_api(); }
  PJRT_Executable* c_executable() const { return executable_->c_executable(); }

  PJRT_LoadedExecutable* c_loaded_executable() const {
    return loaded_executable_.get();
  }

  // True if the `returned_futures` output parameter is supported in the
  // Execute*() methods.
  bool IsReturnedFutureSupported() const override { return true; }

  // std::function version of PJRT_SendCallback
  using SendCallbackFunction = std::function<PJRT_Error*(
      PJRT_Chunk*, PJRT_CallbackError*, size_t, bool)>;
  // std::function version of PJRT_RecvCallback
  using RecvCallbackFunction = std::function<void(PJRT_CopyToDeviceStream*)>;

  // Override to call FingerprintExecutable through the wrapped
  // PjRtCApiExecutable.
  StatusOr<std::string> FingerprintExecutable() const override;

 private:
  // Groups data needed to support send/recv execution callbacks.
  struct SendRecvCallbackData {
    std::vector<std::vector<PJRT_SendCallbackInfo>> c_send_callbacks;
    std::vector<PJRT_SendCallbackInfo*> c_send_callback_lists;
    std::vector<std::vector<PJRT_RecvCallbackInfo>> c_recv_callbacks;
    std::vector<PJRT_RecvCallbackInfo*> c_recv_callback_lists;
    std::vector<SendCallbackFunction> send_callback_functions;
    std::vector<RecvCallbackFunction> recv_callback_functions;
  };

  // Gets common Execute_Args between Execute, ExecuteSharded and
  // ExecutePortable. device_complete_events in the return is set if the input
  // device_complete_events has value.
  xla::StatusOr<PJRT_LoadedExecutable_Execute_Args> GetCommonExecuteArgs(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options, PJRT_ExecuteOptions& c_options,
      std::vector<std::vector<PJRT_Buffer*>>& c_argument_lists_storage,
      std::vector<PJRT_Buffer**>& c_arguments,
      std::vector<std::vector<PJRT_Buffer*>>& c_output_lists_storage,
      std::vector<PJRT_Buffer**>& c_output_lists,
      std::optional<std::vector<PJRT_Event*>>& device_complete_events,
      SendRecvCallbackData& send_recv_callback_data,
      std::vector<int64_t>& non_donatable_input_indices_storage);

  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteWithSingleDevice(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<Status>>& returned_future, bool fill_future);

  PjRtCApiClient* client_;
  std::unique_ptr<PJRT_LoadedExecutable, ::pjrt::PJRT_LoadedExecutableDeleter>
      loaded_executable_;
  std::unique_ptr<PjRtCApiExecutable> executable_;
  std::vector<PjRtDevice*> addressable_devices_;

  void InitDevices();
};

class CApiCopyToDeviceStream : public CopyToDeviceStream {
 public:
  CApiCopyToDeviceStream(PJRT_CopyToDeviceStream* c_stream,
                         const PJRT_Api* c_api);
  ~CApiCopyToDeviceStream() override;

  PjRtFuture<Status> AddChunk(PjRtChunk chunk) override;

 private:
  PJRT_CopyToDeviceStream* c_stream_;
  const PJRT_Api* c_api_;
};

StatusOr<std::unique_ptr<PjRtClient>> GetCApiClient(
    absl::string_view device_type,
    const absl::flat_hash_map<std::string, PjRtValueType>& create_options = {},
    std::shared_ptr<KeyValueStoreInterface> kv_store = nullptr);

absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>> GetCApiTopology(
    const PJRT_Api* c_api, absl::string_view topology_name,
    const absl::flat_hash_map<std::string, PjRtValueType>& create_options);

// A variant that takes `device_type` as an input, used for plugins that are not
// registered with standard way (xla_bridge.register_plugin).
// TODO(b/322357665): Delete this method after TPU plugin changes to use the
// standard registration.
StatusOr<std::unique_ptr<PjRtTopologyDescription>> GetCApiTopology(
    absl::string_view device_type, absl::string_view topology_name,
    const absl::flat_hash_map<std::string, PjRtValueType>& create_options = {});

}  // namespace xla

#endif  // XLA_PJRT_PJRT_C_API_CLIENT_H_
