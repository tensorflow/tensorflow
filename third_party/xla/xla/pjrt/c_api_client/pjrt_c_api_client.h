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

#ifndef XLA_PJRT_C_API_CLIENT_PJRT_C_API_CLIENT_H_
#define XLA_PJRT_C_API_CLIENT_PJRT_C_API_CLIENT_H_

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
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_abi_version_extension.h"
#include "xla/pjrt/c/pjrt_c_api_callback_extension.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_tpu_topology_extension.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_device_dimensions.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/proto/topology_description.pb.h"
#include "xla/pjrt/scoped_async_tracking_event.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

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

  absl::Span<const PjRtMemorySpaceDescription* const> memory_spaces()
      const override;

  absl::StatusOr<const PjRtMemorySpaceDescription*> default_memory_space()
      const override;

 private:
  const PJRT_Api* c_api_;
  // `device_description_` is owned by the `PJRT_Client` wrapped by `client_`
  PJRT_DeviceDescription* device_description_;
  // Device specific attributes with corresponding values.
  absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute> attributes_;
  mutable std::vector<PjRtMemorySpaceDescription> memory_space_descriptions_;
  mutable std::vector<PjRtMemorySpaceDescription*>
      memory_space_description_pointers_;
  mutable absl::StatusOr<PjRtMemorySpaceDescription*>
      default_memory_space_description_;

  // Initializes device specific attributes.
  void InitAttributes();
  // Initialize device specific memory descriptions.
  void InitMemoryDescriptions() const;
};

class PjRtCApiMemorySpace : public PjRtMemorySpace {
 public:
  explicit PjRtCApiMemorySpace(PJRT_Memory* c_memory, PjRtCApiClient* client)
      : client_(client), c_memory_(c_memory) {}

  PjRtClient* client() const override;

  absl::Span<PjRtDevice* const> devices() const override { return devices_; }

  int id() const override;

  absl::string_view kind() const override;
  int kind_id() const override;

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

  LocalChipId local_hardware_id() const override;

  absl::Status TransferToInfeed(const LiteralSlice& literal) override {
    return Unimplemented(
        "PJRT C API does not support TransferToInfeed. Please report an issue "
        "at https://github.com/google/jax/issues if you need this feature.");
  }

  absl::Status TransferFromOutfeed(MutableBorrowingLiteral literal) override {
    return Unimplemented(
        "PJRT C API does not support TransferFromOutfeed. Please report an "
        "issue at https://github.com/google/jax/issues if you need this "
        "feature.");
  }

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override {
    return memory_spaces_;
  }

  absl::StatusOr<PjRtMemorySpace*> default_memory_space() const override;

  absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind(
      absl::string_view kind) const override;

  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override;

  absl::StatusOr<bool> PoisonExecution(int32_t launch_id,
                                       absl::Status error) override;

  PJRT_Device* c_device() const { return device_; }

  const PjRtCApiDeviceDescription& description() const override {
    return description_;
  }

  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }

  absl::StatusOr<tsl::AllocatorStats> GetAllocatorStats() const override;

  absl::StatusOr<std::intptr_t> GetStreamForExternalReadyEvents()
      const override;

 private:
  // Initializes device specific attributes.
  void InitAttributes();

  friend class PjRtCApiClient;

  PjRtCApiClient* client_ = nullptr;
  // `device_` is owned by the `PJRT_Client` wrapped by `client_`
  PJRT_Device* device_;
  PjRtCApiDeviceDescription description_;
  std::vector<PjRtMemorySpace*> memory_spaces_;
  absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute> attributes_;
};

class PjRtCApiCompiler : public PjRtCompiler {
 public:
  explicit PjRtCApiCompiler(const PJRT_Api* c_api) : c_api_(c_api) {}

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, const XlaComputation& computation,
      const PjRtTopologyDescription& topology, PjRtClient* client) override;

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, mlir::ModuleOp module,
      const PjRtTopologyDescription& topology, PjRtClient* client) override;

  absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>>
  DeserializePjRtTopologyDescription(
      const std::string& serialized_topology) override;

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

  PjRtPlatformId platform_id() const override { return platform_id_; }

  absl::string_view platform_name() const override { return platform_name_; }

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

  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type,
      absl::Span<const int64_t> dims) const override;

  absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>> Subslice(
      const PjRtDeviceDimensions& chips_per_host_bounds,
      const PjRtDeviceDimensions& host_bounds) const override;

  bool is_subslice_topology() const override;

  absl::StatusOr<PjRtTopologyDescriptionProto> ToProto() const override;

  absl::StatusOr<int> ProcessCount() const override;

  absl::StatusOr<int> ChipsPerProcess() const override;

  absl::StatusOr<int> CoreCountOfDefaultTypePerChip() const override;

  absl::StatusOr<int> ChipCount() const override;
  absl::StatusOr<int> CoreCountOfDefaultType() const override;
  absl::StatusOr<int> LogicalDeviceCountOfDefaultType() const override;
  absl::StatusOr<int> LogicalDeviceCountOfDefaultTypePerProcess()
      const override;
  absl::StatusOr<int> LogicalDeviceCountOfDefaultTypePerChip() const override;
  absl::StatusOr<int> CoreCountOfDefaultTypePerProcess() const override;
  absl::StatusOr<PjRtIdContainer<ProcessId>> ProcessIds() const override;
  absl::StatusOr<PjRtIdContainer<GlobalDeviceId>>
  LogicalDeviceOfDefaultTypeIdsOnProcess(ProcessId process_id) const override;
  absl::StatusOr<std::pair<ProcessId, int>> ProcessIdAndIndexOnProcessForChip(
      GlobalChipId chip_id) const override;
  absl::StatusOr<std::pair<ProcessId, int>>
  ProcessIdAndIndexOnProcessForLogicalDeviceOfDefaultType(
      xla::GlobalDeviceId device_id) const override;
  absl::StatusOr<PjRtDeviceDimensions> ProcessCoordFromId(
      ProcessId process_id) const override;
  absl::StatusOr<GlobalChipId> ChipIdFromCoord(
      const PjRtDeviceDimensions& chip) const override;
  absl::StatusOr<xla::GlobalDeviceId>
  LogicalDeviceOfDefaultTypeIdFromChipCoordAndCoreIndex(
      const PjRtDeviceDimensions& chip, int core_index) const override;
  absl::StatusOr<std::pair<PjRtDeviceDimensions, int32_t>>
  ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
      xla::GlobalDeviceId device_id) const override;
  absl::StatusOr<PjRtDeviceDimensions> ChipsPerProcessBounds() const override;
  absl::StatusOr<PjRtDeviceDimensions> ChipBounds() const override;
  absl::StatusOr<PjRtDeviceDimensions> ProcessBounds() const override;

 private:
  std::unique_ptr<PjRtCApiCompiler> compiler_;
  const PJRT_Api* c_api_;
  const PJRT_TpuTopology_Extension* tpu_topology_extension_;
  // nullptr iff the PJRT_TopologyDescription isn't owned by this wrapper
  // (i.e. by the caller).
  std::unique_ptr<PJRT_TopologyDescription,
                  ::pjrt::PJRT_TopologyDescriptionDeleter>
      owned_c_topology_;
  PJRT_TopologyDescription* c_topology_;
  // Device specific attributes with corresponding values.
  absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute> attributes_;

  const std::string platform_version_;
  const std::string platform_name_;
  const PjRtPlatformId platform_id_;

  // Initializes device specific attributes.
  void InitAttributes();
};

class PjRtCApiAsyncTrackingEvent : public ScopedAsyncTrackingEvent {
 public:
  PjRtCApiAsyncTrackingEvent(const PJRT_Api* c_api,
                             PJRT_AsyncTrackingEvent* event);
  ~PjRtCApiAsyncTrackingEvent() override;

  void AddDependency(tsl::RCReference<tsl::AsyncValue> dependency) override;

 private:
  const PJRT_Api* c_api_;
  PJRT_AsyncTrackingEvent* event_;
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

  absl::StatusOr<PjRtDevice*> LookupDevice(
      GlobalDeviceId global_device_id) const override;

  absl::StatusOr<PjRtDevice*> LookupAddressableDevice(
      LocalDeviceId local_device_id) const override;

  void UpdateGlobalProcessInfo(
      absl::Span<tensorflow::CoordinatedTaskStateInfo> infos) override;

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

  PjRtPlatformId platform_id() const override { return platform_id_; }

  absl::string_view platform_name() const override { return platform_name_; };

  absl::string_view platform_version() const override;

  absl::StatusOr<std::unique_ptr<PjRtRuntimeAbiVersion>> RuntimeAbiVersion()
      const override;

  std::optional<PjRtPluginAttributes> plugin_attributes() const override;

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  absl::StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis()
      const override {
    return Unimplemented(
        "PJRT C API does not support GetHloCostAnalysis. Please report an "
        "issue at https://github.com/google/jax/issues if you need this "
        "feature.");
  }

  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type, absl::Span<const int64_t> dims) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      const XlaComputation& computation, CompileOptions options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      mlir::ModuleOp module, CompileOptions options) override;

  // `PjRtCApiClient::LoadSerializedExecutable()` ignores `LoadOptions` arg
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
  LoadSerializedExecutable(absl::string_view serialized,
                           std::optional<CompileOptions> options,
                           const LoadOptions& load_options) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateUninitializedBuffer(
      const Shape& shape, PjRtMemorySpace* memory_space) override;

  absl::StatusOr<
      std::pair<std::unique_ptr<PjRtBuffer>, PjRtFulfillAliasBufferCallback>>
  CreateAliasBuffer(const Shape& shape, PjRtMemorySpace* memory_space) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateErrorBuffer(
      absl::Status error, const Shape& shape, PjRtMemorySpace* memory) override;

  absl::StatusOr<const PjRtTopologyDescription*> GetTopologyDescription()
      const override;

  absl::StatusOr<HostAllocator*> GetHostAllocator() const override;

  absl::StatusOr<std::unique_ptr<AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(
      absl::Span<const ShapeSpec> shape_specs,
      std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
      PjRtMemorySpace* memory_space) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtMemorySpace* memory_space, const Layout* device_layout) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtMemorySpace* memory_space,
      const Layout* device_layout) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtMemorySpace* memory_space,
      std::function<void()> on_delete_callback,
      std::optional<std::intptr_t> stream) override;

  absl::StatusOr<std::uintptr_t> UnsafeBufferPointer(
      PjRtBuffer* buffer) override;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffers(absl::Span<const Shape> shapes,
                              PjRtDevice* device,
                              PjRtCrossHostRecvNotifier notifier) override;

  absl::StatusOr<std::vector<Future<>>> CrossHostSendBuffers(
      absl::Span<PjRtBuffer* const> buffers,
      absl::Span<const GlobalDeviceId> dst_global_device_ids,
      std::vector<CrossHostTransferKey> transfer_keys) override;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  CrossHostReceiveBuffers(
      xla::PjRtDevice* device, absl::Span<const xla::Shape> shapes,
      absl::Span<const GlobalDeviceId> src_global_device_ids,
      std::vector<CrossHostTransferKey> transfer_keys) override;

  absl::Status DmaMap(void* data, size_t size) override;

  absl::Status DmaUnmap(void* data) override;

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

  using CrossHostRecvNotifierFunction =
      std::function<void(PJRT_Error*, const char**, size_t*, size_t)>;

  template <typename ExtType>
  ExtType* FindExtension(PJRT_Extension_Type type) const {
    return reinterpret_cast<ExtType*>(FindExtensionImpl(type));  // NOLINT
  }

  template <typename CallbackArgs, int ArgsStructSize>
  absl::Status RegisterCallback(
      PJRT_Callback_Type callback_type,
      std::function<void(CallbackArgs*)> user_callback) {
    return RegisterCallbackImpl(
        callback_type, [user_callback = std::move(user_callback)](void* args) {
          CallbackArgs* callback_args = static_cast<CallbackArgs*>(args);
          CHECK_OK(pjrt::ActualStructSizeIsGreaterOrEqual(
              "CallbackArgs", ArgsStructSize, callback_args->struct_size));
          user_callback(callback_args);
        });
  }

  absl::Status InvokeCallbacks(PJRT_Callback_Type callback_type,
                               void* callback_args);

 private:
  void InitDevicesAndMemorySpaces();
  void InitAttributes();
  PJRT_Extension_Base* FindExtensionImpl(PJRT_Extension_Type type) const;
  absl::Status RegisterCallbackImpl(PJRT_Callback_Type callback_type,
                                    std::function<void(void*)> callback);

  std::unique_ptr<PJRT_Error, ::pjrt::PJRT_ErrorDeleter> CreatePjRtError(
      const absl::Status& error) const;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBufferInternalImpl(
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
  std::vector<PjRtMemorySpace*> addressable_memory_spaces_;
  absl::flat_hash_map<PJRT_Memory*, PjRtCApiMemorySpace*> c_to_cpp_memory_map_;
  // There may be an error fetching the topology desc via the C API
  // (e.g. unimplemented). Save the error during client init so we can return it
  // from GetTopologyDescription().
  absl::StatusOr<const PjRtCApiTopologyDescription> topo_desc_;
  absl::flat_hash_map<PJRT_Extension_Type, PJRT_Extension_Base*> extensions_;
  // Not all PJRT C API implementations support the host allocator extension.
  absl::StatusOr<std::unique_ptr<PjRtClient::HostAllocator>> host_allocator_;

  const std::string platform_version_;
  const std::string platform_name_;
  const PjRtPlatformId platform_id_;
  absl::flat_hash_map<std::string, xla::PjRtValueType> attributes_;
  std::vector<std::unique_ptr<std::function<void(void*)>>>
      registered_callbacks_;
};

class PjRtCApiBuffer : public PjRtBuffer {
 public:
  PjRtCApiBuffer(PjRtCApiClient* client, PJRT_Buffer* buffer);

  PrimitiveType element_type() const override;

  absl::Span<const int64_t> dimensions() const override;

  std::shared_ptr<const PjRtLayout> layout() const override;

  // PJRT C API doesn't support tuple buffers.
  bool IsTuple() const override { return false; }

  const Shape& on_device_shape() const override;

  bool has_dynamic_dimensions() const override;

  absl::Span<const bool> is_dynamic_dimension() const override;

  absl::StatusOr<std::vector<int64_t>> logical_dimensions() override;

  absl::StatusOr<Shape> logical_on_device_shape() override;

  PjRtMemorySpace* memory_space() const override;

  PjRtDevice* device() const override;

  PjRtClient* client() const override { return client_; }

  absl::StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override;

  Future<> ToLiteral(MutableLiteralBase* literal) override;
  Future<> LazyToLiteral(
      absl::AnyInvocable<Future<MutableLiteralBase*>() &&> generator) override;

  absl::StatusOr<size_t> GetOnDeviceSizeInBytes() const override;

  Future<> CopyRawToHost(void* dst, int64_t offset,
                         int64_t transfer_size) override;

  Future<> CopyRawToHostFuture(Future<void*> dst, int64_t offset,
                               int64_t transfer_size) override;

  void Delete() override;

  absl::StatusOr<std::unique_ptr<ExternalReference>>
  ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) override {
    return Unimplemented(
        "PJRT C API does not support ReleaseDeviceMemoryOwnership");
  }

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> DonateWithControlDependency(
      Future<> dependency) override;

  bool IsDeleted() const override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      PjRtMemorySpace* dst_memory_space) override;

  void CopyToRemoteDevice(Future<std::string> serialized_descriptor,
                          RemoteSendCallback on_done) override;

  Future<> GetReadyFuture() override;

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
  std::shared_ptr<Promise<>> readiness_promise_;
  // Future tied to the `readiness_promise_`.
  Future<> readiness_future_;
  // Set and cached the first time layout() is called.
  mutable std::shared_ptr<const PjRtLayout> layout_;
  // Set and cached the first time is_dynamic_dimension() is called.
  mutable std::optional<absl::InlinedVector<bool, InlineRank()>>
      is_dynamic_dimension_;
  // Used to synchronize concurrent setting of cached values.
  mutable absl::Mutex mu_;
  // Cached result of on_device_shape();
  mutable std::optional<Shape> on_device_shape_;
};

class PjRtCApiExternalReference : public PjRtBuffer::ExternalReference {
 public:
  PjRtCApiExternalReference(PjRtCApiClient* client, PjRtCApiBuffer* buffer,
                            void* data_ptr)
      : client_(client), buffer_(buffer) {
    data_ptr_ = data_ptr;
  }
  ~PjRtCApiExternalReference() override;

  absl::Status WaitUntilBufferReadyOnStream(std::intptr_t stream) override;

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

  absl::StatusOr<absl::flat_hash_map<std::string, PjRtValueType>>
  GetCostAnalysis() const override;

  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override;

  absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override {
    return pjrt::GetCompiledMemoryStats(c_api_, executable_.get());
  }

  std::optional<std::vector<OpSharding>> GetParameterShardings() const override;

  absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
  GetParameterLayouts() const override;

  absl::StatusOr<std::vector<Shape>> GetOutputShapes() const override;

  absl::StatusOr<std::vector<std::vector<PrimitiveType>>>
  GetOutputElementTypes() const override;

  absl::StatusOr<std::vector<std::vector<DimensionVector>>>
  GetOutputDimensions() const override;

  std::optional<std::vector<OpSharding>> GetOutputShardings() const override;

  absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
  GetOutputLayouts() const override;

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override;

  const PJRT_Api* pjrt_c_api() const { return c_api_; }
  PJRT_Executable* c_executable() const { return executable_.get(); }

  absl::StatusOr<std::string> SerializeExecutable() const override;

  absl::StatusOr<std::string> FingerprintExecutable() const override;

  absl::StatusOr<std::unique_ptr<PjRtExecutableAbiVersion>> GetAbiVersion()
      const override;

  // TODO(b/438000615): Move this to PjRtLoadedExecutable.
  absl::StatusOr<std::string> GetSerializedExecutableMetadata() const;

  absl::StatusOr<CompileOptions> GetCompileOptions() const override;

 private:
  const PJRT_Api* c_api_;
  std::unique_ptr<PJRT_Executable, ::pjrt::PJRT_ExecutableDeleter> executable_;
};

class PjRtCApiLoadedExecutable : public PjRtLoadedExecutable {
 public:
  PjRtCApiLoadedExecutable(PjRtCApiClient* client,
                           PJRT_LoadedExecutable* executable);

  PjRtExecutable* GetExecutable() const override { return executable_.get(); }

  PjRtClient* client() const override { return client_; }
  absl::string_view name() const override { return executable_->name(); }
  int num_replicas() const override { return executable_->num_replicas(); }
  int num_partitions() const override { return executable_->num_partitions(); }

  int64_t SizeOfGeneratedCodeInBytes() const override {
    return executable_->SizeOfGeneratedCodeInBytes();
  }

  absl::StatusOr<absl::flat_hash_map<std::string, PjRtValueType>>
  GetCostAnalysis() const override {
    return executable_->GetCostAnalysis();
  }

  const DeviceAssignment& device_assignment() const override {
    CHECK(device_assignment_ != nullptr)
        << "device_assignment_ is a nullptr. This is likely because "
           "PjRtCApiLoadedExecutable::device_assignment() was called on a "
           "portable executable, which does not have a device assignment.";
    return *device_assignment_;
  }

  absl::Span<const LogicalDeviceIds> addressable_device_logical_ids()
      const override {
    CHECK(false)
        << "PJRT C API does not support addressable_device_logical_ids";
  }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }

  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
    return executable_->GetHloModules();
  }

  absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override {
    return executable_->GetCompiledMemoryStats();
  }

  std::optional<std::vector<OpSharding>> GetParameterShardings()
      const override {
    return executable_->GetParameterShardings();
  }

  absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
  GetParameterLayouts() const override {
    return executable_->GetParameterLayouts();
  }

  absl::StatusOr<std::vector<Shape>> GetOutputShapes() const override {
    return executable_->GetOutputShapes();
  }

  absl::StatusOr<std::vector<std::vector<PrimitiveType>>>
  GetOutputElementTypes() const override {
    return executable_->GetOutputElementTypes();
  }

  absl::StatusOr<std::vector<std::vector<DimensionVector>>>
  GetOutputDimensions() const override {
    return executable_->GetOutputDimensions();
  }

  std::optional<std::vector<OpSharding>> GetOutputShardings() const override {
    return executable_->GetOutputShardings();
  }

  absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
  GetOutputLayouts() const override {
    return executable_->GetOutputLayouts();
  }

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override {
    return executable_->GetOutputMemoryKinds();
  }

  absl::StatusOr<CompileOptions> GetCompileOptions() const override {
    return executable_->GetCompileOptions();
  }

  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options,
      std::optional<std::vector<Future<>>>& returned_futures) const override;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options, std::optional<Future<>>& returned_future,
      bool fill_future) const override;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options, std::optional<Future<>>& returned_future,
      bool fill_future) const override;

  void Delete() override;
  bool IsDeleted() const override;

  absl::StatusOr<std::string> SerializeExecutable() const override {
    return executable_->SerializeExecutable();
  }

  const PJRT_Api* pjrt_c_api() const { return client_->pjrt_c_api(); }
  PJRT_Executable* c_executable() const { return executable_->c_executable(); }

  PJRT_LoadedExecutable* c_loaded_executable() const {
    return loaded_executable_.get();
  }

  // std::function version of PJRT_SendCallback
  using SendCallbackFunction = std::function<PJRT_Error*(
      PJRT_Chunk*, PJRT_CallbackError*, size_t, bool)>;
  // std::function version of PJRT_RecvCallback
  using RecvCallbackFunction = std::function<void(PJRT_CopyToDeviceStream*)>;

  // Override to call FingerprintExecutable through the wrapped
  // PjRtCApiExecutable.
  absl::StatusOr<std::string> FingerprintExecutable() const override {
    return executable_->FingerprintExecutable();
  }

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

  // Returns the number of outputs of the executable.
  absl::StatusOr<size_t> GetNumOutputs() const;

  // Allocates memory for the `Execute` output.
  // These functions are a little verbose, but allocating the correct amount of
  // memory on initialization (thus avoiding `resize` calls) provides a
  // significant performance optimization.
  absl::StatusOr<std::vector<std::vector<PJRT_Buffer*>>>
  InitializeOutputListsStorage(size_t outer_size) const;
  absl::StatusOr<std::vector<PJRT_Buffer**>> InitializeOutputLists(
      std::vector<std::vector<PJRT_Buffer*>>& c_output_lists_storage) const;

  // Gets common Execute_Args for use in various Execute* functions.
  // device_complete_events in the return is set if the input
  // device_complete_events has value.
  absl::StatusOr<PJRT_LoadedExecutable_Execute_Args> GetCommonExecuteArgs(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options, PJRT_ExecuteOptions& c_options,
      std::vector<std::vector<PJRT_Buffer*>>& c_argument_lists_storage,
      std::vector<PJRT_Buffer**>& c_arguments,
      std::optional<std::vector<PJRT_Event*>>& device_complete_events,
      SendRecvCallbackData& send_recv_callback_data,
      std::vector<int64_t>& non_donatable_input_indices_storage,
      std::vector<int>& task_ids_storage,
      std::vector<int64_t>& incarnation_ids_storage) const;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  ExecuteWithSingleDevice(absl::Span<PjRtBuffer* const> argument_handles,
                          PjRtDevice* device, const ExecuteOptions& options,
                          std::optional<Future<>>& returned_future,
                          bool fill_future) const;

  PjRtCApiClient* client_;
  std::unique_ptr<PJRT_LoadedExecutable, ::pjrt::PJRT_LoadedExecutableDeleter>
      loaded_executable_;
  std::unique_ptr<PjRtCApiExecutable> executable_;
  std::vector<PjRtDevice*> addressable_devices_;
  std::unique_ptr<const DeviceAssignment> device_assignment_;

  void InitDevices();
  void InitDeviceAssignment();
};

class CApiCopyToDeviceStream : public CopyToDeviceStream {
 public:
  CApiCopyToDeviceStream(PJRT_CopyToDeviceStream* c_stream,
                         const PJRT_Api* c_api);
  ~CApiCopyToDeviceStream() override;

  Future<> AddChunk(PjRtChunk chunk) override;

 private:
  PJRT_CopyToDeviceStream* c_stream_;
  const PJRT_Api* c_api_;
};

class PjRtCApiRuntimeAbiVersion : public PjRtRuntimeAbiVersion {
 public:
  PjRtCApiRuntimeAbiVersion(PJRT_RuntimeAbiVersion* c_abi_version,
                            const PJRT_Api* c_api,
                            const PJRT_AbiVersion_Extension* extension);
  ~PjRtCApiRuntimeAbiVersion() override;

  absl::Status IsCompatibleWith(
      const PjRtRuntimeAbiVersion& runtime_abi_version) const override;
  absl::Status IsCompatibleWith(
      const PjRtExecutableAbiVersion& executable_abi_version) const override;

  absl::StatusOr<PjRtRuntimeAbiVersionProto> ToProto() const override;
  PjRtPlatformId platform_id() const override;

  PJRT_RuntimeAbiVersion* c_abi_version() const { return c_abi_version_; }

 private:
  PJRT_RuntimeAbiVersion* c_abi_version_;
  const PJRT_Api* c_api_;
  const PJRT_AbiVersion_Extension* extension_;
};

class PjRtCApiExecutableAbiVersion : public PjRtExecutableAbiVersion {
 public:
  PjRtCApiExecutableAbiVersion(PJRT_ExecutableAbiVersion* c_abi_version,
                               const PJRT_Api* c_api,
                               const PJRT_AbiVersion_Extension* extension);
  ~PjRtCApiExecutableAbiVersion() override;

  absl::StatusOr<PjRtExecutableAbiVersionProto> ToProto() const override;
  PjRtPlatformId platform_id() const override;

  PJRT_ExecutableAbiVersion* c_abi_version() const { return c_abi_version_; }

 private:
  PJRT_ExecutableAbiVersion* c_abi_version_;
  const PJRT_Api* c_api_;
  const PJRT_AbiVersion_Extension* extension_;
};

absl::StatusOr<std::unique_ptr<PjRtClient>> GetCApiClient(
    absl::string_view device_type,
    const absl::flat_hash_map<std::string, PjRtValueType>& create_options = {},
    std::shared_ptr<KeyValueStoreInterface> kv_store = nullptr);

absl::StatusOr<std::unique_ptr<PjRtClient>> WrapClientAroundCApi(
    const PJRT_Api* c_api,
    const absl::flat_hash_map<std::string, PjRtValueType>& create_options = {},
    std::shared_ptr<KeyValueStoreInterface> kv_store = nullptr);

absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>> GetCApiTopology(
    const PJRT_Api* c_api, absl::string_view topology_name,
    const absl::flat_hash_map<std::string, PjRtValueType>& create_options);

// A variant that takes `device_type` as an input, used for plugins that are not
// registered with standard way (xla_bridge.register_plugin).
// TODO(b/322357665): Delete this method after TPU plugin changes to use the
// standard registration.
absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>> GetCApiTopology(
    absl::string_view device_type, absl::string_view topology_name,
    const absl::flat_hash_map<std::string, PjRtValueType>& create_options = {});

absl::StatusOr<std::unique_ptr<PjRtCompiler>> GetCApiCompiler(
    absl::string_view device_type);

absl::StatusOr<std::unique_ptr<PjRtCompiler>> GetCApiCompiler();

absl::StatusOr<std::unique_ptr<PjRtPhaseCompiler>> GetCApiPhaseCompiler(
    absl::string_view device_type);

absl::StatusOr<std::unique_ptr<PjRtPhaseCompiler>> GetCApiPhaseCompiler();

}  // namespace xla

#endif  // XLA_PJRT_C_API_CLIENT_PJRT_C_API_CLIENT_H_
