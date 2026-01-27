/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_PJRT_PJRT_STREAM_EXECUTOR_CLIENT_H_
#define XLA_PJRT_PJRT_STREAM_EXECUTOR_CLIENT_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_device_description.h"
#include "xla/pjrt/tracked_device_buffer.h"
#include "xla/pjrt/transpose.h"
#include "xla/pjrt/utils.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace xla {

struct PjRtStreamExecutorExecutionOutput {
  // Donated inputs which must be freed.
  std::vector<tsl::AsyncValueRef<RawSEDeviceMemory>> to_be_released;
  // For PjRtStreamExecutorClient implementations that
  // use ScopedDeviceAddress for donated inputs.
  std::vector<se::ScopedDeviceAddress<uint8_t>> se_to_be_released;
};

class PjRtStreamExecutorDevice : public PjRtDevice {
 public:
  PjRtStreamExecutorDevice(int id,
                           std::unique_ptr<LocalDeviceState> local_device_state,
                           int local_device_id, int process_index,
                           int process_index_in_partition, int partition_index,
                           std::string device_kind)
      : local_device_id_(local_device_id),
        local_hardware_id_(local_device_state
                               ? local_device_state->local_hardware_id()
                               : PjRtLocalHardwareId(-1)),
        local_device_state_(std::move(local_device_state)),
        description_(id, local_device_id_.value(), process_index,
                     process_index_in_partition, partition_index,
                     std::move(device_kind)) {
    if (local_device_state_ != nullptr) {
      CHECK_EQ(local_device_state_->local_device_id(), local_device_id_);
    }
  }
  ~PjRtStreamExecutorDevice() override = default;

  // Must set client exactly once.
  void SetClient(PjRtClient* client) {
    CHECK(client_ == nullptr);
    client_ = client;
    // We have to define debug_string_ and to_string_ here, because
    // platform_name() requires client_ to be set.
    std::string device_name =
        absl::StrCat(MakeAsciiTitlecase(platform_name()), "Device");

    description().SetDebugString(absl::StrCat(platform_name(), ":", id()));
    description().SetToString(absl::StrCat(device_name, "(id=", id(), ")"));
  }

  PjRtStreamExecutorDeviceDescription& description() { return description_; }
  const PjRtStreamExecutorDeviceDescription& description() const override {
    return description_;
  }

  // Return `platform_id` from client.
  PjRtPlatformId platform_id() const;

  // Return `platform_name` from client.
  absl::string_view platform_name() const;

  PjRtClient* client() const override { return client_; }

  bool IsAddressable() const override { return local_device_state_ != nullptr; }

  PjRtLocalDeviceId local_device_id() const override {
    return local_device_id_;
  }

  PjRtLocalHardwareId local_hardware_id() const override {
    return local_hardware_id_;
  }

  // If this is a device local to this host, returns a LocalDeviceState object
  // that can be used to manipulate the device. Returns nullptr if the device is
  // not local to this host.
  LocalDeviceState* local_device_state() const {
    return local_device_state_.get();
  }

  // If this is a device local to this host, returns a LocalDeviceState object
  // that can be used to manipulate the device. Returns an error if the device
  // is not local to this host.
  absl::StatusOr<LocalDeviceState*> GetLocalDeviceState() const;

  absl::Status TransferToInfeed(const LiteralSlice& literal) override;

  absl::Status TransferFromOutfeed(MutableBorrowingLiteral literal) override;

  void AttachMemorySpace(PjRtMemorySpace* memory_space,
                         bool is_default = false);

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

  absl::StatusOr<PjRtMemorySpace*> default_memory_space() const override;

  absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind(
      absl::string_view memory_space_kind) const override;

  absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind_id(int id) const;

  absl::StatusOr<std::intptr_t> GetStreamForExternalReadyEvents()
      const override;

  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
    return nullptr;
  }

 private:
  const PjRtLocalDeviceId local_device_id_;
  const PjRtLocalHardwareId local_hardware_id_;
  const std::unique_ptr<LocalDeviceState> local_device_state_;
  PjRtStreamExecutorDeviceDescription description_;
  PjRtClient* client_ = nullptr;
  absl::InlinedVector<PjRtMemorySpace*, 1> memory_spaces_;
  absl::flat_hash_map<int, PjRtMemorySpace*> memory_spaces_by_id_;
  PjRtMemorySpace* default_memory_space_ = nullptr;
};

class PjRtStreamExecutorMemorySpace : public PjRtMemorySpace {
 public:
  PjRtStreamExecutorMemorySpace(int id, PjRtDevice* device,
                                absl::string_view kind, int kind_id);

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

class PjRtStreamExecutorClient : public CommonPjRtClient {
 public:
  // `allocator` may null, in which case the platform default allocator is used.
  explicit PjRtStreamExecutorClient(
      std::string platform_name, LocalClient* client,
      std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices,
      int process_index,
      std::vector<std::unique_ptr<PjRtMemorySpace>> memory_spaces,
      std::unique_ptr<se::DeviceAddressAllocator> allocator,
      std::unique_ptr<HostMemoryAllocator> host_memory_allocator,
      bool should_stage_host_to_device_transfers,
      std::unique_ptr<gpu::GpuExecutableRunOptions> gpu_run_options);
  ~PjRtStreamExecutorClient() override = default;

  int process_index() const override { return process_index_; }

  int device_count() const override { return devices_.size(); }
  int addressable_device_count() const override {
    return addressable_devices_.size();
  }
  absl::Span<PjRtDevice* const> devices() const override { return devices_; }
  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }

  absl::StatusOr<PjRtDevice*> LookupDevice(
      PjRtGlobalDeviceId global_device_id) const override {
    auto it = id_to_device_.find(global_device_id.value());
    if (it != id_to_device_.end()) {
      return it->second;
    }
    return InvalidArgument("No matching device found for device_id %d",
                           global_device_id.value());
  }

  absl::StatusOr<PjRtDevice*> LookupAddressableDevice(
      PjRtLocalDeviceId local_device_id) const override;

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

  PjRtPlatformId platform_id() const override { return platform_id_; }
  absl::string_view platform_name() const override { return platform_name_; }
  absl::string_view platform_version() const override { return "<unknown>"; }

  std::optional<PjRtPluginAttributes> plugin_attributes() const override;

  // Most platforms expect device-to-device transfers to be enqueued on the
  // source d2d stream, but some platforms use the destination d2d stream. This
  // function specifies which one the platform expects.
  virtual bool EnqueueD2DTransfersOnSrcStream() const { return true; }

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type, absl::Span<const int64_t> dims) override;

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override;
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      const XlaComputation& computation, CompileOptions options) override;
  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      mlir::ModuleOp mlir_module, CompileOptions options) override;
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      mlir::ModuleOp mlir_module, CompileOptions options) override;

  virtual absl::StatusOr<std::string> SerializeExecutable(
      const PjRtLoadedExecutable& executable) const;

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> DeserializeExecutable(
      absl::string_view serialized,
      std::optional<CompileOptions> options) override;

  // For PjRtStreamExecutorClient, `options` is mandatory.
  // This function returns an InvalidArgument error if `std::nullopt` is passed.
  // TODO(b/237720161): make it actually optional
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
  LoadSerializedExecutable(absl::string_view serialized,
                           std::optional<CompileOptions> options,
                           const LoadOptions& load_options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Load(
      std::unique_ptr<PjRtExecutable> executable,
      const LoadOptions& load_options) override;

  absl::StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis()
      const override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateErrorBuffer(
      absl::Status error, const Shape& shape, PjRtMemorySpace* memory) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtMemorySpace* memory_space,
      std::function<void()> on_delete_callback,
      std::optional<std::intptr_t> stream) override;

  // Caller is responsible to ensure that `data` has allocated enough memory
  // for `buffer_size` to do DMA mapping.
  absl::Status DmaMap(void* data, size_t buffer_size) override;

  absl::Status DmaUnmap(void* data) override;

  bool IsDmaMapped(const void* data_start, int64_t transfer_size);

  LocalDeviceState& device_state(int device_ordinal) const {
    return *absl::down_cast<PjRtStreamExecutorDevice*>(
                LookupAddressableDevice(xla::PjRtLocalDeviceId(device_ordinal))
                    .value())
                ->local_device_state();
  }
  LocalClient* client() const { return client_; }
  se::DeviceAddressAllocator* allocator() const { return allocator_; }
  HostMemoryAllocator* host_memory_allocator() const {
    return host_memory_allocator_.get();
  }

  bool ShouldStageHostToDeviceTransfers(const void* data, int64_t size) {
    // Allocating multi-gigabyte pinned buffers can be very slow. In that case,
    // using a staging buffer is probably worse than not using one.
    // TODO(phawkins): add chunking for transfers.
    return should_stage_host_to_device_transfers_ &&
           size < (int64_t{1} << 30) && !IsDmaMapped(data, size);
  }

  virtual gpu::GpuExecutableRunOptions* gpu_run_options(
      const ExecuteOptions& options) {
    return gpu_run_options_.get();
  }

  virtual absl::StatusOr<PjRtStreamExecutorExecutionOutput> RunAsync(
      LocalExecutable& exec, PjRtDevice* device,
      absl::Span<const tsl::RCReference<CommonPjRtRawBuffer>> flat_arguments,
      absl::Span<const tsl::RCReference<CommonPjRtRawBuffer>> results,
      ExecutableRunOptions run_options, bool parameter_is_tupled_arguments,
      absl::Span<const Shape> executable_parameter_shapes);

  void ThenRecordEvent(BufferSequencingEventRef event,
                       LocalDeviceState* local_device,
                       EventPool::Handle device_event, se::Stream* stream);

  absl::Status AllocateAndRecordEvent(BufferSequencingEventRef event,
                                      LocalDeviceState* local_device,
                                      se::Stream* stream);

  tsl::RCReference<PjRtDeviceEvent> CreateErrorDeviceEvent(absl::Status error);

  void SetEventAsError(BufferSequencingEventRef event, absl::Status s);

  bool IsOnCpu(PjRtMemorySpace* memory_space);

  AsyncWorkRunner* async_work_runner() const override {
    return async_work_runner_.get();
  }

  bool allows_recursion() const override { return false; }

  absl::StatusOr<int64_t> GetOnDeviceBytesCount(
      PjRtMemorySpace* memory_space, const xla::Shape& shape) const override;

  absl::StatusOr<xla::Shape> MakeDefaultShapeForMemorySpace(
      PjRtMemorySpace* memory_space, xla::Shape shape,
      const xla::Layout* layout) const override;

  absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>> AllocateRawBuffer(
      PjRtMemorySpace* memory_space, size_t on_device_bytes_count,
      bool retry_on_oom, tsl::AsyncValueRef<bool> allocate_after) override;

  absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>>
  AllocateRawBufferForExecute(PjRtMemorySpace* memory_space,
                              size_t on_device_bytes_count,
                              bool retry_on_oom) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> DefineBuffer(
      const Shape& on_device_shape, PjRtMemorySpace* memory_space,
      tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
      absl::InlinedVector<tsl::RCReference<PjRtDeviceEvent>, 4>
          definition_device_events) override;

  absl::StatusOr<std::pair<tsl::RCReference<CommonPjRtRawBuffer>,
                           PjRtFulfillAliasRawBufferCallback>>
  CreateRawBufferChannel(PjRtMemorySpace* memory_space,
                         size_t on_device_bytes_count) override;

  absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> LinearizeInto(
      const LiteralSlice& literal, const xla::Shape& device_shape,
      HostBufferSemantics host_buffer_semantics,
      tsl::RCReference<CommonPjRtRawBuffer> raw_buffer) override;

  absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> LinearizeHostBufferInto(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      const xla::Shape& device_shape,
      tsl::RCReference<CommonPjRtRawBuffer> raw_buffer) override;

  absl::StatusOr<std::pair<tsl::RCReference<PjRtDeviceEventPromise>,
                           tsl::RCReference<PjRtDeviceEvent>>>
  CreateLinkedEventPromise(PjRtMemorySpace* memory_space,
                           absl::string_view debug_info) override;

  void WaitForAllocation(se::Stream* stream,
                         const CommonPjRtRawBuffer& raw_buffer);

 protected:
  friend class PjRtStreamExecutorRawBuffer;

  // Helper function for creating PjRtStreamExecutorExecutables. Modifies
  // `options` in-place.
  struct ExecutableExtras {
    std::shared_ptr<DeviceAssignment> device_assignment;
    std::vector<PjRtLoadedExecutable::LogicalDeviceIds>
        addressable_device_logical_ids;
    std::vector<PjRtDevice*> addressable_devices;
  };

  // Updates `options` for compilation.
  absl::Status UpdateCompileOptions(CompileOptions* options,
                                    bool lookup_addressable_devices);

  // Same as above, but also returns the executable extras.
  absl::StatusOr<ExecutableExtras> UpdateCompileOptionsAndGetExecutableExtras(
      CompileOptions* options);

  // Updates `options` for compilation, and gets the executable extras if
  // `returned_extras` is not null. It skips addressable device lookup if
  // `lookup_addressable_devices` is false.
  virtual absl::Status UpdateCompileOptionsInternal(
      CompileOptions* options, ExecutableExtras* returned_extras,
      bool lookup_addressable_devices);

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options,
      bool lookup_addressable_devices);
  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      mlir::ModuleOp mlir_module, CompileOptions options,
      bool lookup_addressable_devices);

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> CompileInternal(
      const XlaComputation& computation,
      const std::vector<const Shape*>& argument_layout_pointers,
      LayoutCanonicalizationCallback layout_canonicalization_callback,
      CompileOptions options, bool lookup_addressable_devices);

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> BuildPjRtExecutable(
      std::optional<HloModuleProto> unoptimized_hlo_module_proto,
      std::unique_ptr<LocalExecutable> local_executables,
      CompileOptions compile_options);

  absl::StatusOr<std::pair<std::unique_ptr<LocalExecutable>, CompileOptions>>
  DeserializeToLocalExecutable(absl::string_view serialized,
                               std::optional<CompileOptions> options);

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> LoadInternal(
      std::optional<HloModuleProto> unoptimized_hlo_module_proto,
      std::unique_ptr<LocalExecutable> local_executables,
      CompileOptions compile_options, bool dump);

  const PjRtPlatformId platform_id_;
  const std::string platform_name_;
  LocalClient* client_;

  // Allocator to be used for staging memory transfers to devices.
  std::unique_ptr<HostMemoryAllocator> host_memory_allocator_;

  // Device memory allocator. If owned, the allocator must outlive the devices,
  // because it is the device destructor that waits for any outstanding work to
  // complete.
  se::DeviceAddressAllocator* allocator_;
  std::unique_ptr<se::DeviceAddressAllocator> owned_allocator_;

  // Includes all devices, including non-local devices on multi-host platforms.
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> owned_devices_;
  // Pointers to `owned_devices_`.
  std::vector<PjRtDevice*> devices_;
  // Maps Device::id() to the corresponding Device. Includes all devices.
  std::map<int, PjRtDevice*> id_to_device_;
  // Local devices indexed by local device ordinal.
  std::vector<PjRtDevice*> addressable_devices_;
  int process_index_;

  std::vector<std::unique_ptr<PjRtMemorySpace>> owned_memory_spaces_;
  // Pointers to `owned_memory_spaces_`.
  std::vector<PjRtMemorySpace*> memory_spaces_;

  // Should we always prefer to stage host-to-device transfers via memory
  // allocated on host_memory_allocator_? True only on GPU, where we prefer to
  // transfer via pinned memory.
  bool should_stage_host_to_device_transfers_;

  std::unique_ptr<gpu::GpuExecutableRunOptions> gpu_run_options_;

  tsl::thread::ThreadPool compile_thread_pool_;
  std::unique_ptr<AsyncWorkRunner> async_work_runner_;

  absl::Mutex transpose_mu_;
  TransposePlanCache transpose_cache_ ABSL_GUARDED_BY(transpose_mu_);

  absl::Mutex dma_maps_mutex_;
  // Maps dma mapped start pointers to their sizes.
  absl::flat_hash_map<void*, size_t> dma_maps_ ABSL_GUARDED_BY(dma_maps_mutex_);
};

// Converts a 2D set of Device objects indexed by [replica][partition] into an
// xla::DeviceAssignment.
absl::StatusOr<DeviceAssignment> DevicesToDeviceAssignment(
    absl::Span<const std::vector<PjRtDevice*>> devices);

class PjRtStreamExecutorRawLoadedExecutable {
 public:
  PjRtStreamExecutorRawLoadedExecutable(
      int replica, int partition, RunId run_id, PjRtDevice* device,
      std::shared_ptr<DeviceAssignment> device_assignment,
      std::shared_ptr<LocalExecutable> executable,
      PjRtStreamExecutorClient* client, bool parameter_is_tupled_arguments,
      std::shared_ptr<std::vector<Shape>> on_device_executable_parameter_shapes)
      : replica_(replica),
        partition_(partition),
        run_id_(run_id),
        device_(device),
        device_assignment_(std::move(device_assignment)),
        executable_(std::move(executable)),
        client_(client),
        parameter_is_tupled_arguments_(parameter_is_tupled_arguments),
        on_device_executable_parameter_shapes_(
            std::move(on_device_executable_parameter_shapes)) {}
  absl::StatusOr<PjRtRawLoadedExecutable::RawExecuteResult> Execute(
      const ExecuteOptions& options,
      absl::Span<const tsl::RCReference<CommonPjRtRawBuffer>> inputs,
      absl::Span<const tsl::RCReference<CommonPjRtRawBuffer>> results,
      PjRtDeviceEventSet& extra_deps, PjRtDeviceEventSet& control_deps,
      bool is_predetermined_error, bool fill_future) &&;

 private:
  int replica_;
  int partition_;
  RunId run_id_;
  PjRtDevice* device_;
  std::shared_ptr<DeviceAssignment> device_assignment_;
  std::shared_ptr<LocalExecutable> executable_;
  PjRtStreamExecutorClient* client_;
  bool parameter_is_tupled_arguments_;
  std::shared_ptr<std::vector<Shape>> on_device_executable_parameter_shapes_;
};

// Wraps one or more XLA LocalExecutables (one per partition, as specified by
// the build options).
class PjRtStreamExecutorLoadedExecutable : public PjRtLoadedExecutable {
 public:
  PjRtStreamExecutorLoadedExecutable(
      std::unique_ptr<LocalExecutable> executables,
      bool parameter_is_tupled_arguments,
      std::shared_ptr<DeviceAssignment> device_assignment,
      CompileOptions compile_options,
      std::vector<LogicalDeviceIds> addressable_device_logical_ids,
      std::vector<PjRtDevice*> addressable_devices,
      PjRtStreamExecutorClient* client, xla::Shape result_shape,
      std::vector<int> output_memory_space_kind_ids);

  ~PjRtStreamExecutorLoadedExecutable() override = default;

  PjRtStreamExecutorClient* client() const override { return client_; }

  absl::string_view name() const override;

  int num_replicas() const override {
    return executable_->build_options().num_replicas();
  }

  int num_partitions() const override {
    return executable_->build_options().num_partitions();
  }

  int64_t SizeOfGeneratedCodeInBytes() const override {
    return executable_->executable()->SizeOfGeneratedCodeInBytes();
  }

  absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override {
    CompiledMemoryStats memory_stats = CompiledMemoryStats();
    memory_stats.generated_code_size_in_bytes = SizeOfGeneratedCodeInBytes();
    const BufferAssignmentProto* proto =
        executable_->executable()->buffer_assignment_proto();
    if (proto != nullptr) {
      memory_stats.serialized_buffer_assignment = proto->SerializeAsString();
      TF_ASSIGN_OR_RETURN(memory_stats.peak_memory_in_bytes,
                          ComputePeakMemory(*proto));
    }
    memory_stats.PopulateBufferStatsFromAllocations(
        executable_->executable()->GetAllocations());
    return memory_stats;
  }

  const DeviceAssignment& device_assignment() const override {
    return *device_assignment_;
  }

  absl::Span<const LogicalDeviceIds> addressable_device_logical_ids()
      const override {
    return addressable_device_logical_ids_;
  }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }

  // Return an HloModule per partition.
  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override;

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override;

  using PjRtLoadedExecutable::Execute;
  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options,
      std::optional<std::vector<Future<>>>& returned_futures) const override;

  using PjRtLoadedExecutable::ExecuteSharded;
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options, std::optional<Future<>>& returned_future,
      bool fill_future) const override;

  using PjRtLoadedExecutable::ExecutePortable;
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options, std::optional<Future<>>& returned_future,
      bool fill_future) const override;

  void Delete() override { executable_.reset(); }

  bool IsDeleted() const override { return executable_ != nullptr; }

  absl::StatusOr<std::string> SerializeExecutable() const override {
    return client_->SerializeExecutable(*this);
  }

  const std::shared_ptr<LocalExecutable>& executable() const {
    return executable_;
  }

  absl::StatusOr<CompileOptions> GetCompileOptions() const override {
    return compile_options_;
  }

  absl::StatusOr<std::string> FingerprintExecutable() const override {
    return fingerprint_;
  };

  void SetInputHloSnapshotBits(HloModuleProto hlo_module,
                               DebugOptions debug_options) {
    input_hlo_snapshot_bits_ =
        std::make_optional<InputHloSnapshotBits>(InputHloSnapshotBits{
            HloModuleProto(std::move(hlo_module)), std::move(debug_options)});
  }

 protected:
  bool parameter_is_tupled_arguments() const {
    return parameter_is_tupled_arguments_;
  }

 private:
  friend class PjRtStreamExecutorClient;
  friend class PjRtTpuClient;
  friend class InternalPjRtTpuClient;
  friend class StreamExecutorGpuClient;
  // Initializes information about which arguments to which executables must be
  // donated due to aliases that were specified by the computation.
  absl::Status SetUpDonation(bool tuple_inputs);

  absl::StatusOr<Result> ExecuteHelper(
      absl::Span<PjRtBuffer* const> argument_handles, int replica,
      int partition, const RunId& run_id, const ExecuteOptions& options,
      bool fill_future, PjRtDevice* device = nullptr) const;

  absl::Status VerifyCompatibleDevices() const;

  // Create shared pointers so we can free them after the execution: with
  // asynchronous execution, the process being executed can outlive the
  // executable itself.
  PjRtStreamExecutorClient* const client_;
  // One executable per partition.
  std::shared_ptr<LocalExecutable> executable_;
  // On device shapes of the executable parameters.
  std::shared_ptr<std::vector<Shape>> on_device_executable_parameter_shapes_;
  // Per-executable sorted vector of parameters that have any aliased buffers
  // and thus must be donated when executing the computation.
  std::vector<int> parameters_that_must_be_donated_;
  std::shared_ptr<DeviceAssignment> device_assignment_;
  CompileOptions compile_options_;

  // True if the executables were compiled expecting arguments in a single
  // tuple.
  const bool parameter_is_tupled_arguments_;
  xla::Shape result_shape_;
  std::vector<int> output_memory_space_kind_ids_;

  // The replica and partition indices of device_assignment_ to be run by this
  // client. On single-host platforms without partitioning, this is all replicas
  // (i.e. addressable_device_logical_ids_[i] = (i, 0)), but this may not be the
  // case on multi-host platforms. If there are 4 replicas and 2 partitions on a
  // single host platform, size of addressable_device_logical_ids_ is 4*2 = 8.
  std::vector<LogicalDeviceIds> addressable_device_logical_ids_;

  // addressable_devices_[i] is the Device to which
  // addressable_device_logical_ids_[i] is assigned. shared_ptrs instead of
  // unique_ptrs to play well with the Python bindings (see xla.cc).
  std::vector<PjRtDevice*> addressable_devices_;
  std::string fingerprint_;

  struct InputHloSnapshotBits {
    HloModuleProto hlo_module;
    DebugOptions debug_options;
  };

  // The unoptimized (unsharded) HloModule. Primarily used for debugging.
  std::optional<InputHloSnapshotBits> input_hlo_snapshot_bits_;
};

}  // namespace xla

#endif  // XLA_PJRT_PJRT_STREAM_EXECUTOR_CLIENT_H_
