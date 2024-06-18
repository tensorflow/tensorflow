/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_PJRT_CPU_CPU_CLIENT_H_
#define XLA_PJRT_CPU_CPU_CLIENT_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "xla/client/xla_computation.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/cpu/abstract_tfrt_cpu_buffer.h"
#include "xla/pjrt/cpu/cpu_topology.h"
#include "xla/pjrt/cpu/tracked_tfrt_cpu_device_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/semaphore.h"
#include "xla/pjrt/transpose.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/cpu/collectives_interface.h"
#include "xla/service/cpu/cpu_event.h"
#include "xla/service/executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/statusor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/threadpool.h"

namespace xla {

class TfrtCpuDevice;  // forward declare

class TfrtCpuDeviceDescription final : public PjRtDeviceDescription {
 public:
  explicit TfrtCpuDeviceDescription(int process_id, int local_device_id);

  int id() const override { return id_.value(); }

  int process_index() const override { return process_index_; }

  int local_hardware_id() const { return local_hardware_id_; }

  absl::string_view device_kind() const override;

  absl::string_view DebugString() const override;

  absl::string_view ToString() const override;

  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }

 private:
  PjRtGlobalDeviceId id_;
  int process_index_;
  int local_hardware_id_;
  std::string debug_string_;
  std::string to_string_;
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes_ = {};
};

class TfrtCpuTopologyDescription : public PjRtTopologyDescription {
 public:
  static TfrtCpuTopologyDescription Create(
      PjRtPlatformId platform_id, absl::string_view platform_name,
      absl::string_view platform_version,
      absl::Span<const std::unique_ptr<TfrtCpuDevice>> devices,
      absl::Span<const std::string> machine_attributes);

  // `cpu_device_ids` is the list of logical device ids for the CPU devices and
  // will be used to initialize the CPU topology.
  TfrtCpuTopologyDescription(
      const PjRtPlatformId platform_id, const absl::string_view platform_name,
      const absl::string_view platform_version,
      const std::vector<CpuTopology::CpuDevice> cpu_devices,
      absl::Span<const std::string> machine_attributes)
      : platform_id_(platform_id),
        platform_name_(platform_name),
        platform_version_(platform_version),
        cpu_topology_(std::move(cpu_devices),
                      std::vector<std::string>(machine_attributes.begin(),
                                               machine_attributes.end())) {}

  bool operator==(const TfrtCpuTopologyDescription& other) const {
    return this->platform_id() == other.platform_id() &&
           this->platform_name() == other.platform_name() &&
           this->platform_version() == other.platform_version() &&
           this->cpu_topology().devices() == other.cpu_topology().devices();
  }

  PjRtPlatformId platform_id() const override { return platform_id_; }

  absl::string_view platform_name() const override { return platform_name_; }

  absl::string_view platform_version() const override {
    return platform_version_;
  }

  std::vector<std::unique_ptr<const PjRtDeviceDescription>> DeviceDescriptions()
      const override;

  const CpuTopology& cpu_topology() const { return cpu_topology_; }
  const CpuTopology* cpu_topology_ptr() const { return &cpu_topology_; }

  // No subslice is supported.
  bool is_subslice_topology() const override { return false; }

  // TODO(b/319478189): We support multi-host CPU computations and should
  // correctly report process count.
  absl::StatusOr<int> ProcessCount() const override { return 1; }

  absl::StatusOr<int> CoreCountOfDefaultType() const override {
    return cpu_topology_.number_of_devices();
  }

  absl::StatusOr<int> LogicalDeviceCountOfDefaultType() const override {
    return cpu_topology_.number_of_devices();
  }

  absl::StatusOr<int> CoreCountOfDefaultTypePerProcess() const override {
    return cpu_topology_.number_of_devices();
  }

  absl::StatusOr<int> CoreCountOfDefaultTypePerChip() const override {
    return 1;
  }

  absl::StatusOr<std::string> Serialize() const override;

  // Returns vendor specific attributes about the topology.
  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }

  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type,
      absl::Span<const int64_t> dims) const override;

 private:
  const PjRtPlatformId platform_id_;
  const std::string platform_name_;
  const std::string platform_version_;
  const CpuTopology cpu_topology_;
  absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute> attributes_;
};

class TfrtCpuDevice final : public PjRtDevice {
 public:
  explicit TfrtCpuDevice(int process_id, int local_device_id,
                         int max_inflight_computations = 32);

  const TfrtCpuDeviceDescription& description() const override {
    return description_;
  }

  void SetClient(PjRtClient* client) {
    CHECK(client_ == nullptr);
    client_ = client;
  }

  PjRtClient* client() const override { return client_; }

  bool IsAddressable() const override {
    return process_index() == client()->process_index();
  }

  int local_hardware_id() const override {
    return local_hardware_id_typed().value();
  }

  PjRtLocalDeviceId local_device_id() const override {
    return PjRtLocalDeviceId(local_hardware_id_typed().value());
  }

  PjRtLocalHardwareId local_hardware_id_typed() const override {
    return PjRtLocalHardwareId(description_.local_hardware_id());
  }

  absl::Status TransferToInfeed(const LiteralSlice& literal) override;

  absl::Status TransferFromOutfeed(MutableBorrowingLiteral literal) override;

  void AttachMemorySpace(PjRtMemorySpace* memory_space);

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

  absl::StatusOr<PjRtMemorySpace*> default_memory_space() const override;

  absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind(
      absl::string_view memory_space_kind) const override;

  absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind_id(int id) const;

  // Returns a semaphore for admission control on inflight computations.
  Semaphore& max_inflight_computations_semaphore() {
    return max_inflight_computations_semaphore_;
  }

  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
    return nullptr;
  }

 private:
  PjRtClient* client_ = nullptr;
  TfrtCpuDeviceDescription description_;
  absl::InlinedVector<PjRtMemorySpace*, 1> memory_spaces_;
  absl::flat_hash_map<int, PjRtMemorySpace*> memory_spaces_by_id_;

  // TODO(zhangqiaorjc): Optimize semaphore related overhead.
  // Semaphore used to limit how many programs can be enqueued by the host
  // ahead of the device.
  Semaphore max_inflight_computations_semaphore_;
};

class TfrtCpuClient final : public PjRtClient {
 public:
  TfrtCpuClient(int process_index,
                std::vector<std::unique_ptr<TfrtCpuDevice>> devices,
                std::shared_ptr<cpu::CollectivesInterface> collectives,
                size_t num_threads, bool asynchronous);
  ~TfrtCpuClient() override;

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
      PjRtGlobalDeviceId global_device_id) const override;

  absl::StatusOr<PjRtDevice*> LookupAddressableDevice(
      PjRtLocalDeviceId local_device_id) const override;

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

  PjRtPlatformId platform_id() const override {
    return tsl::Fingerprint64(CpuName());
  }

  absl::string_view platform_name() const override { return CpuName(); }

  absl::string_view platform_version() const override { return "<unknown>"; }

  PjRtRuntimeType runtime_type() const override { return kTfrt; }

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type, absl::Span<const int64_t> dims) override;

  absl::StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis()
      const override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override;
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      mlir::ModuleOp module, CompileOptions options) override;

  // For TfrtCpuClient, `options` is mandatory.
  // This function returns an InvalidArgument error if `std::nullopt` is passed.
  // TODO(b/237720161): make it actually optional
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> DeserializeExecutable(
      absl::string_view serialized,
      std::optional<CompileOptions> options) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateErrorBuffer(
      absl::Status error, const Shape& shape, PjRtDevice* device) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateErrorBuffer(
      absl::Status error, const Shape& shape, PjRtMemorySpace* memory) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateUninitializedBuffer(
      const Shape& shape, PjRtDevice* device) override;

  absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(absl::Span<const Shape> shapes,
                                    PjRtDevice* device) override;

  absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(absl::Span<const Shape> shapes,
                                    PjRtMemorySpace* memory_space) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtDevice* device) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtDevice* device, const Layout* device_layout) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtMemorySpace* memory_space, const Layout* device_layout) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtDevice* device) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtMemorySpace* memory_space) override;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffers(absl::Span<const Shape> shapes,
                              PjRtDevice* device,
                              PjRtCrossHostRecvNotifier notifier) override {
    return Unimplemented("MakeCrossHostReceiveBuffers not implemented.");
  }

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffersForGather(
      absl::Span<const Shape> shapes, std::vector<GatherDetails> gather_details,
      PjRtDevice* device, PjRtCrossHostRecvNotifier notifier) override {
    return Unimplemented(
        "MakeCrossHostReceiveBuffersForGather not implemented.");
  }

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtDevice* device,
      std::function<void()> on_delete_callback,
      std::optional<std::intptr_t> stream) override;

  absl::StatusOr<ChannelHandle> CreateChannelHandle() override {
    return Unimplemented("CreateChannelHandle not implemented.");
  }
  absl::StatusOr<ChannelHandle> CreateDeviceToHostChannelHandle() override {
    return Unimplemented("CreateDeviceToHostChannelHandle not implemented.");
  }
  absl::StatusOr<ChannelHandle> CreateHostToDeviceChannelHandle() override {
    return Unimplemented("CreateHostToDeviceChannelHandle not implemented.");
  }

  absl::Status Defragment() override {
    return Unimplemented("Defragment not implemented.");
  }

  tsl::thread::ThreadPool* pjrt_client_thread_pool() const {
    return pjrt_client_thread_pool_.get();
  }

  AsyncWorkRunner* async_work_runner() const {
    return async_work_runner_.get();
  }

  Eigen::ThreadPoolDevice* eigen_intraop_device() const {
    return eigen_intraop_device_.get();
  }

  tsl::AsyncValueRef<CpuEvent> GetLastCollectiveLaunchEvent() {
    absl::MutexLock lock(&mu_);
    return last_collective_launch_event_.CopyRef();
  }

  void SetLastCollectiveLaunchEvent(tsl::AsyncValueRef<CpuEvent> event) {
    absl::MutexLock lock(&mu_);
    last_collective_launch_event_ = std::move(event);
  }

  tsl::AsyncValueRef<CpuEvent> GetLastEnqueueEvent() {
    return last_enqueue_event_.CopyRef();
  }

  void SetLastEnqueueEvent(tsl::AsyncValueRef<CpuEvent> event) {
    last_enqueue_event_ = std::move(event);
  }

  absl::StatusOr<const xla::PjRtTopologyDescription*> GetTopologyDescription()
      const override {
    return &topology_;
  }

 private:
  friend class TfrtCpuExecutable;

  int process_index_;
  // Includes all devices, including non-addressable devices.
  std::vector<std::unique_ptr<TfrtCpuDevice>> owned_devices_;
  // Pointers to `owned_devices_`.
  std::vector<PjRtDevice*> devices_;
  // Maps Device::id() to the corresponding Device. Includes all devices.
  absl::flat_hash_map<PjRtGlobalDeviceId, TfrtCpuDevice*> id_to_device_;
  // Addressable devices indexed by core_id.
  std::vector<PjRtDevice*> addressable_devices_;
  std::unique_ptr<ComputationPlacer> computation_placer_;

  // Addressable memory spaces.
  std::vector<std::unique_ptr<PjRtMemorySpace>> owned_memory_spaces_;
  // Pointers to `owned_memory_spaces_`.
  std::vector<PjRtMemorySpace*> memory_spaces_;

  // Thread pool for running PjRtClient tasks.
  std::unique_ptr<tsl::thread::ThreadPool> pjrt_client_thread_pool_;
  std::unique_ptr<AsyncWorkRunner> async_work_runner_;

  // TODO(zhangqiaorjc): Use tsl::compat::EigenHostContextThreadPool.
  std::unique_ptr<tsl::thread::ThreadPool> eigen_intraop_pool_;
  std::unique_ptr<Eigen::ThreadPoolDevice> eigen_intraop_device_;

  // Launching collectives are prone to deadlock when we use fixed-sized
  // threadpools since ExecuteHelper will block until all replicas reach the
  // barrier. We ensure that
  // 1. Threadpool size is at least as large as device_count so one collective
  //    launch over all devices can succeed.
  // 2. Gang-schedule each collective by conservatively ensuring a total order
  //    of collectives and launching only one collective at a time to avoid
  //    having no active threads to make progress
  // TODO(zhangqiaorjc): Explore alternatives that allow multiple concurrent
  // collectives.
  mutable absl::Mutex mu_;
  tsl::AsyncValueRef<CpuEvent> last_collective_launch_event_
      ABSL_GUARDED_BY(mu_);

  // A cache for transpose plans. We use transposes to convert
  // (possibly strided) buffers provided to BufferFromHostBuffer into dense
  // major-to-minor layout.
  absl::Mutex transpose_mu_;
  TransposePlanCache transpose_cache_ ABSL_GUARDED_BY(transpose_mu_);

  std::shared_ptr<cpu::CollectivesInterface> collectives_;

  xla::TfrtCpuTopologyDescription topology_;

  // Used to control whether asynchronous computation dispatch is available for
  // this client. Only applies to non-parallel computations.
  bool asynchronous_;

  // Used to prevent too much parallelism: we will not enqueue next non-parallel
  // computation until last one is done within each user thread.
  // TODO(yueshengys): Consider moving the enqueuing/ordering logic to JAX via
  // token threading.
  inline static thread_local tsl::AsyncValueRef<CpuEvent> last_enqueue_event_ =
      tsl::MakeAvailableAsyncValueRef<CpuEvent>();
};

class TfrtCpuBuffer final : public AbstractTfrtCpuBuffer {
 public:
  TfrtCpuBuffer(
      Shape on_device_shape,
      std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer,
      TfrtCpuClient* client, TfrtCpuDevice* device,
      PjRtMemorySpace* memory_space);

  TfrtCpuBuffer(const TfrtCpuBuffer&) = delete;
  TfrtCpuBuffer(TfrtCpuBuffer&&) = delete;
  TfrtCpuBuffer& operator=(const TfrtCpuBuffer&) = delete;
  TfrtCpuBuffer& operator=(TfrtCpuBuffer&&) = delete;

  PjRtMemorySpace* memory_space() const override { return memory_space_; }
  TfrtCpuDevice* device() const override { return device_; }
  TfrtCpuClient* client() const override { return client_; }

  using PjRtBuffer::ToLiteralSync;
  PjRtFuture<> ToLiteral(MutableLiteralBase* literal) override;
  PjRtFuture<> LazyToLiteral(
      absl::AnyInvocable<absl::StatusOr<MutableLiteralBase*>() &&> generator)
      override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToDevice(
      PjRtDevice* dst_device) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      PjRtMemorySpace* dst_memory_space) override;

 private:
  absl::string_view buffer_name() const override { return "TfrtCpuBuffer"; }

  TfrtCpuClient* client_;
  TfrtCpuDevice* const device_;
  PjRtMemorySpace* const memory_space_;
};

class TfrtCpuExecutable final : public PjRtLoadedExecutable {
 public:
  TfrtCpuExecutable(
      int num_replicas, int num_partitions,
      std::shared_ptr<DeviceAssignment> device_assignment,
      bool parameter_is_tupled_arguments, CompileOptions compile_options,
      std::unique_ptr<Executable> cpu_executable,
      BufferAllocation::Index result_buffer_index,
      absl::InlinedVector<BufferAllocation::Index, 4> result_buffer_indices,
      std::vector<LogicalDeviceIds> addressable_device_logical_ids,
      std::vector<PjRtDevice*> addressable_devices, TfrtCpuClient* client);

  ~TfrtCpuExecutable() override = default;

  TfrtCpuClient* client() const override { return client_; }

  absl::string_view name() const override {
    return cpu_executable_->shared_module()->name();
  }

  int num_replicas() const override { return num_replicas_; }

  int num_partitions() const override { return num_partitions_; }

  int64_t SizeOfGeneratedCodeInBytes() const override {
    return cpu_executable_->SizeOfGeneratedCodeInBytes();
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

  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
    return std::vector<std::shared_ptr<HloModule>>{
        cpu_executable_->shared_module()};
  }

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override {
    return Unimplemented("GetOutputMemoryKinds is not supported.");
  }

  absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override {
    CompiledMemoryStats memory_stats = CompiledMemoryStats();
    memory_stats.generated_code_size_in_bytes = SizeOfGeneratedCodeInBytes();
    const HloProto* proto = cpu_executable_->hlo_proto();
    if (!proto) {
      return tsl::errors::FailedPrecondition(
          "cpu_executable_ has no hlo_proto.");
    }
    memory_stats.serialized_hlo_proto = proto->SerializeAsString();
    memory_stats.PopulateBufferStatsFromAllocations(
        cpu_executable_.get()->GetAllocations());
    return memory_stats;
  }

  using PjRtLoadedExecutable::Execute;
  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options,
      std::optional<std::vector<PjRtFuture<>>>& returned_futures) override;

  using PjRtLoadedExecutable::ExecuteSharded;
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<>>& returned_future, bool fill_future) override;

  using PjRtLoadedExecutable::ExecutePortable;
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<>>& returned_future, bool fill_future) override;

  void Delete() override;

  bool IsDeleted() override;

  absl::StatusOr<std::string> SerializeExecutable() const override;

  bool IsReturnedFutureSupported() const override { return true; }

  absl::StatusOr<std::optional<std::string>> Fingerprint() const;

  std::shared_ptr<Executable> cpu_executable() const { return cpu_executable_; }

  absl::StatusOr<std::string> FingerprintExecutable() const override {
    return Unimplemented("Fingerprinting executable is not supported.");
  }

  absl::StatusOr<CompileOptions> GetCompileOptions() const override {
    return compile_options_;
  }

 private:
  friend class TfrtCpuClient;

  absl::Status SetUpDonation(bool tuple_inputs);

  // Checks that the input buffers passed in by the user have the correct size
  // on device for the compiled program.
  absl::Status CheckBufferCompatibilities(
      absl::Span<std::pair<bool, TrackedTfrtCpuDeviceBuffer*> const>
          input_buffers) const;

  absl::StatusOr<Result> ExecuteHelper(
      absl::Span<PjRtBuffer* const> argument_handles, int replica,
      int partition, const RunId& run_id, const ExecuteOptions& options,
      tsl::AsyncValueRef<CpuEvent> last_collective_launch_event,
      bool fill_future, TfrtCpuDevice* device = nullptr);

  TfrtCpuClient* client_;

  int num_replicas_;
  int num_partitions_;
  std::shared_ptr<DeviceAssignment> device_assignment_;
  bool parameter_is_tupled_arguments_;
  CompileOptions compile_options_;

  std::shared_ptr<Executable> cpu_executable_;

  // Caching `result_buffer_index_` and `result_buffer_indices_` to avoid lookup
  // HLO dataflow analysis data structures in program execution critical path.

  // Buffer allocation index corresponding to root buffer buffer.
  BufferAllocation::Index result_buffer_index_;
  // Buffer allocation indices corresponding to each result buffer leaf buffer.
  absl::InlinedVector<BufferAllocation::Index, 4> result_buffer_indices_;

  // Size on device of each leaf buffer of the compiled program, cached here
  // for performance reasons.
  std::vector<int64_t> input_buffer_sizes_in_bytes_;

  // A sorted vector of parameters that have any aliased buffers and thus must
  // be donated when executing the computation.
  std::vector<int> parameters_that_must_be_donated_;

  // The replica and partition indices of device_assignment_ to be run by this
  // client. On single-host platforms without partitioning, this is all
  // replicas (i.e. addressable_device_logical_ids_[i] = (i, 0)), but this may
  // not be the case on multi-host platforms. If there are 4 replicas and 2
  // partitions on a single host platform, size of
  // addressable_device_logical_ids_ is 4*2 = 8.
  std::vector<LogicalDeviceIds> addressable_device_logical_ids_;

  // addressable_devices_[i] is the Device to which
  // addressable_device_logical_ids_[i] is assigned. shared_ptrs instead of
  // unique_ptrs to play well with the Python bindings (see xla.cc).
  std::vector<PjRtDevice*> addressable_devices_;

  // Cached result of comparing HloCostAnalysis FLOP estimate for execute
  // critical path.
  bool cheap_computation_;
};

struct CpuClientOptions {
  // Used to control whether asynchronous computation dispatch is available for
  // this client. Only applies to non-parallel computations, because collectives
  // may exist when there are multiple cpu devices and we need to do async
  // dispatch in that case. If it is set to be `false`, we will always run
  // computations inline.
  bool asynchronous = true;

  // Number of CPU devices. If not provided, the value of
  // --xla_force_host_platform_device_count is used.
  std::optional<int> cpu_device_count = std::nullopt;

  int max_inflight_computations_per_device = 32;

  // My process ID.
  int process_id = 0;

  // Distributed collectives implementation. Optional. If not provided, an
  // in-process collectives implementation will be used.
  std::shared_ptr<cpu::CollectivesInterface> collectives;
};
absl::StatusOr<std::unique_ptr<PjRtClient>> GetTfrtCpuClient(
    const CpuClientOptions& options);

// Deprecated. Use the overload that takes 'options' instead.
inline absl::StatusOr<std::unique_ptr<PjRtClient>> GetTfrtCpuClient(
    bool asynchronous) {
  CpuClientOptions options;
  options.asynchronous = asynchronous;
  return GetTfrtCpuClient(options);
}

// Deprecated. Use the overload that takes 'options' instead.
inline absl::StatusOr<std::unique_ptr<PjRtClient>> GetTfrtCpuClient(
    bool asynchronous, int cpu_device_count,
    int max_inflight_computations_per_device = 32) {
  CpuClientOptions options;
  options.asynchronous = asynchronous;
  options.cpu_device_count = cpu_device_count;
  options.max_inflight_computations_per_device =
      max_inflight_computations_per_device;
  return GetTfrtCpuClient(options);
}

}  // namespace xla

#endif  // XLA_PJRT_CPU_CPU_CLIENT_H_
