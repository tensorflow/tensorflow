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
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "mlir/IR/BuiltinOps.h"
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/executable_run_options.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/cpu/cpu_device.h"
#include "xla/pjrt/cpu/cpu_event.h"
#include "xla/pjrt/cpu/tracked_cpu_device_buffer.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology_description.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/transpose.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

class PjRtCpuExecutable;

class PjRtCpuClient final : public CommonPjRtClient {
 public:
  ~PjRtCpuClient() override;

  bool allow_fallback_for_donation() const override { return true; }

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
      GlobalDeviceId global_device_id) const override;

  absl::StatusOr<PjRtDevice*> LookupAddressableDevice(
      LocalDeviceId local_device_id) const override;

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

  PjRtPlatformId platform_id() const override { return xla::CpuPlatformId(); }

  absl::string_view platform_name() const override {
    return xla::CpuPlatformName();
  }

  absl::string_view platform_version() const override {
    return xla::CpuPlatformVersion();
  }

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type, absl::Span<const int64_t> dims) override;

  absl::StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis()
      const override;

  // TODO(parkers): These should be moved to be fully client independent in
  // cpu_pjrt_compiler.cc.
  absl::StatusOr<std::pair<std::unique_ptr<PjRtCpuExecutable>,
                           std::shared_ptr<DeviceAssignment>>>
  CompileAndAssignDevices(const XlaComputation& computation,
                          CompileOptions options);
  absl::StatusOr<std::pair<std::unique_ptr<PjRtCpuExecutable>,
                           std::shared_ptr<DeviceAssignment>>>
  CompileAndAssignDevices(mlir::ModuleOp module, CompileOptions options);

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override;
  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      mlir::ModuleOp module, CompileOptions options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      const XlaComputation& computation, CompileOptions options) override;
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      mlir::ModuleOp module, CompileOptions options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Load(
      std::unique_ptr<PjRtExecutable> executable,
      const LoadOptions& load_options) override;

  // TODO(b/403584258): PJRT wants to have just one simple Compile API. When the
  // CPU runtime stops supporting the legacy runtime we will unify our compile
  // paths better and this will be redundant.
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
  CompileAheadOfTimeAndLoad(const XlaComputation& computation,
                            CompileOptions options,
                            const AotCompilationOptions& aot_options);

  // For PjRtCpuClient, `options` is mandatory.
  // This function returns an InvalidArgument error if `std::nullopt` is passed.
  // TODO(b/237720161): make it actually optional
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
  LoadSerializedExecutable(absl::string_view serialized,
                           std::optional<CompileOptions> options,
                           const LoadOptions& load_options) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateErrorBuffer(
      absl::Status error, const Shape& shape, PjRtMemorySpace* memory) override;

  absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(
      absl::Span<const PjRtClient::ShapeSpec> shape_specs,
      std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
      PjRtMemorySpace* memory_space) override;

  using PjRtClient::BufferFromHostLiteral;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffers(absl::Span<const Shape> shapes,
                              PjRtDevice* device,
                              PjRtCrossHostRecvNotifier notifier) override {
    return Unimplemented("MakeCrossHostReceiveBuffers not implemented.");
  }

  absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>> ImportForeignMemory(
      void* device_ptr, absl::AnyInvocable<void() &&> on_delete_callback,
      size_t on_device_bytes_count, PjRtMemorySpace* memory_space,
      bool is_mutable) override;

  tsl::thread::ThreadPool* pjrt_client_thread_pool() const {
    return pjrt_client_thread_pool_.get();
  }

  AsyncWorkRunner* async_work_runner() const override {
    return async_work_runner_.get();
  }

  tsl::thread::ThreadPool* eigen_intraop_pool() const {
    return eigen_intraop_pool_.get();
  }

  Eigen::ThreadPoolDevice* eigen_intraop_device() const {
    return eigen_intraop_device_.get();
  }

  bool IsOnCpu(PjRtMemorySpace* memory_space) override { return true; }

  tsl::AsyncValueRef<CpuEvent> GetCollectiveLaunchEvent(
      RunId run_id, uint64_t executable_id, size_t num_addressable_devices,
      tsl::AsyncValueRef<CpuEvent> execute_event);

  absl::StatusOr<const xla::PjRtTopologyDescription*> GetTopologyDescription()
      const override {
    return topology_.get();
  }

  absl::StatusOr<std::pair<tsl::RCReference<CommonPjRtRawBuffer>,
                           PjRtFulfillAliasRawBufferCallback>>
  CreateRawBufferChannel(PjRtMemorySpace* memory_space,
                         size_t on_device_bytes_count) override;

  absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>> AllocateRawBuffer(
      PjRtMemorySpace* memory_space, size_t on_device_bytes_count,
      bool retry_on_oom, tsl::AsyncValueRef<bool> allocate_after) override;

  absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>>
  AllocateRawBufferForExecute(PjRtMemorySpace* memory_space,
                              size_t on_device_bytes_count,
                              bool retry_on_oom) override;

  absl::StatusOr<std::pair<tsl::RCReference<PjRtDeviceEventPromise>,
                           tsl::RCReference<PjRtDeviceEvent>>>
  CreateLinkedEventPromise(PjRtMemorySpace* memory_space,
                           absl::string_view debug_info) override;

  std::unique_ptr<PjRtDeviceEventSet> CreateDeviceEventSet(
      size_t preallocated_size) const override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> DefineBuffer(
      const Shape& on_device_shape, PjRtMemorySpace* memory_space,
      tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
      absl::InlinedVector<tsl::RCReference<PjRtDeviceEvent>, 4>
          definition_device_events) override;

  absl::StatusOr<int64_t> GetOnDeviceBytesCount(
      PjRtMemorySpace* memory_space, const xla::Shape& shape) const override;

  absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> LinearizeHostBufferInto(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      const xla::Shape& device_shape,
      tsl::RCReference<CommonPjRtRawBuffer> raw_buffer) override;

  absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> LinearizeInto(
      const LiteralSlice& literal, const xla::Shape& device_shape,
      HostBufferSemantics host_buffer_semantics,
      tsl::RCReference<CommonPjRtRawBuffer> raw_buffer) override;

  absl::StatusOr<xla::Shape> MakeDefaultShapeForMemorySpace(
      PjRtMemorySpace* memory_space, xla::Shape shape,
      const xla::Layout* layout) const override;

  bool BufferFromHostBufferSupportsZeroCopy(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides, const Shape& shape,
      PjRtMemorySpace* memory_space,
      const Layout* device_layout) const override;

 private:
  friend class PjRtCpuLoadedExecutable;
  friend class CpuPjRtRawLoadedExecutable;
  friend absl::StatusOr<std::unique_ptr<PjRtClient>> GetPjRtCpuClient(
      CpuClientOptions options);

  PjRtCpuClient(
      int process_index, std::vector<std::unique_ptr<PjRtCpuDevice>> devices,
      std::shared_ptr<CpuDeviceMemory::Allocator> allocator,
      std::shared_ptr<cpu::CpuCollectives> collectives, size_t num_threads,
      bool asynchronous,
      std::function<void(HloModuleConfig&)> customize_hlo_module_config,
      int max_transpose_threads,
      std::unique_ptr<CpuTopologyDescription> topology);

  absl::StatusOr<std::pair<std::unique_ptr<PjRtCpuExecutable>,
                           std::shared_ptr<DeviceAssignment>>>
  CompileInternal(
      const XlaComputation& computation,
      const std::vector<const Shape*>& argument_layout_pointers,
      LayoutCanonicalizationCallback layout_canonicalization_callback,
      CompileOptions options,
      const AotCompilationOptions* absl_nullable aot_options = nullptr);

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> LoadInternal(
      std::shared_ptr<PjRtCpuExecutable> cpu_executable,
      std::shared_ptr<DeviceAssignment> device_assignment);

  CpuDeviceMemory::Allocator* allocator() const { return allocator_.get(); }

  int process_index_;
  // Includes all devices, including non-addressable devices.
  std::vector<std::unique_ptr<PjRtCpuDevice>> owned_devices_;
  // Pointers to `owned_devices_`.
  std::vector<PjRtDevice*> devices_;
  // Maps Device::id() to the corresponding Device. Includes all devices.
  absl::flat_hash_map<GlobalDeviceId, PjRtCpuDevice*> id_to_device_;
  // Addressable devices indexed by core_id.
  std::vector<PjRtDevice*> addressable_devices_;
  std::unique_ptr<ComputationPlacer> computation_placer_;

  // Addressable memory spaces.
  std::vector<std::unique_ptr<PjRtMemorySpace>> owned_memory_spaces_;
  // Pointers to `owned_memory_spaces_`.
  std::vector<PjRtMemorySpace*> memory_spaces_;

  // A memory allocator used to allocate host memory for PjRtBuffers, and
  // temporary allocations passed to XLA:CPU executable.
  std::shared_ptr<CpuDeviceMemory::Allocator> allocator_;

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
  struct CollectiveLaunchEventState {
    tsl::AsyncValueRef<CpuEvent> previous_event;
    tsl::CountDownAsyncValueRef<CpuEvent> countdown_event;
    size_t num_left_in_barrier;
  };
  absl::flat_hash_map<std::pair<RunId, uint64_t>, CollectiveLaunchEventState>
      launch_events_;
  tsl::AsyncValueRef<CpuEvent> last_collective_launch_event_
      ABSL_GUARDED_BY(mu_);

  // A cache for transpose plans. We use transposes to convert
  // (possibly strided) buffers provided to BufferFromHostBuffer into dense
  // major-to-minor layout.
  absl::Mutex transpose_mu_;
  TransposePlanCache transpose_cache_ ABSL_GUARDED_BY(transpose_mu_);

  std::shared_ptr<cpu::CpuCollectives> collectives_;

  std::unique_ptr<xla::CpuTopologyDescription> topology_;

  // Used to control whether asynchronous computation dispatch is available for
  // this client. Only applies to non-parallel computations.
  bool asynchronous_;

  // A callback to customize the HloModuleConfig for each compiled module.
  std::function<void(HloModuleConfig&)> customize_hlo_module_config_;

  // IMPORTANT: All thread pools must be destroyed first, because thread pool
  // destruction guarantees that all scheduled tasks are completed. Otherwise,
  // we might get use-after-free races when dispatched executables try to access
  // the member variables of this class that are already destroyed.

  // TODO(zhangqiaorjc): Use tsl::compat::EigenHostContextThreadPool.
  std::unique_ptr<tsl::thread::ThreadPool> eigen_intraop_pool_;
  std::unique_ptr<Eigen::ThreadPoolDevice> eigen_intraop_device_;

  // Thread pool for running PjRtClient tasks.
  std::unique_ptr<tsl::thread::ThreadPool> pjrt_client_thread_pool_;
  std::unique_ptr<AsyncWorkRunner> async_work_runner_;

  // Maximum number of threads to use for any one transpose. We will use the
  // the lesser of this number and the thread pool size. 1 = no threading.
  int max_transpose_threads_;
};

class PjRtCpuLoadedExecutable;
class PjRtCpuExecutable;

class CpuPjRtRawLoadedExecutable : public PjRtRawLoadedExecutable {
 public:
  explicit CpuPjRtRawLoadedExecutable(RunId run_id) : run_id_(run_id) {}
  PjRtDevice* device() override { return device_; }

  absl::Status Load(const ExecuteOptions& options,
                    size_t host_callback_idx) override {
    return absl::OkStatus();
  }

  PjRtRawLoadedExecutable::RawExecuteResult Execute(
      const ExecuteOptions& options,
      absl::Span<const tsl::RCReference<CommonPjRtRawBuffer>> input_buffers,
      absl::Span<const tsl::RCReference<CommonPjRtRawBuffer>>
          output_leaf_buffers,
      PjRtDeviceEventSet& extra_deps, PjRtDeviceEventSet& control_deps,
      bool is_predetermined_error, bool fill_future) &&
      override;

 private:
  friend class PjRtCpuLoadedExecutable;

  const PjRtCpuExecutable* executable_;
  std::shared_ptr<DeviceAssignment> device_assignment_;
  size_t num_addressable_devices_;
  PjRtCpuDevice* device_;
  PjRtCpuClient* client_;
  RunId run_id_;
};

class PjRtCpuExecutable final : public PjRtExecutable {
 public:
  PjRtCpuExecutable(
      int num_replicas, int num_partitions,
      bool parameter_is_tupled_arguments, CompileOptions compile_options,
      std::unique_ptr<Executable> cpu_executable,
      absl::InlinedVector<BufferAllocation::Index, 4> result_buffer_indices,
      std::unique_ptr<HloModule> unoptimized_hlo_module);

  ~PjRtCpuExecutable() override = default;

  absl::string_view name() const override {
    return cpu_executable_->shared_module()->name();
  }

  int num_replicas() const override { return num_replicas_; }

  int num_partitions() const override { return num_partitions_; }

  int64_t SizeOfGeneratedCodeInBytes() const override {
    return cpu_executable_->SizeOfGeneratedCodeInBytes();
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

  absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override;

  absl::StatusOr<std::string> SerializeExecutable() const override;

  std::shared_ptr<Executable> cpu_executable() const { return cpu_executable_; }

  absl::StatusOr<std::optional<std::string>> Fingerprint() const {
    return fingerprint_;
  }

  absl::StatusOr<std::string> FingerprintExecutable() const override {
    return fingerprint_;
  }

  absl::StatusOr<CompileOptions> GetCompileOptions() const override {
    return compile_options_;
  }

  const CompileOptions& compile_options() const { return compile_options_; }

 private:
  friend class PjRtCpuClient;
  friend class CpuPjRtRawLoadedExecutable;
  friend class PjRtCpuLoadedExecutable;

  absl::Status SetUpDonation(bool tuple_inputs);

  int num_replicas_;
  int num_partitions_;
  bool parameter_is_tupled_arguments_;
  CompileOptions compile_options_;

  std::shared_ptr<Executable> cpu_executable_;

  std::vector<Shape> parameter_device_shapes_;

  // Caching `result_buffer_indices_` to avoid lookup
  // HLO dataflow analysis data structures in program execution critical path.

  // Buffer allocation indices corresponding to each result buffer leaf buffer.
  absl::InlinedVector<BufferAllocation::Index, 4> result_buffer_indices_;
  // Reverse mapping of result_buffer_indices_.
  std::vector<int64_t> output_indices_;

  // Size on device of each leaf buffer of the compiled program, cached here
  // for performance reasons.
  std::vector<int64_t> input_buffer_sizes_in_bytes_;

  // A sorted vector of parameters that have any aliased buffers and thus must
  // be donated when executing the computation.
  std::vector<int> parameters_that_must_be_donated_;

  // Cached list of memory spaces per output.
  std::vector<int> output_memory_space_kind_ids_;

  // Cached result of comparing HloCostAnalysis FLOP estimate for execute
  // critical path.
  bool cheap_computation_;

  std::string fingerprint_;

  std::unique_ptr<HloModule> unoptimized_hlo_module_;
};

class PjRtCpuLoadedExecutable final : public CommonPjRtLoadedExecutable {
 public:
  PjRtCpuLoadedExecutable(
      std::shared_ptr<PjRtCpuExecutable> executable,
      std::shared_ptr<DeviceAssignment> device_assignment,
      std::vector<LogicalDeviceIds> addressable_device_logical_ids,
      std::vector<PjRtDevice*> addressable_devices, PjRtCpuClient* client);

  ~PjRtCpuLoadedExecutable() override = default;

  PjRtCpuExecutable* GetExecutable() const override {
    return executable_.get();
  }

  PjRtCpuClient* client() const override { return client_; }

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

  void Delete() override;

  bool IsDeleted() const override;

  const HloInputOutputAliasConfig& input_output_alias_config() const override {
    return executable_->cpu_executable_->module().input_output_alias_config();
  }

  void LaunchOnDevice(PjRtDevice* device,
                      absl::AnyInvocable<void()> execute_fn) const override {
    client()->async_work_runner()->Schedule(std::move(execute_fn));
  }

 private:
  friend class PjRtCpuClient;
  friend class CpuPjRtRawLoadedExecutable;

  absl::Status SetUpDonation(bool tuple_inputs);

  // Checks that the input buffers passed in by the user have the correct size
  // on device for the compiled program.
  absl::Status CheckBufferCompatibilities(
      absl::Span<const CommonPjRtBuffer::ScopedHold> input_buffers,
      absl::Span<PjRtBuffer* const> argument_handles) const;

  absl::StatusOr<std::unique_ptr<PjRtRawLoadedExecutable>> StartRawExecutable(
      const ExecuteOptions& options, RunId run_id, int replica, int partition,
      PjRtDevice* device) const override;

  absl::StatusOr<Result> ExecuteHelper(
      absl::Span<PjRtBuffer* const> argument_handles, int replica,
      int partition, const RunId& run_id, const ExecuteOptions& options,
      bool fill_future, PjRtCpuDevice* device = nullptr) const;

  PjRtCpuClient* client_;
  std::shared_ptr<PjRtCpuExecutable> executable_;
  std::shared_ptr<DeviceAssignment> device_assignment_;
};

absl::StatusOr<std::unique_ptr<PjRtClient>> ABSL_DEPRECATED(
    "Use public XLA:CPU GetXlaPjRtCpuClient instead")
    GetPjRtCpuClient(CpuClientOptions options);

// Deprecated. Use the overload that takes 'options' instead.
inline absl::StatusOr<std::unique_ptr<PjRtClient>> ABSL_DEPRECATED(
    "Use public XLA:CPU GetXlaPjRtCpuClient instead")
    GetPjRtCpuClient(bool asynchronous) {
  CpuClientOptions options;
  options.asynchronous = asynchronous;
  return GetPjRtCpuClient(std::move(options));
}

}  // namespace xla

#endif  // XLA_PJRT_CPU_CPU_CLIENT_H_
