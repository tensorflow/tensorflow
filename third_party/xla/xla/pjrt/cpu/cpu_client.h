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
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/executable_run_options.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/compiled_memory_stats.h"
#include "xla/pjrt/cpu/cpu_device.h"
#include "xla/pjrt/cpu/cpu_device_memory.h"
#include "xla/pjrt/cpu/cpu_event.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/dynamic_shapes.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology_description.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/raw_pjrt_client.h"
#include "xla/pjrt/thread_pool_async_work_runner.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {

class PjRtCpuExecutable;

// Client-less CPU compilation for XlaComputation.
absl::StatusOr<std::unique_ptr<PjRtCpuExecutable>> CompileCpuExecutable(
    const XlaComputation& computation, CompileOptions options,
    const CpuTopologyDescription& topology,
    std::function<void(HloModuleConfig&)> customize_hlo_module_config =
        nullptr);

// Client-less CPU compilation for MLIR Module.
absl::StatusOr<std::unique_ptr<PjRtCpuExecutable>> CompileCpuExecutable(
    MaybeOwningMlirModule module, CompileOptions options,
    const CpuTopologyDescription& topology,
    std::function<void(HloModuleConfig&)> customize_hlo_module_config =
        nullptr);

class PjRtCpuRawClient : public PjRtRawClient {
 public:
  explicit PjRtCpuRawClient(
      std::shared_ptr<CpuDeviceMemory::Allocator> allocator,
      std::shared_ptr<cpu::CpuCollectives> collectives, size_t num_threads,
      bool asynchronous, int max_transpose_threads,
      std::function<void(HloModuleConfig&)> customize_hlo_module_config =
          nullptr);

  ~PjRtCpuRawClient() override;

  const std::function<void(HloModuleConfig&)>& customize_hlo_module_config()
      const {
    return customize_hlo_module_config_;
  }

  CpuDeviceMemory::Allocator* allocator() const { return allocator_.get(); }

  ThreadPoolAsyncWorkRunner* async_work_runner() const {
    return async_work_runner_.get();
  }

  tsl::thread::ThreadPool* eigen_intraop_pool() const {
    return eigen_intraop_pool_.get();
  }

  Eigen::ThreadPoolDevice* eigen_intraop_device() const {
    return eigen_intraop_device_.get();
  }

  cpu::CpuCollectives* collectives() const { return collectives_.get(); }

  bool asynchronous() const { return asynchronous_; }

  int max_transpose_threads() const { return max_transpose_threads_; }

  tsl::AsyncValueRef<CpuEvent> GetCollectiveLaunchEvent(
      RunId run_id, uint64_t executable_id, size_t num_addressable_devices,
      tsl::AsyncValueRef<CpuEvent> execute_event);

  absl::StatusOr<PjRtRawBufferRef> AllocateRawBuffer(
      PjRtMemorySpace* memory_space, size_t on_device_bytes_count,
      bool retry_on_oom, tsl::AsyncValueRef<bool> allocate_after) override;

  absl::StatusOr<PjRtRawBufferRef> AllocateRawBufferForExecute(
      PjRtMemorySpace* memory_space, size_t on_device_bytes_count,
      bool retry_on_oom) override;

  absl::StatusOr<std::pair<PjRtRawBufferRef,
                           CommonPjRtClient::PjRtFulfillAliasRawBufferCallback>>
  CreateRawBufferChannel(PjRtMemorySpace* memory_space,
                         size_t on_device_bytes_count) override;

  absl::StatusOr<std::pair<PjRtDeviceEventPromiseRef, PjRtDeviceEventRef>>
  CreateLinkedEventPromise(PjRtMemorySpace* memory_space,
                           absl::string_view debug_info) override;

  absl::StatusOr<PjRtDeviceEventRef> CreateDeviceEvent(
      PjRtMemorySpace* memory_space, Future<> dependency) override;

  absl::StatusOr<PjRtRawBufferRef> ImportForeignMemory(
      PjRtMemorySpace* memory_space, void* device_ptr, size_t size,
      absl::AnyInvocable<void() &&> on_delete_callback) override;

  void Stop();

 private:
  friend class PjRtCpuClient;
  friend class CpuExecutableLoadState;
  friend class CpuPjRtRawLoadedExecutable;

  std::shared_ptr<CpuDeviceMemory::Allocator> allocator_;
  std::shared_ptr<cpu::CpuCollectives> collectives_;
  bool asynchronous_;
  int max_transpose_threads_;

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

  // A callback to customize the HloModuleConfig for each compiled module.
  std::function<void(HloModuleConfig&)> customize_hlo_module_config_;

  // IMPORTANT: All thread pools must be destroyed first, because thread pool
  // destruction guarantees that all scheduled tasks are completed. Otherwise,
  // we might get use-after-free races when dispatched executables try to access
  // the member variables of this class that are already destroyed.
  std::unique_ptr<tsl::thread::ThreadPool> eigen_intraop_pool_;
  std::unique_ptr<Eigen::ThreadPoolDevice> eigen_intraop_device_;
  std::unique_ptr<ThreadPoolAsyncWorkRunner> async_work_runner_;
};

class PjRtCpuClient final : public CommonPjRtClient {
 public:
  ~PjRtCpuClient() override;

  PjRtCpuRawClient* raw_client() const override { return raw_client_.get(); }

  bool allow_fallback_for_donation() const override { return true; }
  // This is needed because CPU currently doesn't have per-device dispatching
  // threads for Execute() so two-phase launch can run into thread starvation.
  bool supports_two_phase_launch() const override { return false; }
  // TODO(parkers): implement proper predetermined error support.
  bool supports_predetermined_error() const override { return false; }

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

  PjRtDynamicShapeKind GetDynamicShapeKind(
      int memory_space_kind_id) const override {
    return PjRtDynamicShapeKind::kSuffix;
  }

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
  CompileAndAssignDevices(MaybeOwningMlirModule module, CompileOptions options);

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override;
  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      MaybeOwningMlirModule module, CompileOptions options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      const XlaComputation& computation, CompileOptions options) override;
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      MaybeOwningMlirModule module, CompileOptions options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Load(
      std::shared_ptr<PjRtExecutable> executable,
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

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
  LoadSerializedExecutable(const absl::Cord& serialized,
                           std::optional<CompileOptions> options,
                           const LoadOptions& load_options) override;

  AsyncWorkRunner* async_work_runner() const override {
    return raw_client_->async_work_runner();
  }

  bool IsOnCpu(PjRtMemorySpace* memory_space) override { return true; }

  absl::StatusOr<const xla::PjRtTopologyDescription*> GetTopologyDescription()
      const override {
    return topology_.get();
  }

  absl::StatusOr<PjRtRawBufferRef> AllocateRawBufferForExecute(
      PjRtMemorySpace* memory_space, size_t on_device_bytes_count,
      bool retry_on_oom) override {
    return raw_client_->AllocateRawBufferForExecute(
        memory_space, on_device_bytes_count, retry_on_oom);
  }

  absl::StatusOr<int> GetMemorySpaceKindForShape(
      const Shape& shape) const override;

  absl::StatusOr<PjRtDeviceEventRef> LinearizeHostBufferInto(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      const xla::Shape& device_shape, PjRtRawBufferRef raw_buffer) override;

  absl::StatusOr<PjRtDeviceEventRef> LinearizeInto(
      const LiteralSlice& literal, const xla::Shape& device_shape,
      HostBufferSemantics host_buffer_semantics,
      PjRtRawBufferRef raw_buffer) override;

  bool BufferFromHostBufferSupportsZeroCopy(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides, const Shape& shape,
      PjRtMemorySpace* memory_space,
      const Layout* device_layout) const override;

 private:
  friend class PjRtCpuLoadedExecutable;
  friend class CpuPjRtRawLoadedExecutable;
  friend class CpuExecutableLoadState;
  friend absl::StatusOr<std::unique_ptr<PjRtClient>> GetPjRtCpuClient(
      CpuClientOptions options);

  PjRtCpuClient(int process_index,
                std::vector<std::unique_ptr<PjRtCpuDevice>> devices,
                std::unique_ptr<PjRtCpuRawClient> raw_client,
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

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
  LoadSerializedExecutableInternal(google::protobuf::io::ZeroCopyInputStream* stream,
                                   std::optional<CompileOptions> options,
                                   const LoadOptions& load_options);

  int process_index_;
  // Includes all devices, including non-addressable devices.
  std::vector<std::unique_ptr<PjRtCpuDevice>> owned_devices_;
  // Pointers to `owned_devices_`.
  std::vector<PjRtDevice*> devices_;
  // Maps Device::id() to the corresponding Device. Includes all devices.
  absl::flat_hash_map<GlobalDeviceId, PjRtCpuDevice*> id_to_device_;
  // Addressable devices indexed by core_id.
  std::vector<PjRtDevice*> addressable_devices_;

  // Addressable memory spaces.
  std::vector<std::unique_ptr<PjRtMemorySpace>> owned_memory_spaces_;
  // Pointers to `owned_memory_spaces_`.
  std::vector<PjRtMemorySpace*> memory_spaces_;

  std::unique_ptr<PjRtCpuRawClient> raw_client_;

  std::unique_ptr<xla::CpuTopologyDescription> topology_;
};

class PjRtCpuLoadedExecutable;
class PjRtCpuExecutable;

class CpuPjRtRawLoadedExecutable : public PjRtRawLoadedExecutable {
 public:
  explicit CpuPjRtRawLoadedExecutable(RunId run_id) : run_id_(run_id) {}

  PjRtRawLoadedExecutable::RawExecuteResult Execute(
      const ExecuteOptions& options,
      absl::Span<const PjRtRawBufferRef> input_buffers,
      absl::Span<const PjRtRawBufferRef> output_leaf_buffers,
      PjRtDeviceEventRefVector extra_deps,
      PjRtDeviceEventRefVector control_deps, bool is_predetermined_error,
      bool fill_future) &&
      override;

 private:
  friend class PjRtCpuLoadedExecutable;
  friend class CpuExecutableLoadState;

  const PjRtCpuExecutable* executable_;
  std::shared_ptr<DeviceAssignment> device_assignment_;
  size_t num_addressable_devices_;
  PjRtCpuDevice* device_;
  PjRtCpuRawClient* raw_client_;
  RunId run_id_;
};

class CpuExecutableLoadState : public PjRtExecutableLoadState {
 public:
  explicit CpuExecutableLoadState(PjRtCpuRawClient* raw_client)
      : raw_client_(raw_client) {}

  ~CpuExecutableLoadState() override = default;

  void Delete() override { is_deleted_.store(true); }
  bool IsDeleted() const override { return is_deleted_.load(); }

  absl::StatusOr<std::unique_ptr<PjRtRawLoadedExecutable>> LoadRawExecutable(
      tsl::AsyncValueRef<PjRtExecutable> executable,
      const ExecuteOptions& options, size_t host_callback_idx,
      xla::RunId run_id, DeviceAndAssignment device_and_assign,
      int attempt) override;

  PjRtCpuRawClient* raw_client() const { return raw_client_; }

 private:
  PjRtCpuRawClient* raw_client_;
  std::atomic<bool> is_deleted_{false};
};

class PjRtCpuExecutable final : public PjRtExecutable {
 public:
  PjRtCpuExecutable(
      int num_replicas, int num_partitions, bool parameter_is_tupled_arguments,
      CompileOptions compile_options,
      std::unique_ptr<Executable> cpu_executable,
      absl::InlinedVector<BufferAllocation::Index, 4> result_buffer_indices,
      std::unique_ptr<HloModule> unoptimized_hlo_module,
      const CpuTopologyDescription& topology);

  ~PjRtCpuExecutable() override = default;

  absl::Status SetUpDonation(bool tuple_inputs);

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
  GetParameterMemoryKinds() const override;

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override;

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
  friend class CpuExecutableLoadState;

  int num_replicas_;
  int num_partitions_;
  bool parameter_is_tupled_arguments_;
  CompileOptions compile_options_;

  std::shared_ptr<cpu::CpuExecutable> cpu_executable_;

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

  const CpuTopologyDescription* topology_;
};

class PjRtCpuLoadedExecutable final : public CommonPjRtLoadedExecutable {
 public:
  using CommonPjRtLoadedExecutable::CommonPjRtLoadedExecutable;

  ~PjRtCpuLoadedExecutable() override = default;

  PjRtCpuExecutable* GetExecutable() const override {
    return absl::down_cast<PjRtCpuExecutable*>(
        CommonPjRtLoadedExecutable::GetExecutable());
  }

  PjRtCpuClient* client() const override {
    return absl::down_cast<PjRtCpuClient*>(
        CommonPjRtLoadedExecutable::client());
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

  const HloInputOutputAliasConfig& input_output_alias_config() const override {
    return GetExecutable()
        ->cpu_executable_->module()
        .input_output_alias_config();
  }
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
