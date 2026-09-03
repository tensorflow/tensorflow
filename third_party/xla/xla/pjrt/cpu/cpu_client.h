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
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/device_assignment.h"
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
    const XlaComputation& computation, CompileOptions&& options,
    const CpuTopologyDescription& topology,
    std::function<void(HloModuleConfig&)> customize_hlo_module_config =
        nullptr);

// Client-less CPU compilation for MLIR Module.
absl::StatusOr<std::unique_ptr<PjRtCpuExecutable>> CompileCpuExecutable(
    MaybeOwningMlirModule module, CompileOptions&& options,
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

  ThreadPoolAsyncWorkRunner* async_work_runner() const override {
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
      absl::AnyInvocable<void() &&> on_delete_callback,
      bool is_mutable) override;

  // TODO(b/403584258): PJRT wants to have just one simple Compile API. When the
  // CPU runtime stops supporting the legacy runtime we will unify our compile
  // paths better and this will be redundant.
  absl::StatusOr<std::unique_ptr<PjRtExecutable>> CompileAheadOfTime(
      const XlaComputation& computation, CompileOptions options,
      const CpuTopologyDescription& topology, int process_index,
      const AotCompilationOptions& aot_options);

  // TODO(parkers): These should be moved to be fully client independent in
  // cpu_pjrt_compiler.cc.
  absl::StatusOr<std::unique_ptr<PjRtCpuExecutable>> Compile(
      const XlaComputation& computation, const CpuTopologyDescription& topology,
      int process_index, CompileOptions&& options);
  absl::StatusOr<std::unique_ptr<PjRtCpuExecutable>> Compile(
      MaybeOwningMlirModule module, const CpuTopologyDescription& topology,
      int process_index, CompileOptions&& options);

  tsl::AsyncValueRef<PjRtExecutable> ToAsyncExecutable(
      std::shared_ptr<PjRtExecutable> executable) const override;

  tsl::RCReference<PjRtExecutableLoadState> MakeLoadState() override;

 private:
  friend class PjRtCpuClient;
  friend class CpuExecutableLoadState;
  friend class CpuPjRtRawLoadedExecutable;

  absl::StatusOr<std::unique_ptr<PjRtCpuExecutable>> CompileInternal(
      const XlaComputation& computation,
      const std::vector<const Shape*>& argument_layout_pointers,
      LayoutCanonicalizationCallback layout_canonicalization_callback,
      CompileOptions&& options, const CpuTopologyDescription& topology,
      int process_index,
      const AotCompilationOptions* absl_nullable aot_options = nullptr);

  // A memory allocator used to allocate host memory for PjRtBuffers, and
  // temporary allocations passed to XLA:CPU executable.
  std::shared_ptr<CpuDeviceMemory::Allocator> allocator_;

  std::shared_ptr<cpu::CpuCollectives> collectives_;

  // Used to control whether asynchronous computation dispatch is available for
  // this client. Only applies to non-parallel computations.
  bool asynchronous_;

  // Maximum number of threads to use for any one transpose. We will use the
  // the lesser of this number and the thread pool size. 1 = no threading.
  int max_transpose_threads_;

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

class PjRtCpuClient final : public CommonPjRtClientImpl {
 public:
  ~PjRtCpuClient() override;

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
  explicit CpuExecutableLoadState() = default;

  ~CpuExecutableLoadState() override = default;

  void Delete() override { is_deleted_.store(true); }
  bool IsDeleted() const override { return is_deleted_.load(); }

  absl::StatusOr<std::unique_ptr<PjRtRawLoadedExecutable>> LoadRawExecutable(
      tsl::AsyncValueRef<PjRtExecutable> executable,
      const ExecuteOptions& options, size_t host_callback_idx,
      xla::RunId run_id, DeviceAndAssignment device_and_assign,
      int attempt) override;

 private:
  std::atomic<bool> is_deleted_{false};
};

class PjRtCpuExecutable final : public PjRtExecutable {
 public:
  PjRtCpuExecutable(
      int num_replicas, int num_partitions, CompileOptions compile_options,
      std::unique_ptr<Executable> cpu_executable,
      absl::InlinedVector<BufferAllocation::Index, 4> result_buffer_indices,
      std::unique_ptr<HloModule> unoptimized_hlo_module,
      const CpuTopologyDescription& topology);

  ~PjRtCpuExecutable() override = default;

  absl::string_view name() const override {
    return cpu_executable_->shared_module()->name();
  }

  int num_replicas() const override { return num_replicas_; }

  int num_partitions() const override { return num_partitions_; }

  int64_t SizeOfGeneratedCodeInBytes() const override {
    return cpu_executable_->SizeOfGeneratedCodeInBytes();
  }

  absl::StatusOr<std::shared_ptr<HloModule>> GetHloModule() const override {
    return cpu_executable_->shared_module();
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

  std::optional<HloModuleProto> GetUnoptimizedHloModule() const override {
    if (!unoptimized_hlo_module_) {
      return std::nullopt;
    }
    return unoptimized_hlo_module_->ToProto();
  }

  static absl::StatusOr<std::unique_ptr<PjRtCpuExecutable>> Deserialize(
      riegeli::Any<riegeli::Reader*> reader,
      const xla::CpuTopologyDescription& topology,
      std::optional<CompileOptions>&& options);

 private:
  friend class PjRtCpuClient;
  friend class CpuPjRtRawLoadedExecutable;
  friend class PjRtCpuLoadedExecutable;
  friend class CpuExecutableLoadState;

  int num_replicas_;
  int num_partitions_;
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

  // Cached list of memory spaces per output.
  std::vector<int> output_memory_space_kind_ids_;

  // Cached result of comparing HloCostAnalysis FLOP estimate for execute
  // critical path.
  bool cheap_computation_;

  std::string fingerprint_;

  std::unique_ptr<HloModule> unoptimized_hlo_module_;

  const CpuTopologyDescription* topology_;
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
