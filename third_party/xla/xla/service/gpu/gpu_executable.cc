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

#include "xla/service/gpu/gpu_executable.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "riegeli/bytes/string_writer.h"
#include "riegeli/bytes/writer.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/annotation.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_memory_cache.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_buffer_conversion_pass.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_collective_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_buffer_debug_pass.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/backends/gpu/runtime/thunk_proto_deserialization.h"
#include "xla/client/executable_build_options.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/core/collectives/communicator.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/map_util.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/device_id.h"
#include "xla/runtime/hang_watchdog.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_value.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/maybe_owning_device_address.h"
#include "xla/service/rendezvous.h"
#include "xla/service/riegeli_dump_writer.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/stream_pool.h"
#include "xla/service/xla_debug_info_manager.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/kernel_stats.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/scoped_module_handle.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/stream_executor/vmm_device_address_allocator.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/env_time.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/sorted_range.h"
#include "xla/util.h"
#include "xla/util/split_proto/split_executable_and_options_writer.h"
#include "xla/util/split_proto/split_gpu_executable_writer.h"
#include "xla/xla.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/random.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace gpu {

namespace {

// Chooses the correct allocations to be used within the GpuExecutable code.
std::vector<const BufferAllocation*> GatherAllocationPtrs(
    const std::optional<std::vector<BufferAllocation>>& mlir_allocations,
    const BufferAssignment* buffer_assignment,
    const std::deque<BufferAllocation>& thunk_pass_allocations) {
  const std::vector<BufferAllocation>* allocation_vec = nullptr;
  if (mlir_allocations.has_value()) {
    allocation_vec = &mlir_allocations.value();
  } else if (buffer_assignment != nullptr) {
    allocation_vec = &buffer_assignment->Allocations();
  }

  std::vector<const BufferAllocation*> alloc_ptrs;
  if (allocation_vec != nullptr) {
    alloc_ptrs.reserve(allocation_vec->size());
    for (const BufferAllocation& alloc : *allocation_vec) {
      alloc_ptrs.push_back(&alloc);
    }
  }

  if (!thunk_pass_allocations.empty()) {
    alloc_ptrs.reserve(alloc_ptrs.size() + thunk_pass_allocations.size());
    for (const BufferAllocation& alloc : thunk_pass_allocations) {
      alloc_ptrs.push_back(&alloc);
    }
  }

  return alloc_ptrs;
}

class GpuExecutableThunkPassBufferAllocator : public ThunkPassBufferAllocator {
 public:
  ~GpuExecutableThunkPassBufferAllocator() override = default;

  explicit GpuExecutableThunkPassBufferAllocator(
      BufferAllocation::Index start_idx)
      : next_idx_(start_idx) {}

  absl::StatusOr<BufferAllocation* absl_nonnull> NewEmptyAllocation(
      int64_t size) override {
    allocations_.push_back(BufferAllocation(next_idx_++, size, /*color=*/0));
    return &allocations_.back();
  }

  std::deque<BufferAllocation>& MutableAllocations() { return allocations_; }

 private:
  BufferAllocation::Index next_idx_ = 0;
  // std::deque is used to ensure pointer stability.
  std::deque<BufferAllocation> allocations_;
};

absl::StatusOr<bool> ShouldCollectiveUseMinimalResource(
    const HloModule& module) {
  int64_t sync_collective_count = 0;
  int64_t total_sync_coll_size = 0;

  int64_t async_collective_count = 0;
  int64_t total_async_coll_size = 0;
  for (HloComputation* computation : module.MakeNonfusionComputations()) {
    for (HloInstruction* inst : computation->instructions()) {
      if (IsCollective(inst)) {
        TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                            inst->backend_config<GpuBackendConfig>());

        bool is_sync = gpu_backend_config.collective_backend_config().is_sync();
        int64_t total_size =
            ShapeUtil::ByteSizeOfElementsRecursive(inst->shape());

        if (is_sync) {
          sync_collective_count++;
          total_sync_coll_size += total_size;
        } else {
          async_collective_count++;
          total_async_coll_size += total_size;
        }
      }
    }
  }
  // Simple Heuristics to determine if we should minimize SM usage.
  // If we have more async collective count or larger message sizes
  // for async collectives, then we will choose to minimize SM
  // usage to give resource for computes.
  return (sync_collective_count <= async_collective_count ||
          total_sync_coll_size <= total_async_coll_size);
}
}  // namespace

using ::tsl::profiler::ScopedAnnotation;

// Returns the number of additional compute streams needed by all
// `AsyncStartThunks` in the `GpuExecutable`.
static int64_t GetNumAdditionalComputeStreams(ThunkExecutor& executor) {
  int64_t num_streams = 0;
  CHECK_OK(executor.thunks().WalkNested([&](Thunk* thunk) -> absl::Status {
    if (auto* async_start = dynamic_cast<AsyncStartThunk*>(thunk)) {
      auto stream_id = async_start->execution_stream_id();
      if (stream_id.is_computation()) {
        num_streams = std::max<int64_t>(num_streams,
                                        stream_id.computation_id().value() + 1);
      }
    }
    return absl::OkStatus();
  }));
  return num_streams;
}

static absl::Status RunThunkPasses(const DebugOptions& debug_options,
                                   const se::DeviceDescription& device_info,
                                   SequentialThunk* root_thunk,
                                   HloModule* hlo_module,
                                   ThunkPassBufferAllocator& allocator) {
  ThunkPassPipeline pipeline("thunk-passes");
  if (debug_options.xla_gpu_experimental_enable_checksum_tracing_on_thunks()) {
    pipeline.AddPass(std::make_unique<ThunkBufferDebugPass>(
        ThunkBufferDebugPass::Mode::kChecksum));
  }
  if (debug_options.xla_gpu_experimental_enable_buffer_saver_on_thunks()) {
    pipeline.AddPass(std::make_unique<ThunkBufferDebugPass>(
        ThunkBufferDebugPass::Mode::kBufferSaver));
  }
  if ((debug_options.xla_gpu_detect_nan() !=
       DebugOptions::DETECTION_MODE_NONE) ||
      (debug_options.xla_gpu_detect_inf() !=
       DebugOptions::DETECTION_MODE_NONE) ||
      debug_options.xla_gpu_log_minmax()) {
    LOG(ERROR) << "Adding ThunkBufferDebugPass for nan/inf/minmax checking";
    pipeline.AddPass(std::make_unique<ThunkBufferDebugPass>(
        ThunkBufferDebugPass::Mode::kFloatChecker));
  }
  pipeline.AddPass(std::make_unique<CommandBufferConversionPass>(
      hlo_module ? hlo_module->name() : "Anonymous"));

  ASSIGN_OR_RETURN(bool changed,
                   pipeline.Run(&root_thunk->thunks(), debug_options,
                                hlo_module, device_info, allocator));
  if (changed) {
    VLOG(3) << "Thunk passes changed the thunk tree.";
    if (hlo_module && DumpingEnabledForHloModule(*hlo_module)) {
      DumpToFileInDirOrStdout(
          *hlo_module, "",
          absl::StrCat("thunk_sequence_after_thunk_passes", ".txt"),
          root_thunk->ToString(/*indent=*/0));
    }
  }

  if (hlo_module && DumpingEnabledForHloModule(*hlo_module)) {
    ThunkMetadataListProto metadata_list_proto =
        GetMetadataListProtoFromThunkGraph(root_thunk->executor().thunks());
    DumpPerModuleProtobufToFile(*hlo_module, metadata_list_proto, debug_options,
                                "thunk_metadata");
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<GpuExecutable>> GpuExecutable::Create(
    Params params) {
  int64_t next_idx = 0;
  if (params.mlir_allocations.has_value()) {
    next_idx = params.mlir_allocations->size();
  } else if (params.buffer_assignment != nullptr) {
    next_idx = params.buffer_assignment->Allocations().size();
  }

  GpuExecutableThunkPassBufferAllocator allocator(next_idx);

  // TODO(b/461380690): Remove this once we have a better way to distinguish
  // between compiler-generated and runtime-loaded GPU executables.
  absl::StatusOr<std::vector<ThunkProto>> thunk_sequence_proto =
      [&]() -> absl::StatusOr<std::vector<ThunkProto>> {
    std::vector<ThunkProto> protos;
    protos.reserve(params.executable->thunks().size());
    for (const auto& thunk : params.executable->thunks()) {
      ASSIGN_OR_RETURN(ThunkProto proto, thunk->ToProto());
      protos.push_back(std::move(proto));
    }
    return protos;
  }();

  // Wrap the ThunkExecutor's thunks in a temporary SequentialThunk for running
  // thunk passes (which operate on SequentialThunk).
  auto seq_thunk = std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo(), std::move(params.executable->thunks()));
  RETURN_IF_ERROR(RunThunkPasses(params.debug_options,
                                 params.device_description, seq_thunk.get(),
                                 params.debug_module.get(), allocator));
  // Extract modified thunks back into a ThunkExecutor.
  auto executor =
      std::make_unique<ThunkExecutor>(std::move(seq_thunk->thunks()));

  return std::unique_ptr<GpuExecutable>(new GpuExecutable(
      std::move(params.debug_module), std::move(params.asm_text),
      std::move(params.binary), std::move(params.dnn_compiled_graphs),
      std::move(params.device_description), std::move(executor),
      std::move(params.module_name), std::move(params.program_shape),
      std::move(params.mlir_allocations), std::move(params.buffer_assignment),
      std::move(allocator.MutableAllocations()), std::move(params.alias_info),
      std::move(params.debug_options), std::move(params.constants),
      std::move(params.output_info), params.enable_debug_info_manager,
      std::move(params.module_stats), std::move(thunk_sequence_proto),
      std::move(params.executable_abi_version),
      std::move(params.cpu_target_machine_options)));
}

// Implementation note: HLO profiling is always enabled for GPU executables,
// since we can use timers around thunks.
GpuExecutable::GpuExecutable(
    std::unique_ptr<HloModule> debug_module, std::string asm_text,
    std::vector<uint8_t> binary, BinaryMap dnn_compiled_graphs,
    se::DeviceDescription device_description,
    std::unique_ptr<ThunkExecutor> executable, std::string module_name,
    ProgramShape program_shape,
    std::optional<std::vector<BufferAllocation>> mlir_allocations,
    std::unique_ptr<const BufferAssignment> buffer_assignment,
    std::deque<BufferAllocation> thunk_pass_allocations,
    std::unique_ptr<GpuAliasInfo> alias_info, DebugOptions debug_options,
    std::vector<ConstantInfo> constants,
    absl::flat_hash_map<ShapeIndex, OutputInfo> output_info,
    bool enable_debug_info_manager, ModuleStats module_stats,
    absl::StatusOr<std::vector<ThunkProto>> thunk_sequence_proto,
    stream_executor::ExecutableAbiVersion executable_abi_version,
    std::optional<xla::cpu::TargetMachineOptions> cpu_target_machine_options)
    : Executable(std::move(debug_module)),
      text_(std::move(asm_text)),
      binary_(std::move(binary)),
      dnn_compiled_graphs_(std::move(dnn_compiled_graphs)),
      gpu_version_(device_description.gpu_compute_capability()),
      thunk_executor_(std::move(executable)),
      num_additional_compute_streams_(
          GetNumAdditionalComputeStreams(*thunk_executor_)),
      module_name_(std::move(module_name)),
      program_shape_(std::move(program_shape)),
      allocation_ptrs_(GatherAllocationPtrs(
          mlir_allocations, buffer_assignment.get(), thunk_pass_allocations)),
      allocations_(std::move(mlir_allocations)),
      buffer_assignment_(std::move(buffer_assignment)),
      thunk_pass_allocations_(std::move(thunk_pass_allocations)),
      alias_info_(std::move(alias_info)),
      debug_buffer_assignment_show_max_(
          debug_options.xla_debug_buffer_assignment_show_max()),
      constants_(std::move(constants)),
      output_info_(std::move(output_info)),
      enable_debug_info_manager_(enable_debug_info_manager),
      thunk_sequence_proto_(std::move(thunk_sequence_proto)),
      executable_abi_version_(std::move(executable_abi_version)),
      cpu_target_machine_options_(std::move(cpu_target_machine_options)) {
  if (gpu_version_.IsRocm()) {
    // ROCm uses hsaco hashes to distinguish between modules.
    // Bad things happen if multiple modules with identical code are loaded.
    binary_.resize(binary_.size() + 16);
    *(uint64_t*)(&binary_[binary_.size() - 16]) = tsl::EnvTime::NowNanos();
    *(uint64_t*)(&binary_[binary_.size() - 8]) = tsl::random::New64();
  }
  if (has_module() && enable_debug_info_manager_) {
    XlaDebugInfoManager::Get()->RegisterModule(shared_module(),
                                               buffer_assignment_);
  }
  set_module_stats(std::move(module_stats));

  // Populate command_buffer_allocation_indexes_ with buffer indices accessed by
  // command buffer thunks. Skip constant and zero-size allocations since they
  // don't need VA remapping (constants are allocated as global values with
  // fixed addresses; zero-size allocations have nothing to map).
  //
  // The set of collected indices depends on xla_gpu_command_buffer_update_mode:
  //   ALWAYS_UPDATE - collect nothing (VA remapping disabled)
  //   NEVER_UPDATE - collect all allocations from all command buffer
  //     commands
  //   CAPTURE_CMD_NEVER_UPDATE - collect only allocations from traced
  //     commands (TracedCommandBufferCmd subclasses) and collective commands
  //     (CollectiveCmd subclasses)
  if (thunk_executor_) {
    DebugOptions::CommandBufferUpdateMode update_mode =
        has_module() ? module_config()
                           .debug_options()
                           .xla_gpu_command_buffer_update_mode()
                     : DebugOptions::ALWAYS_UPDATE;

    if (update_mode == DebugOptions::NEVER_UPDATE ||
        update_mode == DebugOptions::CAPTURE_CMD_NEVER_UPDATE) {
      CHECK_OK(thunk_executor_->thunks().WalkNested(
          [&](const Thunk* t) -> absl::Status {
            auto* cbt = dynamic_cast<const CommandBufferThunk*>(t);
            if (cbt == nullptr) return absl::OkStatus();
            return cbt->WalkCommands([&](const Command* cmd) -> absl::Status {
              if (update_mode == DebugOptions::CAPTURE_CMD_NEVER_UPDATE &&
                  !cmd->IsTracedCommand()) {
                return absl::OkStatus();
              }
              for (const BufferUse& use : cmd->buffer_uses()) {
                BufferAllocation::Index index = use.slice().index();
                if (buffer_assignment_) {
                  const auto& alloc = buffer_assignment_->GetAllocation(index);
                  if (alloc.is_constant() || alloc.size() == 0) continue;
                }
                command_buffer_allocation_indexes_.insert(index);
              }
              return absl::OkStatus();
            });
          }));
      VLOG(3) << "VA remapping: collected "
              << command_buffer_allocation_indexes_.size()
              << " allocation indexes for module " << module_name_;
    }
    // update_mode == ALWAYS_UPDATE: collect nothing.
  }
}

GpuExecutable::~GpuExecutable() {
  if (has_module() && enable_debug_info_manager_) {
    XlaDebugInfoManager::Get()->UnregisterModule(module().unique_id());
  }
}

absl::Status GpuExecutable::CheckCompatibilityWithServiceExecutableRunOptions(
    const ServiceExecutableRunOptions* run_options) {
  se::Stream* main_stream = run_options->stream();

  stream_executor::Platform::Id platform_id =
      main_stream->parent()->GetPlatform()->id();
  if (platform_id == stream_executor::rocm::kROCmPlatformId) {
    auto cc = main_stream->GetRocmComputeCapability();
    std::string stream_arch = cc.gcn_arch_name();
    std::string gpu_exec_arch =
        gpu_version_.rocm_compute_capability()->gcn_arch_name();
    TF_RET_CHECK(stream_arch == gpu_exec_arch)
        << "AMDGPU GCN ISA version mismatch; expected {" << gpu_exec_arch
        << ", but was " << stream_arch;
  } else if (platform_id == stream_executor::cuda::kCudaPlatformId) {
    se::CudaComputeCapability cc = main_stream->GetCudaComputeCapability();
    TF_RET_CHECK(cc == *gpu_version_.cuda_compute_capability())
        << "Compute capability mismatch; expected {" << gpu_version_.ToString()
        << "}, but was {" << cc.ToString() << "}";
  } else if (platform_id == stream_executor::sycl::kSyclPlatformId) {
    // TODO: Add check.
  } else {
    return Internal("Unknown platform");
  }

  return absl::OkStatus();
}

namespace {

absl::Status MaybeSyncAndProfile(const ServiceExecutableRunOptions* run_options,
                                 se::EventBasedTimer* execution_timer,
                                 se::Stream* stream_to_sync);

absl::Status RendezvousAfterInitialization(
    const ServiceExecutableRunOptions& run_options,
    const DebugOptions* absl_nullable debug_options);

absl::Status BarrierAfterExecutable(
    const ServiceExecutableRunOptions& run_options,
    const DebugOptions* absl_nullable debug_options, se::Stream& stream_to_sync,
    size_t num_participants);

absl::Status ExecuteThunksImpl(const DebugOptions* debug_options,
                               const std::string& module_name,
                               ModuleIdentifier module_id,
                               ThunkExecutor& thunk_executor,
                               Thunk::ExecutableSource executable_source,
                               const ServiceExecutableRunOptions* run_options,
                               const BufferAllocations& buffer_allocations,
                               bool block_host_until_done,
                               int64_t num_additional_compute_streams,
                               CollectiveMemoryCache& collective_memory_cache,
                               bool collective_use_minimal_resource) {
  bool mock_collectives =
      run_options->run_options().gpu_executable_run_options()
          ? run_options->run_options()
                .gpu_executable_run_options()
                ->enable_mock_collectives()
          : false;

  int64_t collective_max_nchannels =
      debug_options ? debug_options->xla_gpu_nccl_collective_max_nchannels()
                    : 0;
  int64_t p2p_max_nchannels =
      debug_options ? debug_options->xla_gpu_nccl_p2p_max_nchannels() : 0;
  bool use_highest_priority_for_async_stream =
      debug_options
          ? debug_options->xla_gpu_enable_highest_priority_async_stream()
          : false;

  se::Stream* main_stream = run_options->stream();
  se::StreamExecutor* executor = main_stream->parent();
  stream_executor::StreamPriority stream_priority =
      stream_executor::StreamPriority::Default;
  // TODO(intel-tf): Enable stream priorities for sycl backend.
  if (executor->GetPlatform()->id() == stream_executor::sycl::kSyclPlatformId) {
    use_highest_priority_for_async_stream = false;
  }
  if (use_highest_priority_for_async_stream) {
    stream_priority = stream_executor::StreamPriority::Highest;
  }

  // Maybe install progress tracker for this execution.
  int32_t progress_tracking_n =
      debug_options ? debug_options->xla_gpu_execution_progress_tracking() : 0;

  std::optional<ThunkExecutor::ScopedProgressTracker> tracker;
  if (progress_tracking_n > 0) {
    ASSIGN_OR_RETURN(tracker, InstallProgressTracker(executor, thunk_executor));
  }

  // Maybe add a watch guard for this execution.
  absl::Duration watchdog_timeout = absl::InfiniteDuration();
  if (debug_options &&
      !debug_options->xla_gpu_execution_terminate_timeout().empty()) {
    TF_RET_CHECK(absl::ParseDuration(
        debug_options->xla_gpu_execution_terminate_timeout(),
        &watchdog_timeout))
        << "Failed to parse XLA execution terminate timeout";
  }

  std::shared_ptr<HangWatchdog::Guard> guard = nullptr;
  if (watchdog_timeout < absl::InfiniteDuration()) {
    int32_t device_ordinal = executor->device_ordinal();
    std::string watchdog_name = absl::StrFormat("[%d] XLA GPU execution `%s`",
                                                device_ordinal, module_name);

    // If we have installed progress tracker, log how far thunk execution
    // progressed before getting stuck. This is helpful for identifying kernels
    // that never finish and stall the stream execution.
    HangWatchdog::CancelCallback pre_abort;
    if (tracker.has_value()) {
      pre_abort = [&tracker, progress_tracking_n, device_ordinal] {
        auto log_progress = [&](auto label, auto thunks) {
          LOG(ERROR) << absl::StreamFormat("[%d] %s: size=%d", device_ordinal,
                                           label, thunks.size());
          // We want to report all thunks in chronological order for
          // readability according to the time they were executed.
          absl::c_sort(thunks, [](const auto& a, const auto& b) {
            return a.executed < b.executed;
          });
          for (auto& thunk : thunks) {
            std::string loop_info;
            for (const auto& state : thunk.loop_nest) {
              absl::StrAppend(&loop_info,
                              absl::StrFormat(" [%s iter=%d]", state.loop_name,
                                              state.loop_iteration));
            }
            LOG(ERROR) << absl::StreamFormat(
                "  - exec[%d] thunk[%d/%d] %v: %s at %s%s", thunk.exec_idx,
                thunk.thunk_idx, tracker->num_thunks(), thunk.kind, thunk.name,
                absl::FormatTime("%Y-%m-%d %H:%M:%S.%E6f", thunk.executed,
                                 absl::LocalTimeZone()),
                loop_info);
          }
        };

        size_t num_executions = tracker->num_executions();
        LOG(ERROR) << absl::StreamFormat(
            "[%d] Completed thunks: %d/%d (unique thunks: %d)", device_ordinal,
            tracker->NumCompletedThunks(), num_executions,
            tracker->num_thunks());
        LOG(ERROR) << absl::StreamFormat(
            "[%d] Pending thunks: %d/%d (unique thunks: %d)", device_ordinal,
            tracker->NumPendingThunks(), num_executions, tracker->num_thunks());

        log_progress("Last completed thunks",
                     tracker->LastCompletedThunks(progress_tracking_n));
        log_progress("First pending thunks",
                     tracker->FirstPendingThunks(progress_tracking_n));
        log_progress("Last pending thunks",
                     tracker->LastPendingThunks(progress_tracking_n));
      };
    }

    guard = HangWatchdog::Global().Watch(
        watchdog_name, watchdog_timeout,
        HangWatchdog::Abort(watchdog_name, watchdog_timeout,
                            std::move(pre_abort)));
  }

  constexpr int64_t kAsyncStreamTotal =
      static_cast<int64_t>(AsyncStreamKind::ASYNC_STREAM_KIND_MEMCPYP2P) + 1;

  // Borrow streams required for CollectiveThunk.
  absl::InlinedVector<se::Stream*, kAsyncStreamTotal> async_comms_streams(
      kAsyncStreamTotal, nullptr);
  se::Stream* command_buffer_trace_stream = nullptr;
  std::vector<StreamPool::Ptr> async_comms_streams_owner;
  StreamPool::Ptr borrowed_command_buffer_trace_stream;
  if (run_options->HasStreamBorrower()) {
    ASSIGN_OR_RETURN(
        async_comms_streams_owner,
        run_options->BorrowStreams(executor->device_ordinal(),
                                   kAsyncStreamTotal, stream_priority));
    for (int64_t i = 0; i < kAsyncStreamTotal; ++i) {
      async_comms_streams[i] = async_comms_streams_owner[i].get();
    }

    // Borrow stream for tracing command buffers.
    ASSIGN_OR_RETURN(borrowed_command_buffer_trace_stream,
                     run_options->BorrowStream(executor->device_ordinal()));
    command_buffer_trace_stream = borrowed_command_buffer_trace_stream.get();
  }

  // Borrow streams for additional compute streams.
  std::vector<se::Stream*> additional_compute_streams;
  std::vector<StreamPool::Ptr> borrowed_compute_streams;
  if (num_additional_compute_streams > 0) {
    if (run_options->HasStreamBorrower()) {
      ASSIGN_OR_RETURN(
          borrowed_compute_streams,
          run_options->BorrowStreams(executor->device_ordinal(),
                                     num_additional_compute_streams));
      additional_compute_streams.reserve(num_additional_compute_streams);
      for (auto& stream : borrowed_compute_streams) {
        additional_compute_streams.push_back(stream.get());
      }
      XLA_VLOG_DEVICE(2, run_options->device_ordinal())
          << absl::StreamFormat("Using %d additional compute streams.",
                                num_additional_compute_streams);
    } else {
      XLA_VLOG_DEVICE(2, run_options->device_ordinal())
          << "No stream borrower created. "
          << "Assigning the default stream to all parallel computes.";
      additional_compute_streams.assign(num_additional_compute_streams,
                                        main_stream);
    }
  }

  tsl::profiler::TraceMe hlo_module_activity(
      [&] { return absl::StrCat(module_name, ":XLA GPU module"); },
      tsl::profiler::TraceMeLevel::kInfo);

  std::unique_ptr<se::EventBasedTimer> execution_timer;
  if (ExecutionProfile* profile =
          run_options->run_options().execution_profile();
      profile) {
    ASSIGN_OR_RETURN(execution_timer, main_stream->CreateEventBasedTimer(
                                          profile->warmup_run_executed()));
  }

  // A state container for this execution.
  Thunk::ExecutionScopedState execution_scoped_state;

  // Parameters for executing collective operations.
  std::optional<std::string> collectives_impl_name;
  if (debug_options &&
      !debug_options->xla_gpu_collectives_implementation().empty()) {
    collectives_impl_name = debug_options->xla_gpu_collectives_implementation();
  }

  ASSIGN_OR_RETURN(
      CollectiveParams collective_params,
      CollectiveParams::Create(
          *run_options, async_comms_streams,
          LocalDeviceId(main_stream->parent()->device_ordinal()),
          std::move(collectives_impl_name), collective_max_nchannels,
          p2p_max_nchannels, collective_use_minimal_resource));

  CollectiveCliqueRequests collective_clique_requests;
  CollectiveMemoryRequests collective_memory_requests(buffer_allocations);

  {  // Prepare thunks for execution and collect requested GPU cliques.
    Thunk::PrepareParams prepare_params{
        &collective_params,          &collective_clique_requests,
        &collective_memory_requests, executor,
        &buffer_allocations,         &execution_scoped_state};

    tsl::profiler::TraceMe trace_prepare("Thunks::Prepare");
    RETURN_IF_ERROR(thunk_executor.Prepare(prepare_params));
  }

  XLA_VLOG_DEVICE(3, run_options->device_ordinal()) << absl::StreamFormat(
      "Prepared GPU executable module: %s for execution: "
      "#collective=[cliques=%d, symmetric=%d]",
      module_name, collective_clique_requests.size(),
      collective_memory_requests.symmetric_size());

  std::vector<std::unique_ptr<CliqueKey>>* clique_keys =
      run_options->run_options().clique_keys();
  if (clique_keys != nullptr) {
    for (const GpuCliqueKey& clique_key :
         collective_clique_requests.RequestedCliques()) {
      clique_keys->push_back(std::make_unique<GpuCliqueKey>(clique_key));
    }
  }

  // Acquire collective cliques requested by thunks.
  CollectiveCliques collective_cliques;
  if (!mock_collectives) {
    ASSIGN_OR_RETURN(collective_cliques,
                     AcquireCollectiveCliques(collective_params,
                                              collective_clique_requests));
  }

  // Acquire collective memories requested by thunks.
  ASSIGN_OR_RETURN(CollectiveMemory collective_memory,
                   AcquireCollectiveMemory(
                       collective_params, collective_cliques,
                       collective_memory_requests, collective_memory_cache));
  {  // Initialize thunks using prepared resources before execution.
    Thunk::InitializeParams initialize_params{
        executor,
        executable_source,
        &buffer_allocations,
        main_stream,
        command_buffer_trace_stream,
        &collective_params,
        &collective_cliques,
        &collective_memory,
        run_options->run_options().ffi_execution_context(),
        run_options->local_device_count(),
        &execution_scoped_state};

    tsl::profiler::TraceMe trace_initialize("Thunks::Initialize");
    RETURN_IF_ERROR(thunk_executor.Initialize(initialize_params));
  }

  // Join a round of rendezvous after thunk initialization. We do this only in
  // presence of newly acquired collective cliques which means that we have
  // collective operations and clique initialization is famous for introducing
  // deadlocks if we try to execute it concurrently with other potentially
  // memory-allocating operations.
  if (!collective_cliques.empty()) {
    RETURN_IF_ERROR(RendezvousAfterInitialization(*run_options, debug_options));
  }

  // Prepare parameters for thunks execution.
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      *run_options, buffer_allocations, main_stream,
      command_buffer_trace_stream, &collective_params, &collective_cliques,
      &collective_memory, std::move(additional_compute_streams),
      &execution_scoped_state);

  XLA_VLOG_DEVICE(1, run_options->device_ordinal())
      << "Start GpuExecutable::ExecuteOnStream module: " << module_name;
  RETURN_IF_ERROR(thunk_executor.ExecuteOnStream(execute_params));
  XLA_VLOG_DEVICE(1, run_options->device_ordinal())
      << "End GpuExecutable::ExecuteOnStream module: " << module_name;

  // Collective kernel thunks may request a barrier after the module execution.
  // This might be needed for several reasons:
  // 1. To make sure that at the end of graph execution all reads and writes to
  //    the symmetric buffers are finished.
  // 2. To make sure that cuda module which uses a multimem handler used by
  //    another GPU will be unloaded only after all kernels are finished.
  //    Otherwise module unloading can cause a deadlock.
  absl::flat_hash_set<GlobalDeviceId> requested_barrier_devices =
      collective_clique_requests.GetDevicesRequiringBarrier();
  if (absl::c_linear_search(requested_barrier_devices,
                            collective_params.global_device_id.value())) {
    XLA_VLOG_DEVICE(1, collective_params.global_device_id.value())
        << "Barrier after executable required by participants: ("
        << absl::StrJoin(requested_barrier_devices, ", ") << ")";
    RETURN_IF_ERROR(BarrierAfterExecutable(*run_options, debug_options,
                                           *main_stream,
                                           requested_barrier_devices.size()));
  }

  return MaybeSyncAndProfile(run_options, execution_timer.get(),
                             block_host_until_done ? main_stream : nullptr);
}

namespace {
// Wrap RunId into a unique struct to guarantee we do not accidentally try to
// run multiple unrelated rendezvous for a same key.
struct InitializationKey {
  RunId run_id;

  template <typename H>
  friend H AbslHashValue(H h, const InitializationKey& key) {
    return H::combine(std::move(h), key.run_id);
  }
};

bool operator==(const InitializationKey& a, const InitializationKey& b) {
  return a.run_id == b.run_id;
}
}  // namespace

absl::Status RendezvousAfterInitialization(
    const ServiceExecutableRunOptions& run_options,
    const DebugOptions* absl_nullable debug_options) {
  // Thunk initialization can allocate new control data structures on device
  // that can lead to deadlocks if other replicas are executing concurrently
  // (i.e. this happens if we try to instantiate CUDA graph when other replica
  // is executing NCCL kernels). If we detect that we are running in multi-gpu
  // setup we synchronize after first initialization to make sure that all
  // replicas completed initialization process before we start execution.
  auto* gpu_opts = run_options.run_options().gpu_executable_run_options();
  auto* device_assn = run_options.run_options().device_assignment();

  // If we don't have Gpu executable options or device assignment it means we
  // are running in a single Gpu config and don't need a rendezvous.
  if (!gpu_opts || !device_assn) {
    return absl::OkStatus();
  }

  // Assume that all participants execute locally first, if we have a local
  // device id to global device id map we will use it to get the real number of
  // participating local devices.
  int64_t num_local_participants =
      device_assn->replica_count() * device_assn->computation_count();

  // Find what local devices are part of the device assignment.
  if (gpu_opts->gpu_global_device_ids()) {
    auto d2l_map = device_assn->GetDeviceToLogicalIdMap();

    num_local_participants = 0;
    for (auto& [local_id, global_id] : *gpu_opts->gpu_global_device_ids()) {
      num_local_participants += d2l_map.contains(global_id);
    }

    if (num_local_participants == 0) {
      return absl::InternalError(
          "Cound't find the number of local participants");
    }
  }

  XLA_VLOG_DEVICE(1, run_options.device_ordinal()) << absl::StreamFormat(
      "Join thunks initialization rendezvous with %d local participants",
      num_local_participants);

  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode(
        "RendezvousAfterInitialization",
        {{"run_id", run_options.run_options().run_id().ToInt()},
         {"num_local_participants", num_local_participants}});
  });

  auto rendezvous_key = InitializationKey{run_options.run_options().run_id()};
  auto rendezvous_name = absl::StrFormat(
      "thunk initialization completion for device ordinal %d; run_id=%d",
      run_options.device_ordinal(), run_options.run_options().run_id().ToInt());

  return Rendezvous(
      rendezvous_name, rendezvous_key, num_local_participants,
      absl::Seconds(
          debug_options
              ? debug_options->xla_gpu_executable_warn_stuck_timeout_seconds()
              : 10),
      absl::Seconds(
          debug_options
              ? debug_options->xla_gpu_executable_terminate_timeout_seconds()
              : 30));
}

absl::Status MaybeSyncAndProfile(const ServiceExecutableRunOptions* run_options,
                                 se::EventBasedTimer* execution_timer,
                                 se::Stream* stream_to_sync = nullptr) {
  // If we're measuring the execution time then it's important to queue the
  // stop event before triggering any synchronization.
  if (ExecutionProfile* profile =
          run_options->run_options().execution_profile();
      profile) {
    ASSIGN_OR_RETURN(absl::Duration elapsed,
                     execution_timer->GetElapsedDuration());
    profile->set_compute_time_ns(
        std::max(absl::ToDoubleNanoseconds(elapsed), 1.0));
  }

  // Make sure kernels are completed before deallocating temporary buffers or
  // the profiler state.
  // TODO(b/30100571): we could potentially postpone deallocating the temp
  // buffers until a different computation is executed.
  if (stream_to_sync) {
    absl::Status block_status = stream_to_sync->BlockHostUntilDone();
    if (!block_status.ok()) {
      return Internal(
          "Failed to complete all kernels launched on stream %p: %s",
          stream_to_sync, block_status.message());
    }
  }

  return absl::OkStatus();
}

absl::Status BarrierAfterExecutable(
    const ServiceExecutableRunOptions& run_options,
    const DebugOptions* absl_nullable debug_options, se::Stream& stream,
    const size_t num_participants) {
  if (debug_options != nullptr &&
      debug_options->xla_gpu_experimental_enable_nvshmem()) {
    ASSIGN_OR_RETURN(auto* collectives, GetNvshmemCollectivesFromRegistry());
    ASSIGN_OR_RETURN(std::unique_ptr<Communicator> nvshmem_comm,
                     collectives->CreateCommunicator());

    RETURN_IF_ERROR(nvshmem_comm->Barrier(GpuCollectives::On(stream)));
  } else {
    RETURN_IF_ERROR(stream.BlockHostUntilDone());

    XLA_VLOG_DEVICE(1, run_options.device_ordinal()) << absl::StreamFormat(
        "Join thunks in barrier after module execution rendezvous with %d "
        "local "
        "participants",
        num_participants);

    tsl::profiler::TraceMe trace([&] {
      return tsl::profiler::TraceMeEncode(
          "RendezvousAfterExecution",
          {{"run_id", run_options.run_options().run_id().ToInt()},
           {"num_local_participants", num_participants}});
    });

    auto rendezvous_key = InitializationKey{run_options.run_options().run_id()};
    auto rendezvous_name = absl::StrFormat(
        "thunk barrier after module execution completion for device ordinal "
        "%d; run_id=%d",
        run_options.device_ordinal(),
        run_options.run_options().run_id().ToInt());

    return Rendezvous(
        rendezvous_name, rendezvous_key, num_participants,
        absl::Seconds(
            debug_options
                ? debug_options->xla_gpu_executable_warn_stuck_timeout_seconds()
                : 10),
        absl::Seconds(
            debug_options
                ? debug_options->xla_gpu_executable_terminate_timeout_seconds()
                : 30));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<const GpuExecutable::BufferAllocToDeviceMemoryMap*>
GpuExecutable::ResolveConstantGlobals(se::Stream* stream) {
  se::StreamExecutor* executor = stream->parent();

  absl::MutexLock lock(module_handle_mutex_);
  auto it = module_globals_.find(executor);
  if (it != module_globals_.end()) {
    return it->second.get();
  }

  se::MultiModuleLoaderSpec module_spec;
  if (!binary().empty()) {
    module_spec.AddCudaCubinInMemory(binary());
  }
  module_spec.AddCudaPtxInMemory(text().c_str());

  auto globals = std::make_unique<BufferAllocToDeviceMemoryMap>();
  se::ModuleHandle module_handle;
  // The CUDA driver isn't able to load a PTX and a binary which are both empty.
  // It's okay if we skip loading in this case; if the module isn't loaded, all
  // symbol lookups will fail, just as they should for an empty module.
  if (!(executor->GetPlatform()->id() ==
            stream_executor::cuda::kCudaPlatformId &&
        binary().empty() && text().empty())) {
    ASSIGN_OR_RETURN(module_handle, executor->LoadModule(module_spec));
  }

  // A flag signalling if constant initialization submitted memcpy operations
  // to the `stream`.
  int submitted_mem_copies = 0;

  for (const ConstantInfo& info : constants_) {
    absl::StatusOr<stream_executor::DeviceAddressBase> global_status;
    if (static_cast<bool>(module_handle)) {
      global_status = executor->GetSymbol(info.symbol_name, module_handle);
    }

    se::DeviceAddressBase global;

    CHECK(static_cast<bool>(module_handle) && global_status.ok());
    // The constant was defined in the PTX and has been allocated by the CUDA
    // driver.
    global = *global_status;
    XLA_VLOG_DEVICE(3, executor->device_ordinal()) << absl::StreamFormat(
        "Resolved global %s to %p", info.symbol_name, global.opaque());

    if (!info.content.span().empty()) {
      // This means the constant did not have an initializer in the PTX and
      // therefore must be initialized by XLA here.
      RETURN_IF_ERROR(stream->Memcpy(&global, info.content.span().data(),
                                     info.content.span().size()));
      submitted_mem_copies = true;
    }

    if (info.allocation_index != -1) {
      InsertOrDie(globals.get(), info.allocation_index, global);
    }
  }

  // Wait for the completion of all host->device transfers, to guarantee that
  // destructor will not race with any operations in flight (deallocate
  // xla::Literal owned by the HLO module).
  if (submitted_mem_copies) {
    CHECK_OK(stream->BlockHostUntilDone());
  }

  module_handles_.emplace(executor,
                          se::ScopedModuleHandle(executor, module_handle));
  return module_globals_.emplace(executor, std::move(globals))
      .first->second.get();
}

absl::StatusOr<se::DeviceAddressBase> GpuExecutable::BufferForAllocation(
    VariantArguments arguments,
    const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
    const BufferAllocation& allocation,
    se::DeviceAddressAllocator* const memory_allocator, int device_ordinal,
    int64_t arg_idx,
    const absl::flat_hash_map<LogicalBuffer::Color, int64_t>&
        allocate_granularity) {
  if (allocation.is_thread_local()) {
    return se::DeviceAddressBase{};
  } else if (allocation.is_entry_computation_parameter()) {
    int64_t param_no = allocation.parameter_number();
    se::DeviceAddressBase registered_buffer = [&] {
      if (auto unowned_shapedbuffers =
              std::get_if<absl::Span<const ShapedBuffer* const>>(&arguments)) {
        return (*unowned_shapedbuffers)[param_no]->buffers().element(
            allocation.param_shape_index());
      } else {
        return std::get<absl::Span<ExecutionInput>>(arguments)[param_no]
            .Buffer(allocation.param_shape_index())
            .AsDeviceAddress();
      }
    }();
    if (registered_buffer.is_null() && registered_buffer.size() > 0) {
      return FailedPrecondition(
          "Cannot run XLA computation because pointer to (sub-)buffer at "
          "index %s of parameter %d was null.  All pointers to "
          "(sub-)buffers must not be null, unless the (sub-)buffer has "
          "zero elements.",
          allocation.param_shape_index().ToString(), param_no);
    }
    return registered_buffer;
  } else if (allocation.is_constant()) {
    auto it = globals->find(arg_idx);
    if (it == globals->end()) {
      return se::DeviceAddressBase();
    }
    return it->second;
  } else {
    // Allocate each allocation that might escape, or is the temp buffer.
    CHECK(allocation.maybe_live_out() || allocation.IsPreallocatedTempBuffer());
    int64_t buffer_size = allocation.size();
    se::DeviceAddressBase buffer_address;
    if (buffer_size > 0) {
      // Maybe round up buffer allocation size to the requested granulariy.
      if (auto it = allocate_granularity.find(allocation.color());
          it != allocate_granularity.end()) {
        buffer_size = RoundUpTo(buffer_size, it->second);
      }
      ASSIGN_OR_RETURN(
          se::ScopedDeviceAddress<uint8_t> buffer,
          memory_allocator->Allocate(device_ordinal, buffer_size,
                                     /*retry_on_failure=*/true,
                                     /*memory_space=*/allocation.color()));
      buffer_address = buffer.Release();
    }
    return buffer_address;
  }
}

static absl::Status CheckAlignment(const BufferAllocation& allocation,
                                   se::DeviceAddressBase buffer, int arg_idx) {
  const int64_t expected_alignment = [&] {
    if (allocation.is_entry_computation_parameter()) {
      return kEntryParameterAlignBytes;
    } else if (allocation.is_constant()) {
      return kConstantBufferAlignBytes;
    } else {
      return kXlaAllocatedBufferAlignBytes;
    }
  }();
  if (!buffer.is_null() &&
      reinterpret_cast<uintptr_t>(buffer.opaque()) % expected_alignment != 0) {
    return Internal(
        "Address of buffer %d must be a multiple of %x, but "
        "was %p",
        arg_idx, expected_alignment, buffer.opaque());
  }
  return absl::OkStatus();
}

// Resolve GpuCollectives instance that we should use for the run.
// TODO(ezhulenev): We have almost identical method in `collective_params.cc`,
// this one has to be removed.
static GpuCollectives* ResolveGpuCollectives(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options) {
  auto* gpu_options = run_options->run_options().gpu_executable_run_options();
  if (gpu_options && gpu_options->collectives()) {
    return gpu_options->collectives();
  }

  absl::string_view platform_name =
      run_options->run_options().stream()->parent()->GetPlatform()->Name();

  // If debug options specify a collectives implementation by name, look it up
  // in the registry. Otherwise, use the default (highest-priority) one.
  if (debug_options &&
      !debug_options->xla_gpu_collectives_implementation().empty()) {
    absl::StatusOr<Collectives*> collectives = CollectivesRegistry::Get(
        platform_name, debug_options->xla_gpu_collectives_implementation());
    CHECK_OK(collectives)  // Crash OK
        << "Failed to get GPU collectives implementation: "
        << debug_options->xla_gpu_collectives_implementation();
    return tsl::down_cast<GpuCollectives*>(*collectives);
  }

  return GpuCollectives::Default(platform_name);
}

absl::StatusOr<BufferAllocations> GpuExecutable::GenerateBufferAllocations(
    const ServiceExecutableRunOptions* run_options, VariantArguments arguments,
    const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
    se::DeviceAddressAllocator* const memory_allocator, int device_ordinal) {
  tsl::profiler::TraceMe hlo_module_activity(
      [&] { return std::string("Build buffer allocations"); },
      tsl::profiler::TraceMeLevel::kInfo);

  const DebugOptions* debug_options =
      has_module() ? &module_config().debug_options() : nullptr;

  absl::flat_hash_map<LogicalBuffer::Color, int64_t> allocate_granularity;
  if (auto* collectives = ResolveGpuCollectives(run_options, debug_options)) {
    // BFC allocator ignores memory alignment and always allocates 256 byte
    // aligned buffers, however for collective memory underlying libraries
    // require larger alignment. We conservatively round up all allocation
    // sizes to the alignment requirement. Proper fix must be done in BFC
    // allocator and all the other allocator adaptors that we have in XLA, but
    // this is left as an exercise for curious reader. The raw memory allocator
    // that backs the BFC allocator uses correct granularity and alignment.
    static constexpr int64_t kCollectiveMemoryColor = 1;
    allocate_granularity[kCollectiveMemoryColor] =
        collectives->SymmetricMemoryAlignment();
  }

  absl::Span<const BufferAllocation* const> allocations = GetAllocations();
  const int64_t num_buffers = allocations.size();
  std::vector<se::DeviceAddressBase> buffers;
  buffers.reserve(num_buffers);
  for (int64_t i = 0; i < num_buffers; ++i) {
    const BufferAllocation& allocation = *allocations[i];
    ASSIGN_OR_RETURN(
        buffers.emplace_back(),
        BufferForAllocation(arguments, globals, allocation, memory_allocator,
                            device_ordinal, i, allocate_granularity));
    RETURN_IF_ERROR(CheckAlignment(allocation, buffers.back(), i));
  }
  return {{buffers, device_ordinal, memory_allocator}};
}

absl::StatusOr<ExecutionOutput> GpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments) {
  return ExecuteAsyncOnStreamImpl(run_options, absl::MakeSpan(arguments));
}

absl::StatusOr<ScopedShapedBuffer> GpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments) {
  ASSIGN_OR_RETURN(ExecutionOutput out,
                   ExecuteAsyncOnStreamImpl(run_options, arguments));
  return out.ConsumeResult();
}

absl::StatusOr<ExecutionOutput> GpuExecutable::ExecuteAsyncOnStreamImpl(
    const ServiceExecutableRunOptions* run_options,
    VariantArguments arguments) {
  XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
      "GpuExecutable::ExecuteAsyncOnStreamImpl(", module_name_, ")"));
  se::DeviceAddressAllocator* const memory_allocator = run_options->allocator();
  se::StreamExecutor* executor = run_options->stream()->parent();

  // GpuExecutable always bound to a single GpuContext during its execution, so
  // we activate it once to skip expensive context activations later.
  auto activation = executor->Activate();

  // Lock the GPU with a shared lock so that we don't interfere with autotuning
  // that may be running during JIT compilation while allowing multiple XLA
  // computations to use the same GPU simultaneously. We do not add locking for
  // "recursive" invocations, which are done when holding a lock already.
  std::variant<absl::ReaderMutexLock, absl::WriterMutexLock> gpu_lock(
      std::in_place_index_t<0>{}, &GetGpuMutex(executor));

  // Maybe update to a writer lock to get exclusive access to underlying GPU.
  if (auto* gpu_opts = run_options->run_options().gpu_executable_run_options();
      gpu_opts && gpu_opts->requires_exclusive_lock_on_gpu()) {
    gpu_lock.emplace<1>(&GetGpuMutex(executor));
  }

  const GpuExecutable::BufferAllocToDeviceMemoryMap* globals;
  {
    tsl::profiler::TraceMe hlo_module_activity(
        [&] { return std::string("Resolve constant globals"); },
        tsl::profiler::TraceMeLevel::kInfo);

    ASSIGN_OR_RETURN(globals, ResolveConstantGlobals(run_options->stream()));
  }

  // Use the `device_ordinal` from the `run_options` if it is provided. This is
  // the ordinal of the logical devices (e.g., virtual GPUs). If it is not
  // provided, the ordinals of the logical and physical devices are the same.
  const int device_ordinal = run_options->device_ordinal() != -1
                                 ? run_options->device_ordinal()
                                 : executor->device_ordinal();
  ExecutionOutput result(/*on_device_shape=*/program_shape_.result(),
                         memory_allocator, device_ordinal,
                         executor->device_ordinal());

  ASSIGN_OR_RETURN(BufferAllocations buffer_allocations,
                   GenerateBufferAllocations(run_options, arguments, globals,
                                             memory_allocator, device_ordinal));
  XLA_VLOG_DEVICE(3, device_ordinal) << buffer_allocations.ToString();
  absl::Span<const BufferAllocation* const> allocations = GetAllocations();

  std::set<se::DeviceAddressBase> buffers_in_result;

  const bool is_entire_tuple_contents_aliased = [&] {
    for (auto& p : result.MutableResult()->buffers().leaves()) {
      if (!output_info_.contains(p.first)) {
        continue;
      }
      const OutputInfo& output_info = output_info_.at(p.first);
      if (!output_info.alias_config.has_value()) {
        return false;
      }
    }
    return true;
  }();

  for (auto& p : result.MutableResult()->buffers()) {
    const ShapeIndex& index = p.first;
    if (!output_info_.contains(index)) {
      continue;
    }
    const OutputInfo& output_info = output_info_.at(index);
    const BufferAllocation* allocation =
        allocations[output_info.allocation_index];
    se::DeviceAddressBase& result_buffer = p.second;

    XLA_VLOG_DEVICE(4, device_ordinal)
        << "Looking at: allocation " << output_info.allocation_index
        << " @ index: " << index.ToString();

    if (output_info.alias_config) {
      MaybeOwningDeviceAddress* maybe_owning_memory =
          [&]() -> xla::MaybeOwningDeviceAddress* {
        // ShapedBuffer is never an owned buffer.
        if (std::holds_alternative<absl::Span<const ShapedBuffer* const>>(
                arguments)) {
          return nullptr;
        } else {
          auto unowned_execution_input =
              std::get<absl::Span<ExecutionInput>>(arguments);
          ExecutionInput& input =
              unowned_execution_input[allocation->parameter_number()];
          return input.MutableBuffer(allocation->param_shape_index());
        }
      }();
      if (output_info.alias_config->must_alias() && maybe_owning_memory &&
          !maybe_owning_memory->HasOwnership()) {
        return InvalidArgument(
            "An input was configured to be must-alias at "
            "compile time but not donated at runtime: allocation %d",
            output_info.allocation_index);
      }
      if (maybe_owning_memory && maybe_owning_memory->HasOwnership()) {
        std::optional<tensorflow::se::ScopedDeviceAddress<uint8_t>> owning =
            maybe_owning_memory->Release();
        // If the caller passes the ownership of the device memory, reuse it
        // as the output buffer. It is up to the caller whether or not to
        // donate a buffer; the aliasing information describes which buffers
        // may alias, not buffers that must alias.
        se::DeviceAddressBase argument_buffer = owning->Release();
        *maybe_owning_memory = argument_buffer;
        result_buffer = argument_buffer;
        // The caller is giving us the
        // input buffer, but in case of error from the execute call, we should
        // not be releasing it as it contains valid data (for example, it is a
        // parameter which the user wants us to alias, in a gradient update
        // computation). So we store the index into the result in the aliased
        // vector, which will be fed to the ExecutionOutput, which will use
        // the indices to drop the addresses from its own ScopedShapedBuffer
        // result, if the ExecutionOutput is not committed.
        result.AddAliasedIndex(index);
      } else if (!output_info.passthrough &&
                 !ShapeUtil::GetSubshape(program_shape_.result(), index)
                      .IsTuple()) {
        // The guard is above is not to insert copy-protection when aliasing
        // pass-through params, as we do not need to write into the output
        // buffer.
        XLA_VLOG_DEVICE(3, device_ordinal)
            << "Using copy-protection: aliasing is specified, but the "
               "buffer is not donated; allocating a fresh buffer";
        int64_t allocation_size = ShapeUtil::ByteSizeOf(
            ShapeUtil::GetSubshape(program_shape_.result(), index));
        absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> allocated_buffer =
            memory_allocator->Allocate(device_ordinal, allocation_size,
                                       /*retry_on_failure=*/true,
                                       /*memory_space=*/allocation->color());
        if (!allocated_buffer.ok()) {
          return VerboseAllocationError(allocated_buffer.status());
        }
        result_buffer = allocated_buffer->Release();
        se::DeviceAddressBase& aliased_buffer =
            buffer_allocations.GetMutableDeviceAddress(
                output_info.allocation_index);
        CHECK_EQ(aliased_buffer.size(), result_buffer.size());
        RETURN_IF_ERROR(run_options->stream()->MemcpyD2D(
            &result_buffer, aliased_buffer, aliased_buffer.size()));
        aliased_buffer = result_buffer;
      }
    }

    if (result_buffer.is_null()) {
      // The source instruction should have a non-parameter buffer
      // assigned.
      result_buffer =
          buffer_allocations.GetDeviceAddress(output_info.allocation_index);

      // If the entire tuple contents is aliased, the copy insertion will *not*
      // materialize a new tuple, so we mark it as aliased as well.
      if (is_entire_tuple_contents_aliased) {
        result.AddAliasedIndex(index);
      }
    }
    buffers_in_result.insert(result_buffer);
  }

  absl::Status execute_status = ExecuteThunks(buffer_allocations, run_options);

  absl::Status teardown_status =
      buffer_allocations.TearDown(buffers_in_result, GetAllocations());

  RETURN_IF_ERROR(execute_status);
  RETURN_IF_ERROR(teardown_status);

  // Free allocations for arguments.
  if (auto args = std::get_if<absl::Span<ExecutionInput>>(&arguments)) {
    MarkToBeReleasedArguments(*args, result);
  }
  return std::move(result);
}

absl::Status GpuExecutable::VerboseAllocationError(absl::Status s) {
  return ResourceExhausted(
      "%s\n%s\n", s.message(),
      buffer_assignment_->ToVerboseString(alias_info_.get(),
                                          debug_buffer_assignment_show_max_));
}

// VA remapping execution flow for 2 consecutive calls on the same executor:
//
// clang-format off
// NOLINTBEGIN
//                   +---------------------+---------------------++---------------------+---------------------+
// GPU               |  VA1 Execute        |  VA2 Execute        ||  VA1 Execute        |  VA2 Execute        |
//                   +---------------------+---------------------++---------------------+---------------------+
//         +---------++---------+           +---------++---------+ +---------++---------++---------+           +---------+
// CPU     | VA1 Map || VA2 Map |           |VA1 UnMap|| VA1 Map | |VA2 UnMap|| VA2 Map ||VA1 UnMap|           |VA2 UnMap|
//         +---------++---------+           +---------++---------+ +---------++---------++---------+           +---------+
// NOLINTEND
// clang-format on
absl::Status GpuExecutable::ExecuteThunksWithVaRemapping(
    const BufferAllocations& buffer_allocations,
    const ServiceExecutableRunOptions* run_options,
    se::StreamExecutor* executor, int64_t unique_id,
    Thunk::ExecutableSource executable_source, bool block_host_until_done,
    bool collective_use_minimal_resource) {
  // Get or create VaRanges for this executor and VA range index. We hold
  // va_ranges_mutex_ briefly just to access/create the VaRanges entry.
  // The VA range index allows multiplexing: with kNumVaReservationSets=2
  // reservations, the CPU can remap one range while the GPU executes the other.
  int command_buffer_va_range_idx =
      run_options->run_options().command_buffer_va_range_idx();
  VaRanges* va_ranges = nullptr;
  {
    absl::MutexLock lock(&va_ranges_mutex_);
    auto va_ranges_key = std::make_pair(executor, command_buffer_va_range_idx);
    va_ranges = &module_va_ranges_[va_ranges_key];
  }

  XLA_VLOG_DEVICE(3, executor->device_ordinal())
      << "VA remapping: module " << module_name_
      << " va_range_idx=" << command_buffer_va_range_idx
      << " num_allocations=" << command_buffer_allocation_indexes_.size();

  // Get the DeviceAddressVmmAllocator to look up physical allocations.
  // vmm_allocator is guaranteed non-null here because
  // use_command_buffer_va_remapping already checked for it.
  se::DeviceAddressVmmAllocator* vmm_allocator =
      dynamic_cast<se::DeviceAddressVmmAllocator*>(run_options->allocator());
  if (vmm_allocator == nullptr) {
    return Internal("DeviceAddressVmmAllocator cast failed unexpectedly");
  }

  uint64_t granularity = vmm_allocator->GetAllocationGranularity(executor);
  auto round_up_to_granularity = [granularity](uint64_t size) -> uint64_t {
    if (granularity == 0) {
      return size;
    }
    return ((size + granularity - 1) / granularity) * granularity;
  };

  // Acquire per-executor mutex to protect VA range operations.
  // This ensures only one thread uses the VA ranges at a time for this
  // executor.
  absl::MutexLock va_lock(&va_ranges->mutex);

  // Initialize VA ranges if this is first use (va_reservation is null).
  if (va_ranges->va_reservation == nullptr) {
    ScopedAnnotation annotation_va_reserve([&] {
      return absl::StrFormat("command_buffer_va_range_reserve:#module=%s#",
                             module_name_);
    });

    // Calculate total size for all command buffer allocations, rounding each
    // allocation up to the allocation granularity.
    uint64_t total_va_size = 0;
    for (BufferAllocation::Index i : command_buffer_allocation_indexes_) {
      const uint64_t size = buffer_allocations.GetDeviceAddress(i).size();
      total_va_size += round_up_to_granularity(size);
    }

    // Reserve a single large VA range for all command buffer allocations.
    TF_ASSIGN_OR_RETURN(
        va_ranges->va_reservation,
        vmm_allocator->CreateReservation(executor, total_va_size));
    TF_ASSIGN_OR_RETURN(va_ranges->unmap_event, executor->CreateEvent());

    XLA_VLOG_DEVICE(3, executor->device_ordinal()) << absl::StreamFormat(
        "VA remapping: Reserved single VA range for module %s "
        "VA: %p total_size: %d granularity: %d",
        module_name_, va_ranges->va_reservation->address().opaque(),
        total_va_size, granularity);
  } else {
    ScopedAnnotation annotation_va_unmap([&] {
      return absl::StrFormat("command_buffer_va_range_unmap:#module=%s#",
                             module_name_);
    });

    // VA range is already initialized; wait for the unmap event to be marked
    // and then do the VA unmapping.
    TF_RETURN_IF_ERROR(va_ranges->unmap_event->Synchronize());

    // Unmap physical addresses from the single reserved VA range.
    // Clearing ScopedMappings calls UnMap via their destructors.
    va_ranges->scoped_mapping.reset();
  }

  // Build a map from allocation index to its offset within va_reservation.
  // Iterate through command_buffer_allocation_indexes_ in order (btree_set
  // provides deterministic iteration order) and accumulate offsets.
  absl::flat_hash_map<BufferAllocation::Index, uint64_t> allocation_va_offsets;
  uint64_t current_offset = 0;
  for (BufferAllocation::Index idx : command_buffer_allocation_indexes_) {
    const uint64_t size = buffer_allocations.GetDeviceAddress(idx).size();
    allocation_va_offsets[idx] = current_offset;
    current_offset += round_up_to_granularity(size);
  }

  if (!allocation_va_offsets.empty() && va_ranges->va_reservation == nullptr) {
    return Internal("Reserved VA address range is null");
  }

  // Map physical memory to reserved VA addresses.
  std::vector<se::DeviceAddressBase> mapped_buffers;
  mapped_buffers.reserve(buffer_allocations.size());

  {
    ScopedAnnotation annotation_va_remap([&] {
      return absl::StrFormat("command_buffer_va_range_remap:#module=%s#",
                             module_name_);
    });

    // Collect mapping descriptors for the batch MapTo call. Descriptors are
    // accumulated in reservation_offset order (guaranteed because
    // allocation_va_offsets was built from a sorted btree_set and the loop
    // below iterates allocation indices in ascending order).
    std::vector<se::MemoryReservation::MappingDescriptor> mapping_descriptors;

    const BufferAllocation::Index num_allocations =
        static_cast<BufferAllocation::Index>(buffer_allocations.size());
    for (BufferAllocation::Index i = 0; i < num_allocations; ++i) {
      se::DeviceAddressBase original_buffer =
          buffer_allocations.GetDeviceAddress(i);

      // Only do VA mapping for allocations accessed by CommandBufferThunk.
      auto offset_it = allocation_va_offsets.find(i);
      if (offset_it == allocation_va_offsets.end()) {
        // Not a command buffer allocation (or zero-size), use the original
        // buffer.
        mapped_buffers.push_back(original_buffer);
        continue;
      }

      // This allocation is used by command buffer - validate it's not null.
      if (original_buffer.is_null()) {
        return Internal("Command buffer allocation %d has null address", i);
      }

      // Get the physical memory allocation from the VMM allocator.
      se::MemoryAllocation* raw_alloc = vmm_allocator->GetRawAllocation(
          executor->device_ordinal(), original_buffer);
      if (raw_alloc == nullptr) {
        return Internal(
            "No raw allocation found for command buffer allocation %d", i);
      }
      const uint64_t mapping_size = raw_alloc->address().size();

      // Calculate the sub-range VA address for this allocation.
      uint64_t va_offset = offset_it->second;
      void* sub_range_ptr = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(
              va_ranges->va_reservation->address().opaque()) +
          va_offset);
      se::DeviceAddressBase sub_range_va(sub_range_ptr, original_buffer.size());

      XLA_VLOG_DEVICE(3, executor->device_ordinal()) << absl::StreamFormat(
          "Mapping allocation %d physical: %p -> VA: %p "
          "(offset: %d) size: %d",
          i, original_buffer.opaque(), sub_range_va.opaque(), va_offset,
          original_buffer.size());

      mapping_descriptors.push_back(
          {va_offset, /*allocation_offset=*/0, mapping_size, raw_alloc});

      // Use VA address for execution.
      mapped_buffers.push_back(
          se::DeviceAddressBase(sub_range_va.opaque(), original_buffer.size()));
    }

    // Batch-map all command buffer allocations into the reserved VA range in
    // a single call. This maps the contiguous range formed by the descriptors
    // and enables device access before returning.
    if (!mapping_descriptors.empty()) {
      TF_ASSIGN_OR_RETURN(se::MemoryReservation::ScopedMapping scoped_mapping,
                          va_ranges->va_reservation->MapTo(
                              absl::MakeSpan(mapping_descriptors)));
      va_ranges->scoped_mapping = std::move(scoped_mapping);
    }
  }

  if (VLOG_IS_ON(3)) {
    void* va_base = (va_ranges->va_reservation != nullptr)
                        ? va_ranges->va_reservation->address().opaque()
                        : nullptr;
    XLA_VLOG_DEVICE(3, executor->device_ordinal()) << absl::StreamFormat(
        "VA remapping: Mapped %d allocations to single VA range at %p",
        allocation_va_offsets.size(), va_base);
    for (const auto& [alloc_idx, va_offset] : allocation_va_offsets) {
      se::DeviceAddressBase physical_addr =
          buffer_allocations.GetDeviceAddress(alloc_idx);
      void* va_ptr = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(va_base) + va_offset);
      XLA_VLOG_DEVICE(3, executor->device_ordinal()) << absl::StreamFormat(
          "  allocation[%d] physical: %p -> VA: %p (offset: %d) size: %d",
          alloc_idx, physical_addr.opaque(), va_ptr, va_offset,
          physical_addr.size());
    }
  }

  BufferAllocations remapped_buffer_allocations(
      mapped_buffers, buffer_allocations.device_ordinal(),
      buffer_allocations.memory_allocator());

  // Execute thunks with remapped addresses.
  TF_RETURN_IF_ERROR(ExecuteThunksImpl(
      has_module() ? &module_config().debug_options() : nullptr, module_name_,
      unique_id, *thunk_executor_, executable_source, run_options,
      remapped_buffer_allocations, block_host_until_done,
      num_additional_compute_streams_, collective_memory_cache_,
      collective_use_minimal_resource));

  // Record event so VA range can be reclaimed after GPU finishes.
  TF_RETURN_IF_ERROR(
      run_options->stream()->RecordEvent(va_ranges->unmap_event.get()));

  return absl::OkStatus();
}

absl::Status GpuExecutable::ExecuteThunks(
    const BufferAllocations& buffer_allocations,
    const ServiceExecutableRunOptions* run_options) {
  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode(
        absl::StrFormat("[%d] GpuExecutable::ExecuteThunks",
                        run_options->device_ordinal()),
        {{"module_name", module_name_}});
  });

  if (VLOG_IS_ON(5)) {
    // Debug code to compare current allocation's address with previous run's
    // address, and report the allocation info if memory addressed changed.
    // Useful for identify in user's model if it is command buffer perf friendly
    // (no command buffer update cost).
    se::StreamExecutor* executor = run_options->stream()->parent();

    // Collect the set of allocations that changed between executions.
    std::vector<std::pair<int32_t, std::string>> changed_allocations;

    absl::MutexLock lock(module_handle_mutex_);
    if (module_allocations_.find(executor) == module_allocations_.end()) {
      std::vector<se::DeviceAddressBase> allocs_addr;
      allocs_addr.reserve(buffer_allocations.size());
      for (int i = 0; i < buffer_allocations.size(); i++) {
        allocs_addr.push_back(buffer_allocations.GetDeviceAddress(i));
      }
      module_allocations_[executor] = std::move(allocs_addr);
    } else {
      for (int i = 0; i < buffer_allocations.size(); i++) {
        if (module_allocations_[executor][i].IsSameAs(
                buffer_allocations.GetDeviceAddress(i))) {
          continue;
        }
        module_allocations_[executor][i] =
            buffer_allocations.GetDeviceAddress(i);
        const BufferAllocation& allocation =
            buffer_assignment_->GetAllocation(i);
        const char* allocation_type =
            allocation.is_entry_computation_parameter() ? "parameter"
            : allocation.maybe_live_out()               ? "live-out"
                                                        : "temp";
        changed_allocations.emplace_back(i, allocation_type);
      }
    }

    if (!changed_allocations.empty()) {
      XLA_VLOG_DEVICE(5, executor->device_ordinal()) << absl::StreamFormat(
          "Buffer allocations changed address between module %s executions: "
          "[%s]",
          module_name_,
          absl::StrJoin(changed_allocations, ", ", absl::PairFormatter(":")));
    }
  }

  se::DeviceAddressAllocator* const memory_allocator = run_options->allocator();
  // Force synchronous execution if the allocator requires it.
  const bool block_host_until_done =
      !memory_allocator->AllowsAsynchronousDeallocation();

  RETURN_IF_ERROR(
      CheckCompatibilityWithServiceExecutableRunOptions(run_options));

  ScopedAnnotation annotation([&] { return module_annotations_.top_level; });
  ScopedModuleAnnotations module_annotations(&module_annotations_);

  ModuleIdentifier unique_id = has_module() ? module().unique_id() : -1;
  Thunk::ExecutableSource executable_source = {text_, binary_,
                                               dnn_compiled_graphs_};

  se::StreamExecutor* executor = run_options->stream()->parent();

  // Check if command buffer VA remapping is active.
  bool use_command_buffer_va_remapping =
      (command_buffer_allocation_indexes_.size() > 0) && has_module() &&
      module_config().debug_options().xla_gpu_command_buffer_update_mode() !=
          DebugOptions::ALWAYS_UPDATE &&
      dynamic_cast<se::DeviceAddressVmmAllocator*>(memory_allocator) != nullptr;

  XLA_VLOG_DEVICE(3, executor->device_ordinal()) << absl::StreamFormat(
      "ExecuteThunks: command_buffer_allocation_indexes_.size()=%d "
      "use_command_buffer_va_remapping=%d",
      command_buffer_allocation_indexes_.size(),
      use_command_buffer_va_remapping);

  bool collective_use_minimal_resource = false;
  if (has_module()) {
    ASSIGN_OR_RETURN(collective_use_minimal_resource,
                     ShouldCollectiveUseMinimalResource(module()));
  }
  if (use_command_buffer_va_remapping) {
    TF_RETURN_IF_ERROR(ExecuteThunksWithVaRemapping(
        buffer_allocations, run_options, executor, unique_id, executable_source,
        block_host_until_done, collective_use_minimal_resource));
  } else {
    TF_RETURN_IF_ERROR(ExecuteThunksImpl(
        has_module() ? &module_config().debug_options() : nullptr, module_name_,
        unique_id, *thunk_executor_, executable_source, run_options,
        buffer_allocations, block_host_until_done,
        num_additional_compute_streams_, collective_memory_cache_,
        collective_use_minimal_resource));
  }
  return absl::OkStatus();
}

int64_t GpuExecutable::SizeOfGeneratedCodeInBytes() const {
  // Non-empty PTX but empty cubin: compilation must have failed, return
  // "unknown".
  if (binary().empty() && !text_.empty()) {
    return -1;
  }
  int64_t size = binary().size();
  for (const BufferAllocation* allocation : GetAllocations()) {
    if (allocation->is_constant()) {
      size += allocation->size();
    }
  }
  return size;
}

absl::StatusOr<absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>>
GetOutputInfo(const HloModule& hlo_module, const BufferAssignment& assignment) {
  const HloInstruction* root =
      hlo_module.entry_computation()->root_instruction();

  InstructionValueSet root_value_set =
      assignment.dataflow_analysis().GetInstructionValueSet(root);

  if (root_value_set.IsAmbiguous()) {
    return Unimplemented("Points-to set of root instruction is ambiguous");
  }

  using OutputInfoMap =
      absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>;
  OutputInfoMap output;
  RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      root->shape(),
      [&](const Shape& /*sub_shape*/, const ShapeIndex& index) -> absl::Status {
        const auto& sources = root_value_set.element(index);
        // The points-to set is unambiguous so the set should be a
        // singleton. That is, we know exactly which instruction
        // produced the array at this element.
        CHECK_EQ(1, sources.values().size());
        HloInstruction* src_hlo = sources.values()[0]->instruction();

        GpuExecutable::OutputInfo& info = output[index];
        info.passthrough = src_hlo->opcode() == HloOpcode::kParameter;
        ASSIGN_OR_RETURN(
            const BufferAllocation::Slice slice,
            assignment.GetUniqueSlice(src_hlo, sources.values()[0]->index()));
        CHECK_EQ(slice.offset(), 0) << "Parameter should get its own slice";
        info.allocation_index = slice.index();

        output[index].alias_config =
            hlo_module.input_output_alias_config().GetAliasedParameter(index);

        return absl::OkStatus();
      }));
  return output;
}

GpuExecutableProto::OutputInfoProto GpuExecutable::OutputInfo::ToProto() const {
  GpuExecutableProto::OutputInfoProto proto;
  proto.set_allocation_index(allocation_index);
  proto.set_passthrough(passthrough);

  if (alias_config.has_value()) {
    proto.mutable_alias_config()->set_parameter_number(
        alias_config->parameter_number);
    proto.mutable_alias_config()->mutable_parameter_shape_index()->Assign(
        alias_config->parameter_index.begin(),
        alias_config->parameter_index.end());

    switch (alias_config->kind) {
      case xla::HloInputOutputAliasConfig::AliasKind::kMayAlias:
        proto.mutable_alias_config()->set_kind(Kind::MAY_ALIAS);
        break;
      case xla::HloInputOutputAliasConfig::AliasKind::kMustAlias:
        proto.mutable_alias_config()->set_kind(Kind::MUST_ALIAS);
        break;
    }
  }

  return proto;
}

absl::StatusOr<GpuExecutable::OutputInfo> GpuExecutable::OutputInfo::FromProto(
    const GpuExecutableProto::OutputInfoProto& proto) {
  OutputInfo output_info;
  output_info.allocation_index = proto.allocation_index();
  output_info.passthrough = proto.passthrough();
  if (proto.has_alias_config()) {
    xla::HloInputOutputAliasConfig::AliasKind alias_kind;
    switch (proto.alias_config().kind()) {
      case Kind::MAY_ALIAS:
        alias_kind = xla::HloInputOutputAliasConfig::AliasKind::kMayAlias;
        break;
      case Kind::MUST_ALIAS:
        alias_kind = xla::HloInputOutputAliasConfig::AliasKind::kMustAlias;
        break;
      default:
        return absl::InvalidArgumentError("Given alias kind is invalid");
    }
    const auto& parameter_shape_index =
        proto.alias_config().parameter_shape_index();
    output_info.alias_config.emplace(
        proto.alias_config().parameter_number(),
        ShapeIndex{parameter_shape_index.begin(), parameter_shape_index.end()},
        alias_kind);
  }
  return output_info;
}

GpuExecutableProto::ConstantInfoProto GpuExecutable::ConstantInfo::ToProto()
    const {
  GpuExecutableProto::ConstantInfoProto proto;
  proto.set_symbol_name(symbol_name);
  *proto.mutable_content() = content.ToProto();
  proto.set_allocation_index(allocation_index);
  return proto;
}

GpuExecutable::ConstantInfo GpuExecutable::ConstantInfo::FromProto(
    const GpuExecutableProto::ConstantInfoProto& proto) {
  return ConstantInfo{
      /*symbol_name=*/proto.symbol_name(),
      /*content=*/DenseDataIntermediate::FromProto(proto.content()),
      /*allocation_index=*/static_cast<int>(proto.allocation_index())};
}

absl::StatusOr<GpuExecutableProto> GpuExecutable::ToProto() const {
  GpuExecutableProto proto;
  proto.set_binary(binary_.data(), binary_.size());
  proto.set_asm_text(text_);
  proto.mutable_dnn_compiled_graphs()->insert(dnn_compiled_graphs_.cbegin(),
                                              dnn_compiled_graphs_.cend());

  *proto.mutable_gpu_compute_capability() = gpu_version_.ToProto();

  // TODO(b/461380690): Generate the proto on-the-fly once we have a better way
  // to distinguish between compiler-generated and runtime-loaded GPU
  // executables.
  ASSIGN_OR_RETURN(const auto& thunk_sequence_proto, thunk_sequence_proto_);
  proto.mutable_thunks()->Reserve(thunk_sequence_proto.size());
  for (const auto& thunk_proto : thunk_sequence_proto) {
    *proto.add_thunks() = thunk_proto;
  }

  proto.set_module_name(module_name_);
  *proto.mutable_program_shape() = program_shape_.ToProto();

  absl::Span<const BufferAllocation* const> allocations = GetAllocations();
  proto.mutable_buffer_allocations()->mutable_values()->Reserve(
      allocations.size());
  for (const auto& allocation : allocations) {
    proto.mutable_buffer_allocations()->mutable_values()->Add(
        allocation->ToProto());
  }

  if (has_module()) {
    *proto.mutable_hlo_module_with_config() = module().ToProtoWithConfig();
  }

  proto.mutable_output_info_map()->Reserve(output_info_.size());
  for (const auto& [shape_index, output_info] :
       tsl::KeySortedRange(output_info_)) {
    auto map_entry = proto.add_output_info_map();
    *map_entry->mutable_shape_index() = shape_index.ToProto();
    *map_entry->mutable_output_info() = output_info.ToProto();
  }

  proto.mutable_constants()->Reserve(constants_.size());
  for (const auto& constant : constants_) {
    *proto.add_constants() = constant.ToProto();
  }

  *proto.mutable_executable_abi_version() = executable_abi_version_.proto();

  if (cpu_target_machine_options_.has_value()) {
    *proto.mutable_cpu_target_machine_options() =
        cpu_target_machine_options_->ToProto();
  }

  return proto;
}

absl::StatusOr<std::unique_ptr<GpuExecutable>> GpuExecutable::FromProto(
    const GpuExecutableProto& proto,
    const se::DeviceDescription& device_description,
    absl::string_view platform_name, DebugOptions debug_options,
    const std::optional<stream_executor::KernelLoaderSpec::SymbolResolver>&
        symbol_resolver) {
  Params params;
  params.debug_options = std::move(debug_options);
  params.enable_debug_info_manager =
      params.debug_options.xla_gpu_executable_embed_debug_info();
  params.asm_text = proto.asm_text();
  const std::string& binary = proto.binary();
  params.binary.assign(binary.begin(), binary.end());
  params.buffer_assignment = nullptr;
  if (proto.has_hlo_module_with_config()) {
    ASSIGN_OR_RETURN(params.debug_module, HloModule::CreateFromProtoWithConfig(
                                              proto.hlo_module_with_config()));
  }

  params.mlir_allocations.emplace();
  params.mlir_allocations->reserve(proto.buffer_allocations().values_size());
  for (const BufferAllocationProto& allocation_proto :
       proto.buffer_allocations().values()) {
    params.mlir_allocations->push_back(
        BufferAllocation::FromProto(allocation_proto));
  }

  for (const auto& [key, value] : proto.dnn_compiled_graphs()) {
    params.dnn_compiled_graphs.emplace(key, value);
  }

  ASSIGN_OR_RETURN(stream_executor::GpuComputeCapability gpu_compute_capability,
                   stream_executor::GpuComputeCapability::FromProto(
                       proto.gpu_compute_capability()));

  if (gpu_compute_capability != device_description.gpu_compute_capability()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "GPU compute capability of serialized executable doesn't match target "
        "device capability. (serialized: %s, target: %s)",
        gpu_compute_capability.ToString(),
        device_description.gpu_compute_capability().ToString()));
  }

  params.device_description = device_description;

  if (proto.has_cpu_target_machine_options()) {
    ASSIGN_OR_RETURN(params.cpu_target_machine_options,
                     xla::cpu::TargetMachineOptions::FromProto(
                         proto.cpu_target_machine_options()));
  }

  ThunkSequenceProto thunk_sequence_proto;
  *thunk_sequence_proto.mutable_thunks() = proto.thunks();
  ASSIGN_OR_RETURN(
      ThunkSequence thunk_sequence,
      DeserializeThunkSequenceProto(
          thunk_sequence_proto, params.mlir_allocations.value(),
          params.debug_module.get(), platform_name, gpu_compute_capability,
          symbol_resolver, params.cpu_target_machine_options));

  params.executable =
      std::make_unique<ThunkExecutor>(std::move(thunk_sequence));

  params.constants.reserve(proto.constants().size());
  for (const auto& constant_proto : proto.constants()) {
    params.constants.push_back(ConstantInfo::FromProto(constant_proto));
  }

  params.output_info.reserve(proto.output_info_map().size());
  for (const auto& output_info_proto : proto.output_info_map()) {
    ShapeIndex shape_index =
        ShapeIndex::FromProto(output_info_proto.shape_index());
    ASSIGN_OR_RETURN(OutputInfo output_info,
                     OutputInfo::FromProto(output_info_proto.output_info()));
    params.output_info.emplace(std::move(shape_index), std::move(output_info));
  }

  params.module_name = proto.module_name();
  ASSIGN_OR_RETURN(params.program_shape,
                   ProgramShape::FromProto(proto.program_shape()));

  ASSIGN_OR_RETURN(params.executable_abi_version,
                   stream_executor::ExecutableAbiVersion::FromProto(
                       proto.executable_abi_version()));

  return Create(std::move(params));
}

static absl::StatusOr<ExecutableBuildOptionsProto>
CreateSerializableBuildOptionsProto(const ExecutableBuildOptions& options) {
  ExecutableBuildOptions serializable_opts = options;
  // These fields are not serializable, and the toProto will fail if they are
  // set, but we also don't need them for the dump so just clear them.
  serializable_opts.set_layout_canonicalization_callback(nullptr);
  serializable_opts.set_compile_thread_pool(nullptr);

  return serializable_opts.ToProto();
}

absl::Status GpuExecutable::DumpExecutableIfEnabled(
    const ExecutableBuildOptions& options,
    const DebugOptions& debug_options) const {
  if (!debug_options.has_xla_dump_to() ||
      !debug_options.xla_gpu_experimental_dump_gpu_executable()) {
    return absl::OkStatus();
  }

  ASSIGN_OR_RETURN(GpuExecutableProto gpu_executable_proto, ToProto());
  ExecutableAndOptionsProto dump_proto;
  RETURN_IF_ERROR(
      WriteSplitGpuExecutable(std::move(gpu_executable_proto),
                              std::make_unique<riegeli::StringWriter<>>(
                                  dump_proto.mutable_serialized_executable())));
  ASSIGN_OR_RETURN(
      *dump_proto.mutable_compile_options()->mutable_executable_build_options(),
      CreateSerializableBuildOptionsProto(options));

  constexpr absl::string_view kDumpFilename = "gpu_executable.riegeli";
  ASSIGN_OR_RETURN(std::unique_ptr<riegeli::Writer> writer,
                   CreateRiegeliDumpWriter(debug_options, kDumpFilename,
                                           has_module() ? &module() : nullptr));
  RETURN_IF_ERROR(
      WriteSplitExecutableAndOptions(dump_proto, std::move(writer)));

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
