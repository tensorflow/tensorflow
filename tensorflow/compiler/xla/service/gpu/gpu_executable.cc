/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"

#include <set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/copy_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_constants.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_types.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/gpu/sequential_thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/llvm_ir/buffer_assignment_util.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/service/xla_debug_info_manager.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/stream_executor/platform.h"

namespace xla {
namespace gpu {
namespace {

using ::tensorflow::profiler::ScopedAnnotation;

// A helper function to decide whether to use GPU graph capture, which can
// reduce GPU launch latency overheads in some cases.
//
// To enable for all supported cluster set the environment variable
// TF_XLA_ENABLE_GPU_GRAPH_CAPTURE to 1.
//
// The environment variable TF_XLA_GPU_GRAPH_CLUSTER_NAME can be used
// to limit which cluster uses CudaGraph. Multiple names can be added
// by separating them by a comma. Example:
// TF_XLA_GPU_GRAPH_CLUSTER_NAME=cluster_1,cluster_20
bool GpuGraphCaptureEnabled(absl::string_view module_name) {
  static bool is_enabled = [] {
    bool is_enabled = false;
    TF_CHECK_OK(
        tensorflow::ReadBoolFromEnvVar("TF_XLA_ENABLE_GPU_GRAPH_CAPTURE",
                                       /*default_val=*/false, &is_enabled));
    return is_enabled;
  }();

  static string enabled_names = [] {
    string enabled_names;
    TF_CHECK_OK(
        tensorflow::ReadStringFromEnvVar("TF_XLA_GPU_GRAPH_CLUSTER_NAME",
                                         /*default_val=*/"", &enabled_names));
    return enabled_names;
  }();
  if (!enabled_names.empty() && is_enabled) {
    for (auto cluster_name : absl::StrSplit(enabled_names, ',')) {
      if (absl::StartsWith(module_name, cluster_name)) {
        VLOG(0) << "CUDA GRAPH ENABLED FOR CLUSTER " << module_name;
        return true;
      }
    }
    VLOG(0) << "CUDA GRAPH NOT ENABLED FOR CLUSTER " << module_name;
    return false;
  }
  return is_enabled;
}

bool IsThunkSafeForGpuGraphCapture(const std::unique_ptr<Thunk>& thunk) {
  // Thunks that synchronize with the host (i.e., call BlockHostUntilDone)
  // cannot be used with graph capture.
  static const absl::flat_hash_set<Thunk::Kind> thunk_kinds_safe_for_capture = {
      Thunk::kCollectivePermute,
      Thunk::kConvolution,
      Thunk::kCopy,
      Thunk::kCudnnBatchNormBackward,
      Thunk::kCudnnBatchNormForwardInference,
      Thunk::kCudnnBatchNormForwardTraining,
      Thunk::kGemm,
      Thunk::kKernel,
      Thunk::kMemset32BitValue,
      Thunk::kMemzero,
      Thunk::kNcclAllReduce,
      Thunk::kReplicaId,
      Thunk::kTuple,
  };
  if (thunk->kind() == Thunk::kSequential) {
    const auto* seq_thunk = static_cast<const SequentialThunk*>(thunk.get());
    return absl::c_all_of(seq_thunk->thunks(), IsThunkSafeForGpuGraphCapture);
  }
  if (dynamic_cast<const HostToDeviceCopyThunk*>(thunk.get())) {
    VLOG(1) << "HostToDeviceCopyThunk is not supported for a graph capture";
    return false;
  }

  if (!thunk_kinds_safe_for_capture.count(thunk->kind())) {
    VLOG(1) << Thunk::KindToString(thunk->kind())
            << " is not supported for graph capture";
    return false;
  }

  return true;
}

bool GpuExecutableSafeForGraphCapture(const ThunkSchedule* thunk_schedule) {
  return absl::c_all_of(thunk_schedule->TotalOrder(),
                        IsThunkSafeForGpuGraphCapture);
}

}  // namespace

// Implementation note: HLO profiling is always enabled for GPU executables,
// since we can use timers around thunks.
GpuExecutable::GpuExecutable(GpuExecutable::Params params)
    : Executable(std::move(params.debug_module),
                 std::move(params.hlo_profile_printer_data),
                 std::move(params.hlo_profile_index_map)),
      text_(std::move(params.asm_text)),
      binary_(std::move(params.binary)),
      gpu_version_(params.gpu_version),
      thunk_schedule_(std::move(params.thunk_schedule)),
      module_name_(params.module_name),
      output_shape_(params.output_shape),
      allocations_(std::move(params.allocations)),
      debug_buffer_assignment_(std::move(params.debug_buffer_assignment)),
      entry_computation_profile_index_(params.entry_computation_profile_index),
      constants_(std::move(params.constants)),
      output_info_(std::move(params.output_info)) {
  XlaDebugInfoManager::Get()->RegisterModule(module_name_, shared_module(),
                                             debug_buffer_assignment_);
}

GpuExecutable::~GpuExecutable() {
  XlaDebugInfoManager::Get()->UnregisterModule(module_name_, shared_module(),
                                               debug_buffer_assignment_);

  {
    // We could have issued host->device mem copies in ResolveConstantGlobals.
    // Wait for those to finish so that we can safely deallocate the backing HLO
    // module.
    //
    // We need for the host->device memcpies to finish they are concurrently
    // reading memory (xla::Literal's) owned by the HLO module.
    tensorflow::mutex_lock lock(module_handle_mutex_);
    for (const auto& pair : module_globals_) {
      CHECK(pair.first->SynchronizeAllActivity());
    }
  }

  if (GetCanUseGraphCaptureFlag() && VLOG_IS_ON(1)) {
    VLOG(1) << "For gpu_executable " << this
            << " Hits: " << graph_stats_.cache_hits.load()
            << " Misses: " << graph_stats_.cache_miss.load() << " called # "
            << graph_stats_.times_called.load();
    VLOG(4) << "Temp buffer cache hits: "
            << graph_stats_.temp_buffer_cache_hits.load();
    VLOG(4) << "This executable " << this << " has encountered "
            << temp_buffer_base_to_bufs_keys_map_.size()
            << " different temporary buffer base ptrs during the allocation "
               "process";
    if (VLOG_IS_ON(4)) {
      for (auto it : temp_buffer_base_to_bufs_keys_map_) {
        LOG(INFO) << "Temp buffer with TempBufferKey hash of " << it.first
                  << " is the same for " << it.second.size()
                  << " unique buffer keys";
      }
    }
    if (graph_stats_.times_called.load() > 0) {
      VLOG(1) << "Mem cache hit rate of this executable " << this << " is "
              << (graph_stats_.cache_hits.load() * 100) /
                     graph_stats_.times_called.load()
              << "%";
    }
    VLOG(4) << "Most recent enqueued hash hits: "
            << graph_stats_.last_buf_key_hits.load();
    if (graph_stats_.cache_hits.load() > 0 && VLOG_IS_ON(4)) {
      VLOG(4) << " Most recent enqueued hash hit rate: "
              << (graph_stats_.last_buf_key_hits.load() * 100) /
                     graph_stats_.cache_hits.load();
    }
  }

  while (!gpu_exec_graphs_cache_.empty()) {
    auto* gpu_context = static_cast<stream_executor::gpu::GpuContext*>(
        gpu_exec_graphs_cache_.begin()->first);
    VLOG(1) << "Cache size for gpu_executable " << this << " and gpu_context "
            << gpu_context << " is "
            << gpu_exec_graphs_cache_[gpu_context].GetCurrentCacheSize();
    auto& cache = gpu_exec_graphs_cache_.begin()->second;
    auto& exec_graphs = cache.GetGpuExecGraphs();

    while (!exec_graphs.empty()) {
      GetExecutor()->DestroyExecutableGraph(
          gpu_exec_graphs_cache_.begin()->first, exec_graphs.front());
      exec_graphs.erase(exec_graphs.begin());
    }
    gpu_exec_graphs_cache_.erase(gpu_exec_graphs_cache_.begin());
  }
}

bool GpuExecutable::CanUseGpuGraphCapture() {
  if (!can_use_gpu_graph_capture_.first) {
    can_use_gpu_graph_capture_.first = true;
    SetCanUseGraphCaptureFlag(
        GpuExecutableSafeForGraphCapture(thunk_schedule_.get()));
  }
  return GetCanUseGraphCaptureFlag();
}

Status GpuExecutable::CheckCompatibilityWithServiceExecutableRunOptions(
    const ServiceExecutableRunOptions* run_options) {
  se::Stream* main_stream = run_options->stream();

  stream_executor::PlatformKind platform_kind =
      main_stream->parent()->platform_kind();
  if (platform_kind == stream_executor::PlatformKind::kROCm) {
    int stream_isa_version;
    main_stream->parent()->GetDeviceDescription().rocm_amdgpu_isa_version(
        &stream_isa_version);
    int gpu_exec_isa_version =
        absl::get<std::pair<int, std::string>>(gpu_version_).first;
    TF_RET_CHECK(stream_isa_version == gpu_exec_isa_version)
        << "AMDGPU GCN ISA version mismatch; expected {" << gpu_exec_isa_version
        << ", but was " << stream_isa_version;
  } else if (platform_kind == stream_executor::PlatformKind::kCuda) {
    GpuVersion cc = main_stream->GetCudaComputeCapability();
    TF_RET_CHECK(absl::get<se::CudaComputeCapability>(cc) ==
                 absl::get<se::CudaComputeCapability>(gpu_version_))
        << "Compute capability mismatch; expected {"
        << absl::get<se::CudaComputeCapability>(gpu_version_).ToString()
        << "}, but was {" << absl::get<se::CudaComputeCapability>(cc).ToString()
        << "}";
  } else {
    return InternalError("Unknown platform: %d", platform_kind);
  }

  return Status::OK();
}

Status GpuExecutable::ExecuteThunks(
    const ServiceExecutableRunOptions* run_options,
    const BufferAllocations& buffer_allocations, bool block_host_until_done,
    HloExecutionProfile* hlo_execution_profile) {
  TF_RETURN_IF_ERROR(
      CheckCompatibilityWithServiceExecutableRunOptions(run_options));
  XlaDebugInfoManager::Get()->OnModuleStart(module_name_);
  auto cleanup = MakeCleanup(
      [&]() { XlaDebugInfoManager::Get()->OnModuleStop(module_name_); });

  se::Stream* main_stream = run_options->stream();
  se::StreamExecutor* executor = main_stream->parent();
  SetExecutor(executor->implementation());

  bool do_profile = hlo_execution_profile != nullptr;
  if (do_profile) {
    LOG(WARNING) << "PROFILING: profiling is enabled";
  }

  // Stream 0 indicates `main_stream` and substreams start from stream 1.
  std::vector<StreamPool::Ptr> sub_streams;
  sub_streams.reserve(thunk_schedule_->StreamCount() - 1);
  while (sub_streams.size() + 1 < thunk_schedule_->StreamCount()) {
    sub_streams.emplace_back();
    TF_ASSIGN_OR_RETURN(sub_streams.back(),
                        run_options->BorrowStream(executor->device_ordinal()));
  }

  se::Stream* capture_stream = main_stream;
  StreamPool::Ptr private_capture_stream;
  bool use_gpu_graph_capture =
      GpuGraphCaptureEnabled(module_name_) && CanUseGpuGraphCapture() &&
      executor->platform_kind() == stream_executor::PlatformKind::kCuda &&
      !is_graph_capture_costly_;
  if (use_gpu_graph_capture) {
    // We need a private stream for capturing to avoid interference from other
    // threads.
    TF_ASSIGN_OR_RETURN(private_capture_stream,
                        run_options->BorrowStream(executor->device_ordinal()));
    capture_stream = private_capture_stream.get();
  }

  HloExecutionProfiler profiler(do_profile, hlo_execution_profile, main_stream,
                                sub_streams, entry_computation_profile_index_);
  uint64 start_micros = tensorflow::Env::Default()->NowMicros();

  tensorflow::profiler::TraceMe hlo_module_activity(
      [&] { return absl::StrCat(module_name_, ":XLA GPU module"); },
      tensorflow::profiler::TraceMeLevel::kInfo);

  std::vector<std::function<void()>> deferred_host_callbacks;
  if (!use_gpu_graph_capture) {
    TF_RETURN_IF_ERROR(ExecuteThunkSequence(
        run_options, buffer_allocations, profiler, main_stream, capture_stream,
        sub_streams, deferred_host_callbacks));
  } else {
    auto bufs_key = buffer_allocations.Key(temp_buffer_base_);
    auto temp_buf_key = buffer_allocations.TempBufferKey(temp_buffer_base_);
    VLOG(3) << "For gpu executable " << this
            << " tmp buffer base: " << temp_buffer_base_.opaque()
            << " key hash: "
            << buffer_allocations.Key(temp_buffer_base_).hash();

    stream_executor::gpu::GpuContext* gpu_context =
        static_cast<stream_executor::gpu::GpuExecutor*>(GetExecutor())
            ->gpu_context();
    tensorflow::mutex_lock lock(module_handle_mutex_);
    auto& graph_exec_cache = gpu_exec_graphs_cache_[gpu_context];

    // 'Initialize' sets the gpu_context and the max LRU cache size. This is
    // only done once though. Once initialized, the subsequent calls are no-ops.
    graph_exec_cache.Initialize(gpu_context);
    auto exec_graph = graph_exec_cache.GetExecGraph(bufs_key);

    // If there is a cache hit, simply launch the existing executable graph.
    if (exec_graph) {
      // The temp_buffer_base should match the temp_buff_base of the bufs_key
      // that is cached.
      auto buf_keys_set =
          temp_buffer_base_to_bufs_keys_map_[temp_buf_key.hash()];
      DCHECK(buf_keys_set.find(bufs_key) == buf_keys_set.end())
          << " The temp_buffer_base should match the temp_buff_base of "
             "the bufs_key that is cached.";
      graph_stats_.cache_hits++;
      graph_stats_.temp_buffer_cache_hits++;
      graph_stats_.times_called++;
      if (bufs_key.hash() == graph_stats_.last_buf_key_hash.load()) {
        graph_stats_.last_buf_key_hits++;
      }
      if (VLOG_IS_ON(2)) {
        VLOG(2) << "CACHE HIT -> Launching graph " << exec_graph << " blindly!"
                << " Hits: " << graph_stats_.cache_hits.load()
                << " Misses: " << graph_stats_.cache_miss.load() << " called # "
                << graph_stats_.times_called.load()
                << " Cache size: " << graph_exec_cache.GetCurrentCacheSize();
        VLOG(4) << "Most recent enqueued hash: " << bufs_key.hash()
                << "Most recent enqueued hash hits: "
                << graph_stats_.last_buf_key_hits.load();
      }

      // Launch existing exec graph
      main_stream->ThenLaunchGraph(exec_graph);
    } else {
      // In case of a cache miss, do the following:
      // 1. Re-capture the thunk sequence in new template graph,
      // 2. Instantiate the template graph (cuGraph) to construct a new
      // executable (cuExecGraph)graph,
      // 3. Cache the new executable graph with the buffer address key,
      // 4. Launch new executable graph.
      if (temp_buffer_base_to_bufs_keys_map_.find(temp_buf_key.hash()) !=
          temp_buffer_base_to_bufs_keys_map_.end()) {
        graph_stats_.temp_buffer_cache_hits++;
        if (VLOG_IS_ON(3)) {
          VLOG(3) << "CACHE MISS Temp Buffer cache HIT";
          VLOG(3) << "Cache hits till this point: "
                  << graph_stats_.cache_hits.load() << " and Temp Buffer hits: "
                  << graph_stats_.temp_buffer_cache_hits.load();
          VLOG(3)
              << "Number of buf keys for this temp buffer of hash "
              << temp_buf_key.hash() << " = "
              << temp_buffer_base_to_bufs_keys_map_[temp_buf_key.hash()].size();
        }
      }
      graph_stats_.cache_miss++;
      graph_stats_.times_called++;
      graph_stats_.last_buf_key_hash = bufs_key.hash();
      if (VLOG_IS_ON(2)) {
        VLOG(2) << "CACHE MISS"
                << " Hits: " << graph_stats_.cache_hits.load()
                << " Misses: " << graph_stats_.cache_miss.load() << " called # "
                << graph_stats_.times_called
                << " Cache size: " << graph_exec_cache.GetCurrentCacheSize();
        VLOG(3) << "Temp buffer Hits: "
                << graph_stats_.temp_buffer_cache_hits.load();
        VLOG(4) << "Most recently enqueued hash: " << bufs_key.hash()
                << "Most recent enqueued hash hits: "
                << graph_stats_.last_buf_key_hits.load();
      }

      // Begin capture template graph
      capture_stream->ThenBeginGraphCapture();

      TF_RETURN_IF_ERROR(ExecuteThunkSequence(
          run_options, buffer_allocations, profiler, main_stream,
          capture_stream, sub_streams, deferred_host_callbacks));

      void* graph = nullptr;

      // End capture template graph
      capture_stream->ThenEndGraphCapture(graph);

      // Instantiate exec graph
      StatusOr<void*> status =
          GetExecutor()->InstantiateGraph(graph, exec_graph);
      if (!status.ok()) {
        return InternalError(
            "Failed to instantiate GPU execution graph on stream %p: %s",
            main_stream, status.status().error_message());
      }
      exec_graph = status.ValueOrDie();

      // Add exec graph to Cache
      bool has_reached_max_cache_size =
          gpu_exec_graphs_cache_[gpu_context].AddToCache(bufs_key, exec_graph);

      // Heuristic to check whether using graphs for this gpu_executable is
      // proving to be expensive due to low hit rate. If the hit rate is less
      // than equal 20% there is no point in using graphs for this executable.
      if (has_reached_max_cache_size &&
          graph_stats_.get_cache_hit_rate() <= 20) {
        VLOG(1) << "The maximum LRU cache size has been reached but the "
                   "cache hit rate is still "
                << graph_stats_.get_cache_hit_rate()
                << ". Hence aborting graph capture for executable " << this;
        is_graph_capture_costly_ = true;
      }

      temp_buffer_base_to_bufs_keys_map_[temp_buf_key.hash()].insert(bufs_key);

      // Launch exec graph
      main_stream->ThenLaunchGraph(exec_graph);

      // Destroy template graph
      GetExecutor()->DestroyGraph(gpu_context, graph);

    }  // End of graph launch conditional.
  }

  if (!deferred_host_callbacks.empty()) {
    auto fn = [deferred_host_callbacks{std::move(deferred_host_callbacks)}]() {
      for (auto& callback : deferred_host_callbacks) {
        callback();
      }
    };
    if (run_options->run_options().then_execute_function()) {
      (*run_options->run_options().then_execute_function())(main_stream,
                                                            std::move(fn));
    } else {
      main_stream->ThenDoHostCallback(std::move(fn));
    }
  }

  // Make sure kernels are completed before deallocating temporary buffers or
  // the profiler state.
  // TODO(b/30100571): we could potentially postpone deallocating the temp
  // buffers until a different computation is executed.
  if (do_profile || block_host_until_done) {
    Status block_status = main_stream->BlockHostUntilDone();
    if (!block_status.ok()) {
      return InternalError(
          "Failed to complete all kernels launched on stream %p: %s",
          main_stream, block_status.error_message());
    }
  }

  // FinishExecution() blocks until main_stream has completed if profiling is
  // enabled; we therefore do not need to defer profile collection onto a
  // stream.
  profiler.FinishExecution();
  uint64 end_micros = tensorflow::Env::Default()->NowMicros();

  if (run_options->run_options().execution_profile()) {
    ExecutionProfile* profile = run_options->run_options().execution_profile();
    const double nanoseconds = (end_micros - start_micros) * 1000.0;
    profile->set_compute_time_ns(std::max(nanoseconds, 1.0));

    // If hlo profiling was disabled then the cycle count is left empty.
    if (do_profile) {
      profile->set_compute_cycle_count(hlo_execution_profile->GetCyclesTakenBy(
          entry_computation_profile_index_));
    }
  }

  return Status::OK();
}

Status GpuExecutable::ExecuteThunkSequence(
    const ServiceExecutableRunOptions* run_options,
    const BufferAllocations& buffer_allocations, HloExecutionProfiler& profiler,
    se::Stream* main_stream, se::Stream* capture_stream,
    const std::vector<StreamPool::Ptr>& sub_streams,
    std::vector<std::function<void()>>& deferred_host_callbacks) {
  // se::StreamExecutor* executor = main_stream->parent();
  absl::flat_hash_map<const Thunk*, std::unique_ptr<se::Event>>
      thunk_to_finish_event;
  // bool scoped_annotation_enabled = ScopedAnnotation::IsEnabled();
  for (const std::unique_ptr<Thunk>& thunk : thunk_schedule_->TotalOrder()) {
    // Annotate execution of this op if tracing was enabled when we started
    // running this module.  If tracing is enabled *while* we're running the
    // module, we won't get any data, but that's probably an OK trade-off.
    ScopedAnnotation annotation([&] { return thunk->profile_annotation(); });

    int32 stream_no = thunk_schedule_->StreamNumberForThunk(thunk.get());
    se::Stream* stream =
        (stream_no == 0 ? capture_stream : sub_streams[stream_no - 1].get());

    for (const Thunk* dependency : thunk_schedule_->DependsOn(thunk.get())) {
      if (VLOG_IS_ON(3)) {
        VLOG(3) << Thunk::KindToString(thunk->kind()) << " depends on "
                << Thunk::KindToString(dependency->kind());
      }
      stream->ThenWaitFor(FindOrDie(thunk_to_finish_event, dependency).get());
    }

    if (VLOG_IS_ON(3)) {
      std::string copy_type = dynamic_cast<HostToDeviceCopyThunk*>(thunk.get())
                                  ? "HostToDeviceCopyThunk"
                                  : "";
      VLOG(3) << "Executing thunk of kind "
              << Thunk::KindToString(thunk->kind()) << copy_type;
      VLOG(4) << "Executing the thunk for " << thunk->profile_annotation()
              << " on stream " << stream_no;
    }
    const GpuExecutableRunOptions* gpu_options =
        run_options->run_options().gpu_executable_run_options();

    Thunk::ExecuteParams thunk_params{
        &buffer_allocations,
        stream,
        run_options->run_options().run_id(),
        &profiler,
        run_options->run_options().device_assignment(),
        &deferred_host_callbacks,
        gpu_options && gpu_options->gpu_global_device_ids()
            ? &*gpu_options->gpu_global_device_ids()
            : nullptr,
        gpu_options && gpu_options->nccl_unique_id_callback()
            ? &gpu_options->nccl_unique_id_callback()
            : nullptr};
    TF_RETURN_IF_ERROR(thunk->ExecuteOnStream(thunk_params));
    if (thunk_schedule_->Depended(thunk.get())) {
      if (VLOG_IS_ON(3)) {
        VLOG(3) << " " << Thunk::KindToString(thunk->kind())
                << " is depended by another thunk"
                << ". Hence pushing finish_event.";
      }
      auto finish_event = absl::make_unique<se::Event>(main_stream->parent());
      finish_event->Init();
      stream->ThenRecordEvent(finish_event.get());
      thunk_to_finish_event[thunk.get()] = std::move(finish_event);
    }
  }
  capture_stream->ThenWaitFor(&sub_streams);

  return Status::OK();
}

StatusOr<const GpuExecutable::BufferAllocToDeviceMemoryMap*>
GpuExecutable::ResolveConstantGlobals(se::Stream* stream) {
  se::StreamExecutor* executor = stream->parent();

  tensorflow::mutex_lock lock(module_handle_mutex_);
  auto it = module_globals_.find(executor);
  if (it != module_globals_.end()) {
    return &it->second;
  }

  se::MultiModuleLoaderSpec module_spec;
  if (!binary().empty()) {
    module_spec.AddCudaCubinInMemory(binary());
  }
  module_spec.AddCudaPtxInMemory(text().c_str());

  absl::flat_hash_map<int64, se::DeviceMemoryBase> globals;
  if (executor->platform_kind() == se::PlatformKind::kCuda &&
      module_spec.cuda_ptx_in_memory() == nullptr) {
    // No custom PTX => no globals.
    return &module_globals_.emplace(executor, std::move(globals)).first->second;
  }

  se::ModuleHandle module_handle;
  TF_RETURN_IF_ERROR(executor->LoadModule(module_spec, &module_handle));

  for (const auto& info : constants_) {
    TF_ASSIGN_OR_RETURN(auto global, executor->GetUntypedSymbol(
                                         info.symbol_name, module_handle));
    VLOG(3) << "Resolved global " << info.symbol_name << " to "
            << global.opaque();

    if (!info.content.empty()) {
      stream->ThenMemcpy(&global, info.content.data(), info.content.size());
    }

    if (info.allocation_index != -1) {
      InsertOrDie(&globals, info.allocation_index, global);
    }
  }

  module_handles_.emplace(executor,
                          se::ScopedModuleHandle(executor, module_handle));
  return &module_globals_.emplace(executor, std::move(globals)).first->second;
}

StatusOr<se::DeviceMemoryBase> GpuExecutable::BufferForAllocation(
    VariantArguments arguments,
    const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
    const BufferAllocation& allocation,
    se::DeviceMemoryAllocator* const memory_allocator, int device_ordinal,
    int64 arg_idx) {
  if (allocation.is_thread_local()) {
    return se::DeviceMemoryBase{};
  } else if (allocation.is_entry_computation_parameter()) {
    int64 param_no = allocation.parameter_number();
    se::DeviceMemoryBase registered_buffer = [&] {
      if (auto unowned_shapedbuffers =
              absl::get_if<absl::Span<const ShapedBuffer* const>>(&arguments)) {
        return (*unowned_shapedbuffers)[param_no]->buffers().element(
            allocation.param_shape_index());
      } else {
        return absl::get<absl::Span<ExecutionInput>>(arguments)[param_no]
            .Buffer(allocation.param_shape_index())
            .AsDeviceMemoryBase();
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
      return se::DeviceMemoryBase();
    }
    return it->second;
  } else {
    // Allocate each allocation that might escape, or is the temp buffer.
    CHECK(allocation.maybe_live_out() || allocation.IsPreallocatedTempBuffer());
    const int64 buffer_size = allocation.size();
    se::DeviceMemoryBase buffer_address;
    if (buffer_size > 0) {
      TF_ASSIGN_OR_RETURN(
          se::OwningDeviceMemory buffer,
          memory_allocator->Allocate(device_ordinal, buffer_size));
      buffer_address = buffer.Release();
    }
    if (allocation.IsPreallocatedTempBuffer()) {
      temp_buffer_base_ = buffer_address;
    }
    return buffer_address;
  }
}

static Status CheckAlignment(const BufferAllocation& allocation,
                             se::DeviceMemoryBase buffer, int arg_idx) {
  const int64 expected_alignment = [&] {
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
    return InternalError(
        "Address of buffer %d must be a multiple of %x, but "
        "was %p",
        arg_idx, expected_alignment, buffer.opaque());
  }
  return Status::OK();
}

StatusOr<BufferAllocations> GpuExecutable::GenerateBufferAllocations(
    VariantArguments arguments,
    const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
    se::DeviceMemoryAllocator* const memory_allocator,
    se::StreamExecutor* executor) {
  tensorflow::profiler::TraceMe hlo_module_activity(
      [&] { return std::string("Build buffer allocations"); },
      tensorflow::profiler::TraceMeLevel::kInfo);

  const int64 num_buffers = allocations_.size();
  std::vector<se::DeviceMemoryBase> buffers;
  buffers.reserve(num_buffers);
  for (int64 i = 0; i < num_buffers; ++i) {
    const BufferAllocation& allocation = allocations_[i];
    TF_ASSIGN_OR_RETURN(
        se::DeviceMemoryBase buffer,
        BufferForAllocation(arguments, globals, allocation, memory_allocator,
                            executor->device_ordinal(), i));
    buffers.push_back(buffer);
    TF_RETURN_IF_ERROR(CheckAlignment(allocation, buffer, i));
  }
  return {{buffers, executor->device_ordinal(), memory_allocator}};
}

StatusOr<ExecutionOutput> GpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  return ExecuteAsyncOnStreamImpl(run_options, absl::MakeSpan(arguments),
                                  hlo_execution_profile);
}

StatusOr<ScopedShapedBuffer> GpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  TF_ASSIGN_OR_RETURN(
      ExecutionOutput out,
      ExecuteAsyncOnStreamImpl(run_options, arguments, hlo_execution_profile));
  return out.ConsumeResult();
}

StatusOr<ExecutionOutput> GpuExecutable::ExecuteAsyncOnStreamImpl(
    const ServiceExecutableRunOptions* run_options, VariantArguments arguments,
    HloExecutionProfile* hlo_execution_profile) {
  XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
      "GpuExecutable::ExecuteAsyncOnStreamImpl(", module_name_, ")"));
  se::DeviceMemoryAllocator* const memory_allocator = run_options->allocator();
  // Force synchronous execution if the allocator requires it.
  const bool block_host_until_done =
      !memory_allocator->AllowsAsynchronousDeallocation();

  const GpuExecutable::BufferAllocToDeviceMemoryMap* globals;
  {
    tensorflow::profiler::TraceMe hlo_module_activity(
        [&] { return std::string("Resolve constant globals"); },
        tensorflow::profiler::TraceMeLevel::kInfo);

    TF_ASSIGN_OR_RETURN(globals, ResolveConstantGlobals(run_options->stream()));
  }

  se::StreamExecutor* executor = run_options->stream()->parent();

  auto device_ordinal = executor->device_ordinal();
  ExecutionOutput result(/*on_device_shape=*/output_shape_, memory_allocator,
                         device_ordinal);

  TF_ASSIGN_OR_RETURN(BufferAllocations buffer_allocations,
                      GenerateBufferAllocations(arguments, globals,
                                                memory_allocator, executor));
  VLOG(2) << buffer_allocations.ToString();
  std::set<se::DeviceMemoryBase> buffers_in_result;

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
        &allocations_[output_info.allocation_index];
    se::DeviceMemoryBase& result_buffer = p.second;

    VLOG(4) << "Looking at: allocation " << output_info.allocation_index
            << " @ index: " << index.ToString();

    if (output_info.alias_config) {
      MaybeOwningDeviceMemory* maybe_owning_memory =
          [&]() -> xla::MaybeOwningDeviceMemory* {
        // ScopedBuffer is never an owned buffer.
        if (auto* unowned_shapedbuffers =
                absl::get_if<absl::Span<const ShapedBuffer* const>>(
                    &arguments)) {
          return nullptr;
        } else {
          auto unowned_execution_input =
              absl::get<absl::Span<ExecutionInput>>(arguments);
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
        absl::optional<tensorflow::se::OwningDeviceMemory> owning =
            maybe_owning_memory->Release();
        // If the caller passes the ownership of the device memory, reuse it
        // as the output buffer. It is up to the caller whether or not to
        // donate a buffer; the aliasing information describes which buffers
        // may alias, not buffers that must alias.
        se::DeviceMemoryBase argument_buffer = owning->Release();
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
                 !ShapeUtil::GetSubshape(output_shape_, index).IsTuple()) {
        // The guard is above is not to insert copy-protection when aliasing
        // pass-through params, as we do not need to write into the output
        // buffer.
        VLOG(3) << "Using copy-protection: aliasing is specified, but the "
                   "buffer is not donated; allocating a fresh buffer";
        int64 allocation_size =
            ShapeUtil::ByteSizeOf(ShapeUtil::GetSubshape(output_shape_, index));
        TF_ASSIGN_OR_RETURN(
            se::OwningDeviceMemory allocated_buffer,
            memory_allocator->Allocate(device_ordinal, allocation_size));
        result_buffer = allocated_buffer.Release();
        se::DeviceMemoryBase& aliased_buffer =
            buffer_allocations.GetMutableDeviceAddress(
                output_info.allocation_index);
        CHECK_EQ(aliased_buffer.size(), result_buffer.size());
        run_options->stream()->ThenMemcpyD2D(&result_buffer, aliased_buffer,
                                             aliased_buffer.size());
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

  for (const std::unique_ptr<Thunk>& thunk : thunk_schedule_->TotalOrder()) {
    TF_RETURN_IF_ERROR(thunk->Initialize(*this, executor));
  }
  TF_RETURN_IF_ERROR(ExecuteThunks(run_options, buffer_allocations,
                                   block_host_until_done,
                                   hlo_execution_profile));

  // Free all temporary allocations.
  TF_RETURN_IF_ERROR(
      buffer_allocations.TearDown(buffers_in_result, allocations_));

  // Free allocations for arguments.
  if (auto args = absl::get_if<absl::Span<ExecutionInput>>(&arguments)) {
    MarkToBeReleasedArguments(*args, result);
  }
  return std::move(result);
}

int64 GpuExecutable::SizeOfGeneratedCodeInBytes() const {
  // Non-empty PTX but empty cubin: compilation must have failed, return
  // "unknown".
  if (binary().empty() && !text_.empty()) {
    return -1;
  }
  int64 size = binary().size();
  for (BufferAllocation::Index i = 0; i < allocations_.size(); ++i) {
    const BufferAllocation& allocation = allocations_[i];
    if (allocation.is_constant()) {
      size += allocation.size();
    }
  }
  return size;
}

StatusOr<absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>>
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
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      root->shape(),
      [&](const Shape& /*sub_shape*/, const ShapeIndex& index) -> Status {
        const auto& sources = root_value_set.element(index);
        // The points-to set is unambiguous so the set should be a
        // singleton. That is, we know exactly which instruction
        // produced the array at this element.
        CHECK_EQ(1, sources.values().size());
        HloInstruction* src_hlo = sources.values()[0]->instruction();

        GpuExecutable::OutputInfo& info = output[index];
        info.passthrough = src_hlo->opcode() == HloOpcode::kParameter;
        TF_ASSIGN_OR_RETURN(
            const BufferAllocation::Slice slice,
            assignment.GetUniqueSlice(src_hlo, sources.values()[0]->index()));
        CHECK_EQ(slice.offset(), 0) << "Parameter should get its own slice";
        info.allocation_index = slice.index();

        output[index].alias_config =
            hlo_module.input_output_alias_config().GetAliasedParameter(index);

        return Status::OK();
      }));
  return output;
}

}  // namespace gpu
}  // namespace xla
