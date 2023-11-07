/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/runtime/graph_launch.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/runtime/custom_call.h"
#include "xla/runtime/executable.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/non_atomically_upgradeable_rw_lock.h"
#include "xla/service/gpu/runtime/concurrent_region.h"
#include "xla/service/gpu/runtime/conv.h"
#include "xla/service/gpu/runtime/gemm.h"
#include "xla/service/gpu/runtime/kernel_launch.h"
#include "xla/service/gpu/runtime/support.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/statusor.h"
#include "tsl/profiler/lib/profiler_lock.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "xla/stream_executor/gpu/gpu_graph.h"
#endif  // #if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace xla {
namespace gpu {

using tsl::profiler::TraceMe;
using tsl::profiler::TraceMeEncode;

using xla::runtime::Arguments;
using xla::runtime::AsyncTaskRunner;
using xla::runtime::CustomCall;
using xla::runtime::Executable;
using xla::runtime::FunctionRef;
using xla::runtime::FunctionType;
using xla::runtime::MemrefDesc;
using xla::runtime::MemrefType;
using xla::runtime::StridedMemrefView;

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
using se::gpu::OwnedGpuGraph;

// Captures Gpu graph by running given function in capture mode.
static absl::StatusOr<OwnedGpuGraph> CaptureGraph(
    const ServiceExecutableRunOptions* run_options,
    runtime::FunctionRef function_ref, Arguments<MemrefDesc>& args,
    CustomCall::UserData user_data);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

//===----------------------------------------------------------------------===//
// GPU graphs caching.
//===----------------------------------------------------------------------===//

struct GraphInstances::Impl {
  struct State {
    // A flag signalling if `InstantiateAllGraphs` was already called and we
    // have all Gpu graph instantiated ahead of time.
    bool instantiated = false;

    // Last time graph instances were used by a particular stream executor.
    uint64_t last_use_micros = 0;

    std::shared_ptr<StreamExecutorGraphInstances> instances =
        std::make_shared<StreamExecutorGraphInstances>();
  };

  // XLA module name that owns graph instances. We use it only to produce logs
  // that can be attributed back to XLA executables.
  std::string module_name;

  // Number of graphs in the parent module.
  int64_t num_graphs = 0;

  mutable absl::Mutex mu;
  absl::node_hash_map<se::StreamExecutor*, State> graphs ABSL_GUARDED_BY(mu);
};

// Keep track of instantiated graphs on each StreamExecutor, we use this
// information in the graph eviction policy.
using GraphInstancesState = absl::flat_hash_map<se::StreamExecutor*, int64_t>;

static absl::Mutex* GetGraphInstancesStateMutex() {
  static auto* mu = new absl::Mutex();
  return mu;
}

static GraphInstancesState& GetGraphInstancesState() {
  static auto* state = new GraphInstancesState();
  return *state;
}

static int64_t NotifyGraphInstancesCreated(se::StreamExecutor* executor,
                                           int64_t num_graphs) {
  absl::MutexLock lock(GetGraphInstancesStateMutex());
  return GetGraphInstancesState()[executor] += num_graphs;
}

static int64_t NotifyGraphInstancesDestroyed(se::StreamExecutor* executor,
                                             int64_t num_graphs) {
  absl::MutexLock lock(GetGraphInstancesStateMutex());
  return GetGraphInstancesState()[executor] -= num_graphs;
}

// We keep track of all graph instances in the process, to implement graph
// eviction on OOM. Graph instances owned by GpuExecutable, so we rely on
// weak ptr to check if they are still alive.
using GraphInstancesVec = std::vector<std::weak_ptr<GraphInstances::Impl>>;

static absl::Mutex* GetGraphInstancesVecMutex() {
  static auto* mu = new absl::Mutex();
  return mu;
}

static GraphInstancesVec& GetGraphInstancesVec() {
  static auto* vec = new GraphInstancesVec();
  return *vec;
}

static void AddGraphInstances(std::weak_ptr<GraphInstances::Impl> impl) {
  absl::MutexLock lock(GetGraphInstancesVecMutex());
  GetGraphInstancesVec().push_back(std::move(impl));
}

// Evicts all graphs for a given executor in the current process.
static void EvictAllGraphs(
    se::StreamExecutor* executor,
    std::optional<uint64_t> eviction_timeout_seconds = std::nullopt) {
  // We WARN only when we evict all Gpu graphs because it happens when we
  // recover from OOM. Eviction by time out is business as usual.
  if (eviction_timeout_seconds.has_value()) {
    VLOG(3) << "Evict timed out gpu graphs from executor " << executor;
  } else {
    LOG(WARNING) << "Evict all gpu graphs from executor " << executor;
  }

  TraceMe trace_instantiation([&] {
    return TraceMeEncode("cuda.graph.evict_all_graphs",
                         {{"device_ordinal", executor->device_ordinal()}});
  });

  absl::MutexLock lock(GetGraphInstancesVecMutex());
  auto& vec = GetGraphInstancesVec();

  // Erase all expired graph instances.
  vec.erase(std::remove_if(vec.begin(), vec.end(),
                           [](auto& weak_ptr) { return weak_ptr.expired(); }),
            vec.end());

  auto timed_out = [&](GraphInstances::Impl::State& state) -> bool {
    if (!eviction_timeout_seconds.has_value()) {
      return false;
    }

    auto diff = tsl::Env::Default()->NowMicros() - state.last_use_micros;
    return (diff / (1000 * 1000)) > *eviction_timeout_seconds;
  };

  for (auto& weak_ptr : vec) {
    auto ptr = weak_ptr.lock();
    if (!ptr) continue;

    if (!ptr->mu.TryLock()) continue;

    auto it = ptr->graphs.find(executor);
    if (it == ptr->graphs.end()) {
      ptr->mu.Unlock();
      continue;
    }

    // If we have a timeout value, than check it first, otherwise always evict
    // graphs for a given executor.
    bool is_timed_out = timed_out(it->second);
    if (eviction_timeout_seconds.has_value() && !is_timed_out) {
      ptr->mu.Unlock();
      continue;
    }

    if (ptr->num_graphs > 0) {
      VLOG(3) << "Evict " << ptr->num_graphs << " graphs for: @"
              << ptr->module_name << " at executor: " << executor
              << " (timed_out = " << is_timed_out << ")."
              << " Total remaining graphs at given executor: "
              << NotifyGraphInstancesDestroyed(executor, ptr->num_graphs);
    }
    ptr->graphs.erase(it);
    ptr->mu.Unlock();
  }
}

GraphInstances::GraphInstances(std::string module_name, int64_t num_graphs)
    : impl_(std::make_shared<Impl>()) {
  impl_->module_name = std::move(module_name);
  impl_->num_graphs = num_graphs;
  if (impl_->num_graphs > 0) {
    VLOG(3) << "Construct graph instances cache for: @" << impl_->module_name
            << " (num_graphs = " << impl_->num_graphs << ")";
  }
  AddGraphInstances(impl_);
}

GraphInstances::~GraphInstances() {
  if (impl_->num_graphs > 0) {
    VLOG(3) << "Destroy graph instances cache for: @" << impl_->module_name
            << " (num_graphs = " << impl_->num_graphs << ")";

    absl::MutexLock lock(&impl_->mu);
    for (auto& [executor, state] : impl_->graphs) {
      VLOG(3) << "Destroy " << impl_->num_graphs << " graphs for: @"
              << impl_->module_name << " at executor: " << executor
              << ". Total remaining graphs at given executor: "
              << NotifyGraphInstancesDestroyed(executor, impl_->num_graphs);
    }
  }
}

std::shared_ptr<StreamExecutorGraphInstances> GraphInstances::operator()(
    se::StreamExecutor* executor) {
  absl::MutexLock lock(&impl_->mu);

  auto it = impl_->graphs.try_emplace(executor);
  if (it.second && impl_->num_graphs > 0) {
    VLOG(3) << "Instantiate " << impl_->num_graphs << " graphs for: @"
            << impl_->module_name << " at executor: " << executor
            << ". Total graphs at given executor: "
            << NotifyGraphInstancesCreated(executor, impl_->num_graphs);
  }

  Impl::State& state = it.first->second;
  state.last_use_micros = tsl::Env::Default()->NowMicros();
  return state.instances;
}

bool GraphInstances::InstantiatedAllGraphs(
    const ServiceExecutableRunOptions* run_options,
    const Executable& executable) {
  if (executable.num_functions() == 1) return true;

  absl::MutexLock lock(&impl_->mu);
  return impl_->graphs[run_options->stream()->parent()].instantiated;
}

Status GraphInstances::InstantiateAllGraphs(
    const ServiceExecutableRunOptions* run_options,
    const Executable& executable, const CustomCall::UserData& user_data,
    const BufferAllocations& buffer_allocations,
    absl::Span<const int64_t> buffer_sizes,
    absl::Span<const std::vector<int64_t>> allocation_indices,
    std::optional<uint64_t> eviction_timeout_seconds) {
  // We have only "main" function in the executable.
  if (executable.num_functions() == 1) return OkStatus();

  absl::MutexLock lock(&impl_->mu);
  se::StreamExecutor* executor = run_options->stream()->parent();

  Impl::State& state = impl_->graphs[executor];

  // All Gpu graphs are already instantiated for a given executor.
  if (state.instantiated) return OkStatus();

  TraceMe trace("gpu.graph.instantiate_all");

  // Evict all timeout graphs before trying to instantiate new ones.
  EvictAllGraphs(executor, eviction_timeout_seconds);

  // We'll retry graph instantiation on OOM errors after evicting all graphs
  // instantiated on `executor`.
  int32_t num_retries = 0;

  StreamExecutorGraphInstances::Snapshot instances =
      state.instances->snapshot();

  // Instantiate all Gpu graphs by calling graph capture functions with fake
  // arguments. Once we'll execute them first time for real, they'll be updated
  // with correct pointers.
  for (unsigned ordinal = 1; ordinal < executable.num_functions(); ++ordinal) {
    if (!absl::StartsWith(executable.function_name(ordinal),
                          "xla.gpu.graph.capture"))
      continue;

    VLOG(3) << "Instantiate Gpu graph defined by capture function @"
            << executable.function_name(ordinal) << " (ordinal = " << ordinal
            << ")";

    TraceMe trace_instantiation([&] {
      return TraceMeEncode("gpu.graph.instantiate", {{"ordinal", ordinal}});
    });

    FunctionRef function_ref = executable.function_ref(ordinal);

    const FunctionType& signature = executable.signature(ordinal);
    assert(signature.num_results() == 0 && "unexpected number of results");
    Arguments<MemrefDesc> args(signature.num_operands());

    // Mapping from graph capture argument to buffer allocation index.
    absl::Span<const int64_t> capture_allocs = allocation_indices[ordinal];
    if (capture_allocs.size() != signature.num_operands())
      return absl::InternalError(
          "Invalid number of allocation indices for a graph capture function");

    // Prepare arguments for the graph capture function.
    for (size_t j = 0; j < signature.num_operands(); ++j) {
      auto* memref = llvm::dyn_cast<MemrefType>(signature.operand(j));

      if (!memref)
        return absl::InternalError(absl::StrFormat(
            "Unsupported capture function argument type #%d", j));

      if (memref->sizes().size() != 1)
        return absl::InternalError(
            absl::StrFormat("Unsupported capture function memref rank #%d: %d",
                            j, memref->sizes().size()));

      std::array<int64_t, 1> sizes = {memref->size(0)};
      std::array<int64_t, 1> strides = {1};

      int64_t allocation_index = capture_allocs[j];
      args.emplace_back<MemrefDesc>(
          memref->element_type(),
          buffer_allocations.GetDeviceAddress(allocation_index).opaque(),
          /*offset=*/0, sizes, strides);
    }

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    // Instantiate a Gpu graph with fake arguments.
    auto instantiate = [&]() -> absl::StatusOr<GraphInstance> {
      TF_ASSIGN_OR_RETURN(
          auto g, CaptureGraph(run_options, function_ref, args, user_data));
      TF_ASSIGN_OR_RETURN(auto e, se::gpu::InstantiateGpuGraph(std::move(g)));
      return GraphInstance(0, std::move(e));
    };

    absl::StatusOr<GraphInstance*> instance =
        instances.GetOrCreate(ordinal, instantiate);

    if (instance.status().code() == absl::StatusCode::kResourceExhausted) {
      if (num_retries == 0) {
        // Retry on OOM error after evicting all graphs from executor.
        EvictAllGraphs(executor);
        num_retries++;
        ordinal--;  // we'll try to instantiate the same graph one more time
        continue;
      } else {
        LOG(WARNING) << "InstantiateAllGraph failed due to insufficient memory."
                        " Uninitializd graphs will run in op-by-op mode.";
        return OkStatus();
      }
    }

    // Otherwise return an error to the caller.
    if (!instance.ok()) return instance.status();
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  }

  state.instantiated = true;
  return OkStatus();
}

CapturedFunctionExecutionCount* CapturedFunctionExecutionCounts::operator()(
    se::StreamExecutor* executor) {
  absl::MutexLock lock(&mutex_);
  return &counts_[executor];
}

//===----------------------------------------------------------------------===//
// Helper structure to hash the remaining arguments' memref pointers.
//===----------------------------------------------------------------------===//

struct RemainingArgsPtrs {
  CustomCall::RemainingArgs args;
  se::DeviceMemoryBase* temp_buffer;

  template <typename H>
  friend H AbslHashValue(H h, const RemainingArgsPtrs& m);
};

template <typename H>
H AbslHashValue(H h, const RemainingArgsPtrs& m) {
  for (size_t i = 0; i < m.args.size(); ++i) {
    if (auto memref = m.args.get<StridedMemrefView>(i); succeeded(memref))
      h = H::combine(std::move(h), memref->data);
  }
  return std::move(H::combine(std::move(h), m.temp_buffer->opaque()));
}

//----------------------------------------------------------------------------//
// Runs capture function exported by the executable to construct a gpu graph.
//----------------------------------------------------------------------------//

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

static bool InDebugMode() {
#ifdef NDEBUG
  return false;
#endif
  return true;
}

// Forwards custom call arguments to an arguments container that can be passed
// to an executable function.
static absl::Status ForwardArguments(CustomCall::RemainingArgs fwd_args,
                                     Arguments<MemrefDesc>& args) {
  for (size_t i = 0; i < fwd_args.size(); ++i) {
    if (auto memref = fwd_args.get<StridedMemrefView>(i); succeeded(memref)) {
      args.emplace_back<MemrefDesc>(memref->dtype, memref->data, /*offset=*/0,
                                    memref->sizes, memref->strides);
      continue;
    }

    return absl::InvalidArgumentError("Unsupported argument type");
  }

  return OkStatus();
}

static absl::StatusOr<OwnedGpuGraph> CaptureGraph(
    const ServiceExecutableRunOptions* run_options,
    runtime::FunctionRef function_ref, Arguments<MemrefDesc>& args,
    CustomCall::UserData user_data) {
  // We capture graph on a borrowed stream because we do not want to
  // accidentally record any concurrent kernel launches from other XLA
  // executables.
  se::StreamExecutor* executor = run_options->stream()->parent();

  // Initialize (with memoization) BlasSupport here because cublasCreate fails
  // during gpu graph capturing.
  if (function_ref.RequiresBlas()) {
    if (!executor->AsBlas()) {
      return absl::InternalError("Failed to initialize BLAS support");
    }
  }

  StatusOr<StreamPool::Ptr> capture_stream =
      run_options->BorrowStream(executor->device_ordinal());

  if (!capture_stream.ok())
    return absl::InternalError(
        absl::StrFormat("Failed to borrow a stream for graph capture: %s",
                        capture_stream.status().message()));

  TraceMe trace([&] {
    return TraceMeEncode("gpu.graph.capture",
                         {{"ordinal", function_ref.ordinal()}});
  });

  // TODO(ezhulenev): Pass graph capture context explicitly to the custom calls
  // via UserData to be able to detect when executing custom call in graph
  // capture mode. Currently we rely on the fact that we know for sure that
  // operations in the graph capture function do not need anything except the
  // main stream (we capture only kernel launches).
  ExecutableRunOptions capture_run_options;
  capture_run_options.set_stream(capture_stream->get());

  const ServiceExecutableRunOptions capture_opts(capture_run_options);
  user_data.insert(&capture_opts);

  // Collect all emitted diagnostic messages.
  std::string diagnostic;
  runtime::DiagnosticEngine diagnostic_engine;
  AppendDiagnosticToString(diagnostic_engine, &diagnostic);

  // Prepare options for executing graph capture function.
  Executable::ExecuteOpts opts;
  opts.custom_call_data = &user_data;
  opts.diagnostic_engine = &diagnostic_engine;

  // Graph capture function should not launch any async tasks.
  opts.async_task_runner = reinterpret_cast<AsyncTaskRunner*>(0XDEADBEEF);

  // Create a graph from running the graph capture function.
  auto captured = se::gpu::CaptureGpuGraph(capture_stream->get(), [&]() {
    return function_ref(args, runtime::NoResultConverter{}, opts,
                        /*verify_arguments=*/InDebugMode())
        .status();
  });

  if (!captured.ok()) {
    return InternalError("CaptureGpuGraph failed (%s): %s",
                         diagnostic.empty() ? "<no details>" : diagnostic,
                         captured.status().ToString());
  }
  return std::move(*captured);
}

// When graph execution is disabled we run the graph capture function in
// "regular" mode and execute all operation one by one.
static absl::Status RunGraphOpByOp(
    const ServiceExecutableRunOptions* run_options,
    runtime::FunctionRef function_ref, CustomCall::RemainingArgs fwd_args,
    CustomCall::UserData user_data) {
  // Prepare options for executing graph capture function.
  Executable::ExecuteOpts opts;
  auto* concurrent_region_status = user_data.get<ConcurrentRegionStatus>();
  // Ops should not run in parallel during op-by-op execution.
  concurrent_region_status->DisableConcurrentRegion();
  opts.custom_call_data = &user_data;

  TraceMe trace([&] {
    return TraceMeEncode("gpu.graph.run_op_by_op_fallback",
                         {{"ordinal", function_ref.ordinal()}});
  });

  // Collect all emitted diagnostic messages.
  std::string diagnostic;
  runtime::DiagnosticEngine diagnostic_engine;
  AppendDiagnosticToString(diagnostic_engine, &diagnostic);

  opts.diagnostic_engine = &diagnostic_engine;

  // Graph capture function should not launch any async tasks.
  opts.async_task_runner = reinterpret_cast<AsyncTaskRunner*>(0XDEADBEEF);

  Arguments<MemrefDesc> args(fwd_args.size());
  TF_RETURN_IF_ERROR(ForwardArguments(fwd_args, args));

  auto executed =
      function_ref(args, runtime::NoResultConverter{}, opts, InDebugMode());
  concurrent_region_status->EnableConcurrentRegion();
  if (!executed.ok()) {
    return InternalError("RunGraphOpByOp failed (%s): %s",
                         diagnostic.empty() ? "<no details>" : diagnostic,
                         executed.status().ToString());
  }
  return absl::OkStatus();
}

#endif  // #if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

//===----------------------------------------------------------------------===//
// Define the gpu graph launch custom call.
//===----------------------------------------------------------------------===//

static absl::Status LaunchGraph(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, const std::string* ptx,
    const std::vector<uint8_t>* cubin, se::DeviceMemoryBase* temp_buffer,
    StreamExecutorKernels::Snapshot* kernels,
    StreamExecutorConvRunners::Snapshot* convs,
    StreamExecutorGraphInstances::Snapshot* instances,
    CapturedFunctionExecutionCount::Snapshot* counts,
    GemmConfigs::Snapshot* gemm_config, runtime::Executable* executable,
    NonAtomicallyUpgradeableRWLock* gpu_lock,
    ConcurrentRegionStatus* region_status, CustomCall::RemainingArgs fwd_args,
    CustomCall::FunctionOrdinal capture) {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  VLOG(1) << "Launch GPU Graph: ordinal = " << capture.ordinal;

  // Get a reference to exported function that captures the gpu graph.
  runtime::FunctionRef function_ref = executable->function_ref(capture.ordinal);

  // Compute the hash of the buffer arguments.
  size_t ptrs_hash = absl::HashOf(RemainingArgsPtrs{fwd_args, temp_buffer});

  // Forwards user data required for launching kernels.
  auto user_data = [&] {
    return CustomCall::UserData(run_options, debug_options, ptx, cubin,
                                temp_buffer, kernels, convs, executable,
                                gemm_config, gpu_lock, region_status);
  };

  TF_ASSIGN_OR_RETURN(std::unique_ptr<std::atomic<uint64_t>> * get_count,
                      counts->GetOrCreate(capture.ordinal, [] {
                        return std::make_unique<std::atomic<uint64_t>>(0);
                      }));

  int64_t count = (*get_count)->fetch_add(1);
  int64_t num_runs_to_instantiate =
      debug_options->xla_gpu_graph_num_runs_to_instantiate();

  // TODO(b/290773547): Profiler + CUDA graphs lead to memory corruption. As a
  // work around disable graph execution and run everything in op-by-op mode.
  bool is_profiling = tsl::profiler::ProfilerLock::HasActiveSession();

  if (count < num_runs_to_instantiate || is_profiling) {
    VLOG(3) << "Run gpu graph in op-by-op mode: ordinal = " << capture.ordinal;
    return RunGraphOpByOp(run_options, function_ref, fwd_args, user_data());
  }

  // Instantiate Gpu graph by running graph capture function.
  auto instantiate = [&]() -> absl::StatusOr<GraphInstance> {
    Arguments<MemrefDesc> args(fwd_args.size());
    TF_RETURN_IF_ERROR(ForwardArguments(fwd_args, args));

    TF_ASSIGN_OR_RETURN(
        auto g, CaptureGraph(run_options, function_ref, args, user_data()));

    TF_ASSIGN_OR_RETURN(auto e, se::gpu::InstantiateGpuGraph(std::move(g)));

    return GraphInstance(ptrs_hash, std::move(e));
  };

  GraphInstance* instance;
  if (num_runs_to_instantiate < 0) {
    // If num_runs_to_instantiate is less than 0, all graphs should be
    // instantiated ahead-of-time. If we fail to get the graph instance, then
    // graph instantiation failed due to OOM. So we run the graph op-by-op.
    absl::StatusOr<GraphInstance*> try_get_instance =
        instances->Get(capture.ordinal);
    if (try_get_instance.ok()) {
      instance = try_get_instance.value();
    } else {
      return RunGraphOpByOp(run_options, function_ref, fwd_args, user_data());
    }
  } else {
    TF_ASSIGN_OR_RETURN(instance,
                        instances->GetOrCreate(capture.ordinal, instantiate));
  }

  {
    // Lock graph instance for read only access. If we'll have to update the
    // graph, we'll update to a writer lock below.
    absl::ReaderMutexLock lock(instance->mutex.get());

    // If pointers did not change we can run captured graph.
    if (ptrs_hash == instance->ptr_hash) {
      TraceMe trace([&] {
        return TraceMeEncode("gpu.graph.launch_cached",
                             {{"ordinal", capture.ordinal}});
      });

      VLOG(3) << "Execute cached graph instance";
      return instance->exec.Launch(run_options->stream());
    }
  }

  // Otherwise we have to re-capture the graph and update the graph instance.
  VLOG(3) << "Update cached graph instance";

  Arguments<MemrefDesc> args(fwd_args.size());
  TF_RETURN_IF_ERROR(ForwardArguments(fwd_args, args));

  // Capture GPU graph by running capture function.
  TF_ASSIGN_OR_RETURN(
      auto g, CaptureGraph(run_options, function_ref, args, user_data()));

  // At this point we have to grab a writer lock, because we might potentially
  // have concurrent execution of the cached graph instance.
  absl::WriterMutexLock lock(instance->mutex.get());

  // Update captured graph executable.
  TF_RETURN_IF_ERROR(instance->exec.Update(std::move(g)));

  // Update captured pointer hash.
  instance->ptr_hash = ptrs_hash;

  TraceMe trace([&] {
    return TraceMeEncode("gpu.graph.launch_updated",
                         {{"ordinal", capture.ordinal}});
  });

  return instance->exec.Launch(run_options->stream());

#else  // #if !GOOGLE_CUDA && !TENSORFLOW_USE_ROCM

  return absl::InternalError("GPU graphs are not supported");

#endif  // #if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

//===----------------------------------------------------------------------===//

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    Launch, FunctionWrapper<LaunchGraph>(), checks,
    CustomCall::Bind("xla.gpu.graph.launch")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<const DebugOptions*>()
        .UserData<const std::string*>()
        .UserData<const std::vector<uint8_t>*>()
        .UserData<se::DeviceMemoryBase*>()
        .UserData<StreamExecutorKernels::Snapshot*>()
        .UserData<StreamExecutorConvRunners::Snapshot*>()
        .UserData<StreamExecutorGraphInstances::Snapshot*>()
        .UserData<CapturedFunctionExecutionCount::Snapshot*>()
        .UserData<GemmConfigs::Snapshot*>()
        .UserData<Executable*>()
        .UserData<NonAtomicallyUpgradeableRWLock*>()
        .UserData<ConcurrentRegionStatus*>()
        .RemainingArgs()
        .Attr<CustomCall::FunctionOrdinal>("capture"));

void RegisterGraphLaunchCustomCalls(
    runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.graph.launch", Launch);
}

}  // namespace gpu
}  // namespace xla
