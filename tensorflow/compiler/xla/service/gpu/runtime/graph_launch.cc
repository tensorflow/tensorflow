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

#include "tensorflow/compiler/xla/service/gpu/runtime/graph_launch.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/gpu/non_atomically_upgradeable_rw_lock.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/conv.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/gemm.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/kernel_launch.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"

#if GOOGLE_CUDA
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_graph.h"
#endif  // #if GOOGLE_CUDA

namespace xla {
namespace gpu {

using xla::runtime::Arguments;
using xla::runtime::AsyncTaskRunner;
using xla::runtime::CustomCall;
using xla::runtime::Executable;
using xla::runtime::MemrefDesc;
using xla::runtime::ScalarArg;
using xla::runtime::StridedMemrefView;

//===----------------------------------------------------------------------===//
// CUDA graphs caching.
//===----------------------------------------------------------------------===//

StreamExecutorGraphInstances* GraphInstances::operator()(
    se::StreamExecutor* executor) {
  absl::MutexLock lock(&mutex_);
  return &graphs_[executor];
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
// Runs capture function exported by the executable to constuct a CUDA graph.
//----------------------------------------------------------------------------//

#if GOOGLE_CUDA

using se::gpu::OwnedCudaGraph;

static bool InDebugMode() {
#ifdef NDEBUG
  return false;
#endif
  return true;
}

static absl::StatusOr<OwnedCudaGraph> CaptureGraph(
    const ServiceExecutableRunOptions* run_options,
    runtime::FunctionRef function_ref, CustomCall::RemainingArgs fwd_args,
    CustomCall::UserData user_data) {
  // Initialize (with memoization) BlasSupport here because cublasCreate fails
  // during cuda graph capturing.
  // TODO(b/272559361): The initialization should be conditional.
  if (!run_options->stream()->parent()->AsBlas()) {
    return absl::InternalError("Failed to initialize BLAS support");
  }

  std::string error;
  runtime::DiagnosticEngine diagnostic_engine;
  diagnostic_engine.AddHandler([&](runtime::Diagnostic& diagnostic) {
    error.append(diagnostic.status().message());
    return runtime::success();
  });

  // Prepare options for executing graph capture function.
  Executable::ExecuteOpts opts;
  opts.custom_call_data = &user_data;
  opts.diagnostic_engine = &diagnostic_engine;

  // Graph capture function should not launch any async tasks.
  opts.async_task_runner = reinterpret_cast<AsyncTaskRunner*>(0XDEADBEEF);

  // Graph capture functions can only have index arguments for launch
  // dimensions, or memrefs for passing buffers. We need to re-package custom
  // call arguments into a container that can be passed to an executable
  // function.
  Arguments<ScalarArg, MemrefDesc> args(fwd_args.size());

  for (size_t i = 0; i < fwd_args.size(); ++i) {
    // `index` argument passed as int64_t.
    if (auto idx = fwd_args.get<int64_t>(i); succeeded(idx)) {
      args.emplace_back<ScalarArg>(*idx);
      continue;
    }

    // Pass `memref` argument as a MemrefDesc.
    if (auto memref = fwd_args.get<StridedMemrefView>(i); succeeded(memref)) {
      args.emplace_back<MemrefDesc>(memref->dtype, memref->data, /*offset=*/0,
                                    memref->sizes, memref->strides);
      continue;
    }

    return absl::InvalidArgumentError("Unsupported argument type");
  }

  // Create a graph from running the graph capture function.
  auto captured = se::gpu::CaptureCudaGraph(run_options->stream(), [&]() {
    return FromAbslStatus(function_ref(args, runtime::NoResultConverter{}, opts,
                                       /*verify_arguments=*/InDebugMode())
                              .status());
  });

  if (!captured.ok()) return ToAbslStatus(captured.status());
  return std::move(*captured);
}

static absl::Status RunGraphWithoutCapture(
    const ServiceExecutableRunOptions* run_options,
    runtime::FunctionRef function_ref, CustomCall::RemainingArgs fwd_args,
    CustomCall::UserData user_data) {
  // Prepare options for executing graph capture function.
  Executable::ExecuteOpts opts;
  opts.custom_call_data = &user_data;

  std::string error;
  runtime::DiagnosticEngine diagnostic_engine;
  diagnostic_engine.AddHandler([&](runtime::Diagnostic& diagnostic) {
    error.append(diagnostic.status().message());
    return runtime::success();
  });
  opts.diagnostic_engine = &diagnostic_engine;

  // Graph capture function should not launch any async tasks.
  opts.async_task_runner = reinterpret_cast<AsyncTaskRunner*>(0XDEADBEEF);

  Arguments<ScalarArg, MemrefDesc> args(fwd_args.size());

  for (size_t i = 0; i < fwd_args.size(); ++i) {
    // `index` argument passed as int64_t.
    if (auto idx = fwd_args.get<int64_t>(i); succeeded(idx)) {
      args.emplace_back<ScalarArg>(*idx);
      continue;
    }

    // Pass `memref` argument as a MemrefDesc.
    if (auto memref = fwd_args.get<StridedMemrefView>(i); succeeded(memref)) {
      args.emplace_back<MemrefDesc>(memref->dtype, memref->data, /*offset=*/0,
                                    memref->sizes, memref->strides);
      continue;
    }

    return absl::InvalidArgumentError("Unsupported argument type");
  }

  return function_ref(args, runtime::NoResultConverter{}, opts, InDebugMode())
      .status();
}

#endif  // #if GOOGLE_CUDA

//===----------------------------------------------------------------------===//
// Define the cuda graph launch custom call.
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
    CustomCall::RemainingArgs fwd_args, CustomCall::FunctionOrdinal capture) {
#if GOOGLE_CUDA
  VLOG(1) << "Launch Cuda Graph: capture=" << capture.ordinal;

  // Get a reference to exported function that captures the cuda graph.
  runtime::FunctionRef function_ref = executable->function_ref(capture.ordinal);

  // Compute the hash of the buffer arguments.
  size_t ptrs_hash = absl::HashOf(RemainingArgsPtrs{fwd_args, temp_buffer});

  CapturingCudaGraph not_capturing(false);
  CapturingCudaGraph capturing(true);
  // Forwards user data required for launching kernels.
  auto user_data_no_capture = [&] {
    return CustomCall::UserData(run_options, debug_options, ptx, cubin,
                                temp_buffer, kernels, convs, executable,
                                gemm_config, gpu_lock, &not_capturing);
  };
  auto user_data_capture = [&] {
    return CustomCall::UserData(run_options, debug_options, ptx, cubin,
                                temp_buffer, kernels, convs, executable,
                                gemm_config, gpu_lock, &capturing);
  };

  absl::StatusOr<std::unique_ptr<std::atomic<uint64_t>>*> get_count =
      counts->GetOrCreate(
          capture.ordinal,
          []() -> absl::StatusOr<std::unique_ptr<std::atomic<uint64_t>>> {
            return std::make_unique<std::atomic<uint64_t>>(0);
          });
  if (!get_count.ok()) return get_count.status();
  uint64_t count = (**get_count)->fetch_add(1);
  uint64_t instantiation_threshold =
      debug_options->xla_gpu_cuda_graph_instantiation_threshold();
  if (count < instantiation_threshold) {
    // Run captured graph directly.
    absl::Status result = RunGraphWithoutCapture(
        run_options, function_ref, fwd_args, user_data_no_capture());
    if (!result.ok()) return result;
    return absl::OkStatus();
  }

  absl::StatusOr<GraphInstance*> instance = instances->GetOrCreate(
      capture.ordinal, [&]() -> absl::StatusOr<GraphInstance> {
        auto g = CaptureGraph(run_options, function_ref, fwd_args,
                              user_data_capture());
        if (!g.ok()) return g.status();

        auto e = se::gpu::InstantiateCudaGraph(std::move(*g));
        if (!e.ok()) return ToAbslStatus(e.status());

        return GraphInstance(ptrs_hash, std::move(*e));
      });
  if (!instance.ok()) return instance.status();

  // Lock graph instance mutex for exclusive access, because we potentially
  // might have to update it with a new graph version.
  absl::MutexLock lock((*instance)->mutex.get());

  // If pointers did not change we can run captured graph.
  if (ptrs_hash == (*instance)->ptr_hash) {
    VLOG(3) << "Execute cached graph instance";
    return ToAbslStatus((*instance)->exec.Launch(run_options->stream()));
  }

  // Otherwise we have to re-capture the graph and update the graph instance.
  VLOG(3) << "Update cached graph instance";

  // Capture CUDA graph by running capture function.
  auto g =
      CaptureGraph(run_options, function_ref, fwd_args, user_data_capture());
  if (!g.ok()) return g.status();

  // Update captured graph executable.
  auto updated = (*instance)->exec.Update(std::move(*g));
  if (!updated.ok()) return ToAbslStatus(updated);

  // Update captured graph pointers hash.
  (*instance)->ptr_hash = ptrs_hash;

  return ToAbslStatus((*instance)->exec.Launch(run_options->stream()));

#else  // #if !GOOGLE_CUDA

  return absl::InternalError("Cuda graphs are not supported");

#endif  // #if GOOGLE_CUDA
}

//===----------------------------------------------------------------------===//

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    Launch, FunctionWrapper<LaunchGraph>(), checks,
    CustomCall::Bind("xla.gpu.cuda.graph.launch")
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
        .RemainingArgs()
        .Attr<CustomCall::FunctionOrdinal>("capture"));

void RegisterGraphLaunchCustomCalls(
    runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.cuda.graph.launch", Launch);
}

}  // namespace gpu
}  // namespace xla
