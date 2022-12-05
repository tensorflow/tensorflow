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

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/kernel_launch.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#endif  // #if GOOGLE_CUDA

namespace xla {
namespace gpu {

using absl::InternalError;
using absl::OkStatus;
using absl::StrFormat;

using xla::runtime::CustomCall;
using xla::runtime::Executable;
using xla::runtime::StridedMemrefView;

#if GOOGLE_CUDA
using xla::runtime::Arguments;
using xla::runtime::AsyncTaskRunner;
using xla::runtime::MemrefDesc;
using xla::runtime::ScalarArg;
#endif  // #if GOOGLE_CUDA

//===----------------------------------------------------------------------===//
// RAII helpers for CUDA graph types.
//===----------------------------------------------------------------------===//

#if GOOGLE_CUDA

using OwnedGraph = GraphInstances::OwnedGraph;
using OwnedGraphExec = GraphInstances::OwnedGraphExec;

void GraphInstances::DestroyGraph::operator()(cudaGraph_t graph) {
  cudaError_t err = cudaGraphDestroy(graph);
  CHECK(err == cudaSuccess)
      << "Failed to destroy CUDA graph: " << cudaGetErrorString(err);
}

void GraphInstances::DestroyGraphExec::operator()(cudaGraphExec_t instance) {
  cudaError_t err = cudaGraphExecDestroy(instance);
  CHECK(err == cudaSuccess)
      << "Failed to destroy CUDA graph instance: " << cudaGetErrorString(err);
}

#endif  // #if GOOGLE_CUDA

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

static absl::StatusOr<OwnedGraph> CaptureGraph(
    const ServiceExecutableRunOptions* run_options,
    runtime::FunctionRef function_ref, CustomCall::RemainingArgs fwd_args,
    CustomCall::UserData user_data) {
  // We capture graph on a borrowed stream because we do not want to
  // accidentally record any concurrent kernel launches from other XLA
  // executables.
  se::StreamExecutor* executor = run_options->stream()->parent();
  StatusOr<StreamPool::Ptr> capture_stream =
      run_options->BorrowStream(executor->device_ordinal());

  if (!capture_stream.ok())
    return InternalError(
        StrFormat("Failed to borrow a stream for graph capture: %s",
                  capture_stream.status().error_message()));

  // TODO(ezhulenev): Pass graph capture context explicitly to the custom calls
  // via UserData to be able to detect when executing custom call in graph
  // capture mode. Currently we rely on the fact that we know for sure that
  // operations in the graph capture function do not need anything except the
  // main stream (we capture only kernel launches).
  ExecutableRunOptions capture_run_options;
  capture_run_options.set_stream(capture_stream->get());

  const ServiceExecutableRunOptions capture_opts(capture_run_options);
  user_data.insert(&capture_opts);

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

  // Get the underlying CUDA stream for passing to CUDA APIs.
  auto stream = se::gpu::AsGpuStreamValue(capture_stream->get());

  // We know for sure that graph capture function is single-threaded, and we do
  // not want to accidentally record some unrelated command, so we always record
  // graphs in thread local mode.
  auto mode = cudaStreamCaptureModeThreadLocal;

  cudaGraph_t graph;

  // Capture graph constructed by the exported graph capture function.
  if (auto err = cudaStreamBeginCapture(stream, mode); err != cudaSuccess)
    return InternalError(
        StrFormat("Stream begin capture failed: %s", cudaGetErrorString(err)));

  // Call into graph capture function.
  auto captured = function_ref(args, runtime::NoResultConverter{}, opts);

  // Always stop capturing the stream before checking `captured` result.
  if (auto err = cudaStreamEndCapture(stream, &graph); err != cudaSuccess)
    return InternalError(
        StrFormat("Stream end capture failed: %s", cudaGetErrorString(err)));

  if (!captured.ok())
    return InternalError(StrFormat("Failed to capture CUDA graph: %s; %s",
                                   captured.message(), error));

  return OwnedGraph(graph);
}

#endif  // #if GOOGLE_CUDA

//===----------------------------------------------------------------------===//
// Define the cuda graph launch custom call.
//===----------------------------------------------------------------------===//

static absl::Status LaunchGraph(
    const ServiceExecutableRunOptions* run_options, const std::string* ptx,
    const std::vector<uint8_t>* cubin, se::DeviceMemoryBase* temp_buffer,
    StreamExecutorKernels::Snapshot* kernels,
    GraphInstances::Snapshot* instances, runtime::Executable* executable,
    CustomCall::RemainingArgs fwd_args, CustomCall::FunctionOrdinal capture) {
#if GOOGLE_CUDA
  VLOG(1) << "Launch Cuda Graph: capture=" << capture.ordinal;

  // Get a reference to exported function that captures the cuda graph.
  runtime::FunctionRef function_ref = executable->function_ref(capture.ordinal);

  // Compute the hash of the buffer arguments.
  size_t ptrs_hash = absl::HashOf(RemainingArgsPtrs{fwd_args, temp_buffer});

  // Forwards user data required for launching kernels.
  auto user_data = [&] {
    return CustomCall::UserData(run_options, ptx, cubin, temp_buffer, kernels,
                                executable);
  };

  absl::StatusOr<GraphInstance*> instance = instances->GetOrCreate(
      capture.ordinal, [&]() -> absl::StatusOr<GraphInstance> {
        // Get a graph defined by the graph capture function.
        auto g = CaptureGraph(run_options, function_ref, fwd_args, user_data());
        if (!g.ok()) return g.status();

        // Instantiate captured CUDA graph into an executable instance.
        cudaGraphExec_t exec;
        if (auto err = cudaGraphInstantiate(&exec, &**g, nullptr, nullptr, 0);
            err != cudaSuccess) {
          return InternalError(StrFormat("Graph instantiation failed: %s",
                                         cudaGetErrorString(err)));
        }

        return GraphInstance(ptrs_hash, exec);
      });

  if (!instance.ok()) return instance.status();

  // Get the underlying cuda stream.
  auto stream = se::gpu::AsGpuStreamValue(run_options->stream());

  // Lock graph instance mutex for exclusive access, because we potentially
  // might have to update it with a new graph version.
  absl::MutexLock lock((*instance)->mutex.get());

  // If pointers did not change we can run captured graph.
  if (ptrs_hash == (*instance)->ptr_hash) {
    VLOG(3) << "Execute cached graph instance";
    return (cudaGraphLaunch((*instance)->exec.get(), stream) == cudaSuccess)
               ? OkStatus()
               : InternalError("Failed to run captured graph");
  }

  // Otherwise we have to re-capture the graph and update the graph
  // instance.
  VLOG(3) << "Update cached graph instance";

  // Capture CUDA graph by running capture function.
  auto g = CaptureGraph(run_options, function_ref, fwd_args, user_data());
  if (!g.ok()) return g.status();

  cudaGraphExecUpdateResult update_result;
  cudaGraphNode_t error_node;

  auto err = cudaGraphExecUpdate((*instance)->exec.get(), g->get(), &error_node,
                                 &update_result);
  if (err != cudaSuccess || update_result != cudaGraphExecUpdateSuccess)
    return InternalError("Failed to update cuda graph");

  // Update captured graph pointers hash.
  (*instance)->ptr_hash = ptrs_hash;

  return (cudaGraphLaunch((*instance)->exec.get(), stream) == cudaSuccess)
             ? OkStatus()
             : InternalError("Failed to run captured graph");

#else  // #if !GOOGLE_CUDA

  return InternalError("Cuda graphs are not supported");

#endif  // #if GOOGLE_CUDA
}

//===----------------------------------------------------------------------===//

static bool Launch(runtime::ExecutionContext* ctx, void** args, void** attrs,
                   void** rets) {
  static auto* handler = CustomCall::Bind("xla.gpu.cuda.graph.launch")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .UserData<const std::string*>()
                             .UserData<const std::vector<uint8_t>*>()
                             .UserData<se::DeviceMemoryBase*>()
                             .UserData<StreamExecutorKernels::Snapshot*>()
                             .UserData<GraphInstances::Snapshot*>()
                             .UserData<Executable*>()
                             .RemainingArgs()
                             .Attr<CustomCall::FunctionOrdinal>("capture")
                             .To<checks>(LaunchGraph)
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

void RegisterGraphLaunchCustomCalls(
    runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.cuda.graph.launch", Launch);
}

}  // namespace gpu
}  // namespace xla
