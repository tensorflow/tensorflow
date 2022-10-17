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

#include <string>
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

using xla::runtime::CustomCall;
using xla::runtime::Executable;

#if GOOGLE_CUDA
using xla::runtime::Arguments;
using xla::runtime::AsyncTaskRunner;
using xla::runtime::MemrefDesc;
using xla::runtime::ScalarArg;
using xla::runtime::StridedMemrefView;
#endif  // #if GOOGLE_CUDA

//===----------------------------------------------------------------------===//
// Define the cuda graph launch custom call.
//===----------------------------------------------------------------------===//

static absl::Status LaunchGraph(
    const ServiceExecutableRunOptions* run_options, const std::string* ptx,
    const std::vector<uint8_t>* cubin, se::DeviceMemoryBase* temp_buffer,
    GpuExecutableKernelsCache* kernels_cache, runtime::Executable* executable,
    CustomCall::RemainingArgs fwd_args, CustomCall::FunctionOrdinal capture) {
#if GOOGLE_CUDA
  // Get a reference to exported function that captures the cuda graph.
  runtime::FunctionRef function_ref = executable->function_ref(capture.ordinal);

  VLOG(1) << "Launch Cuda Graph: capture=" << capture.ordinal;

  // Forward user data required for launching kernels.
  CustomCall::UserData user_data;
  user_data.insert_all(run_options, ptx, cubin, temp_buffer, kernels_cache,
                       executable);

  // Graph capture function should not launch any async tasks.
  Executable::ExecuteOpts opts;
  opts.custom_call_data = &user_data;
  opts.async_task_runner = reinterpret_cast<AsyncTaskRunner*>(0XDEADBEEF);

  // Graph capture functions can only have index arguments for launch
  // dimensions, or memrefs for passing buffers. We need to re-package custom
  // call arguments into a container that can be passed to an executable
  // function.
  Arguments<ScalarArg, MemrefDesc> args(fwd_args.size());

  for (size_t i = 0; i < fwd_args.size(); ++i) {
    // `index` argument passed as intptr_t.
    if (auto idx = fwd_args.get<intptr_t>(i); succeeded(idx)) {
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

  // TODO(ezhulenev): This function instantiates cuda graphs on every call,
  // which is absolutely not how it should be done. Graphs have to be cached,
  // and updated whenever we receive new arguments. This is a proof of concept
  // demonstration of integrating Cuda Graphs with Xla runtime and trivial
  // compiler pass that outlines sequences of device function launches.

  // Construct cuda graph from the exported function.
  cudaGraph_t graph;
  cudaGraphExec_t instance;

  // Get the underlying cuda stream.
  auto stream = se::gpu::AsGpuStreamValue(run_options->stream());

  cudaError_t err;

  // Capture graph constructed by the exported graph capture function.
  err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  if (err != cudaSuccess)
    return absl::InternalError("Stream begin capture failed");

  // Call into graph capture function.
  auto captured = function_ref(args, runtime::NoResultConverter{}, opts);
  if (!captured.ok()) return captured;

  err = cudaStreamEndCapture(stream, &graph);
  if (err != cudaSuccess)
    return absl::InternalError("Stream end capture failed");

  err = cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
  if (err != cudaSuccess)
    return absl::InternalError("Graph instantiation failed");

  // Run captured graph.
  cudaGraphLaunch(instance, stream);
  if (err != cudaSuccess)
    return absl::InternalError("Failed to run captured graph");

  // Destroy captured graph.
  err = cudaGraphExecDestroy(instance);
  if (err != cudaSuccess) return absl::InternalError("Instance destroy failed");
  err = cudaGraphDestroy(graph);
  if (err != cudaSuccess) return absl::InternalError("Graph destroy failed");

  return absl::OkStatus();
#else
  return absl::InternalError("Cuda graphs are not supported");
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
                             .UserData<GpuExecutableKernelsCache*>()
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
