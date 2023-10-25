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

#include "xla/service/gpu/runtime/kernel_launch.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "xla/runtime/custom_call.h"
#include "xla/runtime/executable.h"
#include "xla/runtime/state.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/runtime/concurrent_region.h"
#include "xla/service/gpu/runtime/support.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/kernel.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "xla/stream_executor/gpu/gpu_graph.h"
#endif  // #if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace xla {
namespace gpu {

using xla::runtime::CustomCall;
using xla::runtime::State;
using xla::runtime::StridedMemrefView;

StreamExecutorKernels* GpuExecutableKernels::operator()(
    se::StreamExecutor* executor) {
  absl::MutexLock lock(&mutex_);
  return &kernels_[executor];
}

//===----------------------------------------------------------------------===//
// Define the kernel launch custom call.
//===----------------------------------------------------------------------===//

static absl::Status LaunchImpl(
    const ServiceExecutableRunOptions* run_options, const std::string* ptx,
    const std::vector<uint8_t>* cubin, se::DeviceMemoryBase* temp_buffer,
    ConcurrentRegionStatus* region_status,
    State<std::unique_ptr<se::KernelBase>> device_kernel,
    int32_t shared_memory_bytes, int32_t grid_size_x, int32_t grid_size_y,
    int32_t grid_size_z, int32_t block_size_x, int32_t block_size_y,
    int32_t block_size_z, CustomCall::RemainingArgs args, std::string_view name,
    int64_t stream_id) {
  se::Stream* stream = run_options->stream();
  se::StreamExecutor* executor = stream->parent();

  LaunchDimensions launch_dimensions(
      {grid_size_x, grid_size_y, grid_size_z},
      {block_size_x, block_size_y, block_size_z});

  const int args_size_including_temp_buffer = args.size() + 1;

  // If kernel does not exist create it from the ptx and cubin.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::KernelBase> * kernel, device_kernel.GetOrCreate([&] {
        return ToAbsl(CreateKernel(absl::string_view(name.data(), name.size()),
                                   args_size_including_temp_buffer, *ptx,
                                   *cubin, executor, shared_memory_bytes));
      }));
  assert((*kernel)->name() == name && "unexpected loaded kernel");

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  if (VLOG_IS_ON(3)) {
    TF_ASSIGN_OR_RETURN(bool is_capturing, se::gpu::IsStreamCapturing(stream));
    if (is_capturing) {
      if (region_status->IsInConcurrentRegion()) {
        LOG(INFO) << "Launching " << (*kernel)->name()
                  << "in a concurrent region during GPU graph capture";
      } else {
        LOG(INFO) << "Launching " << (*kernel)->name()
                  << "during GPU graph capture";
      }
    } else {
      LOG(INFO) << "Launching " << (*kernel)->name();
    }
  }
#else
  VLOG(3) << "Launching " << (*kernel)->name();
#endif

  absl::InlinedVector<se::DeviceMemoryBase, 8> buffer_args(
      args_size_including_temp_buffer);

  // Add MemRef arguments as buffer arguments.
  for (unsigned i = 0; i < args.size(); ++i) {
    // We get arguments corresponding to XLA allocations required by the
    // compiled device kernel, and not the actual memrefs that device kernel
    // writes/reads, so we don't have to pass the size along with the pointer.
    if (auto strided = args.get<StridedMemrefView>(i); succeeded(strided)) {
      buffer_args[i] = se::DeviceMemoryBase(strided->data);
      continue;
    }

    return absl::InvalidArgumentError(
        absl::StrFormat("Unsupported argument #%d type", i));
  }

  // Always add temporary buffer as the last kernel argument.
  buffer_args.back() = *temp_buffer;

  // If we are capturing a concurrent region in a GPU graph, then use the
  // stream provided by ConcurrentRegionStatus to execute the kernel.
  se::Stream* execution_stream = stream;
  if (stream_id != 0) {
    DCHECK(region_status->IsInConcurrentRegion());
    TF_ASSIGN_OR_RETURN(execution_stream, region_status->GetStream(stream_id));
  } else if (region_status->IsInConcurrentRegion()) {
    execution_stream = region_status->GetNextStream();
  }

  // Execute device kernel on the execution stream.
  return ExecuteKernelOnStream(**kernel, buffer_args, launch_dimensions,
                               execution_stream);
}

//===----------------------------------------------------------------------===//

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    Launch, FunctionWrapper<LaunchImpl>(), checks,
    CustomCall::Bind("xla.gpu.func.launch")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<const std::string*>()
        .UserData<const std::vector<uint8_t>*>()
        .UserData<se::DeviceMemoryBase*>()
        .UserData<ConcurrentRegionStatus*>()
        .State<std::unique_ptr<se::KernelBase>>("uid")
        .Arg<int32_t>()   // shared_memory_bytes
        .Arg<int32_t>()   // grid_size_x
        .Arg<int32_t>()   // grid_size_y
        .Arg<int32_t>()   // grid_size_z
        .Arg<int32_t>()   // block_size_x
        .Arg<int32_t>()   // block_size_y
        .Arg<int32_t>()   // block_size_x
        .RemainingArgs()  // args
        .Attr<std::string_view>("kernel")
        .Attr<int64_t>("stream"));

void RegisterKernelLaunchCustomCalls(
    runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.func.launch", Launch);
}

}  // namespace gpu
}  // namespace xla
