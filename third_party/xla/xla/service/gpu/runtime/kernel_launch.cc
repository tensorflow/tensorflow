/* Copyright 2022 The OpenXLA Authors.

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

#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/runtime/custom_call.h"
#include "xla/runtime/custom_call_registry.h"
#include "xla/runtime/executable.h"
#include "xla/runtime/memref_view.h"
#include "xla/runtime/state.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/runtime/concurrent_region.h"
#include "xla/service/gpu/runtime/support.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

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
    State<std::unique_ptr<se::Kernel>> device_kernel,
    int32_t shared_memory_bytes, int32_t grid_size_x, int32_t grid_size_y,
    int32_t grid_size_z, int32_t block_size_x, int32_t block_size_y,
    int32_t block_size_z, CustomCall::RemainingArgs args, std::string_view name,
    int64_t stream_id) {
  se::Stream* stream = run_options->stream();
  se::StreamExecutor* executor = stream->parent();

  LaunchDimensions launch_dimensions(
      se::BlockDim(grid_size_x, grid_size_y, grid_size_z),
      se::ThreadDim(block_size_x, block_size_y, block_size_z));

  const int args_size_including_temp_buffer = args.size() + 1;

  // If kernel does not exist create it from the ptx and cubin.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::Kernel> * kernel, device_kernel.GetOrCreate([&] {
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
// Define the custom kernel (fusion) launch custom call.
//===----------------------------------------------------------------------===//

static absl::StatusOr<std::unique_ptr<se::Kernel>> CreateCustomKernel(
    se::StreamExecutor* executor, std::string_view name,
    std::string_view custom_fusion_computation) {
  auto* registry = CustomKernelFusionRegistry::Default();
  auto* custom_kernel_fusion = registry->Lookup(name);

  // If custom fusion is not found it means that some of the build targets might
  // not be statically linked into the binary.
  if (custom_kernel_fusion == nullptr) {
    return absl::InternalError(absl::StrCat(
        "Custom kernel fusion ", name, " not found in a default registry."));
  }

  // Parse attached custom fusion computation.
  HloComputationProto computation_proto;
  if (!computation_proto.ParseFromArray(custom_fusion_computation.data(),
                                        custom_fusion_computation.size())) {
    return absl::InternalError("Failed to parse custom fusion computation");
  }

  // Build HloComputation from a proto for passing to custom fusion.
  absl::flat_hash_map<int64_t, HloComputation*> computation_map;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloComputation> computation,
      HloComputation::CreateFromProto(computation_proto, computation_map));

  // Load custom kernels that can implement a fusion computation.
  TF_ASSIGN_OR_RETURN(std::vector<CustomKernel> kernels,
                      custom_kernel_fusion->LoadKernels(
                          executor->GetDeviceDescription(), computation.get()));

  // This should never happen, it means that compilation pipeline created a
  // fusion operation that is not supported by a given custom fusion.
  if (kernels.empty()) {
    return absl::InternalError(
        absl::StrCat("Custom kernel fusion ", name,
                     " returned empty custom kernels for a fused computation"));
  }

  auto kernel = std::make_unique<se::Kernel>(executor);
  TF_RETURN_IF_ERROR(
      executor->GetKernel(kernels[0].kernel_spec(), kernel.get()));

  return kernel;
}

static absl::Status CustomLaunchImpl(
    const ServiceExecutableRunOptions* run_options, const std::string* ptx,
    const std::vector<uint8_t>* cubin, se::DeviceMemoryBase* temp_buffer,
    ConcurrentRegionStatus* region_status,
    State<std::unique_ptr<se::Kernel>> device_kernel,
    int32_t shared_memory_bytes, int32_t grid_size_x, int32_t grid_size_y,
    int32_t grid_size_z, int32_t block_size_x, int32_t block_size_y,
    int32_t block_size_z, CustomCall::RemainingArgs args, std::string_view name,
    int64_t stream_id, std::string_view custom_fusion_computation) {
  se::Stream* stream = run_options->stream();
  se::StreamExecutor* executor = stream->parent();

  LaunchDimensions launch_dimensions(
      se::BlockDim(grid_size_x, grid_size_y, grid_size_z),
      se::ThreadDim(block_size_x, block_size_y, block_size_z));

  // If kernel does not exist load it from a custom fusion computation.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::Kernel> * kernel, device_kernel.GetOrCreate([&] {
        return ToAbsl(
            CreateCustomKernel(executor, name, custom_fusion_computation));
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

  absl::InlinedVector<se::DeviceMemoryBase, 8> buffer_args(args.size());

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

  // If we are capturing a concurrent region in a GPU graph, then use the
  // stream provided by ConcurrentRegionStatus to execute the kernel.
  se::Stream* execution_stream = stream;
  if (stream_id != 0) {
    DCHECK(region_status->IsInConcurrentRegion());
    TF_ASSIGN_OR_RETURN(execution_stream, region_status->GetStream(stream_id));
  } else if (region_status->IsInConcurrentRegion()) {
    execution_stream = region_status->GetNextStream();
  }

  se::KernelArgsDeviceMemoryArray kernel_args(buffer_args, shared_memory_bytes);
  return executor->Launch(
      stream, se::ThreadDim(block_size_x, block_size_y, block_size_z),
      se::BlockDim(grid_size_x, grid_size_y, grid_size_z), **kernel,
      kernel_args);
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
        .State<std::unique_ptr<se::Kernel>>("uid")
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

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    CustomLaunch, FunctionWrapper<CustomLaunchImpl>(), checks,
    CustomCall::Bind("xla.gpu.func.custom_launch")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<const std::string*>()
        .UserData<const std::vector<uint8_t>*>()
        .UserData<se::DeviceMemoryBase*>()
        .UserData<ConcurrentRegionStatus*>()
        .State<std::unique_ptr<se::Kernel>>("uid")
        .Arg<int32_t>()   // shared_memory_bytes
        .Arg<int32_t>()   // grid_size_x
        .Arg<int32_t>()   // grid_size_y
        .Arg<int32_t>()   // grid_size_z
        .Arg<int32_t>()   // block_size_x
        .Arg<int32_t>()   // block_size_y
        .Arg<int32_t>()   // block_size_x
        .RemainingArgs()  // args
        .Attr<std::string_view>("kernel")
        .Attr<int64_t>("stream")
        .Attr<std::string_view>("__custom_fusion_computation"));

void RegisterKernelLaunchCustomCalls(
    runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.func.launch", Launch);
  registry.Register("xla.gpu.func.custom_launch", CustomLaunch);
}

}  // namespace gpu
}  // namespace xla
