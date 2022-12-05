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

#include "tensorflow/compiler/xla/service/gpu/runtime/kernel_launch.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/runtime/state.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/stream_executor/kernel.h"

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

namespace {
struct KernelLaunch {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  absl::Status operator()(
      const ServiceExecutableRunOptions* run_options, const std::string* ptx,
      const std::vector<uint8_t>* cubin, se::DeviceMemoryBase* temp_buffer,
      State<std::unique_ptr<se::KernelBase>> device_kernel, int32_t grid_size_x,
      int32_t grid_size_y, int32_t grid_size_z, int32_t block_size_x,
      int32_t block_size_y, int32_t block_size_z,
      CustomCall::RemainingArgs args, std::string_view name) const;
  static KernelLaunch Handler() { return KernelLaunch(); }
};
}  // namespace

absl::Status KernelLaunch::operator()(
    const ServiceExecutableRunOptions* run_options, const std::string* ptx,
    const std::vector<uint8_t>* cubin, se::DeviceMemoryBase* temp_buffer,
    State<std::unique_ptr<se::KernelBase>> device_kernel, int32_t grid_size_x,
    int32_t grid_size_y, int32_t grid_size_z, int32_t block_size_x,
    int32_t block_size_y, int32_t block_size_z, CustomCall::RemainingArgs args,
    std::string_view name) const {
  se::Stream* stream = run_options->stream();
  se::StreamExecutor* executor = stream->parent();

  LaunchDimensions launch_dimensions(
      {grid_size_x, grid_size_y, grid_size_z},
      {block_size_x, block_size_y, block_size_z});

  const int args_size_including_temp_buffer = args.size() + 1;

  // If kernel does not exists create it from the ptx and cubin.
  absl::StatusOr<std::unique_ptr<se::KernelBase>*> kernel =
      device_kernel.GetOrCreate([&] {
        return ToAbsl(CreateKernel(absl::string_view(name.data(), name.size()),
                                   args_size_including_temp_buffer, *ptx,
                                   *cubin, executor));
      });
  if (!kernel.ok()) return kernel.status();
  assert((**kernel)->name() == name && "unexpected loaded kernel");

  VLOG(3) << "Launching " << (**kernel)->name();
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

  // Execute device kernel on a main stream.
  auto executed =
      ExecuteKernelOnStream(***kernel, buffer_args, launch_dimensions, stream);
  if (!executed.ok()) return ToAbslStatus(executed);

  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//

XLA_RUNTIME_DEFINE_CUSTOM_CALL_WITH_CHECKS(
    Launch, KernelLaunch::Handler(), checks,
    CustomCall::Bind("xla.gpu.func.launch")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<const std::string*>()
        .UserData<const std::vector<uint8_t>*>()
        .UserData<se::DeviceMemoryBase*>()
        .State<std::unique_ptr<se::KernelBase>>("uid")
        .Arg<int32_t>()   // grid_size_x
        .Arg<int32_t>()   // grid_size_y
        .Arg<int32_t>()   // grid_size_z
        .Arg<int32_t>()   // block_size_x
        .Arg<int32_t>()   // block_size_y
        .Arg<int32_t>()   // block_size_x
        .RemainingArgs()  // args
        .Attr<std::string_view>("kernel"));

void RegisterKernelLaunchCustomCalls(
    runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.func.launch", Launch);
}

}  // namespace gpu
}  // namespace xla
