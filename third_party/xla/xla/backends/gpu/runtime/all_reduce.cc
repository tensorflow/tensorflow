/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/all_reduce.h"

#include <array>
#include <cstddef>
#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {
template <typename T>
absl::Status LaunchTypedKernel(
    se::Stream* stream, se::StreamExecutor* executor,
    const se::ThreadDim& thread_dims, const se::BlockDim& block_dims,
    const std::array<T*, stream_executor::gpu::kMaxNumAllReduceInputPtrs>&
        input_ptrs,
    se::DeviceMemoryBase output_buffer, int64_t num_inputs,
    int64_t num_elements) {
  TF_ASSIGN_OR_RETURN(auto kernel,
                      se::gpu::GpuKernelRegistry::GetGlobalRegistry()
                          .LoadKernel<se::gpu::AllReduceKernel<T>>(executor));

  return kernel.Launch(thread_dims, block_dims, stream, input_ptrs,
                       output_buffer, num_inputs, num_elements);
}
}  // namespace

bool IsAllReduceKernelSupported(int64_t num_inputs, int64_t num_elements,
                                PrimitiveType element_type) {
  // The kernel always vectorizes to 4 elements per thread.
  if (num_elements % 4 != 0) {
    return false;
  }

  // The kernel is only supported for up to 8 devices.
  if (num_inputs > stream_executor::gpu::kMaxNumAllReduceInputPtrs) {
    return false;
  }

  return element_type == BF16 || element_type == F32;
}

absl::Status RunAllReduceKernel(
    se::Stream* stream, PrimitiveType element_type,
    absl::Span<const se::DeviceMemoryBase> input_buffers,
    se::DeviceMemoryBase output_buffer, int64_t num_inputs,
    int64_t num_elements) {
  if (input_buffers.size() > stream_executor::gpu::kMaxNumAllReduceInputPtrs) {
    return absl::InvalidArgumentError(
        "Number of input pointers exceeds the maximum supported number of "
        "input pointers.");
  }

  se::StreamExecutor* executor = stream->parent();

  // TODO(b/383125489): Fine tune the block and thread dimensions.
  static constexpr size_t kBlocks = 8;
  static constexpr size_t kThreads = 512;
  se::ThreadDim thread_dims(kThreads, 1, 1);
  se::BlockDim block_dims(kBlocks, 1, 1);

  auto launch_kernel = [&](auto type) -> absl::Status {
    using T = decltype(type);

    std::array<T*, stream_executor::gpu::kMaxNumAllReduceInputPtrs> input_ptrs;
    absl::c_transform(input_buffers, input_ptrs.begin(),
                      [](se::DeviceMemoryBase buffer) {
                        return tsl::safe_reinterpret_cast<T*>(buffer.opaque());
                      });

    return LaunchTypedKernel<T>(stream, executor, thread_dims, block_dims,
                                input_ptrs, output_buffer, num_inputs,
                                num_elements);
  };

  switch (element_type) {
    case BF16:
      return launch_kernel(xla::bfloat16{});
    case F32:
      return launch_kernel(float{});
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported element type: ",
                       primitive_util::LowercasePrimitiveTypeName(element_type),
                       " for AllReduce kernel."));
  }
}

}  // namespace xla::gpu
