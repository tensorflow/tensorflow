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

#include "xla/service/gpu/kernels/all_reduce_kernel.h"

#include <array>
#include <cstddef>
#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/service/gpu/kernels/all_reduce_kernel_common.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/typed_kernel_factory.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

void* GetKernel(PrimitiveType element_type) {
  switch (element_type) {
    case F32:
      return GetAllReduceKernel<float>();
    default:
      return nullptr;
  }
}

}  // namespace

bool IsAllReduceKernelSupported(int64_t num_outputs,
                                PrimitiveType element_type) {
  return num_outputs <= kMaxNumAllReduceInputPtrs &&
         GetKernel(element_type) != nullptr;
}

absl::Status RunAllReduceKernel(
    se::Stream* stream, PrimitiveType element_type,
    absl::Span<const se::DeviceMemoryBase> input_buffers,
    se::DeviceMemoryBase output_buffer, int64_t num_inputs,
    int64_t num_elements) {
  if (input_buffers.size() > kMaxNumAllReduceInputPtrs) {
    return absl::InvalidArgumentError(
        "Number of input pointers exceeds the maximum supported number of "
        "input pointers.");
  }

  se::StreamExecutor* executor = stream->parent();

  // TODO(b/383125489): Fine tune the block and thread dimensions.
  static constexpr size_t kBlocks = 8;
  static constexpr size_t kThreads = 512;

  TF_ASSIGN_OR_RETURN(
      auto kernel,
      (se::TypedKernelFactory<std::array<void*, kMaxNumAllReduceInputPtrs>,
                              se::DeviceMemoryBase, int64_t,
                              int64_t>::Create(executor, "one_shot_all_reduce",
                                               GetKernel(element_type))));

  std::array<void*, kMaxNumAllReduceInputPtrs> input_ptrs;
  absl::c_transform(
      input_buffers, input_ptrs.begin(),
      [](se::DeviceMemoryBase buffer) { return buffer.opaque(); });

  return kernel.Launch(se::ThreadDim(kThreads, 1, 1),
                       se::BlockDim(kBlocks, 1, 1), stream, input_ptrs,
                       output_buffer, num_inputs, num_elements);
}

}  // namespace xla::gpu
