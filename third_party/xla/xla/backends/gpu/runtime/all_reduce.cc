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
#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

struct AddF32Tag {
  using ElementType = float;
  static constexpr ReductionKind kReductionKind = ReductionKind::SUM;
};

struct AddBF16Tag {
  using ElementType = bfloat16;
  static constexpr ReductionKind kReductionKind = ReductionKind::SUM;
};

struct OrPredTag {
  using ElementType = bool;
  static constexpr ReductionKind kReductionKind = ReductionKind::MAX;
};

template <typename T>
absl::Status LaunchTypedKernel(
    se::Stream* stream, const LaunchDimensions& launch_dimensions,
    absl::Span<const se::DeviceMemoryBase> remote_input_buffers,
    se::DeviceMemoryBase local_input_buffer, se::DeviceMemoryBase output_buffer,
    int64_t rank, int64_t num_ranks, int64_t num_elements,
    absl::Span<const se::DeviceMemoryBase> signal_flags_buffers,
    uint32_t signal_value) {
  using ElementType = typename T::ElementType;

  TF_ASSIGN_OR_RETURN(
      auto kernel,
      (se::gpu::GpuKernelRegistry::GetGlobalRegistry()
           .LoadKernel<
               se::gpu::AllReduceKernel<ElementType, T::kReductionKind>>(
               stream->parent())));

  std::array<ElementType*, stream_executor::gpu::kMaxNumAllReduceInputPtrs>
      remote_input_ptrs;
  absl::c_transform(
      remote_input_buffers, remote_input_ptrs.begin(),
      [](se::DeviceMemoryBase buffer) {
        return tsl::safe_reinterpret_cast<ElementType*>(buffer.opaque());
      });

  std::array<uint32_t*, stream_executor::gpu::kMaxNumAllReduceInputPtrs>
      signal_flags_ptrs;
  absl::c_transform(
      signal_flags_buffers, signal_flags_ptrs.begin(),
      [](se::DeviceMemoryBase buffer) {
        return tsl::safe_reinterpret_cast<uint32_t*>(buffer.opaque());
      });

  return kernel.Launch(launch_dimensions.thread_counts_per_block(),
                       launch_dimensions.block_counts(), stream,
                       remote_input_ptrs, local_input_buffer, output_buffer,
                       rank, num_ranks, num_elements, signal_flags_ptrs,
                       signal_value);
}
}  // namespace

bool IsAllReduceKernelSupported(int64_t num_inputs, int64_t num_elements,
                                PrimitiveType element_type,
                                ReductionKind reduction_kind) {
  // The kernel always vectorizes to 4 elements per thread.
  if (num_elements % 4 != 0) {
    return false;
  }

  // The kernel is only supported for up to 8 devices.
  if (num_inputs > stream_executor::gpu::kMaxNumAllReduceInputPtrs) {
    return false;
  }

  // More types of one-shot all-reduce kernel can be supported. Each element
  // type + reduction kind combination need a new template instantiation.
  // Register more kernel in xla/stream_executor/cuda/all_reduce_kernel_cuda.cc
  switch (reduction_kind) {
    case ReductionKind::SUM:
      return element_type == PrimitiveType::F32 ||
             element_type == PrimitiveType::BF16;
    case ReductionKind::MAX:
      return element_type == PrimitiveType::PRED;
    default:
      return false;
  }
}

absl::Status RunAllReduceKernel(
    se::Stream* stream,                                           //
    const LaunchDimensions& launch_dimensions,                    //
    PrimitiveType element_type,                                   //
    ReductionKind reduction_kind,                                 //
    absl::Span<const se::DeviceMemoryBase> remote_input_buffers,  //
    se::DeviceMemoryBase local_input_buffer,                      //
    se::DeviceMemoryBase output_buffer,                           //
    RankId rank,                                                  //
    int64_t num_ranks,                                            //
    int64_t num_elements,                                         //
    absl::Span<const se::DeviceMemoryBase> signal_flags_buffers,  //
    uint32_t signal_value                                         //
) {
  if (!IsAllReduceKernelSupported(num_ranks, num_elements, element_type,
                                  reduction_kind)) {
    return absl::InvalidArgumentError(
        absl::StrCat("AllReduce kernel is not supported for the given number "
                     "of ranks, elements, element type and reduction kind: ",
                     num_ranks, ", ", num_elements, ", ",
                     primitive_util::LowercasePrimitiveTypeName(element_type),
                     ", ", ReductionKindToString(reduction_kind)));
  }

  if (remote_input_buffers.size() >
      stream_executor::gpu::kMaxNumAllReduceInputPtrs) {
    return absl::InvalidArgumentError(
        "Number of input pointers exceeds the maximum supported number of "
        "input pointers.");
  }

  auto launch_kernel = [&](auto type) -> absl::Status {
    using T = decltype(type);

    return LaunchTypedKernel<T>(stream, launch_dimensions, remote_input_buffers,
                                local_input_buffer, output_buffer, rank.value(),
                                num_ranks, num_elements, signal_flags_buffers,
                                signal_value);
  };

  if (element_type == F32 && reduction_kind == ReductionKind::SUM) {
    return launch_kernel(AddF32Tag{});
  }

  if (element_type == BF16 && reduction_kind == ReductionKind::SUM) {
    return launch_kernel(AddBF16Tag{});
  }

  if (element_type == PRED && reduction_kind == ReductionKind::MAX) {
    return launch_kernel(OrPredTag{});
  }

  return absl::InvalidArgumentError(
      "Unsupported AllReduce kernel. This line should never be reached if the "
      "result of `IsAllReduceKernelSupported` is correct.");
}

}  // namespace xla::gpu
