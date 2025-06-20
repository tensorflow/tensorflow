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

#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
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

using ::stream_executor::gpu::AllReduceStrategy;

template <typename T, ReductionKind kReductionKindV>
class TagRegistry {
 private:
  template <AllReduceStrategy kAllReduceStrategyV>
  struct Impl {
    using ElementType = T;
    static constexpr ReductionKind kReductionKind = kReductionKindV;
    static constexpr AllReduceStrategy kAllReduceStrategy = kAllReduceStrategyV;
  };

 public:
  static constexpr auto kOneShot = Impl<AllReduceStrategy::kOneShot>{};
  static constexpr auto kTwoShot = Impl<AllReduceStrategy::kTwoShot>{};
};

// Static set of supported kernel tags.
static constexpr auto kAddF32Tags = TagRegistry<float, ReductionKind::SUM>{};
static constexpr auto kAddBF16Tags =
    TagRegistry<bfloat16, ReductionKind::SUM>{};
static constexpr auto kOrPredTags = TagRegistry<bool, ReductionKind::MAX>{};

template <typename TagType>
absl::Status LaunchTypedKernel(
    TagType, se::Stream* stream, const LaunchDimensions& launch_dimensions,
    absl::Span<const se::DeviceMemoryBase> remote_input_buffers,
    se::DeviceMemoryBase local_input_buffer, se::DeviceMemoryBase output_buffer,
    int64_t rank, int64_t num_ranks, int64_t num_elements,
    absl::Span<const se::DeviceMemoryBase> signal_flags_buffers,
    uint32_t signal_value) {
  using ElementType = typename TagType::ElementType;
  static constexpr bool kIsTwoShot =
      TagType::kAllReduceStrategy == AllReduceStrategy::kTwoShot;

  TF_ASSIGN_OR_RETURN(
      auto kernel,
      (se::gpu::GpuKernelRegistry::GetGlobalRegistry()
           .LoadKernel<
               se::gpu::AllReduceKernel<ElementType, TagType::kReductionKind,
                                        TagType::kAllReduceStrategy>>(
               stream->parent())));

  se::gpu::AllReduceKernelParams<ElementType> params{};
  absl::c_transform(
      remote_input_buffers, params.remote_input_buffers.begin(),
      [](se::DeviceMemoryBase buffer) {
        return tsl::safe_reinterpret_cast<ElementType*>(buffer.opaque());
      });
  params.input_buffer =
      tsl::safe_reinterpret_cast<ElementType*>(local_input_buffer.opaque());
  params.output_buffer =
      tsl::safe_reinterpret_cast<ElementType*>(output_buffer.opaque());
  params.rank = rank;
  params.num_ranks = num_ranks;
  params.num_elements = num_elements;
  params.num_elements_per_rank = num_elements / (kIsTwoShot ? num_ranks : 1);
  params.num_elements_per_block = RoundUpTo(
      CeilOfRatio(params.num_elements_per_rank,
                  absl::implicit_cast<int64_t>(launch_dimensions.num_blocks())),
      se::gpu::kNumElementsPerThread);
  absl::c_transform(
      signal_flags_buffers, params.signal_flags_buffers.begin(),
      [](se::DeviceMemoryBase buffer) {
        return tsl::safe_reinterpret_cast<uint32_t*>(buffer.opaque());
      });
  params.signal_value = signal_value;

  return kernel.Launch(launch_dimensions.thread_counts_per_block(),
                       launch_dimensions.block_counts(), stream,
                       std::move(params));
}
}  // namespace

bool IsAllReduceKernelSupported(int64_t num_ranks, int64_t num_elements,
                                PrimitiveType element_type,
                                ReductionKind reduction_kind,
                                AllReduceStrategy all_reduce_strategy) {
  // For twoShot each rank processes: num_elements / num_ranks elements.
  const int64_t alignment_requirement =
      all_reduce_strategy == AllReduceStrategy::kOneShot
          ? se::gpu::kNumElementsPerThread
          : se::gpu::kNumElementsPerThread * num_ranks;

  if (num_elements % alignment_requirement != 0) {
    return false;
  }

  // The kernel is only supported for up to 8 devices.
  if (num_ranks > stream_executor::gpu::kMaxNumAllReduceInputPtrs) {
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
    AllReduceStrategy all_reduce_strategy,                        //
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
                                  reduction_kind, all_reduce_strategy)) {
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

  const auto launch_kernel_impl = [&](auto tag) -> absl::Status {
    return LaunchTypedKernel(tag, stream, launch_dimensions,
                             remote_input_buffers, local_input_buffer,
                             output_buffer, rank.value(), num_ranks,
                             num_elements, signal_flags_buffers, signal_value);
  };
  const auto launch_kernel = [&](auto tag_registry,
                                 AllReduceStrategy strategy) -> absl::Status {
    switch (strategy) {
      case AllReduceStrategy::kOneShot:
        return launch_kernel_impl(tag_registry.kOneShot);
      case AllReduceStrategy::kTwoShot:
        return launch_kernel_impl(tag_registry.kTwoShot);
    }
  };

  if (element_type == F32 && reduction_kind == ReductionKind::SUM) {
    return launch_kernel(kAddF32Tags, all_reduce_strategy);
  }

  if (element_type == BF16 && reduction_kind == ReductionKind::SUM) {
    return launch_kernel(kAddBF16Tags, all_reduce_strategy);
  }

  if (element_type == PRED && reduction_kind == ReductionKind::MAX) {
    return launch_kernel(kOrPredTags, all_reduce_strategy);
  }

  return absl::InvalidArgumentError(
      "Unsupported AllReduce kernel. This line should never be reached if the "
      "result of `IsAllReduceKernelSupported` is correct.");
}

}  // namespace xla::gpu
