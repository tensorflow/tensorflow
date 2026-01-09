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

#include <algorithm>
#include <cstdint>

#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/stream_executor/gpu/collective_kernel_metadata.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

using se::gpu::AllReduceStrategy;
static constexpr int64_t kMaxOneShotAllReduceSizeBytes = 256 * 1024;  // 256 KB
static constexpr int64_t kMaxTwoShotAllReduceSizeBytes =
    2 * 1024 * 1024;  // 2 MB

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
  static constexpr auto kMultimem = Impl<AllReduceStrategy::kMultimem>{};
};

// Static set of supported kernel tags.
static constexpr auto kAddF32Tags = TagRegistry<float, ReductionKind::SUM>{};
static constexpr auto kAddBF16Tags =
    TagRegistry<bfloat16, ReductionKind::SUM>{};
static constexpr auto kOrPredTags = TagRegistry<bool, ReductionKind::MAX>{};
// Heuristic maxima after some benchmarking.
static constexpr int64_t kMaxBlocksPerGrid = 24;
static constexpr int64_t kMaxThreadsPerBlock = 512;
static constexpr int64_t kWarpSize = 32;

template <typename TagType>
absl::Status LaunchTypedKernel(
    TagType, se::Stream* stream, const LaunchDimensions& launch_dimensions,
    se::DeviceAddressBase symmetric_input_buffer,
    se::DeviceAddressBase local_input_buffer,
    se::DeviceAddressBase output_buffer, int64_t rank, int64_t num_ranks,
    int64_t num_elements, se::DeviceAddressBase symmetric_signal_buffer,
    uint32_t signal_value, se::DeviceAddressBase metadata) {
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
  params.input_buffer =
      tsl::safe_reinterpret_cast<ElementType*>(local_input_buffer.opaque());
  params.output_buffer =
      tsl::safe_reinterpret_cast<ElementType*>(output_buffer.opaque());
  params.symmetric_input_ptrs =
      tsl::safe_reinterpret_cast<ElementType*>(symmetric_input_buffer.opaque());
  params.symmetric_signal_ptrs =
      tsl::safe_reinterpret_cast<uint32_t*>(symmetric_signal_buffer.opaque());
  params.rank = rank;
  params.num_ranks = num_ranks;
  params.num_elements = num_elements;
  params.num_elements_per_rank = num_elements / (kIsTwoShot ? num_ranks : 1);
  // NB: num_elements_per_block can be bigger or smaller than blockDim.x.
  // If its smaller, then the block stride loop will run just once.
  // If its bigger, then the block stride loop will run multiple times.
  params.num_elements_per_block = RoundUpTo(
      CeilOfRatio(params.num_elements_per_rank,
                  absl::implicit_cast<int64_t>(launch_dimensions.num_blocks())),
      se::gpu::kNumElementsPerThread);
  params.rank_offset =
      kIsTwoShot ? params.rank * params.num_elements_per_rank : 0;
  for (int i = 0; i < params.num_ranks; ++i) {
    params.rotated_ranks[i] = (i + rank) % params.num_ranks;
  }
  params.signal_value = signal_value;
  params.metadata =
      tsl::safe_reinterpret_cast<CollectiveKernelMetadata*>(metadata.opaque());

  VLOG(3) << "Launching all-reduce kernel with params: " << "strategy: "
          << absl::StrFormat("%v", TagType::kAllReduceStrategy)
          << ", rank: " << params.rank << ", num_ranks: " << params.num_ranks
          << ", num_elements: " << params.num_elements
          << ", num_elements_per_rank: " << params.num_elements_per_rank
          << ", num_elements_per_block: " << params.num_elements_per_block
          << ", num_threads_per_block: "
          << launch_dimensions.num_threads_per_block()
          << ", num_blocks_per_grid: " << launch_dimensions.num_blocks()
          << ", rank_offset: {" << params.rank_offset << ", rotated_ranks: "
          << absl::StrJoin(
                 absl::MakeSpan(params.rotated_ranks.data(), params.num_ranks),
                 ", ")
          << "}, signal_value: " << params.signal_value;

  return kernel.Launch(launch_dimensions.thread_counts_per_block(),
                       launch_dimensions.block_counts(), stream,
                       std::move(params));
}

// More types of one-shot all-reduce kernel can be supported. Each element
// type + reduction kind combination need a new template instantiation.
// Register more kernel in xla/stream_executor/cuda/all_reduce_kernel_cuda.cc
bool IsElementReductionSupported(PrimitiveType element_type,
                                 ReductionKind reduction_kind) {
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

}  // namespace

AllReduceStrategy GetAllReduceStrategy(int64_t input_size_bytes,
                                       bool is_multimem_enabled) {
  if (input_size_bytes > kMaxOneShotAllReduceSizeBytes) {
    return AllReduceStrategy::kTwoShot;
  }
  if (is_multimem_enabled) {
    return AllReduceStrategy::kMultimem;
  }
  return AllReduceStrategy::kOneShot;
}

int64_t GetMaxSupportedAllReduceSizeBytes(AllReduceStrategy strategy) {
  switch (strategy) {
    case AllReduceStrategy::kOneShot:
      return kMaxOneShotAllReduceSizeBytes;
    case AllReduceStrategy::kTwoShot:
      return kMaxTwoShotAllReduceSizeBytes;
    case AllReduceStrategy::kMultimem:
      return kMaxTwoShotAllReduceSizeBytes;
  }
}

LaunchDimensions AllReduceLaunchDimensions(int64_t elements, int64_t num_ranks,
                                           AllReduceStrategy strategy) {
  int64_t threads_per_block;
  int64_t blocks_per_grid;
  const int64_t elements_per_rank =
      elements / (strategy == AllReduceStrategy::kTwoShot ? num_ranks : 1);
  // Maximum number of threads such that each thread has elements to process.
  const int64_t total_threads =
      RoundUpTo(elements_per_rank / se::gpu::kNumElementsPerThread, kWarpSize);
  threads_per_block = std::min(kMaxThreadsPerBlock, total_threads);
  blocks_per_grid = std::min(kMaxBlocksPerGrid,
                             CeilOfRatio(total_threads, threads_per_block));
  return LaunchDimensions(blocks_per_grid, threads_per_block);
}

bool IsAllReduceKernelSupported(int64_t num_ranks, int64_t num_elements,
                                PrimitiveType element_type,
                                ReductionKind reduction_kind,
                                AllReduceStrategy all_reduce_strategy) {
  if (!IsElementReductionSupported(element_type, reduction_kind)) {
    VLOG(3) << "Element type and reduction kind combination is not supported.";
    return false;
  }
  const int64_t alignment_requirement =
      all_reduce_strategy == AllReduceStrategy::kOneShot ||
              all_reduce_strategy == AllReduceStrategy::kMultimem
          ? se::gpu::kNumElementsPerThread
          : se::gpu::kNumElementsPerThread * num_ranks;

  if (num_elements % alignment_requirement != 0) {
    VLOG(3)
        << "Number of elements is not aligned to the alignment requirement.";
    return false;
  }

  // The kernel is only supported for up to 8 devices.
  return num_ranks <= stream_executor::gpu::kMaxNumAllReduceInputPtrs;
}

absl::Status RunAllReduceKernel(
    se::Stream* stream,                             //
    const LaunchDimensions& launch_dimensions,      //
    PrimitiveType element_type,                     //
    ReductionKind reduction_kind,                   //
    AllReduceStrategy all_reduce_strategy,          //
    se::DeviceAddressBase symmetric_input_buffer,   //
    se::DeviceAddressBase local_input_buffer,       //
    se::DeviceAddressBase output_buffer,            //
    RankId rank,                                    //
    int64_t num_ranks,                              //
    int64_t num_elements,                           //
    se::DeviceAddressBase symmetric_signal_buffer,  //
    uint32_t signal_value,                          //
    se::DeviceAddressBase metadata) {
  if (!IsAllReduceKernelSupported(num_ranks, num_elements, element_type,
                                  reduction_kind, all_reduce_strategy)) {
    return absl::InvalidArgumentError(
        absl::StrCat("AllReduce kernel is not supported for the given number "
                     "of ranks, elements, element type and reduction kind: ",
                     num_ranks, ", ", num_elements, ", ",
                     primitive_util::LowercasePrimitiveTypeName(element_type),
                     ", ", reduction_kind));
  }

  const auto launch_kernel_impl = [&](auto tag) -> absl::Status {
    return LaunchTypedKernel(
        tag, stream, launch_dimensions, symmetric_input_buffer,
        local_input_buffer, output_buffer, rank.value(), num_ranks,
        num_elements, symmetric_signal_buffer, signal_value, metadata);
  };
  const auto launch_kernel = [&](auto tag_registry,
                                 AllReduceStrategy strategy) -> absl::Status {
    switch (strategy) {
      case AllReduceStrategy::kOneShot:
        return launch_kernel_impl(tag_registry.kOneShot);
      case AllReduceStrategy::kTwoShot:
        return launch_kernel_impl(tag_registry.kTwoShot);
      case AllReduceStrategy::kMultimem:
        return launch_kernel_impl(tag_registry.kMultimem);
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
