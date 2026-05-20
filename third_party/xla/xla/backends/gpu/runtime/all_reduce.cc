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
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/stream_executor/gpu/collective_kernel_metadata.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

using se::gpu::AllReduceStrategy;
static constexpr int64_t kKB = 1024;
static constexpr int64_t kMB = kKB * 1024;
static constexpr int64_t kMaxOneShotAllReduceSizeBytes = 256 * kKB;
static constexpr int64_t kMaxTwoShotAllReduceSizeBytes = 4 * kMB;

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
// It's nicer to have this as a power of 2 because we consume blocks in powers
// of 2 from a higher dimension to a lower dimension.
static constexpr int64_t kMaxBlocksPerGrid = 32;
static constexpr uint64_t kMaxThreadsPerBlock = 512;
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

bool IsElementReductionSupported(PrimitiveType element_type,
                                 ReductionKind reduction_kind) {
  absl::Span<const HloOpcode> supported_ops =
      SupportedReductionOps(element_type);
  HloOpcode opcode = ReductionKindToOpcode(reduction_kind, element_type);
  return absl::c_linear_search(supported_ops, opcode);
}

}  // namespace

absl::Span<const HloOpcode> SupportedReductionOps(PrimitiveType element_type) {
  const auto numeric_types = llvm::concat<const PrimitiveType>(
      kSupportedFloatingPointTypes, kSupportedIntegralTypes);
  if (absl::c_linear_search(numeric_types, element_type)) {
    return absl::MakeSpan(gpu::kSupportedNumericReductionOps);
  }
  if (absl::c_linear_search(kSupportedPredicateTypes, element_type)) {
    return absl::MakeSpan(gpu::kSupportedLogicalReductionOps);
  }
  return {};
}

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
      RoundUpTo(CeilOfRatio(elements_per_rank, se::gpu::kNumElementsPerThread),
                kWarpSize);
  // Triton expects power of 2 for threads_per_block / threads_per_warp.
  // Since threads_per_warp is 32 for all NVIDIA GPUs, power of 2 is guaranteed.
  threads_per_block =
      std::min(kMaxThreadsPerBlock,
               llvm::bit_ceil(static_cast<uint64_t>(total_threads)));
  blocks_per_grid = std::min(kMaxBlocksPerGrid,
                             CeilOfRatio(total_threads, threads_per_block));
  return LaunchDimensions(blocks_per_grid, threads_per_block);
}

absl::Status IsAllReduceKernelSupported(int64_t num_ranks, int64_t num_elements,
                                        PrimitiveType element_type,
                                        ReductionKind reduction_kind,
                                        AllReduceStrategy all_reduce_strategy) {
  if (!IsElementReductionSupported(element_type, reduction_kind)) {
    return absl::UnimplementedError(absl::StrCat(
        "Element type (",
        primitive_util::LowercasePrimitiveTypeName(element_type),
        ") and reduction kind (", absl::StrFormat("%v", reduction_kind),
        ") combination is not supported."));
  }
  if (num_ranks > stream_executor::gpu::kMaxNumAllReduceInputPtrs) {
    return absl::UnimplementedError(
        absl::StrCat("AllReduce XLA implementation does not support more than ",
                     se::gpu::kMaxNumAllReduceInputPtrs,
                     " ranks. "
                     " Required number of ranks: ",
                     num_ranks));
  }
  return absl::OkStatus();
}

absl::Status IsAllReduceKernelSupported(
    bool is_collective_kernel_enabled, const se::DeviceDescription& device_info,
    int32_t num_operands, std::optional<ReductionKind> reduction_kind,
    int64_t num_devices, int64_t num_elements, PrimitiveType element_type,
    bool is_local, bool is_multimem_enabled,
    const std::vector<ReplicaGroup>& replica_groups) {
  if (!is_collective_kernel_enabled) {
    return absl::UnimplementedError("Collective kernel is not enabled.");
  }
  // Check if the device supports Triton collective codegen:
  // CUDA: Requires compute capability 9.0+ (Hopper or newer)
  // ROCm: All versions with Triton support are enabled
  if (!device_info.cuda_compute_capability().IsAtLeastHopper() &&
      !device_info.gpu_compute_capability().IsRocm()) {
    return absl::UnimplementedError(absl::StrCat(
        "Triton collective codegen requires CUDA compute capability >= 9.0 "
        "(Hopper or newer) or a ROCm device with Triton support. "
        "Got: ",
        device_info.gpu_compute_capability().ToString(), "."));
  }
  // TODO(b/383125489): Support variadic arguments.
  if (num_operands != 1) {
    return absl::UnimplementedError(
        absl::StrCat("Collective kernel is not supported for number of "
                     "operands not equal to 1. Got ",
                     num_operands, "."));
  }
  if (replica_groups.empty()) {
    return absl::UnimplementedError(
        "Replica groups must be explicitly provided for collective kernels.");
  }
  if (!reduction_kind.has_value()) {
    return absl::UnimplementedError("Could not match reduction computation.");
  }
  if (!is_local) {
    return absl::UnimplementedError(
        "Cross-host symmetric memory collectives are not supported.");
  }
  if (!llvm::has_single_bit(static_cast<uint64_t>(num_devices))) {
    return absl::UnimplementedError(
        absl::StrCat("Collective kernels are only supported for power of 2 "
                     "number of devices. Got ",
                     num_devices, "."));
  }
  const int64_t byte_size =
      num_elements * primitive_util::ByteWidth(element_type);
  const AllReduceStrategy strategy =
      GetAllReduceStrategy(byte_size, is_multimem_enabled);
  if (byte_size > GetMaxSupportedAllReduceSizeBytes(strategy)) {
    return absl::UnimplementedError(
        "Custom all-reduce strategy is only supported for small inputs.");
  }
  return IsAllReduceKernelSupported(num_devices, num_elements, element_type,
                                    *reduction_kind, strategy);
}

absl::StatusOr<AllReduceInfo> BuildAllReduceInfo(
    bool is_collective_kernel_enabled, bool is_multimem_enabled,
    const se::DeviceDescription& device_info,
    const HloAllReduceInstruction* all_reduce) {
  const int64_t num_devices =
      all_reduce->device_list()->num_devices_per_group();
  const std::optional<ReductionKind> reduction_kind =
      MatchReductionComputation(all_reduce->called_computations().front());
  const int64_t num_elements =
      ShapeUtil::ElementsIn(all_reduce->operand(0)->shape());
  const PrimitiveType element_type =
      all_reduce->operand(0)->shape().element_type();
  const int32_t num_operands = all_reduce->operand_count();
  const int64_t byte_size =
      num_elements * primitive_util::ByteWidth(element_type);
  const AllReduceStrategy strategy =
      GetAllReduceStrategy(byte_size, is_multimem_enabled);
  TF_RETURN_IF_ERROR(IsAllReduceKernelSupported(
      is_collective_kernel_enabled, device_info, num_operands, reduction_kind,
      num_devices, num_elements, element_type, /*is_local=*/true,
      is_multimem_enabled, all_reduce->replica_groups()));
  return AllReduceInfo{
      /*.reduction_kind=*/*reduction_kind,
      /*.num_devices =*/num_devices,
      /*.num_elements =*/num_elements,
      /*.element_type =*/element_type,
      /*.all_reduce_strategy =*/strategy,
  };
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
  TF_RETURN_IF_ERROR(IsAllReduceKernelSupported(num_ranks, num_elements,
                                                element_type, reduction_kind,
                                                all_reduce_strategy));
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
