/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_GPU_MEMORY_SPACE_ASSIGNMENT_H_
#define XLA_SERVICE_GPU_GPU_MEMORY_SPACE_ASSIGNMENT_H_

#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

<<<<<<< HEAD
inline constexpr int64_t kCollectiveMemorySpaceColor = 1;
inline constexpr int64_t kTempBufferMemorySpaceColor = 2;

// Set memory space to kCollectiveMemorySpaceColor for all allocations used by
// all-reduce, all-gather, and reduce-scatter. This memory space maps to
// collective memory using ncclMemAlloc in the runtime.
inline BufferAssigner::Colorer CollectiveColorer(bool use_user_buffers,
                                                 bool use_nvshmem) {
  return [use_user_buffers, use_nvshmem](HloAliasAnalysis* alias_analysis,
                                         const HloOrdering&) {
    // NOTE: The explicit internal constructor is needed as an explicitly typed
    // variable to avoid a method ambiguity error when compiling for CUDA 12.4.
    static const absl::NoDestructor<absl::flat_hash_set<HloOpcode>>
        kSupportedOpcodes(absl::flat_hash_set<HloOpcode>{
            HloOpcode::kAllReduce,
            HloOpcode::kAllReduceStart,
            HloOpcode::kAllReduceDone,
            HloOpcode::kAllGather,
            HloOpcode::kAllGatherStart,
            HloOpcode::kAllGatherDone,
            HloOpcode::kReduceScatter,
            HloOpcode::kCollectivePermute,
            HloOpcode::kCollectivePermuteStart,
            HloOpcode::kCollectivePermuteDone,
            HloOpcode::kAllToAll,
        });

    auto is_nvshmem_op = [](const HloInstruction* inst) {
      bool is_nvshmem_collective = false;
      if (inst->has_backend_config()) {
        auto gpu_config = inst->backend_config<GpuBackendConfig>();
        if (!gpu_config.ok()) {
          return false;
        }
        const CollectiveBackendConfig& backend_config =
            gpu_config.value().collective_backend_config();
        is_nvshmem_collective =
            backend_config.backend() == CollectiveBackendConfig::NVSHMEM;
      }
      return is_nvshmem_collective;
    };
=======
enum class MemorySpaceColor {
  // Corresponds to stream_executor::MemoryTypes::kDefault or kUnified.
  // This memory can be allocated with any device allocation API.
  kDefault = 0,
>>>>>>> upstream/master

  // Corresponds to stream_executor::MemoryTypes::kCollective.
  // This memory should be allocated with ncclMemAlloc in the runtime.
  kCollective = 1,

  // Temp buffers can be allocated within separate memory space (if
  // xla_gpu_temp_buffer_use_separate_color is set). This improves cuda-graphs
  // performance. See more details in the corresponding flag description.
  kTempBuffer = 2,
};

BufferAssigner::Colorer CreateColorer(const DebugOptions& option);
}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_MEMORY_SPACE_ASSIGNMENT_H_
