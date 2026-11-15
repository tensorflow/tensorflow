/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/config_aware_cse.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/tiling_space.h"
#include "xla/codegen/xtile/xtile_config.pb.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/hlo_cse.h"
#include "xla/status_macros.h"

namespace xla {
namespace gpu {
namespace {

// A sequential dimension: the unique ID of the instruction that defines it and
// its position in that instruction's dimensions.
using SequentialDim = std::pair<int64_t, int64_t>;

// Returns the sequential dimensions of `fusion`'s tiling space, in order.
absl::StatusOr<std::vector<SequentialDim>> GetSequentialDims(
    const HloInstruction& fusion, mlir::MLIRContext* mlir_context) {
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<TilingSpace> tiling_space,
      TilingSpace::Create(*HloFusionAdaptor::ForInstruction(&fusion),
                          mlir_context));
  std::vector<SequentialDim> dims;
  for (const TilingSpace::DimensionInfo& dim : tiling_space->dimensions()) {
    if (dim.type == TilingSpace::DimensionSemantics::kSequential) {
      dims.push_back({dim.hlo->unique_id(), dim.dim_position});
    }
  }
  return dims;
}

}  // namespace

absl::StatusOr<bool> ConfigAwareCSE::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // CSE keeps one of the instructions it merges, so only fusions with more than
  // one sequential dimension can lose one. CSE only removes instructions from
  // fusion computations, so fusions and the remaining instructions keep their
  // IDs, which are unique in the module.
  absl::flat_hash_map<SequentialDim, int64_t> tile_sizes;
  std::vector<std::pair<HloInstruction*, int64_t>> instruction_counts;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : computation->instructions()) {
      if (!IsGenericTritonFusion(*instr)) {
        continue;
      }
      ABSL_ASSIGN_OR_RETURN(GpuBackendConfig config,
                       instr->backend_config<GpuBackendConfig>());
      const auto& sizes = config.fusion_backend_config()
                              .block_level_fusion_config()
                              .sequential_tile_sizes();
      if (sizes.size() <= 1) {
        continue;
      }
      ABSL_ASSIGN_OR_RETURN(std::vector<SequentialDim> dims,
                       GetSequentialDims(*instr, mlir_context_));
      TF_RET_CHECK(dims.size() == sizes.size());
      instruction_counts.push_back(
          {instr,
           instr->fused_instructions_computation()->instruction_count()});
      for (int i = 0; i < dims.size(); ++i) {
        tile_sizes[dims[i]] = sizes[i];
      }
    }
  }

  ABSL_ASSIGN_OR_RETURN(bool changed, HloCSE(/*is_layout_sensitive=*/true,
                                        /*ignore_control_dependencies=*/false,
                                        /*should_eliminate_computation=*/
                                        &HloComputation::IsFusionComputation)
                                     .Run(module, execution_threads));
  if (!changed) {
    return false;
  }

  // Keep the tile sizes of the sequential dimensions that CSE didn't remove.
  for (auto& [fusion, instruction_count] : instruction_counts) {
    if (fusion->fused_instructions_computation()->instruction_count() ==
        instruction_count) {
      continue;
    }
    ABSL_ASSIGN_OR_RETURN(std::vector<SequentialDim> dims,
                     GetSequentialDims(*fusion, mlir_context_));
    ABSL_ASSIGN_OR_RETURN(GpuBackendConfig config,
                     fusion->backend_config<GpuBackendConfig>());
    auto* sizes = config.mutable_fusion_backend_config()
                      ->mutable_block_level_fusion_config()
                      ->mutable_sequential_tile_sizes();
    sizes->Clear();
    for (const SequentialDim& dim : dims) {
      auto it = tile_sizes.find(dim);
      TF_RET_CHECK(it != tile_sizes.end());
      sizes->Add(it->second);
    }
    ABSL_RETURN_IF_ERROR(fusion->set_backend_config(config));
  }
  return true;
}

}  // namespace gpu
}  // namespace xla
