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

#include "xla/backends/gpu/core/autotuner/triton.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/core/autotuner/codegen_backend.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/backends/gpu/core/autotuner/triton/cost_model_config_optimization.h"
#include "xla/backends/gpu/core/autotuner/triton/dot_search_space.h"
#include "xla/backends/gpu/core/autotuner/triton/triton_configs.h"
#include "xla/backends/gpu/hlo/transforms/convert_triton_gemm_config.h"
#include "xla/backends/gpu/hlo/transforms/fusion/fusion_wrapper.h"
#include "xla/backends/gpu/hlo/transforms/fusion/priority_fusion.h"
#include "xla/codegen/tiling/tiled_hlo.h"
#include "xla/codegen/tiling/tiling_space.h"
#include "xla/codegen/xtile/xtile_config.pb.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/instruction_fusion.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_float_support.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/mlir_context_pool.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/tsl/platform/env.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "triton/Version.h"

namespace xla {
namespace gpu {

namespace {

bool IsWarpSpecializationAvailable(
    se::GpuComputeCapability compute_capability) {
  return compute_capability.IsCuda() &&
         compute_capability.cuda_compute_capability()->IsAtLeastBlackwell();
}

// Config every ragged-dot fusion falls back to: used when autotuning is off
// and when the search space prunes everything.
constexpr TritonGemmConfig kDefaultRaggedDotConfig(
    /*block_m=*/32, /*block_n=*/32, /*block_k=*/32, /*num_stages=*/1,
    /*num_warps=*/4);

// Sizes of the ragged-dot dimensions that bound the tile search space.
struct RaggedDotDimensions {
  // The ragged dim is an LHS contracting dim (kRaggedContracting).
  bool is_contracting = false;
  // The dims that block_m, block_k and block_n tile.
  int64_t m = 1;
  int64_t k = 1;
  int64_t n = 1;
  // kRaggedContracting only: the number of groups and the number of rows in
  // the smallest of them.
  int64_t num_groups = 0;
  int64_t min_group_size = 0;
};

RaggedDotDimensions GetRaggedDotDimensions(
    const HloRaggedDotInstruction* ragged_dot) {
  const RaggedDotDimensionNumbers& ragged_dims =
      ragged_dot->ragged_dot_dimension_numbers();
  const DotDimensionNumbers& dot_dims = ragged_dims.dot_dimension_numbers();
  const Shape& lhs_shape = ragged_dot->operand(0)->shape();
  const Shape& rhs_shape = ragged_dot->operand(1)->shape();
  const int64_t lhs_rank = lhs_shape.dimensions().size();
  const int64_t rhs_rank = rhs_shape.dimensions().size();
  const int64_t lhs_ragged_dim = ragged_dims.lhs_ragged_dimensions(0);
  const bool is_batch =
      absl::c_linear_search(dot_dims.lhs_batch_dimensions(), lhs_ragged_dim);

  RaggedDotDimensions dims;
  dims.is_contracting =
      !is_batch && absl::c_linear_search(dot_dims.lhs_contracting_dimensions(),
                                         lhs_ragged_dim);

  // block_m tiles the ragged dim, except for kRaggedBatch, where that is the
  // batch dim — tiled with 1 by ApplyConfig — and M is a separate dim.
  dims.m = lhs_shape.dimensions(lhs_ragged_dim);
  if (is_batch) {
    for (int64_t i = 0; i < lhs_rank; ++i) {
      if (!absl::c_linear_search(dot_dims.lhs_batch_dimensions(), i) &&
          !absl::c_linear_search(dot_dims.lhs_contracting_dimensions(), i)) {
        dims.m = lhs_shape.dimensions(i);
      }
    }
  }

  // K is the contracting dim, except for kRaggedContracting, where that is the
  // ragged dim and K is the LHS non-ragged, non-batch dim (the output K).
  if (dims.is_contracting) {
    for (int64_t i = 0; i < lhs_rank; ++i) {
      if (i != lhs_ragged_dim &&
          !absl::c_linear_search(dot_dims.lhs_batch_dimensions(), i)) {
        dims.k = lhs_shape.dimensions(i);
      }
    }
  } else {
    dims.k = lhs_shape.dimensions(dot_dims.lhs_contracting_dimensions(0));
  }

  // N is the non-contracting, non-group, non-batch RHS dim.
  for (int64_t i = 0; i < rhs_rank; ++i) {
    if (!absl::c_linear_search(dot_dims.rhs_contracting_dimensions(), i) &&
        !absl::c_linear_search(ragged_dims.rhs_group_dimensions(), i) &&
        !absl::c_linear_search(dot_dims.rhs_batch_dimensions(), i)) {
      dims.n = rhs_shape.dimensions(i);
    }
  }

  if (dims.is_contracting) {
    dims.num_groups = ragged_dot->operand(2)->shape().dimensions(0);
    if (dims.num_groups > 0) {
      dims.min_group_size = dims.m / dims.num_groups;
    }
  }
  return dims;
}

// Representative configs for real profiling, obtained by evaluating a few
// different input problems on MI300 and MI350 on H100. It might require to be
// extended for more targets.
std::vector<TritonGemmConfig> GenerateDefaultRaggedDotConfigs(
    const RaggedDotDimensions& dims) {
  struct SimpleConfig {
    int block_m, block_n, block_k, num_stages, num_warps, group_size;
  };
  static constexpr SimpleConfig kDefaultConfigs[] = {
      {32, 32, 32, 1, 4, 1},
      {64, 32, 32, 1, 4, 1},
      {64, 64, 32, 1, 4, 2},
      {128, 32, 32, 1, 8, 1},
      {128, 64, 32, 1, 8, 2},
      {16, 16, 128, 2, 2, 1},
      {64, 64, 64, 2, 4, 1},
      {32, 64, 64, 2, 4, 1},
      {32, 128, 64, 2, 4, 2},
      {32, 128, 128, 2, 8, 1},
      {16, 256, 256, 2, 8, 1},
      {256, 64, 64, 1, 4, 1},
      {256, 16, 32, 2, 4, 1},
      {128, 64, 64, 1, 4, 1},
      {64, 128, 64, 2, 4, 2},
      {64, 128, 128, 1, 8, 1},
      {32, 128, 32, 2, 4, 2},
      {64, 64, 64, 1, 8, 1},
      {64, 128, 64, 1, 4, 1},
      // Exhaustive-autotuner winners over real MoE model shapes
      // (mixtral / deepseek / qwen3 / gpt-oss / llama4)
      // on MI300, MI350 and H100.
      {16, 256, 32, 2, 2, 2},
      {64, 128, 64, 2, 4, 8},
      {64, 128, 32, 1, 2, 4},
      {128, 256, 64, 1, 4, 2},
      {32, 128, 64, 1, 4, 8},
      {16, 128, 32, 1, 2, 2},
      {64, 128, 64, 1, 4, 8},
      {64, 256, 64, 1, 8, 2},
      {128, 128, 128, 1, 4, 1},
      {32, 256, 32, 2, 4, 1},
      {128, 256, 32, 2, 8, 8},
      {64, 128, 128, 1, 4, 4},
      {64, 256, 128, 1, 2, 4},
  };

  std::vector<TritonGemmConfig> configs;
  for (const auto& c : kDefaultConfigs) {
    if (c.block_m > dims.m || c.block_n > dims.n) {
      continue;
    }
    // For kRaggedContracting: skip configs where block_m * num_stages reaches
    // min_group_size (the pipeline would preload beyond the group's
    // contracting slice).
    if (dims.is_contracting &&
        static_cast<int64_t>(c.block_m) * c.num_stages >= dims.min_group_size) {
      continue;
    }
    // For kRaggedContracting: group_size controls G-dim grouping (not M).
    // Skip if group_size > num_groups — impossible to group more slices than
    // exist.
    if (dims.is_contracting && c.group_size > dims.num_groups) {
      continue;
    }
    configs.push_back(TritonGemmConfig(c.block_m, c.block_n, c.block_k,
                                       /*num_stages=*/c.num_stages,
                                       /*num_warps=*/c.num_warps,
                                       /*num_ctas=*/1, /*is_tma_allowed=*/false,
                                       /*is_warp_specialization_allowed=*/false,
                                       /*waves_per_eu=*/0,
                                       /*group_size=*/c.group_size));
  }
  return configs;
}

// Search space: {16,32,64,128,256}³ block sizes × {2,4,8} num_warps × {1,2}
// num_stages × {1,2,4,8} group_size, pruned by the dimension sizes and by
// `elems_per_thread_limit` to avoid obviously invalid configs.
//
// The upper bound of 256 matches the MI300X benchmark's best configs, where
// BLOCK_M=256 and BLOCK_N=256 achieve peak GMM throughput on that platform.
//
// group_size controls L2 tile reordering (see BlockLevelFusionConfig):
//   1 = no reordering (default), 2/4/8 = reorder up to 2/4/8 M-tiles
//   before advancing to the next N-tile, improving L2 hit rate.
std::vector<TritonGemmConfig> GenerateExhaustiveRaggedDotConfigs(
    const RaggedDotDimensions& dims, int64_t elems_per_thread_limit) {
  std::vector<TritonGemmConfig> configs;
  for (int block_m : {16, 32, 64, 128, 256}) {
    if (block_m > dims.m) {
      continue;
    }
    for (int block_n : {16, 32, 64, 128, 256}) {
      if (block_n > dims.n) {
        continue;
      }
      for (int block_k : {16, 32, 64, 128, 256}) {
        if (block_k > dims.k) {
          continue;
        }
        for (int num_warps : {2, 4, 8}) {
          for (int num_stages : {1, 2}) {
            if (dims.is_contracting &&
                static_cast<int64_t>(block_m) * num_stages >=
                    dims.min_group_size) {
              continue;
            }
            int64_t elems_per_thread =
                static_cast<int64_t>(block_m) * block_n / (num_warps * 32);
            if (elems_per_thread > elems_per_thread_limit) {
              continue;
            }
            for (int group_size : {1, 2, 4, 8}) {
              if (dims.is_contracting) {
                if (group_size > dims.num_groups) {
                  continue;
                }
              } else {
                // group_size groups M-tiles for the other ragged modes.
                int64_t num_m_tiles = (dims.m + block_m - 1) / block_m;
                if (group_size > num_m_tiles) {
                  continue;
                }
              }
              configs.push_back(
                  TritonGemmConfig(block_m, block_n, block_k,
                                   /*num_stages=*/num_stages,
                                   /*num_warps=*/num_warps,
                                   /*num_ctas=*/1,
                                   /*is_tma_allowed=*/false,
                                   /*is_warp_specialization_allowed=*/false,
                                   /*waves_per_eu=*/0,
                                   /*group_size=*/group_size));
            }
          }
        }
      }
    }
  }
  return configs;
}

// Translates `config` into the fusion-level BlockLevelFusionConfig in
// `fusion_config` and the sequential Tile backend config on `ragged_dot`
// (read by GetTilingSpaceConcreteSizes).
absl::Status ApplyConfigToRaggedDot(const AutotuneResult::TritonGemmKey& config,
                                    FusionBackendConfig& fusion_config,
                                    HloInstruction& ragged_dot) {
  const RaggedDotDimensionNumbers& ragged_dims =
      Cast<HloRaggedDotInstruction>(&ragged_dot)
          ->ragged_dot_dimension_numbers();
  const DotDimensionNumbers& dot_dims = ragged_dims.dot_dimension_numbers();
  const int64_t lhs_ragged_dim = ragged_dims.lhs_ragged_dimensions(0);
  const bool is_batch =
      absl::c_linear_search(dot_dims.lhs_batch_dimensions(), lhs_ragged_dim);
  const bool is_contracting =
      !is_batch && absl::c_linear_search(dot_dims.lhs_contracting_dimensions(),
                                         lhs_ragged_dim);

  xtile::BlockLevelFusionConfig* blk_cfg =
      fusion_config.mutable_block_level_fusion_config();
  blk_cfg->clear_output_tiles();
  auto* output_tile = blk_cfg->add_output_tiles();
  xtile::Tile inner_tile;

  if (is_batch) {
    // kRaggedBatch: parallel dims = [B=1, M, N].
    for (int64_t b_dim : dot_dims.lhs_batch_dimensions()) {
      (void)b_dim;
      output_tile->add_sizes(1);  // B tile = 1 per program
    }
    output_tile->add_sizes(config.block_m());
    output_tile->add_sizes(config.block_n());
    for (auto lhs_k : dot_dims.lhs_contracting_dimensions()) {
      (void)lhs_k;
      inner_tile.add_sizes(config.block_k());  // K sequential
    }
  } else if (!is_contracting) {
    // kRaggedNonContracting: parallel dims = [batch..., M, N].
    // Sequential dims: G=1 (outer loop), K=block_k (inner K-loop).
    for (int64_t b_dim : dot_dims.lhs_batch_dimensions()) {
      (void)b_dim;
      output_tile->add_sizes(1);
    }
    output_tile->add_sizes(config.block_m());
    output_tile->add_sizes(config.block_n());
    inner_tile.add_sizes(1);                 // G sequential
    inner_tile.add_sizes(config.block_k());  // K sequential
  } else {
    // kRaggedContracting: Grid = G × K_tiles × N_tiles.
    // output_tile = [G=1, batch..., K, N].
    // inner_tile  = [M=block_m].
    output_tile->add_sizes(1);  // G tile = 1
    for (int64_t b_dim : dot_dims.lhs_batch_dimensions()) {
      (void)b_dim;
      output_tile->add_sizes(1);
    }
    output_tile->add_sizes(config.block_k());  // K output
    output_tile->add_sizes(config.block_n());  // N output
    inner_tile.add_sizes(config.block_m());    // M sequential
  }

  blk_cfg->set_num_warps(config.num_warps());
  blk_cfg->set_num_ctas(std::max(1, static_cast<int>(config.num_ctas())));
  blk_cfg->set_num_stages(config.num_stages());
  // group_size == 0 in old protos means "no reordering" (same as 1).
  blk_cfg->set_group_size(
      std::max(static_cast<int64_t>(1), config.group_size()));
  return ragged_dot.set_backend_config(inner_tile);
}

}  // namespace

absl::StatusOr<std::vector<CodegenBackend::EstimatedConfig>>
TritonBackend::GetSupportedConfigsWithEstimates(const HloInstruction& instr) {
  ABSL_ASSIGN_OR_RETURN(std::vector<TritonGemmConfig> gemm_configs,
                   GetSupportedGemmConfigs(instr));
  if (gemm_configs.empty()) {
    return std::vector<CodegenBackend::EstimatedConfig>();
  }

  const HloInstruction* dot_instr = hlo_query::GetFirstInstructionWithOpcode(
      *instr.fused_instructions_computation(), HloOpcode::kDot);

  absl::flat_hash_map<TritonGemmConfig, absl::Duration> estimates_map;
  if (dot_instr != nullptr) {
    const auto* dot = Cast<HloDotInstruction>(dot_instr);
    ABSL_ASSIGN_OR_RETURN(MlirContextPool::BorrowedObject mlir_context,
                     mlir_context_pool_->GetOrCreate());
    ABSL_ASSIGN_OR_RETURN(estimates_map,
                     EstimateConfigsWithCostModel(
                         dot, gemm_configs, target_config().device_description,
                         debug_options(), mlir_context->get()));
  }

  std::vector<CodegenBackend::EstimatedConfig> result;
  result.reserve(gemm_configs.size());
  for (const auto& gemm_config : gemm_configs) {
    auto backend_config = std::make_unique<BackendConfig>();
    *backend_config->mutable_triton() = gemm_config.ToProto();

    std::optional<absl::Duration> duration;
    if (auto it = estimates_map.find(gemm_config); it != estimates_map.end()) {
      duration = it->second;
    }

    result.push_back({std::move(backend_config), duration});
  }
  return result;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
TritonBackend::GetSupportedConfigs(const HloInstruction& instr) {
  ABSL_ASSIGN_OR_RETURN(std::vector<TritonGemmConfig> gemm_configs,
                   GetSupportedGemmConfigs(instr));
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.reserve(gemm_configs.size());
  for (const auto& gemm_config : gemm_configs) {
    auto config = std::make_unique<BackendConfig>();
    *config->mutable_triton() = gemm_config.ToProto();
    configs.push_back(std::move(config));
  }
  return configs;
}

absl::StatusOr<std::vector<TritonGemmConfig>>
TritonBackend::GetSupportedGemmConfigs(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return std::vector<TritonGemmConfig>();
  }
  ABSL_ASSIGN_OR_RETURN(std::vector<TritonGemmConfig> overridden_configs,
                   GetOverriddenConfigs(&instr));
  if (!overridden_configs.empty()) {
    return overridden_configs;
  }

  const HloComputation* fused_comp = instr.fused_instructions_computation();

  const HloInstruction* dot_instr =
      hlo_query::GetFirstInstructionWithOpcode(*fused_comp, HloOpcode::kDot);
  if (dot_instr != nullptr) {
    return GetSupportedConfigsForDot(dot_instr);
  }
  const HloInstruction* scaled_dot_instr =
      hlo_query::GetFirstInstructionWithOpcode(*fused_comp,
                                               HloOpcode::kScaledDot);
  if (scaled_dot_instr != nullptr) {
    return GetSupportedConfigsForScaledDot(scaled_dot_instr);
  }
  // kRaggedDot fusions routed through the Triton XTile backend
  // (kTritonFusionKind / "__triton").
  const HloInstruction* ragged_dot_instr =
      hlo_query::GetFirstInstructionWithOpcode(*fused_comp,
                                               HloOpcode::kRaggedDot);
  if (ragged_dot_instr != nullptr) {
    return GetSupportedConfigsForRaggedDot(ragged_dot_instr);
  }
  return std::vector<TritonGemmConfig>();
}

absl::StatusOr<std::vector<TritonGemmConfig>>
TritonBackend::GetSupportedConfigsForDot(const HloInstruction* instr) {
  const HloDotInstruction* dot = Cast<HloDotInstruction>(instr);
  TritonDotFusionSearchSpace search_space(target_config().device_description,
                                          dot);
  bool autotune_warp_specialization =
      debug_options()
          .xla_gpu_experimental_enable_triton_warp_specialization() &&
      IsWarpSpecializationAvailable(
          target_config().device_description.gpu_compute_capability());

  VLOG(1) << "Generating configs from search space: "
          << search_space.ToString();

  if (debug_options().xla_gpu_exhaustive_tiling_search()) {
    return search_space.GenerateConfigs(autotune_warp_specialization);
  }

  const std::vector<TritonGemmConfig>& default_configs =
      GetDefaultTritonConfigs(
          target_config().device_description.gpu_compute_capability());

  if (!debug_options()
           .xla_gpu_experimental_cost_model_gemm_tiling_options()
           .empty()) {
    VLOG(1) << "Optimizing configs with the cost model.";
    std::vector<TritonGemmConfig> all_configs =
        search_space.GenerateConfigs(autotune_warp_specialization);
    std::vector<TritonGemmConfig> candidate_configs =
        search_space.OptimizeConfigSet(all_configs, default_configs);
    ABSL_ASSIGN_OR_RETURN(MlirContextPool::BorrowedObject mlir_context,
                     mlir_context_pool_->GetOrCreate());
    return OptimizeConfigsWithCostModel(dot, all_configs, candidate_configs,
                                        target_config().device_description,
                                        debug_options(), mlir_context->get());
  }

  VLOG(1) << "Restricting configs to the default set.";
  return search_space.GenerateAndOptimizeConfigs(default_configs,
                                                 autotune_warp_specialization);
}

absl::StatusOr<std::vector<TritonGemmConfig>>
TritonBackend::GetSupportedConfigsForScaledDot(const HloInstruction* instr) {
  // The ROCm Triton backend does not support mixed FP4/FP8 scaled-dot inputs.
  const auto& gpu_cc =
      target_config().device_description.gpu_compute_capability();
  if (gpu_cc.IsRocm()) {
    PrimitiveType lhs_type = instr->operand(0)->shape().element_type();
    PrimitiveType rhs_type = instr->operand(1)->shape().element_type();
    auto is_fp4 = [](PrimitiveType t) { return t == F4E2M1FN; };
    auto is_fp8 = [](PrimitiveType t) { return t == F8E4M3FN || t == F8E5M2; };
    if ((is_fp4(lhs_type) && is_fp8(rhs_type)) ||
        (is_fp8(lhs_type) && is_fp4(rhs_type))) {
      return std::vector<TritonGemmConfig>();
    }
  }

  std::vector<TritonGemmConfig> configs;

  const bool exhaustive_search =
      debug_options().xla_gpu_exhaustive_tiling_search();

  // Only MI300 measured a win, matching mi300.txtpb for regular dots.
  const bool autotune_mfma_size =
      gpu_cc.IsRocm() &&
      gpu_cc.rocm_compute_capability()->has_mfma_instr_support() &&
      (exhaustive_search || gpu_cc.rocm_compute_capability()->gfx9_mi300());
  for (int block_m = 128; block_m <= 256; block_m *= 2) {
    for (int block_n = 16; block_n <= 256; block_n *= 2) {
      for (int block_k = 128; block_k <= 256; block_k *= 2) {
        // TODO(b/436988479): fine tune the search space.
        const int elements_per_thread = (block_m * block_n) / (4 * 32);
        if (!exhaustive_search &&
            (elements_per_thread > 64 ||
             (block_k >= 256 && elements_per_thread >= 32))) {
          VLOG(3) << "Ignoring spill over config: block_m=" << block_m
                  << " block_n=" << block_n << " block_k=" << block_k;
          continue;
        }

        TritonGemmConfig config(block_m, block_n,
                                /*block_k=*/block_k,
                                /*num_stages=*/1,
                                /*num_warps=*/4,
                                /*num_ctas=*/1,
                                /*is_tma_allowed=*/false);
        configs.push_back(config);

        const int mfma_size = MfmaSizeToTry(block_m, block_n);
        if (autotune_mfma_size && mfma_size != 0) {
          config.mfma_size = mfma_size;
          configs.push_back(config);
        }
      }
    }
  }
  return configs;
}

absl::StatusOr<std::vector<TritonGemmConfig>>
TritonBackend::GetOverriddenConfigs(const HloInstruction* instr) {
  std::vector<TritonGemmConfig> configs;
  const std::string& override_file =
      debug_options().xla_gpu_override_triton_configs_file();
  if (!override_file.empty()) {
    std::string file_content;
    ABSL_RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(), override_file,
                                          &file_content));
    TritonGemmConfigsProto gemm_configs;
    if (!tsl::protobuf::TextFormat::ParseFromString(file_content,
                                                    &gemm_configs)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Could not parse override file: ", override_file));
    }
    configs.reserve(gemm_configs.config_size());
    for (const auto& gemm_config : gemm_configs.config()) {
      ABSL_ASSIGN_OR_RETURN(TritonGemmConfig config,
                       TritonGemmConfig::FromProto(gemm_config));
      configs.push_back(config);
    }
  }
  if (!debug_options().xla_gpu_override_triton_config().empty()) {
    AutotuneResult::TritonGemmKey gemm_config;
    CHECK(tsl::protobuf::TextFormat::ParseFromString(
        debug_options().xla_gpu_override_triton_config(), &gemm_config));
    ABSL_ASSIGN_OR_RETURN(TritonGemmConfig config,
                     TritonGemmConfig::FromProto(gemm_config));
    configs.push_back(config);
  }
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>> TritonBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  return absl::UnimplementedError(
      "TritonBackend does not support a default config.");
}

absl::Status TritonBackend::ApplyConfig(HloInstruction& instr,
                                        const BackendConfig& config) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "TritonBackend does not support this instruction.");
  }
  if (!config.has_triton()) {
    return absl::InvalidArgumentError(
        "Expected TritonGemmKey config for TritonBackend.");
  }
  const AutotuneResult::TritonGemmKey& triton_config_proto = config.triton();

  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   instr.backend_config<GpuBackendConfig>());
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();

  // Detect ragged-dot XTile fusions by the presence of kRaggedDot inside the
  // fused computation.
  HloComputation* fused_comp = instr.fused_instructions_computation();
  HloInstruction* inner_ragged_dot = hlo_query::GetFirstInstructionWithOpcode(
      *fused_comp, HloOpcode::kRaggedDot);

  if (inner_ragged_dot != nullptr) {
    ABSL_RETURN_IF_ERROR(ApplyConfigToRaggedDot(triton_config_proto, backend_config,
                                           *inner_ragged_dot));
    return instr.set_backend_config(gpu_config);
  }

  // Regular dot / scaled-dot path: write a TritonGemmConfig into the fusion.
  backend_config.set_kind(kTritonGemmFusionKind);
  *backend_config.mutable_triton_gemm_config() = triton_config_proto;
  ABSL_RETURN_IF_ERROR(instr.set_backend_config(gpu_config));

  // FromProto has validation checks, that's why we call it here.
  ABSL_RETURN_IF_ERROR(TritonGemmConfig::FromProto(triton_config_proto).status());
  if (triton_config_proto.split_k() > 1) {
    return absl::InvalidArgumentError(
        "TritonBackend no longer supports split-k (split_k > 1).");
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<TritonGemmConfig>>
TritonBackend::GetSupportedConfigsForRaggedDot(const HloInstruction* instr) {
  const bool exhaustive_search =
      debug_options().xla_gpu_exhaustive_tiling_search();
  const bool select_first =
      debug_options().xla_gpu_autotune_level() == 0 ||
      debug_options().xla_gpu_deterministic_ops() ||
      debug_options().xla_gpu_exclude_nondeterministic_ops();
  if (!exhaustive_search && select_first) {
    return std::vector<TritonGemmConfig>{kDefaultRaggedDotConfig};
  }

  const RaggedDotDimensions dims =
      GetRaggedDotDimensions(Cast<HloRaggedDotInstruction>(instr));

  std::vector<TritonGemmConfig> configs;
  if (exhaustive_search) {
    // On ROCm (MI300X), 512 vector registers/thread allow larger output tiles
    // without spilling than the 64 appropriate for NVIDIA Ampere/Hopper.
    const bool is_rocm =
        target_config().device_description.gpu_compute_capability().IsRocm();
    configs = GenerateExhaustiveRaggedDotConfigs(
        dims, /*elems_per_thread_limit=*/is_rocm ? 256 : 64);
  } else {
    configs = GenerateDefaultRaggedDotConfigs(dims);
  }
  if (configs.empty()) {
    configs.push_back(kDefaultRaggedDotConfig);
  }
  return configs;
}

absl::StatusOr<std::unique_ptr<HloModule>> TritonBackend::RunHloPasses(
    std::unique_ptr<HloModule> hlo_module,
    const Compiler::CompileOptions& options) {
  auto gpu_device_info = target_config().device_description;
  for (PrimitiveType type :
       {BF16, F8E5M2, F8E4M3FN, F8E4M3B11FNUZ, F8E5M2FNUZ, F8E4M3FNUZ}) {
    GpuFloatSupport float_support(gpu_device_info.gpu_compute_capability(),
                                  type);
    FloatNormalization float_normalization(&float_support);
    ABSL_RETURN_IF_ERROR(float_normalization.Run(hlo_module.get()).status());
  }

  ABSL_ASSIGN_OR_RETURN(MlirContextPool::BorrowedObject mlir_context,
                   mlir_context_pool_->GetOrCreate());

  HloCostAnalysis::Options priority_fusion_options;
  priority_fusion_options.count_multiple_input_accesses = true;
  PriorityFusion priority_fusion(
      /*thread_pool=*/nullptr, gpu_device_info, alias_info_,
      priority_fusion_options, mlir_context->get());
  ABSL_RETURN_IF_ERROR(priority_fusion.Run(hlo_module.get()).status());

  // If the priority fusion pass above skipped some instructions, turn them
  // into fusions.
  FusionWrapper fusion_wrapper(gpu_device_info);
  ABSL_RETURN_IF_ERROR(fusion_wrapper.Run(hlo_module.get()).status());
  ConvertTritonGemmConfig convert_triton_gemm_config(gpu_device_info,
                                                     mlir_context->get());
  ABSL_RETURN_IF_ERROR(convert_triton_gemm_config.Run(hlo_module.get()).status());
  return hlo_module;
}

bool TritonBackend::IsSupported(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return false;
  }
  auto gpu_config = instr.backend_config<GpuBackendConfig>();
  if (!gpu_config.ok()) {
    return false;
  }
  const FusionBackendConfig& backend_config =
      gpu_config->fusion_backend_config();

  // Ragged-dot XTile fusions created by GemmRewriter::HandleRaggedDot carry
  // the generic Triton fusion kind. Only claim those: all other
  // kTritonFusionKind fusions are tuned by BlockLevelEmitterBackend.
  if (backend_config.kind() == kTritonFusionKind) {
    return IsRaggedDotTritonFusion(instr);
  }

  if (backend_config.kind() != kTritonGemmFusionKind &&
      backend_config.kind() != kCuDnnFusionKind &&
      backend_config.kind() != kCustomFusionKind) {
    return false;
  }

  if (const HloInstruction* scaled_dot =
          hlo_query::GetFirstInstructionWithOpcode(
              *instr.fused_instructions_computation(), HloOpcode::kScaledDot)) {
    // cuDNN scaled dot fusions take pre-swizzled scales and must not be
    // retuned.
    if (backend_config.kind() != kTritonGemmFusionKind) {
      return false;
    }
    // GemmFusion also wraps scaled dots that only CUDNN_FISSION or
    // CUBLASLT_FISSION support.
    if (!debug_options().xla_gpu_experimental_scaled_dot_with_triton() ||
        !IsTritonSupportedInstruction(
             *scaled_dot,
             target_config().device_description.gpu_compute_capability())
             .IsAllowed()) {
      return false;
    }
  }

  // TODO: b/487920266 - sometimes we create fusions that can't be tiled.
  // Bail out here if that's the case.
  if (backend_config.kind() == kTritonGemmFusionKind) {
    auto fusion = Cast<HloFusionInstruction>(&instr);
    std::unique_ptr<HloFusionAdaptor> fusion_adaptor =
        HloFusionAdaptor::ForInstruction(fusion);
    absl::StatusOr<MlirContextPool::BorrowedObject> mlir_context =
        mlir_context_pool_->GetOrCreate();
    if (!mlir_context.ok()) {
      VLOG(1) << "Failed to borrow MLIRContext: " << mlir_context.status();
      return false;
    }
    auto ts = TilingSpace::Create(*fusion_adaptor, (*mlir_context)->get());
    if (!ts.ok()) {
      VLOG(1) << "Failed to create tiling space: " << ts.status().message();
      return false;
    }
    auto tiled_computation_or =
        TiledHloComputation::Tile(*fusion_adaptor, std::move(ts.value()));
    if (!tiled_computation_or.ok()) {
      VLOG(1) << "Fusion is not tileable with experimental tiling: "
              << tiled_computation_or.status().message();
      return false;
    }
    // We don't have concrete tile sizes here and don't validate Triton
    // constraints here.
    return true;
  }
  return true;
}

std::string TritonBackend::version() const { return TRITON_VERSION; }

}  // namespace gpu
}  // namespace xla
