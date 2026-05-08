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

#include "xla/backends/gpu/autotuner/triton/cost_model_config_optimization.h"

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/transforms/convert_triton_gemm_config.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/tiled_hlo_computation.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/gpu/model/tiling_from_block_parameters.h"
#include "xla/service/gpu/model/triton_emitter_constraints.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/instruction_fusion.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/sorted_range.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace cost_model_config_optimization_detail {

// Helper struct for fields always used together.
struct EstimationContext {
  // Fusion that contains the dot.
  const HloFusionInstruction* fusion = nullptr;
  const HloDotInstruction* dot = nullptr;
  const se::DeviceDescription& device_description;
};

absl::StatusOr<absl::Duration> EstimateRunTimeWithConfig(
    const SymbolicTileAnalysis& analysis,
    const HloFusionAdaptor& fusion_adaptor, const EstimationContext& context,
    const TritonGemmConfig& config,
    GpuPerformanceModelWithIndexingAnalysis& cost_model,
    mlir::MLIRContext* mlir_context) {
  TF_ASSIGN_OR_RETURN(
      BlockLevelParameters block_params,
      FindBlockLevelParameters(context.dot, config, mlir_context,
                               context.device_description));

  Tile dot_tiling;
  dot_tiling.add_sizes(config.block_k);

  TF_ASSIGN_OR_RETURN(Tiling tiling, TilingFromAnnotatedFusion(
                                         analysis, block_params, &dot_tiling));

  TF_ASSIGN_OR_RETURN(TiledHloComputation tiled_hlo_computation,
                      analysis.ComputeTiledComputation(tiling));

  TF_ASSIGN_OR_RETURN(
      EstimateRunTimeData estimate,
      cost_model.EstimateRunTimeForTiledHloComputation(
          fusion_adaptor, tiled_hlo_computation, block_params.num_warps));

  return estimate.exec_time;
}

absl::StatusOr<OrderedEstimatesAndConfigs> EstimateConfigs(
    const EstimationContext& context,
    const std::vector<TritonGemmConfig>& configs,
    mlir::MLIRContext* mlir_context) {
  HloFusionAnalysisCache fusion_analysis_cache{context.device_description};
  GpuPerformanceModelWithIndexingAnalysis cost_model{
      &context.device_description, &fusion_analysis_cache,
      HloCostAnalysis::DefaultShapeSize, mlir_context};

  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(context.fusion);

  SymbolicTileAnalysisOrError analysis_or_error =
      SymbolicTileAnalysis::AnalyzeFusion(
          *fusion_adaptor, mlir_context,
          TritonEmitterConstraints::GetBuilder(context.device_description));

  if (const auto* fusion_decision =
          std::get_if<FusionDecision>(&analysis_or_error)) {
    return absl::InternalError(absl::StrCat("SymbolicTileAnalysis failed: ",
                                            fusion_decision->Explain()));
  }

  SymbolicTileAnalysis analysis =
      std::get<SymbolicTileAnalysis>(std::move(analysis_or_error));

  OrderedEstimatesAndConfigs estimates_and_confs;
  for (const TritonGemmConfig& config : configs) {
    absl::StatusOr<absl::Duration> estimate = EstimateRunTimeWithConfig(
        analysis, *fusion_adaptor, context, config, cost_model, mlir_context);
    if (estimate.ok()) {
      VLOG(10) << "Estimated cost for config: " << config.ToString() << " is "
               << *estimate;
      estimates_and_confs.insert({*estimate, config});
    } else {
      VLOG(10) << "Failed to estimate cost for config: " << config.ToString()
               << " - " << estimate.status();
    }
  }
  return estimates_and_confs;
}

OrderedEstimatesAndConfigs GetTopEstimatedConfigs(
    const OrderedEstimatesAndConfigs& estimates_and_confs, int64_t n,
    const OrderedEstimatesAndConfigs* configs_to_skip) {
  absl::flat_hash_set<TritonGemmConfig> exclude_set;
  if (configs_to_skip) {
    VLOG(5) << "Skipping " << configs_to_skip->size() << " provided configs.";
    for (const auto& pair : *configs_to_skip) {
      exclude_set.insert(pair.second);
    }
  }

  OrderedEstimatesAndConfigs top_configs;
  for (const auto& pair : estimates_and_confs) {
    if (top_configs.size() >= n) {
      break;
    }
    if (exclude_set.contains(pair.second)) {
      continue;
    }
    VLOG(5) << "Top config #" << top_configs.size() << ": "
            << pair.second.ToString() << " with estimate: " << pair.first;
    top_configs.insert(pair);
  }
  return top_configs;
}

OrderedEstimatesAndConfigs FilterConfigsByRatioVsFastest(
    const OrderedEstimatesAndConfigs& configs, float filter_threshold) {
  if (configs.empty()) {
    VLOG(5) << "No configs to filter.";
    return configs;
  }

  absl::Duration fastest_time = configs.begin()->first;
  OrderedEstimatesAndConfigs filtered_configs;
  absl::Duration limit = fastest_time * (1.0 + filter_threshold);
  for (const auto& pair : configs) {
    if (pair.first > limit) {
      break;
    }
    filtered_configs.insert(pair);
  }

  VLOG(1) << "Minimum estimated time: " << fastest_time << ". Filtered down to "
          << filtered_configs.size() << " configs.";
  return filtered_configs;
}

std::string CostModelGemmTilingOptions::ToString() const {
  std::string s = "CostModelGemmTilingOptions {";
  if (top.has_value()) {
    absl::StrAppend(&s, " top: ", *top);
    absl::StrAppend(&s, " top_from_default: ", top_from_default);
  }
  if (mixin.has_value()) {
    absl::StrAppend(&s, " mixin: ", *mixin);
  }
  if (filter.has_value()) {
    absl::StrAppend(&s, " filter: ", *filter);
  }
  absl::StrAppend(&s, " }");
  return s;
}

absl::StatusOr<CostModelGemmTilingOptions> ParseCostModelGemmTilingOptions(
    const google::protobuf::Map<std::string, std::string>& options) {
  CostModelGemmTilingOptions parsed_options;
  for (const auto& [key, value] : tsl::SortedRange(options)) {
    if (key == "top") {
      int val = 0;
      if (absl::SimpleAtoi(value, &val)) {
        parsed_options.top = val;
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat("Could not parse 'top' value: ", value));
      }
    } else if (key == "top_from_default") {
      int val = 0;
      if (absl::SimpleAtoi(value, &val)) {
        parsed_options.top_from_default = (val != 0);
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat("Could not parse 'top_from_default' value: ", value));
      }
    } else if (key == "mixin") {
      int val = 0;
      if (absl::SimpleAtoi(value, &val)) {
        parsed_options.mixin = val;
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat("Could not parse 'mixin' value: ", value));
      }
    } else if (key == "filter") {
      float val = 0.f;
      if (absl::SimpleAtof(value, &val)) {
        parsed_options.filter = val;
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat("Could not parse 'filter' value: ", value));
      }
    }
  }
  return parsed_options;
}

}  // namespace cost_model_config_optimization_detail

absl::StatusOr<std::vector<TritonGemmConfig>> OptimizeConfigsWithCostModel(
    const HloDotInstruction* dot,
    const std::vector<TritonGemmConfig>& all_configs,
    const std::vector<TritonGemmConfig>& optimized_configs,
    const se::DeviceDescription& device_description,
    const DebugOptions& debug_options, mlir::MLIRContext* mlir_context) {
  namespace detail = cost_model_config_optimization_detail;

  const HloFusionInstruction* fusion =
      Cast<HloFusionInstruction>(dot->parent()->FusionInstruction());

  detail::EstimationContext context{fusion, dot, device_description};

  TF_ASSIGN_OR_RETURN(
      detail::CostModelGemmTilingOptions options,
      detail::ParseCostModelGemmTilingOptions(
          debug_options.xla_gpu_experimental_cost_model_gemm_tiling_options()));

  VLOG(1) << "Performing Cost Model config optimizations with "
          << options.ToString();

  std::optional<absl::StatusOr<detail::OrderedEstimatesAndConfigs>>
      estimated_all_configs;

  auto get_estimated_all_configs =
      [&]() -> const absl::StatusOr<detail::OrderedEstimatesAndConfigs>& {
    if (!estimated_all_configs.has_value()) {
      estimated_all_configs =
          detail::EstimateConfigs(context, all_configs, mlir_context);
    }
    return *estimated_all_configs;
  };

  detail::OrderedEstimatesAndConfigs current_set;

  // Create the base set by either picking the top configs or estimating the
  // existing set.
  if (options.top.has_value()) {
    TF_ASSIGN_OR_RETURN(
        detail::OrderedEstimatesAndConfigs base_config_set,
        options.top_from_default
            ? EstimateConfigs(context, optimized_configs, mlir_context)
            : get_estimated_all_configs());

    VLOG(1) << "Cost Model: Selecting top " << *options.top << " configs from "
            << (options.top_from_default ? "default" : "exhaustive") << " set";

    current_set =
        detail::GetTopEstimatedConfigs(base_config_set, *options.top, nullptr);
  } else {
    VLOG(1) << "Cost Model: Using default set";
    TF_ASSIGN_OR_RETURN(
        detail::OrderedEstimatesAndConfigs base_config_set,
        EstimateConfigs(context, optimized_configs, mlir_context));
    current_set = std::move(base_config_set);
  }

  // Mixin step inserts best configs not already present in the current set.
  if (options.mixin.has_value()) {
    VLOG(1) << "Cost Model: Mixing in top " << *options.mixin << " configs";

    TF_ASSIGN_OR_RETURN(const detail::OrderedEstimatesAndConfigs& all,
                        get_estimated_all_configs());

    detail::OrderedEstimatesAndConfigs top_non_present =
        detail::GetTopEstimatedConfigs(all, *options.mixin, &current_set);

    current_set.insert(top_non_present.begin(), top_non_present.end());
  }

  // Filter step removes slow configs based on threshold.
  if (options.filter.has_value()) {
    VLOG(1) << "Cost Model: Filtering with threshold " << *options.filter;
    current_set =
        detail::FilterConfigsByRatioVsFastest(current_set, *options.filter);
  }

  if (!current_set.empty()) {
    VLOG(5) << "Fastest estimated config: "
            << current_set.begin()->second.ToString() << " with time "
            << current_set.begin()->first;
    VLOG(5) << "Slowest estimated config: "
            << current_set.rbegin()->second.ToString() << " with time "
            << current_set.rbegin()->first;
  }

  std::vector<TritonGemmConfig> result;
  result.reserve(current_set.size());
  for (const auto& pair : current_set) {
    result.push_back(pair.second);
  }
  VLOG(1) << "Returning " << result.size() << " processed configs";
  return result;
}

}  // namespace xla::gpu
