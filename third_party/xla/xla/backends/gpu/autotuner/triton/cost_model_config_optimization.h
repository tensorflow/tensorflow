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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_TRITON_COST_MODEL_CONFIG_OPTIMIZATION_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_TRITON_COST_MODEL_CONFIG_OPTIMIZATION_H_

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

// Optimizes a given set of configs using the gpu cost model controlled by
// `debug_options`.
// Assumes `optimized_configs` is a subset of `all_configs` optimized by other
// means (e.g. via the default set).
absl::StatusOr<std::vector<TritonGemmConfig>> OptimizeConfigsWithCostModel(
    const HloDotInstruction* dot,
    const std::vector<TritonGemmConfig>& all_configs,
    const std::vector<TritonGemmConfig>& optimized_configs,
    const se::DeviceDescription& device_description,
    const DebugOptions& debug_options, mlir::MLIRContext* mlir_context);

namespace cost_model_config_optimization_detail {

using OrderedEstimatesAndConfigs =
    std::set<std::pair<absl::Duration, TritonGemmConfig>>;

// Retrieves the top `n` estimated configurations.
// If `configs_to_skip` is passed, skips these configs in the result.
OrderedEstimatesAndConfigs GetTopEstimatedConfigs(
    const OrderedEstimatesAndConfigs& estimates_and_confs, int64_t n,
    const OrderedEstimatesAndConfigs* configs_to_skip);

// Filters configurations slower than `fastest_time * (1.0 + filter_threshold)`.
OrderedEstimatesAndConfigs FilterConfigsByRatioVsFastest(
    const OrderedEstimatesAndConfigs& configs, float filter_threshold);

// Controls optimization steps. Unset values disable the steps.
struct CostModelGemmTilingOptions {
  // Number of top estimated configs to select.
  std::optional<int> top;
  // Select top configs from the default set instead of the exhaustive set.
  bool top_from_default = false;
  // Number of top configs from the exhaustive set to mix back in.
  std::optional<int> mixin;
  // The threshold ratio failing vs best estimated runtime to filter out.
  std::optional<float> filter;

  std::string ToString() const;
};

absl::StatusOr<CostModelGemmTilingOptions> ParseCostModelGemmTilingOptions(
    const google::protobuf::Map<std::string, std::string>& options);

}  // namespace cost_model_config_optimization_detail

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_TRITON_COST_MODEL_CONFIG_OPTIMIZATION_H_
