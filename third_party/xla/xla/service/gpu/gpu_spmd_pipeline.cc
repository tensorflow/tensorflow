/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_spmd_pipeline.h"

#include <cstdint>
#include <optional>

#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/hlo_constant_folding.h"
#include "xla/hlo/transforms/simplifiers/hlo_constant_splitter.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/reshape_mover.h"
#include "xla/hlo/transforms/simplifiers/sort_simplifier.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/service/conditional_simplifier.h"
#include "xla/service/gather_expander.h"
#include "xla/service/gpu/transforms/algebraic_simplifier.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/scatter_expander.h"
#include "xla/service/sharding_propagation.h"
#include "xla/service/spmd/collective_permute_motion.h"
#include "xla/service/spmd/shardy/shardy_xla_pass.h"
#include "xla/service/spmd/stateful_rng_spmd_partitioner.h"
#include "xla/service/while_loop_constant_sinking.h"
#include "xla/service/while_loop_simplifier.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

void AddSPMDPasses(
    const HloModule* hlo_module,
    const AlgebraicSimplifierOptions& layout_insensitive_algsimp_opts,
    const se::GpuComputeCapability& compute_capability,
    HloPassPipeline& spmd_pipeline,
    std::optional<const absl::FunctionRef<void(HloPassPipeline&)>>
        auto_sharding_func) {
  const int64_t num_partitions = hlo_module->config().num_partitions();
  CHECK_GE(num_partitions, 1);

  HloPassPipeline& spmd_simplify =
      spmd_pipeline.AddPass<HloPassFix<HloPassPipeline>>("spmd-simplify");

  spmd_simplify.AddPass<GpuAlgebraicSimplifier>(layout_insensitive_algsimp_opts,
                                                compute_capability);
  spmd_simplify.AddPass<SortSimplifier>();
  spmd_simplify.AddPass<TupleSimplifier>();
  spmd_simplify.AddPass<ScatterExpander>(
      ScatterExpander::kEliminateSimpleScatters);
  spmd_simplify.AddPass<GatherExpander>(
      GatherExpander::kEliminateSimpleGathers);
  spmd_simplify.AddPass<WhileLoopConstantSinking>();
  spmd_simplify.AddPass<WhileLoopSimplifier>();

  ReshapeMoverOptions reshape_mover_options;
  reshape_mover_options.reshape_of_1d_broadcast_is_cheap = true;
  spmd_simplify.AddPass<ReshapeMover>(reshape_mover_options);
  // Run AlgebraicSimplifier directly before HloConstantFolding, because we
  // need to simplify DynamicSlice(Broadcast) away. Constant folding of
  // DynamicSlice can be quite costly, as the whole operand will be evaluated.
  // We run AlgebraicSimplifier as HloPassFix to make sure all simplifications
  // have been done before running HloConstantFolding. This is necessary
  // because simplifications create new instructions which may not be visited
  // in the same iteration of AlgebraicSimplifier.
  spmd_simplify.AddPass<HloPassFix<GpuAlgebraicSimplifier>>(
      layout_insensitive_algsimp_opts, compute_capability);
  spmd_simplify.AddPass<HloConstantFolding>();
  spmd_simplify.AddPass<ConditionalSimplifier>();

  const HloModuleConfig& config = hlo_module->config();

  if (config.use_shardy_partitioner()) {
    spmd_pipeline.AddPass<sdy::ShardyXLA>();
  } else {
    spmd_pipeline.AddPass<HloConstantSplitter>();
    spmd_simplify.AddPass<HloDCE>();

    if (auto_sharding_func.has_value()) {
      (*auto_sharding_func)(spmd_pipeline);
    }
    spmd_pipeline.AddPass<ShardingPropagation>(
        /*is_spmd=*/true, /*propagate_metadata=*/false,
        config.allow_spmd_sharding_propagation_to_output());
  }
  std::optional<int64_t> oper_size_threshold = std::nullopt;
  if (hlo_module->config()
          .debug_options()
          .xla_gpu_operand_bytes_threshold_for_windowed_einsum() >= 0) {
    oper_size_threshold =
        hlo_module->config()
            .debug_options()
            .xla_gpu_operand_bytes_threshold_for_windowed_einsum();
  }
  spmd_pipeline.AddPass<spmd::StatefulRngSpmdPartitioner>(
      num_partitions, hlo_module->config().replica_count(),
      hlo_module->config()
          .debug_options()
          .xla_gpu_threshold_for_windowed_einsum_mib(),
      hlo_module->config()
          .debug_options()
          .xla_gpu_multi_streamed_windowed_einsum(),
      /*skip_checking_windowed_einsum_users=*/true,
      /*disable_ag_rewrite_for_multiple_consumers=*/true,
      /*enable_partial_windowed_einsums=*/true, oper_size_threshold);
  spmd_pipeline.AddPass<CollectivePermuteMotion>();
}

}  // namespace gpu
}  // namespace xla
