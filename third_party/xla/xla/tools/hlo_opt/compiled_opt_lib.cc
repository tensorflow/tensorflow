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

#include "xla/tools/hlo_opt/compiled_opt_lib.h"

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/all_reduce_simplifier.h"
#include "xla/service/all_to_all_decomposer.h"
#include "xla/service/batched_gather_scatter_normalizer.h"
#include "xla/service/bitcast_dtypes_expander.h"
#include "xla/service/call_inliner.h"
#include "xla/service/compiler.h"
#include "xla/service/conditional_simplifier.h"
#include "xla/service/conditional_to_select.h"
#include "xla/service/copy_insertion.h"
#include "xla/service/executable.h"
#include "xla/service/gather_expander.h"
#include "xla/service/gpu/transforms/all_gather_dynamic_slice_simplifier.h"
#include "xla/service/gpu/transforms/all_reduce_splitter.h"
#include "xla/service/gpu/transforms/collective_permute_valid_iteration_annotator.h"
#include "xla/service/gpu/transforms/scatter_expander.h"
#include "xla/service/gpu/transforms/scatter_slice_simplifier.h"
#include "xla/service/map_inliner.h"
#include "xla/service/platform_util.h"
#include "xla/service/reduce_scatter_reassociate.h"
#include "xla/service/scatter_determinism_expander.h"
#include "xla/service/scatter_simplifier.h"
#include "xla/service/select_and_scatter_expander.h"
#include "xla/service/sharding_remover.h"
#include "xla/service/spmd/shardy/shardy_xla_pass.h"
#include "xla/service/topk_rewriter.h"
#include "xla/service/triangular_solve_expander.h"
#include "xla/service/while_loop_all_reduce_code_motion.h"
#include "xla/service/while_loop_constant_sinking.h"
#include "xla/service/while_loop_invariant_code_motion.h"
#include "xla/service/while_loop_simplifier.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<se::StreamExecutor*> CompiledOptProvider::GetExecutor() {
  DebugOptions debug_opts = GetDebugOptionsFromFlags();
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform(GetPlatformName()));
  if (debug_opts.xla_gpu_target_config_filename().empty()) {
    TF_ASSIGN_OR_RETURN(std::vector<se::StreamExecutor*> stream_executors,
                        PlatformUtil::GetStreamExecutors(
                            platform, /*allowed_devices=*/std::nullopt));
    return stream_executors[0];
  }
  return nullptr;
}

absl::StatusOr<std::optional<std::string>> CompiledOptProvider::GenerateStage(
    std::unique_ptr<HloModule> module, absl::string_view stage) {
  if (stage == "hlo") {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> optimized_module,
                        GetOptimizedHlo(std::move(module)));
    return optimized_module->ToString();
  } else if (stage == "html") {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> optimized_module,
                        GetOptimizedHlo(std::move(module)));
    TF_ASSIGN_OR_RETURN(std::string cmps,
                        RenderAllComputationsToHtml(*optimized_module));
    return cmps;
  } else if (stage == "hlo-backend") {
    TF_ASSIGN_OR_RETURN(auto executable, GetExecutable(std::move(module)));
    return executable->module().ToString();
  }

  return std::nullopt;
}

absl::StatusOr<Compiler*> CompiledOptProvider::GetCompiler() {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform(GetPlatformName()));

  TF_ASSIGN_OR_RETURN(Compiler * compiler, Compiler::GetForPlatform(platform));
  return compiler;
}

absl::StatusOr<std::unique_ptr<HloModule>> CompiledOptProvider::GetOptimizedHlo(
    std::unique_ptr<HloModule> input_module) {
  TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor, GetExecutor());

  DebugOptions debug_opts = GetDebugOptionsFromFlags();
  Compiler::CompileOptions opts;
  TF_ASSIGN_OR_RETURN(Compiler * compiler, GetCompiler());
  DebugOptions d = input_module->config().debug_options();
  d.set_xla_embed_ir_in_executable(true);
  input_module->mutable_config().set_debug_options(d);

  if (input_module->has_schedule()) {
    return input_module;
  }

  // But run-hlo-passes does not actually run the scheduling.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> optimized_module,
      compiler->RunHloPasses(std::move(input_module), executor, opts));

  return optimized_module;
}

absl::StatusOr<std::unique_ptr<Executable>> CompiledOptProvider::GetExecutable(
    std::unique_ptr<HloModule> input_module) {
  Compiler::CompileOptions opts;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> optimized_module,
                      GetOptimizedHlo(std::move(input_module)));
  TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor, GetExecutor());
  TF_ASSIGN_OR_RETURN(Compiler * compiler, GetCompiler());
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      compiler->RunBackend(std::move(optimized_module), executor, opts));
  return executable;
}

std::set<std::string> CompiledOptProvider::SupportedStages() {
  return {"hlo", "html", "hlo-backend"};
}

void CompiledOptProvider::RegisterSharedHardwareSpecificPasses() {
  // go/keep-sorted start
  RegisterPass<AllGatherDynamicSliceSimplifier>();
  RegisterPass<AllReduceSimplifier>();
  RegisterPass<AllReduceSplitter>();
  RegisterPass<AllToAllDecomposer>();
  RegisterPass<BatchedGatherScatterNormalizer>();
  RegisterPass<BitcastDtypesExpander>();
  RegisterPass<CallInliner>();
  RegisterPass<CollectivePermuteValidIterationAnnotator>();
  RegisterPass<ConditionalSimplifier>();
  RegisterPass<ConditionalToSelect>();
  RegisterPass<CopyInsertion>();
  RegisterPass<GatherExpander>(GatherExpander::kEliminateSimpleGathers);
  RegisterPass<GpuScatterExpander>();
  RegisterPass<MapInliner>();
  RegisterPass<ReduceScatterReassociate>();
  RegisterPass<ScatterDeterminismExpander>();
  RegisterPass<ScatterSimplifier>();
  RegisterPass<ScatterSliceSimplifier>();
  RegisterPass<SelectAndScatterExpander>();
  RegisterPass<ShardingRemover>();
  RegisterPass<TopkDecomposer>();
  RegisterPass<TriangularSolveExpander>();
  RegisterPass<WhileLoopAllReduceCodeMotion>();
  RegisterPass<WhileLoopConstantSinking>();
  RegisterPass<WhileLoopInvariantCodeMotion>();
  RegisterPass<WhileLoopSimplifier>();
  RegisterPass<sdy::ShardyXLA>();
  // go/keep-sorted end
}

}  // namespace xla
