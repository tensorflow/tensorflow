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

#include "xla/hlo/tools/hlo_opt/opt_lib.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/const_init.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/analysis/indexed_array_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/collectives/all_gather_broadcast_reorder.h"
#include "xla/hlo/transforms/collectives/all_reduce_contiguous.h"
#include "xla/hlo/transforms/collectives/collective_quantizer.h"
#include "xla/hlo/transforms/convert_memory_placement_to_internal_annotations.h"
#include "xla/hlo/transforms/expanders/cholesky_expander.h"
#include "xla/hlo/transforms/expanders/comparison_expander.h"
#include "xla/hlo/transforms/expanders/convolution_4d_expander.h"
#include "xla/hlo/transforms/expanders/convolution_pred_expander.h"
#include "xla/hlo/transforms/expanders/dot_decomposer.h"
#include "xla/hlo/transforms/expanders/dynamic_index_splitter.h"
#include "xla/hlo/transforms/expanders/eigh_expander.h"
#include "xla/hlo/transforms/expanders/logistic_expander.h"
#include "xla/hlo/transforms/expanders/optimization_barrier_expander.h"
#include "xla/hlo/transforms/expanders/qr_expander.h"
#include "xla/hlo/transforms/expanders/real_imag_expander.h"
#include "xla/hlo/transforms/expanders/reduce_decomposer.h"
#include "xla/hlo/transforms/expanders/reshape_decomposer.h"
#include "xla/hlo/transforms/expanders/rng_expander.h"
#include "xla/hlo/transforms/expanders/stable_sort_expander.h"
#include "xla/hlo/transforms/expanders/stochastic_convert_decomposer.h"
#include "xla/hlo/transforms/operand_upcaster.h"
#include "xla/hlo/transforms/simplifiers/all_reduce_folder.h"
#include "xla/hlo/transforms/simplifiers/batch_dot_simplification.h"
#include "xla/hlo/transforms/simplifiers/broadcast_canonicalizer.h"
#include "xla/hlo/transforms/simplifiers/conditional_canonicalizer.h"
#include "xla/hlo/transforms/simplifiers/convert_mover.h"
#include "xla/hlo/transforms/simplifiers/convolution_group_converter.h"
#include "xla/hlo/transforms/simplifiers/dynamic_dimension_simplifier.h"
#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/hlo/transforms/simplifiers/gather_simplifier.h"
#include "xla/hlo/transforms/simplifiers/hlo_constant_folding.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/optimize_input_output_buffer_alias.h"
#include "xla/hlo/transforms/simplifiers/reshape_mover.h"
#include "xla/hlo/transforms/simplifiers/result_caster.h"
#include "xla/hlo/transforms/simplifiers/simplify_fp_conversions.h"
#include "xla/hlo/transforms/simplifiers/slice_sinker.h"
#include "xla/hlo/transforms/simplifiers/sort_simplifier.h"
#include "xla/hlo/transforms/simplifiers/sub_byte_normalization.h"
#include "xla/hlo/transforms/simplifiers/tree_reduction_rewriter.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/hlo/transforms/simplifiers/zero_sized_hlo_elimination.h"
#include "xla/hlo/transforms/tests/dummy_passes.h"
#include "xla/hlo/transforms/while_loop_trip_count_annotator.h"
#include "xla/service/float_support.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform/initialize.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {
using ProviderMap =
    absl::flat_hash_map<std::string, std::unique_ptr<OptProvider>>;
static absl::Mutex provider_mu(absl::kConstInit);

static ProviderMap& GetProviderMap() {
  static auto& provider_map = *new ProviderMap();
  return provider_map;
}
}  // namespace

/*static*/ void OptProvider::RegisterForPlatform(
    std::string platform, std::unique_ptr<OptProvider> translate_provider) {
  absl::MutexLock l(&provider_mu);
  CHECK(!GetProviderMap().contains(platform));
  absl::StatusOr<std::string> canonical_name =
      xla::PlatformUtil::CanonicalPlatformName(platform);
  CHECK_OK(canonical_name);
  GetProviderMap()[*canonical_name] = std::move(translate_provider);
}

/*static*/ absl::StatusOr<OptProvider*> OptProvider::GetProviderForPlatform(
    std::string platform) {
  absl::MutexLock l(&provider_mu);

  TF_ASSIGN_OR_RETURN(std::string canonical_name,
                      xla::PlatformUtil::CanonicalPlatformName(platform));
  auto it = GetProviderMap().find(canonical_name);
  if (it == GetProviderMap().end()) {
    return absl::UnimplementedError(absl::StrCat(
        "Provider not found for platform ", platform, "; canonical expansion: ",
        canonical_name, "; supported platforms are: ",
        absl::StrJoin(GetProviderMap(), ", ",
                      [&](std::string* s, const auto& p) {
                        absl::StrAppend(s, p.first);
                      })));
  }

  return it->second.get();
}

// Placeholder for `key function` of the class to avoid an error due to
// missing vtable entry.
absl::StatusOr<std::optional<std::string>> OptProvider::GenerateStage(
    std::unique_ptr<HloModule> module, absl::string_view stage) {
  return module->ToString();
}

absl::StatusOr<std::optional<std::string>>
OptProvider::BuildAndRunTransformPipeline(std::unique_ptr<HloModule> module,
                                          const std::string& input_pass_names) {
  HloPassPipeline transforms_pipeline{"transforms_pipeline"};
  for (const auto& pass_name :
       std::vector<std::string>(absl::StrSplit(input_pass_names, ','))) {
    auto it = pass_registry_.find(pass_name);
    if (it != pass_registry_.end()) {
      it->second(transforms_pipeline);
    } else {
      LOG(ERROR) << "Pass " << pass_name << " not found.";
    }
  }
  CHECK_OK(transforms_pipeline.Run(module.get(), {}));
  return module->ToString();
}

std::set<std::string> OptProvider::SupportedStages() { return {"hlo"}; }

// Hardware Independent passes are already registered in the constructor.
// Hardware Specific passes can be populated by respective hardware provider
// subclasses using this method.
void OptProvider::RegisterProviderPasses(HloModule& module) {}

std::string OptProvider::GetRegisteredPassNames() {
  return GetRegisteredPassNamesHelper(pass_registry_);
}

std::string OptProvider::GetRegisteredPassNamesHelper(
    const absl::flat_hash_map<
        std::string, std::function<void(HloPassPipeline&)>>& pass_registry_) {
  std::vector<std::string> names;
  names.reserve(pass_registry_.size());
  for (const auto& [name, pass_func] : pass_registry_) {
    names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  return absl::StrJoin(names, ",");
}

// Register Hardware-independent HLO passes here if you want the hlo-opt tool
// to be able to apply them.
void OptProvider::RegisterAllHardwareIndependentPasses() {
  // Dummy passes
  RegisterPass<FooToBarModulePass>();
  RegisterPass<BarToHelloModulePass>();
  // Hardware-independent HLO passes
  // go/keep-sorted start
  RegisterPass<AllGatherBroadcastReorder>();
  RegisterPass<AllReduceContiguous>();
  RegisterPass<AllReduceFolder>();
  RegisterPass<BatchDotSimplification>();
  RegisterPass<BroadcastCanonicalizer>();
  RegisterPass<CholeskyExpander>();
  RegisterPass<CollectiveQuantizer>();
  RegisterPass<ComparisonExpander>();
  RegisterPass<ConditionalCanonicalizer>();
  RegisterPass<ConvertMemoryPlacementToInternalAnnotations>();
  RegisterPass<ConvertMover>();
  RegisterPass<Convolution4DExpander>();
  RegisterPass<ConvolutionPredExpander>();
  RegisterPass<DotDecomposer>();
  RegisterPass<DynamicDimensionSimplifier>();
  RegisterPass<DynamicIndexSplitter>();
  RegisterPass<EighExpander>();
  RegisterPass<FlattenCallGraph>();
  RegisterPass<GatherSimplifier>();
  RegisterPass<HloConstantFolding>();
  RegisterPass<HloDCE>();
  RegisterPass<IndexedArrayAnalysisPrinterPass>();
  RegisterPass<LogisticExpander>();
  RegisterPass<OperandUpcaster>();
  RegisterPass<OptimizationBarrierExpander>();
  RegisterPass<OptimizeInputOutputBufferAlias>(true);
  RegisterPass<QrExpander>();
  RegisterPass<RealImagExpander>();
  RegisterPass<ReduceDecomposer>();
  RegisterPass<ReshapeDecomposer>();
  RegisterPass<ReshapeMover>();
  RegisterPass<ResultCaster>();
  RegisterPass<RngExpander>();
  RegisterPass<SimplifyFPConversions>();
  RegisterPass<SliceSinker>();
  RegisterPass<SortSimplifier>();
  RegisterPass<StableSortExpander>();
  RegisterPass<StochasticConvertDecomposer>();
  RegisterPass<SubByteNormalization>(SubByteNormalization::SET_ELEMENT_SIZE);
  RegisterPass<TreeReductionRewriter>();
  RegisterPass<TupleSimplifier>();
  RegisterPass<WhileLoopTripCountAnnotator>();
  RegisterPass<ZeroSizedHloElimination>();
  // go/keep-sorted end
  FloatSupport bf16_support(BF16);
  RegisterPass<FloatNormalization>(&bf16_support);
  auto cost_model = [](HloInstruction* conv) {
    // We need a cost model for CPUs. Currently, do nothing.
    return false;
  };
  RegisterPass<ConvolutionGroupConverter>(
      /*should_expand=*/[](HloInstruction* conv) { return true; }, cost_model,
      /*convert_batch_groups_only=*/true);
}

}  // namespace xla

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(transforms_opt_provider, {
  xla::OptProvider::RegisterForPlatform("transforms",
                                        std::make_unique<xla::OptProvider>());
});
