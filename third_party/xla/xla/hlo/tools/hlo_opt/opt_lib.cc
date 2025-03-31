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
#include <cstdint>
#include <functional>
#include <limits>
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
#include "xla/hlo/tools/tests/hlo_opt_test_only_passes.h"
#include "xla/hlo/transforms/add_original_value.h"
#include "xla/hlo/transforms/bfloat16_propagation.h"
#include "xla/hlo/transforms/collectives/all_gather_broadcast_reorder.h"
#include "xla/hlo/transforms/collectives/all_gather_combiner.h"
#include "xla/hlo/transforms/collectives/all_gather_cse.h"
#include "xla/hlo/transforms/collectives/all_reduce_combiner.h"
#include "xla/hlo/transforms/collectives/all_reduce_contiguous.h"
#include "xla/hlo/transforms/collectives/async_collective_creator.h"
#include "xla/hlo/transforms/collectives/collective_quantizer.h"
#include "xla/hlo/transforms/collectives/collective_transformation_reorderer.h"
#include "xla/hlo/transforms/collectives/collectives_schedule_linearizer.h"
#include "xla/hlo/transforms/collectives/convert_async_collectives_to_sync.h"
#include "xla/hlo/transforms/collectives/infeed_token_propagation.h"
#include "xla/hlo/transforms/collectives/while_loop_all_reduce_code_motion_setup.h"
#include "xla/hlo/transforms/convert_memory_placement_to_internal_annotations.h"
#include "xla/hlo/transforms/defuser.h"
#include "xla/hlo/transforms/despecializer.h"
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
#include "xla/hlo/transforms/expanders/rng_bit_generator_expander.h"
#include "xla/hlo/transforms/expanders/rng_expander.h"
#include "xla/hlo/transforms/expanders/stable_sort_expander.h"
#include "xla/hlo/transforms/expanders/stochastic_convert_decomposer.h"
#include "xla/hlo/transforms/host_offload_legalize.h"
#include "xla/hlo/transforms/host_offloader.h"
#include "xla/hlo/transforms/host_offloading_prepare.h"
#include "xla/hlo/transforms/literal_canonicalizer.h"
#include "xla/hlo/transforms/memory_space_propagation.h"
#include "xla/hlo/transforms/operand_upcaster.h"
#include "xla/hlo/transforms/sharding_format_picker.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/all_reduce_folder.h"
#include "xla/hlo/transforms/simplifiers/ar_crs_combiner.h"
#include "xla/hlo/transforms/simplifiers/batch_dot_simplification.h"
#include "xla/hlo/transforms/simplifiers/bfloat16_conversion_folding.h"
#include "xla/hlo/transforms/simplifiers/broadcast_canonicalizer.h"
#include "xla/hlo/transforms/simplifiers/conditional_canonicalizer.h"
#include "xla/hlo/transforms/simplifiers/convert_mover.h"
#include "xla/hlo/transforms/simplifiers/convert_operand_folder.h"
#include "xla/hlo/transforms/simplifiers/convolution_group_converter.h"
#include "xla/hlo/transforms/simplifiers/dot_dimension_merger.h"
#include "xla/hlo/transforms/simplifiers/dot_merger.h"
#include "xla/hlo/transforms/simplifiers/dynamic_dimension_simplifier.h"
#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/hlo/transforms/simplifiers/fusion_constant_sinking.h"
#include "xla/hlo/transforms/simplifiers/gather_simplifier.h"
#include "xla/hlo/transforms/simplifiers/hlo_computation_deduplicator.h"
#include "xla/hlo/transforms/simplifiers/hlo_constant_folding.h"
#include "xla/hlo/transforms/simplifiers/hlo_constant_splitter.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/hlo_element_type_converter.h"
#include "xla/hlo/transforms/simplifiers/hlo_memory_scheduler.h"
#include "xla/hlo/transforms/simplifiers/host_memory_transfer_asyncifier.h"
#include "xla/hlo/transforms/simplifiers/instruction_hoister.h"
#include "xla/hlo/transforms/simplifiers/optimize_input_output_buffer_alias.h"
#include "xla/hlo/transforms/simplifiers/reduce_window_rewriter.h"
#include "xla/hlo/transforms/simplifiers/reshape_mover.h"
#include "xla/hlo/transforms/simplifiers/result_caster.h"
#include "xla/hlo/transforms/simplifiers/root_instruction_sinker.h"
#include "xla/hlo/transforms/simplifiers/simplify_fp_conversions.h"
#include "xla/hlo/transforms/simplifiers/slice_sinker.h"
#include "xla/hlo/transforms/simplifiers/sort_simplifier.h"
#include "xla/hlo/transforms/simplifiers/sub_byte_normalization.h"
#include "xla/hlo/transforms/simplifiers/tree_reduction_rewriter.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/hlo/transforms/simplifiers/zero_sized_hlo_elimination.h"
#include "xla/hlo/transforms/while_loop_trip_count_annotator.h"
#include "xla/literal_pool.h"
#include "xla/service/buffer_value.h"
#include "xla/service/float_support.h"
#include "xla/service/platform_util.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

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

absl::StatusOr<std::string> OptProvider::BuildAndRunTransformPipeline(
    std::unique_ptr<HloModule> module, const std::string& input_pass_names) {
  HloPassPipeline transforms_pipeline{"transforms_pipeline"};
  for (const auto& pass_name :
       std::vector<std::string>(absl::StrSplit(input_pass_names, ','))) {
    auto it = pass_registry_.find(pass_name);
    if (it != pass_registry_.end()) {
      it->second(transforms_pipeline);
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Pass ", pass_name, " not found."));
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

////////////////////////////////////////////////////////////////////////////////
// Registration of Hardware-independent HLO Passes                            //
////////////////////////////////////////////////////////////////////////////////
void OptProvider::RegisterAllHardwareIndependentPasses() {
  // Dummy pass configs necessary for pass registration.
  FloatSupport* bfloat16_support = new FloatSupport(BF16, F32);
  auto size_fn = [](const BufferValue& buffer) {
    return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
  };
  LiteralPool* literal_pool = new LiteralPool();

  // Hardware-independent HLO passes
  // go/keep-sorted start
  RegisterPass<AddOriginalValue>();
  RegisterPass<AlgebraicSimplifier>(AlgebraicSimplifierOptions());
  RegisterPass<AllGatherBroadcastReorder>();
  RegisterPass<AllGatherCSE>();
  RegisterPass<AllGatherCombiner>(/*combine_threshold_in_bytes=*/1024,
                                  /*combine_threshold_count=*/1024,
                                  /*combine_by_dim=*/true);
  RegisterPass<AllReduceCombiner>(/*combine_threshold_in_bytes=*/1024,
                                  /*combine_threshold_count=*/1024);
  RegisterPass<AllReduceContiguous>();
  RegisterPass<AllReduceFolder>();
  RegisterPass<ArCrsCombiner>(/*num_spatial_partitions=*/0,
                              /*spmd_partition=*/1);
  RegisterPass<AssumeGatherIndicesInBoundRewriteToCopy>();
  RegisterPass<AsyncCollectiveCreator>(
      AsyncCollectiveCreator::CollectiveCreatorConfig());
  RegisterPass<BFloat16ConversionFolding>(
      /*bfloat16_support=*/bfloat16_support);
  RegisterPass<BFloat16MixedPrecisionRemoval>();
  RegisterPass<BFloat16Propagation>(/*bfloat16_support=*/bfloat16_support);
  RegisterPass<BatchDotSimplification>();
  RegisterPass<BroadcastCanonicalizer>();
  RegisterPass<CholeskyExpander>();
  RegisterPass<CollectiveQuantizer>();
  RegisterPass<CollectiveTransformationReorder>();
  RegisterPass<CollectivesScheduleLinearizer>();
  RegisterPass<ComparisonExpander>();
  RegisterPass<ConditionalCanonicalizer>();
  RegisterPass<ControlDepRemover>();
  RegisterPass<ConvertAsyncCollectivesToSync>();
  RegisterPass<ConvertMemoryPlacementToInternalAnnotations>();
  RegisterPass<ConvertMover>();
  RegisterPass<ConvertOperandFolding>();
  RegisterPass<Convolution4DExpander>();
  RegisterPass<ConvolutionGroupConverter>(
      /*should_expand=*/[](HloInstruction* conv) { return true; },
      /*is_cost_viable=*/[](HloInstruction* conv) { return true; },
      /*convert_batch_groups_only=*/false);
  RegisterPass<ConvolutionPredExpander>();
  RegisterPass<DeconstructReduceWindowToReduceBroadcast>();
  RegisterPass<Defuser>();
  RegisterPass<Despecializer>();
  RegisterPass<DotDecomposer>();
  RegisterPass<DotDimensionMerger>();
  RegisterPass<DotMerger>(
      /*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  RegisterPass<DynamicDimensionSimplifier>();
  RegisterPass<DynamicIndexSplitter>();
  RegisterPass<EighExpander>();
  RegisterPass<FlattenCallGraph>();
  RegisterPass<FloatNormalization>(/*float_support=*/bfloat16_support);
  RegisterPass<FusionConstantSinking>();
  RegisterPass<GatherSimplifier>();
  RegisterPass<HloComputationDeduplicator>();
  RegisterPass<HloConstantFolding>();
  RegisterPass<HloConstantSplitter>();
  RegisterPass<HloDCE>();
  RegisterPass<HloDescheduler>();
  RegisterPass<HloElementTypeConverter>(/*eliminate_type=*/BF16,
                                        /*replace_with_type=*/F32);
  RegisterPass<HloMemoryScheduler>(/*size_fn*/ size_fn);
  RegisterPass<HloTrivialScheduler>();
  RegisterPass<HostMemoryTransferAsyncifier>(/*host_memory_space_color=*/5);
  RegisterPass<HostOffloadLegalize>(
      /*host_memory_space_color=*/5, /*after_layout=*/false);
  RegisterPass<HostOffloader>();
  RegisterPass<HostOffloadingPrepare>(
      /*rewrite=*/HostOffloadingPrepare::Rewrite::kElideMoveToHost);
  RegisterPass<IndexedArrayAnalysisPrinterPass>();
  RegisterPass<InfeedTokenPropagation>();
  RegisterPass<InstructionHoister>();
  RegisterPass<LiteralCanonicalizer>(
      /*literal_pool=*/literal_pool, /*min_size_bytes=*/0);
  RegisterPass<LogisticExpander>();
  RegisterPass<MemorySpacePropagation>();
  RegisterPass<OperandUpcaster>();
  RegisterPass<OptimizationBarrierExpander>();
  RegisterPass<OptimizeInputOutputBufferAlias>(true);
  RegisterPass<QrExpander>();
  RegisterPass<RealImagExpander>();
  RegisterPass<ReduceDecomposer>();
  RegisterPass<ReduceWindowRewriter>(/*base_length=*/16);
  RegisterPass<ReorderConvertReduceAdd>();
  RegisterPass<ReorderReduceTranspose>();
  RegisterPass<ReshapeDecomposer>();
  RegisterPass<ReshapeMover>();
  RegisterPass<ResultCaster>();
  RegisterPass<RngBitGeneratorExpander>(RandomAlgorithm::RNG_THREE_FRY);
  RegisterPass<RngExpander>();
  RegisterPass<RootInstructionSinker>();
  RegisterPass<ShardingFormatPicker>(
      /*sharding_type=*/ShardingFormatPicker::ShardingType::kBestEffortV2);
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
  // Excluded passes:
  // 1. HloRematerialization : The `RegisterPass` template applies
  //   `std::as_const` to pass arguments, but the `HloRematerialization`
  //   constructor requires a non-const lvalue reference for 2nd
  //   argument(`RematerializationSizes`). For now, we don't want to add any
  //   pass specific customization to the `RegisterPass`.

  // Dummy passes for unit-testing the `hlo-opt` tool itself.
  RegisterPass<test_only::FooToBarModulePass>();
  RegisterPass<test_only::BarToHelloModulePass>();

  // Test-only passes exposing behavior that isn't easily testable through
  // standard passes, e.g. internal or config-dependent behavior.
  RegisterPass<test_only::AlgebraicSimplifierWithOnednnEnabled>();
  RegisterPass<test_only::XlaBuilderTestPass>();
}

}  // namespace xla

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(transforms_opt_provider, {
  xla::OptProvider::RegisterForPlatform("transforms",
                                        std::make_unique<xla::OptProvider>());
});
