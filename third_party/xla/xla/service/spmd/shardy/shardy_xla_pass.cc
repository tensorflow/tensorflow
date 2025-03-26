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

#include "xla/service/spmd/shardy/shardy_xla_pass.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/file_utils.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#include "re2/re2.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/layout.h"
#include "xla/map_util.h"
#include "xla/service/computation_layout.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/sdy_round_trip/pipelines.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_export.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_import.h"
#include "xla/service/spmd/shardy/utils.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"

namespace xla {
namespace sdy {

namespace {

std::string uniqueModuleName(const HloModule& module) {
  std::string result;
  absl::StrAppendFormat(&result, "module_%04d", module.unique_id());
  if (!module.name().empty()) {
    absl::StrAppend(&result, ".", module.name());
  }
  return result;
}

// Creates a vector of HloComputation, which is used to replace the old
// computations in the HloModule. It is adapted from CreateAndSanitizeFromProto
// in internal xla/tests/fuzzing/hlo_fuzzer_utils.cc.
absl::Status createFromProtoAndReplaceComputations(
    HloModule* module, const HloModuleProto& proto) {
  absl::flat_hash_map<int64_t, HloComputation*> idToComputation;
  std::vector<std::unique_ptr<HloComputation>> computations;
  HloComputation* entryComputation = nullptr;

  // Create HLO computations from proto.
  for (const HloComputationProto& computationProto : proto.computations()) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloComputation> computation,
        HloComputation::CreateFromProto(computationProto, idToComputation));
    CHECK_NE(computation.get(), nullptr);
    const int64_t computationId = computationProto.id();
    CHECK_NE(computationId, -1);
    CHECK(!ContainsKey(idToComputation, computationId));
    idToComputation[computationId] = computation.get();
    if (computationId == proto.entry_computation_id()) {
      CHECK_EQ(entryComputation, nullptr);
      entryComputation = computation.get();
    }
    computations.push_back(std::move(computation));
  }

  CHECK_NE(entryComputation, nullptr);

  // Sort the computations by their proto id.
  absl::c_sort(computations, [](const std::unique_ptr<HloComputation>& a,
                                const std::unique_ptr<HloComputation>& b) {
    return a->unique_id() < b->unique_id();
  });
  // Add computations to the module. Make computation and instruction names
  // unique. Re-assign unique IDs to the computations and instructions.
  for (std::unique_ptr<HloComputation>& computation : computations) {
    HloComputation* newComputation =
        module->AddComputationAndUnifyNamesAndIds(std::move(computation),
                                                  /*is_entry=*/false);
    if (newComputation == entryComputation) {
      module->ReplaceEntryComputation(newComputation);
    }
  }

  // Remove the old computations, which are currently dead.
  CHECK_OK(HloDCE().Run(module));

  return absl::OkStatus();
}

// A map from the original shape indices of each parameter to the flattened
// parameter number.
using OriginalParamIndexToFlattenedNum =
    std::vector<absl::flat_hash_map<ShapeIndex, int64_t>>;

int64_t getFlattenedParamNumber(
    const OriginalParamIndexToFlattenedNum& originalParamIndexToFlattenedNum,
    int64_t paramNumber, const ShapeIndex& paramIndex) {
  return originalParamIndexToFlattenedNum[paramNumber].at(paramIndex);
}

// Returns a map from the original shape indices of each parameter to the
// flattened parameter number.
OriginalParamIndexToFlattenedNum getOriginalParamIndexToFlattenedNum(
    HloModule* hloModule) {
  OriginalParamIndexToFlattenedNum originalParamIndexToFlattened;
  HloComputation* entryComputation = hloModule->entry_computation();
  originalParamIndexToFlattened.reserve(entryComputation->num_parameters());
  int64_t paramNumber = 0;
  for (HloInstruction* paramInstruction :
       entryComputation->parameter_instructions()) {
    auto& paramMap = originalParamIndexToFlattened.emplace_back();
    ShapeUtil::ForEachLeafShape(paramInstruction->shape(),
                                [&](const Shape&, const ShapeIndex& index) {
                                  paramMap[index] = paramNumber++;
                                });
  }
  return originalParamIndexToFlattened;
}

// Flattens the given `shape`.
Shape getFlattenedShape(const Shape& shape) {
  std::vector<Shape> flattenedShapes;
  ShapeUtil::ForEachLeafShape(
      shape, [&](const Shape& subShape, const ShapeIndex& index) {
        flattenedShapes.push_back(subShape);
      });
  return ShapeUtil::MakeMaybeTupleShape(flattenedShapes);
}

// Get the flattened version of a computation layout.
//
// If `useTupleArgs` is true, the returned flattened computation layout will
// account for the flattened parameters being wrapped in a single tuple
// parameter.
ComputationLayout getFlattenedComputationLayout(
    const ComputationLayout& computationLayout, bool useTupleArgs) {
  if (!computationLayout.AnyLayoutSet()) {
    return computationLayout;
  }
  // Flatten the result layout.
  ComputationLayout flattenedComputationLayout = ComputationLayout(
      ShapeLayout(getFlattenedShape(computationLayout.result_shape())));

  // Flatten the parameter layout.
  // When `useTupleArgs` is true, we will use a single flattened tuple for the
  // params. So this single Shape will hold the layout of all params.
  Shape tupleShape;
  tupleShape.set_element_type(PrimitiveType::TUPLE);
  for (int64_t i = 0; i != computationLayout.parameter_count(); ++i) {
    ShapeUtil::ForEachLeafShape(
        computationLayout.parameter_shape(i),
        [&](const Shape& subShape, const ShapeIndex& index) {
          if (useTupleArgs) {
            *tupleShape.add_tuple_shapes() = subShape;
          } else {
            flattenedComputationLayout.add_parameter_layout(
                ShapeLayout(subShape));
          }
        });
  }
  if (useTupleArgs) {
    flattenedComputationLayout.add_parameter_layout(ShapeLayout(tupleShape));
  }
  return flattenedComputationLayout;
}

// Returns the corresponding parameter number and index dependent on whether the
// returned HLO module will have a single tuple level tuple.
// - If `useTupleArgs` is true, then the param number will be 0 since there is
//   only one tuple param (holding the flattened params), with the param index
//   holding the location in that single tuple.
// - If `useTupleArgs` is false, then the param index will be ShapeIndex() since
//   all params are fully flattened with no tuple, and param number will be the
//   location in the flattened params.
std::pair<int64_t, ShapeIndex> getFlattenedParamNumberAndIndex(
    const OriginalParamIndexToFlattenedNum& originalParamIndexToFlattenedNum,
    int64_t parameterNumber, const ShapeIndex& parameterIndex,
    bool useTupleArgs) {
  int64_t flattenedIndex = getFlattenedParamNumber(
      originalParamIndexToFlattenedNum, parameterNumber, parameterIndex);
  if (useTupleArgs) {
    return {0, ShapeIndex{flattenedIndex}};
  }
  return {flattenedIndex, ShapeIndex()};
}

// Get the flattened version of a input-output alias config.
//
// If `useTupleArgs` is true, the returned flattened alias config will
// account for the flattened parameters being wrapped in a single tuple
// parameter.
HloInputOutputAliasConfig getFlattenedInputOutputAliasConfig(
    const HloInputOutputAliasConfig& inputOutputAliasConfig,
    const OriginalParamIndexToFlattenedNum& originalParamIndexToFlattenedNum,
    bool useTupleArgs) {
  HloInputOutputAliasConfig flattenedInputOutputAliasConfig(
      getFlattenedShape(inputOutputAliasConfig.shape()));
  int64_t resultIndex = 0;
  ShapeUtil::ForEachLeafShape(
      inputOutputAliasConfig.shape(),
      [&](const Shape&, const ShapeIndex& index) {
        if (const std::optional<HloInputOutputAliasConfig::Alias>& alias =
                inputOutputAliasConfig.GetAliasedParameter(index)) {
          auto [paramNumber, paramIndex] = getFlattenedParamNumberAndIndex(
              originalParamIndexToFlattenedNum, alias->parameter_number,
              alias->parameter_index, useTupleArgs);
          CHECK_OK(flattenedInputOutputAliasConfig.SetUpAlias(
              flattenedInputOutputAliasConfig.shape().IsTuple()
                  ? ShapeIndex{resultIndex}
                  : ShapeIndex(),
              paramNumber, paramIndex, alias->kind));
        }
        ++resultIndex;
      });
  return flattenedInputOutputAliasConfig;
}

// Get the flattened version of a buffer donors config.
//
// If `useTupleArgs` is true, the returned flattened buffer donors config will
// account for the flattened parameters being wrapped in a single tuple
// parameter.
HloBufferDonorConfig getFlattenedBufferDonorsConfig(
    const HloBufferDonorConfig& bufferDonorsConfig,
    const OriginalParamIndexToFlattenedNum& originalParamIndexToFlattenedNum,
    bool useTupleArgs) {
  HloBufferDonorConfig flattenedBufferDonorsConfig;
  for (const HloBufferDonorConfig::BufferDonor& bufferDonor :
       bufferDonorsConfig.buffer_donor()) {
    auto [paramNumber, paramIndex] = getFlattenedParamNumberAndIndex(
        originalParamIndexToFlattenedNum, bufferDonor.param_number,
        bufferDonor.param_index, useTupleArgs);
    CHECK_OK(
        flattenedBufferDonorsConfig.AddBufferDonor(paramNumber, paramIndex));
  }
  return flattenedBufferDonorsConfig;
}

// Remove `attributeNames` from the frontend attributes of `hloModule`.
void removeFrontendAttributes(HloModule* hloModule,
                              mlir::ArrayRef<mlir::StringRef> attributeNames) {
  FrontendAttributes feAttrs = hloModule->frontend_attributes();
  auto* map = feAttrs.mutable_map();
  for (const auto& attributeName : attributeNames) {
    map->erase(attributeName);
  }
  hloModule->set_frontend_attributes(feAttrs);
}

}  // namespace

absl::StatusOr<bool> ShardyXLA::Run(
    HloModule* hloModule,
    const absl::flat_hash_set<absl::string_view>& executionThreads) {
  LOG(INFO) << "Using Shardy for XLA SPMD propagation.";

  // HLO -> StableHLO
  auto mlirContext = std::make_unique<mlir::MLIRContext>();
  loadAllRequiredDialects(mlirContext.get());
  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> mlirModule,
      xla::ConvertHloToStablehlo(*mlirContext.get(), hloModule));

  const DebugOptions& debugOptions = hloModule->config().debug_options();
  std::string shardyDir = debugOptions.xla_dump_to();

  if (!shardyDir.empty() &&
      !RE2::PartialMatch(name(), debugOptions.xla_dump_hlo_pass_re())) {
    shardyDir.clear();
  }

  if (shardyDir == "sponge") {
    shardyDir = getenv("TEST_UNDECLARED_OUTPUTS_DIR");
    if (shardyDir.empty()) {
      LOG(WARNING) << "\"sponge\" specified as dump directory but "
                      "TEST_UNDECLARED_OUTPUTS_DIR is not set!";
    } else {
      LOG(INFO) << "Shardy dump directory is sponge on undeclared outputs dir: "
                << shardyDir;
    }
  }

  if (!shardyDir.empty()) {
    shardyDir =
        tsl::io::JoinPath(shardyDir, "shardy", uniqueModuleName(*hloModule));
    LOG(INFO) << "Using Shardy output directory: " << shardyDir;
  }
  TF_RETURN_IF_ERROR(tsl::Env::Default()->RecursivelyCreateDir(shardyDir));
  // MLIR pipeline: (1) import, (2) Shardy, and (3) export.

  bool enableVerifier = false;
#ifndef NDEBUG
  enableVerifier = true;
#endif

  mlir::PassManager pm(mlirContext.get());
  pm.enableVerifier(enableVerifier);
  pm.addPass(mlir::sdy::createSaveModuleOpPass(shardyDir,
                                               "sdy_module_before_xla_import"));
  bool useTupleArgs = false;
  mlir::DictionaryAttr moduleFrontendAttrs = getFrontendAttrs(*mlirModule);
  if (hasKey(moduleFrontendAttrs, kUseTupleArgs)) {
    useTupleArgs = true;
    removeFrontendAttribute(*mlirModule, kUseTupleArgs);
  }

  if (hasKey(moduleFrontendAttrs, kImportMhloShardings)) {
    removeFrontendAttribute(*mlirModule, kImportMhloShardings);
    auto spanToArrayRef = [](absl::Span<const bool> span) {
      return mlir::ArrayRef<bool>(span.data(), span.size());
    };

    addStablehloImportPipeline(
        pm,
        spanToArrayRef(hloModule->config()
                           .allow_spmd_sharding_propagation_to_parameters()),
        spanToArrayRef(
            hloModule->config().allow_spmd_sharding_propagation_to_output()));
  } else {
    // This is the default path.
    addSdyRoundTripImportPipeline(pm);
  }

  // Store the entry computation layout, input-output alias config, and buffer
  // donors, which will be restored in the end, since MLIR does not preserve
  // those properties.
  ComputationLayout flattenedEntryComputationLayout =
      getFlattenedComputationLayout(hloModule->entry_computation_layout(),
                                    useTupleArgs);
  OriginalParamIndexToFlattenedNum originalParamIndexToFlattenedNum =
      getOriginalParamIndexToFlattenedNum(hloModule);
  HloInputOutputAliasConfig flattenedInputOutputAliasConfig =
      getFlattenedInputOutputAliasConfig(hloModule->input_output_alias_config(),
                                         originalParamIndexToFlattenedNum,
                                         useTupleArgs);
  HloBufferDonorConfig flattenedBufferDonorsConfig =
      getFlattenedBufferDonorsConfig(hloModule->buffer_donor_config(),
                                     originalParamIndexToFlattenedNum,
                                     useTupleArgs);

  if (runSdyShardingPropagation) {
    // NOTE: if we are using auto-spmd, we will use conservative propagation
    // since the TOAST cost model cannot account for split axes or padding.
    mlir::sdy::PropagationOptions options = defaultOptions;
    options.dumpDirectory = shardyDir;
    options.conservativePropagation = hloModule->use_auto_spmd_partitioning();
    mlir::sdy::addPropagationPipeline(pm, options);
  }
  addStablehloExportPipeline(pm);
  pm.addPass(mlir::sdy::createSaveModuleOpPass(shardyDir,
                                               "sdy_module_after_xla_export"));
  tsl::StatusScopedDiagnosticHandler diagnosticHandler(mlirContext.get());
  TF_RETURN_IF_ERROR(diagnosticHandler.consumeStatus(pm.run(*mlirModule)));

  // StableHlo -> HLO
  HloProto hloProto;
  TF_RETURN_IF_ERROR(ConvertStablehloWithManyArgsToHloProto(
      *mlirModule, &hloProto, useTupleArgs));
  TF_RETURN_IF_ERROR(
      createFromProtoAndReplaceComputations(hloModule, hloProto.hlo_module()));

  // If the module returns a single tensor as result with sharding,
  // ConvertMlirHloToHlo still generates the tuple and get-tuple-element as the
  // root instructions. We use the TupleSimplifier as a temporary solution.
  CHECK_OK(TupleSimplifier().Run(hloModule));

  // Restore entry computation layout.
  *hloModule->mutable_entry_computation_layout() =
      flattenedEntryComputationLayout;
  hloModule->set_input_output_alias_config(
      std::move(flattenedInputOutputAliasConfig));
  hloModule->set_buffer_donor_config(std::move(flattenedBufferDonorsConfig));

  // Shardy currently propagates shardings to parameters and root
  // instructions. Hence, we specify `true` for update_output_layout and
  // update_parameters_layout.
  TF_RETURN_IF_ERROR(
      hlo_sharding_util::CanonicalizeLayoutAfterShardingPropagation(
          hloModule, /*update_output_layout=*/{true},
          /*update_parameters_layout=*/{true}));

  // We don't fully replace the HLO module, so it will continue to have the
  // temporary frontend attributes. So clean them up as XLA won't need them.
  removeFrontendAttributes(hloModule, {kUseTupleArgs, kMeshesRoundTripAttr});

  return true;
}

}  // namespace sdy
}  // namespace xla
