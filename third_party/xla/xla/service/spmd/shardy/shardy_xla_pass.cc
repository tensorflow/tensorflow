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
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/file_utils.h"
#include "shardy/dialect/sdy/transforms/common/partitioner_stage.h"
#include "shardy/dialect/sdy/transforms/common/propagation_options.h"
#include "shardy/dialect/sdy/transforms/export/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#include "re2/re2.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/stack_frames.h"
#include "xla/hlo/pass/hlo_pass_context.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/pass/hlo_pass_options.h"
#include "xla/hlo/pass/hlo_pass_registry.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/map_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
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
#include "xla/status_macros.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/tsl/platform/env.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"
#include "tsl/platform/platform.h"

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
    ABSL_ASSIGN_OR_RETURN(
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

  ABSL_ASSIGN_OR_RETURN(StackFrames stack_frames,
                   StackFrames::FromProto(proto.stack_frame_index()));
  module->set_stack_frames(std::move(stack_frames));

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
  ShapeUtil::ForEachLeafShape(shape,
                              [&](const Shape& subShape, const ShapeIndex&) {
                                flattenedShapes.push_back(subShape);
                              });
  return ShapeUtil::MakeValidatedMaybeTupleShape(flattenedShapes).value();
}

// Get the flattened version of a computation layout.
//
// If `useTupleArgs` is true, the returned flattened computation layout will
// account for the flattened parameters being wrapped in a single tuple
// parameter.
ComputationLayout getFlattenedComputationLayout(
    const ComputationLayout& computationLayout, bool useTupleArgs) {
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
        [&](const Shape& subShape, const ShapeIndex&) {
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

// Updates the entry computation layout of `hloModule` after Shardy device-local
// code generation by copying layouts from `oldLayout` to the new local program
// shape and canonicalizing layouts for any parameter or output whose shape
// changed during partitioning.
absl::Status updateEntryComputationLayout(HloModule* hloModule,
                                          const ComputationLayout& oldLayout) {
  ProgramShape newProgramShape =
      hloModule->entry_computation()->ComputeProgramShape();

  // Record which parameters changed shape and copy the original parameter
  // layouts onto the new local parameter shapes.
  mlir::SmallVector<bool> updateParametersLayout(
      newProgramShape.parameters_size());
  for (int64_t i = 0; i < newProgramShape.parameters_size(); ++i) {
    const Shape& oldParamShape = oldLayout.parameter_shape(i);
    Shape* newParamShape = newProgramShape.mutable_parameters(i);
    updateParametersLayout[i] =
        !Shape::Equal().IgnoreLayout()(oldParamShape, *newParamShape);
    ABSL_RETURN_IF_ERROR(
        LayoutUtil::CopyLayoutBetweenShapes(oldParamShape, newParamShape));
  }

  // Record which outputs changed shape and copy the original result layout onto
  // the new local result shape.
  const Shape& oldResultShape = oldLayout.result_shape();
  Shape* newResultShape = newProgramShape.mutable_result();
  mlir::SmallVector<bool> updateOutputLayout;
  if (newResultShape->IsTuple()) {
    updateOutputLayout.reserve(newResultShape->tuple_shapes().size());
    for (int64_t i = 0; i < newResultShape->tuple_shapes().size(); ++i) {
      updateOutputLayout.push_back(!Shape::Equal().IgnoreLayout()(
          oldResultShape.tuple_shapes(i), newResultShape->tuple_shapes(i)));
    }
  } else {
    updateOutputLayout.push_back(
        !Shape::Equal().IgnoreLayout()(oldResultShape, *newResultShape));
  }
  ABSL_RETURN_IF_ERROR(
      LayoutUtil::CopyLayoutBetweenShapes(oldResultShape, newResultShape));
  *hloModule->mutable_entry_computation_layout() =
      ComputationLayout(newProgramShape, /*ignore_layouts=*/false);

  // Give the changed parameter and output shapes canonical layouts.
  return hlo_sharding_util::CanonicalizeLayoutAfterShardingPropagation(
      hloModule, updateOutputLayout, updateParametersLayout);
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
    const Shape& flattenedOutputShape,
    const OriginalParamIndexToFlattenedNum& originalParamIndexToFlattenedNum,
    bool useTupleArgs) {
  HloInputOutputAliasConfig flattenedInputOutputAliasConfig(
      flattenedOutputShape);
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

std::string getShardyDirIfShouldDump(const DebugOptions& debugOptions,
                                     absl::string_view passName,
                                     bool isShardyVerbose) {
  std::string shardyDir = debugOptions.xla_dump_to();
  if (shardyDir.empty() || isShardyVerbose) {
    return shardyDir;
  }
  absl::string_view regex = debugOptions.xla_dump_hlo_pass_re();
  if (regex.empty() || !RE2::PartialMatch(passName, regex)) {
    return "";
  }
  return shardyDir;
}

std::optional<mlir::RankedTensorType> toBoundedStaticType(mlir::Type type) {
  auto rankedTy = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!rankedTy || rankedTy.hasStaticShape()) {
    return std::nullopt;
  }
  std::optional<mlir::ArrayRef<int64_t>> bounds;
  if (auto ext = mlir::dyn_cast_or_null<mlir::stablehlo::TypeExtensionsAttr>(
          rankedTy.getEncoding())) {
    bounds = ext.getBounds();
  } else if (auto ext = mlir::dyn_cast_or_null<mlir::mhlo::TypeExtensionsAttr>(
                 rankedTy.getEncoding())) {
    bounds = ext.getBounds();
  }
  if (!bounds.has_value() ||
      bounds->size() != static_cast<size_t>(rankedTy.getRank())) {
    return std::nullopt;
  }
  mlir::SmallVector<int64_t> staticShape(rankedTy.getRank());
  for (int64_t d = 0; d < rankedTy.getRank(); ++d) {
    staticShape[d] =
        rankedTy.isDynamicDim(d) ? (*bounds)[d] : rankedTy.getDimSize(d);
    if (staticShape[d] == mlir::ShapedType::kDynamic || staticShape[d] < 0) {
      return std::nullopt;
    }
  }
  return mlir::RankedTensorType::get(staticShape, rankedTy.getElementType());
}

HloSharding getValueLeafHloSharding(mlir::Value val) {
  if (mlir::Operation* defOp = val.getDefiningOp()) {
    if (auto shardingAttr =
            defOp->getAttrOfType<mlir::StringAttr>(kXlaShardingAttr)) {
      HloSharding sharding = xla::sdy::parseShardingFromString(shardingAttr);
      if (sharding.IsTuple()) {
        uint64_t idx = mlir::cast<mlir::OpResult>(val).getResultNumber();
        if (idx < sharding.tuple_elements().size()) {
          return sharding.tuple_elements()[idx];
        }
      } else {
        return sharding;
      }
    }
  } else if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
    mlir::Operation* parentOp = blockArg.getOwner()->getParentOp();
    if (auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(parentOp)) {
      if (auto shardingAttr = funcOp.getArgAttrOfType<mlir::StringAttr>(
              blockArg.getArgNumber(), kXlaShardingAttr)) {
        return xla::sdy::parseShardingFromString(shardingAttr);
      }
    } else if (parentOp) {
      if (auto shardingAttr =
              parentOp->getAttrOfType<mlir::StringAttr>(kXlaShardingAttr)) {
        HloSharding sharding = xla::sdy::parseShardingFromString(shardingAttr);
        if (sharding.IsTuple() &&
            blockArg.getArgNumber() < sharding.tuple_elements().size()) {
          return sharding.tuple_elements()[blockArg.getArgNumber()];
        }
      }
    }
  }
  return HloSharding::Replicate();
}

mlir::Value repackTupleValue(mlir::OpBuilder& builder, mlir::Location loc,
                             mlir::ValueRange& flatValues, mlir::Type type,
                             std::vector<HloSharding>& leafShardings) {
  auto tupleType = mlir::dyn_cast<mlir::TupleType>(type);
  if (!tupleType) {
    CHECK(!flatValues.empty());
    mlir::Value val = flatValues.front();
    flatValues = flatValues.drop_front();
    leafShardings.push_back(getValueLeafHloSharding(val));
    return val;
  }
  mlir::SmallVector<mlir::Value> subValues;
  subValues.reserve(tupleType.size());
  size_t startLeafIdx = leafShardings.size();
  for (mlir::Type childType : tupleType.getTypes()) {
    subValues.push_back(
        repackTupleValue(builder, loc, flatValues, childType, leafShardings));
  }
  auto tupleOp = mlir::stablehlo::TupleOp::create(builder, loc, subValues);
  std::vector<HloSharding> tupleLeafShardings(
      leafShardings.begin() + startLeafIdx, leafShardings.end());
  if (!tupleLeafShardings.empty()) {
    tupleOp->setAttr(
        kXlaShardingAttr,
        builder.getStringAttr(
            HloSharding::FlatTuple(std::move(tupleLeafShardings)).ToString()));
  }
  return tupleOp.getResult();
}

void prepareBoundedDynamicTypesAndTupleCustomCalls(
    mlir::ModuleOp mlirModule, bool& hasBoundedDynamicTypes,
    bool& hasTupleCustomCallOperands) {
  mlir::MLIRContext* ctx = mlirModule->getContext();
  for (auto funcOp : mlirModule.getOps<mlir::func::FuncOp>()) {
    bool sigChanged = false;
    mlir::SmallVector<mlir::Type> newInTypes(funcOp.getArgumentTypes().begin(),
                                             funcOp.getArgumentTypes().end());
    mlir::SmallVector<mlir::Type> newOutTypes(funcOp.getResultTypes().begin(),
                                              funcOp.getResultTypes().end());
    for (auto [i, argTy] : llvm::enumerate(newInTypes)) {
      if (auto staticTy = toBoundedStaticType(argTy)) {
        funcOp.setArgAttr(i, kOrigTypeAttr, mlir::TypeAttr::get(argTy));
        newInTypes[i] = *staticTy;
        if (!funcOp.isExternal()) {
          funcOp.getArgument(i).setType(*staticTy);
        }
        sigChanged = true;
        hasBoundedDynamicTypes = true;
      }
    }
    for (auto [i, resTy] : llvm::enumerate(newOutTypes)) {
      if (auto staticTy = toBoundedStaticType(resTy)) {
        funcOp.setResultAttr(i, kOrigTypeAttr, mlir::TypeAttr::get(resTy));
        newOutTypes[i] = *staticTy;
        sigChanged = true;
        hasBoundedDynamicTypes = true;
      }
    }
    if (sigChanged) {
      funcOp.setFunctionType(
          mlir::FunctionType::get(ctx, newInTypes, newOutTypes));
    }
    if (!funcOp.isExternal()) {
      funcOp.front().walk([&](mlir::Operation* op) {
        bool opResChanged = false;
        mlir::SmallVector<mlir::Attribute> origResAttrs;
        for (auto [resIdx, res] : llvm::enumerate(op->getResults())) {
          if (auto staticTy = toBoundedStaticType(res.getType())) {
            if (!opResChanged) {
              origResAttrs.assign(op->getNumResults(),
                                  mlir::UnitAttr::get(ctx));
            }
            origResAttrs[resIdx] = mlir::TypeAttr::get(res.getType());
            res.setType(*staticTy);
            opResChanged = true;
            hasBoundedDynamicTypes = true;
          }
        }
        if (opResChanged) {
          op->setAttr(kOrigResultTypesAttr,
                      mlir::ArrayAttr::get(ctx, origResAttrs));
        }

        if (auto customCall =
                mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op)) {
          if (llvm::any_of(customCall.getInputs(), [](mlir::Value opnd) {
                return mlir::isa<mlir::TupleType>(opnd.getType());
              })) {
            mlir::SmallVector<mlir::Attribute> origOpndTypes;
            origOpndTypes.reserve(customCall.getNumOperands());
            for (mlir::Type ty : customCall.getOperandTypes()) {
              origOpndTypes.push_back(mlir::TypeAttr::get(ty));
            }
            customCall->setAttr(kOrigOperandTypesAttr,
                                mlir::ArrayAttr::get(ctx, origOpndTypes));
            hasTupleCustomCallOperands = true;
          }
        }

        if (op->getNumRegions() > 0) {
          bool regArgsChanged = false;
          mlir::SmallVector<mlir::Attribute> perRegionAttrs;
          perRegionAttrs.reserve(op->getNumRegions());
          for (mlir::Region& region : op->getRegions()) {
            mlir::SmallVector<mlir::Attribute> regionArgAttrs;
            bool thisRegionChanged = false;
            size_t flatIdx = 0;
            for (mlir::Block& block : region) {
              for (mlir::BlockArgument blockArg : block.getArguments()) {
                if (auto staticTy = toBoundedStaticType(blockArg.getType())) {
                  if (!thisRegionChanged) {
                    regionArgAttrs.assign(flatIdx, mlir::UnitAttr::get(ctx));
                    thisRegionChanged = true;
                  }
                  regionArgAttrs.push_back(
                      mlir::TypeAttr::get(blockArg.getType()));
                  blockArg.setType(*staticTy);
                  regArgsChanged = true;
                  hasBoundedDynamicTypes = true;
                } else if (thisRegionChanged) {
                  regionArgAttrs.push_back(mlir::UnitAttr::get(ctx));
                }
                ++flatIdx;
              }
            }
            perRegionAttrs.push_back(mlir::ArrayAttr::get(ctx, regionArgAttrs));
          }
          if (regArgsChanged) {
            op->setAttr(kOrigRegionArgTypesAttr,
                        mlir::ArrayAttr::get(ctx, perRegionAttrs));
          }
        }
      });
    }
  }
}

void restoreBoundedDynamicTypesAndTupleCustomCalls(
    mlir::ModuleOp mlirModule, bool hasBoundedDynamicTypes,
    bool hasTupleCustomCallOperands) {
  mlir::MLIRContext* ctx = mlirModule->getContext();
  if (hasBoundedDynamicTypes) {
    for (auto funcOp : mlirModule.getOps<mlir::func::FuncOp>()) {
      mlir::SmallVector<mlir::Type> newInTypes(
          funcOp.getArgumentTypes().begin(), funcOp.getArgumentTypes().end());
      mlir::SmallVector<mlir::Type> newOutTypes(funcOp.getResultTypes().begin(),
                                                funcOp.getResultTypes().end());
      bool sigChanged = false;
      for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
        if (auto origTy =
                funcOp.getArgAttrOfType<mlir::TypeAttr>(i, kOrigTypeAttr)) {
          newInTypes[i] = origTy.getValue();
          if (!funcOp.isExternal()) {
            funcOp.getArgument(i).setType(origTy.getValue());
          }
          funcOp.removeArgAttr(i, kOrigTypeAttr);
          sigChanged = true;
        }
      }
      if (!funcOp.isExternal()) {
        funcOp.front().walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation*
                                                               op) {
          if (auto origRes =
                  op->getAttrOfType<mlir::ArrayAttr>(kOrigResultTypesAttr)) {
            for (auto [i, attr] : llvm::enumerate(origRes)) {
              if (i < op->getNumResults()) {
                if (auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr)) {
                  mlir::Type origTy = typeAttr.getValue();
                  if (toBoundedStaticType(origTy) ==
                      op->getResult(i).getType()) {
                    op->getResult(i).setType(origTy);
                  }
                }
              }
            }
            op->removeAttr(kOrigResultTypesAttr);
          } else if (op->getNumOperands() == 1 && op->getNumResults() == 1) {
            bool isBoundaryShapeOp = false;
            if (auto customCall =
                    mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op)) {
              mlir::StringRef target = customCall.getCallTargetName();
              isBoundaryShapeOp =
                  target == kSPMDFullToShardShapeCallTargetName ||
                  target == kSPMDShardToFullShapeCallTargetName ||
                  target == kGlobalToLocalShapeCallTargetName ||
                  target == kLocalToGlobalShapeCallTargetName;
            }
            if (!isBoundaryShapeOp) {
              if (auto opndTy = mlir::dyn_cast<mlir::RankedTensorType>(
                      op->getOperand(0).getType());
                  opndTy && !opndTy.hasStaticShape() &&
                  toBoundedStaticType(opndTy) == op->getResult(0).getType()) {
                op->getResult(0).setType(opndTy);
              }
            }
          }

          if (auto origRegArgs =
                  op->getAttrOfType<mlir::ArrayAttr>(kOrigRegionArgTypesAttr)) {
            for (auto [regIdx, regAttr] : llvm::enumerate(op->getRegions())) {
              if (regIdx >= origRegArgs.size()) {
                break;
              }
              auto perBlockArgs =
                  mlir::dyn_cast<mlir::ArrayAttr>(origRegArgs[regIdx]);
              if (!perBlockArgs) {
                continue;
              }
              size_t argIdx = 0;
              for (mlir::Block& block : regAttr) {
                for (mlir::BlockArgument blockArg : block.getArguments()) {
                  if (argIdx < perBlockArgs.size()) {
                    if (auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(
                            perBlockArgs[argIdx])) {
                      mlir::Type origTy = typeAttr.getValue();
                      if (toBoundedStaticType(origTy) == blockArg.getType()) {
                        blockArg.setType(origTy);
                      }
                    }
                  }
                  ++argIdx;
                }
              }
            }
            op->removeAttr(kOrigRegionArgTypesAttr);
          }

          if (auto whileOp = mlir::dyn_cast<mlir::stablehlo::WhileOp>(op)) {
            for (auto [i, opnd] : llvm::enumerate(whileOp->getOperands())) {
              auto opndTy =
                  mlir::dyn_cast<mlir::RankedTensorType>(opnd.getType());
              if (!opndTy || opndTy.hasStaticShape()) {
                continue;
              }
              if (i < whileOp.getCond().getNumArguments() &&
                  toBoundedStaticType(opndTy) ==
                      whileOp.getCond().getArgument(i).getType()) {
                whileOp.getCond().getArgument(i).setType(opndTy);
              }
              if (i < whileOp.getBody().getNumArguments() &&
                  toBoundedStaticType(opndTy) ==
                      whileOp.getBody().getArgument(i).getType()) {
                whileOp.getBody().getArgument(i).setType(opndTy);
              }
              if (i < whileOp->getNumResults() &&
                  toBoundedStaticType(opndTy) ==
                      whileOp->getResult(i).getType()) {
                whileOp->getResult(i).setType(opndTy);
              }
            }
          }
        });
      }
      for (unsigned i = 0; i < funcOp.getNumResults(); ++i) {
        if (auto origTy =
                funcOp.getResultAttrOfType<mlir::TypeAttr>(i, kOrigTypeAttr)) {
          newOutTypes[i] = origTy.getValue();
          funcOp.removeResultAttr(i, mlir::StringAttr::get(ctx, kOrigTypeAttr));
          sigChanged = true;
        }
      }
      if (!funcOp.isExternal()) {
        if (auto retOp = mlir::dyn_cast<mlir::func::ReturnOp>(
                funcOp.front().getTerminator())) {
          for (auto [i, opndTy] : llvm::enumerate(retOp.getOperandTypes())) {
            if (i < newOutTypes.size() && newOutTypes[i] != opndTy) {
              newOutTypes[i] = opndTy;
              sigChanged = true;
            }
          }
        }
      }
      if (sigChanged) {
        funcOp.setFunctionType(
            mlir::FunctionType::get(ctx, newInTypes, newOutTypes));
      }
    }
  }
  if (hasTupleCustomCallOperands) {
    for (auto funcOp : mlirModule.getOps<mlir::func::FuncOp>()) {
      if (funcOp.isExternal()) {
        continue;
      }
      funcOp.front().walk([&](mlir::stablehlo::CustomCallOp customCall) {
        auto origOpndAttrs =
            customCall->getAttrOfType<mlir::ArrayAttr>(kOrigOperandTypesAttr);
        if (!origOpndAttrs) {
          return;
        }
        mlir::OpBuilder builder(customCall);
        mlir::ValueRange flatOperands(customCall.getInputs());
        mlir::SmallVector<mlir::Value> repackedOperands;
        repackedOperands.reserve(origOpndAttrs.size());
        for (mlir::Attribute attr : origOpndAttrs) {
          auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr);
          if (!typeAttr || flatOperands.empty()) {
            break;
          }
          std::vector<HloSharding> leafShardings;
          repackedOperands.push_back(
              repackTupleValue(builder, customCall.getLoc(), flatOperands,
                               typeAttr.getValue(), leafShardings));
        }
        if (flatOperands.empty() &&
            repackedOperands.size() == origOpndAttrs.size()) {
          customCall->setOperands(repackedOperands);
        }
        customCall->removeAttr(kOrigOperandTypesAttr);
      });
    }
  }
}

bool hasNonV3MhloShardings(mlir::ModuleOp module, bool enableHloShardingV3) {
  mlir::DictionaryAttr moduleDict = xla::sdy::getFrontendAttrs(module);
  const bool hasInTupleSdy = xla::sdy::hasKey(moduleDict, kInTupleShardings);
  const bool hasOutTupleSdy = xla::sdy::hasKey(moduleDict, kOutTupleShardings);
  const mlir::StringRef sdyShardingAttrName(
      xla::HloSharding::kShardingFrontendAttrName.data(),
      xla::HloSharding::kShardingFrontendAttrName.size());
  bool hasNamedSharding = false;
  bool hasNonV3Sharding = false;
  auto inspectAttr = [&](mlir::StringAttr attr, bool isShardingOp) {
    if (!attr) {
      return;
    }
    mlir::StringRef val = attr.getValue();
    if (val.contains("mesh[")) {
      hasNamedSharding = true;
    } else if (val != "{replicated}" || isShardingOp) {
      hasNonV3Sharding = true;
    }
  };
  for (auto funcOp : module.getOps<mlir::func::FuncOp>()) {
    if (funcOp.getSymName().contains(kManualComputationFuncName)) {
      continue;
    }
    const bool isMain = funcOp.getSymName() == "main";
    if (!(isMain && hasInTupleSdy)) {
      for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
        if (xla::sdy::hasKey(xla::sdy::getFuncArgFrontendAttrs(funcOp, i),
                             sdyShardingAttrName)) {
          continue;
        }
        inspectAttr(
            funcOp.getArgAttrOfType<mlir::StringAttr>(i, kXlaShardingAttr),
            /*isShardingOp=*/false);
        if (hasNamedSharding) {
          return false;
        }
        if (!enableHloShardingV3 && hasNonV3Sharding) {
          return true;
        }
      }
    }
    bool hasFuncResultSdy = isMain && hasOutTupleSdy;
    if (!funcOp.isExternal()) {
      funcOp.front().walk([&](mlir::Operation* op) {
        if (auto customCall =
                mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op)) {
          mlir::StringRef target = customCall.getCallTargetName();
          if (target == kFuncResultShardingTargetName) {
            hasFuncResultSdy = true;
            return mlir::WalkResult::advance();
          }
          if (target == kGlobalToLocalShapeCallTargetName ||
              target == kLocalToGlobalShapeCallTargetName) {
            return mlir::WalkResult::advance();
          }
        }
        if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op);
            callOp && xla::sdy::isManualComputation(callOp)) {
          return mlir::WalkResult::advance();
        }
        mlir::DictionaryAttr dict = xla::sdy::getFrontendAttrs(op);
        if (xla::sdy::hasKey(dict, sdyShardingAttrName) ||
            xla::sdy::hasKey(dict, kInShardings) ||
            xla::sdy::hasKey(dict, kOutShardings) ||
            xla::sdy::hasKey(dict, kManualAxes)) {
          return mlir::WalkResult::advance();
        }
        bool isShardingOp = false;
        if (auto customCall =
                mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op)) {
          mlir::StringRef target = customCall.getCallTargetName();
          isShardingOp = target == kShardingCustomCallTargetName ||
                         target == kSPMDFullToShardShapeCallTargetName ||
                         target == kSPMDShardToFullShapeCallTargetName;
        }
        inspectAttr(op->getAttrOfType<mlir::StringAttr>(kXlaShardingAttr),
                    isShardingOp);
        if (hasNamedSharding || (!enableHloShardingV3 && hasNonV3Sharding)) {
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
      if (hasNamedSharding) {
        return false;
      }
      if (!enableHloShardingV3 && hasNonV3Sharding) {
        return true;
      }
    }
    if (!hasFuncResultSdy) {
      for (unsigned i = 0; i < funcOp.getNumResults(); ++i) {
        inspectAttr(
            funcOp.getResultAttrOfType<mlir::StringAttr>(i, kXlaShardingAttr),
            /*isShardingOp=*/false);
        if (hasNamedSharding) {
          return false;
        }
        if (!enableHloShardingV3 && hasNonV3Sharding) {
          return true;
        }
      }
    }
  }
  return hasNonV3Sharding && !hasNamedSharding;
}

absl::Status runShardingPropagation(HloModule* hloModule,
                                    mlir::ModuleOp mlirModule,
                                    bool importMhloShardings,
                                    mlir::sdy::PropagationOptions options,
                                    absl::string_view passName) {
  VLOG(1) << "Using Shardy for XLA SPMD propagation.";

  const DebugOptions& debugOptions = hloModule->config().debug_options();
  bool isShardyVerbose =
      absl::StrContains(debugOptions.xla_dump_hlo_pass_re(), kShardyVerbose);
  std::string shardyDir =
      getShardyDirIfShouldDump(debugOptions, passName, isShardyVerbose);

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

  if (isShardyVerbose) {
    options.debugPropagationEdgeSharding = true;
    options.debugShardingOrigins = true;
  }

  if (!shardyDir.empty()) {
    shardyDir =
        tsl::io::JoinPath(shardyDir, "shardy", uniqueModuleName(*hloModule));
    LOG(INFO) << "Using Shardy output directory: " << shardyDir;
  }
  ABSL_RETURN_IF_ERROR(tsl::Env::Default()->RecursivelyCreateDir(shardyDir));
  // MLIR pipeline: (1) import, (2) Shardy, and (3) export.

  bool enableVerifier = false;
  if constexpr (tsl::kIsDebugBuild) {
    enableVerifier = true;
  }

  bool hasBoundedDynamicTypes = false;
  bool hasTupleCustomCallOperands = false;
  prepareBoundedDynamicTypesAndTupleCustomCalls(
      mlirModule, hasBoundedDynamicTypes, hasTupleCustomCallOperands);

  mlir::PassManager pm(mlirModule->getContext());
  pm.enableVerifier(enableVerifier);
  int dumpIndex = 0;
  pm.addPass(mlir::sdy::createSaveModuleOpPass(shardyDir, "input_module",
                                               dumpIndex++));

  bool enableHloShardingV3 =
      hloModule->config().debug_options().xla_enable_hlo_sharding_v3();
  const bool hasSdyAttrs = xla::sdy::hasFrontendSdyAttrs(mlirModule);
  const bool hasMeshes = xla::sdy::hasFrontendMeshes(mlirModule);
  const bool hasNonV3Mhlo =
      ((enableHloShardingV3 && !hasSdyAttrs) || hasMeshes) &&
      hasNonV3MhloShardings(mlirModule, enableHloShardingV3);
  // Fallback is required to support models exported earlier with mixed
  // serialization disabled or lowered from frontends using HloShardingV1/V2
  // (such as tf2xla).
  // TODO (b/519501636): Remove the frontend_attributes fallback once 6 months
  // compatibility window for ifrt has passed in Nov'2026.
  if (enableHloShardingV3 && (hasSdyAttrs || hasNonV3Mhlo)) {
    if (hasSdyAttrs) {
      LOG(WARNING)
          << "Module contains sdy shardings or meshes in frontend_attributes "
             "even when HloShardingV3 is enabled. This is probably due to "
             "stale model export, try re-exporting the model on HEAD (with "
             "HloShardingV3 enabled). Disabling HloShardingV3 here to prevent "
             "conflicts during Shardy import";
    }
    hloModule->mutable_config()
        .mutable_debug_options()
        .set_xla_enable_hlo_sharding_v3(false);
    enableHloShardingV3 = false;
  }

  auto spanToArrayRef = [](absl::Span<const bool> span) {
    return mlir::ArrayRef<bool>(span.data(), span.size());
  };
  const mlir::ArrayRef<bool> allowPropToArgs = spanToArrayRef(
      hloModule->config().allow_spmd_sharding_propagation_to_parameters());
  const mlir::ArrayRef<bool> allowPropToResults = spanToArrayRef(
      hloModule->config().allow_spmd_sharding_propagation_to_output());
  if (importMhloShardings ||
      (!enableHloShardingV3 && (!hasMeshes || hasNonV3Mhlo))) {
    if (importMhloShardings) {
      LOG(WARNING)
          << "ShardyXLA is run against a module with HLO shardings. It "
             "should be used only for testing.";
    }
    addStablehloImportPipeline(
        pm, allowPropToArgs, allowPropToResults, enableHloShardingV3,
        /*inlineMesh=*/hasSdyAttrs,
        /*liftAndDedupMeshes=*/true, hloModule->config().num_partitions());
  } else {
    // This branch is in production.
    addSdyRoundTripImportPipeline(pm, /*enableConstantImport=*/true,
                                  /*liftAndDedupMeshes=*/true,
                                  enableHloShardingV3);
  }

  bool shardyGeneratesDeviceCode =
      !options.avoidExportForPartitioning &&
      !options.enablePerInstructionPartitioning &&
      options.partitionerStage ==
          mlir::sdy::PartitionerStage::kConvertGlobalToLocal;
  if (!shardyGeneratesDeviceCode && !enableHloShardingV3 && !hasSdyAttrs) {
    auto isLockedAt = [](mlir::ArrayRef<bool> allowProp, size_t idx) {
      if (allowProp.empty()) {
        return true;
      }
      if (allowProp.size() == 1) {
        return !allowProp.front();
      }
      return idx < allowProp.size() && !allowProp[idx];
    };
    auto isUnevenV2Sharding = [](mlir::Type type, mlir::StringAttr attr) {
      auto rankedTy = mlir::dyn_cast<mlir::RankedTensorType>(type);
      if (!rankedTy || !rankedTy.hasStaticShape() || !attr) {
        return false;
      }
      const xla::HloSharding sharding = xla::sdy::parseShardingFromString(attr);
      if (sharding.IsTuple() || sharding.UseNamedShardingLeaf() ||
          !sharding.IsTiledLeaf()) {
        return false;
      }
      for (int64_t d = 0; d < rankedTy.getRank() &&
                          d < sharding.tile_assignment().num_dimensions();
           ++d) {
        if (rankedTy.getDimSize(d) % sharding.tile_assignment().dim(d) != 0) {
          return true;
        }
      }
      return false;
    };
    if (auto mainFunc = mlirModule.lookupSymbol<mlir::func::FuncOp>("main")) {
      bool hasLockedUnevenBoundary = false;
      for (unsigned i = 0; i < mainFunc.getNumArguments(); ++i) {
        if (isLockedAt(allowPropToArgs, i) &&
            isUnevenV2Sharding(mainFunc.getArgumentTypes()[i],
                               mainFunc.getArgAttrOfType<mlir::StringAttr>(
                                   i, kXlaShardingAttr))) {
          hasLockedUnevenBoundary = true;
          break;
        }
      }
      if (!hasLockedUnevenBoundary) {
        for (unsigned i = 0; i < mainFunc.getNumResults(); ++i) {
          if (isLockedAt(allowPropToResults, i) &&
              isUnevenV2Sharding(mainFunc.getResultTypes()[i],
                                 mainFunc.getResultAttrOfType<mlir::StringAttr>(
                                     i, kXlaShardingAttr))) {
            hasLockedUnevenBoundary = true;
            break;
          }
        }
      }
      if (hasLockedUnevenBoundary) {
        options.updateNonDivisibleInputOutputShardings = false;
      }
    }
  }

  // NOTE: if we are using auto-spmd, we will use conservative propagation
  // since the TOAST cost model cannot account for split axes or padding.
  options.dumpDirectory = shardyDir;
  options.conservativePropagation = hloModule->use_auto_spmd_partitioning();
  options.enableAutoPartitioning = hloModule->use_auto_spmd_partitioning();
  mlir::sdy::addPropagationPipeline(pm, dumpIndex, options);

  xla::sdy::StablehloExportPipelineOptions stablehloExportPipelineOptions;
  stablehloExportPipelineOptions.addMissingShardingToControlFlow =
      !shardyGeneratesDeviceCode;
  stablehloExportPipelineOptions.enableHloShardingV3 = enableHloShardingV3;
  stablehloExportPipelineOptions.exportAllReduceScatter =
      debugOptions.xla_sdy_export_all_reduce_scatter();
  stablehloExportPipelineOptions.simplifyReplicatedShardings = true;
  stablehloExportPipelineOptions.clearReverseOpSharding =
      options.clearReverseOpSharding;
  addStablehloExportPipeline(pm, stablehloExportPipelineOptions);
  pm.addPass(mlir::sdy::createSaveModuleOpPass(shardyDir, "output_module",
                                               dumpIndex++));
  tsl::StatusScopedDiagnosticHandler diagnosticHandler(
      mlirModule->getContext());
  ABSL_RETURN_IF_ERROR(diagnosticHandler.consumeStatus(pm.run(mlirModule)));

  restoreBoundedDynamicTypesAndTupleCustomCalls(
      mlirModule, hasBoundedDynamicTypes, hasTupleCustomCallOperands);
  return absl::OkStatus();
}

absl::StatusOr<bool> eraseInlineableAttrForShardyManualComputations(
    HloModule* module) {
  bool changed = false;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCall &&
          absl::StrContains(instruction->to_apply()->name(),
                            sdy::kManualComputationFuncName.str())) {
        auto it =
            instruction->frontend_attributes().map().find(kXlaInlineableAttr);
        if (it != instruction->frontend_attributes().map().end()) {
          absl::string_view value = it->second;
          TF_RET_CHECK(value == "false" || value == "xla_late" ||
                       value == "auto")
              << "Unexpected inlineable attribute value: " << value;
          instruction->erase_frontend_attribute(kXlaInlineableAttr);
          // TODO(b/436603025). CallInliner do not inline the Shardy related
          // manual computations based on the callee name. We have to rename the
          // callee to a name such that it can be inlined. If we can remove the
          // special handling in CallInliner, we can remove this renaming.
          module->SetAndUniquifyComputationName(instruction->to_apply(),
                                                "inlineable_callee");
          changed = true;
        }
      }
    }
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> ShardyXLA::RunImpl(
    HloModule* hloModule,
    const absl::flat_hash_set<absl::string_view>& executionThreads) {
  auto moduleFrontendAttrs = hloModule->frontend_attributes().map();
  bool useTupleArgs = moduleFrontendAttrs.contains(kUseTupleArgs);
  bool importMhloShardings = moduleFrontendAttrs.contains(kImportMhloShardings);

  // If propagation is enabled, we don't need to erase the inlineable attribute
  // for manual computations, since StablehloExportPipeline can handle it.
  if (!runSdyShardingPropagation) {
    ABSL_ASSIGN_OR_RETURN(bool changed,
                     eraseInlineableAttrForShardyManualComputations(hloModule));
    if (!useTupleArgs) {
      // Nothing more to do.
      return changed;
    }
  }

  // The auto-spmd flag is present in both the HLO module and the config. Apply
  // auto spmd partitioning if either is true.
  if (hloModule->use_auto_spmd_partitioning() ||
      hloModule->config().use_auto_spmd_partitioning()) {
    hloModule->set_use_auto_spmd_partitioning(true);
  }

  // HLO -> StableHLO
  auto mlirContext = std::make_unique<mlir::MLIRContext>();
  loadAllRequiredDialects(mlirContext.get());
  ABSL_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> mlirModule,
                   xla::ConvertHloToStablehlo(*mlirContext.get(), hloModule));

  // Store the entry computation layout, input-output alias config, and buffer
  // donors, which will be restored in the end, since MLIR does not preserve
  // those properties.
  ComputationLayout flattenedEntryComputationLayout =
      getFlattenedComputationLayout(hloModule->entry_computation_layout(),
                                    useTupleArgs);
  OriginalParamIndexToFlattenedNum originalParamIndexToFlattenedNum =
      getOriginalParamIndexToFlattenedNum(hloModule);
  HloInputOutputAliasConfig oldInputOutputAliasConfig =
      hloModule->input_output_alias_config();
  HloBufferDonorConfig flattenedBufferDonorsConfig =
      getFlattenedBufferDonorsConfig(hloModule->buffer_donor_config(),
                                     originalParamIndexToFlattenedNum,
                                     useTupleArgs);

  if (runSdyShardingPropagation) {
    propagationOptions.replicaCount = hloModule->config().replica_count();
    propagationOptions.partitionCount = hloModule->config().num_partitions();
    ABSL_RETURN_IF_ERROR(runShardingPropagation(hloModule, mlirModule.get(),
                                           importMhloShardings,
                                           propagationOptions, name()));
  }

  // TODO(b/431836696): Remove once issue is fixed.
  if (useTupleArgs) {
    mlirModule.get()->removeAttr(
        "mhlo.xla_entry_computation_parameter_layouts");
    mlirModule.get()->removeAttr("mhlo.xla_entry_computation_parameter_tiles");
  }

  // StableHlo -> HLO
  HloProto hloProto;
  ABSL_RETURN_IF_ERROR(ConvertStablehloWithManyArgsToHloProto(*mlirModule, &hloProto,
                                                         useTupleArgs));
  ABSL_RETURN_IF_ERROR(
      createFromProtoAndReplaceComputations(hloModule, hloProto.hlo_module()));

  // If the module returns a single tensor as result with sharding,
  // ConvertMlirHloToHlo still generates the tuple and get-tuple-element as the
  // root instructions. We use the TupleSimplifier as a temporary solution.
  CHECK_OK(TupleSimplifier().Run(hloModule));

  hloModule->set_input_output_alias_config(getFlattenedInputOutputAliasConfig(
      oldInputOutputAliasConfig,
      hloModule->entry_computation()->root_instruction()->shape(),
      originalParamIndexToFlattenedNum, useTupleArgs));
  hloModule->set_buffer_donor_config(std::move(flattenedBufferDonorsConfig));

  if (runSdyShardingPropagation &&
      mlir::sdy::shardyGeneratesDeviceCode(propagationOptions)) {
    const HloModuleProto& proto = hloProto.hlo_module();
    if (proto.has_spmd_output_sharding()) {
      ABSL_ASSIGN_OR_RETURN(HloSharding hlo_sharding,
                       HloSharding::FromProto(proto.spmd_output_sharding()));
      hloModule->set_spmd_output_sharding(std::move(hlo_sharding));
    }
    if (!proto.spmd_parameters_shardings().empty() ||
        proto.has_spmd_output_sharding()) {
      std::vector<HloSharding> param_shardings;
      param_shardings.reserve(proto.spmd_parameters_shardings_size());
      for (const auto& sharding_proto : proto.spmd_parameters_shardings()) {
        ABSL_ASSIGN_OR_RETURN(HloSharding sharding,
                         HloSharding::FromProto(sharding_proto));
        param_shardings.push_back(std::move(sharding));
      }
      hloModule->set_spmd_parameters_shardings(std::move(param_shardings));
    }

    ABSL_RETURN_IF_ERROR(updateEntryComputationLayout(
        hloModule, flattenedEntryComputationLayout));
  } else {
    // Restore entry computation layout.
    *hloModule->mutable_entry_computation_layout() =
        std::move(flattenedEntryComputationLayout);

    ABSL_RETURN_IF_ERROR(
        hlo_sharding_util::CanonicalizeLayoutAfterShardingPropagation(
            hloModule,
            hloModule->config().allow_spmd_sharding_propagation_to_output(),
            hloModule->config()
                .allow_spmd_sharding_propagation_to_parameters()));
  }

  // We don't fully replace the HLO module, so it will continue to have the
  // temporary frontend attributes. So clean them up as XLA won't need them.
  removeFrontendAttributes(
      hloModule, {kUseTupleArgs, kImportMhloShardings, kMeshesRoundTripAttr,
                  kInTupleShardings, kOutTupleShardings});

  return true;
}

namespace {

class ShardyXLAFactory : public HloPassFactory {
 public:
  absl::string_view name() const override { return "shardy-xla"; }

  absl::StatusOr<std::unique_ptr<HloPassInterface>> Create(
      const HloPassContext&, const HloPassOptions& options) const override {
    return std::make_unique<ShardyXLA>(
        options.Get<bool>("run_sharding_propagation").value_or(true));
  }
};

const HloPassFactory* const kShardyXLARegistration =
    RegisterHloPassFactory<ShardyXLAFactory>();

}  // namespace

}  // namespace sdy
}  // namespace xla
