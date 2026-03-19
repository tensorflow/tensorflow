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

#include "xla/core/host_offloading/host_offloading_transforms.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/computation_layout.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {
// Returns true if all parameters are already flattened and program does not
// have any tuple parameters which are not supported by PjRt.
bool IsAllParametersFlattened(const ProgramShape& program_shape) {
  return absl::c_all_of(program_shape.parameters(),
                        [](const Shape& shape) { return !shape.IsTuple(); });
}

// Returns true if all outputs are already aliased with parameters.
bool IsAllOutputsAliased(const ProgramShape& program_shape,
                         const HloInputOutputAliasConfig& alias_config) {
  return absl::c_all_of(
      ShapeUtil::GetLeafShapes(program_shape.result()),
      [&](const ShapeUtil::IndexedShape& indexed) {
        auto alias = alias_config.GetAliasedParameter(indexed.index);
        return alias.has_value() &&
               alias->kind == HloInputOutputAliasConfig::kMustAlias;
      });
}

// Returns a computation layout with all tuple parameters flattened.
ComputationLayout FlattenComputationLayout(const ComputationLayout& layout) {
  ProgramShape flat_shape;

  for (size_t i = 0; i < layout.parameter_count(); ++i) {
    for (const auto& indexed :
         ShapeUtil::GetLeafShapes(layout.parameter_shape(i))) {
      flat_shape.AddParameter(indexed.shape, "");
    }
  }

  *flat_shape.mutable_result() = layout.result_shape();
  return ComputationLayout(flat_shape, /*ignore_layouts=*/false);
}

// Returns an input-output alias config with all tuple parameters flattened.
absl::StatusOr<HloInputOutputAliasConfig> FlattenInputOutputAliasConfig(
    const ComputationLayout& layout,
    const HloInputOutputAliasConfig& alias_config) {
  HloInputOutputAliasConfig flat_alias_config(layout.result_shape());

  // A mapping from original parameter number and shape index to a flatten
  // parameter number (flatten shape index is always `{}`).
  using Key = std::pair<size_t, ShapeIndex>;
  absl::flat_hash_map<Key, size_t> flat_params;

  for (size_t i = 0; i < layout.parameter_count(); ++i) {
    for (auto& indexed : ShapeUtil::GetLeafShapes(layout.parameter_shape(i))) {
      flat_params[{i, indexed.index}] = flat_params.size();
    }
  }

  // Output to parameter aliasing can be set up only for leaf arrays, so we
  // don't need to worry about tuples.
  for (auto& indexed : ShapeUtil::GetLeafShapes(layout.result_shape())) {
    if (auto alias = alias_config.GetAliasedParameter(indexed.index)) {
      Key key = {alias->parameter_number, alias->parameter_index};
      TF_RETURN_IF_ERROR(
          flat_alias_config.SetUpAlias(indexed.index, flat_params.at(key),
                                       /*param_index=*/{}, alias->kind));
    }
  }

  return flat_alias_config;
}

// Updates HLO module to have all tuple parameters flattened.
absl::Status FlattenEntryParameters(HloModule* hlo_module) {
  HloComputation* entry = hlo_module->entry_computation();

  // If we don't have tuple parameters we don't have to do anything.
  if (absl::c_all_of(entry->parameter_instructions(),
                     [](auto* inst) { return !inst->shape().IsTuple(); })) {
    return absl::OkStatus();
  }

  // Compute new input-output alias config for flattened parameters.
  TF_ASSIGN_OR_RETURN(
      HloInputOutputAliasConfig flat_alias_config,
      FlattenInputOutputAliasConfig(hlo_module->entry_computation_layout(),
                                    hlo_module->input_output_alias_config()));

  // Compute new computation layout for flattened parameters.
  ComputationLayout flat_layout =
      FlattenComputationLayout(hlo_module->entry_computation_layout());

  // Mapping from an original parameter number and index to a new instruction.
  absl::btree_map<std::pair<size_t, ShapeIndex>,
                  std::unique_ptr<HloInstruction>>
      flat_params;

  for (size_t i = 0; i < entry->num_parameters(); ++i) {
    HloInstruction* param = entry->parameter_instruction(i);

    // Forward non-tuple parameters as is, potentially with a new parameter
    // number of they follow tupled parameter(s).
    if (!param->shape().IsTuple()) {
      flat_params[{i, {}}] = HloInstruction::CreateParameter(
          flat_params.size(), param->shape(), param->name());
      continue;
    }

    // Flatten tuple parameters and keep track of new parameters instructions.
    CHECK(param->shape().IsTuple()) << "Parameter " << i << " is not a tuple";
    for (auto& indexed : ShapeUtil::GetLeafShapes(param->shape())) {
      flat_params[{i, indexed.index}] = HloInstruction::CreateParameter(
          flat_params.size(), indexed.shape, param->name());
    }
  }

  // Create tuples in the entry computation that reconstructs original tuple
  // parameter from flattened parameters.
  for (size_t i = 0; i < entry->num_parameters(); ++i) {
    HloInstruction* param = entry->parameter_instruction(i);

    // Forward non-tuple parameters to the new instructions.
    if (!param->shape().IsTuple()) {
      TF_RETURN_IF_ERROR(
          param->ReplaceAllUsesWith(flat_params.at({i, {}}).get()));
      continue;
    }

    // Recursively builds a tuple from flattened parameters to reconstruct
    // an HLO value with original parameter shape to replace all uses with it.
    std::function<HloInstruction*(const ShapeIndex& index)> make_tuple;
    make_tuple = [&](const ShapeIndex& index) -> HloInstruction* {
      const auto& subshape = ShapeUtil::GetSubshape(param->shape(), index);

      // Forward non-tuple parameters to the caller.
      if (!subshape.IsTuple()) {
        return flat_params.at({i, index}).get();
      }

      // Create a tuple from instructions corresponding to nested elements.
      std::vector<HloInstruction*> tuple_elements;
      for (int j = 0; j < subshape.tuple_shapes().size(); ++j) {
        ShapeIndex nested_index = index;
        nested_index.push_back(j);
        tuple_elements.push_back(make_tuple(nested_index));
      }

      return entry->AddInstruction(HloInstruction::CreateTuple(tuple_elements));
    };

    // Replace original tuple parameter with a tuple of flatten parameters.
    TF_RETURN_IF_ERROR(param->ReplaceAllUsesWith(make_tuple({})));
  }

  // Remove all original parameters from the entry computation.
  for (int32_t i = entry->num_parameters() - 1; i >= 0; --i) {
    TF_RETURN_IF_ERROR(entry->RemoveParameter(i));
  }

  // Add flattened parameters to the entry computation.
  for (auto& [_, flat_param] : flat_params) {
    entry->AddParameter(std::move(flat_param));
  }

  // Update HLO module alias config and computation layout after rewriting entry
  // computation parameters.
  hlo_module->set_input_output_alias_config(flat_alias_config);
  *hlo_module->mutable_entry_computation_layout() = flat_layout;

  return absl::OkStatus();
}

// Converts HLO module into destination passing style computation by appending
// parameters for each non-aliased output of entry computation.
absl::Status AppendDestinationParameters(HloModule* hlo_module) {
  HloComputation* entry = hlo_module->entry_computation();

  const ComputationLayout& layout = hlo_module->entry_computation_layout();
  HloInputOutputAliasConfig& alias_config =
      hlo_module->input_output_alias_config();

  // Appends output parameter for outputs that are not already aliased.
  auto append_output_param = [&](const Shape& shape, const ShapeIndex& index) {
    // Skip non-leaf outputs.
    if (shape.IsTuple()) {
      return absl::OkStatus();
    }

    // Skip outputs that already must alias.
    if (auto alias = alias_config.GetAliasedParameter(index)) {
      if (alias->kind == HloInputOutputAliasConfig::kMayAlias) {
        return absl::InternalError("May-alias output is not supported");
      }
      return absl::OkStatus();
    }

    // Add a destination parameter aliased with an output.
    size_t parameter_number = entry->num_parameters();
    entry->AddEntryComputationParameter(HloInstruction::CreateParameter(
        parameter_number, shape, "output_param"));
    return alias_config.SetUpAlias(index, parameter_number, {},
                                   HloInputOutputAliasConfig::kMustAlias);
  };

  // Append output parameters for aliased outputs.
  Shape original_result_shape = layout.result_shape();
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(original_result_shape,
                                                          append_output_param));

  return absl::OkStatus();
}

}  // namespace

absl::Status RewriteToDestinationPassingStyle(
    HloModule* hlo_module, const ProgramShape& program_shape,
    const HloInputOutputAliasConfig& alias_config) {
  bool is_flattened = IsAllParametersFlattened(program_shape);
  bool is_aliased = IsAllOutputsAliased(program_shape, alias_config);

  // If host offloading module is flattened and all outputs are aliased we
  // don't have to do anything, otherwise we have to convert HLO module into
  // destination passing style computation compatible with PJRT CPU client.
  if (is_flattened && is_aliased) {
    return absl::OkStatus();
  }

  TF_RETURN_IF_ERROR(FlattenEntryParameters(hlo_module));
  TF_RETURN_IF_ERROR(AppendDestinationParameters(hlo_module));

  return absl::OkStatus();
}

}  // namespace xla
