/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/service/optimize_input_output_buffer_alias.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_input_output_alias_config.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_reachability.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/hlo_alias_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_buffer.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {

StatusOr<bool> OptimizeInputOutputBufferAlias::Build(
    absl::Span<const Shape> input_shapes, const Shape& output_shape,
    HloInputOutputAliasConfig* alias_config,
    HloBufferDonorConfig* buffer_donor_config) {
  bool changed = false;

  // Collects all buffer donors in a vector.
  struct DonorEntry {
    int64_t param_number;
    ShapeIndex index;
    int64_t shape_size;
  };
  std::vector<DonorEntry> donor_vectors;

  for (int64_t param_number = 0; param_number < input_shapes.size();
       ++param_number) {
    const Shape& input_shape = input_shapes[param_number];
    TF_RET_CHECK(LayoutUtil::HasLayout(input_shape));
    VLOG(1) << "input_shape: " << input_shape.ToString();
    ShapeUtil::ForEachSubshape(input_shape, [&](const Shape& subshape,
                                                const ShapeIndex& index) {
      if (!LayoutUtil::IsDenseArray(subshape)) {
        return;
      }
      if (alias_config->ParameterHasAlias(param_number, index)) {
        return;
      }
      if (registered_buffer_donor_only_ &&
          !buffer_donor_config->ParameterIsBufferDonor(param_number, index)) {
        return;
      }
      donor_vectors.emplace_back(
          DonorEntry{param_number, index, shape_size_fn_(subshape)});
    });
  }

  // Collects all buffer donees in a vector.
  struct DoneeEntry {
    ShapeIndex index;
    int64_t shape_size;
  };
  std::vector<DoneeEntry> donee_vectors;
  TF_RET_CHECK(LayoutUtil::HasLayout(output_shape));
  VLOG(1) << "output_shape: " << output_shape.ToString();
  ShapeUtil::ForEachSubshape(
      output_shape, [&](const Shape& subshape, const ShapeIndex& index) {
        if (!LayoutUtil::IsDenseArray(subshape)) {
          return;
        }
        if (alias_config->OutputHasAlias(index)) {
          return;
        }
        donee_vectors.emplace_back(DoneeEntry{index, shape_size_fn_(subshape)});
      });

  // Sort donor and donees by their shape size in non-increasing order.
  absl::c_stable_sort(donor_vectors,
                      [](const DonorEntry& a, const DonorEntry& b) -> bool {
                        return a.shape_size > b.shape_size;
                      });
  absl::c_stable_sort(donee_vectors,
                      [](const DoneeEntry& a, const DoneeEntry& b) -> bool {
                        return a.shape_size > b.shape_size;
                      });

  // Match donors and donees with two pointers. The larger size a donee has, the
  // more prioritized the donee will get matched.
  int64_t donor_vector_index = 0;
  int64_t donee_vector_index = 0;
  while (donor_vector_index < donor_vectors.size() &&
         donee_vector_index < donee_vectors.size()) {
    const auto& donor = donor_vectors[donor_vector_index];
    const auto& donee = donee_vectors[donee_vector_index];
    if (donor.shape_size > donee.shape_size) {
      donor_vector_index += 1;
    } else if (donor.shape_size < donee.shape_size) {
      donee_vector_index += 1;
    } else {
      // The current donor and donee match.
      TF_RETURN_IF_ERROR(alias_config->SetUpAlias(
          donee.index, donor.param_number, donor.index));
      TF_RETURN_IF_ERROR(buffer_donor_config->RemoveBufferDonor(
          donor.param_number, donor.index));
      donor_vector_index += 1;
      donee_vector_index += 1;
      changed = true;
    }
  }

  return changed;
}

StatusOr<bool> AddControlDependencyForAlias(
    const HloInputOutputAliasConfig::Alias& alias,
    const ShapeIndex& output_index,
    const HloComputation* const entry_computation,
    const HloAliasAnalysis& alias_analysis, HloReachabilityMap* reachability) {
  bool changed = false;

  // Step 1. Collect where the input and output buffers are used.
  const auto& input_buffers = alias_analysis.ComputeBuffersAt(
      entry_computation->parameter_instruction(alias.parameter_number),
      alias.parameter_index);
  const auto& output_buffers = alias_analysis.ComputeBuffersAt(
      entry_computation->root_instruction(), output_index);

  // We only consider the instructions in the entry computation for data
  // dependency analysis.
  std::vector<HloInstruction*> input_instructions;
  for (const HloBuffer* const buffer : input_buffers) {
    for (const HloPosition& position : buffer->ComputePositions()) {
      if (position.instruction->parent() == entry_computation) {
        input_instructions.push_back(position.instruction);
      }
    }
  }
  std::vector<HloInstruction*> output_instructions;
  for (const HloBuffer* const buffer : output_buffers) {
    for (const HloPosition& position : buffer->ComputePositions()) {
      if (position.instruction->parent() == entry_computation) {
        output_instructions.push_back(position.instruction);
      }
    }
  }

  // Step 2. We only need the potential earliest instructions for the
  // output buffer.
  std::vector<HloInstruction*> potential_earlist_output_instructions;
  {
    std::vector<bool> potential_earlist_output_flag(output_instructions.size(),
                                                    true);
    for (int64_t i = 0; i < output_instructions.size(); ++i) {
      if (!potential_earlist_output_flag[i]) {
        continue;
      }
      for (int64_t j = i + 1; j < output_instructions.size(); ++j) {
        if (reachability->IsReachable(output_instructions[i],
                                      output_instructions[j])) {
          potential_earlist_output_flag[j] = false;
        } else if (reachability->IsReachable(output_instructions[j],
                                             output_instructions[i])) {
          potential_earlist_output_flag[i] = false;
          break;
        }
      }
    }
    for (int64_t i = 0; i < output_instructions.size(); ++i) {
      if (potential_earlist_output_flag[i]) {
        potential_earlist_output_instructions.push_back(output_instructions[i]);
      }
    }
  }

  // Step 3. Similar to Step 2, we only need the potential latest
  // instructions for the input buffer.
  std::vector<HloInstruction*> potential_latest_input_instructions;
  {
    std::vector<bool> potential_latest_input_flag(input_instructions.size(),
                                                  true);
    for (int64_t i = 0; i < input_instructions.size(); ++i) {
      if (!potential_latest_input_flag[i]) {
        continue;
      }
      for (int64_t j = i + 1; j < input_instructions.size(); ++j) {
        if (reachability->IsReachable(input_instructions[i],
                                      input_instructions[j])) {
          potential_latest_input_flag[i] = false;
          break;
        } else if (reachability->IsReachable(input_instructions[j],
                                             input_instructions[i])) {
          potential_latest_input_flag[j] = false;
        }
      }
    }
    for (int64_t i = 0; i < input_instructions.size(); ++i) {
      if (potential_latest_input_flag[i]) {
        potential_latest_input_instructions.push_back(input_instructions[i]);
      }
    }
  }

  // Step 4. Collect the users of the input_instructions.
  // The input_instructions is where buffers are used. We further collect
  // the users of these positions. An example is listed below. Input P1
  // and Output O1 are aliased. P1 is used by two instructions I1, I2. We
  // have to ensure that I1 and I2 are executed before O1. The
  // input_instructions is {P1}, the input_users is {I1, I2}.
  absl::flat_hash_set<HloInstruction*> input_users;
  for (const auto& input_instr : potential_latest_input_instructions) {
    for (const auto& user : input_instr->users()) {
      input_users.emplace(user);
    }
  }

  // Step 5. If there is a input user later than the output_instr, the
  // current input and output share memory. It is not necessary to add
  // control dependency.
  for (const auto& input_user : input_users) {
    for (const auto& output_instr : potential_earlist_output_instructions) {
      if (input_user != output_instr &&
          reachability->IsReachable(output_instr, input_user)) {
        return changed;
      }
    }
  }

  // Step 6. Add control dependency if the input_user is not
  // strictly before the output_instruction.
  for (const auto& input_user : input_users) {
    for (const auto& output_instr : potential_earlist_output_instructions) {
      if (!reachability->IsReachable(input_user, output_instr)) {
        // If there is a path from input_user to output_instr (including
        // the case that two instructions are the same), it is redundant
        // to add control dependency.
        VLOG(1) << "Add control dependency between predecessor "
                << input_user->ToString() << " and successor "
                << output_instr->ToString();
        TF_RETURN_IF_ERROR(input_user->AddControlDependencyTo(output_instr));
        reachability->UpdateReachabilityThroughInstruction(output_instr);
        changed = true;
      }
    }
  }

  return changed;
}

StatusOr<bool> OptimizeInputOutputBufferAlias::AddControlDependencyForAlias(
    HloModule* module) {
  bool global_changed = false;

  // When we call HloAliasAnalysis, we need to reset input_output_alias_config.
  // Otherwise, HloAliasAnalysis will consider the input output alias.
  const auto real_alias_config = module->input_output_alias_config();
  module->input_output_alias_config() = HloInputOutputAliasConfig(
      module->entry_computation()->root_instruction()->shape());
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module));
  module->input_output_alias_config() = real_alias_config;

  auto reachability = HloReachabilityMap::Build(module->entry_computation());

  // Add control dependency for must-alias at first. Then consider the
  // may-alias.
  TF_RETURN_IF_ERROR(module->input_output_alias_config().ForEachAliasWithStatus(
      [&](const ShapeIndex& output_index,
          const HloInputOutputAliasConfig::Alias& alias) -> Status {
        if (alias.kind == HloInputOutputAliasConfig::AliasKind::kMustAlias) {
          TF_ASSIGN_OR_RETURN(
              bool local_changed,
              ::xla::AddControlDependencyForAlias(
                  alias, output_index, module->entry_computation(),
                  *alias_analysis, reachability.get()));
          global_changed |= local_changed;
        }
        return OkStatus();
      }));
  TF_RETURN_IF_ERROR(module->input_output_alias_config().ForEachAliasWithStatus(
      [&](const ShapeIndex& output_index,
          const HloInputOutputAliasConfig::Alias& alias) -> Status {
        if (alias.kind == HloInputOutputAliasConfig::AliasKind::kMayAlias) {
          TF_ASSIGN_OR_RETURN(
              bool local_changed,
              ::xla::AddControlDependencyForAlias(
                  alias, output_index, module->entry_computation(),
                  *alias_analysis, reachability.get()));
          global_changed |= local_changed;
        }
        return OkStatus();
      }));
  return global_changed;
}

StatusOr<bool> OptimizeInputOutputBufferAlias::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // We exactly follow HloInputOutputAliasConfig::Verify to create input_shapes
  // and output_shape.
  const auto& entry_computation_layout = module->entry_computation_layout();
  std::vector<Shape> input_shapes;
  for (int64_t i = 0; i < module->entry_computation()->num_parameters(); ++i) {
    input_shapes.push_back(entry_computation_layout.parameter_shape(i));
  }
  const Shape& output_shape = entry_computation_layout.result_shape();

  HloInputOutputAliasConfig* alias_config =
      &module->input_output_alias_config();
  HloBufferDonorConfig* buffer_donor_config = &module->buffer_donor_config();

  TF_ASSIGN_OR_RETURN(
      bool alias_added,
      Build(input_shapes, output_shape, alias_config, buffer_donor_config));
  TF_RETURN_IF_ERROR(alias_config->Verify(*module, shape_size_fn_));
  TF_ASSIGN_OR_RETURN(bool control_dependency_added,
                      AddControlDependencyForAlias(module));

  return alias_added || control_dependency_added;
}

}  // namespace xla
