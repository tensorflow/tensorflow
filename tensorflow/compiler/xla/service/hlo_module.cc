/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include <iterator>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

HloModule::HloModule(const string& name,
                     const VersionedComputationHandle& entry_computation_handle,
                     const HloModuleConfig& config)
    : name_(name),
      config_(config),
      has_entry_computation_handle_(true),
      entry_computation_handle_(entry_computation_handle) {}

HloModule::HloModule(const string& name) : name_(name) {}
HloModule::HloModule(const string& name, const HloModuleConfig& config)
    : name_(name), config_(config) {}

HloComputation* HloModule::AddComputationInternal(
    std::unique_ptr<HloComputation> computation, bool is_entry,
    bool uniquify_names) {
  if (is_entry) {
    CHECK_EQ(nullptr, entry_computation_);
    entry_computation_ = computation.get();

    // If the module configuration has no entry layout computation set, create a
    // default one based on the program shape.
    if (!config_.has_entry_computation_layout()) {
      config_.SetDefaultComputationLayout(
          entry_computation_->ComputeProgramShape());
    }
  }

  if (uniquify_names) {
    computation->UniquifyName(&computation_name_uniquer_);
    for (auto* instruction : computation->instructions()) {
      instruction->UniquifyName(&instruction_name_uniquer_);
    }
  } else {
    // Don't uniquify the names of the computation or instruction, but we must
    // run the names through the uniquifiers to prevent future name collisions
    // for computations and instructions created later.
    computation_name_uniquer_.GetUniqueName(computation->name());
    for (auto* instruction : computation->instructions()) {
      instruction_name_uniquer_.GetUniqueName(instruction->name());
    }
  }

  // Pick unique IDs for each instruction.
  for (auto* instruction : computation->instructions()) {
    instruction->SetUniqueId(NewUniqueInstructionId());
  }
  computation->set_parent(this);
  computations_.push_back(std::move(computation));
  return computations_.back().get();
}

HloComputation* HloModule::AddEntryComputation(
    std::unique_ptr<HloComputation> computation) {
  return AddComputationInternal(std::move(computation), /*is_entry=*/true,
                                /*uniquify_names=*/true);
}

Status HloModule::RemoveEmbeddedComputation(HloComputation* to_remove) {
  auto it =
      std::find_if(computations_.begin(), computations_.end(),
                   [&to_remove](const std::unique_ptr<HloComputation>& comp) {
                     return comp.get() == to_remove;
                   });
  TF_RET_CHECK(it->get() == to_remove);
  computations_.erase(it);
  return Status::OK();
}

HloComputation* HloModule::AddEmbeddedComputation(
    std::unique_ptr<HloComputation> computation) {
  return AddComputationInternal(std::move(computation), /*is_entry=*/false,
                                /*uniquify_names=*/true);
}

void HloModule::ReplaceComputations(
    const std::unordered_map<HloComputation*, HloComputation*>& replacements) {
  // Replace all uses of non-canonical computations with their
  // representatives.
  std::vector<std::unique_ptr<HloComputation>> new_computations;
  new_computations.reserve(computations_.size());

  for (std::unique_ptr<HloComputation>& computation : computations_) {
    for (auto* instruction : computation->instructions()) {
      switch (instruction->opcode()) {
        case HloOpcode::kCall:
        case HloOpcode::kMap:
        case HloOpcode::kReduce:
        case HloOpcode::kReduceWindow: {
          HloComputation* new_arg = tensorflow::gtl::FindWithDefault(
              replacements, instruction->to_apply(), nullptr);
          if (new_arg != nullptr) {
            instruction->set_to_apply(new_arg);
          }
          break;
        }
        case HloOpcode::kWhile: {
          HloComputation* new_condition = tensorflow::gtl::FindWithDefault(
              replacements, instruction->while_condition(), nullptr);
          if (new_condition != nullptr) {
            instruction->set_while_condition(new_condition);
          }
          HloComputation* new_body = tensorflow::gtl::FindWithDefault(
              replacements, instruction->while_body(), nullptr);
          if (new_body != nullptr) {
            instruction->set_while_body(new_body);
          }
          break;
        }
        case HloOpcode::kSelectAndScatter: {
          HloComputation* new_select = tensorflow::gtl::FindWithDefault(
              replacements, instruction->select(), nullptr);
          if (new_select != nullptr) {
            instruction->set_select(new_select);
          }
          HloComputation* new_scatter = tensorflow::gtl::FindWithDefault(
              replacements, instruction->scatter(), nullptr);
          if (new_scatter != nullptr) {
            instruction->set_scatter(new_scatter);
          }
          break;
        }
        default:
          break;
      }
    }

    if (replacements.find(computation.get()) == replacements.end()) {
      new_computations.push_back(std::move(computation));
    }
  }

  // Replace entry_computation if necessary.
  entry_computation_ = tensorflow::gtl::FindWithDefault(
      replacements, entry_computation_, entry_computation_);

  computations_ = std::move(new_computations);
}

string HloModule::ToString(bool include_large_constants) const {
  std::ostringstream s;
  s << "HloModule " << name() << ":\n\n";
  for (const HloComputation* computation : MakeComputationPostOrder()) {
    // Fusion computations are emitted with their fusion instruction and
    // therefore don't need to be emitted as a separate comptutation in the
    // module.
    if (computation->IsFusionComputation()) {
      continue;
    }
    if (computation == entry_computation()) {
      s << "ENTRY ";
    }
    s << computation->ToString(
             /*nested_level=*/0,
             /*include_large_constants=*/include_large_constants)
      << "\n\n";
  }
  return s.str();
}

HloModuleProto HloModule::ToProto() const {
  HloModuleProto proto;
  proto.set_name(name_);
  proto.set_entry_computation_name(entry_computation_->name());
  for (const HloComputation* computation : MakeComputationPostOrder()) {
    // Fusion computations are added when the fusion instructions are created by
    // HloInstruction::CreateFromProto.
    if (computation->IsFusionComputation()) {
      continue;
    }
    HloComputationProto computation_proto = computation->ToProto();
    proto.add_computations()->Swap(&computation_proto);
  }
  return proto;
}

namespace {

// Construct a ProgramShape matching the shape of the parameters and root of the
// given module's entry computation.
StatusOr<ProgramShape> ProgramShapeFromProto(const HloModuleProto& module) {
  const HloComputationProto* entry_computation = nullptr;
  for (const HloComputationProto& computation : module.computations()) {
    if (computation.name() == module.entry_computation_name()) {
      entry_computation = &computation;
      break;
    }
  }
  TF_RET_CHECK(entry_computation != nullptr)
      << "No computation with entry computation name"
      << module.entry_computation_name();

  tensorflow::gtl::FlatMap<int64, std::pair<string, const Shape*>> parameters;
  const HloInstructionProto* root = nullptr;
  for (const HloInstructionProto& instruction :
       entry_computation->instructions()) {
    if (instruction.name() == entry_computation->root_name()) {
      TF_RET_CHECK(root == nullptr) << "Entry computation has more than "
                                       "one instruction with (root) name "
                                    << instruction.name();
      root = &instruction;
    }
    if (instruction.opcode() == HloOpcodeString(HloOpcode::kParameter)) {
      TF_RET_CHECK(!ContainsKey(parameters, instruction.parameter_number()))
          << "Entry computation has more than one parameter instruction "
             "with parameter number "
          << instruction.parameter_number();
      parameters[instruction.parameter_number()] = {
          instruction.parameter_name(), &instruction.shape()};
    }
  }
  TF_RET_CHECK(root != nullptr)
      << "Entry computation is missing root instruction named "
      << entry_computation->root_name();

  ProgramShape program_shape;
  *program_shape.mutable_result() = root->shape();
  for (int64 i = 0; i < parameters.size(); ++i) {
    TF_RET_CHECK(ContainsKey(parameters, i))
        << "Entry computation missing parameter number " << i;
    const string& name = parameters.at(i).first;
    const Shape& shape = *parameters.at(i).second;
    *program_shape.add_parameters() = shape;
    program_shape.add_parameter_names(name);
  }

  return std::move(program_shape);
}

}  // namespace

/* static */
StatusOr<std::unique_ptr<HloModule>> HloModule::CreateFromProto(
    const HloModuleProto& proto, const HloModuleConfig& module_config,
    const VersionedComputationHandle& entry_computation_handle) {
  // The ProgramShape in the passed in module config must match the shapes of
  // the entry parameters and root.
  TF_ASSIGN_OR_RETURN(ProgramShape expected_program_shape,
                      ProgramShapeFromProto(proto));
  TF_RET_CHECK(expected_program_shape.parameters_size() ==
               module_config.entry_computation_layout().parameter_count());
  for (int i = 0; i < expected_program_shape.parameters_size(); ++i) {
    const Shape& parameter_shape =
        module_config.entry_computation_layout().parameter_layout(i).shape();
    TF_RET_CHECK(
        ShapeUtil::Equal(expected_program_shape.parameters(i), parameter_shape))
        << "HloModuleConfig has different shape for parameter " << i
        << " than the HLO module. Expected: "
        << ShapeUtil::HumanStringWithLayout(
               expected_program_shape.parameters(i))
        << ", actual: " << ShapeUtil::HumanStringWithLayout(parameter_shape);
  }
  const Shape& result_shape =
      module_config.entry_computation_layout().result_layout().shape();
  TF_RET_CHECK(ShapeUtil::Equal(expected_program_shape.result(), result_shape))
      << "HloModuleConfig has different result shape than the HLO module. "
         "Expected: "
      << ShapeUtil::HumanStringWithLayout(expected_program_shape.result())
      << ", actual: " << ShapeUtil::HumanStringWithLayout(result_shape);

  auto module = MakeUnique<HloModule>(proto.name(), entry_computation_handle,
                                      module_config);

  tensorflow::gtl::FlatMap<string, HloComputation*> computation_map;
  for (const HloComputationProto& computation_proto : proto.computations()) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloComputation> computation,
                        HloComputation::CreateFromProto(
                            module.get(), computation_proto, &computation_map));
    CHECK_NE(computation.get(), nullptr);
    TF_RET_CHECK(!ContainsKey(computation_map, computation->name()));
    string computation_name = computation->name();
    // Don't uniquify names because we want names to be stable across
    // serialization and deserialization.
    computation_map[computation_name] = module->AddComputationInternal(
        std::move(computation),
        /*is_entry=*/proto.entry_computation_name() == computation_name,
        /*uniquify_names=*/false);
  }
  TF_RET_CHECK(module->entry_computation_ != nullptr);

  // Because we didn't uniquify the names, double-check that the instruction and
  // computation names are unique from the proto.
  tensorflow::gtl::FlatSet<string> computation_names;
  tensorflow::gtl::FlatSet<string> instruction_names;
  for (HloComputation* computation : module->computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }

    TF_RET_CHECK(!ContainsKey(computation_names, computation->name()))
        << "Computation name is not unique: " << computation->name();
    computation_names.insert(computation->name());
    for (HloInstruction* instruction : computation->instructions()) {
      TF_RET_CHECK(!ContainsKey(instruction_names, instruction->name()))
          << "Instruction name is not unique: " << instruction->name();
      instruction_names.insert(instruction->name());
    }
  }

  return std::move(module);
}

/* static */
StatusOr<HloModuleConfig> HloModule::CreateModuleConfigFromProto(
    const HloModuleProto& module) {
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      ProgramShapeFromProto(module));

  HloModuleConfig module_config(program_shape);

  // The module config is constructed with default layouts regardless of what is
  // passed in via the ProgramShape. Set the layouts to the appropriate values.
  ComputationLayout* entry_layout =
      module_config.mutable_entry_computation_layout();
  for (int64 i = 0; i < entry_layout->parameter_count(); ++i) {
    TF_RETURN_IF_ERROR(
        entry_layout->mutable_parameter_layout(i)->CopyLayoutFromShape(
            program_shape.parameters(i)));
  }
  TF_RETURN_IF_ERROR(entry_layout->mutable_result_layout()->CopyLayoutFromShape(
      program_shape.result()));

  return module_config;
}

namespace {
// Returns whether `hlo` is used outside the given subcomputation.
// `instructions_in_subcomputation` is the instruction set of the given
// subcomputation.
bool IsUsedOutsideSubcomputation(
    const HloInstruction& hlo,
    const std::unordered_set<HloInstruction*>& instructions_in_subcomputation) {
  for (HloInstruction* user : hlo.users()) {
    if (!instructions_in_subcomputation.count(user)) {
      return true;
    }
  }
  return false;
}
}  // anonymous namespace

HloInstruction* HloModule::OutlineExpressionFromComputation(
    tensorflow::gtl::ArraySlice<HloInstruction*> instructions_to_outline,
    const string& outlined_computation_name, HloComputation* computation) {
  auto builder = HloComputation::Builder(outlined_computation_name);

  // A map from original instructions to their counterparts in the new outlined
  // function.
  std::unordered_map<HloInstruction*, HloInstruction*> outlined_instructions;
  // A set that contains all instructions to be outlined.
  std::unordered_set<HloInstruction*> instruction_set_to_outline(
      instructions_to_outline.begin(), instructions_to_outline.end());
  std::vector<HloInstruction*> arguments;
  std::vector<HloInstruction*> outputs;
  int64 parameter_count = 0;
  for (HloInstruction* instruction_to_outline : instructions_to_outline) {
    // Clone the original instruction.
    HloInstruction* outlined_instruction =
        builder.AddInstruction(instruction_to_outline->Clone());

    // Replace its operands to their counterparts in the new function.
    for (int64 operand_num = 0;
         operand_num < outlined_instruction->operand_count(); ++operand_num) {
      HloInstruction* old_operand =
          outlined_instruction->mutable_operand(operand_num);

      HloInstruction** operand_slot = &(outlined_instructions[old_operand]);
      if (*operand_slot == nullptr) {
        // Because instructions_to_outline is in topological order, if
        // old_operand is not in outlined_instructions, old_operand must be an
        // input of the outlined subcomputation and thus should be represented
        // as a parameter in the new function.
        arguments.push_back(old_operand);
        *operand_slot = builder.AddInstruction(HloInstruction::CreateParameter(
            parameter_count, old_operand->shape(), ""));
        ++parameter_count;
      }
      TF_CHECK_OK(
          outlined_instruction->ReplaceOperandWith(operand_num, *operand_slot));
    }

    // Insert the new instruction into the outlined_instructions map.
    InsertOrDie(&outlined_instructions, instruction_to_outline,
                outlined_instruction);

    // Mark instruction_to_outline an output if it is used outside the
    // subcomputation or is the output of the original computation (i.e. used
    // externally).
    if (instruction_to_outline->user_count() == 0 ||
        IsUsedOutsideSubcomputation(*instruction_to_outline,
                                    instruction_set_to_outline)) {
      outputs.push_back(instruction_to_outline);
    }
  }

  if (outputs.size() != 1) {
    string error_message =
        "The subcomputation to outline has multiple outputs:\n";
    for (HloInstruction* output : outputs) {
      tensorflow::strings::StrAppend(&error_message, output->ToString(), "\n");
    }
    LOG(FATAL) << error_message;
  }
  HloInstruction* output = outputs[0];

  // Creates a call to the nested computation.
  HloComputation* nested_computation = AddEmbeddedComputation(
      builder.Build(FindOrDie(outlined_instructions, output)));
  HloInstruction* call = computation->AddInstruction(HloInstruction::CreateCall(
      output->shape(), arguments, nested_computation));

  VLOG(2) << "Outlining the following instructions";
  for (auto* instruction_to_outline : instructions_to_outline) {
    VLOG(2) << "  " << instruction_to_outline->ToString();
  }
  VLOG(2) << "as a call " << call->ToString();
  VLOG(2) << "to " << nested_computation->ToString();

  TF_CHECK_OK(output->ReplaceAllUsesWith(call));
  for (auto i = instructions_to_outline.rbegin();
       i != instructions_to_outline.rend(); ++i) {
    TF_CHECK_OK(computation->RemoveInstruction(*i));
  }

  return call;
}

std::list<HloComputation*> HloModule::MakeComputationPostOrder() const {
  // First determine all root computations by building a set of nonroot
  // computations (computations which are called by an instruction in the
  // module).
  std::set<HloComputation*> nonroot_computations;
  for (auto& computation : computations_) {
    for (auto* instruction : computation->instructions()) {
      for (HloComputation* called_computation :
           instruction->called_computations()) {
        nonroot_computations.insert(called_computation);
      }
    }
  }

  // Keep track of computations which have already been added to the post
  // order. This prevents duplication as an embedded computation may be called
  // from two different root computations.
  std::set<HloComputation*> added_computations;
  std::list<HloComputation*> post_order;
  for (auto& computation : computations_) {
    if (nonroot_computations.count(computation.get()) == 0) {
      for (HloComputation* embedded_computation :
           computation->MakeEmbeddedComputationsList()) {
        if (added_computations.count(embedded_computation) == 0) {
          post_order.push_back(embedded_computation);
          added_computations.insert(embedded_computation);
        }
      }
      // Root computations should only be encountered once.
      CHECK_EQ(0, added_computations.count(computation.get()));
      post_order.push_back(computation.get());
      added_computations.insert(computation.get());
    }
  }
  CHECK_EQ(post_order.size(), computations_.size());
  return post_order;
}

std::vector<HloComputation*> HloModule::MakeNonfusionComputations() const {
  std::vector<HloComputation*> result;
  for (auto* c : computations()) {
    if (c->IsFusionComputation()) {
      continue;
    }
    result.push_back(c);
  }
  return result;
}

std::unique_ptr<HloModule> HloModule::Clone(const string& suffix) const {
  VLOG(1) << "Cloning module :" << name_ << " --> " << suffix << "\n";
  auto module = MakeUnique<HloModule>(name_ + "-" + suffix);
  module->config_ = config_;
  module->entry_computation_handle_ = entry_computation_handle_;
  module->has_entry_computation_handle_ = has_entry_computation_handle_;

  std::unordered_map<HloComputation*, HloComputation*> clone_map;
  for (auto& computation : computations_) {
    auto cloned_computation = computation->Clone(suffix);
    InsertOrDie(&clone_map, computation.get(), cloned_computation.get());

    if (entry_computation_ == computation.get()) {
      module->AddEntryComputation(std::move(cloned_computation));
    } else {
      module->AddEmbeddedComputation(std::move(cloned_computation));
    }
  }

  for (auto& cloned_computation : module->computations_) {
    for (auto* instruction : cloned_computation->instructions()) {
      // Rewrite instruction's called_computation to point to the cloned
      // computations.
      instruction->ReplaceCalledComputations(
          [&](HloComputation* hlo) { return FindOrDie(clone_map, hlo); });
    }
  }
  return module;
}

uint64 HloModule::RandomNew64() const {
  tensorflow::mutex_lock l(rng_mutex_);
  return rng_();
}

}  // namespace xla
