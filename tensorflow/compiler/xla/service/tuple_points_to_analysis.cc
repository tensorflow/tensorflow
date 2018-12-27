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

#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"

#include <ostream>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

string BufferAlias::ToString() const {
  return absl::StrCat("BufferAlias(", instruction_->name(), "[",
                      absl::StrJoin(index_, ","), "])");
}

std::ostream& operator<<(std::ostream& out, const BufferAlias& buffer_alias) {
  out << buffer_alias.ToString();
  return out;
}

bool PointsToSet::IsAmbiguous() const {
  bool ambiguous = false;
  ForEachElement(
      [&ambiguous](const ShapeIndex& /*index*/, const BufferList& points_to) {
        ambiguous |= points_to.size() > 1;
      });
  return ambiguous;
}

bool PointsToSet::IsDistinct() const {
  bool distinct = true;
  std::set<const LogicalBuffer*> all_points_to;
  ForEachElement([&distinct, &all_points_to](const ShapeIndex& /*index*/,
                                             const BufferList& points_to) {
    for (auto& buffer : points_to) {
      if (all_points_to.count(buffer) != 0) {
        distinct = false;
      }
      all_points_to.insert(buffer);
    }
  });
  return distinct;
}

size_t PointsToSet::size() const {
  // Because pointed-to elements may be duplicated we have to create a flattened
  // set and return the size.
  return CreateFlattenedSet().size();
}

PointsToSet::BufferSet PointsToSet::CreateFlattenedSet() const {
  BufferSet flat_set;
  ForEachElement(
      [&flat_set](const ShapeIndex& /*index*/, const BufferList& buffers) {
        flat_set.insert(buffers.begin(), buffers.end());
      });
  return flat_set;
}

bool PointsToSet::ContainsBuffer(const LogicalBuffer& buffer) const {
  bool found = false;
  ForEachElement([&found, &buffer](const ShapeIndex& /*index*/,
                                   const BufferList& pointed_to_buffers) {
    if (!found &&
        std::find(pointed_to_buffers.begin(), pointed_to_buffers.end(),
                  &buffer) != pointed_to_buffers.end()) {
      found = true;
    }
  });
  return found;
}

bool PointsToSet::ContainsBufferAtIndex(const LogicalBuffer& buffer,
                                        const ShapeIndex& index) const {
  const auto& pointed_to_buffers = element(index);
  return std::find(pointed_to_buffers.begin(), pointed_to_buffers.end(),
                   &buffer) != pointed_to_buffers.end();
}

void PointsToSet::AddPointedToBuffer(const LogicalBuffer& buffer,
                                     const ShapeIndex& index) {
  if (ContainsBufferAtIndex(buffer, index)) {
    return;
  }
  mutable_element(index)->push_back(&buffer);
}

const PointsToSet::SourceSet& PointsToSet::tuple_sources(
    const ShapeIndex& index) const {
  return tree_.element(index).tuple_sources;
}

void PointsToSet::add_tuple_source(const ShapeIndex& index,
                                   HloInstruction* tuple) {
  tree_.mutable_element(index)->tuple_sources.insert(tuple);
}

namespace {
// Gather fusion instructions from 'instruction' into 'fusion_instructions'.
void GatherFusionInstructions(
    HloInstruction* instruction,
    std::vector<HloInstruction*>* fusion_instructions) {
  CHECK_EQ(HloOpcode::kFusion, instruction->opcode());
  for (auto* fused : instruction->fused_instructions()) {
    if (fused->opcode() == HloOpcode::kFusion) {
      GatherFusionInstructions(fused, fusion_instructions);
    }
  }
  fusion_instructions->push_back(instruction);
}

}  // namespace

/* static */ StatusOr<std::unique_ptr<TuplePointsToAnalysis>>
TuplePointsToAnalysis::Run(const HloModule* module) {
  auto logical_buffer_analysis = LogicalBufferAnalysis::Run(module);
  std::unique_ptr<TuplePointsToAnalysis> analysis(new TuplePointsToAnalysis(
      module, logical_buffer_analysis.ConsumeValueOrDie()));
  TF_RETURN_IF_ERROR(analysis->Analyze());
  return std::move(analysis);
}

Status TuplePointsToAnalysis::Analyze() {
  per_instruction_.clear();
  per_instruction_.reserve(module_->instruction_count());

  logical_buffer_aliases_.clear();
  logical_buffer_aliases_.resize(
      logical_buffer_analysis_->num_logical_buffers());

  std::vector<HloInstruction*> fusion_instructions;
  for (auto* computation : module_->MakeNonfusionComputations()) {
    TF_RETURN_IF_ERROR(computation->Accept(this));
    TF_RETURN_IF_ERROR(
        PopulateDefinedBuffersAndAliases(computation->instructions()));
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kFusion) {
        GatherFusionInstructions(instruction, &fusion_instructions);
      }
    }
  }
  // Run points-to analysis on fusion instructions in 'computation'.
  for (auto* instruction : fusion_instructions) {
    TF_RETURN_IF_ERROR(instruction->fused_expression_root()->Accept(this));
    TF_RETURN_IF_ERROR(
        PopulateDefinedBuffersAndAliases(instruction->fused_instructions()));
  }

  XLA_VLOG_LINES(3, ToString());

  return Status::OK();
}

Status TuplePointsToAnalysis::PopulateDefinedBuffersAndAliases(const decltype(
    std::declval<HloComputation>().instructions())& instructions) {
  for (auto* instruction : instructions) {
    PerInstruction* pi = PerInst(instruction);
    TF_RETURN_IF_ERROR(GatherBuffersDefinedByInstruction(
        instruction, &pi->instruction_defined_buffers));

    const PointsToSet& points_to_set = GetPointsToSet(instruction);
    points_to_set.ForEachElement(
        [this, &instruction](
            const ShapeIndex& index,
            const PointsToSet::BufferList& pointed_to_buffers) {
          for (const LogicalBuffer* buffer : pointed_to_buffers) {
            logical_buffer_aliases_[buffer->id()].emplace_back(instruction,
                                                               index);
          }
        });
  }
  return Status::OK();
}

Status TuplePointsToAnalysis::DefaultAction(HloInstruction* hlo_instruction) {
  // Create trivial points-to set for instruction. Each points-to set at index i
  // contains a single element LogicalBuffer(hlo_instruction, i). This indicates
  // that this instruction is the source of all buffers in its own output.
  PointsToSet& points_to_set = CreateEmptyPointsToSet(hlo_instruction);
  points_to_set.ForEachMutableElement(
      [this, hlo_instruction](const ShapeIndex& index,
                              PointsToSet::BufferList* buffers) {
        buffers->push_back(
            &logical_buffer_analysis_->GetBuffer(hlo_instruction, index));
      });

  if (hlo_instruction->shape().IsTuple()) {
    // If the hlo instruction is a tuple-shaped, then trivially the instruction
    // itself is the source of the tuple.
    points_to_set.add_tuple_source({}, hlo_instruction);
  }

  return Status::OK();
}

Status TuplePointsToAnalysis::HandleGetTupleElement(
    HloInstruction* get_tuple_element) {
  // GetTupleElement forwards a pointer to a particular element of the tuple
  // operand.
  int64 element_index = get_tuple_element->tuple_index();

  PointsToSet& points_to_set = CreateEmptyPointsToSet(get_tuple_element);
  const PointsToSet& operand_points_to_set =
      *PerInst(get_tuple_element->operand(0))->points_to_set;

  // Copy the points-to set (and tuple sources) at index {element_index} of the
  // operand to the points-to set for this GetTupleElement instruction.
  points_to_set.ForEachMutableElement(
      [&](const ShapeIndex& target_index, PointsToSet::BufferList* points_to) {
        // Construct an index into the operand by prepending element_index to
        // the index for the GetTupleElement instruction's points-to set.
        ShapeIndex src_index;
        src_index.push_back(element_index);
        for (auto element : target_index) {
          src_index.push_back(element);
        }

        *points_to = operand_points_to_set.element(src_index);
        for (HloInstruction* tuple :
             operand_points_to_set.tuple_sources(src_index)) {
          points_to_set.add_tuple_source(target_index, tuple);
        }
      });

  return Status::OK();
}

Status TuplePointsToAnalysis::HandleCopy(HloInstruction* copy) {
  // A kCopy instruction performs a shallow copy of the operand. The top-level
  // buffer (index={}) is newly created, but all other buffers (in the case of a
  // tuple shape) come from the operand
  PointsToSet& points_to_set = CreateCopiedPointsToSet(copy, copy->operand(0));
  points_to_set.mutable_element(/*index=*/{})->clear();
  points_to_set.AddPointedToBuffer(
      logical_buffer_analysis_->GetBuffer(copy, /*index=*/{}),
      /*index=*/{});

  return Status::OK();
}

Status TuplePointsToAnalysis::HandleBitcast(HloInstruction* bitcast) {
  // A kBitcast instruction aliases its operand. That is, the buffer of its
  // result *is* the buffer of its operand, so just copy the operands points-to
  // set.
  CreateCopiedPointsToSet(bitcast, bitcast->operand(0));
  return Status::OK();
}

Status TuplePointsToAnalysis::HandleDomain(HloInstruction* domain) {
  // A kDomain instruction aliases its operand. That is, the buffer of its
  // result *is* the buffer of its operand, so just copy the operands points-to
  // set.
  CreateCopiedPointsToSet(domain, domain->operand(0));
  return Status::OK();
}

Status TuplePointsToAnalysis::HandleAddDependency(
    HloInstruction* add_dependency) {
  // AddDependency just forwards the value of its zero-th operand.
  CreateCopiedPointsToSet(add_dependency, add_dependency->operand(0));
  return Status::OK();
}

Status TuplePointsToAnalysis::HandleRecvDone(HloInstruction* recv_done) {
  // RecvDone aliases its input (Recv) tuple element {0} to element {0} of its
  // output. The other indices ({} and {1}) define their own buffers.
  PointsToSet& points_to_set = CreateEmptyPointsToSet(recv_done);
  points_to_set.AddPointedToBuffer(
      logical_buffer_analysis_->GetBuffer(recv_done, /*index=*/{}),
      /*index=*/{});
  points_to_set.AddPointedToBuffer(
      logical_buffer_analysis_->GetBuffer(recv_done, /*index=*/{1}),
      /*index=*/{1});

  const PointsToSet& operand_points_to_set =
      GetPointsToSet(recv_done->operand(0));

  // Recursively copy the points to set of the operand tuple {0} to the output
  // element {0}.
  points_to_set.ForEachMutableElement(
      [&points_to_set, &operand_points_to_set](
          const ShapeIndex& index, PointsToSet::BufferList* buffers) {
        if (index.empty() || index[0] != 0) {
          return;
        }
        *buffers = operand_points_to_set.element(index);
        for (auto& tuple_source : operand_points_to_set.tuple_sources(index)) {
          points_to_set.add_tuple_source(index, tuple_source);
        }
      });
  return Status::OK();
}

Status TuplePointsToAnalysis::HandleSend(HloInstruction* send) {
  // Send creates a tuple of {aliased operand, U32 context, token}.
  PointsToSet& points_to_set = CreateEmptyPointsToSet(send);

  // Creates the points to set for the tuple and its element at {1}.
  auto top_buffer = points_to_set.mutable_element(ShapeIndex({}));
  top_buffer->push_back(
      &logical_buffer_analysis_->GetBuffer(send, ShapeIndex({})));
  points_to_set.add_tuple_source({}, send);

  auto context_buffer = points_to_set.mutable_element(ShapeIndex({1}));
  context_buffer->push_back(
      &logical_buffer_analysis_->GetBuffer(send, ShapeIndex({1})));

  auto token_buffer = points_to_set.mutable_element(ShapeIndex({2}));
  token_buffer->push_back(
      &logical_buffer_analysis_->GetBuffer(send, ShapeIndex({2})));

  // Recursively copy the points to set of the operand to output tuple {0}.
  const PointsToSet& operand_points_to_set = GetPointsToSet(send->operand(0));
  operand_points_to_set.ForEachElement(
      [&points_to_set, &operand_points_to_set](
          const ShapeIndex& src_index,
          const PointsToSet::BufferList& points_to) {
        ShapeIndex target_index({0});
        for (auto element : src_index) {
          target_index.push_back(element);
        }
        *points_to_set.mutable_element(target_index) = points_to;

        for (HloInstruction* tuple :
             operand_points_to_set.tuple_sources(src_index)) {
          points_to_set.add_tuple_source(target_index, tuple);
        }
      });

  return Status::OK();
}

Status TuplePointsToAnalysis::HandleTuple(HloInstruction* tuple) {
  absl::Span<HloInstruction* const> operands(tuple->operands());
  PointsToSet& points_to_set = CreateEmptyPointsToSet(tuple);
  points_to_set.AddPointedToBuffer(
      logical_buffer_analysis_->GetBuffer(tuple, /*index=*/{}),
      /*index=*/{});

  // A tuple contains references to all input operands and transitively any
  // references in those operands.
  for (int64 i = 0; i < operands.size(); ++i) {
    const PointsToSet& operand_points_to_set =
        *PerInst(operands[i])->points_to_set;

    // Copy the points-to set (and tuple sources) of the operand into the
    // respective subtree of the tuple instructions points-to set.
    operand_points_to_set.ForEachElement(
        [&points_to_set, &operand_points_to_set, i](
            const ShapeIndex& src_index,
            const PointsToSet::BufferList& points_to) {
          ShapeIndex target_index;
          target_index.push_back(i);
          for (auto element : src_index) {
            target_index.push_back(element);
          }

          *points_to_set.mutable_element(target_index) = points_to;

          for (HloInstruction* tuple :
               operand_points_to_set.tuple_sources(src_index)) {
            points_to_set.add_tuple_source(target_index, tuple);
          }
        });
  }

  points_to_set.add_tuple_source({}, tuple);

  return Status::OK();
}

Status TuplePointsToAnalysis::HandleTupleSelect(HloInstruction* tuple_select) {
  // Select allocates a new buffer and then shallow copies the on_true or
  // on_false buffer into this new buffer. Which side is chosen cannot be
  // determined statically so conservatively set the points-to set to the union
  // of these on_true and on_false operands.
  //
  // First create a copy of the on_true points-to set (and tuple sources), then
  // add in elements of the on_false points-to set (tuple sources).
  auto on_true = tuple_select->operand(1);
  auto on_false = tuple_select->operand(2);
  PointsToSet& points_to_set = CreateCopiedPointsToSet(tuple_select, on_true);
  const PointsToSet& false_points_to_set = *PerInst(on_false)->points_to_set;
  points_to_set.ForEachMutableElement(
      [&](const ShapeIndex& index, PointsToSet::BufferList* buffers) {
        for (const LogicalBuffer* false_buffer :
             false_points_to_set.element(index)) {
          points_to_set.AddPointedToBuffer(*false_buffer, index);
        }

        for (HloInstruction* tuple : false_points_to_set.tuple_sources(index)) {
          points_to_set.add_tuple_source(index, tuple);
        }
      });

  // Select creates a new (top-level) buffer to store its result, so its
  // respective element in the points-to set should contain only itself.
  points_to_set.mutable_element({})->clear();
  points_to_set.AddPointedToBuffer(
      logical_buffer_analysis_->GetBuffer(tuple_select, /*index=*/{}),
      /*index=*/{});
  return Status::OK();
}

const PointsToSet& TuplePointsToAnalysis::GetPointsToSet(
    const HloInstruction* hlo_instruction) const {
  return *PerInst(hlo_instruction)->points_to_set;
}

PointsToSet& TuplePointsToAnalysis::CreateEmptyPointsToSet(
    const HloInstruction* instruction) {
  PerInstruction* pi = PerInst(instruction);
  CHECK(pi->points_to_set == nullptr)
      << "instruction should not have been present in the map.";
  auto set = absl::make_unique<PointsToSet>(&instruction->shape());
  pi->points_to_set = std::move(set);
  // Return *set using the iterator returned by emplace.
  return *pi->points_to_set;
}

bool TuplePointsToAnalysis::InstructionDefinesBufferAtIndex(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  const auto& buffers = GetPointsToSet(instruction).element(index);
  return (buffers.size() == 1 && buffers[0]->instruction() == instruction);
}

Status TuplePointsToAnalysis::VerifyBuffer(const LogicalBuffer& buffer) const {
  if (!InstructionDefinesBufferAtIndex(buffer.instruction(), buffer.index())) {
    return FailedPrecondition(
        "LogicalBuffer %s is ill-defined: instruction %s does not define a "
        "buffer at that index",
        buffer.ToString(), buffer.instruction()->name());
  }

  if (buffer.id() < 0 ||
      buffer.id() >= logical_buffer_analysis_->num_logical_buffers()) {
    return FailedPrecondition("LogicalBuffer %s is ill-defined: invalid id %d",
                              buffer.ToString(), buffer.id());
  }
  if (GetBuffer(buffer.id()).instruction() != buffer.instruction() ||
      GetBuffer(buffer.id()).index() != buffer.index()) {
    return FailedPrecondition(
        "LogicalBuffer %s is ill-defined: buffer with same id differs: %s",
        buffer.ToString(), GetBuffer(buffer.id()).ToString());
  }

  return Status::OK();
}

const LogicalBuffer& TuplePointsToAnalysis::GetBuffer(
    LogicalBuffer::Id id) const {
  CHECK_GE(id, 0);
  CHECK_LT(id, logical_buffer_analysis_->num_logical_buffers());
  return logical_buffer_analysis_->GetBuffer(id);
}

StatusOr<const LogicalBuffer*> TuplePointsToAnalysis::GetBufferDefinedAt(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  const auto& buffers = GetPointsToSet(instruction).element(index);
  if (buffers.size() != 1 || buffers[0]->instruction() != instruction) {
    return FailedPrecondition(
        "instruction %s does not define buffer at index {%s}",
        instruction->name(), absl::StrJoin(index, ","));
  }
  return buffers[0];
}

const TuplePointsToAnalysis::BufferAliasVector&
TuplePointsToAnalysis::GetBufferAliases(const LogicalBuffer& buffer) const {
  return logical_buffer_aliases_.at(buffer.id());
}

const TuplePointsToAnalysis::BufferDefinitionVector&
TuplePointsToAnalysis::GetBuffersDefinedByInstruction(
    const HloInstruction* instruction) const {
  return PerInst(instruction)->instruction_defined_buffers;
}

Status TuplePointsToAnalysis::GatherBuffersDefinedByInstruction(
    const HloInstruction* instruction,
    TuplePointsToAnalysis::BufferDefinitionVector* buffers) {
  GetPointsToSet(instruction)
      .ForEachElement([buffers, instruction](
                          const ShapeIndex& index,
                          const PointsToSet::BufferList& source_buffers) {
        // Add buffers which 'instruction' is the source of.
        CHECK(!source_buffers.empty());
        if (source_buffers.size() == 1 &&
            source_buffers[0]->instruction() == instruction) {
          // If this instruction is the source of this buffer the
          // indices must match.
          DCHECK(source_buffers[0]->index() == index);
          buffers->push_back(source_buffers[0]);
        } else {
          // If the points-to set includes more than one buffer then
          // necessarily this instruction did not produce the
          // buffer.
          for (const LogicalBuffer* source_buffer : source_buffers) {
            DCHECK(source_buffer->instruction() != instruction);
          }
        }
      });
  return Status::OK();
}

PointsToSet& TuplePointsToAnalysis::CreateCopiedPointsToSet(
    const HloInstruction* instruction, const HloInstruction* src) {
  // PointsToSet doesn't have a copy constructor so copy over element-by-element
  // from src PointsToSet.
  PointsToSet& dst_points_to_set = CreateEmptyPointsToSet(instruction);
  const PointsToSet& src_points_to_set = GetPointsToSet(src);
  dst_points_to_set.ForEachMutableElement(
      [&dst_points_to_set, &src_points_to_set](
          const ShapeIndex& index, PointsToSet::BufferList* buffers) {
        *buffers = src_points_to_set.element(index);
        for (auto& tuple_source : src_points_to_set.tuple_sources(index)) {
          dst_points_to_set.add_tuple_source(index, tuple_source);
        }
      });
  return *PerInst(instruction)->points_to_set;
}

string TuplePointsToAnalysis::ToString() const {
  string output =
      absl::StrFormat("TuplePointsToSet for module %s:\n", module_->name());
  for (const auto* computation : module_->MakeNonfusionComputations()) {
    const char* entry =
        computation == module_->entry_computation() ? "entry " : "";
    absl::StrAppend(&output, entry, "computation ", computation->name(), ":\n");
    for (const HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      InstructionToString(instruction, &output);
      if (instruction->opcode() == HloOpcode::kFusion) {
        for (auto* fused : instruction->fused_instructions()) {
          InstructionToString(fused, &output);
        }
      }
    }
  }

  absl::StrAppend(&output, "LogicalBuffers:\n");
  for (const auto& b : logical_buffer_analysis_->logical_buffers()) {
    absl::StrAppend(&output, "  buffer ", b->ToString(), ":\n");
    for (const BufferAlias& alias : logical_buffer_aliases_.at(b->id())) {
      absl::StrAppend(&output, "    alias ", alias.ToString(), "\n");
    }
  }
  return output;
}

void TuplePointsToAnalysis::InstructionToString(
    const HloInstruction* instruction, string* output) const {
  const string prefix = instruction->IsFused() ? "    " : "";
  absl::StrAppend(output, prefix, "  instruction ",
                  instruction->ToShortString(), ":\n");
  const PointsToSet& points_to_set = GetPointsToSet(instruction);
  points_to_set.ForEachElement([&prefix, &output](
                                   const ShapeIndex& index,
                                   const PointsToSet::BufferList& points_to) {
    absl::StrAppend(output, prefix, "    {", absl::StrJoin(index, ","), "}: ",
                    absl::StrJoin(points_to, ", ",
                                  [](string* out, const LogicalBuffer* source) {
                                    out->append(source->ToString());
                                  }),
                    "\n");
  });
}

bool TuplePointsToAnalysis::DoesNotUseOperandBuffer(
    const HloInstruction* operand, const ShapeIndex& index,
    const HloInstruction* user) const {
  CHECK(user->IsUserOf(operand))
      << "user: " << user->ToString() << " operand: " << operand->ToString();
  if (user->opcode() == HloOpcode::kGetTupleElement && !index.empty()) {
    // GetTupleElement instructions only access the top-level buffer of their
    // operand.
    return true;
  } else if (user->opcode() == HloOpcode::kFusion &&
             user->fusion_kind() == HloInstruction::FusionKind::kLoop) {
    // Find fusion parameter associated with 'operand'.
    auto it = std::find_if(
        user->fused_parameters().begin(), user->fused_parameters().end(),
        [=](HloInstruction* fused_param) {
          return user->operand(fused_param->parameter_number()) == operand;
        });
    CHECK(it != user->fused_parameters().end());
    // Iterate through all users of all buffer aliases of the buffer in the
    // points-to set of fusion parameter at 'index'.
    // Return false if any uses are detected at 'index', returns true otherwise.
    const LogicalBuffer* buffer = GetBufferDefinedAt(*it, index).ValueOrDie();
    for (const BufferAlias& alias : GetBufferAliases(*buffer)) {
      for (HloInstruction* alias_user : alias.instruction()->users()) {
        if (DoesNotUseOperandBuffer(alias.instruction(), alias.index(),
                                    alias_user)) {
          continue;
        }
        // Return false: use detected at 'buffer' -> 'alias' -> 'alias_user'.
        return false;
      }
    }
    // Return true: found no uses of 'operand' at 'index' in 'user'.
    return true;
  }
  return false;
}

// Returns all uses of all aliases of 'instruction' at 'index' in 'uses'.
// Each use in 'uses' is a pair (HloInstruction* user, int64 operand_index)
// where 'user' is a user of an alias of 'instruction' at 'index', and
// 'operand_index' is the operand index at which the alias appears in the
// operand list of 'user'.
std::vector<std::pair<HloInstruction*, int64>>
TuplePointsToAnalysis::GetAllUsesOfInstructionAtIndex(
    HloInstruction* instruction, const ShapeIndex& index) const {
  std::vector<std::pair<HloInstruction*, int64>> uses;
  const PointsToSet::BufferList& points_to =
      GetPointsToSet(instruction).element(index);
  for (const LogicalBuffer* buffer : points_to) {
    for (const BufferAlias& alias : GetBufferAliases(*buffer)) {
      for (HloInstruction* alias_user : alias.instruction()->users()) {
        if (DoesNotUseOperandBuffer(alias.instruction(), alias.index(),
                                    alias_user)) {
          continue;
        }
        for (int64 op_idx : alias_user->OperandIndices(alias.instruction())) {
          uses.emplace_back(alias_user, op_idx);
        }
      }
    }
  }
  return uses;
}

// Returns true if there is exactly one use of 'operand' at 'operand_index'
// in 'fusion.fused_instructions', where the singleton use is the fused
// root at operand index 'use_operand_index'. Returns false otherwise.
//
// REQUIRES: 'fusion' opcode is a kFusion instruction.
bool TuplePointsToAnalysis::HasUniqueFusedUseOfOperandAt(
    HloInstruction* operand, const ShapeIndex& operand_index,
    HloInstruction* fusion, const int64 use_operand_index) const {
  CHECK_EQ(HloOpcode::kFusion, fusion->opcode());
  // Check that 'operand' is unique in the operand list of 'fusion'.
  if (fusion->OperandIndices(operand).size() > 1) {
    return false;
  }
  // Find fusion parameter associated with 'operand'.
  const auto& fused_params = fusion->fused_parameters();
  auto fused_param_it = std::find_if(
      fused_params.begin(), fused_params.end(),
      [&](HloInstruction* fused_param) {
        return fusion->operand(fused_param->parameter_number()) == operand;
      });
  if (fused_param_it == fused_params.end()) {
    return false;
  }
  auto* fused_param = *fused_param_it;
  // Get all uses of 'operand' at 'index' from 'fusion.fused_instructions'.
  auto fused_param_uses =
      GetAllUsesOfInstructionAtIndex(fused_param, operand_index);
  // Return true iff there is exactly one use of 'operand' at 'index', and
  // this singleton use is the fused root (at index in 'use_operand_indices').
  return fused_param_uses.size() == 1 &&
         fused_param_uses[0].first == fusion->fused_expression_root() &&
         fused_param_uses[0].second == use_operand_index;
}

// User and operand can share buffers iff both instructions emit the same shape
// and layout, and 'user' meets one of the following qualifications:
//
// (1) Is element-wise. Or...
// (2) Is a loop fusion instruction where the only use of 'operand' at 'index'
//     in the set 'user.fused_instructions' is a DynamicUpdateSlice fused root
//     at operand 0. Or...
// (3) Is a kDot -> kAdd output fusion instruction where the only use of
//     'operand' at 'index' in the set 'user.fused_instructions' is a kAdd fused
//     root at operand 0 or 1. Or...
// (4) The 'user' of 'operand' is DynamicUpdateSlice or While at operand index
//     0.
// (5) The 'user' of 'operand' is Sort, and it is the only user.
//
// (2) and (3) can only be determined if points-to analysis is available.
bool TuplePointsToAnalysis::CanShareOperandBufferWithUser(
    HloInstruction* operand, const ShapeIndex& operand_index,
    HloInstruction* user, const ShapeIndex& user_index) const {
  CHECK(user->IsUserOf(operand))
      << "user: " << user->ToString() << " operand: " << operand->ToString();
  const Shape& operand_subshape =
      ShapeUtil::GetSubshape(operand->shape(), operand_index);
  const Shape& user_subshape =
      ShapeUtil::GetSubshape(user->shape(), user_index);
  // Check that operand and user emit the same shape and layout.
  if (!ShapeUtil::Equal(operand_subshape, user_subshape)) {
    return false;
  }
  if (user->opcode() == HloOpcode::kFusion) {
    if (user->fusion_kind() == HloInstruction::FusionKind::kLoop ||
        user->fusion_kind() == HloInstruction::FusionKind::kInput) {
      if (user->fused_expression_root()->opcode() ==
          HloOpcode::kDynamicUpdateSlice) {
        // Loop fusion with kDynamicUpdateSlice fused root.
        //
        // Returns true iff there is exactly one use of 'operand' at shape index
        // 'operand_index', and this singleton use is the fused root at operand
        // index 0.
        return HasUniqueFusedUseOfOperandAt(operand, operand_index, user, 0);
      } else {
        HloInstruction* fusion_param =
            user->fused_parameter(user->operand_index(operand));
        return HloDataflowAnalysis::AreTransitiveUsesElementwiseOrTuple(
            fusion_param);
      }
    } else if (user->fusion_kind() == HloInstruction::FusionKind::kOutput &&
               user->fused_expression_root()->opcode() == HloOpcode::kAdd) {
      // Output fusion with kAdd fused root.

      // Check if one operand of kAdd fused root is kDot or kConvolution.
      auto* add = user->fused_expression_root();
      auto add_operand_it =
          std::find_if(add->operands().begin(), add->operands().end(),
                       [&](HloInstruction* operand) {
                         return operand->opcode() == HloOpcode::kConvolution ||
                                operand->opcode() == HloOpcode::kDot;
                       });
      if (add_operand_it == add->operands().end()) {
        return false;
      }
      auto* matched_add_operand = *add_operand_it;
      // Calculate operand index of 'add' operand which was not matched above.
      const int64 other_add_operand_index =
          matched_add_operand == add->operand(0) ? 1 : 0;
      // Returns true iff there is exactly one use of 'operand' at shape index
      // 'operand_index', and this singleton use is the fused root (at operand
      // index 'other_add_operand_index').
      return HasUniqueFusedUseOfOperandAt(operand, operand_index, user,
                                          other_add_operand_index);
    }
  }
  if (user->opcode() == HloOpcode::kDynamicUpdateSlice ||
      user->opcode() == HloOpcode::kScatter ||
      user->opcode() == HloOpcode::kWhile) {
    // We eliminated other users in BufferLiveness::live_range_strictly_before,
    // so here we just need to check that the use is at operand index 0.
    std::vector<int64> operand_indices = user->OperandIndices(operand);
    return operand_indices.size() == 1 && operand_indices[0] == 0;
  }
  if (user->opcode() == HloOpcode::kSort) {
    // Only valid if there are no other users.
    if (operand->users().size() != 1) {
      return false;
    }
    // If we only sort keys, the output of sort is not a tuple, so we can always
    // share the buffer.
    if (user->operand_count() == 1) {
      return true;
    }
    CHECK(!user_index.empty());
    // Only share with the right tuple element buffer.
    std::vector<int64> operand_indices = user->OperandIndices(operand);
    return operand_indices.size() == 1 && user_index[0] == operand_indices[0];
  }
  if (user->opcode() == HloOpcode::kCall) {
    // TODO(b/62548313): Remove when buffer assignment is module scoped and
    // does not assign buffers to calls.
    // Find called computation parameter associated with 'operand'.
    const std::vector<int64> operand_indices = user->OperandIndices(operand);
    if (operand_indices.size() > 1) {
      return false;
    }
    CHECK_EQ(1, operand_indices.size());
    auto* param = user->to_apply()->parameter_instruction(operand_indices[0]);
    // Get all uses of 'operand' at 'index' in called computation.
    auto param_uses = GetAllUsesOfInstructionAtIndex(param, operand_index);

    // Return true iff:
    // *) There exists exactly one use of 'operand' in called computation.
    // *) The unique use is by the root instruction of called computation.
    //    (Note: we check the root of the called computation, because the
    //     root result buffer is required to alias with the Call result buffer).
    // *) The root instruction of the called computation is element-wise on
    //    'operand'.
    auto* callee_root = user->to_apply()->root_instruction();
    return param_uses.size() == 1 && param_uses[0].first == callee_root &&
           callee_root->IsElementwiseOnOperand(param_uses[0].second);
  }
  // Loop fusions that contain transposing copies won't reach here as they have
  // different layouts, which fails the check in the beginning of this function.
  //
  // Multi-output fusion will fail the check here as tuples are not considered
  // an elementwise operation.
  return user->IsElementwiseOnOperand(user->operand_index(operand));
}

}  // namespace xla
