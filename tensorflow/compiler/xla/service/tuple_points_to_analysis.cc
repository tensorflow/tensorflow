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

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

string BufferAlias::ToString() const {
  return tensorflow::strings::StrCat("BufferAlias(", instruction_->name(), "[",
                                     tensorflow::str_util::Join(index_, ","),
                                     "])");
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
  per_instruction_.resize(module_->NumUniqueInstructionIds());

  logical_buffer_aliases_.clear();
  logical_buffer_aliases_.resize(
      logical_buffer_analysis_->num_logical_buffers());

  for (auto& computation : module_->computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    TF_RETURN_IF_ERROR(computation->Accept(this));
    TF_RETURN_IF_ERROR(
        PopulateDefinedBuffersAndAliases(computation->instructions()));
    // Run points-to analysis on fusion instructions in 'computation'.
    for (auto& instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kFusion) {
        continue;
      }
      TF_RETURN_IF_ERROR(instruction->fused_expression_root()->Accept(this));
      TF_RETURN_IF_ERROR(
          PopulateDefinedBuffersAndAliases(instruction->fused_instructions()));
    }
  }

  XLA_VLOG_LINES(3, ToString());

  return Status::OK();
}

Status TuplePointsToAnalysis::PopulateDefinedBuffersAndAliases(
    const std::list<std::unique_ptr<HloInstruction>>& instructions) {
  for (auto& instruction : instructions) {
    PerInstruction* pi = PerInst(instruction.get());
    TF_RETURN_IF_ERROR(GatherBuffersDefinedByInstruction(
        instruction.get(), &pi->instruction_defined_buffers));

    const PointsToSet& points_to_set = GetPointsToSet(instruction.get());
    points_to_set.ForEachElement(
        [this, &instruction](
            const ShapeIndex& index,
            const PointsToSet::BufferList& pointed_to_buffers) {
          for (const LogicalBuffer* buffer : pointed_to_buffers) {
            logical_buffer_aliases_[buffer->id()].emplace_back(
                instruction.get(), index);
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

  if (ShapeUtil::IsTuple(hlo_instruction->shape())) {
    // If the hlo instruction is a tuple-shaped, then trivially the instruction
    // itself is the source of the tuple.
    points_to_set.add_tuple_source({}, hlo_instruction);
  }

  return Status::OK();
}

Status TuplePointsToAnalysis::HandleGetTupleElement(
    HloInstruction* get_tuple_element, HloInstruction* operand) {
  // GetTupleElement forwards a pointer to a particular element of the tuple
  // operand.
  int64 element_index = get_tuple_element->tuple_index();

  PointsToSet& points_to_set = CreateEmptyPointsToSet(get_tuple_element);
  const PointsToSet& operand_points_to_set = *PerInst(operand)->points_to_set;

  // Copy the points-to set (and tuple sources) at index {element_index} of the
  // operand to the points-to set for this GetTupleElement instruction.
  points_to_set.ForEachMutableElement(
      [&, this](const ShapeIndex& target_index,
                PointsToSet::BufferList* points_to) {
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

Status TuplePointsToAnalysis::HandleTuple(
    HloInstruction* tuple,
    tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
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

Status TuplePointsToAnalysis::HandleSelect(HloInstruction* select,
                                           HloInstruction* /*pred*/,
                                           HloInstruction* on_true,
                                           HloInstruction* on_false) {
  // Select allocates a new buffer and then shallow copies the on_true or
  // on_false buffer into this new buffer. Which side is chosen cannot be
  // determined statically so conservatively set the points-to set to the union
  // of these on_true and on_false operands.
  //
  // First create a copy of the on_true points-to set (and tuple sources), then
  // add in elements of the on_false points-to set (tuple sources).
  PointsToSet& points_to_set = CreateCopiedPointsToSet(select, on_true);
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
      logical_buffer_analysis_->GetBuffer(select, /*index=*/{}),
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
  auto set = MakeUnique<PointsToSet>(&instruction->shape());
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
        buffer.ToString().c_str(), buffer.instruction()->name().c_str());
  }

  if (buffer.id() < 0 ||
      buffer.id() >= logical_buffer_analysis_->num_logical_buffers()) {
    return FailedPrecondition(
        "LogicalBuffer %s is ill-defined: invalid id %lld",
        buffer.ToString().c_str(), buffer.id());
  }
  if (GetBuffer(buffer.id()).instruction() != buffer.instruction() ||
      GetBuffer(buffer.id()).index() != buffer.index()) {
    return FailedPrecondition(
        "LogicalBuffer %s is ill-defined: buffer with same id differs: %s",
        buffer.ToString().c_str(), GetBuffer(buffer.id()).ToString().c_str());
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
        instruction->name().c_str(),
        tensorflow::str_util::Join(index, ",").c_str());
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
      .ForEachElement([this, buffers, instruction](
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
      [this, &dst_points_to_set, &src_points_to_set](
          const ShapeIndex& index, PointsToSet::BufferList* buffers) {
        *buffers = src_points_to_set.element(index);
        for (auto& tuple_source : src_points_to_set.tuple_sources(index)) {
          dst_points_to_set.add_tuple_source(index, tuple_source);
        }
      });
  return *PerInst(instruction)->points_to_set;
}

string TuplePointsToAnalysis::ToString() const {
  string output = tensorflow::strings::Printf(
      "TuplePointsToSet for module %s:\n", module_->name().c_str());
  for (const auto& computation : module_->computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    const char* entry =
        computation.get() == module_->entry_computation() ? "entry " : "";
    tensorflow::strings::StrAppend(&output, entry, "computation ",
                                   computation->name(), ":\n");
    for (const HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      InstructionToString(instruction, &output);
      if (instruction->opcode() == HloOpcode::kFusion) {
        for (auto& fused : instruction->fused_instructions()) {
          InstructionToString(fused.get(), &output);
        }
      }
    }
  }

  tensorflow::strings::StrAppend(&output, "LogicalBuffers:\n");
  for (const auto& b : logical_buffer_analysis_->logical_buffers()) {
    tensorflow::strings::StrAppend(&output, "  buffer ", b->ToString(), ":\n");
    for (const BufferAlias& alias : logical_buffer_aliases_.at(b->id())) {
      tensorflow::strings::StrAppend(&output, "    alias ", alias.ToString(),
                                     "\n");
    }
  }
  return output;
}

void TuplePointsToAnalysis::InstructionToString(
    const HloInstruction* instruction, string* output) const {
  const string prefix = instruction->IsFused() ? "    " : "";
  tensorflow::strings::StrAppend(output, prefix, "  instruction ",
                                 instruction->ToShortString(), ":\n");
  const PointsToSet& points_to_set = GetPointsToSet(instruction);
  points_to_set.ForEachElement([&prefix, &output](
                                   const ShapeIndex& index,
                                   const PointsToSet::BufferList& points_to) {
    tensorflow::strings::StrAppend(
        output, prefix, "    {", tensorflow::str_util::Join(index, ","), "}: ",
        tensorflow::str_util::Join(
            points_to, ", ",
            [](string* out, const LogicalBuffer* source) {
              out->append(source->ToString());
            }),
        "\n");
  });
}

}  // namespace xla
