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
  return tensorflow::strings::StrCat("BufferAlias(",
                                     instruction_->FullyQualifiedName(), "[",
                                     tensorflow::str_util::Join(index_, ","),
                                     "] => ", buffer_->ToString(), ")");
}

std::ostream& operator<<(std::ostream& out, const BufferAlias& buffer_alias) {
  out << buffer_alias.ToString();
  return out;
}

bool PointsToSet::IsAmbiguous() const {
  bool ambiguous = false;
  TF_CHECK_OK(ForEachElement(
      [&ambiguous](const ShapeIndex& /*index*/, bool /*is_leaf*/,
                   const std::vector<const LogicalBuffer*>& points_to) {
        ambiguous |= points_to.size() > 1;
        return Status::OK();
      }));
  return ambiguous;
}

bool PointsToSet::IsDistinct() const {
  bool distinct = true;
  std::set<const LogicalBuffer*> all_points_to;
  TF_CHECK_OK(ForEachElement([&distinct, &all_points_to](
      const ShapeIndex& /*index*/, bool /*is_leaf*/,
      const std::vector<const LogicalBuffer*>& points_to) {
    for (auto& buffer : points_to) {
      if (all_points_to.count(buffer) != 0) {
        distinct = false;
      }
      all_points_to.insert(buffer);
    }
    return Status::OK();
  }));
  return distinct;
}

size_t PointsToSet::size() const {
  // Because pointed-to elements may be duplicated we have to create a flattened
  // set and return the size.
  return CreateFlattenedSet().size();
}

tensorflow::gtl::FlatSet<const LogicalBuffer*> PointsToSet::CreateFlattenedSet()
    const {
  tensorflow::gtl::FlatSet<const LogicalBuffer*> flat_set;
  TF_CHECK_OK(ForEachElement(
      [&flat_set](const ShapeIndex& /*index*/, bool /*is_leaf*/,
                  const std::vector<const LogicalBuffer*>& buffers) {
        flat_set.insert(buffers.begin(), buffers.end());
        return Status::OK();
      }));
  return flat_set;
}

bool PointsToSet::ContainsBuffer(const LogicalBuffer& buffer) const {
  bool found = false;
  TF_CHECK_OK(ForEachElement([&found, &buffer](
      const ShapeIndex& /*index*/, bool /*is_leaf*/,
      const std::vector<const LogicalBuffer*>& pointed_to_buffers) {
    if (!found &&
        std::find(pointed_to_buffers.begin(), pointed_to_buffers.end(),
                  &buffer) != pointed_to_buffers.end()) {
      found = true;
    }
    return Status::OK();
  }));
  return found;
}

bool PointsToSet::ContainsBufferAtIndex(const LogicalBuffer& buffer,
                                        const ShapeIndex& index) const {
  const std::vector<const LogicalBuffer*>& pointed_to_buffers = element(index);
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

const std::set<HloInstruction*>& PointsToSet::tuple_sources(
    const ShapeIndex& index) const {
  return tuple_sources_.element(index);
}

void PointsToSet::add_tuple_source(const ShapeIndex& index,
                                   HloInstruction* tuple) {
  tuple_sources_.mutable_element(index)->insert(tuple);
}

/* static */ StatusOr<std::unique_ptr<TuplePointsToAnalysis>>
TuplePointsToAnalysis::Run(const HloModule* module,
                           const bool include_loop_fusion_instructions) {
  std::unique_ptr<TuplePointsToAnalysis> analysis(
      new TuplePointsToAnalysis(module, include_loop_fusion_instructions));
  TF_RETURN_IF_ERROR(analysis->Analyze());
  return std::move(analysis);
}

Status TuplePointsToAnalysis::Analyze() {
  points_to_.clear();
  for (auto& computation : module_->computations()) {
    TF_RETURN_IF_ERROR(computation->Accept(this));
    TF_RETURN_IF_ERROR(
        PopulateDefinedBuffersAndAliases(computation->instructions()));
    if (include_loop_fusion_instructions_) {
      // Run points-to analysis on loop fusion instructions in 'computation'.
      for (auto& instruction : computation->instructions()) {
        if (instruction->opcode() != HloOpcode::kFusion ||
            instruction->fusion_kind() != HloInstruction::FusionKind::kLoop) {
          continue;
        }
        TF_RETURN_IF_ERROR(instruction->fused_expression_root()->Accept(this));
        TF_RETURN_IF_ERROR(PopulateDefinedBuffersAndAliases(
            instruction->fused_instructions()));
      }
    }
  }

  XLA_VLOG_LINES(3, ToString());

  return Status::OK();
}

Status TuplePointsToAnalysis::PopulateDefinedBuffersAndAliases(
    const std::list<std::unique_ptr<HloInstruction>>& instructions) {
  for (auto& instruction : instructions) {
    TF_RETURN_IF_ERROR(GatherBuffersDefinedByInstruction(
        instruction.get(), &instruction_defined_buffers_[instruction.get()]));

    const PointsToSet& points_to_set = GetPointsToSet(instruction.get());
    TF_RETURN_IF_ERROR(points_to_set.ForEachElement([this, &instruction](
        const ShapeIndex& index, bool /*is_leaf*/,
        const std::vector<const LogicalBuffer*>& pointed_to_buffers) {
      for (const LogicalBuffer* buffer : pointed_to_buffers) {
        if (buffer_aliases_.count(buffer) == 0) {
          buffer_aliases_.insert({buffer, std::vector<BufferAlias>()});
        }
        buffer_aliases_[buffer].emplace_back(*buffer, instruction.get(), index);
      }
      return Status::OK();
    }));
  }
  return Status::OK();
}

const LogicalBuffer& TuplePointsToAnalysis::NewLogicalBuffer(
    HloInstruction* instruction, const ShapeIndex& index) {
  CHECK_EQ(logical_buffers_.size(), next_buffer_id_);
  logical_buffers_.push_back(
      MakeUnique<LogicalBuffer>(instruction, index, next_buffer_id_));
  ++next_buffer_id_;
  return *logical_buffers_.back();
}

Status TuplePointsToAnalysis::DefaultAction(HloInstruction* hlo_instruction) {
  // Create trivial points-to set for instruction. Each points-to set at index i
  // contains a single element LogicalBuffer(hlo_instruction, i). This indicates
  // that this instruction is the source of all buffers in its own output.
  PointsToSet& points_to_set = CreateEmptyPointsToSet(hlo_instruction);
  TF_RETURN_IF_ERROR(points_to_set.ForEachMutableElement(
      [this, hlo_instruction](const ShapeIndex& index, bool /*is_leaf*/,
                              std::vector<const LogicalBuffer*>* buffers) {
        const LogicalBuffer& buffer = NewLogicalBuffer(hlo_instruction, index);
        buffers->push_back(&buffer);
        return Status::OK();
      }));

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
  const PointsToSet& operand_points_to_set = *FindOrDie(points_to_, operand);

  // Copy the points-to set (and tuple sources) at index {element_index} of the
  // operand to the points-to set for this GetTupleElement instruction.
  TF_RETURN_IF_ERROR(points_to_set.ForEachMutableElement([&, this](
      const ShapeIndex& target_index, bool /*is_leaf*/,
      std::vector<const LogicalBuffer*>* points_to) {
    // Construct an index into the operand by prepending element_index to the
    // index for the GetTupleElement instruction's points-to set.
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
    return Status::OK();
  }));

  return Status::OK();
}

Status TuplePointsToAnalysis::HandleCopy(HloInstruction* copy,
                                         HloInstruction* operand) {
  // A kCopy instruction performs a shallow copy of the operand. The top-level
  // buffer (index={}) is newly created, but all other buffers (in the case of a
  // tuple shape) come from the operand
  PointsToSet& points_to_set = CreateCopiedPointsToSet(copy, operand);
  points_to_set.mutable_element(/*index=*/{})->clear();
  points_to_set.AddPointedToBuffer(NewLogicalBuffer(copy, /*index=*/{}),
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
  points_to_set.AddPointedToBuffer(NewLogicalBuffer(tuple, /*index=*/{}),
                                   /*index=*/{});

  // A tuple contains references to all input operands and transitively any
  // references in those operands.
  for (int64 i = 0; i < operands.size(); ++i) {
    const PointsToSet& operand_points_to_set =
        *FindOrDie(points_to_, operands[i]);

    // Copy the points-to set (and tuple sources) of the operand into the
    // respective subtree of the tuple instructions points-to set.
    TF_RETURN_IF_ERROR(operand_points_to_set.ForEachElement(
        [&points_to_set, &operand_points_to_set, i](
            const ShapeIndex& src_index, bool /*is_leaf*/,
            const std::vector<const LogicalBuffer*>& points_to) {
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
          return Status::OK();
        }));
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
  const PointsToSet& false_points_to_set = *FindOrDie(points_to_, on_false);
  TF_RETURN_IF_ERROR(points_to_set.ForEachMutableElement(
      [&](const ShapeIndex& index, bool /*is_leaf*/,
          std::vector<const LogicalBuffer*>* buffers) {
        for (const LogicalBuffer* false_buffer :
             false_points_to_set.element(index)) {
          points_to_set.AddPointedToBuffer(*false_buffer, index);
        }

        for (HloInstruction* tuple : false_points_to_set.tuple_sources(index)) {
          points_to_set.add_tuple_source(index, tuple);
        }
        return Status::OK();
      }));

  // Select creates a new (top-level) buffer to store its result, so its
  // respective element in the points-to set should contain only itself.
  points_to_set.mutable_element({})->clear();
  points_to_set.AddPointedToBuffer(NewLogicalBuffer(select, /*index=*/{}),
                                   /*index=*/{});
  return Status::OK();
}

Status TuplePointsToAnalysis::HandleFusion(HloInstruction* fusion) {
  return ShapeUtil::IsTuple(fusion->shape())
             ? Unimplemented("HandleFusion with tuple output")
             : DefaultAction(fusion);
}

const PointsToSet& TuplePointsToAnalysis::GetPointsToSet(
    const HloInstruction* hlo_instruction) const {
  return *FindOrDie(points_to_, hlo_instruction);
}

PointsToSet& TuplePointsToAnalysis::CreateEmptyPointsToSet(
    const HloInstruction* instruction) {
  CHECK_EQ(0, points_to_.count(instruction));
  points_to_[instruction] = MakeUnique<PointsToSet>(instruction->shape());
  return *FindOrDie(points_to_, instruction);
}

bool TuplePointsToAnalysis::InstructionDefinesBufferAtIndex(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  const std::vector<const LogicalBuffer*>& buffers =
      GetPointsToSet(instruction).element(index);
  return (buffers.size() == 1 && buffers[0]->instruction() == instruction);
}

Status TuplePointsToAnalysis::VerifyBuffer(const LogicalBuffer& buffer) const {
  if (!InstructionDefinesBufferAtIndex(buffer.instruction(), buffer.index())) {
    return FailedPrecondition(
        "LogicalBuffer %s is ill-defined: instruction %s does not define a "
        "buffer at that index",
        buffer.ToString().c_str(), buffer.instruction()->name().c_str());
  }

  if (buffer.id() < 0 || buffer.id() >= next_buffer_id_) {
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
  CHECK_LT(id, logical_buffers_.size());
  return *logical_buffers_[id];
}

StatusOr<const LogicalBuffer*> TuplePointsToAnalysis::GetBufferDefinedAt(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  const std::vector<const LogicalBuffer*>& buffers =
      GetPointsToSet(instruction).element(index);
  if (buffers.size() != 1 || buffers[0]->instruction() != instruction) {
    return FailedPrecondition(
        "instruction %s does not define buffer at index {%s}",
        instruction->name().c_str(),
        tensorflow::str_util::Join(index, ",").c_str());
  }
  return buffers[0];
}

const std::vector<BufferAlias>& TuplePointsToAnalysis::GetBufferAliases(
    const LogicalBuffer& buffer) const {
  return buffer_aliases_.at(&buffer);
}

const std::vector<const LogicalBuffer*>&
TuplePointsToAnalysis::GetBuffersDefinedByInstruction(
    const HloInstruction* instruction) const {
  return instruction_defined_buffers_.at(instruction);
}

Status TuplePointsToAnalysis::GatherBuffersDefinedByInstruction(
    const HloInstruction* instruction,
    std::vector<const LogicalBuffer*>* buffers) {
  return GetPointsToSet(instruction)
      .ForEachElement([this, buffers, instruction](
          const ShapeIndex& index, bool /*is_leaf*/,
          const std::vector<const LogicalBuffer*>& source_buffers) {
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
        return Status::OK();
      });
}

PointsToSet& TuplePointsToAnalysis::CreateCopiedPointsToSet(
    const HloInstruction* instruction, const HloInstruction* src) {
  // PointsToSet doesn't have a copy constructor so copy over element-by-element
  // from src PointsToSet.
  PointsToSet& dst_points_to_set = CreateEmptyPointsToSet(instruction);
  const PointsToSet& src_points_to_set = GetPointsToSet(src);
  TF_CHECK_OK(dst_points_to_set.ForEachMutableElement(
      [this, &dst_points_to_set, &src_points_to_set](
          const ShapeIndex& index, bool /*is_leaf*/,
          std::vector<const LogicalBuffer*>* buffers) {
        *buffers = src_points_to_set.element(index);
        for (auto& tuple_source : src_points_to_set.tuple_sources(index)) {
          dst_points_to_set.add_tuple_source(index, tuple_source);
        }
        return Status::OK();
      }));
  return *FindOrDie(points_to_, instruction);
}

string TuplePointsToAnalysis::ToString() const {
  string output = tensorflow::strings::Printf(
      "TuplePointsToSet for module %s:\n", module_->name().c_str());
  for (const auto& computation : module_->computations()) {
    const char* entry =
        computation.get() == module_->entry_computation() ? "entry " : "";
    tensorflow::strings::StrAppend(&output, entry, "computation ",
                                   computation->name(), ":\n");
    for (const HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      InstructionToString(instruction, &output);
      if (include_loop_fusion_instructions_ &&
          instruction->opcode() == HloOpcode::kFusion &&
          instruction->fusion_kind() == HloInstruction::FusionKind::kLoop) {
        for (auto& fused : instruction->fused_instructions()) {
          InstructionToString(fused.get(), &output);
        }
      }
    }
  }

  tensorflow::strings::StrAppend(&output, "LogicalBuffers:\n");
  for (auto& buffer : logical_buffers_) {
    tensorflow::strings::StrAppend(&output, "  buffer ", buffer->ToString(),
                                   ":\n");
    for (const BufferAlias& buffer_alias : buffer_aliases_.at(buffer.get())) {
      tensorflow::strings::StrAppend(&output, "    alias ",
                                     buffer_alias.ToString(), "\n");
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
  TF_CHECK_OK(points_to_set.ForEachElement([&prefix, &output](
      const ShapeIndex& index, bool /*is_leaf*/,
      const std::vector<const LogicalBuffer*>& points_to) {
    tensorflow::strings::StrAppend(
        output, prefix, "    {", tensorflow::str_util::Join(index, ","), "}: ",
        tensorflow::str_util::Join(
            points_to, ", ",
            [](string* out, const LogicalBuffer* source) {
              out->append(source->ToString());
            }),
        "\n");
    return Status::OK();
  }));
}

}  // namespace xla
