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

#include "tensorflow/compiler/xla/service/copy_insertion.h"

#include <memory>
#include <set>
#include <string>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace {

// InstructionCopier encapsulates indices at which to copy 'instruction'.
// All 'instruction' users in 'copy_users' are updated to use the copy.
//
// Instruction copies are generated in two phases:
// 1) Recording buffer indices at which 'instruction' requires copies (i.e.
//    setting 'indices_to_copy_[index]'=true).
// 2) Inserting kCopy instructions based on indices recorded in phase 1).
//   *) Array instructions are copied by inserting a single kCopy instruction.
//   *) Tuple-shaped instructions are copied by recursively expanding tuples
//      (and tuple-shaped elements), and inserting kCopy instructions for any
//      tuple elements which require a copy. As the recursion unwinds, new tuple
//      instructions are added to gather the copied (and uncopied) references
//      into the output tuple (i.e. the copy of the tuple-shaped instruction).
//
//      Example two-element tuple with one element that needs a copy:
//
//                    Tuple  // instruction
//                   /    \
//                GTE(0)  GTE(1)
//                  |       |
//                 Copy     |
//                   \     /
//                    Tuple  // copied-instruction
//
class InstructionCopier {
 public:
  InstructionCopier(const bool init_value, HloInstruction* instruction,
                    const std::vector<HloInstruction*>& copy_users);

  // Returns true if all recorded indices are false (returns true otherwise).
  bool HasAllIndicesFalse() const;

  // Records instruction buffer indices which point-to a Parameter or Constant.
  tensorflow::Status RecordIndicesWhichPointToParamOrConstant(
      const TuplePointsToAnalysis& points_to_analysis);

  // Records instruction buffer indices to copy which are necessary to ensure:
  // *) PointsToSet of 'instruction_' is unambiguous and distinct.
  // *) No liveness interference between 'instruction_' and 'other_instruction'.
  tensorflow::Status RecordIndicesToCopyForColocatingBuffers(
      BufferLiveness* liveness, HloInstruction* other_instruction);

  // Inserts copies of 'instruction' buffers at indices in 'indices_to_copy',
  // and replaces all uses for instructions in 'copy_users_' with copy.
  // Returns the instruction which is a copy 'instruction'.
  HloInstruction* Copy();

  HloInstruction* instruction() { return instruction_; }

  const std::vector<HloInstruction*>& copy_users() const { return copy_users_; }

 private:
  // Records instruction buffer indices which have ambiguous or non-distinct
  // points-to sets.
  tensorflow::Status RecordAmbiguousOrNonDistinctIndices(
      const TuplePointsToAnalysis& points_to_analysis);

  // Records instruction buffer indices which have interferring live ranges
  // with 'other_instruction' buffers at same index.
  tensorflow::Status RecordIndicesWhichInterfereWithOtherInstruction(
      BufferLiveness* liveness, HloInstruction* other_instruction);

  // Recursively inserts copies of 'instruction' tuple elements at indices
  // specified in 'indices_to_copy', and returns the copy of 'instruction'.
  HloInstruction* CopyTuple(HloInstruction* instruction, ShapeIndex* index);

  void RecordIndex(const ShapeIndex& index) {
    *indices_to_copy_.mutable_element(index) = true;
  }

  HloInstruction* instruction_;
  std::vector<HloInstruction*> copy_users_;
  ShapeTree<bool> indices_to_copy_;
};

InstructionCopier::InstructionCopier(
    const bool init_value, HloInstruction* instruction,
    const std::vector<HloInstruction*>& copy_users)
    : instruction_(instruction),
      copy_users_(copy_users),
      indices_to_copy_(instruction->shape(), init_value) {}

bool InstructionCopier::HasAllIndicesFalse() const {
  bool all_indices_false = true;
  TF_CHECK_OK(indices_to_copy_.ForEachElement([&all_indices_false](
      const ShapeIndex& /*index*/, bool /*is_leaf*/, const bool& data) {
    if (data) all_indices_false = false;
    return tensorflow::Status::OK();
  }));
  return all_indices_false;
}

tensorflow::Status InstructionCopier::RecordIndicesWhichPointToParamOrConstant(
    const TuplePointsToAnalysis& points_to_analysis) {
  const PointsToSet& points_to =
      points_to_analysis.GetPointsToSet(instruction_);
  // Shallow copy the instruction if the points-to set of the top-level
  // buffer is ambiguous. This is necessary because the backends must know
  // statically what the top-level buffer of the result is.
  if (points_to.element(/*index=*/{}).size() > 1) {
    RecordIndex({});
  }

  // Multiple buffers within a parameter/constant may be live out, so collect
  // a set of indices at which to copy first.
  TF_RETURN_IF_ERROR(points_to.ForEachElement([this](
      const ShapeIndex& index, bool /*is_leaf*/,
      const std::vector<const LogicalBuffer*>& buffers) {
    for (auto buffer : buffers) {
      // pointee is the HloInstruction producing the buffer which may be
      // liveout.
      HloInstruction* pointee = buffer->instruction();
      if (pointee->opcode() == HloOpcode::kParameter ||
          pointee->opcode() == HloOpcode::kConstant) {
        VLOG(2) << "Parameter or constant buffer " << buffer->ToString()
                << " index: " << tensorflow::str_util::Join(index, ",")
                << " may be live out of computation: " << pointee->ToString();
        RecordIndex(index);
      }
    }
    return tensorflow::Status::OK();
  }));
  return tensorflow::Status::OK();
}

tensorflow::Status InstructionCopier::RecordIndicesToCopyForColocatingBuffers(
    BufferLiveness* liveness, HloInstruction* other_instruction) {
  TF_RETURN_IF_ERROR(
      RecordAmbiguousOrNonDistinctIndices(liveness->points_to_analysis()));
  TF_RETURN_IF_ERROR(RecordIndicesWhichInterfereWithOtherInstruction(
      liveness, other_instruction));
  return tensorflow::Status::OK();
}

tensorflow::Status InstructionCopier::RecordAmbiguousOrNonDistinctIndices(
    const TuplePointsToAnalysis& points_to_analysis) {
  const PointsToSet& points_to =
      points_to_analysis.GetPointsToSet(instruction_);
  // Mapping from LogicalBuffer to index (used to detect non-distinct indices).
  std::unordered_map<const LogicalBuffer*, std::vector<ShapeIndex>>
      buffer_to_source_indices;
  TF_RETURN_IF_ERROR(points_to.ForEachElement([this, &buffer_to_source_indices](
      const ShapeIndex& index, bool /*is_leaf*/,
      const std::vector<const LogicalBuffer*>& buffers) {
    if (buffers.size() > 1) {
      // Record ambiguous points-to set at 'index'.
      if (!indices_to_copy_.element(index)) {
        VLOG(2) << "Adding copy of buffer for instruction: "
                << instruction_->name()
                << " at index: " << tensorflow::str_util::Join(index, ",")
                << " with ambiguous points-to set.";
        RecordIndex(index);
      }
    }
    // For each 'buffer': record a mapping from 'buffer' to 'index'.
    for (auto& buffer : buffers) {
      auto it = buffer_to_source_indices.find(buffer);
      if (it == buffer_to_source_indices.end()) {
        buffer_to_source_indices.insert({buffer, std::vector<ShapeIndex>()});
      }
      buffer_to_source_indices[buffer].push_back(index);
    }
    return tensorflow::Status::OK();
  }));

  // Record all non-distinct indices detected in 'buffer_to_source_indices'.
  for (auto& buff_to_src : buffer_to_source_indices) {
    if (buff_to_src.second.size() == 1) {
      continue;
    }
    for (auto& src_index : buff_to_src.second) {
      // Record non-distinct points-to set at 'src_index'.
      if (!indices_to_copy_.element(src_index)) {
        VLOG(2) << "Adding copy of buffer for instruction: "
                << instruction_->name()
                << " at index: " << tensorflow::str_util::Join(src_index, ",")
                << " because of non-distinct points-to set.";
        RecordIndex(src_index);
      }
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status
InstructionCopier::RecordIndicesWhichInterfereWithOtherInstruction(
    BufferLiveness* liveness, HloInstruction* other_instruction) {
  // Record all buffer indices for 'instruction_', which interfere with
  // 'other_instruction' at the same index.
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshape(
      instruction_->shape(),
      [this, &liveness, &other_instruction](const Shape& /*subshape*/,
                                            const ShapeIndex& index) {
        if (indices_to_copy_.element(index)) {
          // Return if previous pass already set index.
          return tensorflow::Status::OK();
        }
        auto& points_to_analysis = liveness->points_to_analysis();
        // Lookup buffers for 'instruction_' and 'other_instruction'.
        const std::vector<const LogicalBuffer*> instruction_buffers =
            points_to_analysis.GetPointsToSet(instruction_).element(index);
        // If 'instruction_' has ambiguous points-to-set  at 'index', it would
        // have been recorded in a previous pass (and we would have returned
        // early at the entry to this function). As a result, here we know that
        // 'instruction_' has just one buffer in its points-to-set.
        CHECK_EQ(1, instruction_buffers.size());
        const LogicalBuffer* instruction_buffer = instruction_buffers[0];

        const std::vector<const LogicalBuffer*> other_instruction_buffers =
            points_to_analysis.GetPointsToSet(other_instruction).element(index);
        // Do not insert a copy if both instructions point at the same buffer.
        // This eliminates unnecessary copies of read-only tuple elements.
        // If 'instruction_' and 'other_instruction' point to the same buffer,
        // then that buffer is not updated on the path between the two
        // instructions. Therefore, any other (possibly interference-causing)
        // users of that buffer from 'other_instruction' will see the same data,
        // irrespecive of whether we insert a copy of this buffer at
        // 'instruction_' or not.
        if (other_instruction_buffers.size() == 1 &&
            other_instruction_buffers[0]->id() == instruction_buffer->id()) {
          return tensorflow::Status::OK();
        }
        // We cant say anything about the ambiguity of 'other_instruction' at
        // this point, so we need to check interference between the single
        // buffer in the points-to set of 'instruction_' and all buffers in
        // 'other_instruction_buffers'.
        for (auto& other_buffer : other_instruction_buffers) {
          if (liveness->MayInterfere(*instruction_buffer, *other_buffer)) {
            VLOG(2) << "Adding copy of buffer for instruction: "
                    << instruction_->name()
                    << " at index: " << tensorflow::str_util::Join(index, ",")
                    << " because of interference with buffer: "
                    << other_buffer->ToString();
            RecordIndex(index);
            break;
          }
        }
        return tensorflow::Status::OK();
      }));
  return tensorflow::Status::OK();
}

// Recursively inserts copies of 'instruction' tuple element buffers at
// indices in 'indices_to_copy_', expanding tuples as needed.
// TODO(b/31159897) Remove superfluous Tuple->GTE->Tuple expressions.
HloInstruction* InstructionCopier::CopyTuple(HloInstruction* instruction,
                                             ShapeIndex* index) {
  std::vector<HloInstruction*> element_copies;
  const int64 num_tuple_elements =
      ShapeUtil::TupleElementCount(instruction->shape());
  for (int64 i = 0; i < num_tuple_elements; ++i) {
    HloInstruction* gte = instruction->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(
            ShapeUtil::GetSubshape(instruction->shape(), {i}), instruction, i));
    HloInstruction* element_copy;
    index->push_back(i);
    if (ShapeUtil::IsTuple(gte->shape())) {
      element_copy = CopyTuple(gte, index);
    } else {
      if (indices_to_copy_.element(*index)) {
        element_copy = gte->parent()->AddInstruction(
            HloInstruction::CreateUnary(gte->shape(), HloOpcode::kCopy, gte));
      } else {
        element_copy = gte;
      }
    }
    index->pop_back();
    element_copies.push_back(element_copy);
  }
  return instruction->parent()->AddInstruction(
      HloInstruction::CreateTuple(element_copies));
}

// Inserts copies of 'instruction_' buffers at indices in 'indices_to_copy_'.
HloInstruction* InstructionCopier::Copy() {
  ShapeIndex index;
  HloInstruction* copy;
  if (ShapeUtil::IsTuple(instruction_->shape())) {
    copy = CopyTuple(instruction_, &index);
  } else {
    copy = instruction_->parent()->AddInstruction(HloInstruction::CreateUnary(
        instruction_->shape(), HloOpcode::kCopy, instruction_));
  }
  for (HloInstruction* user : copy_users_) {
    VLOG(2) << "Adding copy between instruction: " << instruction_->name()
            << " and user: " << user->name();
    TF_CHECK_OK(instruction_->ReplaceUseWith(user, copy));
  }
  return copy;
}

}  // anonymous namespace

StatusOr<HloInstruction*> CopyInsertion::FindOrInsertCopy(HloInstruction* hlo) {
  auto copy_it = inserted_copies_.find(hlo);
  if (copy_it == inserted_copies_.end()) {
    HloInstruction* copy = hlo->parent()->DeepCopyInstruction(hlo).ValueOrDie();
    inserted_copies_.insert({hlo, copy});
    return copy;
  } else {
    return copy_it->second;
  }
}

StatusOr<bool> CopyInsertion::Run(HloModule* module) {
  bool changed = false;
  VLOG(2) << "CopyInsertion for module " << module->name();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BufferLiveness> liveness,
      BufferLiveness::Run(module, MakeUnique<DependencyHloOrdering>(module)));
  auto& points_to_analysis = liveness->points_to_analysis();
  XLA_VLOG_LINES(2, points_to_analysis.ToString());
  XLA_VLOG_LINES(2, module->ToString());

  // Gather references to all while body computations in 'module'.
  std::unordered_set<const HloComputation*> while_body_computations;
  // Gather references to all while instructions in 'module' by computation.
  std::unordered_map<const HloComputation*, std::vector<HloInstruction*>>
      while_instructions;
  for (auto& computation : module->computations()) {
    for (auto& instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kWhile) {
        continue;
      }
      while_body_computations.insert(instruction->while_body());
      auto it = while_instructions.find(computation.get());
      if (it == while_instructions.end()) {
        while_instructions.insert(
            {computation.get(), std::vector<HloInstruction*>()});
      }
      while_instructions[computation.get()].emplace_back(instruction.get());
    }
  }

  for (auto& computation : module->computations()) {
    VLOG(2) << "computation " << computation->name();

    // Collect instruction buffer indices to copy in 'instructions_to_copy'.
    std::vector<InstructionCopier> instructions_to_copy;

    // Add copies of while 'init' operand instructions (if needed).
    // TODO(b/33301720) Remove redundant while instruction copies.
    auto it = while_instructions.find(computation.get());
    if (it != while_instructions.end()) {
      for (auto& while_hlo : it->second) {
        // Create InstructionCopier for init operand of while instruction.
        HloInstruction* init_hlo = while_hlo->mutable_operand(0);
        instructions_to_copy.push_back(
            InstructionCopier(/*init_value=*/false, init_hlo, {while_hlo}));
        InstructionCopier& init_copier = instructions_to_copy.back();
        // Record 'init' buffer indices which point-to a Constant or Parameter.
        TF_RETURN_IF_ERROR(init_copier.RecordIndicesWhichPointToParamOrConstant(
            liveness->points_to_analysis()));
        // Record indices necessary to colocate while and init operand buffers.
        TF_RETURN_IF_ERROR(init_copier.RecordIndicesToCopyForColocatingBuffers(
            liveness.get(), while_hlo));
      }
    }

    // Create InstructionCopier for computation root instruction.
    instructions_to_copy.push_back(InstructionCopier(
        /*init_value=*/false, computation->root_instruction(), {}));
    InstructionCopier& root_copier = instructions_to_copy.back();

    if (while_body_computations.count(computation.get()) > 0) {
      // Record root indices to copy for while body sub-computations.
      // We do not need to call RecordIndicesWhichPointToParamOrConstant for
      // the while root instruction here, because any neccessary copies needed
      // to avoid constant or parameters in the output are handled by while.init
      // operand copy insertion above (which will share an allocation).
      TF_RETURN_IF_ERROR(root_copier.RecordIndicesToCopyForColocatingBuffers(
          liveness.get(), computation->parameter_instruction(0)));
    } else {
      // Record root indices to copy for general computations.
      TF_RETURN_IF_ERROR(root_copier.RecordIndicesWhichPointToParamOrConstant(
          liveness->points_to_analysis()));
    }

    for (auto& to_copy : instructions_to_copy) {
      if (to_copy.HasAllIndicesFalse()) {
        continue;
      }
      changed = true;

      // Copy instruction at recorded buffer indices.
      HloInstruction* copy = to_copy.Copy();
      if (to_copy.instruction() == computation->root_instruction()) {
        computation->set_root_instruction(copy);
      }
    }
  }

  VLOG(3) << "After copy insertion for module " << module->name();
  XLA_VLOG_LINES(3, module->ToString());

  return changed;
}

}  // namespace xla
