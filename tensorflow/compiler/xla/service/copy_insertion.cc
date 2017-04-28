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
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace {

using tensorflow::gtl::FlatMap;
using tensorflow::gtl::FlatSet;

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
//             original-instruction
//                   /    \
//                GTE(0)  GTE(1)
//                  |       |
//                 Copy     |
//                   \     /
//                    Tuple  // copied-instruction
//
//      As an optimization, if the original instruction is itself a Tuple
//      instruction, we elide the unnecessary extra GTE and Tuple instructions,
//      and just insert the copy into a new Tuple instruction, with control
//      dependencies to ensure the copy occurs after any possible interference.
class InstructionCopier {
 public:
  InstructionCopier(HloInstruction* instruction,
                    const std::vector<HloInstruction*>& copy_users)
      : instruction_(instruction),
        copy_users_(copy_users),
        indices_to_copy_(instruction->shape()),
        control_predecessors_(instruction->shape()) {}

  // Sets indices that are read-only, and thus do not need to be copied.
  void SetReadOnlyIndices(const ShapeTree<bool>& read_only_indices) {
    read_only_indices_ = read_only_indices;
  }

  // Sets copy overrides, which are copy instructions to use at each index. This
  // is used to share a single copy of read-only entry parameters and constants
  // between multiple While loops.
  void SetCopyOverrides(const ShapeTree<HloInstruction*>& copy_overrides) {
    copy_overrides_ = copy_overrides;
  }

  // Returns true if all recorded indices are false (returns true otherwise).
  bool HasAllIndicesFalse() const;

  // Records instruction buffer indices which point-to a Parameter or Constant.
  Status RecordIndicesWhichPointToParamOrConstant(
      const TuplePointsToAnalysis& points_to_analysis);

  // Records instruction buffer indices to copy which are necessary to ensure:
  // *) PointsToSet of 'instruction_' is unambiguous and distinct.
  // *) No liveness interference between 'instruction_' and 'other_instruction'.
  //
  // If 'read_only_indices_out' is non-null, read-only indices are set to true.
  Status RecordIndicesToCopyForColocatingBuffers(
      const BufferLiveness& liveness, const HloInstruction* other_instruction,
      ShapeTree<bool>* read_only_indices_out);

  // Records control predecessors to add for inserted copy instructions.
  // 'parameter' must have the same shape as the instruction that will be
  // copied, and must define all buffers in the shape. Control predecessors are
  // only recorded for indices that have already been marked for copying.
  Status RecordControlPredecessors(
      const TuplePointsToAnalysis& points_to_analysis,
      HloInstruction* parameter);

  // Inserts copies of 'instruction' buffers at indices in 'indices_to_copy',
  // and replaces all uses for instructions in 'copy_users_' with copy.
  // Returns the instruction which is a copy 'instruction'.
  HloInstruction* Copy();

  HloInstruction* instruction() { return instruction_; }

  const std::vector<HloInstruction*>& copy_users() const { return copy_users_; }

 private:
  // Does the given index represent a read-only buffer?
  bool IsReadOnlyIndex(const ShapeIndex& index) const {
    return !ShapeUtil::IsNil(read_only_indices_.shape()) &&
           read_only_indices_.element(index);
  }

  // Returns the copy override at the given index, or nullptr.
  HloInstruction* GetCopyOverride(const ShapeIndex& index) const {
    return ShapeUtil::IsNil(copy_overrides_.shape())
               ? nullptr
               : copy_overrides_.element(index);
  }

  // Records instruction buffer indices which have ambiguous or non-distinct
  // points-to sets.
  Status RecordAmbiguousOrNonDistinctIndices(
      const TuplePointsToAnalysis& points_to_analysis);

  // Records instruction buffer indices which have interferring live ranges
  // with 'other_instruction' buffers at same index.
  Status RecordIndicesWhichInterfereWithOtherInstruction(
      const BufferLiveness& liveness, const HloInstruction* other_instruction,
      ShapeTree<bool>* read_only_indices_out);

  // Recursively inserts copies of 'instruction' tuple elements at indices
  // specified in 'indices_to_copy', and returns the copy of 'instruction'.
  HloInstruction* CopyTuple(HloInstruction* instruction, ShapeIndex* index);

  void RecordIndex(const ShapeIndex& index) {
    *indices_to_copy_.mutable_element(index) = true;
  }

  HloInstruction* instruction_;
  const std::vector<HloInstruction*> copy_users_;
  ShapeTree<bool> indices_to_copy_;
  ShapeTree<std::vector<HloInstruction*>> control_predecessors_;
  ShapeTree<bool> read_only_indices_;
  ShapeTree<HloInstruction*> copy_overrides_;
};

bool InstructionCopier::HasAllIndicesFalse() const {
  bool all_indices_false = true;
  TF_CHECK_OK(indices_to_copy_.ForEachElement(
      [&all_indices_false](const ShapeIndex& /*index*/, bool /*is_leaf*/,
                           bool data) {
        if (data) all_indices_false = false;
        return tensorflow::Status::OK();
      }));
  return all_indices_false;
}

Status InstructionCopier::RecordIndicesWhichPointToParamOrConstant(
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
  TF_RETURN_IF_ERROR(points_to.ForEachElement(
      [this](const ShapeIndex& index, bool /*is_leaf*/,
             const std::vector<const LogicalBuffer*>& buffers) {
        if (IsReadOnlyIndex(index)) {
          return Status::OK();
        }
        for (const LogicalBuffer* buffer : buffers) {
          // pointee is the HloInstruction producing the buffer which may be
          // liveout.
          HloInstruction* pointee = buffer->instruction();
          if (pointee->opcode() == HloOpcode::kParameter ||
              pointee->opcode() == HloOpcode::kConstant) {
            VLOG(2) << "Parameter or constant buffer " << buffer->ToString()
                    << " index: " << tensorflow::str_util::Join(index, ",")
                    << " may be live out of computation: "
                    << pointee->ToString();
            RecordIndex(index);
            break;
          }
        }
        return Status::OK();
      }));
  return Status::OK();
}

Status InstructionCopier::RecordIndicesToCopyForColocatingBuffers(
    const BufferLiveness& liveness, const HloInstruction* other_instruction,
    ShapeTree<bool>* read_only_indices_out) {
  TF_RETURN_IF_ERROR(
      RecordAmbiguousOrNonDistinctIndices(liveness.points_to_analysis()));
  TF_RETURN_IF_ERROR(RecordIndicesWhichInterfereWithOtherInstruction(
      liveness, other_instruction, read_only_indices_out));
  return Status::OK();
}

Status InstructionCopier::RecordAmbiguousOrNonDistinctIndices(
    const TuplePointsToAnalysis& points_to_analysis) {
  const PointsToSet& points_to =
      points_to_analysis.GetPointsToSet(instruction_);
  // Mapping from LogicalBuffer to index (used to detect non-distinct indices).
  FlatMap<const LogicalBuffer*, std::vector<ShapeIndex>>
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
    for (const LogicalBuffer* buffer : buffers) {
      buffer_to_source_indices[buffer].push_back(index);
    }
    return Status::OK();
  }));

  // Record all non-distinct indices detected in 'buffer_to_source_indices'.
  for (const auto& buff_to_src : buffer_to_source_indices) {
    if (buff_to_src.second.size() == 1) {
      continue;
    }
    for (const ShapeIndex& src_index : buff_to_src.second) {
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
  return Status::OK();
}

Status InstructionCopier::RecordIndicesWhichInterfereWithOtherInstruction(
    const BufferLiveness& liveness, const HloInstruction* other_instruction,
    ShapeTree<bool>* read_only_indices_out) {
  // Record all buffer indices for 'instruction_', which interfere with
  // 'other_instruction' at the same index.
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshape(
      instruction_->shape(),
      [this, &liveness, other_instruction, read_only_indices_out](
          const Shape& /*subshape*/, const ShapeIndex& index) {
        if (IsReadOnlyIndex(index)) {
          return Status::OK();
        }
        if (indices_to_copy_.element(index)) {
          // Return if previous pass already set index.
          return Status::OK();
        }
        const auto& points_to_analysis = liveness.points_to_analysis();
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
        // irrespective of whether we insert a copy of this buffer at
        // 'instruction_' or not.
        if (other_instruction_buffers.size() == 1 &&
            other_instruction_buffers[0]->id() == instruction_buffer->id()) {
          if (read_only_indices_out != nullptr) {
            *read_only_indices_out->mutable_element(index) = true;
          }
          return Status::OK();
        }
        // We can't say anything about the ambiguity of 'other_instruction' at
        // this point, so we need to check interference between the single
        // buffer in the points-to set of 'instruction_' and all buffers in
        // 'other_instruction_buffers'.
        for (const LogicalBuffer* other_buffer : other_instruction_buffers) {
          if (liveness.MayInterfere(*instruction_buffer, *other_buffer)) {
            VLOG(2) << "Adding copy of buffer for instruction: "
                    << instruction_->name()
                    << " at index: " << tensorflow::str_util::Join(index, ",")
                    << " because of interference with buffer: "
                    << other_buffer->ToString();
            RecordIndex(index);
            break;
          }
        }
        return Status::OK();
      }));
  return Status::OK();
}

// This is called when 'instruction_' is a while body root, and 'parameter' is
// the while body parameter. We record all users of all aliases of 'parameter'
// as control predecessors, so that when we add a copy of 'instruction_', we can
// mark the control dependencies. This is necessary because points-to and
// liveness analysis doesn't know about the aliasing between the while body root
// and param. Without these control dependencies, the copy might get scheduled
// to run at a point that interferes with users of the buffer.
Status InstructionCopier::RecordControlPredecessors(
    const TuplePointsToAnalysis& points_to_analysis,
    HloInstruction* parameter) {
  return indices_to_copy_.ForEachElement(
      [this, &points_to_analysis, parameter](const ShapeIndex& index,
                                             bool /*is_leaf*/, bool will_copy) {
        if (will_copy) {
          TF_ASSIGN_OR_RETURN(
              const LogicalBuffer* buffer,
              points_to_analysis.GetBufferDefinedAt(parameter, index));
          for (const BufferAlias& alias :
               points_to_analysis.GetBufferAliases(*buffer)) {
            for (HloInstruction* user : alias.instruction()->users()) {
              if (user != instruction_) {
                control_predecessors_.mutable_element(index)->push_back(user);
              }
            }
          }
        }
        return Status::OK();
      });
}

// Recursively inserts copies of 'instruction' tuple element buffers at
// indices in 'indices_to_copy_', expanding tuples as needed.
HloInstruction* InstructionCopier::CopyTuple(HloInstruction* instruction,
                                             ShapeIndex* index) {
  const int64 num_tuple_elements =
      ShapeUtil::TupleElementCount(instruction->shape());
  std::vector<HloInstruction*> elem_copies(num_tuple_elements);
  for (int64 i = 0; i < num_tuple_elements; ++i) {
    HloInstruction* elem;
    if (instruction->opcode() == HloOpcode::kTuple) {
      // If the instruction is already a Tuple instruction, we know that the
      // element buffers are aliased, so we can just grab the operand directly.
      elem = instruction->mutable_operand(i);
    } else {
      // Otherwise we need to add a GTE to unpack the element out of the tuple.
      elem = instruction->parent()->AddInstruction(
          HloInstruction::CreateGetTupleElement(
              ShapeUtil::GetSubshape(instruction->shape(), {i}), instruction,
              i));
    }
    index->push_back(i);
    if (ShapeUtil::IsTuple(elem->shape())) {
      elem_copies[i] = CopyTuple(elem, index);
    } else if (!indices_to_copy_.element(*index)) {
      elem_copies[i] = elem;
    } else if (HloInstruction* copy_override = GetCopyOverride(*index)) {
      elem_copies[i] = copy_override;
    } else {
      HloInstruction* elem_copy = elem->parent()->AddInstruction(
          HloInstruction::CreateUnary(elem->shape(), HloOpcode::kCopy, elem));
      for (HloInstruction* control_predecessor :
           control_predecessors_.element(*index)) {
        VLOG(2) << "Adding control dependency from "
                << control_predecessor->ToString() << " to "
                << elem_copy->ToString();
        TF_CHECK_OK(control_predecessor->AddControlDependencyTo(elem_copy));
      }
      elem_copies[i] = elem_copy;
    }
    index->pop_back();
  }
  return instruction->parent()->AddInstruction(
      HloInstruction::CreateTuple(elem_copies));
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

// The 'read_only_indices' are initalized based on points-to analysis on the
// while body corresponding to 'while_hlo'. If the init buffer corresponding to
// a read-only index aliases with an entry parameter (or constant), it cannot be
// considered read-only, and must be copied. This is necessary because some
// backends don't support entry-parameter (or constant) aliasing with regular
// instructions. This function performs this fix-up of 'read_only_indices'.
//
// Returns a ShapeTree of copy_overrides, which implements an optimization to
// allow multiple while loops that share the same read-only entry parameters to
// share a single copy.
StatusOr<ShapeTree<HloInstruction*>>
RevertReadOnlyIndicesForEntryParamsAndConstants(
    const HloInstruction* while_hlo,
    const TuplePointsToAnalysis& points_to_analysis,
    ShapeTree<bool>* read_only_indices,
    FlatMap<const HloInstruction*, HloInstruction*>* shared_copies) {
  const HloInstruction* init_hlo = while_hlo->operand(0);
  const PointsToSet& points_to = points_to_analysis.GetPointsToSet(init_hlo);
  ShapeTree<HloInstruction*> copy_overrides(init_hlo->shape());
  TF_RETURN_IF_ERROR(points_to.ForEachElement(
      [init_hlo, read_only_indices, shared_copies, &copy_overrides](
          const ShapeIndex& index, bool /*is_leaf*/,
          const std::vector<const LogicalBuffer*>& buffers) {
        // Look for read-only entry parameters.
        if (!read_only_indices->element(index)) {
          return Status::OK();
        }
        for (const LogicalBuffer* buffer : buffers) {
          HloInstruction* pointee = buffer->instruction();
          const HloComputation* computation = pointee->parent();
          const bool is_entry_parameter =
              pointee->opcode() == HloOpcode::kParameter &&
              computation == computation->parent()->entry_computation();
          const bool is_constant = pointee->opcode() == HloOpcode::kConstant;
          if (!is_entry_parameter && !is_constant) {
            continue;
          }
          // We have found an entry parameter or constant that is read-only in
          // the while body. These buffers are managed by the caller, and cannot
          // be aliased with non-parameter buffers. Revert this read-only index,
          // to allow it to be copied.
          *read_only_indices->mutable_element(index) = false;

          // Optimization to allow multiple while loops that share the same
          // read-only entry parameters (or constants) to share a single copy.
          // Only unambiguous array-shaped buffers are allowed, to reduce code
          // complexity. The shape of the entry parameter must be identical to
          // the shape of the init_hlo at this index, to ensure there were no
          // intervening bitcast or GTE instructions, which are also hard to
          // handle.
          const Shape& pointee_shape = pointee->shape();
          const Shape& init_shape =
              ShapeUtil::GetSubshape(init_hlo->shape(), index);
          if (buffers.size() == 1 && ShapeUtil::IsArray(pointee_shape) &&
              ShapeUtil::Equal(pointee_shape, init_shape)) {
            HloInstruction** copy = &(*shared_copies)[pointee];
            if (*copy == nullptr) {
              *copy =
                  pointee->parent()->AddInstruction(HloInstruction::CreateUnary(
                      pointee_shape, HloOpcode::kCopy, pointee));
            }
            // Add the copy as an override.
            *copy_overrides.mutable_element(index) = *copy;
          }

          // We've already reverted the read-only index and handled the
          // single-copy optimization above, so there's nothing more to do.
          break;
        }
        return Status::OK();
      }));
  return copy_overrides;
}

}  // anonymous namespace

// NOTE: This is only called by gpu::CopyInsertion. It's not called here in the
// base class, since the regular CopyInsertion logic above selectively copies
// tuple elements, while this method assumes all buffers need to be deep copied.
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
  const auto& points_to_analysis = liveness->points_to_analysis();
  XLA_VLOG_LINES(2, points_to_analysis.ToString());
  XLA_VLOG_LINES(2, module->ToString());

  // Gather all while body computations and while instructions.
  FlatSet<const HloComputation*> while_body_computations;
  std::vector<HloInstruction*> while_instructions;
  for (auto& computation : module->computations()) {
    for (auto& instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        while_body_computations.insert(instruction->while_body());
        while_instructions.push_back(instruction.get());
      }
    }
  }

  // Collect instruction buffer indices to copy in 'instructions_to_copy'.
  std::vector<InstructionCopier> instructions_to_copy;

  // Add copies of computation root instructions, if needed.
  FlatMap<const HloComputation*, ShapeTree<bool>> while_body_read_only_indices;
  for (auto& computation : module->computations()) {
    VLOG(2) << "computation " << computation->name();
    InstructionCopier root_copier(computation->root_instruction(),
                                  /*copy_users=*/{});
    if (while_body_computations.count(computation.get()) > 0) {
      // Record root indices to copy for while body sub-computations. We do not
      // need to call RecordIndicesWhichPointToParamOrConstant for the while
      // body root instruction here, because any necessary copies needed to
      // avoid constants or parameters in the output are handled by while.init
      // operand copy insertion below (which will share an allocation).
      HloInstruction* while_body_param = computation->parameter_instruction(0);
      ShapeTree<bool> read_only_indices(while_body_param->shape());
      TF_RETURN_IF_ERROR(root_copier.RecordIndicesToCopyForColocatingBuffers(
          *liveness, while_body_param, &read_only_indices));
      while_body_read_only_indices[computation.get()] = read_only_indices;

      // Mark control predecessors, based on the body param, for any copies
      // we'll be inserting. This ensures the copy doesn't run too early.
      TF_RETURN_IF_ERROR(root_copier.RecordControlPredecessors(
          points_to_analysis, while_body_param));
    } else {
      // Record root indices to copy for general computations.
      TF_RETURN_IF_ERROR(root_copier.RecordIndicesWhichPointToParamOrConstant(
          points_to_analysis));
    }
    instructions_to_copy.push_back(root_copier);
  }

  // Add copies of while 'init' operand instructions, if needed. 'shared_copies'
  // is used to ensure that multiple while loops can share a single copy of the
  // same entry parameter or constant, if all loops use it read-only.
  //
  // TODO(b/33301720) Remove redundant while instruction copies.
  FlatMap<const HloInstruction*, HloInstruction*> shared_copies;
  for (HloInstruction* while_hlo : while_instructions) {
    // Fix read_only_indices to account for entry parameters and constants. Also
    // initialize copy_overrides, which ensures a single copy for each read-only
    // entry parameter or constant that is used in multiple while loops.
    ShapeTree<bool>* read_only_indices =
        &while_body_read_only_indices[while_hlo->while_body()];
    TF_ASSIGN_OR_RETURN(
        const ShapeTree<HloInstruction*> copy_overrides,
        RevertReadOnlyIndicesForEntryParamsAndConstants(
            while_hlo, points_to_analysis, read_only_indices, &shared_copies));
    // Create InstructionCopier for init operand of while instruction.
    HloInstruction* init_hlo = while_hlo->mutable_operand(0);
    InstructionCopier init_copier(init_hlo, {while_hlo});
    init_copier.SetReadOnlyIndices(*read_only_indices);
    init_copier.SetCopyOverrides(copy_overrides);
    // Record 'init' buffer indices which point-to a Constant or Parameter.
    TF_RETURN_IF_ERROR(init_copier.RecordIndicesWhichPointToParamOrConstant(
        points_to_analysis));
    // Record indices necessary to colocate while and init operand buffers.
    TF_RETURN_IF_ERROR(init_copier.RecordIndicesToCopyForColocatingBuffers(
        *liveness, while_hlo, /*read_only_indices_out=*/nullptr));
    instructions_to_copy.push_back(init_copier);
  }

  for (InstructionCopier& to_copy : instructions_to_copy) {
    if (to_copy.HasAllIndicesFalse()) {
      continue;
    }
    changed = true;

    // Copy instruction at recorded buffer indices.
    HloComputation* computation = to_copy.instruction()->parent();
    HloInstruction* copy = to_copy.Copy();
    if (to_copy.instruction() == computation->root_instruction()) {
      computation->set_root_instruction(copy);
    }
  }

  VLOG(3) << "After copy insertion for module " << module->name();
  XLA_VLOG_LINES(3, module->ToString());

  return changed;
}

}  // namespace xla
