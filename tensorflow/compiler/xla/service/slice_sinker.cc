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

#include "tensorflow/compiler/xla/service/slice_sinker.h"
#include <algorithm>
#include <utility>
#include <vector>
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

namespace {

// Returns whether two slices are taken from the same indices of the bigger
// tensor with the same dimensions.
bool SameSliceConfiguration(const HloInstruction* slice_1,
                            const HloInstruction* slice_2) {
  return slice_1->slice_starts() == slice_2->slice_starts()
      && slice_1->slice_limits() == slice_2->slice_limits()
      && slice_1->slice_strides() == slice_2->slice_strides();
}

// Collects slice sources about inst. The inst's operands should all be slices
// and the slice sources are the operands of slices. Returns slice sources
// vector if all slice sources are found, or return an empty vector.
absl::optional<std::vector<HloInstruction*>>
FindSourceOperandsOfSlicesForElementwiseOperation(const HloInstruction* inst) {
  CHECK(inst->IsElementwise());
  std::vector<HloInstruction*> operands;
  // Check operand:
  // The inst's operand should be a slice.
  // The operands-vector keeps all slice sources of inst.
  for (HloInstruction* operand_slice : inst->operands()) {
    if (operand_slice->opcode() != HloOpcode::kSlice) {
      // If the operand is not operand-slice, returns empty vector.
      return absl::nullopt;
    }
    HloInstruction* operand = operand_slice->mutable_operand(0);
    operands.push_back(operand);
  }

  // No slice sources found. returns empty vector.
  if (operands.empty()) {
    return absl::nullopt;
  }

  // Check operands:
  // True operands should have the same shape because inst is elementwise.
  const Shape shape = operands[0]->shape();
  for (const HloInstruction* operand : operands) {
    // Only support element-wise now
    if (!ShapeUtil::CompatibleIgnoringElementType(operand->shape(), shape)) {
      return absl::nullopt;
    }
  }
  // Inst's operand should be slices taken from the same indices of bigger
  // tensors with the same dimensions.
  const HloInstruction* operand0 = inst->operand(0);
  for (const HloInstruction* operand : inst->operands()) {
    if (!SameSliceConfiguration(operand0, operand)) {
      return absl::nullopt;
    }
  }
  // Returns all slice sources about inst.
  return operands;
}

// Returns whether peer candidate is a peer instruction.
bool IsPeerInstruction(const HloInstruction* peer_candidate,
    absl::Span<HloInstruction* const> slice_sources) {
  // Checks slices:
  const HloInstruction* operand_slice0 = peer_candidate->operand(0);
  for (int64 j = 0; j < slice_sources.size(); ++j) {
    // The operands of each operation are slices taken from the same indices of
    // bigger tensors with the same dimensions. The corresponding operands of
    // all operations are slices taken from the same bigger tensors.
    const HloInstruction* operand_slice = peer_candidate->operand(j);
    if (operand_slice->opcode() != HloOpcode::kSlice ||
        operand_slice->operand(0) != slice_sources[j] ||
        !SameSliceConfiguration(operand_slice0, operand_slice)) {
      return false;
      break;
    }
  }
  return true;
}

// Computes the cost of implimentation of sinking slice, and returns whether
// it should be changed.
// This routine assumes that we will generate a new elementwise operation using
// the "slice_sources" as operands. For the following example, we only need the
// result of add([0:9] slice of p) while we generate add(p).
//
// p = f32[10] parameter(0)
// a = f32[8] slice(p), slice=[0:8]
// aa = add(a, a)
// b = f32[7] slice(p), slice=[2:9]
// bb = add(b, b)
// (there are 15 scalar add operations, but only 10 scalar add operations when
// add(p) )
bool ShouldReplace(const std::vector<HloInstruction*>& operands,
    const std::vector<HloInstruction*>& users) {
  int64 sum = 0;
  // Sums the total element number of the peer operations.
  for (HloInstruction* user : users) {
    sum += ShapeUtil::ElementsIn(user->shape());
  }
  // Operand and user have the same element number in shape because of
  // elementwise operation, so the elements in operands[0] is the same as in
  // new user if new user is generated.
  // Compares the total elements in all peer operations and the elements of the
  // new user with whole shape.
  return sum >= xla::ShapeUtil::ElementsIn(operands[0]->shape());
}

// Collects the peer operations of inst included inst.
// The operands of each operation in group have the same opcode and they are
// slices taken from the same indices of bigger tensors with the same
// dimensions. The corresponding operands of all operations are slices taken
// from the same bigger tensors.
// Returns an empty vector if the accumulated size of the operations in group
// is less than the size of such a bigger tensor. This is a heuristic to
// ensure that the transformation never causes us to do more elementwise
// operations
absl::optional<std::vector<HloInstruction*>> FindPeerElementwiseOperations(
    const HloInstruction* inst,
    const std::vector<HloInstruction*>& slice_sources) {
  std::vector<HloInstruction*> peer_operations;
  HloInstruction* slice_source0 = slice_sources[0];

  // Traverses the slices which are taken from slice source 0.
  for (const HloInstruction* operand_slice0 : slice_source0->users()) {
    // Skips not-slices
    if (operand_slice0->opcode() != HloOpcode::kSlice) {
      continue;
    }

    // The user of slice is a candidate of peer operation.
    for (HloInstruction* user : operand_slice0->users()) {
      // The inst's peers should be the same operation and same operand count as
      // inst.
      if (user->opcode() != inst->opcode() ||
          user->operand_count() != inst->operand_count() ||
          !IsPeerInstruction(user, slice_sources)) {
        continue;
      }

      // Found the peer operation.
      peer_operations.push_back(user);
    }
  }

  // Calculates the costs. If cost is more than profit, returns nullopt.
  return ShouldReplace(slice_sources, peer_operations) ?
      absl::make_optional(peer_operations) : absl::nullopt;
}

// Generates a new elementwise operation using the slice_sources as operands,
// and replaces the uses of elementwise operation_on_slices with slices of the
// new elementwise operations.
void SinkSlices(const std::vector<HloInstruction*>& slice_sources,
    const std::vector<HloInstruction*>& operation_on_slices) {
  // Generates new user.
  const Shape shape = slice_sources[0]->shape();
  PrimitiveType element_type = operation_on_slices[0]->shape().element_type();
  Shape new_shape = ShapeUtil::ChangeElementType(shape, element_type);

  HloComputation* computation = operation_on_slices[0]->parent();
  auto new_user = computation->AddInstruction(
      operation_on_slices[0]->CloneWithNewOperands(new_shape, slice_sources));
  VLOG(10) << "Add NewUser: " << new_user->ToString();

  // Replaces the peer operations with new user's slices.
  for (HloInstruction* user : operation_on_slices) {
    const HloInstruction* operand_slice = user->operand(0);
    // Generates user slices of new user.
    auto user_slice = computation->AddInstruction(
        operand_slice->CloneWithNewOperands(user->shape(), {new_user}));
    VLOG(10) << "Add NewSlice: " << user_slice->ToString()
             << " Replace: " << user->ToString();
    // Replaces peer operations with user slices.
    user->ReplaceAllUsesWith(user_slice);
  }
}

}  // namespace

// The group of elementwise operations being transformed meet the following
// requirements:
// (condition-1) The operands of each operation are slices taken from the same
// indices of bigger tensors with the same dimensions.
// (condition-2) All operations have the same opcode.
// (condition-3) The corresponding operands of all operations are slices taken
// from the same bigger tensors.
// (condition-4) The accumulated size of the group of operations is not less
// than the size of such a bigger tensor. This is a heuristic to ensure that the
// transformation never causes us to do more elementwise operations.
//
// TODO(xinan): Supports more non-elementwise instructions. Some non-elementwise
// instruction's slice operand could also sink in some circumstances. e.g. dot,
// broadcast, reduce.
StatusOr<bool> SliceSinker::Run(HloModule* module) {
  XLA_VLOG_LINES(2, "SliceSinker::Run(), before:\n" + module->ToString());
  bool changed = false;

  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction :
        computation->MakeInstructionPostOrder()) {
      if (!instruction->IsElementwise() || instruction->operand_count() == 0) {
        continue;
      }
      VLOG(10) << "Merge inst: " << instruction->ToString();
      // If instruction is an elementwise operation on similar slices, return
      // the source operands of the slices. This check condition-1 described
      // above.
      absl::optional<std::vector<HloInstruction*>> source_operands_of_slices =
          FindSourceOperandsOfSlicesForElementwiseOperation(instruction);
      if (!source_operands_of_slices.has_value()) {
        continue;
      }
      // If we can find a group of elementwise operations on similar slices that
      // meet condition 2~4 and includes instruction, return such a group of
      // instructions.
      absl::optional<std::vector<HloInstruction*>> peer_elementwise_operations =
          FindPeerElementwiseOperations(instruction,
                                        source_operands_of_slices.value());
      if (!peer_elementwise_operations.has_value()) {
        continue;
      }
      SinkSlices(source_operands_of_slices.value(),
                 peer_elementwise_operations.value());
      changed = true;
    }
  }

  XLA_VLOG_LINES(2, "SliceSinker::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
