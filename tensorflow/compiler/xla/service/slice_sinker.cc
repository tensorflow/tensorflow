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
#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

namespace {

// Returns whether two slices are taken from the same indices, assuming the
// slices are taking from tensors with the same dimensions.
bool SameSliceConfiguration(const HloInstruction* slice_1,
                            const HloInstruction* slice_2) {
  CHECK_EQ(slice_1->opcode(), HloOpcode::kSlice);
  CHECK_EQ(slice_2->opcode(), HloOpcode::kSlice);
  CHECK(absl::c_equal(slice_1->operand(0)->shape().dimensions(),
                      slice_2->operand(0)->shape().dimensions()));
  return absl::c_equal(slice_1->slice_starts(), slice_2->slice_starts())
      && absl::c_equal(slice_1->slice_limits(), slice_2->slice_limits())
      && absl::c_equal(slice_1->slice_strides(), slice_2->slice_strides());
}

// Given an elementwise operation, if all the operands of the operation are
// slices from the same indices of tensors with compatible shapes, returns a
// vector of the slice sources. Otherwise returns nullopt.
absl::optional<std::vector<HloInstruction*>>
FindSourceOperandsOfSlicesForElementwiseOperation(const HloInstruction* inst) {
  CHECK(inst->IsElementwise());

  // Check that all operands are slices.
  if (absl::c_any_of(inst->operands(),
      [](const HloInstruction* operand) {
        return operand->opcode() != HloOpcode::kSlice;
      })) {
    return absl::nullopt;
  }

  // Check that all slices are from the same indices of slice sources with
  // compatible shapes.
  const HloInstruction* slice0 = inst->operand(0);
  if (absl::c_any_of(absl::MakeSpan(inst->operands()).subspan(1),
      [slice0](const HloInstruction* slice) {
        return !ShapeUtil::CompatibleIgnoringElementType(
                   slice0->operand(0)->shape(), slice->operand(0)->shape())
            || !SameSliceConfiguration(slice0, slice);
      })) {
    return absl::nullopt;
  }

  // Construct and return a vector of the slice sources.
  std::vector<HloInstruction*> slice_sources;
  absl::c_transform(inst->operands(), std::back_inserter(slice_sources),
      [](HloInstruction* slice) { return slice->mutable_operand(0); });
  return slice_sources;
}

// Given an instruction with a slice of slice_sources[i] as its ith operand,
// returns true if peer_candidate hasn't been transformed by and is a peer of
// the instruction that meets the following requirements:
// 1) Peer_candidate has the same opcode as the given instruction.
// 2) The ith operand of peer_candidate is a slice of slice_sources[i].
// 3) All operands of peer_candidate are slices taken from the same indices.
bool IsPeerOperation(const HloInstruction* inst,
    const HloInstruction* peer_candidate,
    absl::Span<HloInstruction* const> slice_sources) {
  // Instructions that have already been transformed have user_count 0. Avoid
  // transforming such instructions again.
  if (peer_candidate->user_count() == 0) {
    return false;
  }

  if (peer_candidate->opcode() != inst->opcode()) {
    return false;
  }

  const HloInstruction* operand_slice0 = peer_candidate->operand(0);
  for (int64 i = 0; i < slice_sources.size(); ++i) {
    const HloInstruction* operand_slice = peer_candidate->operand(i);
    if (operand_slice->opcode() != HloOpcode::kSlice ||
        operand_slice->operand(0) != slice_sources[i] ||
        !SameSliceConfiguration(operand_slice0, operand_slice)) {
      return false;
      break;
    }
  }
  return true;
}

// Compares the cost of implementing one elementwise operations on the
// slice_sources with the cost of implementing all the individual elementwise
// operations in operations_on_slices and returns true if the former is less
// expensive. Currently we don't support the following transformation because we
// don't have such a use case yet.
// Transform
//   p = f32[20] parameter(0)
//   a = f32[8] slice(p), slice=[0:8]
//   aa = add(a, a)
//   b = f32[7] slice(p), slice=[2:9]
//   bb = add(b, b)
//
// to
//   p = f32[20] parameter(0)
//   x = f32[9] slice(p), slice=[0:8]
//   xx = add(x,x)
//   aa = f32[8] slice(xx), slice=[0:8]
//   bb = f32[7] slice(xx), slice=[2:9]
bool ShouldReplace(const std::vector<HloInstruction*>& slice_sources,
    const std::vector<HloInstruction*>& operations_on_slices) {
  int64 sum = 0;
  for (HloInstruction* user : operations_on_slices) {
    sum += ShapeUtil::ElementsIn(user->shape());
  }
  return sum >= xla::ShapeUtil::ElementsIn(slice_sources[0]->shape());
}

// Collects the peer operations of inst including inst itself. See
// IsPeerOperation for the definition of peer operation.
absl::optional<std::vector<HloInstruction*>> FindPeerElementwiseOperations(
    const HloInstruction* inst,
    const std::vector<HloInstruction*>& slice_sources) {
  std::vector<HloInstruction*> peer_operations;
  HloInstruction* slice_source0 = slice_sources[0];

  // Traverse the slices taken from the first slice sources.
  for (const HloInstruction* operand_slice0 : slice_source0->users()) {
    if (operand_slice0->opcode() != HloOpcode::kSlice) {
      continue;
    }

    // A user of the slice is a candidate of peer operations on slices.
    for (HloInstruction* user : operand_slice0->users()) {
      if (IsPeerOperation(inst, user, slice_sources)) {
        peer_operations.push_back(user);
      }
    }
  }

  return ShouldReplace(slice_sources, peer_operations) ?
      absl::make_optional(peer_operations) : absl::nullopt;
}

// Generates a new elementwise operation using the slice_sources as operands,
// and replaces the uses of elementwise operation_on_slices with slices of the
// new elementwise operations.
Status SinkSlices(const std::vector<HloInstruction*>& slice_sources,
    const std::vector<HloInstruction*>& operation_on_slices) {
  // Generates operation on slice source.
  const Shape shape = slice_sources[0]->shape();
  PrimitiveType element_type = operation_on_slices[0]->shape().element_type();
  Shape new_shape = ShapeUtil::ChangeElementType(shape, element_type);

  HloComputation* computation = operation_on_slices[0]->parent();
  auto operation_on_slice_sources = computation->AddInstruction(
      operation_on_slices[0]->CloneWithNewOperands(new_shape, slice_sources));
  VLOG(10) << "Add operation_on_slice_sources: "
           << operation_on_slice_sources->ToString();

  // Replace each operation on slices with a slice of the operation on the slice
  // sources.
  for (HloInstruction* user : operation_on_slices) {
    const HloInstruction* operand_slice = user->operand(0);
    auto user_slice = computation->AddInstruction(
        operand_slice->CloneWithNewOperands(user->shape(),
                                            {operation_on_slice_sources}));
    VLOG(10) << "Add NewSlice: " << user_slice->ToString()
             << " Replace: " << user->ToString();
    TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(user_slice));
  }
  return Status::OK();
}

}  // namespace

// There are two purposes of this pass.
//
// Eliminates redundant work that occurs when two slices overlap. For example:
//   p = f32[10] parameter(0)
//   a = f32[9] slice(p), slice=[0:9]
//   aa = add(a, a)
//   b = f32[8] slice(p), slice=[2:10]
//   bb = add(b, b)
//   ...
// Here we do 17 scalar add operations, while we actually only need to do 10 if
// we can transform the code to the following:
//   p = f32[10] parameter(0)
//   add = add(p, p)
//   aa = f32[9] slice(add), slice=[0:9]
//   bb = f32[8] slice(add), slice=[2:10]
//   ...
// Merges elementwise when two slices are "adjacent".
//   p = f32[10] parameter(0)
//   a = f32[6] slice(p), slice=[0:6]
//   aa = add(a, a)
//   b = f32[4] slice(p), slice=[6:10]
//   bb = add(b, b)
//   ...
// Here we’re not doing any redundant work, but transforming this graph to  the
// following graph allows us to run fewer kernels:
//   p = f32[10] parameter(0)
//   add = add(p, p)
//   aa = f32[6] slice(add), slice=[0:6]
//   bb = f32[4] slice(add), slice=[6:10]
//
// As can be seen from the examples, the group of elementwise operations being
// transformed meet the following requirements:
// (condition-1) The operands of each operation are slices taken from the same
// indices of bigger tensors with the same dimensions.
// (condition-2) All operations have the same opcode.
// (condition-3) The corresponding operands of all operations are slices taken
// from the same bigger tensors.
// (condition-4) The accumulated size of the group of operations is not less
// than the size of such a bigger tensor. This is a heuristic to ensure that the
// transformation never causes us to do more elementwise operations.
//
// TODO(xinan): Supports more non-elementwise instructions, such as dot,
// broadcast and reduce.
StatusOr<bool> SliceSinker::Run(HloModule* module) {
  bool changed = false;

  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction :
        computation->MakeInstructionPostOrder()) {
      if (!instruction->IsElementwise() || instruction->operand_count() == 0
          || instruction->user_count() == 0) {
        continue;
      }
      VLOG(10) << "Merges inst: " << instruction->ToString();
      // If the current operation is an elementwise operation on similar slices,
      // return the source operands of the slices. This check condition-1
      // described above.
      absl::optional<std::vector<HloInstruction*>> source_operands_of_slices =
          FindSourceOperandsOfSlicesForElementwiseOperation(instruction);
      if (!source_operands_of_slices.has_value()) {
        continue;
      }
      // If we can find a group of elementwise operations on similar slices that
      // meet condition 2~4, return such a group of operations including the
      // current operation.
      absl::optional<std::vector<HloInstruction*>> peer_elementwise_operations =
          FindPeerElementwiseOperations(instruction,
                                        source_operands_of_slices.value());
      if (!peer_elementwise_operations.has_value()) {
        continue;
      }
      TF_RETURN_IF_ERROR(SinkSlices(source_operands_of_slices.value(),
                                    peer_elementwise_operations.value()));
      changed = true;
    }
  }

  return changed;
}

}  // namespace xla
