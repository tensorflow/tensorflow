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

#include "absl/algorithm/container.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

namespace {

// Returns whether two slices are taken from the same indices, assuming the
// slices are taking from tensors with the same dimensions.
bool SameSliceConfiguration(const HloInstruction* slice_1,
                            const HloInstruction* slice_2) {
  CHECK_EQ(slice_1->opcode(), HloOpcode::kSlice);
  CHECK_EQ(slice_2->opcode(), HloOpcode::kSlice);
  CHECK(slice_1->operand(0)->shape().dimensions() ==
        slice_2->operand(0)->shape().dimensions());
  return slice_1->slice_starts() == slice_2->slice_starts() &&
         slice_1->slice_limits() == slice_2->slice_limits() &&
         slice_1->slice_strides() == slice_2->slice_strides();
}

// Returns true if all the operands of the given elementwise operation are
// slices from the same indices of tensors with compatible shapes.
bool IsElementwiseOperationOnSimilarSlices(const HloInstruction* inst) {
  CHECK(inst->IsElementwise());

  // Check that all operands are slices.
  if (absl::c_any_of(inst->operands(), [](const HloInstruction* operand) {
        return operand->opcode() != HloOpcode::kSlice;
      })) {
    return false;
  }

  // Check that all slices are from the same indices of slice sources with
  // compatible shapes.
  const HloInstruction* slice0 = inst->operand(0);
  return absl::c_all_of(absl::MakeSpan(inst->operands()).subspan(1),
                        [slice0](const HloInstruction* slice) {
                          return ShapeUtil::CompatibleIgnoringElementType(
                                     slice0->operand(0)->shape(),
                                     slice->operand(0)->shape()) &&
                                 SameSliceConfiguration(slice0, slice);
                        });
}

// Given an elementwise operation with all slice operands, operation_on_slices,
// checks whether another operation, candidate, is an operation that hasn't been
// transformed and is similar to operation_on_slices as defined by the following
// criteria:
// (1) candidate has the same opcode and result element type as
//     operation_on_slices. The check for same result element type is necessary
//     because kConvert can produce different result element types for the same
//     input element type.
// (2) The ith operand of candidate is a slice from the same slice source of
//     the ith operand in operation_on_slices.
// (3) All operands of candidate are slices taken from the same indices as the
//     operands of operation_on_slices are.
bool IsSimilarOperationOnSlices(const HloInstruction* operation_on_slices,
                                const HloInstruction* candidate) {
  // Instructions that have already been transformed have user_count 0. Avoid
  // transforming such instructions again.
  if (candidate->user_count() == 0) {
    return false;
  }

  if (candidate->opcode() != operation_on_slices->opcode() ||
      operation_on_slices->shape().element_type() !=
          candidate->shape().element_type()) {
    return false;
  }

  const HloInstruction* operand_slice0 = candidate->operand(0);
  for (int64 i = 0; i < candidate->operand_count(); ++i) {
    const HloInstruction* operand_slice = candidate->operand(i);
    if (operand_slice->opcode() != HloOpcode::kSlice ||
        operand_slice->operand(0) !=
            operation_on_slices->operand(i)->operand(0) ||
        !SameSliceConfiguration(operand_slice0, operand_slice)) {
      return false;
    }
  }
  return true;
}

// Given a group of elementwise operations on slices that can be transformed to
// one elementwise operation on the slice sources, compares the cost of
// implementing the new elementwise operation on the slice sources with the cost
// of implementing all the individual elementwise operations independently.
// Returns true if the former is less expensive.
//
// Currently we don't support the following transformation that produces a new
// elementwise operation on bigger slices of the slice sources. This is because
// we don't have such a use case yet:
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
bool ShouldTransform(const std::vector<HloInstruction*>& operations_on_slices) {
  int64 sum = 0;
  for (HloInstruction* user : operations_on_slices) {
    sum += ShapeUtil::ElementsIn(user->shape());
  }
  return sum >= xla::ShapeUtil::ElementsIn(
                    operations_on_slices[0]->operand(0)->operand(0)->shape());
}

// Returns a group of elementwise operations on slices that are similar to the
// given operations_on_slices. See IsSimilarOperationOnSlices for what are
// considered similar operation on slices.
absl::optional<std::vector<HloInstruction*>> FindElementwiseOperationGroup(
    const HloInstruction* operation_on_slices) {
  std::vector<HloInstruction*> operations;
  const HloInstruction* slice_source0 =
      operation_on_slices->operand(0)->operand(0);

  // Traverse the slices taken from the first slice sources.
  for (const HloInstruction* operand_slice0 : slice_source0->users()) {
    if (operand_slice0->opcode() != HloOpcode::kSlice) {
      continue;
    }

    for (HloInstruction* user : operand_slice0->users()) {
      if (IsSimilarOperationOnSlices(operation_on_slices, user)) {
        operations.push_back(user);
      }
    }
  }

  return ShouldTransform(operations) ? absl::make_optional(operations)
                                     : absl::nullopt;
}

// Generates a new elementwise operation using the slice_sources as operands,
// and replaces the uses of elementwise operation_on_slices with slices of the
// new elementwise operations.
Status SinkSlices(const std::vector<HloInstruction*>& slice_sources,
                  const std::vector<HloInstruction*>& operation_on_slices) {
  const Shape shape = slice_sources[0]->shape();
  PrimitiveType element_type = operation_on_slices[0]->shape().element_type();
  Shape new_shape = ShapeUtil::ChangeElementType(shape, element_type);

  HloComputation* computation = operation_on_slices[0]->parent();
  auto operation_on_slice_sources = computation->AddInstruction(
      operation_on_slices[0]->CloneWithNewOperands(new_shape, slice_sources));
  VLOG(10) << "Adding operation_on_slice_sources: "
           << operation_on_slice_sources->ToString();

  // Replace each operation on slices with a slice of the operation on the slice
  // sources.
  for (HloInstruction* user : operation_on_slices) {
    const HloInstruction* operand_slice = user->operand(0);
    auto user_slice =
        computation->AddInstruction(operand_slice->CloneWithNewOperands(
            user->shape(), {operation_on_slice_sources}));
    VLOG(10) << "Adding new slice: " << user_slice->ToString()
             << " to replace: " << user->ToString();
    TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(user_slice));
  }
  return Status::OK();
}

}  // namespace

// There are two purposes of this pass.
//
// 1. Eliminates redundant work that occurs when two slices overlap. For
// example:
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
//
// 2. Merges elementwise operations when two slices are "adjacent".
//   p = f32[10] parameter(0)
//   a = f32[6] slice(p), slice=[0:6]
//   aa = add(a, a)
//   b = f32[4] slice(p), slice=[6:10]
//   bb = add(b, b)
//   ...
// Here we're not doing any redundant work, but transforming this graph to the
// following graph allows us to run fewer kernels:
//   p = f32[10] parameter(0)
//   add = add(p, p)
//   aa = f32[6] slice(add), slice=[0:6]
//   bb = f32[4] slice(add), slice=[6:10]
//
// As can be seen from the examples, the group of elementwise operations being
// transformed must meet the following requirements:
// (1) The operands of each operation are slices taken from the same indices of
//     bigger tensors with the same dimensions.
// (2) All operations have the same opcode.
// (3) The corresponding operands of all operations are slices taken
//     from the same bigger tensors.
// (4) The accumulated size of the group of operations is not less than the size
//     of such a bigger tensor. This is a heuristic to ensure that the
// transformation never causes us to do more elementwise operations.
//
// This pass currently doesn't transform non-elementwise instructions. We may
// extend this pass to transform non-elementwise instructions, such as dot,
// broadcast and reduce in the future.
StatusOr<bool> SliceSinker::Run(HloModule* module) {
  bool changed = false;

  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      // When processing instruction A in this loop, we may transform A along
      // with instruction B, which is after A in the post order. An instruction
      // that has been transformed has a user_count 0. We use this fact to
      // avoid transforming an instruction that has been transformed.
      if (!instruction->IsElementwise() || instruction->operand_count() == 0 ||
          instruction->user_count() == 0) {
        continue;
      }
      VLOG(10) << "Processing instruction : " << instruction->ToString();

      // This checks condition (1).
      if (!IsElementwiseOperationOnSimilarSlices(instruction)) {
        continue;
      }

      // Try to find a group of elementwise operations that are similar to
      // the current instruction. This checks conditions (2)-(4).
      absl::optional<std::vector<HloInstruction*>> similar_operations =
          FindElementwiseOperationGroup(instruction);
      if (!similar_operations.has_value()) {
        continue;
      }

      std::vector<HloInstruction*> slice_sources;
      absl::c_transform(
          instruction->operands(), std::back_inserter(slice_sources),
          [](HloInstruction* slice) { return slice->mutable_operand(0); });

      TF_RETURN_IF_ERROR(SinkSlices(slice_sources, similar_operations.value()));
      changed = true;
    }
  }

  return changed;
}

}  // namespace xla
