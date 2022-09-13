/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/scatter_slice_simplifier.h"

#include <iterator>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace {

// Returns whether the instruction could be an operand for a slice instruction.
bool IsValidIntermediaryUser(const HloInstruction* instruction) {
  // Allow elementwise instructions, as they don't depend on the truncated
  // elements. In case of multi-output scatters, the resulting shape is a tuple.
  return instruction->IsElementwise() ||
         instruction->opcode() == HloOpcode::kGetTupleElement;
}

// Matches the "Scatter -> Elementwise (zero or more) -> Slice" pattern.
// Calculates the resulting scatter dimensions from the slice users.
class ScatterSliceMatcher {
 public:
  explicit ScatterSliceMatcher(const HloScatterInstruction* scatter)
      : scatter_(scatter),
        operand_dimensions_(
            scatter->scatter_operands()[0]->shape().dimensions()),
        result_dimensions_(operand_dimensions_.begin(),
                           operand_dimensions_.end()) {}

  // Determine the scatter shape from the user slice instructions.
  // If any of the users are not truncation slices, return `nullopt`.
  std::optional<Shape> InferShape() {
    VLOG(10) << "Evaluating scatter " << scatter_->name();
    if (!AreAllUsersValid(scatter_)) {
      return std::nullopt;
    }
    std::vector<Shape> result_shapes;
    absl::c_transform(scatter_->scatter_operands(),
                      std::back_inserter(result_shapes),
                      [&](const HloInstruction* op) {
                        return ShapeUtil::MakeShape(op->shape().element_type(),
                                                    result_dimensions_);
                      });
    return ShapeUtil::MakeMaybeTupleShape(result_shapes);
  }

 private:
  // Update the resulting scatter dimensions from the slice configuration and
  // the original scatter dimensions. Return `false` if the update is not
  // possible.
  bool UpdateDimensions(const HloSliceInstruction* slice) {
    int64_t rank = slice->shape().rank();
    for (int64_t i = 0; i < rank; ++i) {
      if (slice->slice_starts(i) != 0 || slice->slice_strides(i) != 1) {
        return false;  // The slice is not a truncation.
      }
      if (slice->slice_limits(i) != result_dimensions_[i]) {
        if (result_dimensions_[i] != operand_dimensions_[i]) {
          return false;  // Another slice has incompatible dimensions.
        }
        result_dimensions_[i] = slice->slice_limits(i);
        VLOG(10) << "Dimension " << i << " truncated to size "
                 << result_dimensions_[i];
      }
    }
    return true;
  }

  // Verify that the instruction is a valid scatter user, i.e. is either a slice
  // operation or is an elementwise operation that has slice users (recursive).
  bool IsUserValid(const HloInstruction* op) {
    VLOG(10) << "Visiting user " << op->name();

    // If the user is a slice operation, verify the configuration and update
    // the resulting dimensions.
    if (auto* slice = DynCast<HloSliceInstruction>(op)) {
      return UpdateDimensions(slice);
    }
    // If the user is an elementwise operation, verify the users recursively
    // (unless already visited).
    bool is_valid = visited_set_.contains(op) ||
                    (IsValidIntermediaryUser(op) && AreAllUsersValid(op));
    if (is_valid) {
      visited_set_.emplace(op);
    }
    return is_valid;
  }

  // Verify that all users are valid (see the definition of IsValidUser).
  // If we reach the root instruction, fail the matching (slice is not found).
  bool AreAllUsersValid(const HloInstruction* op) {
    if (op->user_count() == 0) {
      return !op->IsRoot();
    }
    return absl::c_all_of(op->users(), [this](const HloInstruction* user) {
      return IsUserValid(user);
    });
  }

  const HloScatterInstruction* scatter_;
  absl::flat_hash_set<const HloInstruction*> visited_set_;
  absl::Span<const int64_t> operand_dimensions_;
  DimensionVector result_dimensions_;
};

// Create a replacement operand for the scatter instruction.
HloInstruction* CreateSliceFrom(HloInstruction* operand, const Shape& shape) {
  std::vector<int64_t> start_indices(shape.rank(), 0);
  std::vector<int64_t> limit_indices(shape.rank());
  std::vector<int64_t> strides(shape.rank(), 1);
  for (int64_t i = 0; i < shape.rank(); ++i) {
    limit_indices[i] = shape.dimensions(i);
  }
  return operand->AddInstruction(HloInstruction::CreateSlice(
      shape, operand, start_indices, limit_indices, strides));
}

// Create a replacement for the scatter instruction.
HloInstruction* CreateScatterFrom(HloScatterInstruction* scatter,
                                  const Shape& shape) {
  std::vector<HloInstruction*> operands(scatter->scatter_operand_count());
  for (int64_t i = 0; i < operands.size(); ++i) {
    operands[i] =
        CreateSliceFrom(scatter->scatter_operands()[i],
                        shape.IsTuple() ? shape.tuple_shapes(i) : shape);
  }
  return scatter->AddInstruction(HloInstruction::CreateScatter(
      shape, absl::MakeSpan(operands), scatter->scatter_indices(),
      scatter->scatter_updates(), scatter->called_computations()[0],
      scatter->scatter_dimension_numbers(), scatter->indices_are_sorted(),
      scatter->unique_indices()));
}

class ScatterSliceSimplifierVisitor : public DfsHloRewriteVisitor {
 public:
  Status HandleScatter(HloInstruction* instruction) override {
    auto* scatter = Cast<HloScatterInstruction>(instruction);

    // Infer scatter shape from the slice users.
    std::optional<Shape> result_shape =
        ScatterSliceMatcher(scatter).InferShape();
    if (!result_shape.has_value()) {
      return OkStatus();
    }
    VLOG(2) << "Matched scatter " << scatter->name() << " with shape "
            << scatter->shape().ToString() << ", inferred result shape "
            << result_shape->ToString() << " (from the slice users)";

    // Replace slice user instructions.
    HloInstruction* new_scatter = CreateScatterFrom(scatter, *result_shape);
    return ReplaceAllUsersRecursive(scatter, new_scatter);
  }

 private:
  // Create a replacement for every user. If the user is a slice operation,
  // replace it in the computation graph, the old branch will be removed.
  Status ReplaceAllUsersRecursive(HloInstruction* old_instruction,
                                  HloInstruction* new_instruction) {
    // Maintain the replacement map, needed for non-unary elementwise users.
    replacements_[old_instruction] = new_instruction;

    // It's importand to make a copy of the users list, as it may be modified
    // during the iteration.
    std::vector<HloInstruction*> users = old_instruction->users();
    for (HloInstruction* user : users) {
      if (user->parent() == nullptr) {
        VLOG(3) << "Skipping user " << user->name() << " (already replaced)";
        continue;
      }
      TF_RETURN_IF_ERROR(ReplaceUserRecursive(user, new_instruction));
    }
    return OkStatus();
  }

  // Replace the slice user with a new scatter (or a new chain of operations
  // starting with a scatter). For elementwise operations, create a new user
  // with updated operands (build the chain).
  Status ReplaceUserRecursive(HloInstruction* user, HloInstruction* operand) {
    VLOG(3) << "Replacing scatter user " << user->name();
    if (user->opcode() == HloOpcode::kSlice) {
      return ReplaceInstruction(user, operand);
    }

    // Create the replacement instruction with new shape.
    HloInstruction* new_user = nullptr;
    if (user->IsElementwise()) {
      auto new_shape = [operand](HloInstruction* from) {
        return ShapeUtil::MakeShape(from->shape().element_type(),
                                    operand->shape().dimensions());
      };
      std::vector<HloInstruction*> new_operands;
      absl::c_transform(user->operands(), std::back_inserter(new_operands),
                        [&](HloInstruction* op) {
                          auto it = replacements_.find(op);
                          return it != replacements_.end()
                                     ? it->second
                                     : CreateSliceFrom(op, new_shape(op));
                        });
      new_user = user->AddInstruction(
          user->CloneWithNewOperands(new_shape(user), new_operands));
    } else {
      auto* gte = Cast<HloGetTupleElementInstruction>(user);
      TF_ASSIGN_OR_RETURN(new_user,
                          MakeGetTupleElementHlo(operand, gte->tuple_index(),
                                                 &user->metadata()));
    }

    // Replace slice user instructions recursively.
    return ReplaceAllUsersRecursive(user, new_user);
  }

  absl::flat_hash_map<HloInstruction*, HloInstruction*> replacements_;
};

}  // namespace

StatusOr<bool> ScatterSliceSimplifier::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return ScatterSliceSimplifierVisitor{}.RunOnModule(module, execution_threads);
}

}  // namespace xla
