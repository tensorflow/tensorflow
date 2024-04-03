/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_sort_rewriter.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/runtime/cub_sort_thunk.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

// Analyze sort comparer function.
struct SortComputationAnalysis {
  int key_operand;  // 0 or 1
  bool descending;
};

std::optional<SortComputationAnalysis> AnalyzeSortComputation(
    const HloComputation* computation) {
  // Root instruction must be a comparison with a valid direction.
  const HloCompareInstruction* compare =
      DynCast<HloCompareInstruction>(computation->root_instruction());
  if (compare == nullptr || compare->direction() == ComparisonDirection::kEq ||
      compare->direction() == ComparisonDirection::kNe) {
    return std::nullopt;
  }

  // Compare should operate on the function parameters for a single tensor.
  const HloParameterInstruction* param0 =
      DynCast<HloParameterInstruction>(compare->operand(0));
  const HloParameterInstruction* param1 =
      DynCast<HloParameterInstruction>(compare->operand(1));
  if (param0 == nullptr || param1 == nullptr) {
    return std::nullopt;
  }

  // When sorting a pair of tensors, the parameters should be adjacent.
  int index0 = param0->parameter_number();
  int index1 = param1->parameter_number();
  int first_index = std::min(index0, index1);
  if (first_index % 2 != 0 || std::max(index0, index1) != first_index + 1) {
    return std::nullopt;
  }

  // Return the tensor index and the sort direction.
  bool descending = compare->direction() == ComparisonDirection::kGt ||
                    compare->direction() == ComparisonDirection::kGe;
  bool reverse = first_index != index0;
  return SortComputationAnalysis{first_index / 2, descending != reverse};
}

// Create runner for CUB sort operation.
absl::StatusOr<std::unique_ptr<CubSortRunnerInterface>> CreateRunner(
    HloSortInstruction* sort_op, const SortComputationAnalysis& sort_config) {
  int value_index = 1 - sort_config.key_operand;
  return CubSortRunnerInterface::Create(
      sort_op->operand(sort_config.key_operand)->shape().element_type(),
      sort_op->operand_count() == 2
          ? std::optional(sort_op->operand(value_index)->shape().element_type())
          : std::nullopt);
}

// Verify that the sort tensor shape is supported by CUB.
bool IsCubCompatibleSort(HloSortInstruction* sort_op) {
  VLOG(1) << "Sort instruction: " << sort_op->name();
  if (sort_op->operand_count() != 1 && sort_op->operand_count() != 2) {
    VLOG(2) << "Unsupported operand count: " << sort_op->operand_count();
    return false;
  }
  if (sort_op->operand(0)->shape().rank() != 1) {
    VLOG(2) << "Only 1D shapes are supported";
    return false;
  }
  if (sort_op->operand(0)->shape().dimensions(0) <
      GpuSortRewriter::kSortSizeThreshold) {
    VLOG(2) << "Tensor shape size is too small to see an improvement";
    return false;
  }

  auto sort_config =
      AnalyzeSortComputation(sort_op->called_computations().front());
  if (!sort_config.has_value()) {
    VLOG(2) << "Only simple compare computations are supported";
    return false;
  }
  if (!CreateRunner(sort_op, *sort_config).ok()) {
    VLOG(2) << "Unsupported operand types (no compiled CUB kernels)";
    return false;
  }
  VLOG(2) << "Sort operation is compatible";
  return true;
}

// Restore the result shape after sorting a pair of tensors.
// The trailing argument is the scratch buffer which should be discarded.
HloInstruction* UnpackResultPair(HloSortInstruction* sort_op,
                                 HloInstruction* custom_call, bool swap) {
  HloComputation* parent = sort_op->parent();
  HloInstruction* gte0 =
      parent->AddInstruction(HloInstruction::CreateGetTupleElement(
          sort_op->operand(0)->shape(), custom_call, swap ? 1 : 0));
  HloInstruction* gte1 =
      parent->AddInstruction(HloInstruction::CreateGetTupleElement(
          sort_op->operand(1)->shape(), custom_call, swap ? 0 : 1));
  return parent->AddInstruction(HloInstruction::CreateTuple({gte0, gte1}));
}

}  // namespace

// Rewrites a single sort instruction with a custom call.
absl::StatusOr<bool> GpuSortRewriter::RunOnInstruction(
    HloSortInstruction* sort_op) {
  // Get the sort tensor index and direction.
  SortComputationAnalysis sort_config =
      AnalyzeSortComputation(sort_op->called_computations().front()).value();

  // Get scratch size requirements from CUB.
  TF_ASSIGN_OR_RETURN(auto runner, CreateRunner(sort_op, sort_config));
  TF_ASSIGN_OR_RETURN(
      int64_t scratch_size,
      runner->GetScratchSize(sort_op->operand(0)->shape().dimensions(0)));

  // Values are only present if sorting a pair of tensors.
  HloInstruction* keys = sort_op->mutable_operand(0);
  HloInstruction* values = nullptr;
  if (sort_op->operand_count() == 2) {
    values = sort_op->mutable_operand(1);
    if (sort_config.key_operand == 1) {
      std::swap(keys, values);
    }
  }

  // Build the resulting shape for the custom call.
  std::vector<Shape> shapes{keys->shape()};
  std::vector<HloInstruction*> operands{keys};
  if (values != nullptr) {
    shapes.push_back(values->shape());
    operands.push_back(values);
  }
  shapes.push_back(ShapeUtil::MakeShape(U8, {scratch_size}));
  Shape call_shape = ShapeUtil::MakeTupleShape(absl::MakeSpan(shapes));

  // Build the custom call instruction.
  HloInstruction* custom_call =
      sort_op->parent()->AddInstruction(HloInstruction::CreateCustomCall(
          call_shape, absl::MakeSpan(operands), kCubDeviceRadixSortTarget));

  xla::SortOptions backend_config;
  backend_config.set_descending(sort_config.descending);
  TF_RETURN_IF_ERROR(custom_call->set_backend_config(backend_config));

  // Build the replacement instruction.
  HloInstruction* replacement;
  if (sort_op->operand_count() == 1) {
    replacement =
        sort_op->parent()->AddInstruction(HloInstruction::CreateGetTupleElement(
            sort_op->shape(), custom_call, 0));
  } else {
    replacement = UnpackResultPair(sort_op, custom_call,
                                   /*swap=*/sort_config.key_operand == 1);
  }

  // Replace sort operation with custom call followed by GTE.
  TF_RETURN_IF_ERROR(
      sort_op->parent()->ReplaceInstruction(sort_op, replacement));
  return true;
}

// Rewrites the sorts in the given computation into calls to CUB.
absl::StatusOr<bool> GpuSortRewriter::RunOnComputation(
    HloComputation* computation) {
  std::vector<HloSortInstruction*> sort_ops;
  for (auto* inst : computation->instructions()) {
    HloSortInstruction* sort = DynCast<HloSortInstruction>(inst);
    if (sort != nullptr && IsCubCompatibleSort(sort)) {
      sort_ops.push_back(sort);
    }
  }
  bool changed = false;
  for (auto* sort : sort_ops) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnInstruction(sort));
    changed |= result;
  }
  return changed;
}

// Replace compatible sort operations with custom calls.
absl::StatusOr<bool> GpuSortRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(2, "GpuSortRewriter::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  XLA_VLOG_LINES(2, "GpuSortRewriter::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace gpu
}  // namespace xla
