/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/tools/comparison/original_tensor_summary_propagator.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/hlo/tools/comparison/tensor_summary_util.h"
#include "xla/literal.h"
#include "xla/permutation_util.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla::numerics::comparison {

OriginalTensorSummaryPropagator::OriginalTensorSummaryPropagator(
    const HloModule* original_module,
    OriginalTensorSummaryCallback&& on_propagated_tensor_summary,
    IsOriginalTensorAlreadyRecoveredCallback&&
        is_original_tensor_already_recovered)
    : original_module_(original_module),
      on_propagated_tensor_summary_(std::move(on_propagated_tensor_summary)),
      is_original_tensor_already_recovered_(
          std::move(is_original_tensor_already_recovered)) {
  for (HloComputation* computation : original_module_->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      name_to_instruction_[instruction->name()] = instruction;
    }
  }
}

absl::Status OriginalTensorSummaryPropagator::Initialize() {
  call_stack_.push_back(
      CallFrame::ForComputation(original_module_->entry_computation()));
  return PropagateConstantsAndParameters();
}

absl::Status OriginalTensorSummaryPropagator::OnEnterCall(
    ScopeInstruction call_instruction,
    absl::string_view instruction_in_called_computation) {
  CallFrame child_call_frame = CallFrame::ForCallInstruction(call_instruction);
  CallFrame& current_call_frame = call_stack_.back();
  auto child_call_frame_state_handle =
      current_call_frame.child_states.extract(call_instruction);
  if (!child_call_frame_state_handle.empty()) {
    child_call_frame.current_state =
        std::move(child_call_frame_state_handle.mapped());
  }
  auto hlo_instruction_in_called_computation_ptr =
      name_to_instruction_.find(instruction_in_called_computation);
  if (hlo_instruction_in_called_computation_ptr == name_to_instruction_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Instruction ", instruction_in_called_computation,
                     " not found in the original module."));
  }
  child_call_frame.computation =
      hlo_instruction_in_called_computation_ptr->second->parent();
  auto hlo_call_instruction_ptr =
      name_to_instruction_.find(call_instruction.instruction_name);
  if (hlo_call_instruction_ptr == name_to_instruction_.end()) {
    if (absl::EndsWith(call_instruction.instruction_name, "?")) {
      LOG(WARNING) << "Skipping guessed scope instruction '"
                   << call_instruction.instruction_name
                   << "' as it was not found in the original module.";
      return absl::OkStatus();
    }
    return absl::NotFoundError(
        absl::StrCat("Instruction ", call_instruction.instruction_name,
                     " not found in the original module."));
  }
  HloOpcode opcode = hlo_call_instruction_ptr->second->opcode();
  // Parameter propagation is only enabled for kCall, kWhile, and kConditional
  // because these are the only opcodes where the full operand tensors are
  // passed as parameters to the invoked computations. Other call-like
  // instructions (e.g., kMap, kReduce) operate on elements or slices of the
  // operand tensors.
  if (opcode == HloOpcode::kCall || opcode == HloOpcode::kWhile ||
      opcode == HloOpcode::kConditional) {
    child_call_frame.should_propagate_parameters = true;
  }
  call_stack_.push_back(std::move(child_call_frame));
  return PropagateConstantsAndParameters();
}

absl::Status OriginalTensorSummaryPropagator::OnExitCall(
    ScopeInstruction call_instruction) {
  CHECK(!call_stack_.empty());
  CallFrame call_frame = std::move(call_stack_.back());
  call_stack_.pop_back();

  // Mismatching call instructions indicates a bug in the call propagation
  // logic in this file. Hence we use a CHECK here rather than returning an
  // error status.
  CHECK_EQ(call_frame.call_instruction, call_instruction);

  auto it = name_to_instruction_.find(call_instruction.instruction_name);
  if (it != name_to_instruction_.end()) {
    HloInstruction* instr = it->second;
    if (IsCallLike(*instr)) {
      absl::Status status = ShapeUtil::ForEachSubshapeWithStatus(
          instr->shape(),
          [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
            if (!subshape.IsArray()) {
              return absl::OkStatus();
            }
            if (!ShouldPropagateTo(instr, index)) {
              return absl::OkStatus();
            }
            OriginalTensorSummary summary;
            AbsoluteScopedTensorKey scoped_key =
                GetCurrentAbsoluteScopedTensorKey(instr->name(), index);
            RETURN_IF_ERROR(
                InvokePropagatedCallback(scoped_key, nullptr, summary));
            return absl::OkStatus();
          });
      RETURN_IF_ERROR(status);
    }
  }

  return absl::OkStatus();
}

absl::Status OriginalTensorSummaryPropagator::OnNextIteration(
    ScopeInstruction next_iteration_instruction) {
  CHECK(!call_stack_.empty());
  CallFrame& current_call_frame = call_stack_.back();
  current_call_frame.call_instruction = next_iteration_instruction;
  current_call_frame.child_states.clear();
  current_call_frame.propagated_tensors.clear();
  return PropagateConstantsAndParameters();
}

absl::Status OriginalTensorSummaryPropagator::Process(
    const AbsoluteScopedTensorKey& original_tensor_key,
    std::shared_ptr<const tensor_transformation::TensorTransformation>
        pending_transformation,
    const OriginalTensorSummary& root_tensor_summary) {
  for (const auto& scope : original_tensor_key.scope_instructions) {
    if (absl::EndsWith(scope.instruction_name, "?")) {
      ++processing_metrics_.skipped_unrecoverable_tensor_summaries;
      return absl::OkStatus();
    }
  }

  // If any scope instruction contains an any-iteration wildcard, we delay its
  // propagation. We store it in wildcard_summaries_ so that it can be
  // instantiated and processed later when a specific iteration of the matching
  // loop is encountered.
  for (const auto& scope : original_tensor_key.scope_instructions) {
    if (scope.MatchesAnyIteration()) {
      wildcard_summaries_.push_back(
          {original_tensor_key, pending_transformation, root_tensor_summary});
      return absl::OkStatus();
    }
  }

  ++processing_metrics_.recovered_from_runtime_count;

  const auto& new_scopes = original_tensor_key.scope_instructions;
  auto current_scopes = GetCurrentScopeInstructions();

  size_t i = 0;
  while (i < new_scopes.size() && i < current_scopes.size() &&
         new_scopes[i] == current_scopes[i]) {
    i++;
  }

  std::vector<RecoveredTensorSummary> wildcards_to_process;
  // Checks pending wildcard summaries to see if they match the current scopes
  // up to `index`. If a wildcard summary matches the scope prefix and has a
  // wildcard at `index` for the same loop instruction, we instantiate the
  // wildcard with the concrete iteration index. We also track instantiated keys
  // in `processed_wildcards_` to ensure we only process them once.
  auto check_and_add_wildcards = [&](size_t index) {
    for (const auto& w : wildcard_summaries_) {
      if (w.original_tensor_key.scope_instructions.size() > index) {
        bool prefix_match = true;
        for (size_t k = 0; k < index; ++k) {
          if (w.original_tensor_key.scope_instructions[k] != new_scopes[k]) {
            prefix_match = false;
            break;
          }
        }
        if (prefix_match) {
          const auto& w_scope = w.original_tensor_key.scope_instructions[index];
          if (w_scope.instruction_name == new_scopes[index].instruction_name &&
              w_scope.MatchesAnyIteration()) {
            auto instantiated_key = w.original_tensor_key;
            instantiated_key.scope_instructions[index].iteration_index =
                new_scopes[index].iteration_index;
            if (processed_wildcards_.insert(instantiated_key.ToString())
                    .second) {
              wildcards_to_process.push_back({instantiated_key,
                                              w.pending_transformation,
                                              w.original_tensor_summary});
            }
          }
        }
      }
    }
  };

  if (i < new_scopes.size() && i < current_scopes.size() &&
      new_scopes[i].instruction_name == current_scopes[i].instruction_name) {
    while (call_stack_.size() > i + 2) {
      RETURN_IF_ERROR(OnExitCall(call_stack_.back().call_instruction));
    }
    RETURN_IF_ERROR(OnNextIteration(new_scopes[i]));
    check_and_add_wildcards(i);
    for (size_t j = i + 1; j < new_scopes.size(); ++j) {
      check_and_add_wildcards(j);
      absl::string_view instruction_in_called_computation;
      if (j < new_scopes.size()) {
        instruction_in_called_computation = new_scopes[j + 1].instruction_name;
      } else {
        instruction_in_called_computation =
            original_tensor_key.tensor_key.instruction_name;
      }
      RETURN_IF_ERROR(
          OnEnterCall(new_scopes[j], instruction_in_called_computation));
    }
  } else {
    while (call_stack_.size() > i + 1) {
      RETURN_IF_ERROR(OnExitCall(call_stack_.back().call_instruction));
    }
    for (size_t j = i; j < new_scopes.size(); ++j) {
      check_and_add_wildcards(j);
      absl::string_view instruction_in_called_computation;
      if (j + 1 < new_scopes.size()) {
        instruction_in_called_computation = new_scopes[j + 1].instruction_name;
      } else {
        instruction_in_called_computation =
            original_tensor_key.tensor_key.instruction_name;
      }
      RETURN_IF_ERROR(
          OnEnterCall(new_scopes[j], instruction_in_called_computation));
    }
  }

  for (const auto& w : wildcards_to_process) {
    RETURN_IF_ERROR(Process(w.original_tensor_key, w.pending_transformation,
                            w.original_tensor_summary));
  }

  HloInstruction* instruction = nullptr;
  auto it = name_to_instruction_.find(
      original_tensor_key.tensor_key.instruction_name);
  if (it != name_to_instruction_.end()) {
    instruction = it->second;
  }

  if (instruction == nullptr) {
    return absl::NotFoundError(absl::StrCat(
        "Instruction ", original_tensor_key.tensor_key.instruction_name,
        " not found in the original module."));
  }

  RETURN_IF_ERROR(
      PropagateBackward(instruction, original_tensor_key.tensor_key.shape_index,
                        pending_transformation, root_tensor_summary));
  RETURN_IF_ERROR(InvokePropagatedCallback(
      original_tensor_key, pending_transformation, root_tensor_summary));
  RETURN_IF_ERROR(
      PropagateForward(instruction, original_tensor_key.tensor_key.shape_index,
                       pending_transformation, root_tensor_summary));

  return absl::OkStatus();
}

absl::Status OriginalTensorSummaryPropagator::Finish() {
  while (!call_stack_.empty()) {
    RETURN_IF_ERROR(OnExitCall(call_stack_.back().call_instruction));
  }
  return absl::OkStatus();
}
std::vector<ScopeInstruction>
OriginalTensorSummaryPropagator::GetCurrentScopeInstructions() const {
  std::vector<ScopeInstruction> scope_instructions;
  if (call_stack_.empty()) {
    return scope_instructions;
  }
  scope_instructions.reserve(call_stack_.size() - 1);

  // Skip the right call frame.
  for (size_t i = 1; i < call_stack_.size(); ++i) {
    scope_instructions.push_back(call_stack_[i].call_instruction);
  }
  return scope_instructions;
}
namespace {
using FloatSummary = ::xla::comparison::FloatSummary;
using FloatBlockSummary = ::xla::comparison::FloatBlockSummary;

absl::StatusOr<FloatSummary> CalculateStats(const Literal& literal) {
  if (!literal.shape().IsArray()) {
    return absl::InvalidArgumentError(
        "Can only calculate stats for array literals");
  }
  double min_v = std::numeric_limits<double>::infinity();
  double max_v = -std::numeric_limits<double>::infinity();
  double sum = 0;
  double sum_sq = 0;
  const int64_t num_elements = ShapeUtil::ElementsIn(literal.shape());
  if (num_elements == 0) {
    return FloatSummary{/*block_summaries=*/{FloatBlockSummary{}},
                        /*split_spec=*/{}};
  }

  auto element_visitor = [&](auto native_value) {
    double val;
    if constexpr (is_complex_v<decltype(native_value)>) {
      val = static_cast<double>(native_value.real());
    } else {
      val = static_cast<double>(native_value);
    }
    if (val < min_v) {
      min_v = val;
    }
    if (val > max_v) {
      max_v = val;
    }
    sum += val;
    sum_sq += val * val;
  };
  RETURN_IF_ERROR(primitive_util::ArrayTypeSwitch(
      [&](auto primitive_type_constant) -> absl::Status {
        if constexpr (primitive_util::IsComplexType(primitive_type_constant) ||
                      primitive_util::IsFloatingPointType(
                          primitive_type_constant) ||
                      primitive_util::IsIntegralType(primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          for (NativeT value : literal.data<NativeT>()) {
            element_visitor(value);
          }
        }
        return absl::OkStatus();
      },
      literal.shape().element_type()));

  double mean = sum / num_elements;
  double variance = (sum_sq / num_elements) - (mean * mean);
  double stddev = variance > 0 ? std::sqrt(variance) : 0.0;
  return FloatSummary{/*block_summaries=*/{FloatBlockSummary{
                          /*block_indices=*/{},
                          /*min=*/static_cast<float>(min_v),
                          /*max=*/static_cast<float>(max_v),
                          /*mean=*/static_cast<float>(mean),
                          /*stddev=*/static_cast<float>(stddev),
                          /*count=*/static_cast<float>(num_elements)}},
                      /*split_spec=*/{}};
}
}  // namespace

absl::Status
OriginalTensorSummaryPropagator::PropagateConstantsAndParameters() {
  CallFrame& current_call_frame = call_stack_.back();
  if (current_call_frame.call_instruction.iteration_index == -1) {
    // -1 means the instruction is hoisted out of the loop, in this case the
    // actual loop computation is not actually executed. Hence we don't need
    // to propagate constants in this case.
    return absl::OkStatus();
  }
  HloComputation* current_computation = current_call_frame.computation;
  for (HloInstruction* instruction : current_computation->instructions()) {
    switch (instruction->opcode()) {
      case HloOpcode::kConstant: {
        absl::Status status = ShapeUtil::ForEachSubshapeWithStatus(
            instruction->shape(),
            [&](const Shape& subshape,
                const ShapeIndex& index) -> absl::Status {
              if (!subshape.IsArray()) {
                return absl::OkStatus();
              }
              if (!ShouldPropagateTo(instruction, index)) {
                return absl::OkStatus();
              }
              ASSIGN_OR_RETURN(Literal sub_literal, Literal::Make(subshape));
              RETURN_IF_ERROR(sub_literal.CopyFrom(
                  LiteralSlice(instruction->literal(), index)));
              absl::StatusOr<FloatSummary> stats_or =
                  CalculateStats(sub_literal);
              if (!stats_or.ok()) {
                return stats_or.status();
              }
              auto stats = *std::move(stats_or);
              OriginalTensorSummary summary;
              summary.dimensions.assign(subshape.dimensions().begin(),
                                        subshape.dimensions().end());
              summary.summaries.push_back(stats);
              AbsoluteScopedTensorKey scoped_key =
                  GetCurrentAbsoluteScopedTensorKey(instruction->name(), index);
              RETURN_IF_ERROR(
                  InvokePropagatedCallback(scoped_key, nullptr, summary));
              call_stack_.back().propagated_tensors.insert(
                  TensorKey::Create(instruction->name(), index));
              return PropagateForward(instruction, index, nullptr, summary);
            });
        RETURN_IF_ERROR(status);
        break;
      }
      case HloOpcode::kIota: {
        if (!instruction->shape().IsArray()) {
          continue;
        }
        if (!ShouldPropagateTo(instruction, {})) {
          continue;
        }
        ASSIGN_OR_RETURN(Literal iota_literal,
                         Literal::Make(instruction->shape()));
        const int64_t iota_dimension =
            Cast<HloIotaInstruction>(instruction)->iota_dimension();
        RETURN_IF_ERROR(primitive_util::ArrayTypeSwitch(
            [&](auto primitive_type_constant) -> absl::Status {
              if constexpr (primitive_util::IsIntegralType(
                                primitive_type_constant) ||
                            primitive_util::IsFloatingPointType(
                                primitive_type_constant) ||
                            primitive_util::IsComplexType(
                                primitive_type_constant)) {
                using NativeT =
                    primitive_util::NativeTypeOf<primitive_type_constant>;
                RETURN_IF_ERROR(iota_literal.Populate<NativeT>(
                    [&](absl::Span<const int64_t> multi_index) -> NativeT {
                      if constexpr (is_complex_v<NativeT>) {
                        return NativeT(multi_index[iota_dimension], 0);
                      } else {
                        return static_cast<NativeT>(
                            multi_index[iota_dimension]);
                      }
                    }));
              }
              return absl::OkStatus();
            },
            instruction->shape().element_type()));
        absl::StatusOr<FloatSummary> stats_or = CalculateStats(iota_literal);
        if (!stats_or.ok()) {
          return stats_or.status();
        }
        OriginalTensorSummary summary;
        summary.dimensions.assign(instruction->shape().dimensions().begin(),
                                  instruction->shape().dimensions().end());
        summary.summaries.push_back(*std::move(stats_or));
        AbsoluteScopedTensorKey scoped_key =
            GetCurrentAbsoluteScopedTensorKey(instruction->name(), {});
        RETURN_IF_ERROR(InvokePropagatedCallback(scoped_key, nullptr, summary));
        call_stack_.back().propagated_tensors.insert(
            TensorKey::Create(instruction->name(), {}));
        RETURN_IF_ERROR(PropagateForward(instruction, {}, nullptr, summary));
        break;
      }
      case HloOpcode::kParameter: {
        const int64_t param_no = instruction->parameter_number();
        absl::Status status = ShapeUtil::ForEachSubshapeWithStatus(
            instruction->shape(),
            [&](const Shape& subshape,
                const ShapeIndex& index) -> absl::Status {
              if (!subshape.IsArray()) {
                return absl::OkStatus();
              }
              if (!ShouldPropagateTo(instruction, index)) {
                return absl::OkStatus();
              }
              CallArgKey key{/*arg_number=*/param_no,
                             /*shape_index=*/ShapeIndex(index)};
              if (current_call_frame.should_propagate_parameters) {
                auto it =
                    current_call_frame.current_state.arg_summaries.find(key);
                if (it !=
                    current_call_frame.current_state.arg_summaries.end()) {
                  const OriginalTensorSummary& summary = it->second;
                  AbsoluteScopedTensorKey scoped_key =
                      GetCurrentAbsoluteScopedTensorKey(instruction->name(),
                                                        index);
                  RETURN_IF_ERROR(
                      InvokePropagatedCallback(scoped_key, nullptr, summary));
                  call_stack_.back().propagated_tensors.insert(
                      TensorKey::Create(instruction->name(), index));
                  return PropagateForward(instruction, index, nullptr, summary);
                }
              }
              OriginalTensorSummary summary;
              AbsoluteScopedTensorKey scoped_key =
                  GetCurrentAbsoluteScopedTensorKey(instruction->name(), index);
              RETURN_IF_ERROR(
                  InvokePropagatedCallback(scoped_key, nullptr, summary));
              return absl::OkStatus();
            });
        RETURN_IF_ERROR(status);
        break;
      }
      default:
        break;
    }
  }
  return absl::OkStatus();
}

absl::Status OriginalTensorSummaryPropagator::PropagateForward(
    const HloInstruction* starting_instruction,
    ShapeIndexView starting_shape_index,
    std::shared_ptr<const tensor_transformation::TensorTransformation>
        pending_transformation,
    const OriginalTensorSummary& original_tensor_summary) {
  if (starting_instruction->shape().IsTuple()) {
    if (starting_shape_index.empty()) {
      return absl::InternalError(
          "Shape index cannot be empty for tuple-shaped instruction");
    }
  } else {
    if (!starting_shape_index.empty()) {
      return absl::InternalError(
          "Shape index must be empty for non-tuple-shaped instruction");
    }
  }

  for (HloInstruction* user : starting_instruction->users()) {
    if (IsCallLike(*user)) {
      for (int i = 0; i < user->operand_count(); ++i) {
        if (user->operand(i) == starting_instruction) {
          call_stack_.back()
              .child_states[ScopeInstruction::Create(user->name())]
              .arg_summaries[{
                  /*arg_number=*/static_cast<int64_t>(i),
                  /*shape_index=*/ShapeIndex(starting_shape_index)}] =
              original_tensor_summary;
        }
      }
      continue;
    }

    if (user->opcode() == HloOpcode::kTuple) {
      for (int i = 0; i < user->operand_count(); ++i) {
        if (user->operand(i) != starting_instruction) {
          continue;
        }
        ShapeIndex new_shape_index = {static_cast<int64_t>(i)};
        new_shape_index.insert(new_shape_index.end(),
                               starting_shape_index.begin(),
                               starting_shape_index.end());
        if (!ShouldPropagateTo(user, new_shape_index)) {
          continue;
        }
        RETURN_IF_ERROR(DoPropagateForward(user, new_shape_index,
                                           pending_transformation,
                                           original_tensor_summary));
      }
      continue;
    }

    if (starting_instruction->shape().IsTuple()) {
      if (user->opcode() == HloOpcode::kGetTupleElement &&
          user->tuple_index() == starting_shape_index[0]) {
        ShapeIndex new_shape_index(starting_shape_index.begin() + 1,
                                   starting_shape_index.end());

        if (!ShouldPropagateTo(user, new_shape_index)) {
          continue;
        }
        RETURN_IF_ERROR(DoPropagateForward(user, new_shape_index,
                                           pending_transformation,
                                           original_tensor_summary));
      }
    } else {
      // Non-tuple-shaped instruction, non-call-like user.
      if (!ShouldPropagateTo(user, {})) {
        continue;
      }

      std::shared_ptr<const tensor_transformation::TensorTransformation>
          new_transform = pending_transformation;
      switch (user->opcode()) {
        case HloOpcode::kCopy: {
          break;
        }
        case HloOpcode::kBitcast:
          if (starting_instruction->shape().element_type() !=
              user->shape().element_type()) {
            // We don't propagate bitcasts if element type changes because the
            // meaning of the tensor changes and the summary is not valid
            // anymore.
            break;
          }
          [[fallthrough]];
        case HloOpcode::kReshape: {
          std::vector<int64_t> output_dimensions(
              user->shape().dimensions().begin(),
              user->shape().dimensions().end());
          auto reshape_op =
              std::make_shared<tensor_transformation::TensorTransformation>(
                  tensor_transformation::Reshape{
                      /*continuation=*/nullptr,
                      /*output_dimensions=*/output_dimensions});
          new_transform = tensor_transformation::AppendContinuation(
              pending_transformation, reshape_op);
          break;
        }
        case HloOpcode::kBroadcast: {
          std::vector<int64_t> output_dimensions(
              user->shape().dimensions().begin(),
              user->shape().dimensions().end());
          std::vector<int64_t> broadcast_dimensions(user->dimensions().begin(),
                                                    user->dimensions().end());
          auto bcast_op =
              std::make_shared<tensor_transformation::TensorTransformation>(
                  tensor_transformation::Broadcast{
                      /*continuation=*/nullptr,
                      /*output_dimensions=*/output_dimensions,
                      /*broadcast_dimensions=*/broadcast_dimensions});
          new_transform = tensor_transformation::AppendContinuation(
              pending_transformation, bcast_op);
          break;
        }
        case HloOpcode::kTranspose: {
          std::vector<int64_t> output_dimensions(
              user->shape().dimensions().begin(),
              user->shape().dimensions().end());
          std::vector<int64_t> inverse_permutation =
              xla::InversePermutation(user->dimensions());

          auto transpose_as_bcast_op =
              std::make_shared<tensor_transformation::TensorTransformation>(
                  tensor_transformation::Broadcast{
                      /*continuation=*/nullptr,
                      /*output_dimensions=*/output_dimensions,
                      /*broadcast_dimensions=*/inverse_permutation});
          new_transform = tensor_transformation::AppendContinuation(
              pending_transformation, transpose_as_bcast_op);
          break;
        }
        default:
          continue;
      }
      RETURN_IF_ERROR(
          DoPropagateForward(user, {}, new_transform, original_tensor_summary));
    }
  }

  return absl::OkStatus();
}

absl::Status OriginalTensorSummaryPropagator::PropagateBackward(
    const HloInstruction* starting_instruction,
    ShapeIndexView starting_shape_index,
    std::shared_ptr<const tensor_transformation::TensorTransformation>
        pending_transformation,
    const OriginalTensorSummary& original_tensor_summary) {
  if (starting_instruction->shape().IsTuple()) {
    if (starting_shape_index.empty()) {
      return absl::InternalError(
          "Shape index cannot be empty for tuple-shaped instruction");
    }
  } else {
    if (!starting_shape_index.empty()) {
      return absl::InternalError(
          "Shape index must be empty for non-tuple-shaped instruction");
    }
  }

  if (starting_instruction->opcode() == HloOpcode::kGetTupleElement) {
    const HloInstruction* operand = starting_instruction->operand(0);
    ShapeIndex new_shape_index = {starting_instruction->tuple_index()};
    new_shape_index.insert(new_shape_index.end(), starting_shape_index.begin(),
                           starting_shape_index.end());
    if (!ShouldPropagateTo(operand, new_shape_index)) {
      return absl::OkStatus();
    }
    return DoPropagateBackward(operand, new_shape_index, pending_transformation,
                               original_tensor_summary);
  }

  if (starting_instruction->opcode() == HloOpcode::kTuple) {
    const HloInstruction* operand =
        starting_instruction->operand(starting_shape_index[0]);
    ShapeIndex new_shape_index(starting_shape_index.begin() + 1,
                               starting_shape_index.end());
    if (!ShouldPropagateTo(operand, new_shape_index)) {
      return absl::OkStatus();
    }
    return DoPropagateBackward(operand, new_shape_index, pending_transformation,
                               original_tensor_summary);
  }
  if (starting_instruction->operand_count() != 1 ||
      starting_instruction->shape().IsTuple()) {
    // All supported ops are unary.
    return absl::OkStatus();
  }
  // Non-tuple-shaped instruction.
  const HloInstruction* operand = starting_instruction->operand(0);
  if (!ShouldPropagateTo(operand, {})) {
    return absl::OkStatus();
  }
  std::shared_ptr<const tensor_transformation::TensorTransformation>
      new_transform = pending_transformation;
  switch (starting_instruction->opcode()) {
    case HloOpcode::kCopy:
      break;
    case HloOpcode::kBitcast:
      if (starting_instruction->shape().element_type() !=
          operand->shape().element_type()) {
        // We don't propagate bitcasts if element type changes because the
        // meaning of the tensor changes and the summary is not valid
        // anymore.
        break;
      }
      [[fallthrough]];
    case HloOpcode::kReshape: {
      std::vector<int64_t> output_dimensions(
          operand->shape().dimensions().begin(),
          operand->shape().dimensions().end());
      auto reshape_op =
          std::make_shared<tensor_transformation::TensorTransformation>(
              tensor_transformation::Reshape{
                  /*continuation=*/nullptr,
                  /*output_dimensions=*/output_dimensions});
      new_transform = tensor_transformation::AppendContinuation(
          pending_transformation, reshape_op);
      break;
    }
    case HloOpcode::kTranspose: {
      std::vector<int64_t> output_dimensions(
          operand->shape().dimensions().begin(),
          operand->shape().dimensions().end());
      std::vector<int64_t> permutation(
          starting_instruction->dimensions().begin(),
          starting_instruction->dimensions().end());
      auto transpose_as_bcast_op =
          std::make_shared<tensor_transformation::TensorTransformation>(
              tensor_transformation::Broadcast{
                  /*continuation=*/nullptr,
                  /*output_dimensions=*/output_dimensions,
                  /*broadcast_dimensions=*/permutation});
      new_transform = tensor_transformation::AppendContinuation(
          pending_transformation, transpose_as_bcast_op);
      break;
    }
    case HloOpcode::kBroadcast: {
      std::vector<int64_t> output_dimensions(
          operand->shape().dimensions().begin(),
          operand->shape().dimensions().end());
      // We use -1 to indicate dimensions that are created by broadcast, which
      // didn't exist in the original tensor.
      std::vector<int64_t> inverse_broadcast_dims(
          starting_instruction->shape().dimensions().size(), -1);
      const auto broadcast_dims = starting_instruction->dimensions();
      for (int j = 0; j < broadcast_dims.size(); ++j) {
        inverse_broadcast_dims[broadcast_dims[j]] = j;
      }
      auto bcast_op =
          std::make_shared<tensor_transformation::TensorTransformation>(
              tensor_transformation::Broadcast{
                  /*continuation=*/nullptr,
                  /*output_dimensions=*/output_dimensions,
                  /*broadcast_dimensions=*/inverse_broadcast_dims});
      new_transform = tensor_transformation::AppendContinuation(
          pending_transformation, bcast_op);
      break;
    }
    default:
      return absl::OkStatus();
  }
  return DoPropagateBackward(operand, {}, new_transform,
                             original_tensor_summary);
}

bool OriginalTensorSummaryPropagator::ShouldPropagateTo(
    const HloInstruction* instruction, ShapeIndexView shape_index) {
  TensorKey tensor_key =
      TensorKey::Create(instruction->name(), ShapeIndex(shape_index));
  CHECK(!call_stack_.empty());
  if (call_stack_.back().propagated_tensors.contains(tensor_key)) {
    return false;
  }
  AbsoluteScopedTensorKey scoped_key =
      GetCurrentAbsoluteScopedTensorKey(instruction->name(), shape_index);
  return !is_original_tensor_already_recovered_(scoped_key);
}

absl::Status OriginalTensorSummaryPropagator::DoPropagateBackward(
    const HloInstruction* instruction, ShapeIndexView shape_index,
    std::shared_ptr<const tensor_transformation::TensorTransformation>
        transformation,
    const OriginalTensorSummary& original_tensor_summary) {
  TensorKey tensor_key =
      TensorKey::Create(instruction->name(), ShapeIndex(shape_index));
  AbsoluteScopedTensorKey scoped_key =
      GetCurrentAbsoluteScopedTensorKey(instruction->name(), shape_index);
  call_stack_.back().propagated_tensors.insert(tensor_key);
  RETURN_IF_ERROR(PropagateBackward(instruction, shape_index, transformation,
                                    original_tensor_summary));
  // Here we invoke the callback after backward propagation is done so that
  // the callback is invoked in the order of execution.
  return InvokePropagatedCallback(scoped_key, transformation,
                                  original_tensor_summary);
}

absl::Status OriginalTensorSummaryPropagator::DoPropagateForward(
    const HloInstruction* instruction, ShapeIndexView shape_index,
    std::shared_ptr<const tensor_transformation::TensorTransformation>
        transformation,
    const OriginalTensorSummary& original_tensor_summary) {
  TensorKey tensor_key =
      TensorKey::Create(instruction->name(), ShapeIndex(shape_index));
  AbsoluteScopedTensorKey scoped_key =
      GetCurrentAbsoluteScopedTensorKey(instruction->name(), shape_index);
  RETURN_IF_ERROR(InvokePropagatedCallback(scoped_key, transformation,
                                           original_tensor_summary));
  call_stack_.back().propagated_tensors.insert(tensor_key);
  return PropagateForward(instruction, shape_index, transformation,
                          original_tensor_summary);
}

AbsoluteScopedTensorKey
OriginalTensorSummaryPropagator::GetCurrentAbsoluteScopedTensorKey(
    absl::string_view instruction_name, ShapeIndexView shape_index) const {
  return AbsoluteScopedTensorKey::Create(
      TensorKey::Create(instruction_name, ShapeIndex(shape_index)),
      GetCurrentScopeInstructions());
}

absl::Status OriginalTensorSummaryPropagator::InvokePropagatedCallback(
    const AbsoluteScopedTensorKey& original_tensor_key,
    std::shared_ptr<const tensor_transformation::TensorTransformation>
        pending_transformation,
    const OriginalTensorSummary& root_tensor_summary) {
  ++processing_metrics_.total_propagated_tensor_count;
  return on_propagated_tensor_summary_(
      original_tensor_key, pending_transformation, root_tensor_summary);
}

}  // namespace xla::numerics::comparison
