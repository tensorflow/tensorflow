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

#include "xla/tests/constraint_propagator.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/constraint_state.h"
#include "xla/tsl/platform/errors.h"
#include "xla/types.h"
#include "xla/util.h"

namespace xla {

IdentityElementType GetReductionIdentityElementType(
    const HloComputation& computation) {
  // TODO(b/77635120): Add init values, for min, max, and their arg variants.
  const HloInstruction* const root = computation.root_instruction();
  if (computation.num_parameters() != 2 || root->operand_count() != 2 ||
      root->operand(0)->opcode() != HloOpcode::kParameter ||
      root->operand(1)->opcode() != HloOpcode::kParameter ||
      root->operand(0) == root->operand(1)) {
    return IdentityElementType::kUnknown;
  }

  switch (root->opcode()) {
    case HloOpcode::kAdd:
      return IdentityElementType::kZero;
    case HloOpcode::kMultiply:
      return IdentityElementType::kOne;
    default:
      return IdentityElementType::kUnknown;
  }
}

namespace {

template <typename NativeT>
void SetConstraint(
    const HloInstruction* inst,
    absl::flat_hash_map<const HloInstruction*, ConstraintState>& states) {
  const Literal& literal = inst->literal();
  NativeT min_val = std::numeric_limits<NativeT>::max();
  NativeT max_val = std::numeric_limits<NativeT>::lowest();
  bool contains_zero = false;
  literal.EachCell<NativeT>(
      [&](absl::Span<const int64_t> indices, NativeT value) {
        if (value == static_cast<NativeT>(0)) {
          contains_zero = true;
        }
        min_val = std::min(min_val, value);
        max_val = std::max(max_val, value);
      });
  states[inst].AddConstraint(ConstraintInterval{
      static_cast<double>(min_val), static_cast<double>(max_val),
      /*exclude_zero=*/!contains_zero});
}

}  // namespace

absl::StatusOr<absl::flat_hash_map<const HloInstruction*, ConstraintState>>
ConstraintPropagator::Run(
    const HloModule& module,
    std::function<std::optional<uint64_t>(const HloInstruction*, int64_t)>
        get_index_known_zeroes) {
  ConstraintPropagator propagator(get_index_known_zeroes);
  auto computations = module.MakeComputationPostOrder();
  for (HloComputation* computation : computations) {
    TF_RETURN_IF_ERROR(propagator.Propagate(computation));
  }

  // Extract only the parameters
  absl::flat_hash_map<const HloInstruction*, ConstraintState> result;
  for (const HloInstruction* param :
       module.entry_computation()->parameter_instructions()) {
    result[param] = propagator.states_[param];
  }
  return result;
}

absl::Status ConstraintPropagator::Propagate(
    const HloComputation* computation) {
  TF_RETURN_IF_ERROR(SeedConstraints(computation));
  TF_RETURN_IF_ERROR(PropagateSeedConstraints(computation));
  absl::flat_hash_map<const HloInstruction*, ConstraintState> before;
  do {
    before = states_;
    TF_RETURN_IF_ERROR(PropagateConstraints(computation));
  } while (before != states_);
  return absl::OkStatus();
}

absl::Status ConstraintPropagator::SeedConstraints(
    const HloComputation* computation) {
  auto instructions = computation->MakeInstructionPostOrder();
  // Reverse topological (Use before Definition)
  for (auto it = instructions.rbegin(); it != instructions.rend(); ++it) {
    const HloInstruction* inst = *it;

    // Seed Hard Constraints depending on opcode
    switch (inst->opcode()) {
      case HloOpcode::kAbs:
        // Output is guaranteed to be non-negative.
        states_[inst].AddConstraint(ConstraintInterval::Positive());
        break;
      case HloOpcode::kConstant: {
        const Literal& literal = inst->literal();
        if (literal.shape().IsArray() && literal.element_count() > 0) {
          switch (literal.shape().element_type()) {
            case PRED:
              SetConstraint<bool>(inst, states_);
              break;
            case S8:
              SetConstraint<int8_t>(inst, states_);
              break;
            case S16:
              SetConstraint<int16_t>(inst, states_);
              break;
            case S32:
              SetConstraint<int32_t>(inst, states_);
              break;
            case S64:
              SetConstraint<int64_t>(inst, states_);
              break;
            case U8:
              SetConstraint<uint8_t>(inst, states_);
              break;
            case U16:
              SetConstraint<uint16_t>(inst, states_);
              break;
            case U32:
              SetConstraint<uint32_t>(inst, states_);
              break;
            case U64:
              SetConstraint<uint64_t>(inst, states_);
              break;
            case F16:
              SetConstraint<half>(inst, states_);
              break;
            case F32:
              SetConstraint<float>(inst, states_);
              break;
            case F64:
              SetConstraint<double>(inst, states_);
              break;
            case BF16:
              SetConstraint<bfloat16>(inst, states_);
              break;
            default:
              break;
          }
        }
        break;
      }
      case HloOpcode::kLog:
        // Log(x) => x > 0
        states_[inst->operand(0)].AddConstraint(
            ConstraintInterval::StrictPositive());
        break;
      case HloOpcode::kSqrt:
        // y = Sqrt(x) => x >= 0, y >= 0
        states_[inst->operand(0)].AddConstraint(ConstraintInterval::Positive());
        states_[inst].AddConstraint(ConstraintInterval::Positive());
        break;
      case HloOpcode::kRsqrt:
        // y = Rsqrt(x) => x > 0, y > 0
        states_[inst->operand(0)].AddConstraint(
            ConstraintInterval::StrictPositive());
        states_[inst].AddConstraint(ConstraintInterval::StrictPositive());
        break;
      case HloOpcode::kDivide:
        // Div(x, y) => y != 0
        states_[inst->operand(1)].AddConstraint(ConstraintInterval::NonZero());
        break;
      case HloOpcode::kPower:
        // Power(x, y) => x > 0
        // We heuristically force base to > 0 because:
        // - if exponent is negative, zero base would result in a NaN.
        // - if exponent is non-integer, zero base would result in a NaN.
        states_[inst->operand(0)].AddConstraint(ConstraintInterval::Positive());
        break;

      case HloOpcode::kDynamicSlice:
      case HloOpcode::kDynamicUpdateSlice: {
        // Indexes must be in the range [0, bound) where bound is determined by
        // the size of the slice and the shape of the tensor being sliced.
        const Shape& indexed_shape = inst->operand(0)->shape();
        const Shape& slice_shape = inst->opcode() == HloOpcode::kDynamicSlice
                                       ? inst->shape()
                                       : inst->operand(1)->shape();
        int64_t first_index = Cast<HloDynamicIndexInstruction>(inst)
                                  ->first_index_operand_number();

        for (int64_t i = first_index; i < inst->operand_count(); ++i) {
          int64_t sliced_dim = i - first_index;
          int64_t bound = ShapeUtil::GetDimension(indexed_shape, sliced_dim) -
                          ShapeUtil::GetDimension(slice_shape, sliced_dim);

          states_[inst->operand(i)].AddConstraint(
              ConstraintInterval{0.0, static_cast<double>(bound), false});

          StructuralConstraints s;
          int64_t physical_dim = PositionInContainer(
              inst->shape().layout().minor_to_major(), sliced_dim);
          if (physical_dim == 0) {
            // Lanes
            s.alignment = 128;
          } else if (physical_dim == 1) {
            // Sublanes
            s.alignment = 8;
          }
          if (get_index_known_zeroes_ != nullptr) {
            if (std::optional<uint64_t> current_known_zeroes =
                    get_index_known_zeroes_(inst, sliced_dim)) {
              s.known_zeroes_mask = current_known_zeroes;
            }
          }
          states_[inst->operand(i)].MergeStructural(s);
        }
        break;
      }

      case HloOpcode::kGather:
      case HloOpcode::kScatter: {
        const Shape& operand_shape = inst->operand(0)->shape();
        auto index_map =
            inst->opcode() == HloOpcode::kGather
                ? inst->gather_dimension_numbers().start_index_map()
                : inst->scatter_dimension_numbers()
                      .scatter_dims_to_operand_dims();
        int64_t bound = INT64_MAX;
        for (const auto dim_in_operand : index_map) {
          bound = std::min(bound, operand_shape.dimensions(dim_in_operand) - 1);
        }
        if (bound != INT64_MAX) {
          states_[inst->operand(1)].AddConstraint(
              ConstraintInterval{0.0, static_cast<double>(bound), false});
        }

        StructuralConstraints s;
        if (inst->opcode() == HloOpcode::kScatter) {
          s.needs_sorted_indices =
              Cast<HloScatterInstruction>(inst)->indices_are_sorted();
        } else {
          s.needs_sorted_indices =
              Cast<HloGatherInstruction>(inst)->indices_are_sorted();
        }
        states_[inst->operand(1)].MergeStructural(s);
        break;
      }

      case HloOpcode::kSort: {
        if (ShapeUtil::ElementIsIntegral(inst->operand(0)->shape())) {
          // Turn on no_duplicates for integer keys. It's basically shuffled
          // iota from [0, N) for unsigned, or [-N/2, N/2) for signed.
          StructuralConstraints s;
          s.no_duplicates = true;
          states_[inst->operand(0)].MergeStructural(s);
        }
        break;
      }

      case HloOpcode::kReduce:
      case HloOpcode::kReduceWindow: {
        int64_t first_init = inst->operand_count() / 2;
        if (inst->opcode() == HloOpcode::kReduceWindow) {
          first_init = 1;
        }
        IdentityElementType etype =
            GetReductionIdentityElementType(*inst->to_apply());
        for (int64_t i = first_init; i < inst->operand_count(); ++i) {
          if (etype == IdentityElementType::kZero) {
            states_[inst->operand(i)].AddConstraint(
                ConstraintInterval{0.0, 0.0, false});
          } else if (etype == IdentityElementType::kOne) {
            states_[inst->operand(i)].AddConstraint(
                ConstraintInterval{1.0, 1.0, true});
          }
        }
        break;
      }

      case HloOpcode::kSelectAndScatter: {
        IdentityElementType etype =
            GetReductionIdentityElementType(*inst->scatter());
        if (etype == IdentityElementType::kZero) {
          states_[inst->operand(2)].AddConstraint(
              ConstraintInterval{0.0, 0.0, false});
        } else if (etype == IdentityElementType::kOne) {
          states_[inst->operand(2)].AddConstraint(
              ConstraintInterval{1.0, 1.0, true});
        }
        break;
      }

      default:
        break;
    }

    if (inst->opcode() == HloOpcode::kFusion) {
      HloComputation* fused_computation =
          inst->fused_instructions_computation();
      for (int i = 0; i < inst->operand_count(); ++i) {
        ConstraintState source_state =
            states_[fused_computation->parameter_instruction(i)];
        states_[inst->operand(i)].Merge(source_state);
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ConstraintPropagator::PropagateConstraintsExact(
    const HloInstruction* instruction) {
  ConstraintState output_state = states_[instruction];
  ConstraintInterval output_interval = output_state.GetConstraintInterval();
  StructuralConstraints output_structural =
      output_state.GetStructuralConstraints();
  if (output_interval.IsEmpty() &&
      output_structural == StructuralConstraints{}) {
    return absl::OkStatus();
  }
  switch (instruction->opcode()) {
    case HloOpcode::kAbs: {
      states_[instruction->operand(0)].AddConstraint(ConstraintInterval{
          -output_interval.max, output_interval.max,
          output_interval.exclude_zero || output_interval.min > 0});
      StructuralConstraints sc = output_structural;
      sc.no_duplicates = false;
      states_[instruction->operand(0)].MergeStructural(sc);
      break;
    }

    case HloOpcode::kNegate: {
      states_[instruction->operand(0)].AddConstraint(
          ConstraintInterval{-output_interval.max, -output_interval.min,
                             output_interval.exclude_zero});
      states_[instruction->operand(0)].MergeStructural(output_structural);
      break;
    }
    case HloOpcode::kGetTupleElement:
      if (instruction->operand(0)->opcode() == HloOpcode::kTuple ||
          instruction->operand(0)->opcode() == HloOpcode::kSort) {
        states_[instruction->operand(0)->operand(instruction->tuple_index())]
            .Merge(output_state);
      }
      break;
    case HloOpcode::kSort: {
      if (instruction->operand_count() == 1) {
        states_[instruction->operand(0)].Merge(output_state);
      }
      break;
    }
    case HloOpcode::kBitcast:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kDynamicReshape:
    case HloOpcode::kReshape:
    case HloOpcode::kTranspose:
      states_[instruction->operand(0)].Merge(output_state);
      break;
    case HloOpcode::kReverse: {
      states_[instruction->operand(0)].AddConstraint(output_interval);
      StructuralConstraints sc = output_structural;
      sc.needs_sorted_indices = false;
      states_[instruction->operand(0)].MergeStructural(sc);
      break;
    }
    case HloOpcode::kSlice: {
      states_[instruction->operand(0)].AddConstraint(output_interval);
      StructuralConstraints sc = output_structural;
      sc.no_duplicates = false;
      sc.needs_sorted_indices = false;
      states_[instruction->operand(0)].MergeStructural(sc);
      break;
    }
    case HloOpcode::kBroadcast:
    case HloOpcode::kGather:
    case HloOpcode::kScatter:
    case HloOpcode::kTopK:
      states_[instruction->operand(0)].AddConstraint(output_interval);
      break;

    default:
      break;
  }
  return absl::OkStatus();
}

absl::Status ConstraintPropagator::PropagateConstraintsApprox(
    const HloInstruction* instruction) {
  ConstraintInterval output_interval =
      states_[instruction].GetConstraintInterval();
  if (output_interval.IsEmpty() || output_interval.IsUnconstrained()) {
    return absl::OkStatus();
  }
  switch (instruction->opcode()) {
    case HloOpcode::kAdd: {
      if (output_interval.IsNegative()) {
        // Handle constraint Add(x, y) < 0
        // Heuristic: both should be negative
        states_[instruction->operand(0)].AddConstraint(
            ConstraintInterval::Negative());
        states_[instruction->operand(1)].AddConstraint(
            ConstraintInterval::Negative());
      } else if (output_interval.IsPositive()) {
        // Handle constraint Add(x, y) > 0
        // Heuristic: both should be positive
        states_[instruction->operand(0)].AddConstraint(
            ConstraintInterval::Positive());
        states_[instruction->operand(1)].AddConstraint(
            ConstraintInterval::Positive());
      }
      if (output_interval.exclude_zero) {
        // We have Add(y, z) != 0
        // Since both have the same sign, forcing both to be non-zero is
        // sufficient.
        states_[instruction->operand(0)].AddConstraint(
            ConstraintInterval::NonZero());
        states_[instruction->operand(1)].AddConstraint(
            ConstraintInterval::NonZero());
        if (output_interval.CrossesZero()) {
          states_[instruction->operand(0)].AddConstraint(
              ConstraintInterval::Positive());
          states_[instruction->operand(1)].AddConstraint(
              ConstraintInterval::Positive());
        }
      }
      break;
    }

    case HloOpcode::kSubtract: {
      if (output_interval.IsPositive()) {
        // x - y >= 0 => x>=0, y<=0 .
        states_[instruction->operand(0)].AddConstraint(
            ConstraintInterval::Positive());
        states_[instruction->operand(1)].AddConstraint(
            ConstraintInterval::Negative());
      }
      if (output_interval.IsNegative()) {
        // x - y < 0 => x<0, y>0 .
        states_[instruction->operand(0)].AddConstraint(
            ConstraintInterval::Negative());
        states_[instruction->operand(1)].AddConstraint(
            ConstraintInterval::Positive());
      }
      if (output_interval.exclude_zero) {
        states_[instruction->operand(0)].AddConstraint(
            ConstraintInterval::NonZero());
        states_[instruction->operand(1)].AddConstraint(
            ConstraintInterval::NonZero());
      }
      break;
    }

    case HloOpcode::kMultiply: {
      ConstraintInterval x_interval =
          states_[instruction->operand(0)].GetConstraintInterval();
      ConstraintInterval y_interval =
          states_[instruction->operand(1)].GetConstraintInterval();
      if (output_interval.IsPositive()) {
        // Mul(x, y) > 0
        if (x_interval.IsNegative()) {
          states_[instruction->operand(1)].AddConstraint(
              ConstraintInterval::Negative());
        } else if (y_interval.IsNegative()) {
          states_[instruction->operand(0)].AddConstraint(
              ConstraintInterval::Negative());
        } else {
          states_[instruction->operand(0)].AddConstraint(
              ConstraintInterval::Positive());
          states_[instruction->operand(1)].AddConstraint(
              ConstraintInterval::Positive());
        }
      } else if (output_interval.IsNegative()) {
        // Mul(x, y) < 0
        if (x_interval.IsNegative()) {
          states_[instruction->operand(1)].AddConstraint(
              ConstraintInterval::Positive());
        } else if (y_interval.IsNegative()) {
          states_[instruction->operand(0)].AddConstraint(
              ConstraintInterval::Positive());
        } else {
          states_[instruction->operand(0)].AddConstraint(
              ConstraintInterval::Positive());
          states_[instruction->operand(1)].AddConstraint(
              ConstraintInterval::Negative());
        }
      }
      if (output_interval.exclude_zero) {
        states_[instruction->operand(0)].AddConstraint(
            ConstraintInterval::NonZero());
        states_[instruction->operand(1)].AddConstraint(
            ConstraintInterval::NonZero());
      }
      break;
    }

    case HloOpcode::kDivide: {
      ConstraintInterval x_interval =
          states_[instruction->operand(0)].GetConstraintInterval();
      ConstraintInterval y_interval =
          states_[instruction->operand(1)].GetConstraintInterval();
      if (output_interval.IsPositive()) {
        // if x/y >= 0, then bias to positive, x and y > 0.
        states_[instruction->operand(0)].AddConstraint(
            ConstraintInterval::Positive());
        states_[instruction->operand(1)].AddConstraint(
            ConstraintInterval::StrictPositive());
      } else if (output_interval.IsNegative()) {
        // if x/y < 0,
        //  if x < 0, y > 0 OR
        //  if y < 0, x > 0 OR
        //  bias towards x < 0, y > 0.
        if (x_interval.IsNegative()) {
          states_[instruction->operand(1)].AddConstraint(
              ConstraintInterval::StrictPositive());
        } else if (y_interval.IsNegative()) {
          states_[instruction->operand(0)].AddConstraint(
              ConstraintInterval::StrictPositive());
        } else {
          // Heuristic: For no specific reason, bias towards x < 0, y > 0.
          states_[instruction->operand(0)].AddConstraint(
              ConstraintInterval::StrictNegative());
          states_[instruction->operand(1)].AddConstraint(
              ConstraintInterval::StrictPositive());
        }
      }
      if (output_interval.exclude_zero) {
        states_[instruction->operand(0)].AddConstraint(
            ConstraintInterval::NonZero());
      }
      break;
    }

    case HloOpcode::kMaximum: {
      ConstraintInterval input_interval = {ConstraintInterval::kMin,
                                           output_interval.max,
                                           output_interval.exclude_zero};
      states_[instruction->operand(0)].AddConstraint(input_interval);
      states_[instruction->operand(1)].AddConstraint(input_interval);
      break;
    }
    case HloOpcode::kMinimum: {
      ConstraintInterval input_interval = {output_interval.min,
                                           ConstraintInterval::kMax,
                                           output_interval.exclude_zero};
      states_[instruction->operand(0)].AddConstraint(input_interval);
      states_[instruction->operand(1)].AddConstraint(input_interval);
      break;
    }
    default:
      break;
  }
  return absl::OkStatus();
}

absl::Status ConstraintPropagator::PropagateSeedConstraints(
    const HloComputation* computation) {
  auto instructions = computation->MakeInstructionPostOrder();
  for (auto it = instructions.rbegin(); it != instructions.rend(); ++it) {
    const HloInstruction* inst = *it;
    TF_RETURN_IF_ERROR(PropagateConstraintsExact(inst));
  }
  return absl::OkStatus();
}

absl::Status ConstraintPropagator::PropagateConstraints(
    const HloComputation* computation) {
  auto instructions = computation->MakeInstructionPostOrder();
  for (auto it = instructions.rbegin(); it != instructions.rend(); ++it) {
    const HloInstruction* inst = *it;
    TF_RETURN_IF_ERROR(PropagateConstraintsExact(inst));
    TF_RETURN_IF_ERROR(PropagateConstraintsApprox(inst));
  }
  return absl::OkStatus();
}

}  // namespace xla
