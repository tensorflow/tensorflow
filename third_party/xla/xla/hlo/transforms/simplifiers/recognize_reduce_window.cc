/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/recognize_reduce_window.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"
#include "xla/window_util.h"

namespace xla {
namespace {

HloComputation* GetOrCreateReducer(HloModule* module, HloOpcode opcode,
                                   PrimitiveType type) {
  HloComputation::Builder b("recognize_rw_reducer");
  auto p0 = b.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(type, {}), "lhs"));
  auto p1 = b.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(type, {}), "rhs"));
  b.AddInstruction(HloInstruction::CreateBinary(ShapeUtil::MakeShape(type, {}),
                                                opcode, p0, p1));
  return module->AddEmbeddedComputation(b.Build());
}

HloInstruction* CreateInitValue(HloComputation* comp, HloOpcode opcode,
                                PrimitiveType type) {
  switch (opcode) {
    case HloOpcode::kAdd:
      return comp->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(type)));
    case HloOpcode::kMultiply:
      return comp->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::One(type)));
    case HloOpcode::kMaximum:
      return comp->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::MinValue(type)));
    case HloOpcode::kMinimum:
      return comp->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::MaxValue(type)));
    case HloOpcode::kAnd:
      return comp->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::One(type)));
    case HloOpcode::kOr:
      return comp->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(type)));
    case HloOpcode::kXor:
      return comp->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(type)));
    default:
      return nullptr;
  }
}

struct WeightedSlice {
  HloInstruction* slice;
  Literal weight;
  std::vector<HloInstruction*> array_weights;

  bool is_scalar() const { return array_weights.empty(); }
};

bool CompareInstruction(const HloInstruction* a, const HloInstruction* b) {
  if (a->opcode() != b->opcode()) {
    return a->opcode() < b->opcode();
  }
  if (a->opcode() == HloOpcode::kSlice && b->opcode() == HloOpcode::kSlice) {
    if (a->operand(0)->unique_id() != b->operand(0)->unique_id()) {
      return CompareInstruction(a->operand(0), b->operand(0));
    }
    if (!absl::c_equal(a->slice_starts(), b->slice_starts())) {
      return absl::c_lexicographical_compare(a->slice_starts(),
                                             b->slice_starts());
    }
    if (!absl::c_equal(a->slice_limits(), b->slice_limits())) {
      return absl::c_lexicographical_compare(a->slice_limits(),
                                             b->slice_limits());
    }
    if (!absl::c_equal(a->slice_strides(), b->slice_strides())) {
      return absl::c_lexicographical_compare(a->slice_strides(),
                                             b->slice_strides());
    }
  }
  if (a->opcode() == HloOpcode::kPad && b->opcode() == HloOpcode::kPad) {
    if (a->operand(0)->unique_id() != b->operand(0)->unique_id()) {
      return CompareInstruction(a->operand(0), b->operand(0));
    }
    if (a->operand(1)->unique_id() != b->operand(1)->unique_id()) {
      return CompareInstruction(a->operand(1), b->operand(1));
    }
    const PaddingConfig& a_pad = a->padding_config();
    const PaddingConfig& b_pad = b->padding_config();
    for (int i = 0; i < a_pad.dimensions_size(); ++i) {
      if (a_pad.dimensions(i).edge_padding_low() !=
          b_pad.dimensions(i).edge_padding_low()) {
        return a_pad.dimensions(i).edge_padding_low() >
               b_pad.dimensions(i).edge_padding_low();
      }
      if (a_pad.dimensions(i).edge_padding_high() !=
          b_pad.dimensions(i).edge_padding_high()) {
        return a_pad.dimensions(i).edge_padding_high() <
               b_pad.dimensions(i).edge_padding_high();
      }
      if (a_pad.dimensions(i).interior_padding() !=
          b_pad.dimensions(i).interior_padding()) {
        return a_pad.dimensions(i).interior_padding() <
               b_pad.dimensions(i).interior_padding();
      }
    }
  }
  if (a->opcode() == HloOpcode::kParameter &&
      b->opcode() == HloOpcode::kParameter) {
    return a->parameter_number() < b->parameter_number();
  }
  return a->unique_id() < b->unique_id();
}

bool CompareWeightedSlices(const WeightedSlice& a, const WeightedSlice& b) {
  if (a.slice->unique_id() != b.slice->unique_id()) {
    return CompareInstruction(a.slice, b.slice);
  }
  if (a.array_weights.size() != b.array_weights.size()) {
    return a.array_weights.size() < b.array_weights.size();
  }
  for (size_t i = 0; i < a.array_weights.size(); ++i) {
    if (a.array_weights[i]->unique_id() != b.array_weights[i]->unique_id()) {
      return CompareInstruction(a.array_weights[i], b.array_weights[i]);
    }
  }
  if (a.weight != b.weight) {
    PrimitiveType type = a.weight.shape().element_type();
    return primitive_util::ArrayTypeSwitch(
        [&](auto primitive_type_constant) -> bool {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          if constexpr (primitive_util::IsComplexType(
                            primitive_type_constant)) {
            auto a_complex = a.weight.template GetFirstElement<NativeT>();
            auto b_complex = b.weight.template GetFirstElement<NativeT>();
            if (a_complex.real() != b_complex.real()) {
              return a_complex.real() < b_complex.real();
            }
            return a_complex.imag() < b_complex.imag();
          } else {
            return a.weight.template GetFirstElement<NativeT>() <
                   b.weight.template GetFirstElement<NativeT>();
          }
        },
        type);
  }
  return false;
}

void ExtractWeightedSlices(
    bool support_additional_factor, HloInstruction* inst,
    std::vector<WeightedSlice>& slices,
    std::optional<Literal> current_weight = std::nullopt,
    const std::vector<HloInstruction*>& array_weights = {}) {
  if (!support_additional_factor) {
    CHECK(array_weights.empty());
  }
  HloEvaluator evaluator;
  Literal cw = LiteralUtil::One(inst->shape().element_type());
  if (current_weight.has_value()) {
    cw = std::move(current_weight.value());
  }

  if (inst->opcode() == HloOpcode::kMultiply) {
    HloInstruction* lhs = inst->mutable_operand(0);
    HloInstruction* rhs = inst->mutable_operand(1);
    std::optional<Literal> val = std::nullopt;
    HloInstruction* other = nullptr;

    auto get_const = [](HloInstruction* op) -> std::optional<Literal> {
      if (op->opcode() == HloOpcode::kBroadcast) {
        op = op->mutable_operand(0);
      }
      if (op->opcode() == HloOpcode::kConstant) {
        if (ShapeUtil::IsScalar(op->shape())) {
          return op->literal().Clone();
        }
        if (op->literal().IsAllFirst()) {
          return LiteralUtil::GetFirstScalarLiteral(op->literal());
        }
      }
      return std::nullopt;
    };

    if (auto l_val = get_const(lhs)) {
      val = std::move(*l_val);
      other = rhs;
    } else if (auto r_val = get_const(rhs)) {
      val = std::move(*r_val);
      other = lhs;
    }

    if (val.has_value() && other != nullptr) {
      Literal next_weight =
          evaluator.EvaluateElementwiseBinaryOp(HloOpcode::kMultiply, cw, *val)
              .value();
      ExtractWeightedSlices(support_additional_factor, other, slices,
                            std::move(next_weight), array_weights);
      return;
    } else if (support_additional_factor) {
      auto is_traversable = [](HloInstruction* op) {
        return op->opcode() == HloOpcode::kAdd ||
               op->opcode() == HloOpcode::kSubtract ||
               op->opcode() == HloOpcode::kMultiply;
      };
      if (!is_traversable(lhs) && is_traversable(rhs)) {
        std::swap(lhs, rhs);
      }
      auto new_weights = array_weights;
      new_weights.push_back(rhs);
      ExtractWeightedSlices(support_additional_factor, lhs, slices,
                            std::move(cw), new_weights);
      return;
    }
  }

  if (inst->opcode() == HloOpcode::kAdd) {
    ExtractWeightedSlices(support_additional_factor, inst->mutable_operand(0),
                          slices, cw.Clone(), array_weights);
    ExtractWeightedSlices(support_additional_factor, inst->mutable_operand(1),
                          slices, std::move(cw), array_weights);
    return;
  }

  if (inst->opcode() == HloOpcode::kSubtract) {
    Literal neg_weight =
        evaluator.EvaluateElementwiseUnaryOp(HloOpcode::kNegate, cw).value();
    ExtractWeightedSlices(support_additional_factor, inst->mutable_operand(0),
                          slices, cw.Clone(), array_weights);
    ExtractWeightedSlices(support_additional_factor, inst->mutable_operand(1),
                          slices, std::move(neg_weight), array_weights);
    return;
  }

  if (inst->opcode() == HloOpcode::kDot) {
    HloInstruction* lhs = inst->mutable_operand(0);
    HloInstruction* rhs = inst->mutable_operand(1);

    if (lhs->opcode() == HloOpcode::kConcatenate &&
        rhs->opcode() == HloOpcode::kConstant &&
        inst->dot_dimension_numbers().lhs_contracting_dimensions_size() == 1 &&
        inst->dot_dimension_numbers().lhs_contracting_dimensions(0) == 0 &&
        inst->dot_dimension_numbers().rhs_contracting_dimensions_size() == 1 &&
        inst->dot_dimension_numbers().rhs_contracting_dimensions(0) == 0 &&
        lhs->shape().dimensions_size() > 0 &&
        rhs->shape().dimensions_size() == 1 &&
        lhs->shape().dimensions(0) == rhs->shape().dimensions(0)) {
      int64_t num_slices = lhs->operand_count();
      bool all_good = true;
      std::vector<WeightedSlice> temp_slices;
      for (int64_t i = 0; i < num_slices; ++i) {
        HloInstruction* reshape = lhs->mutable_operand(i);
        if (reshape->opcode() != HloOpcode::kReshape) {
          all_good = false;
          break;
        }
        std::vector<int64_t> starts = {i};
        std::vector<int64_t> limits = {i + 1};
        Literal scalar_w = LiteralUtil::GetFirstScalarLiteral(
            rhs->literal().Slice(starts, limits));
        Literal next_w =
            evaluator
                .EvaluateElementwiseBinaryOp(HloOpcode::kMultiply, cw, scalar_w)
                .value();
        temp_slices.push_back(
            {reshape->mutable_operand(0), std::move(next_w), array_weights});
      }
      if (all_good) {
        slices.insert(slices.end(),
                      std::make_move_iterator(temp_slices.begin()),
                      std::make_move_iterator(temp_slices.end()));
        return;
      }
    }
  }

  // Universal Element-wise extraction fallback
  slices.push_back({inst, std::move(cw), array_weights});
  return;
}

absl::StatusOr<HloInstruction*> CreateReduceWindow(
    HloComputation* computation, Shape shape, HloInstruction* new_base_op,
    HloOpcode opcode, int64_t dim, int64_t current_window_size,
    int64_t window_dilation) {
  HloComputation* reducer =
      GetOrCreateReducer(computation->parent(), opcode, shape.element_type());
  HloInstruction* init_val =
      CreateInitValue(computation, opcode, shape.element_type());
  if (!init_val || !reducer) {
    return absl::StatusOr<HloInstruction*>{};
  }

  int rank = new_base_op->shape().dimensions_size();
  Window window = window_util::MakeWindow(std::vector<int64_t>(rank, 1));
  window.mutable_dimensions(dim)->set_size(current_window_size);
  window.mutable_dimensions(dim)->set_window_dilation(window_dilation);

  HloInstruction* rw =
      computation->AddInstruction(HloInstruction::CreateReduceWindow(
          shape, new_base_op, init_val, window, reducer));
  return rw;
}

absl::StatusOr<bool> RunOnComputation(HloComputation* computation) {
  bool changed = false;
  int32_t optimization_level =
      computation->parent()
          ->config()
          .debug_options()
          .xla_recognize_reduction_optimization_level();
  if (optimization_level == 0) {
    return false;
  }
  bool support_dot = optimization_level >= 2;
  bool support_additional_factor = optimization_level >= 3;

  std::vector<HloInstruction*> all_insts =
      computation->MakeInstructionPostOrder();
  for (HloInstruction* inst : all_insts) {
    if (inst->opcode() == HloOpcode::kDynamicSlice) {
      bool all_static = true;
      for (int i = 1; i < inst->operand_count(); ++i) {
        if (inst->operand(i)->opcode() != HloOpcode::kConstant) {
          all_static = false;
          break;
        }
      }
      if (all_static) {
        std::vector<int64_t> starts;
        std::vector<int64_t> limits;
        std::vector<int64_t> strides(inst->shape().dimensions_size(), 1);
        for (int i = 1; i < inst->operand_count(); ++i) {
          int64_t dim = i - 1;
          int64_t start = *inst->operand(i)->literal().GetFirstInteger();
          int64_t size = inst->dynamic_slice_sizes()[dim];
          int64_t dim_size = inst->operand(0)->shape().dimensions(dim);
          start =
              std::max<int64_t>(0, std::min<int64_t>(start, dim_size - size));
          starts.push_back(start);
          limits.push_back(start + size);
        }
        HloInstruction* static_slice = computation->AddInstruction(
            HloInstruction::CreateSlice(inst->shape(), inst->mutable_operand(0),
                                        starts, limits, strides));
        TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(static_slice));
        changed = true;
      }
    } else if (inst->opcode() == HloOpcode::kGetTupleElement) {
      if (inst->operand(0)->opcode() == HloOpcode::kTuple) {
        TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(
            inst->mutable_operand(0)->mutable_operand(inst->tuple_index())));
        changed = true;
      }
    }
  }

  std::vector<HloInstruction*> instructions =
      computation->MakeInstructionPostOrder();

  for (auto it = instructions.rbegin(); it != instructions.rend(); ++it) {
    HloInstruction* inst = *it;
    if (inst->IsDead()) {
      continue;
    }
    if (inst->operand_count() != 2) {
      continue;
    }
    if (!HloOpcodeIsBinaryCommutative(inst->opcode()) &&
        inst->opcode() != HloOpcode::kSubtract) {
      continue;
    }
    // Don't do anything for scalars.
    if (inst->shape().dimensions_size() == 0) {
      continue;
    }

    HloInstruction* lhs = inst->mutable_operand(0);
    HloInstruction* rhs = inst->mutable_operand(1);

    int64_t dim = -1;
    int64_t window_dilation = -1;
    HloInstruction* new_base_op = nullptr;
    int64_t current_window_size = 2;

    // Pattern 3: add(reduce_window, slice)
    if (!new_base_op && (lhs->opcode() == HloOpcode::kReduceWindow ||
                         rhs->opcode() == HloOpcode::kReduceWindow)) {
      HloInstruction* rw =
          (lhs->opcode() == HloOpcode::kReduceWindow) ? lhs : rhs;
      HloInstruction* other =
          (lhs->opcode() == HloOpcode::kReduceWindow) ? rhs : lhs;

      bool other_valid = other->opcode() == HloOpcode::kSlice ||
                         (other->opcode() == HloOpcode::kPad &&
                          other->operand(1)->opcode() == HloOpcode::kConstant);
      bool rw_valid =
          rw->operand(0)->opcode() == HloOpcode::kSlice ||
          (rw->operand(0)->opcode() == HloOpcode::kPad &&
           rw->operand(0)->operand(1)->opcode() == HloOpcode::kConstant);

      // Make sure the reducer is the same!
      if (rw->to_apply()->root_instruction()->opcode() == inst->opcode() &&
          other_valid && rw_valid) {
        HloInstruction* rw_slice = rw->mutable_operand(0);
        if (rw_slice->opcode() == other->opcode() &&
            rw_slice->operand(0) == other->operand(0)) {
          bool valid_strides = true;
          if (rw_slice->opcode() == HloOpcode::kSlice) {
            valid_strides = absl::c_all_of(rw_slice->slice_strides(),
                                           [](int64_t s) { return s == 1; }) &&
                            absl::c_all_of(other->slice_strides(),
                                           [](int64_t s) { return s == 1; });
          } else {
            valid_strides = rw_slice->operand(1)->Identical(*other->operand(1));
          }
          if (valid_strides) {
            int rank = rw_slice->shape().dimensions_size();
            bool matched = true;
            std::vector<int64_t> new_starts(rank);
            std::vector<int64_t> new_limits(rank);

            auto get_start = [&](HloInstruction* op, int d) -> int64_t {
              if (op->opcode() == HloOpcode::kSlice) {
                return op->slice_starts(d);
              }
              return -op->padding_config().dimensions(d).edge_padding_low();
            };
            auto get_limit = [&](HloInstruction* op, int d) -> int64_t {
              if (op->opcode() == HloOpcode::kSlice) {
                return op->slice_limits(d);
              }
              return op->operand(0)->shape().dimensions(d) +
                     op->padding_config().dimensions(d).edge_padding_high();
            };
            auto get_interior = [&](HloInstruction* op, int d) -> int64_t {
              if (op->opcode() == HloOpcode::kSlice) {
                return 0;
              }
              return op->padding_config().dimensions(d).interior_padding();
            };

            for (int i = 0; i < rank; ++i) {
              if (get_interior(rw_slice, i) != 0 ||
                  get_interior(other, i) != 0) {
                matched = false;
                break;
              }
              if (get_start(rw_slice, i) == get_start(other, i) &&
                  get_limit(rw_slice, i) == get_limit(other, i)) {
                new_starts[i] = get_start(rw_slice, i);
                new_limits[i] = get_limit(rw_slice, i);
              } else if (dim != -1) {
                matched = false;
                break;
              } else {
                int64_t rw_start = get_start(rw_slice, i);
                int64_t o_start = get_start(other, i);
                int64_t rw_limit = get_limit(rw_slice, i);
                int64_t o_limit = get_limit(other, i);
                dim = i;

                int64_t D = rw->window().dimensions(dim).window_dilation();
                int64_t size = rw->window().dimensions(dim).size();

                bool is_append = (o_start == rw_start + D * size) &&
                                 (o_limit == rw_limit + D);
                bool is_prepend = (rw_start == o_start + D) &&
                                  (rw_limit == o_limit + D * size);

                if (is_append || is_prepend) {
                  window_dilation = D;
                } else {
                  matched = false;
                  break;
                }

                new_starts[i] = std::min(rw_start, o_start);
                new_limits[i] = std::max(rw_limit, o_limit);
                current_window_size = size + 1;
              }
            }
            if (matched && dim != -1) {
              if (rw_slice->opcode() == HloOpcode::kSlice) {
                std::vector<int64_t> strides(rank, 1);
                Shape slice_shape = rw_slice->shape();
                slice_shape.set_dimensions(dim,
                                           new_limits[dim] - new_starts[dim]);
                new_base_op =
                    computation->AddInstruction(HloInstruction::CreateSlice(
                        slice_shape, rw_slice->mutable_operand(0), new_starts,
                        new_limits, strides));
              } else {
                PaddingConfig new_pad_config = rw_slice->padding_config();
                for (int d = 0; d < rank; ++d) {
                  new_pad_config.mutable_dimensions(d)->set_edge_padding_low(
                      -new_starts[d]);
                  new_pad_config.mutable_dimensions(d)->set_edge_padding_high(
                      new_limits[d] -
                      rw_slice->operand(0)->shape().dimensions(d));
                  new_pad_config.mutable_dimensions(d)->set_interior_padding(0);
                }
                Shape pad_shape = rw_slice->shape();
                pad_shape.set_dimensions(dim,
                                         new_limits[dim] - new_starts[dim]);
                new_base_op =
                    computation->AddInstruction(HloInstruction::CreatePad(
                        pad_shape, rw_slice->mutable_operand(0),
                        rw_slice->mutable_operand(1), new_pad_config));
              }
            }
          }
        }
      }
    }

    if (new_base_op) {
      auto opcode = (lhs->opcode() == HloOpcode::kReduceWindow)
                        ? lhs->to_apply()->root_instruction()->opcode()
                        : inst->opcode();

      TF_ASSIGN_OR_RETURN(
          auto rw,
          CreateReduceWindow(computation, inst->shape(), new_base_op, opcode,
                             dim, current_window_size, window_dilation));

      TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(rw));
      changed = true;
      continue;
    }

    if (inst->opcode() == HloOpcode::kSubtract ||
        inst->opcode() == HloOpcode::kAdd) {
      std::vector<WeightedSlice> combined_slices;

      ExtractWeightedSlices(support_additional_factor, inst, combined_slices);
      {
        Shape slice_shape = inst->shape();
        int64_t rank = slice_shape.dimensions_size();

        bool any_array_weights = false;
        bool all_unit_stride_slices_on_same_operand = true;
        bool all_pads_on_same_operand = true;
        for (auto& ws : combined_slices) {
          if (!ws.array_weights.empty()) {
            any_array_weights = true;
          }
          CHECK_EQ(slice_shape, ws.slice->shape());
          // TODO here down
          if (ws.slice->opcode() != HloOpcode::kSlice ||
              absl::c_any_of(ws.slice->slice_strides(),
                             [](int64_t s) { return s != 1; })) {
            all_unit_stride_slices_on_same_operand = false;
          } else if (all_unit_stride_slices_on_same_operand &&
                     ws.slice->operand(0) !=
                         combined_slices[0].slice->operand(0)) {
            all_unit_stride_slices_on_same_operand = false;
          }

          if (ws.slice->opcode() != HloOpcode::kPad ||
              ws.slice->operand(1)->opcode() != HloOpcode::kConstant) {
            all_pads_on_same_operand = false;
          } else if (all_pads_on_same_operand &&
                     (ws.slice->operand(0) !=
                          combined_slices[0].slice->operand(0) ||
                      !ws.slice->operand(1)->Identical(
                          *combined_slices[0].slice->operand(1)))) {
            all_pads_on_same_operand = false;
          }

          if (!ws.array_weights.empty()) {
            std::vector<HloInstruction*> all_terms;
            all_terms.push_back(ws.slice);
            for (auto* aw : ws.array_weights) {
              CHECK_EQ(slice_shape, aw->shape());
              all_terms.push_back(aw);
            }
            absl::c_sort(all_terms, CompareInstruction);
            ws.slice = all_terms.back();
            ws.array_weights.assign(all_terms.begin(), all_terms.end() - 1);
          }
        }

        // Sort to group identical (slice, array_weight) pairs
        absl::c_sort(combined_slices, CompareWeightedSlices);

        std::vector<WeightedSlice> folded;
        for (auto& ws : combined_slices) {
          if (folded.empty()) {
            folded.push_back(std::move(ws));
          } else {
            if (ws.slice == folded.back().slice &&
                ws.array_weights == folded.back().array_weights) {
              HloEvaluator evaluator;
              folded.back().weight =
                  evaluator
                      .EvaluateElementwiseBinaryOp(
                          HloOpcode::kAdd, folded.back().weight, ws.weight)
                      .value();
            } else {
              folded.push_back(std::move(ws));
            }
          }
        }

        PrimitiveType ptype = inst->shape().element_type();
        Literal zero = LiteralUtil::Zero(ptype);
        Literal one = LiteralUtil::One(ptype);
        Literal minus_one =
            HloEvaluator{}
                .EvaluateElementwiseUnaryOp(HloOpcode::kNegate, one)
                .value();

        folded.erase(std::remove_if(folded.begin(), folded.end(),
                                    [&](const WeightedSlice& ws) {
                                      return ws.weight == zero;
                                    }),
                     folded.end());

        if (folded.empty()) {
          auto zero_replacement = computation->AddInstruction(
              HloInstruction::CreateConstant(LiteralUtil::Zero(ptype)));
          auto broadcasted_zero =
              computation->AddInstruction(HloInstruction::CreateBroadcast(
                  inst->shape(), zero_replacement, {}));
          TF_RETURN_IF_ERROR(
              computation->ReplaceInstruction(inst, broadcasted_zero));
          changed = true;
          continue;
        }

        if (folded.size() == 1) {
          if (folded[0].weight == one && !any_array_weights) {
            TF_RETURN_IF_ERROR(
                computation->ReplaceInstruction(inst, folded[0].slice));
            changed = true;
            continue;
          }
          if (!folded[0].array_weights.empty()) {
            HloInstruction* replacement = folded[0].slice;
            for (auto* aw : folded[0].array_weights) {
              replacement =
                  computation->AddInstruction(HloInstruction::CreateBinary(
                      inst->shape(), HloOpcode::kMultiply, replacement, aw));
            }
            if (folded[0].weight != one) {
              auto literal_replacement = computation->AddInstruction(
                  HloInstruction::CreateConstant(std::move(folded[0].weight)));
              auto broadcasted =
                  computation->AddInstruction(HloInstruction::CreateBroadcast(
                      inst->shape(), literal_replacement, {}));
              replacement =
                  computation->AddInstruction(HloInstruction::CreateBinary(
                      inst->shape(), HloOpcode::kMultiply, replacement,
                      broadcasted));
            }
            TF_RETURN_IF_ERROR(
                computation->ReplaceInstruction(inst, replacement));
            changed = true;
            continue;
          }
          auto literal_replacement = computation->AddInstruction(
              HloInstruction::CreateConstant(std::move(folded[0].weight)));
          auto broadcasted =
              computation->AddInstruction(HloInstruction::CreateBroadcast(
                  inst->shape(), literal_replacement, {}));
          HloInstruction* replacement = computation->AddInstruction(
              HloInstruction::CreateBinary(inst->shape(), HloOpcode::kMultiply,
                                           folded[0].slice, broadcasted));
          TF_RETURN_IF_ERROR(
              computation->ReplaceInstruction(inst, replacement));
          changed = true;
          continue;
        }

        // Pattern 2: add(slice, slice, ..., slice)
        bool all_weights_one = true;
        for (const auto& ws : folded) {
          if (ws.weight != one) {
            all_weights_one = false;
            break;
          }
        }

        if ((all_unit_stride_slices_on_same_operand ||
             all_pads_on_same_operand) &&
            !any_array_weights && all_weights_one && folded.size() >= 2) {
          int rank = folded[0].slice->shape().dimensions_size();
          bool matched = true;
          std::vector<int64_t> new_starts(rank);
          std::vector<int64_t> new_limits(rank);

          int64_t dim = -1;
          int64_t window_dilation = -1;
          int64_t expected_size = -1;

          auto get_start = [&](HloInstruction* op, int d) -> int64_t {
            if (op->opcode() == HloOpcode::kSlice) return op->slice_starts(d);
            return -op->padding_config().dimensions(d).edge_padding_low();
          };
          auto get_limit = [&](HloInstruction* op, int d) -> int64_t {
            if (op->opcode() == HloOpcode::kSlice) return op->slice_limits(d);
            return op->operand(0)->shape().dimensions(d) +
                   op->padding_config().dimensions(d).edge_padding_high();
          };
          auto get_interior = [&](HloInstruction* op, int d) -> int64_t {
            if (op->opcode() == HloOpcode::kSlice) return 0;
            return op->padding_config().dimensions(d).interior_padding();
          };

          for (int i = 0; i < rank; ++i) {
            bool all_same = true;
            for (size_t j = 1; j < folded.size(); ++j) {
              if (get_start(folded[j].slice, i) !=
                      get_start(folded[0].slice, i) ||
                  get_limit(folded[j].slice, i) !=
                      get_limit(folded[0].slice, i) ||
                  get_interior(folded[j].slice, i) !=
                      get_interior(folded[0].slice, i) ||
                  get_interior(folded[j].slice, i) != 0) {
                all_same = false;
                break;
              }
            }
            if (get_interior(folded[0].slice, i) != 0) {
              matched = false;
              break;
            }
            if (all_same) {
              new_starts[i] = get_start(folded[0].slice, i);
              new_limits[i] = get_limit(folded[0].slice, i);
            } else if (dim != -1) {
              matched = false;
              break;
            } else {
              int64_t first_start = get_start(folded[0].slice, i);
              int64_t first_limit = get_limit(folded[0].slice, i);
              expected_size = first_limit - first_start;
              dim = i;

              window_dilation =
                  get_start(folded[1].slice, i) - get_start(folded[0].slice, i);
              if (window_dilation <= 0) {
                matched = false;
                break;
              }

              for (size_t j = 0; j < folded.size(); ++j) {
                int64_t s = get_start(folded[j].slice, i);
                int64_t l = get_limit(folded[j].slice, i);
                if (l - s != expected_size ||
                    s != first_start + j * window_dilation) {
                  matched = false;
                  break;
                }
              }

              if (matched) {
                new_starts[i] = first_start;
                new_limits[i] = get_limit(folded.back().slice, i);
              }
            }
          }

          if (matched && dim != -1) {
            HloInstruction* new_base_op = nullptr;
            if (all_unit_stride_slices_on_same_operand) {
              std::vector<int64_t> strides(rank, 1);
              Shape slice_shape = folded[0].slice->shape();
              slice_shape.set_dimensions(dim,
                                         new_limits[dim] - new_starts[dim]);
              new_base_op =
                  computation->AddInstruction(HloInstruction::CreateSlice(
                      slice_shape, folded[0].slice->mutable_operand(0),
                      new_starts, new_limits, strides));
            } else {
              PaddingConfig new_pad_config = folded[0].slice->padding_config();
              for (int d = 0; d < rank; ++d) {
                new_pad_config.mutable_dimensions(d)->set_edge_padding_low(
                    -new_starts[d]);
                new_pad_config.mutable_dimensions(d)->set_edge_padding_high(
                    new_limits[d] -
                    folded[0].slice->operand(0)->shape().dimensions(d));
                new_pad_config.mutable_dimensions(d)->set_interior_padding(0);
              }
              Shape pad_shape = folded[0].slice->shape();
              pad_shape.set_dimensions(
                  dim, folded[0].slice->shape().dimensions(dim) +
                           (folded.size() - 1) * window_dilation);
              new_base_op =
                  computation->AddInstruction(HloInstruction::CreatePad(
                      pad_shape, folded[0].slice->mutable_operand(0),
                      folded[0].slice->mutable_operand(1), new_pad_config));
            }
            int64_t current_window_size = folded.size();
            TF_ASSIGN_OR_RETURN(
                HloInstruction * replacement,
                CreateReduceWindow(computation, inst->shape(), new_base_op,
                                   HloOpcode::kAdd, dim, current_window_size,
                                   window_dilation));
            TF_RETURN_IF_ERROR(
                computation->ReplaceInstruction(inst, replacement));
            changed = true;
            continue;
          }
        }

        if (folded.size() == 2 && !any_array_weights &&
            !(support_dot && (all_unit_stride_slices_on_same_operand ||
                              all_pads_on_same_operand))) {
          if (folded[0].weight == one && folded[1].weight == one) {
            if (folded[0].slice == inst->operand(0) &&
                folded[1].slice == inst->operand(1)) {
              if (inst->opcode() == HloOpcode::kAdd) {
                continue;
              }
            }
            if (folded[0].slice == inst->operand(1) &&
                folded[1].slice == inst->operand(0)) {
              if (inst->opcode() == HloOpcode::kAdd) {
                continue;
              }
            }

            HloInstruction* replacement = computation->AddInstruction(
                HloInstruction::CreateBinary(inst->shape(), HloOpcode::kAdd,
                                             folded[0].slice, folded[1].slice));
            TF_RETURN_IF_ERROR(
                computation->ReplaceInstruction(inst, replacement));
            changed = true;
            continue;
          }
          if (folded[0].weight == one && folded[1].weight == minus_one) {
            if (folded[0].slice == inst->operand(0) &&
                folded[1].slice == inst->operand(1)) {
              if (inst->opcode() == HloOpcode::kSubtract) {
                continue;
              }
            }
            HloInstruction* replacement =
                computation->AddInstruction(HloInstruction::CreateBinary(
                    inst->shape(), HloOpcode::kSubtract, folded[0].slice,
                    folded[1].slice));
            TF_RETURN_IF_ERROR(
                computation->ReplaceInstruction(inst, replacement));
            changed = true;
            continue;
          }

          if (folded[0].weight == minus_one && folded[1].weight == one) {
            if (folded[0].slice == inst->operand(1) &&
                folded[1].slice == inst->operand(0)) {
              if (inst->opcode() == HloOpcode::kSubtract) {
                continue;
              }
            }
            HloInstruction* replacement =
                computation->AddInstruction(HloInstruction::CreateBinary(
                    inst->shape(), HloOpcode::kSubtract, folded[1].slice,
                    folded[0].slice));
            TF_RETURN_IF_ERROR(
                computation->ReplaceInstruction(inst, replacement));
            changed = true;
            continue;
          }
        }

        int64_t num_with_real_data = 0;
        for (const auto& ws : folded) {
          if (ws.slice->opcode() == HloOpcode::kBroadcast &&
              ws.slice->operand(0)->shape().dimensions_size() == 0) {
            continue;
          }
          if (ws.slice->opcode() == HloOpcode::kIota) {
            continue;
          }
          num_with_real_data++;
        }

        if (num_with_real_data >= 2 && support_dot) {
          Shape reshape_shape = slice_shape;
          std::vector<int64_t> concat_dims = {1};
          for (int64_t d : reshape_shape.dimensions()) concat_dims.push_back(d);
          Shape new_shape =
              ShapeUtil::MakeShape(reshape_shape.element_type(), concat_dims);

          std::vector<HloInstruction*> reshaped_slices;
          std::vector<HloInstruction*> reshaped_array_weights;

          for (auto& ws : folded) {
            reshaped_slices.push_back(computation->AddInstruction(
                HloInstruction::CreateReshape(new_shape, ws.slice)));
            if (any_array_weights) {
              HloInstruction* aw = nullptr;
              for (auto* inner_aw : ws.array_weights) {
                if (aw == nullptr) {
                  aw = inner_aw;
                } else {
                  aw = computation->AddInstruction(HloInstruction::CreateBinary(
                      aw->shape(), HloOpcode::kMultiply, aw, inner_aw));
                }
              }
              if (aw) {
                aw = computation->AddInstruction(
                    HloInstruction::CreateReshape(new_shape, aw));
              }
              if (!aw) {
                HloInstruction* scalar = computation->AddInstruction(
                    HloInstruction::CreateConstant(ws.weight.Clone()));
                aw = computation->AddInstruction(
                    HloInstruction::CreateBroadcast(new_shape, scalar, {}));
              } else if (ws.weight != one) {
                HloInstruction* scalar = computation->AddInstruction(
                    HloInstruction::CreateConstant(ws.weight.Clone()));
                HloInstruction* weight = computation->AddInstruction(
                    HloInstruction::CreateBroadcast(aw->shape(), scalar, {}));
                aw = computation->AddInstruction(HloInstruction::CreateBinary(
                    aw->shape(), HloOpcode::kMultiply, aw, weight));
              }
              reshaped_array_weights.push_back(aw);
            }
          }

          Shape concat_shape = new_shape;
          concat_shape.set_dimensions(0, folded.size());
          HloInstruction* concat =
              computation->AddInstruction(HloInstruction::CreateConcatenate(
                  concat_shape, reshaped_slices, 0));

          HloInstruction* replacement = nullptr;

          if (any_array_weights) {
            HloInstruction* concat_weights =
                computation->AddInstruction(HloInstruction::CreateConcatenate(
                    concat_shape, reshaped_array_weights, 0));

            DotDimensionNumbers dnums;
            dnums.add_lhs_contracting_dimensions(0);
            dnums.add_rhs_contracting_dimensions(0);
            for (int i = 1; i <= rank; ++i) {
              dnums.add_lhs_batch_dimensions(i);
              dnums.add_rhs_batch_dimensions(i);
            }
            PrecisionConfig precision_config;
            precision_config.add_operand_precision(PrecisionConfig::DEFAULT);
            precision_config.add_operand_precision(PrecisionConfig::DEFAULT);

            replacement = computation->AddInstruction(
                HloInstruction::CreateDot(inst->shape(), concat, concat_weights,
                                          dnums, precision_config));
            TF_RETURN_IF_ERROR(
                computation->ReplaceInstruction(inst, replacement));
            changed = true;
            continue;
          } else {
            Literal weights_literal = primitive_util::PrimitiveTypeSwitch<
                Literal>(
                [&](auto primitive_type_constant) -> Literal {
                  if constexpr (primitive_util::IsArrayType(
                                    primitive_type_constant)) {
                    if constexpr (primitive_type_constant == PRED) {
                      std::unique_ptr<bool[]> bool_vals(
                          new bool[folded.size()]);
                      for (size_t i = 0; i < folded.size(); ++i) {
                        bool_vals[i] = folded[i].weight.GetFirstElement<bool>();
                      }
                      return LiteralUtil::CreateR1<bool>(absl::Span<const bool>(
                          bool_vals.get(), folded.size()));
                    } else {
                      using NativeT =
                          primitive_util::NativeTypeOf<primitive_type_constant>;
                      std::vector<NativeT> vals;
                      vals.reserve(folded.size());
                      for (auto& ws : folded) {
                        vals.push_back(ws.weight.GetFirstElement<NativeT>());
                      }
                      return LiteralUtil::CreateR1<NativeT>(vals);
                    }
                  }
                  LOG(FATAL) << "Unsupported array type";
                },
                ptype);

            HloInstruction* weights = computation->AddInstruction(
                HloInstruction::CreateConstant(std::move(weights_literal)));
            DotDimensionNumbers dnums;
            dnums.add_lhs_contracting_dimensions(0);
            dnums.add_rhs_contracting_dimensions(0);
            PrecisionConfig precision_config;
            precision_config.add_operand_precision(PrecisionConfig::DEFAULT);
            precision_config.add_operand_precision(PrecisionConfig::DEFAULT);

            replacement = computation->AddInstruction(HloInstruction::CreateDot(
                inst->shape(), concat, weights, dnums, precision_config));
            TF_RETURN_IF_ERROR(
                computation->ReplaceInstruction(inst, replacement));
            changed = true;
            continue;
          }
        }
      }
    }
  }

  return changed;
}

}  // namespace

absl::StatusOr<bool> RecognizeReduceWindow::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  std::vector<HloComputation*> computations =
      module->MakeNonfusionComputations(execution_threads);
  for (HloComputation* computation : computations) {
    TF_ASSIGN_OR_RETURN(bool computation_changed,
                        RunOnComputation(computation));
    changed |= computation_changed;

    // Post-order traversal for DCE
    if (computation_changed) {
      bool dce_changed = true;
      while (dce_changed) {
        dce_changed = false;
        std::vector<HloInstruction*> dce_instructions =
            computation->MakeInstructionPostOrder();
        for (auto it = dce_instructions.rbegin(); it != dce_instructions.rend();
             ++it) {
          HloInstruction* dce_inst = *it;
          if (dce_inst->user_count() == 0 && !dce_inst->HasSideEffect() &&
              computation->root_instruction() != dce_inst &&
              computation->IsSafelyRemovable(dce_inst)) {
            TF_RETURN_IF_ERROR(computation->RemoveInstruction(dce_inst));
            dce_changed = true;
          }
        }
      }
    }
  }
  return changed;
}

}  // namespace xla
