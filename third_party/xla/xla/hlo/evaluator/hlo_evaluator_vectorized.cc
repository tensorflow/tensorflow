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
#include <cstddef>
#include <cstdint>
#include <vector>

#include "xla/comparison_util.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/evaluator/stack_literal.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/xla_data.pb.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "third_party/tensorflow/compiler/xla/hlo/evaluator/hlo_evaluator_vectorized.cc" // NOLINT
// clang-format on

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "third_party/highway/hwy/foreach_target.h"  // IWYU pragma: keep
#include "third_party/highway/hwy/highway.h"
#include "xla/hlo/ir/hlo_instruction.h"

HWY_BEFORE_NAMESPACE();
namespace xla {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Evaluates a scheduled computation for a chunk of elements in parallel using
// Highway SIMD. Supports F32, S32, BF16 and PRED types.
absl::StatusOr<std::vector<StackLiteral>> EvaluateScheduleVectorized_Impl(
    absl::Span<const HloInstruction* const> schedule,
    absl::Span<const StackLiteral* const> args, int chunk_size,
    const absl::flat_hash_map<const HloInstruction*, int>& inst_to_index,
    std::vector<StackLiteral>& values) {
  const hn::CappedTag<float, 64> df;
  const hn::CappedTag<int32_t, 64> di;

  for (int i = 0; i < schedule.size(); ++i) {
    const HloInstruction* inst = schedule[i];
    if (inst->opcode() == HloOpcode::kParameter) {
      values[i] = *args[inst->parameter_number()];
    } else if (inst->opcode() == HloOpcode::kCompare) {
      const HloInstruction* lhs = inst->operand(0);
      const HloInstruction* rhs = inst->operand(1);
      int lhs_idx = inst_to_index.at(lhs);
      int rhs_idx = inst_to_index.at(rhs);

      PrimitiveType type = lhs->shape().element_type();
      values[i] = StackLiteral(xla::PRED);

      bool* p_res = values[i].data<bool>();

      const size_t N = hn::Lanes(di);

      auto do_compare = [&](auto get_lhs, auto get_rhs) {
        for (int j = 0; j < chunk_size; j += N) {
          auto v_lhs = get_lhs(j);
          auto v_rhs = get_rhs(j);

          bool handled = false;
          auto store_mask = [&](auto mask) {
            auto int_mask = hn::RebindMask(di, mask);
            auto int_vec =
                hn::IfThenElse(int_mask, hn::Set(di, 1), hn::Zero(di));
            alignas(64) int32_t tmp[64];
            hn::Store(int_vec, di, tmp);
            for (size_t k = 0; k < N; ++k) p_res[j + k] = (tmp[k] != 0);
          };

          switch (inst->comparison_direction()) {
            case ComparisonDirection::kGt:
              store_mask(hn::Gt(v_lhs, v_rhs));
              handled = true;
              break;
            case ComparisonDirection::kLt:
              store_mask(hn::Lt(v_lhs, v_rhs));
              handled = true;
              break;
            case ComparisonDirection::kEq:
              store_mask(hn::Eq(v_lhs, v_rhs));
              handled = true;
              break;
            case ComparisonDirection::kGe:
              store_mask(hn::Ge(v_lhs, v_rhs));
              handled = true;
              break;
            case ComparisonDirection::kLe:
              store_mask(hn::Le(v_lhs, v_rhs));
              handled = true;
              break;
            case ComparisonDirection::kNe:
              store_mask(hn::Ne(v_lhs, v_rhs));
              handled = true;
              break;
            default:
              break;
          }
          if (!handled) {
            for (size_t k = 0; k < N; ++k) p_res[j + k] = false;
          }
        }
      };

      if (type == xla::F32) {
        auto* p_lhs = values[lhs_idx].data<float>();
        auto* p_rhs = values[rhs_idx].data<float>();
        do_compare([&](int j) { return hn::Load(df, p_lhs + j); },
                   [&](int j) { return hn::Load(df, p_rhs + j); });
      } else if (type == xla::BF16) {
        const hn::Rebind<hwy::bfloat16_t, decltype(df)> d_bf16_half;
        auto* p_lhs = values[lhs_idx].data<hwy::bfloat16_t>();
        auto* p_rhs = values[rhs_idx].data<hwy::bfloat16_t>();
        do_compare(
            [&](int j) {
              return hn::PromoteTo(df, hn::Load(d_bf16_half, p_lhs + j));
            },
            [&](int j) {
              return hn::PromoteTo(df, hn::Load(d_bf16_half, p_rhs + j));
            });
      } else if (type == xla::S32) {
        auto* p_lhs = values[lhs_idx].data<int32_t>();
        auto* p_rhs = values[rhs_idx].data<int32_t>();
        do_compare([&](int j) { return hn::Load(di, p_lhs + j); },
                   [&](int j) { return hn::Load(di, p_rhs + j); });
      } else {
        for (int j = 0; j < chunk_size; j += N) {
          for (size_t k = 0; k < N; ++k) p_res[j + k] = false;
        }
      }
    } else if (inst->opcode() == HloOpcode::kSelect) {
      const HloInstruction* pred = inst->operand(0);
      const HloInstruction* on_true = inst->operand(1);
      const HloInstruction* on_false = inst->operand(2);

      int pred_idx = inst_to_index.at(pred);
      int true_idx = inst_to_index.at(on_true);
      int false_idx = inst_to_index.at(on_false);

      PrimitiveType type = on_true->shape().element_type();
      values[i] = StackLiteral(type);

      const size_t N = hn::Lanes(di);

      alignas(64) int32_t tmp[64];
      const bool* p_pred = values[pred_idx].data<bool>();

      auto do_select = [&](auto d_eval, auto prepare_mask, const auto* p_true,
                           const auto* p_false, auto* p_res) {
        for (int j = 0; j < chunk_size; j += N) {
          auto mask = [&]() {
            for (int k = 0; k < N; ++k) {
              tmp[k] = p_pred[j + k] ? 1 : 0;
            }
            auto v_pred = hn::Load(di, tmp);
            return hn::Gt(v_pred, hn::Zero(di));
          }();

          auto d_mask = prepare_mask(mask);
          auto v_true = hn::Load(d_eval, p_true + j);
          auto v_false = hn::Load(d_eval, p_false + j);
          auto v_res = hn::IfThenElse(d_mask, v_true, v_false);
          hn::Store(v_res, d_eval, p_res + j);
        }
      };

      if (type == xla::F32) {
        do_select(
            df, [df](auto m) { return hn::RebindMask(df, m); },
            values[true_idx].data<float>(), values[false_idx].data<float>(),
            values[i].data<float>());
      } else if (type == xla::S32) {
        do_select(
            di, [](auto m) { return m; }, values[true_idx].data<int32_t>(),
            values[false_idx].data<int32_t>(), values[i].data<int32_t>());
      } else if (type == xla::BF16) {
        const hn::Rebind<hwy::bfloat16_t, decltype(df)> d_bf16_half;
        const hn::Rebind<int16_t, decltype(di)> di16_half;
        do_select(
            d_bf16_half,
            [di, di16_half, d_bf16_half](auto m) {
              auto v_mask32 = hn::VecFromMask(di, m);
              auto v_mask16 = hn::DemoteTo(di16_half, v_mask32);
              return hn::RebindMask(d_bf16_half, hn::MaskFromVec(v_mask16));
            },
            values[true_idx].data<hwy::bfloat16_t>(),
            values[false_idx].data<hwy::bfloat16_t>(),
            values[i].data<hwy::bfloat16_t>());
      } else if (type == xla::PRED) {
        const bool* p_true = values[true_idx].data<bool>();
        const bool* p_false = values[false_idx].data<bool>();
        bool* p_res = values[i].data<bool>();
        for (int j = 0; j < chunk_size; j += N) {
          for (int k = 0; k < N; ++k) {
            p_res[j + k] = p_pred[j + k] ? p_true[j + k] : p_false[j + k];
          }
        }
      }
    } else if (inst->opcode() == HloOpcode::kAdd) {
      const HloInstruction* lhs = inst->operand(0);
      const HloInstruction* rhs = inst->operand(1);
      int lhs_idx = inst_to_index.at(lhs);
      int rhs_idx = inst_to_index.at(rhs);

      PrimitiveType type = values[lhs_idx].element_type();
      values[i] = StackLiteral(type);

      const size_t N = hn::Lanes(di);

      auto do_add = [&](auto d_eval, const auto* p_lhs, const auto* p_rhs,
                        auto* p_res) {
        for (int j = 0; j < chunk_size; j += N) {
          auto v_lhs = hn::Load(d_eval, p_lhs + j);
          auto v_rhs = hn::Load(d_eval, p_rhs + j);
          auto v_res = hn::Add(v_lhs, v_rhs);
          hn::Store(v_res, d_eval, p_res + j);
        }
      };

      if (type == xla::F32) {
        do_add(df, values[lhs_idx].data<float>(), values[rhs_idx].data<float>(),
               values[i].data<float>());
      } else if (type == xla::S32) {
        do_add(di, values[lhs_idx].data<int32_t>(),
               values[rhs_idx].data<int32_t>(), values[i].data<int32_t>());
      } else if (type == xla::BF16) {
        const hn::Rebind<hwy::bfloat16_t, decltype(df)> d_bf16_half;
        hwy::bfloat16_t* p_lhs = values[lhs_idx].data<hwy::bfloat16_t>();
        hwy::bfloat16_t* p_rhs = values[rhs_idx].data<hwy::bfloat16_t>();
        hwy::bfloat16_t* p_res = values[i].data<hwy::bfloat16_t>();

        for (int j = 0; j < chunk_size; j += N) {
          auto v_bf16_lhs = hn::Load(d_bf16_half, p_lhs + j);
          auto v_bf16_rhs = hn::Load(d_bf16_half, p_rhs + j);
          auto v_lhs = hn::PromoteTo(df, v_bf16_lhs);
          auto v_rhs = hn::PromoteTo(df, v_bf16_rhs);
          auto v_res = hn::DemoteTo(d_bf16_half, hn::Add(v_lhs, v_rhs));
          hn::Store(v_res, d_bf16_half, p_res + j);
        }
      }
    } else if (inst->opcode() == HloOpcode::kAnd ||
               inst->opcode() == HloOpcode::kOr) {
      const HloInstruction* lhs = inst->operand(0);
      const HloInstruction* rhs = inst->operand(1);
      int lhs_idx = inst_to_index.at(lhs);
      int rhs_idx = inst_to_index.at(rhs);

      PrimitiveType type = values[lhs_idx].element_type();
      values[i] = StackLiteral(type);

      if (type == xla::S32) {
        int32_t* p_lhs = values[lhs_idx].data<int32_t>();
        int32_t* p_rhs = values[rhs_idx].data<int32_t>();
        int32_t* p_res = values[i].data<int32_t>();

        const size_t N = hn::Lanes(di);

        if (inst->opcode() == HloOpcode::kAnd) {
          for (int j = 0; j < chunk_size; j += N) {
            auto v_lhs = hn::Load(di, p_lhs + j);
            auto v_rhs = hn::Load(di, p_rhs + j);
            auto v_res = hn::And(v_lhs, v_rhs);
            hn::Store(v_res, di, p_res + j);
          }
        } else {
          for (int j = 0; j < chunk_size; j += N) {
            auto v_lhs = hn::Load(di, p_lhs + j);
            auto v_rhs = hn::Load(di, p_rhs + j);
            auto v_res = hn::Or(v_lhs, v_rhs);
            hn::Store(v_res, di, p_res + j);
          }
        }
      } else if (type == xla::PRED) {
        const bool* p_lhs = values[lhs_idx].data<bool>();
        const bool* p_rhs = values[rhs_idx].data<bool>();
        bool* p_res = values[i].data<bool>();

        if (inst->opcode() == HloOpcode::kAnd) {
          for (int j = 0; j < chunk_size; ++j) {
            p_res[j] = p_lhs[j] && p_rhs[j];
          }
        } else {
          for (int j = 0; j < chunk_size; ++j) {
            p_res[j] = p_lhs[j] || p_rhs[j];
          }
        }
      }
    }
  }

  const HloInstruction* root = schedule.back();
  if (root->opcode() == HloOpcode::kTuple) {
    std::vector<StackLiteral> result;
    for (const HloInstruction* operand : root->operands()) {
      result.push_back(values[inst_to_index.at(operand)]);
    }
    return result;
  } else {
    return std::vector<StackLiteral>{values.back()};
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace xla
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace xla {

HWY_EXPORT(EvaluateScheduleVectorized_Impl);

absl::StatusOr<std::vector<StackLiteral>>
HloEvaluator::EvaluateScheduleVectorized(
    absl::Span<const HloInstruction* const> schedule,
    absl::Span<const StackLiteral* const> args, int chunk_size,
    const absl::flat_hash_map<const HloInstruction*, int>& inst_to_index,
    std::vector<StackLiteral>& values) {
  return HWY_DYNAMIC_DISPATCH(EvaluateScheduleVectorized_Impl)(
      schedule, args, chunk_size, inst_to_index, values);
}

}  // namespace xla
#endif  // HWY_ONCE
