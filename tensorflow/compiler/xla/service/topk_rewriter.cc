/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/topk_rewriter.h"

#include "absl/algorithm/container.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

static bool IsNanSafeGt(HloComputation* comp) {
  namespace m = match;
  auto match_bitcast_f32 = [](int64 parameter_number) {
    auto param = m::Parameter(parameter_number)
                     .WithShape(m::Shape().WithElementType(F32));
    auto param_s32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(S32));
    auto param_u32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(U32));
    return m::Select(
        m::Lt(param_s32, m::ConstantScalar(0)),
        m::BitcastConvert(
            m::Subtract(m::ConstantScalar(std::numeric_limits<int32>::max()),
                        param_u32))
            .WithShape(m::Shape().WithElementType(S32)),
        param_s32);
  };

  auto match_bitcast_f32_with_convert = [](int64 parameter_number) {
    auto param = m::Parameter(parameter_number)
                     .WithShape(m::Shape().WithElementType(F32));
    auto param_s32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(S32));
    auto param_u32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(U32));
    auto max_u32 =
        m::Convert(m::ConstantScalar(std::numeric_limits<int32>::max()))
            .WithShape(m::Shape().WithElementType(U32));
    return m::Select(m::Lt(param_s32, m::ConstantScalar(0)),
                     m::BitcastConvert(m::Subtract(max_u32, param_u32))
                         .WithShape(m::Shape().WithElementType(S32)),
                     param_s32);
  };

  auto match_bitcast_bf16 = [](int64 parameter_number) {
    auto param = m::Convert(m::Parameter(parameter_number)
                                .WithShape(m::Shape().WithElementType(BF16)))
                     .WithShape(m::Shape().WithElementType(F32));
    auto param_s32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(S32));
    auto param_u32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(U32));
    return m::Select(
        m::Lt(param_s32, m::ConstantScalar(0)),
        m::BitcastConvert(
            m::Subtract(m::ConstantScalar(std::numeric_limits<int32>::max()),
                        param_u32))
            .WithShape(m::Shape().WithElementType(S32)),
        param_s32);
  };

  auto match_bitcast_bf16_with_convert = [](int64 parameter_number) {
    auto param = m::Convert(m::Parameter(parameter_number)
                                .WithShape(m::Shape().WithElementType(BF16)))
                     .WithShape(m::Shape().WithElementType(F32));
    auto param_s32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(S32));
    auto param_u32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(U32));
    auto max_u32 =
        m::Convert(m::ConstantScalar(std::numeric_limits<int32>::max()))
            .WithShape(m::Shape().WithElementType(U32));
    return m::Select(m::Lt(param_s32, m::ConstantScalar(0)),
                     m::BitcastConvert(m::Subtract(max_u32, param_u32))
                         .WithShape(m::Shape().WithElementType(S32)),
                     param_s32);
  };

  return Match(comp->root_instruction(),
               m::Gt(match_bitcast_f32(0), match_bitcast_f32(1))) ||
         Match(comp->root_instruction(),
               m::Gt(match_bitcast_bf16(0), match_bitcast_bf16(1))) ||
         Match(comp->root_instruction(),
               m::Gt(match_bitcast_f32_with_convert(0),
                     match_bitcast_f32_with_convert(1))) ||
         Match(comp->root_instruction(),
               m::Gt(match_bitcast_bf16_with_convert(0),
                     match_bitcast_bf16_with_convert(1)));
}

absl::optional<int64> TopkRewriter::SortIsInTopK(HloInstruction* inst) {
  HloSortInstruction* sort = DynCast<HloSortInstruction>(inst);
  if (sort == nullptr) {
    return absl::nullopt;
  }
  if (sort->operand_count() != 1 && sort->operand_count() != 2) {
    return absl::nullopt;
  }
  HloInstruction* data = sort->mutable_operand(0);

  if (sort->operand_count() == 2) {
    HloIotaInstruction* iota =
        DynCast<HloIotaInstruction>(sort->mutable_operand(1));
    if (iota == nullptr || iota->shape().rank() != data->shape().rank() ||
        iota->shape().element_type() != S32 ||
        iota->opcode() != HloOpcode::kIota ||
        iota->iota_dimension() != sort->sort_dimension()) {
      return absl::nullopt;
    }
  }
  if (!IsNanSafeGt(sort->to_apply())) {
    return absl::nullopt;
  }
  const int64 sort_dim = sort->sort_dimension();
  const int64 batch_dim = sort_dim == 1 ? 0 : 1;
  const bool has_batch = data->shape().rank() == 2;

  bool supported = true;
  absl::optional<int64> k;
  for (HloInstruction* user : sort->users()) {
    const HloInstruction* slice = user;
    if (sort->operand_count() == 2) {
      if (user->opcode() != HloOpcode::kGetTupleElement ||
          user->user_count() != 1) {
        supported = false;
        break;
      }
      slice = user->users()[0];
    }
    if (slice->opcode() != HloOpcode::kSlice) {
      // Non-slice user means we are not doing a TopK
      supported = false;
      break;
    }
    if (absl::c_any_of(slice->slice_starts(), [](int x) { return x != 0; }) ||
        absl::c_any_of(slice->slice_strides(), [](int x) { return x != 1; })) {
      // Strided slice or slicing at the beginning isn't supported.
      supported = false;
      break;
    }
    if (has_batch && slice->slice_limits(batch_dim) !=
                         slice->operand(0)->shape().dimensions(batch_dim)) {
      // Slicing along the batch dimension isn't supported.
      supported = false;
      break;
    }
    if (k == absl::nullopt) {
      k = slice->slice_limits(sort_dim);
    } else if (k != slice->slice_limits(sort_dim)) {
      // Different k for the different operands isn't supported.
      supported = false;
      break;
    }
  }
  if (k == absl::nullopt || !supported) {
    return absl::nullopt;
  }
  return k;
}

StatusOr<bool> TopkRewriter::TransformToCustomCall(HloModule* module) {
  bool changed = false;
  for (HloComputation* comp : module->computations()) {
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      // Check if sort is in TopK.
      absl::optional<int64> k = SortIsInTopK(inst);
      if (!k) {
        continue;
      }

      HloSortInstruction* sort = DynCast<HloSortInstruction>(inst);
      HloInstruction* data = sort->mutable_operand(0);
      const PrimitiveType element_type = data->shape().element_type();

      if ((data->shape().rank() != 1 && data->shape().rank() != 2) ||
          (element_type != F32 && element_type != BF16)) {
        continue;
      }

      const int64 sort_dim = sort->sort_dimension();
      const int64 batch_dim = sort_dim == 1 ? 0 : 1;
      const bool has_batch = data->shape().rank() == 2;

      // Profitability check.
      if (!is_profitable_to_convert_(sort, *k)) {
        continue;
      }

      const int64 batch_size =
          has_batch ? sort->operand(0)->shape().dimensions(batch_dim) : 1;
      const int64 input_size = sort->operand(0)->shape().dimensions(sort_dim);
      HloInstruction* input = sort->mutable_operand(0);
      if (has_batch && sort_dim == 0) {
        input = comp->AddInstruction(HloInstruction::CreateTranspose(
            ShapeUtil::MakeShape(element_type, {batch_size, input_size}), input,
            {1, 0}));
      }

      Shape topk_shape =
          has_batch ? ShapeUtil::MakeTupleShape(
                          {ShapeUtil::MakeShape(element_type,
                                                {batch_size, k.value()}),
                           ShapeUtil::MakeShape(S32, {batch_size, k.value()})})
                    : ShapeUtil::MakeTupleShape(
                          {ShapeUtil::MakeShape(element_type, {k.value()}),
                           ShapeUtil::MakeShape(S32, {k.value()})});
      HloInstruction* topk = comp->AddInstruction(
          HloInstruction::CreateCustomCall(topk_shape, {input}, "TopK"));
      HloInstruction* value_gte =
          comp->AddInstruction(HloInstruction::CreateGetTupleElement(
              topk->shape().tuple_shapes(0), topk, 0));
      HloInstruction* index_gte =
          comp->AddInstruction(HloInstruction::CreateGetTupleElement(
              topk->shape().tuple_shapes(1), topk, 1));

      if (has_batch && sort_dim == 0) {
        value_gte = comp->AddInstruction(HloInstruction::CreateTranspose(
            ShapeUtil::MakeShape(element_type, {k.value(), batch_size}),
            value_gte, {1, 0}));
        index_gte = comp->AddInstruction(HloInstruction::CreateTranspose(
            ShapeUtil::MakeShape(S32, {k.value(), batch_size}), index_gte,
            {1, 0}));
      }

      for (HloInstruction* user : sort->users()) {
        if (sort->operand_count() == 2) {
          HloInstruction* gte = user;
          for (HloInstruction* slice : gte->users()) {
            if (gte->tuple_index() == 0) {
              TF_RETURN_IF_ERROR(slice->ReplaceAllUsesWith(value_gte));
            } else if (gte->tuple_index() == 1) {
              TF_RETURN_IF_ERROR(slice->ReplaceAllUsesWith(index_gte));
            } else {
              LOG(FATAL) << "Sort with more than 2 output isn't supported in "
                            "topk rewriter";
            }
          }
        } else {
          TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(value_gte));
        }
      }
      changed = true;
    }
  }
  return changed;
}

StatusOr<bool> TopkRewriter::Run(HloModule* module) {
  bool changed = false;
  TF_ASSIGN_OR_RETURN(auto transform_to_customcall_changed,
                      TransformToCustomCall(module));
  changed |= transform_to_customcall_changed;
  return changed;
}

}  // namespace xla
