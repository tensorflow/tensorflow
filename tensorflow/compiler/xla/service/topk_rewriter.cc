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

#include <memory>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/match.h"
#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/tsl/platform/logging.h"

namespace xla {

namespace m = match;

// TODO(cheshire): Avoid duplication w/ cudnn_vectorize_convolutions.
static StatusOr<HloComputation*> BuilderToHloComputation(
    XlaComputation& comp, HloComputation* sibling_computation) {
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape, comp.GetProgramShape());
  HloModuleConfig config(program_shape);
  TF_ASSIGN_OR_RETURN(auto new_module,
                      HloModule::CreateFromProto(comp.proto(), config));

  HloModule* dest_module = sibling_computation->parent();
  HloCloneContext context(dest_module);
  return dest_module->DeepCloneComputation(new_module->entry_computation(),
                                           &context);
}

static bool IsNanSafeGt(HloComputation* comp) {
  namespace m = match;
  auto match_bitcast_f32 = [](int64_t parameter_number) {
    auto param = m::Parameter(parameter_number)
                     .WithShape(m::Shape().WithElementType(F32));
    auto param_s32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(S32));
    auto param_u32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(U32));
    return m::Select(
        m::Lt(param_s32, m::ConstantScalar(0)),
        m::BitcastConvert(
            m::Subtract(m::ConstantScalar(std::numeric_limits<int32_t>::max()),
                        param_u32))
            .WithShape(m::Shape().WithElementType(S32)),
        param_s32);
  };

  auto match_bitcast_f32_with_convert = [](int64_t parameter_number) {
    auto param = m::Parameter(parameter_number)
                     .WithShape(m::Shape().WithElementType(F32));
    auto param_s32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(S32));
    auto param_u32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(U32));
    auto max_u32 =
        m::Convert(m::ConstantScalar(std::numeric_limits<int32_t>::max()))
            .WithShape(m::Shape().WithElementType(U32));
    return m::Select(m::Lt(param_s32, m::ConstantScalar(0)),
                     m::BitcastConvert(m::Subtract(max_u32, param_u32))
                         .WithShape(m::Shape().WithElementType(S32)),
                     param_s32);
  };

  auto match_bitcast_bf16 = [](int64_t parameter_number) {
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
            m::Subtract(m::ConstantScalar(std::numeric_limits<int32_t>::max()),
                        param_u32))
            .WithShape(m::Shape().WithElementType(S32)),
        param_s32);
  };

  auto match_bitcast_bf16_with_convert = [](int64_t parameter_number) {
    auto param = m::Convert(m::Parameter(parameter_number)
                                .WithShape(m::Shape().WithElementType(BF16)))
                     .WithShape(m::Shape().WithElementType(F32));
    auto param_s32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(S32));
    auto param_u32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(U32));
    auto max_u32 =
        m::Convert(m::ConstantScalar(std::numeric_limits<int32_t>::max()))
            .WithShape(m::Shape().WithElementType(U32));
    return m::Select(m::Lt(param_s32, m::ConstantScalar(0)),
                     m::BitcastConvert(m::Subtract(max_u32, param_u32))
                         .WithShape(m::Shape().WithElementType(S32)),
                     param_s32);
  };

  auto match_s32 = [](int64_t parameter_number) {
    auto param = m::Parameter(parameter_number)
                     .WithShape(m::Shape().WithElementType(S32));
    return param;
  };

  auto match_compare = [](PrimitiveType type) {
    auto param0 = m::Parameter(0).WithShape(m::Shape().WithElementType(type));
    auto param1 = m::Parameter(1).WithShape(m::Shape().WithElementType(type));
    return m::Gt(param0, param1);
  };

  auto match_default_compare = [](PrimitiveType type) {
    auto params_with_type = [&](int i, PrimitiveType t) {
      return m::Parameter(i).WithShape(m::Shape().WithElementType(t));
    };
    auto params =
        std::vector({// Values
                     params_with_type(0, type), params_with_type(1, type),
                     // Indices
                     params_with_type(2, S32), params_with_type(3, S32)});
    auto const_true = m::Broadcast(m::Constant());
    auto values_gt = m::Gt(params[0], params[1]);
    return m::Select(const_true, values_gt, const_true);
  };

  auto match_all_types = [](HloInstruction* root, auto callback) {
    bool result = false;
    for (auto type : {BF16, F32, S32, U32}) {
      result = result || Match(root, callback(type));
    }
    return result;
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
                     match_bitcast_bf16_with_convert(1))) ||
         Match(comp->root_instruction(), m::Gt(match_s32(0), match_s32(1))) ||
         match_all_types(comp->root_instruction(), match_compare) ||
         match_all_types(comp->root_instruction(), match_default_compare);
}

// Look for the instructions emitted from: xla/client/lib/sorting.cc
static bool HasIota(HloSortInstruction* sort, HloInstruction* data) {
  namespace m = match;
  const auto sort_dims = {data->shape().dimensions(sort->sort_dimension())};
  auto match_iota = [](auto dims) {
    return m::Iota().WithShape(m::Shape().WithElementType(S32).WithDims(dims));
  };
  return Match(sort->operand(1), match_iota(data->shape().dimensions())) ||
         Match(sort->operand(1), m::Broadcast(match_iota(sort_dims)));
}

std::optional<int64_t> TopkRewriter::SortIsInTopK(HloInstruction* inst) {
  HloSortInstruction* sort = DynCast<HloSortInstruction>(inst);
  if (sort == nullptr) {
    return std::nullopt;
  }
  if (sort->operand_count() != 1 && sort->operand_count() != 2) {
    return std::nullopt;
  }
  HloInstruction* data = sort->mutable_operand(0);

  if (sort->operand_count() == 2 && !HasIota(sort, data)) {
    return std::nullopt;
  }
  if (!IsNanSafeGt(sort->to_apply())) {
    return std::nullopt;
  }
  const int64_t sort_dim = sort->sort_dimension();
  const int64_t batch_dim = sort_dim == 1 ? 0 : 1;
  const bool has_batch = data->shape().rank() == 2;

  bool supported = true;
  std::optional<int64_t> k;
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
    if (k == std::nullopt) {
      k = slice->slice_limits(sort_dim);
    } else if (k != slice->slice_limits(sort_dim)) {
      // Different k for the different operands isn't supported.
      supported = false;
      break;
    }
  }
  if (k == std::nullopt || !supported) {
    return std::nullopt;
  }
  return k;
}

StatusOr<bool> TopkRewriter::TransformToCustomCall(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp : module->computations(execution_threads)) {
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      // Check if sort is in TopK.
      std::optional<int64_t> k = SortIsInTopK(inst);
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

      const int64_t sort_dim = sort->sort_dimension();
      const int64_t batch_dim = sort_dim == 1 ? 0 : 1;
      const bool has_batch = data->shape().rank() == 2;

      // Profitability check.
      if (!is_profitable_to_convert_(sort, *k)) {
        continue;
      }

      const int64_t batch_size =
          has_batch ? sort->operand(0)->shape().dimensions(batch_dim) : 1;
      const int64_t input_size = sort->operand(0)->shape().dimensions(sort_dim);
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
      HloInstruction* topk =
          comp->AddInstruction(HloInstruction::CreateCustomCall(
              topk_shape, {input}, /*to_apply=*/sort->to_apply(), "TopK"));
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
      VLOG(2) << "Rewritten Topk: " << topk->ToString();
      changed = true;
    }
  }
  return changed;
}

StatusOr<bool> TopkRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  TF_ASSIGN_OR_RETURN(auto transform_to_customcall_changed,
                      TransformToCustomCall(module, execution_threads));
  changed |= transform_to_customcall_changed;
  return changed;
}

class TopkDecomposerVisitor : public DfsHloRewriteVisitor {
 public:
  explicit TopkDecomposerVisitor(HloPredicate should_decompose)
      : should_decompose_(should_decompose) {}

  Status HandleCustomCall(HloInstruction* inst) override {
    if (should_decompose_ && !should_decompose_(inst)) {
      return OkStatus();
    }
    HloCustomCallInstruction* call = DynCast<HloCustomCallInstruction>(inst);
    if (call == nullptr || call->custom_call_target() != "TopK") {
      return OkStatus();
    }
    HloComputation* comparator = call->to_apply();
    return DecomposeTopK(call, comparator);
  }

  Status HandleTopK(HloInstruction* topk) override {
    if (should_decompose_ && !should_decompose_(topk)) {
      return OkStatus();
    }
    TF_ASSIGN_OR_RETURN(HloComputation * comparator,
                        CreateVariadicComparator(topk));
    return DecomposeTopK(topk, comparator);
  }

 private:
  StatusOr<HloComputation*> CreateVariadicComparator(HloInstruction* topk) {
    XlaBuilder b(absl::StrCat("comparator_", topk->name()));
    std::vector<PrimitiveType> ptypes = {
        topk->operand(0)->shape().element_type(), PrimitiveType::S32};
    HloComputation* comparison_computation = topk->to_apply();

    auto comparison = [&]() -> StatusOr<XlaComputation> {
      if (Match(comparison_computation->root_instruction(),
                m::Compare(m::Parameter(0), m::Parameter(1))
                    .WithComparisonDirection(ComparisonDirection::kGt)) ||
          Match(comparison_computation->root_instruction(),
                m::Compare(m::Parameter(1), m::Parameter(0))
                    .WithComparisonDirection(ComparisonDirection::kLt))) {
        return CreateScalarGtComputation(ptypes, &b);
      } else if (Match(
                     comparison_computation->root_instruction(),
                     m::Compare(m::Parameter(0), m::Parameter(1))
                         .WithComparisonDirection(ComparisonDirection::kLt)) ||
                 Match(
                     comparison_computation->root_instruction(),
                     m::Compare(m::Parameter(1), m::Parameter(0))
                         .WithComparisonDirection(ComparisonDirection::kGt))) {
        return CreateScalarLtComputation(ptypes, &b);
      } else {
        return InternalError("Unexpected comparator: %s",
                             comparison_computation->ToString());
      }
    }();
    TF_RETURN_IF_ERROR(comparison.status());
    TF_ASSIGN_OR_RETURN(HloComputation * comparator,
                        BuilderToHloComputation(*comparison, topk->parent()));
    return comparator;
  }

  Status DecomposeTopK(HloInstruction* call,
                       HloComputation* variadic_comparator) {
    HloComputation* comp = call->parent();
    HloInstruction* input = call->mutable_operand(0);
    Shape iota_shape = input->shape();
    iota_shape.set_element_type(S32);
    size_t sort_dimension = input->shape().dimensions_size() - 1;
    std::vector<int64_t> zeroes(iota_shape.rank(), 0);
    std::vector<int64_t> ones(iota_shape.rank(), 1);
    // Apply a slice to a tuple.
    auto slice_tuple = [&](HloInstruction* sort, const size_t index) {
      return comp->AddInstruction(HloInstruction::CreateSlice(
          call->shape().tuple_shapes(index),
          comp->AddInstruction(HloInstruction::CreateGetTupleElement(
              sort->shape().tuple_shapes(index), sort, index)),
          zeroes, call->shape().tuple_shapes(index).dimensions(), ones));
    };
    CHECK_NE(variadic_comparator, nullptr);
    // If only the topk values are necessary, skip the iota.
    if (call->user_count() == 1 && call->users().front()->tuple_index() == 0) {
      HloInstruction* sort = comp->AddInstruction(HloInstruction::CreateSort(
          {input->shape()}, sort_dimension, {input}, call->to_apply(),
          /*is_stable=*/true));
      TF_RETURN_IF_ERROR(ReplaceInstruction(
          call->users().front(),
          comp->AddInstruction(HloInstruction::CreateSlice(
              call->shape().tuple_shapes(0), sort, zeroes,
              call->shape().tuple_shapes(0).dimensions(), ones))));
      sort->set_metadata(call->metadata());
    } else {
      HloInstruction* iota = comp->AddInstruction(
          HloInstruction::CreateIota(iota_shape, iota_shape.rank() - 1));
      HloInstruction* sort = comp->AddInstruction(HloInstruction::CreateSort(
          ShapeUtil::MakeTupleShape({input->shape(), iota_shape}),
          sort_dimension, {input, iota}, variadic_comparator,
          /*is_stable=*/true));
      TF_RETURN_IF_ERROR(ReplaceInstruction(
          call, comp->AddInstruction(HloInstruction::CreateTuple(
                    {slice_tuple(sort, 0), slice_tuple(sort, 1)}))));
      sort->set_metadata(call->metadata());
    }
    return OkStatus();
  }

 private:
  HloPredicate should_decompose_;
};

StatusOr<bool> TopkDecomposer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return TopkDecomposerVisitor(should_decompose_)
      .RunOnModule(module, execution_threads);
}

}  // namespace xla
