/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/topk_rewriter.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "xla/hlo/builder/lib/comparators.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla {

namespace m = match;

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

  auto match_generic_iec559 = [](int64_t parameter_number,
                                 PrimitiveType fp_type,
                                 PrimitiveType int_type) {
    auto param = m::Parameter(parameter_number)
                     .WithShape(m::Shape().WithElementType(fp_type));
    auto signed_value = m::BitcastConvert(param).WithShape(
        m::Shape().WithElementType(int_type));
    int64_t bit_width = primitive_util::BitWidth(fp_type);
    auto max_value = m::ConstantScalar(LsbMask<uint64_t>(bit_width - 1));
    auto flipped_value = m::XorAnyOrder(max_value, signed_value);
    auto is_negative = m::Lt(signed_value, m::ConstantScalar(0));
    return m::Select(is_negative, flipped_value, signed_value);
  };

  auto match_generic_iec559_with_convert =
      [](int64_t parameter_number, PrimitiveType param_type,
         PrimitiveType fp_type, PrimitiveType int_type) {
        auto param = m::Parameter(parameter_number)
                         .WithShape(m::Shape().WithElementType(param_type));
        auto convert =
            m::Convert(param).WithShape(m::Shape().WithElementType(fp_type));
        auto signed_value = m::BitcastConvert(convert).WithShape(
            m::Shape().WithElementType(int_type));
        int64_t bit_width = primitive_util::BitWidth(fp_type);
        auto max_value = m::ConstantScalar(LsbMask<uint64_t>(bit_width - 1));
        auto flipped_value = m::XorAnyOrder(max_value, signed_value);
        auto is_negative = m::Lt(signed_value, m::ConstantScalar(0));
        return m::Select(is_negative, flipped_value, signed_value);
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
               m::Gt(match_generic_iec559(0, F32, S32),
                     match_generic_iec559(1, F32, S32))) ||
         Match(comp->root_instruction(),
               m::Gt(match_generic_iec559(0, BF16, S16),
                     match_generic_iec559(1, BF16, S16))) ||
         Match(comp->root_instruction(),
               m::Gt(match_generic_iec559_with_convert(0, BF16, F32, S32),
                     match_generic_iec559_with_convert(1, BF16, F32, S32))) ||
         Match(comp->root_instruction(),
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
  const std::array<int64_t, 1> sort_dims = {
      data->shape().dimensions(sort->sort_dimension())};
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
    for (int64_t i = 0; i < slice->slice_limits().size(); ++i) {
      if (i != sort_dim &&
          slice->slice_limits(i) != slice->operand(0)->shape().dimensions(i)) {
        // Slicing along a non-sort dimension isn't supported.
        supported = false;
        break;
      }
    }
    if (!supported) {
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

struct TopKCustomCall {
  HloInstruction* topk;
  HloInstruction* value_gte;
  HloInstruction* index_gte;
};

TopKCustomCall CreateTopKCustomCall(HloSortInstruction* sort, const int64_t k) {
  HloInstruction* input = sort->mutable_operand(0);
  Shape data_shape = input->shape();
  PrimitiveType element_type = data_shape.element_type();
  bool has_batch = data_shape.dimensions().size() >= 2;
  int64_t sort_dim = sort->sort_dimension();
  int64_t input_size = data_shape.dimensions(sort_dim);
  int64_t batch_size = 1;
  Shape topk_input_shape;

  if (has_batch) {
    // The TopK custom call expects either a 1d tensor or a 2d tensor with
    // the last dimension being the sort dimension. An input with rank > 2
    // is reshaped into a 2d tensor by combining non-sort dimensions into a
    // single batch dimension. The original non-sort dimensions are
    // restored for the outputs with another reshape after the custom call.
    batch_size =
        ShapeUtil::ElementsIn(data_shape) / data_shape.dimensions(sort_dim);
    topk_input_shape =
        ShapeUtil::MakeValidatedShape(element_type, {batch_size, input_size})
            .value();

    if (data_shape.dimensions().size() > 2) {
      // Reshape to 2d.
      input = sort->AddInstruction(HloInstruction::CreateReshape(
          sort_dim == 0 ? ShapeUtil::MakeValidatedShape(
                              element_type, {input_size, batch_size})
                              .value()
                        : ShapeUtil::MakeValidatedShape(
                              element_type, {batch_size, input_size})
                              .value(),
          input));
    }

    if (sort_dim == 0) {
      // Transpose for the custom call when sorting the first dimension.
      input = sort->AddInstruction(
          HloInstruction::CreateTranspose(topk_input_shape, input, {1, 0}));
    }
  } else {
    topk_input_shape = data_shape;
  }

  Shape topk_shape =
      has_batch ? ShapeUtil::MakeValidatedTupleShape(
                      {ShapeUtil::MakeShape(element_type, {batch_size, k}),
                       ShapeUtil::MakeShape(S32, {batch_size, k})})
                      .value()
                : ShapeUtil::MakeValidatedTupleShape(
                      {ShapeUtil::MakeShape(element_type, {k}),
                       ShapeUtil::MakeShape(S32, {k})})
                      .value();
  HloInstruction* topk = sort->AddInstruction(HloInstruction::CreateCustomCall(
      topk_shape, {input}, sort->to_apply(), "TopK"));
  HloInstruction* value_gte =
      sort->AddInstruction(HloInstruction::CreateGetTupleElement(
          topk->shape().tuple_shapes(0), topk, 0));
  HloInstruction* index_gte =
      sort->AddInstruction(HloInstruction::CreateGetTupleElement(
          topk->shape().tuple_shapes(1), topk, 1));

  if (has_batch) {
    if (sort_dim == 0) {
      // Transpose back.
      value_gte = sort->AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeValidatedShape(element_type, {k, batch_size}).value(),
          value_gte, {1, 0}));
      index_gte = sort->AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeValidatedShape(S32, {k, batch_size}).value(),
          index_gte, {1, 0}));
    }
    if (data_shape.dimensions().size() > 2) {
      // Reshape back.
      std::vector<int64_t> shape_dim(data_shape.dimensions().begin(),
                                     data_shape.dimensions().end());
      shape_dim[sort_dim] = k;
      value_gte = sort->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeValidatedShape(element_type, shape_dim).value(),
          value_gte));
      index_gte = sort->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeValidatedShape(S32, shape_dim).value(), index_gte));
    }
  }
  return {topk, value_gte, index_gte};
}

absl::StatusOr<HloInstruction*> TopkRewriter::TransformPatternToCustomCall(
    HloInstruction* inst) {
  // Check if sort is in TopK.
  std::optional<int64_t> k = SortIsInTopK(inst);
  if (!k) {
    return nullptr;
  }

  HloSortInstruction* sort = DynCast<HloSortInstruction>(inst);
  HloInstruction* data = sort->mutable_operand(0);
  const PrimitiveType element_type = data->shape().element_type();

  if (element_type != F32 && element_type != BF16) {
    return nullptr;
  }

  // Sort dimension must be the first or last dimension.
  const int64_t sort_dim = sort->sort_dimension();
  if (sort_dim != 0 && sort_dim != data->shape().dimensions().size() - 1) {
    return nullptr;
  }

  // Profitability check.
  if (!is_profitable_to_convert_(sort, *k)) {
    return nullptr;
  }

  TopKCustomCall topkcc = CreateTopKCustomCall(sort, k.value());

  for (HloInstruction* user : sort->users()) {
    if (sort->operand_count() == 2) {
      HloInstruction* gte = user;
      for (HloInstruction* slice : gte->users()) {
        if (gte->tuple_index() == 0) {
          TF_RETURN_IF_ERROR(slice->ReplaceAllUsesWith(topkcc.value_gte));
        } else if (gte->tuple_index() == 1) {
          TF_RETURN_IF_ERROR(slice->ReplaceAllUsesWith(topkcc.index_gte));
        } else {
          // The line below should be unreachable. SortIsInTopK() already checks
          // that sort has either 1 or 2 operands. Reaching this line indicates
          // a programming error (not a bad input), so crashing is OK.
          LOG(FATAL) << "Sort with more than 2 output isn't supported in "
                        "topk rewriter";
        }
      }
    } else {
      TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(topkcc.value_gte));
    }
  }

  return topkcc.topk;
}

absl::StatusOr<bool> TopkRewriter::TransformToCustomCall(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp : module->computations(execution_threads)) {
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(HloInstruction * topkcc,
                          TransformPatternToCustomCall(inst));
      if (topkcc != nullptr) {
        VLOG(2) << "Rewritten Topk: " << topkcc->ToString();
        changed = true;
      }
    }
  }
  return changed;
}

absl::StatusOr<bool> TopkRewriter::Run(
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

  absl::Status HandleCustomCall(HloInstruction* inst) override {
    if (should_decompose_ && !should_decompose_(inst)) {
      return absl::OkStatus();
    }
    HloCustomCallInstruction* call = DynCast<HloCustomCallInstruction>(inst);
    if (call == nullptr || call->custom_call_target() != "TopK") {
      return absl::OkStatus();
    }
    HloComputation* comparator = call->to_apply();
    return DecomposeTopK(call, comparator);
  }

  absl::Status HandleTopK(HloInstruction* topk) override {
    if (should_decompose_ && !should_decompose_(topk)) {
      return absl::OkStatus();
    }
    TF_ASSIGN_OR_RETURN(HloComputation * comparator,
                        CreateVariadicComparator(topk));
    return DecomposeTopK(topk, comparator);
  }

 private:
  bool HasSingleUserReadingOnlyTheValueOutput(HloInstruction* inst) {
    return inst->user_count() == 1 && inst->users().front()->tuple_index() == 0;
  }

  absl::StatusOr<HloComputation*> CreateVariadicComparator(
      HloInstruction* inst) {
    HloTopKInstruction* topk = DynCast<HloTopKInstruction>(inst);
    XlaBuilder b(absl::StrCat("comparator_", topk->name()));
    std::vector<PrimitiveType> ptypes = {
        topk->operand(0)->shape().element_type()};

    if (!HasSingleUserReadingOnlyTheValueOutput(inst)) {
      ptypes.emplace_back(PrimitiveType::S32);
    }

    XlaComputation comparison = topk->largest()
                                    ? CreateScalarGtComputation(ptypes, &b)
                                    : CreateScalarLtComputation(ptypes, &b);
    TF_ASSIGN_OR_RETURN(
        HloComputation * comparator,
        XlaComputationToHloComputation(comparison, topk->parent()->parent()));
    return comparator;
  }

  absl::Status DecomposeTopK(HloInstruction* call,
                             HloComputation* variadic_comparator) {
    HloInstruction* input = call->mutable_operand(0);
    Shape iota_shape = input->shape();
    iota_shape.set_element_type(S32);
    size_t sort_dimension = input->shape().dimensions().size() - 1;
    std::vector<int64_t> zeroes(iota_shape.dimensions().size(), 0);
    std::vector<int64_t> ones(iota_shape.dimensions().size(), 1);
    CHECK_NE(variadic_comparator, nullptr);
    // If only the topk values are necessary, skip the iota.
    if (HasSingleUserReadingOnlyTheValueOutput(call) &&
        variadic_comparator->num_parameters() == 2) {
      HloInstruction* sort = call->AddInstruction(HloInstruction::CreateSort(
          input->shape(), sort_dimension, {input}, variadic_comparator,
          /*is_stable=*/true));
      TF_RETURN_IF_ERROR(ReplaceInstruction(
          call->users().front(),
          call->AddInstruction(HloInstruction::CreateSlice(
              call->shape().tuple_shapes(0), sort, zeroes,
              call->shape().tuple_shapes(0).dimensions(), ones))));
    } else {
      HloInstruction* iota = call->AddInstruction(HloInstruction::CreateIota(
          iota_shape, iota_shape.dimensions().size() - 1));
      HloInstruction* sort = call->AddInstruction(HloInstruction::CreateSort(
          ShapeUtil::MakeValidatedTupleShape({input->shape(), iota_shape})
              .value(),
          sort_dimension, {input, iota}, variadic_comparator,
          /*is_stable=*/true));
      // Apply a slice to a tuple.
      auto slice_tuple = [&](const size_t index) {
        return call->AddInstruction(HloInstruction::CreateSlice(
            call->shape().tuple_shapes(index),
            call->AddInstruction(HloInstruction::CreateGetTupleElement(
                sort->shape().tuple_shapes(index), sort, index)),
            zeroes, call->shape().tuple_shapes(index).dimensions(), ones));
      };
      TF_RETURN_IF_ERROR(ReplaceInstruction(
          call, call->AddInstruction(HloInstruction::CreateTuple(
                    {slice_tuple(0), slice_tuple(1)}))));
    }
    return absl::OkStatus();
  }

 private:
  HloPredicate should_decompose_;
};

absl::StatusOr<bool> TopkDecomposer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return TopkDecomposerVisitor(should_decompose_)
      .RunOnModule(module, execution_threads);
}

}  // namespace xla
