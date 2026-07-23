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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/builder/lib/comparators.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

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
        ShapeUtil::MakeShape(element_type, {batch_size, input_size});

    if (data_shape.dimensions().size() > 2) {
      // Reshape to 2d.
      input = sort->AddInstruction(HloInstruction::CreateReshape(
          sort_dim == 0
              ? ShapeUtil::MakeShape(element_type, {input_size, batch_size})
              : ShapeUtil::MakeShape(element_type, {batch_size, input_size}),
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
      has_batch
          ? ShapeUtil::MakeTupleShape(
                {ShapeUtil::MakeShape(element_type, {batch_size, k}),
                 ShapeUtil::MakeShape(S32, {batch_size, k})})
          : ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(element_type, {k}),
                                       ShapeUtil::MakeShape(S32, {k})});
  HloInstruction* topk = sort->AddInstruction(HloInstruction::CreateCustomCall(
      topk_shape, {input}, sort->to_apply(), "TopK"));
  topk->set_raw_backend_config_string(absl::StrFormat(
      "{is_stable = %s}", sort->is_stable() ? "true" : "false"));
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
          ShapeUtil::MakeShape(element_type, {k, batch_size}), value_gte,
          {1, 0}));
      index_gte = sort->AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(S32, {k, batch_size}), index_gte, {1, 0}));
    }
    if (data_shape.dimensions().size() > 2) {
      // Reshape back.
      std::vector<int64_t> shape_dim(data_shape.dimensions().begin(),
                                     data_shape.dimensions().end());
      shape_dim[sort_dim] = k;
      value_gte = sort->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(element_type, shape_dim), value_gte));
      index_gte = sort->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(S32, shape_dim), index_gte));
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
          RETURN_IF_ERROR(slice->ReplaceAllUsesWith(topkcc.value_gte));
        } else if (gte->tuple_index() == 1) {
          RETURN_IF_ERROR(slice->ReplaceAllUsesWith(topkcc.index_gte));
        } else {
          // The line below should be unreachable. SortIsInTopK() already checks
          // that sort has either 1 or 2 operands. Reaching this line indicates
          // a programming error (not a bad input), so crashing is OK.
          LOG(FATAL) << "Sort with more than 2 output isn't supported in "
                        "topk rewriter";
        }
      }
    } else {
      RETURN_IF_ERROR(user->ReplaceAllUsesWith(topkcc.value_gte));
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
      ASSIGN_OR_RETURN(HloInstruction * topkcc,
                       TransformPatternToCustomCall(inst));
      if (topkcc != nullptr) {
        VLOG(2) << "Rewritten Topk: " << topkcc->ToString();
        changed = true;
      }
    }
  }
  return changed;
}

absl::StatusOr<bool> TopkRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  ASSIGN_OR_RETURN(auto transform_to_customcall_changed,
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
    // TODO: Support packed BF16 sort for legacy custom calls if needed.
    return DecomposeTopKFallback(call, comparator);
  }

  absl::Status HandleTopK(HloInstruction* inst) override {
    if (should_decompose_ && !should_decompose_(inst)) {
      return absl::OkStatus();
    }
    auto* topk = DynCast<HloTopKInstruction>(inst);

    // Use packed BF16 sort optimization when applicable: BF16 input with
    // largest=true, both values and indices used, and the last dimension size
    // fits in 16 bits so indices can be packed.
    constexpr int32_t kLow16BitsLimit = int32_t{1} << 16;
    if (topk->largest() && topk->operand(0)->shape().element_type() == BF16 &&
        !HasSingleUserReadingOnlyTheValueOutput(topk) &&
        topk->operand(0)->shape().dimensions().back() < kLow16BitsLimit) {
      return DecomposeTopKWithSorting(topk);
    }

    ASSIGN_OR_RETURN(HloComputation * comparator,
                     CreateVariadicComparator(topk));
    return DecomposeTopKFallback(topk, comparator);
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
    ASSIGN_OR_RETURN(
        HloComputation * comparator,
        XlaComputationToHloComputation(comparison, topk->parent()->parent()));
    return comparator;
  }

  // Decompose BF16 TopK by packing values and indices into a single S32 value,
  // sorting, then unpacking. This avoids the two-operand sort and uses an
  // unstable single-operand sort on packed S32 values. The BF16 values occupy
  // the high 16 bits (in ones' complement form for correct ordering) and the
  // indices occupy the low 16 bits.
  //
  // This implementation constructs HLO directly rather than going through
  // XlaBuilder to avoid generating SetDimensionSize instructions that are
  // incompatible with while loop bodies on some GPU backends.
  absl::Status DecomposeTopKWithSorting(HloInstruction* call) {
    HloInstruction* input = call->mutable_operand(0);
    const Shape& input_shape = input->shape();
    int64_t rank = input_shape.dimensions().size();
    int64_t last_dim = rank - 1;

    HloTopKInstruction* topk = DynCast<HloTopKInstruction>(call);
    int64_t k = topk->k();

    constexpr int32_t kLow16BitsLimit = int32_t{1} << 16;
    constexpr int32_t kLow16BitsMask = kLow16BitsLimit - 1;
    constexpr int32_t kHigh16BitsMask = ~kLow16BitsMask;
    constexpr int32_t kAllNonSignBits = 0x7fffffff;

    HloComputation* parent_comp = call->parent();
    Shape s32_shape = ShapeUtil::ChangeElementType(input_shape, S32);
    Shape f32_shape = ShapeUtil::ChangeElementType(input_shape, F32);

    // Helper: broadcast an S32 scalar constant to the given shape.
    auto broadcast_s32 = [&](int32_t value, const Shape& shape) {
      return parent_comp->AddInstruction(HloInstruction::CreateBroadcast(
          shape,
          parent_comp->AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<int32_t>(value))),
          {}));
    };

    // Sign-magnitude to ones' complement conversion: converts IEEE 754 floats
    // (viewed as S32 after bitcast) to a representation where integer
    // comparison preserves float ordering.
    auto sign_mag_to_ones_comp = [&](HloInstruction* s32_input,
                                     const Shape& shape) {
      HloInstruction* non_sign = parent_comp->AddInstruction(
          HloInstruction::CreateBinary(shape, HloOpcode::kAnd, s32_input,
                                       broadcast_s32(kAllNonSignBits, shape)));
      HloInstruction* sign_mask = parent_comp->AddInstruction(
          HloInstruction::CreateBinary(shape, HloOpcode::kShiftRightArithmetic,
                                       s32_input, broadcast_s32(31, shape)));
      return parent_comp->AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kXor, non_sign, sign_mask));
    };

    // Step 1: Convert BF16 -> F32 -> bitcast to S32.
    HloInstruction* input_f32 = parent_comp->AddInstruction(
        HloInstruction::CreateConvert(f32_shape, input));
    HloInstruction* input_s32 = parent_comp->AddInstruction(
        HloInstruction::CreateBitcastConvert(s32_shape, input_f32));

    // Step 2: Apply sign-magnitude to ones' complement conversion.
    HloInstruction* input_ones_comp =
        sign_mag_to_ones_comp(input_s32, s32_shape);

    // Step 3: OR with kLow16BitsMask to set low 16 bits, reversing the index
    // order for tie-breaking in the sort.
    HloInstruction* input_with_low_bits = parent_comp->AddInstruction(
        HloInstruction::CreateBinary(s32_shape, HloOpcode::kOr, input_ones_comp,
                                     broadcast_s32(kLow16BitsMask, s32_shape)));

    // Step 4: Create iota for indices.
    HloInstruction* iota = parent_comp->AddInstruction(
        HloInstruction::CreateIota(s32_shape, last_dim));

    // Step 5: XOR input with iota to pack values and indices together.
    HloInstruction* packed =
        parent_comp->AddInstruction(HloInstruction::CreateBinary(
            s32_shape, HloOpcode::kXor, input_with_low_bits, iota));

    // Step 6: Sort in descending order (GT comparator on S32, unstable).
    XlaBuilder b(absl::StrCat("packed_comparator_", call->name()));
    XlaComputation gt_comp = CreateScalarGtComputation({S32}, &b);
    ASSIGN_OR_RETURN(
        HloComputation * comparator,
        XlaComputationToHloComputation(gt_comp, parent_comp->parent()));

    HloInstruction* sorted = parent_comp->AddInstruction(
        HloInstruction::CreateSort(s32_shape, last_dim, {packed}, comparator,
                                   /*is_stable=*/false));

    // Step 7: Slice top-k elements.
    std::vector<int64_t> zeroes(rank, 0);
    std::vector<int64_t> ones(rank, 1);
    std::vector<int64_t> limits(input_shape.dimensions().begin(),
                                input_shape.dimensions().end());
    limits[last_dim] = k;
    Shape sliced_s32_shape = call->shape().tuple_shapes(1);  // S32[..., k]

    HloInstruction* sliced =
        parent_comp->AddInstruction(HloInstruction::CreateSlice(
            sliced_s32_shape, sorted, zeroes, limits, ones));

    // Step 8: Extract values. Apply ones' complement conversion back, mask
    // high 16 bits, bitcast to F32, convert to BF16.
    HloInstruction* sliced_ones_comp =
        sign_mag_to_ones_comp(sliced, sliced_s32_shape);
    HloInstruction* high_bits =
        parent_comp->AddInstruction(HloInstruction::CreateBinary(
            sliced_s32_shape, HloOpcode::kAnd, sliced_ones_comp,
            broadcast_s32(kHigh16BitsMask, sliced_s32_shape)));
    Shape sliced_f32_shape =
        ShapeUtil::ChangeElementType(sliced_s32_shape, F32);
    HloInstruction* values_f32 = parent_comp->AddInstruction(
        HloInstruction::CreateBitcastConvert(sliced_f32_shape, high_bits));
    HloInstruction* values =
        parent_comp->AddInstruction(HloInstruction::CreateConvert(
            call->shape().tuple_shapes(0), values_f32));

    // Step 9: Extract indices. XOR with kLow16BitsMask to invert the index
    // bits back, AND with kLow16BitsMask to isolate the index bits.
    HloInstruction* indices_xor =
        parent_comp->AddInstruction(HloInstruction::CreateBinary(
            sliced_s32_shape, HloOpcode::kXor, sliced,
            broadcast_s32(kLow16BitsMask, sliced_s32_shape)));
    HloInstruction* indices =
        parent_comp->AddInstruction(HloInstruction::CreateBinary(
            sliced_s32_shape, HloOpcode::kAnd, indices_xor,
            broadcast_s32(kLow16BitsMask, sliced_s32_shape)));

    // Step 10: Create tuple of (values, indices) and replace.
    RETURN_IF_ERROR(ReplaceInstruction(
        call, parent_comp->AddInstruction(
                  HloInstruction::CreateTuple({values, indices}))));
    return absl::OkStatus();
  }

  absl::Status DecomposeTopKFallback(HloInstruction* call,
                                     HloComputation* variadic_comparator) {
    HloInstruction* input = call->mutable_operand(0);
    const Shape& input_shape = input->shape();
    int64_t rank = input_shape.dimensions().size();
    size_t sort_dimension = rank - 1;
    Shape iota_shape = input->shape();
    iota_shape.set_element_type(S32);
    bool is_stable = true;
    if (auto* topk_inst = DynCast<HloTopKInstruction>(call)) {
      is_stable = topk_inst->is_stable();
    } else if (auto* custom_call = DynCast<HloCustomCallInstruction>(call)) {
      is_stable = hlo_instruction_utils::IsTopKStable(custom_call);
    }
    std::vector<int64_t> zeroes(rank, 0);
    std::vector<int64_t> ones(rank, 1);

    // If only the topk values are necessary, skip the iota.
    if (HasSingleUserReadingOnlyTheValueOutput(call) &&
        variadic_comparator->num_parameters() == 2) {
      HloInstruction* sort = call->AddInstruction(
          HloInstruction::CreateSort(input->shape(), sort_dimension, {input},
                                     variadic_comparator, is_stable));
      RETURN_IF_ERROR(ReplaceInstruction(
          call->users().front(),
          call->AddInstruction(HloInstruction::CreateSlice(
              call->shape().tuple_shapes(0), sort, zeroes,
              call->shape().tuple_shapes(0).dimensions(), ones))));
    } else {
      HloInstruction* iota = call->AddInstruction(HloInstruction::CreateIota(
          iota_shape, iota_shape.dimensions().size() - 1));
      HloInstruction* sort = call->AddInstruction(HloInstruction::CreateSort(
          ShapeUtil::MakeTupleShape({input->shape(), iota_shape}),
          sort_dimension, {input, iota}, variadic_comparator, is_stable));
      // Apply a slice to a tuple.
      auto slice_tuple = [&](const size_t index) {
        return call->AddInstruction(HloInstruction::CreateSlice(
            call->shape().tuple_shapes(index),
            call->AddInstruction(HloInstruction::CreateGetTupleElement(
                sort->shape().tuple_shapes(index), sort, index)),
            zeroes, call->shape().tuple_shapes(index).dimensions(), ones));
      };
      RETURN_IF_ERROR(ReplaceInstruction(
          call, call->AddInstruction(HloInstruction::CreateTuple(
                    {slice_tuple(0), slice_tuple(1)}))));
    }
    return absl::OkStatus();
  }

 private:
  HloPredicate should_decompose_;
};

absl::StatusOr<bool> TopkDecomposer::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return TopkDecomposerVisitor(should_decompose_)
      .RunOnModule(module, execution_threads);
}

}  // namespace xla
