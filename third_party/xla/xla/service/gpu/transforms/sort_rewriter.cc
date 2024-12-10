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

#include "xla/service/gpu/transforms/sort_rewriter.h"

#include <algorithm>
#include <cstddef>
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
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/runtime/cub_sort_thunk.h"
#include "xla/service/pattern_matcher.h"
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

namespace m = match;

// Floating point numbers can be sorted in two ways:
// * Default order (aka total order):
//   -NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN.
// * Numpy sorts NaNs last, even when negative:
//   -Inf < -Finite < +/-0 < +Finite < +Inf < +/-NaN.
//   Note that negative and positive zeros are considered equal and appear in
//   the result in the same order as they appear in the input. The same applies
//   to negative and positive NaNs.
enum class SortOrderType {
  kDefaultOrder,
  kNumpyOrder,
};

// Analyze sort comparer function.
struct SortComputationAnalysis {
  int key_operand;  // 0 or 1
  bool descending;
  SortOrderType sort_order;
  PrimitiveType key_type;
  std::optional<PrimitiveType> value_type;
};

bool MatchConstNan(const HloInstruction* op) {
  const auto const_nan = DynCast<HloConstantInstruction>(op);
  if (const_nan == nullptr) {
    return false;
  }
  return const_nan->literal().GetAsString({}) == "nan";
}

// Matches the HLO pattern used to ensure Numpy sort order. This is how JAX
// lowers `lax.sort` to HLO comparators.
int ParamNumberOfCanonicalizedZerosAndNans(const HloInstruction* select) {
  const HloInstruction* param = nullptr;
  const HloInstruction* maybe_const_nan;
  if (!Match(select,
             m::Select(
                 m::Compare(m::Parameter(&param), m::Parameter(&param))
                     .WithComparisonDirection(ComparisonDirection::kNe),
                 m::Constant(&maybe_const_nan),
                 m::Select(
                     m::Compare(m::Parameter(&param),
                                m::ConstantEffectiveScalar(0))
                         .WithComparisonDirection(ComparisonDirection::kEq),
                     m::ConstantEffectiveScalar(0), m::Parameter(&param))))) {
    return -1;
  }
  if (!MatchConstNan(maybe_const_nan)) {
    return -1;
  }
  return param->parameter_number();
}

// Returns numbers of the parameters used in a comparator for Numpy sort order.
std::pair<int64_t, int64_t> ParamNumberOfNumpySortComparator(
    const HloCompareInstruction* cmp_op) {
  const HloInstruction *select0, *select1;
  if (!Match(cmp_op, m::Compare(m::Op(&select0), m::Op(&select1)))) {
    return std::pair<int64_t, int64_t>(-1, -1);
  }
  return std::pair<int64_t, int64_t>(
      ParamNumberOfCanonicalizedZerosAndNans(select0),
      ParamNumberOfCanonicalizedZerosAndNans(select1));
}

// Returns numbers of the parameters used in a simple comparator.
std::pair<int64_t, int64_t> ParamNumberOfSimpleSortComparator(
    const HloCompareInstruction* cmp_op) {
  if (cmp_op == nullptr) {
    return std::pair<int64_t, int64_t>(-1, -1);
  }
  const HloParameterInstruction* param0 =
      DynCast<HloParameterInstruction>(cmp_op->operand(0));
  const HloParameterInstruction* param1 =
      DynCast<HloParameterInstruction>(cmp_op->operand(1));
  return (param0 && param1) ? std::make_pair(param0->parameter_number(),
                                             param1->parameter_number())
                            : std::pair<int64_t, int64_t>(-1, -1);
}

// Returns sort info on compatible compare instructions. The instruction may
// belong to a computation that has 2 or 4 operands. If this is the root
// instruction of a computation with 4 parameters only succeeds in cases where
// 2 of the parameters are ignored.
std::optional<SortComputationAnalysis> AnalyzeCompareOp(
    const HloInstruction* maybe_compare_op) {
  // Root instruction must be a comparison with a valid direction.
  const HloCompareInstruction* compare =
      DynCast<HloCompareInstruction>(maybe_compare_op);
  if (compare == nullptr || compare->direction() == ComparisonDirection::kEq ||
      compare->direction() == ComparisonDirection::kNe) {
    return std::nullopt;
  }

  // Determine the sort order and the parameters used in the comparator.
  SortOrderType sort_order;
  int64_t index0, index1;
  auto [simple_sort_index0, simple_sort_index1] =
      ParamNumberOfSimpleSortComparator(compare);
  if (simple_sort_index0 != -1 && simple_sort_index1 != -1) {
    sort_order = SortOrderType::kDefaultOrder;
    index0 = simple_sort_index0;
    index1 = simple_sort_index1;
  } else {
    auto [numpy_sort_index0, numpy_sort_index1] =
        ParamNumberOfNumpySortComparator(compare);
    if (numpy_sort_index0 != -1 && numpy_sort_index1 != -1) {
      sort_order = SortOrderType::kNumpyOrder;
      index0 = numpy_sort_index0;
      index1 = numpy_sort_index1;
    } else {
      return std::nullopt;
    }
  }

  // When sorting a pair of tensors, the parameters should be adjacent.
  int first_index = std::min(index0, index1);
  if (first_index % 2 != 0 || std::max(index0, index1) != first_index + 1) {
    return std::nullopt;
  }

  // Return the tensor index and the sort direction.
  bool descending = compare->direction() == ComparisonDirection::kGt ||
                    compare->direction() == ComparisonDirection::kGe;
  bool reverse = first_index != index0;
  return SortComputationAnalysis{first_index / 2, descending != reverse,
                                 sort_order};
}

std::optional<SortComputationAnalysis> AnalyzeSortOp(
    const HloSortInstruction& sort_op) {
  auto computation = sort_op.called_computations().front();

  auto sort_analysis = AnalyzeCompareOp(computation->root_instruction());
  if (!sort_analysis.has_value()) {
    return std::nullopt;
  }

  PrimitiveType sort_key_type =
      sort_op.operand(sort_analysis->key_operand)->shape().element_type();
  // Sort values are only present if sorting a pair of tensors.
  std::optional<PrimitiveType> sort_value_type;
  if (sort_op.operand_count() == 2) {
    // The value operand of the sort op is either 0 or 1, the opposite of the
    // key operand.
    int value_index = 1 - sort_analysis->key_operand;
    sort_value_type = sort_op.operand(value_index)->shape().element_type();
  }
  // For sorting in Numpy order, synthetic keys are materialized. The synthetic
  // keys and the original values are sorted as pairs.
  if (sort_analysis->sort_order == SortOrderType::kNumpyOrder) {
    // TODO(tjoerg): Add support for dtypes besides bf16.
    if (sort_key_type != BF16) {
      return std::nullopt;
    }
    // Sorting a pair of input tensors is not supported. The keys to sort on
    // will be generated synthetically.
    if (sort_op.operand_count() != 1) {
      return std::nullopt;
    }
    sort_key_type = U16;
    sort_value_type = BF16;
  }
  return SortComputationAnalysis{
      sort_analysis->key_operand, sort_analysis->descending,
      sort_analysis->sort_order, sort_key_type, sort_value_type};
}

// Create runner for CUB sort operation.
absl::StatusOr<std::unique_ptr<CubSortRunnerInterface>> CreateRunner(
    const SortComputationAnalysis& sort_analysis) {
  return CubSortRunnerInterface::Create(sort_analysis.key_type,
                                        sort_analysis.value_type);
}

// Restore the result shape after sorting a pair of tensors.
// The trailing argument is the scratch buffer which should be discarded.
HloInstruction* UnpackResultPair(HloSortInstruction* sort_op,
                                 HloInstruction* custom_call, bool swap) {
  HloInstruction* gte0 =
      sort_op->AddInstruction(HloInstruction::CreateGetTupleElement(
          sort_op->operand(0)->shape(), custom_call, swap ? 1 : 0));
  HloInstruction* gte1 =
      sort_op->AddInstruction(HloInstruction::CreateGetTupleElement(
          sort_op->operand(1)->shape(), custom_call, swap ? 0 : 1));
  return sort_op->AddInstruction(HloInstruction::CreateTuple({gte0, gte1}));
}

// Add HLO ops to materialize sort keys for Numpy sort order from the sort op's
// operand.
HloInstruction* AddNumpySortKey(HloInstruction* operand) {
  Shape value_shape = operand->shape();
  Shape key_shape = ShapeUtil::ChangeElementType(value_shape, U16);
  Shape pred_shape = ShapeUtil::ChangeElementType(value_shape, PRED);
  // Canonicalize zeros, i.e. replace -0 with +0.
  HloInstruction* const_zero = operand->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(BF16)));
  HloInstruction* broadcasted_zero = operand->AddInstruction(
      HloInstruction::CreateBroadcast(value_shape, const_zero, {}));
  HloInstruction* is_zero =
      operand->AddInstruction(HloInstruction::CreateCompare(
          pred_shape, operand, broadcasted_zero, ComparisonDirection::kEq));
  HloInstruction* canonicalized_zeros =
      operand->AddInstruction(HloInstruction::CreateTernary(
          value_shape, HloOpcode::kSelect, is_zero, broadcasted_zero, operand));
  // Canonicalize NaNs, i.e. replace -NaN with NaN.
  HloInstruction* const_nan = operand->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::NanValue(BF16).value()));
  HloInstruction* broadcasted_nan = operand->AddInstruction(
      HloInstruction::CreateBroadcast(value_shape, const_nan, {}));
  // Only NaNs are not equal to themselves.
  HloInstruction* is_nan =
      operand->AddInstruction(HloInstruction::CreateCompare(
          pred_shape, operand, operand, ComparisonDirection::kNe));
  HloInstruction* canonicalized_nans = operand->AddInstruction(
      HloInstruction::CreateTernary(value_shape, HloOpcode::kSelect, is_nan,
                                    broadcasted_nan, canonicalized_zeros));
  // To convert the input values into a radix-sortable bitwise representation,
  // the following transformations take place prior to sorting:
  // * For positive floating point values, the sign bit is inverted.
  // * For negative floating point values, the full key is inverted.
  HloInstruction* is_negative =
      operand->AddInstruction(HloInstruction::CreateCompare(
          pred_shape, canonicalized_nans, broadcasted_zero,
          ComparisonDirection::kLt));
  HloInstruction* bitcast_convert = operand->AddInstruction(
      HloInstruction::CreateBitcastConvert(key_shape, canonicalized_nans));
  HloInstruction* constant_8000 = operand->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint16_t>(32768)));
  HloInstruction* broadcasted_8000 = operand->AddInstruction(
      HloInstruction::CreateBroadcast(key_shape, constant_8000, {}));
  HloInstruction* inverted_sign =
      operand->AddInstruction(HloInstruction::CreateBinary(
          key_shape, HloOpcode::kXor, broadcasted_8000, bitcast_convert));
  HloInstruction* constant_ffff = operand->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint16_t>(65535)));
  HloInstruction* broadcasted_ffff = operand->AddInstruction(
      HloInstruction::CreateBroadcast(key_shape, constant_ffff, {}));
  HloInstruction* inverted_bits =
      operand->AddInstruction(HloInstruction::CreateBinary(
          key_shape, HloOpcode::kXor, broadcasted_ffff, bitcast_convert));
  HloInstruction* sort_keys = operand->AddInstruction(
      HloInstruction::CreateTernary(key_shape, HloOpcode::kSelect, is_negative,
                                    inverted_bits, inverted_sign));
  return sort_keys;
}

}  // namespace

// Rewrites a single sort instruction with a custom call.
absl::StatusOr<bool> SortRewriter::RunOnInstruction(
    HloSortInstruction* sort_op) {
  // Get the sort tensor index and direction.
  SortComputationAnalysis sort_analysis = AnalyzeSortOp(*sort_op).value();

  // Get scratch size requirements from CUB.
  const Shape& operand_shape = sort_op->operand(0)->shape();
  int64_t batch_size = Product(operand_shape.dimensions()) /
                       operand_shape.dimensions(sort_op->sort_dimension());

  TF_ASSIGN_OR_RETURN(auto runner, CreateRunner(sort_analysis));
  TF_ASSIGN_OR_RETURN(
      int64_t scratch_size,
      runner->GetScratchSize(Product(operand_shape.dimensions()), batch_size));

  // Align and increase scratch size to fit the offsets.
  if (batch_size > 1) {
    scratch_size += sizeof(int) - scratch_size % sizeof(int);
    scratch_size += (batch_size + 1) * sizeof(int);
  }

  // Values are only present if sorting a pair of tensors.
  HloInstruction* keys;
  HloInstruction* values = nullptr;
  bool sorting_pairs = sort_op->operand_count() == 2;

  keys = sort_op->mutable_operand(sort_analysis.key_operand);
  int value_index = 1 - sort_analysis.key_operand;
  if (sorting_pairs) {
    values = sort_op->mutable_operand(value_index);
  }
  // For sorting in Numpy order, materialize synthetic keys and treat the
  // original input as values.
  if (sort_analysis.sort_order == SortOrderType::kNumpyOrder) {
    sorting_pairs = true;
    keys = AddNumpySortKey(sort_op->mutable_operand(sort_analysis.key_operand));
    values = sort_op->mutable_operand(sort_analysis.key_operand);
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
      sort_op->AddInstruction(HloInstruction::CreateCustomCall(
          call_shape, absl::MakeSpan(operands), kCubDeviceRadixSortTarget));

  xla::SortOptions backend_config;
  backend_config.set_descending(sort_analysis.descending);
  TF_RETURN_IF_ERROR(custom_call->set_backend_config(backend_config));

  // Build the replacement instruction.
  HloInstruction* replacement;
  if (!sorting_pairs) {
    replacement =
        sort_op->parent()->AddInstruction(HloInstruction::CreateGetTupleElement(
            sort_op->shape(), custom_call, 0));
  } else if (sort_analysis.sort_order == SortOrderType::kNumpyOrder) {
    // Discard the synthetic keys generated for sorting in Numpy order.
    replacement = sort_op->AddInstruction(
        HloInstruction::CreateGetTupleElement(values->shape(), custom_call, 1));
  } else {
    replacement = UnpackResultPair(sort_op, custom_call,
                                   /*swap=*/sort_analysis.key_operand == 1);
  }

  // Replace sort operation with custom call followed by GTE.
  TF_RETURN_IF_ERROR(
      sort_op->parent()->ReplaceInstruction(sort_op, replacement));
  return true;
}

// Rewrites the sorts in the given computation into calls to CUB.
absl::StatusOr<bool> SortRewriter::RunOnComputation(
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
absl::StatusOr<bool> SortRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(3, "SortRewriter::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  XLA_VLOG_LINES(3, "SortRewriter::Run(), after:\n" + module->ToString());
  return changed;
}

bool IsCubCompatibleSort(const HloSortInstruction* sort_op) {
  VLOG(1) << "Sort instruction: " << sort_op->name();
  if (sort_op->operand_count() != 1 && sort_op->operand_count() != 2) {
    VLOG(2) << "Unsupported operand count: " << sort_op->operand_count();
    return false;
  }

  const Shape& operand_shape = sort_op->operand(0)->shape();
  if (sort_op->sort_dimension() != operand_shape.rank() - 1) {
    VLOG(2) << "Sort dimension should be the minor one";
    return false;
  }
  if (Product(operand_shape.dimensions()) < SortRewriter::SortSizeThreshold()) {
    VLOG(2) << "Tensor shape size is too small to see an improvement";
    return false;
  }

  auto sort_analysis = AnalyzeSortOp(*sort_op);
  if (!sort_analysis.has_value()) {
    VLOG(2) << "Only simple compare computations are supported";
    return false;
  }
  if (!CreateRunner(*sort_analysis).ok()) {
    VLOG(2) << "Unsupported operand types (no compiled CUB kernels): "
            << PrimitiveType_Name(sort_analysis->key_type) << " "
            << (sort_analysis->value_type.has_value()
                    ? PrimitiveType_Name(sort_analysis->value_type.value())
                    : "");
    return false;
  }
  VLOG(2) << "Sort operation is compatible";
  return true;
}

}  // namespace gpu
}  // namespace xla
