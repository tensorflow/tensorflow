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
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/cub_sort_thunk.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

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
    if (sort_key_type != BF16 && sort_key_type != F16 && sort_key_type != F32 &&
        sort_key_type != F64) {
      return std::nullopt;
    }
    // Sorting a pair of input tensors is not supported. The keys to sort on
    // will be generated synthetically.
    if (sort_op.operand_count() != 1) {
      return std::nullopt;
    }
    // Cub cannot sort the original keys directly, hence treat them as values in
    // a key-value pair sort.
    sort_value_type = sort_key_type;
    // The synthetic keys used for sorting are unsigned integers.
    sort_key_type = primitive_util::UnsignedIntegralTypeForBitWidth(
        primitive_util::BitWidth(sort_key_type));
  }
  return SortComputationAnalysis{
      sort_analysis->key_operand, sort_analysis->descending,
      sort_analysis->sort_order, sort_key_type, sort_value_type};
}

// Create runner for CUB sort operation.
absl::StatusOr<std::unique_ptr<CubSortRunnerInterface>> CreateRunner(
    const SortComputationAnalysis& sort_analysis,
    absl::string_view platform_name) {
  return CubSortRunnerInterface::Create(
      sort_analysis.key_type, sort_analysis.value_type, platform_name);
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
HloInstruction* AddNumpySortKey(HloInstruction* operand, PrimitiveType key_type,
                                PrimitiveType value_type) {
  Shape value_shape = operand->shape();
  int64_t bit_width = primitive_util::BitWidth(value_type);
  Shape key_shape = ShapeUtil::ChangeElementType(value_shape, key_type);
  Shape pred_shape = ShapeUtil::ChangeElementType(value_shape, PRED);
  // Canonicalize zeros, i.e. replace -0 with +0.
  HloInstruction* const_zero = operand->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(value_type)));
  HloInstruction* broadcasted_zero = operand->AddInstruction(
      HloInstruction::CreateBroadcast(value_shape, const_zero, {}));
  HloInstruction* is_zero =
      operand->AddInstruction(HloInstruction::CreateCompare(
          pred_shape, operand, broadcasted_zero, ComparisonDirection::kEq));
  HloInstruction* canonicalized_zeros =
      operand->AddInstruction(HloInstruction::CreateTernary(
          value_shape, HloOpcode::kSelect, is_zero, broadcasted_zero, operand));
  // Canonicalize NaNs, i.e. replace -NaN with NaN.
  HloInstruction* const_nan =
      operand->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::NanValue(value_type).value()));
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
  // * For negative floating point values, the full key is inverted (kNot op).
  HloInstruction* is_negative =
      operand->AddInstruction(HloInstruction::CreateCompare(
          pred_shape, canonicalized_nans, broadcasted_zero,
          ComparisonDirection::kLt));
  HloInstruction* bitcast_convert = operand->AddInstruction(
      HloInstruction::CreateBitcastConvert(key_shape, canonicalized_nans));
  HloInstruction* constant_8000 =
      operand->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0(key_type, pow(2, bit_width - 1))));
  HloInstruction* broadcasted_8000 = operand->AddInstruction(
      HloInstruction::CreateBroadcast(key_shape, constant_8000, {}));
  HloInstruction* inverted_sign =
      operand->AddInstruction(HloInstruction::CreateBinary(
          key_shape, HloOpcode::kXor, broadcasted_8000, bitcast_convert));
  HloInstruction* inverted_bits = operand->AddInstruction(
      HloInstruction::CreateUnary(key_shape, HloOpcode::kNot, bitcast_convert));
  HloInstruction* sort_keys = operand->AddInstruction(
      HloInstruction::CreateTernary(key_shape, HloOpcode::kSelect, is_negative,
                                    inverted_bits, inverted_sign));
  return sort_keys;
}

bool IsCubSortFasterOnH100(int bitwidth, int batch_size, int num_elements,
                           int sm_count) {
  // The numbers below are based on extensive benchmarks: see
  // b/407689882#comment35 and b/410480351 for more details.
  switch (bitwidth) {
    case 8:
      return batch_size == 1 ||
             (num_elements > 1300 && (batch_size > 8 || num_elements < 26000));
    case 16:
      return (batch_size == 1 && num_elements > (1 << 9)) ||
             (batch_size > 12 && num_elements > (1 << 16)) ||
             (batch_size > 14 && num_elements > (1 << 15)) ||
             (batch_size > 16 && num_elements > (1 << 14)) ||
             (batch_size > 18 && num_elements > (1 << 13)) ||
             (batch_size > 33 && num_elements > (1 << 12)) ||
             (batch_size > 66 && num_elements > (1 << 11));
    case 32:
      return (batch_size == 1 && num_elements > 22000) ||
             (batch_size > 26 && num_elements > (1 << 17)) ||
             (batch_size > 31 && num_elements > (1 << 16)) ||
             (batch_size > 38 && num_elements > (1 << 15)) ||
             (batch_size > 44 && num_elements > (1 << 14)) ||
             (batch_size > 52 && num_elements > (1 << 13)) ||
             (batch_size > 88 && batch_size <= sm_count &&
              num_elements > (1 << 12));
    case 64:
      return (batch_size == 1 && num_elements > (1 << 17)) ||
             (batch_size > 55 && num_elements > (1 << 17)) ||
             (batch_size > 70 && num_elements > (1 << 16)) ||
             (batch_size > 92 && num_elements > (1 << 15)) ||
             (((batch_size > 160 && batch_size <= 2 * sm_count) ||
               (batch_size > 354)) &&
              num_elements > (1 << 14));
    default:
      return false;
  }
}

bool IsCubSortFasterOnA100(int bitwidth, int batch_size, int num_elements,
                           int sm_count) {
  // The numbers below are based on extensive benchmarks: see
  // b/410480351#comment4 for more details.
  switch (bitwidth) {
    case 8:
      return batch_size == 1 ||
             (num_elements > 1000 && (batch_size > 5 || num_elements < 43000));
    case 16:
      return (batch_size == 1 && num_elements > (1 << 16)) ||
             (batch_size > 9 && num_elements > (1 << 17)) ||
             (batch_size > 13 && num_elements > (1 << 16)) ||
             (batch_size > 13 && num_elements > (1 << 15)) ||
             (batch_size > 13 && num_elements > (1 << 14)) ||
             (batch_size > 13 && num_elements > (1 << 13)) ||
             (batch_size > 27 && num_elements > (1 << 12)) ||
             (batch_size > 54 && num_elements > (1 << 11));
    case 32:
      return (batch_size == 1 && num_elements > (2 << 14)) ||
             (batch_size > 24 && num_elements > (1 << 17)) ||
             (batch_size > 30 && num_elements > (1 << 16)) ||
             (batch_size > 36 && num_elements > (1 << 15)) ||
             (batch_size > 39 && num_elements > (1 << 14)) ||
             (batch_size > 52 && num_elements > (1 << 13)) ||
             (batch_size > 144 && num_elements > (1 << 12));
    case 64:
      return (batch_size == 1 && num_elements > (1 << 16)) ||
             (batch_size > 46 && num_elements > (1 << 17)) ||
             (batch_size > 55 && num_elements > (1 << 16)) ||
             (batch_size > 72 && num_elements > (1 << 15)) ||
             (((batch_size > 138 && batch_size <= 2 * sm_count) ||
               (batch_size > 289)) &&
              num_elements > (1 << 14));
    default:
      return false;
  }
}

// Returns whether a compatible sort should be rewritten based on the current
// sort mode and possibly a heuristic.
bool ShouldRewriteCompatibleSort(se::DeviceDescription device_description,
                                 const HloSortInstruction* sort_op) {
  if (SortRewriter::SortMode() == SortRewriter::Mode::kAlways) {
    return true;
  }

  const Shape& operand_shape = sort_op->operand(0)->shape();
  int num_elements = operand_shape.dimensions().back();
  if (num_elements == 0) {
    return false;
  }

  if (SortRewriter::SortMode() == SortRewriter::Mode::kAuto) {
    if (auto cuda_cc = std::get_if<se::CudaComputeCapability>(
            &device_description.gpu_compute_capability())) {
      int bitwidth = primitive_util::BitWidth(operand_shape.element_type());
      int batch_size = Product(operand_shape.dimensions()) / num_elements;

      if (cuda_cc->IsBlackwell()) {
        // TODO(b/410480351): Verify that the H100 heuristic also works well for
        // Blackwell or implement a custom heuristic.
        return IsCubSortFasterOnH100(bitwidth, batch_size, num_elements,
                                     device_description.core_count());
      }
      if (cuda_cc->IsHopper()) {
        return IsCubSortFasterOnH100(bitwidth, batch_size, num_elements,
                                     device_description.core_count());
      }
      if (cuda_cc->IsAmpere()) {
        return IsCubSortFasterOnA100(bitwidth, batch_size, num_elements,
                                     device_description.core_count());
      }
    }
  }

  // TODO(b/410480351): The default heuristic below is pretty bad in the general
  // case. Run benchmarks on different devices and add a heuristic per device.
  return Product(operand_shape.dimensions()) > 16384;
}

bool IsCubCompatibleSort(const se::DeviceDescription& device_description,
                         const HloSortInstruction* sort_op,
                         absl::string_view platform_name) {
  VLOG(1) << "Sort instruction: " << sort_op->name();
  if (sort_op->operand_count() != 1 && sort_op->operand_count() != 2) {
    VLOG(2) << "Unsupported operand count: " << sort_op->operand_count();
    return false;
  }

  for (const auto& op : sort_op->operands()) {
    if (op->shape().is_dynamic()) {
      VLOG(2) << "Dynamic shape is not supported: " << op->shape().ToString();
      return false;
    }
  }

  const Shape& operand_shape = sort_op->operand(0)->shape();
  if (sort_op->sort_dimension() != operand_shape.dimensions().size() - 1) {
    VLOG(2) << "Sort dimension should be the minor one";
    return false;
  }

  if (!ShouldRewriteCompatibleSort(device_description, sort_op)) {
    VLOG(2) << "Tensor shape and type will not see an improvement.";
    return false;
  }

  auto sort_analysis = AnalyzeSortOp(*sort_op);
  if (!sort_analysis.has_value()) {
    VLOG(2) << "Only simple compare computations are supported";
    return false;
  }
  if (!CreateRunner(*sort_analysis, platform_name).ok()) {
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

  TF_ASSIGN_OR_RETURN(auto runner, CreateRunner(sort_analysis, platform_name_));
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
    keys = AddNumpySortKey(sort_op->mutable_operand(sort_analysis.key_operand),
                           sort_analysis.key_type,
                           sort_analysis.value_type.value());
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
    if (sort != nullptr &&
        IsCubCompatibleSort(device_description_, sort, platform_name_)) {
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

}  // namespace gpu
}  // namespace xla
