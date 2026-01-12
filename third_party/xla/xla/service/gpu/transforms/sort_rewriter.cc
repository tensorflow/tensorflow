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
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
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

namespace xla::gpu {
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

// Returns whether the argsort operation exceeds the memory threshold for CUB
// sort rewrite.
// The packing for CUB numpy order argsort consumes ~2x the memory of the input
// because it creates packed values that combine keys and indices.
bool IsNumpySortMemoryExpensive(const Shape& shape, PrimitiveType key_type,
                                PrimitiveType value_type) {
  const int64_t num_elements = ShapeUtil::ElementsIn(shape);
  const int64_t memory_increase_bytes =
      num_elements * (primitive_util::ByteWidth(key_type) +
                      primitive_util::ByteWidth(value_type));

  // Threshold is 2GB.
  // Note: We can consider making this configurable via a flag in the future.
  if (memory_increase_bytes >= 2LL * 1024 * 1024 * 1024) {
    VLOG(2) << "Sort memory increase (" << memory_increase_bytes
            << " bytes) exceeds the threshold for Numpy order argsort rewrite "
               "(2GB).";
    return true;
  }
  return false;
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
    // Sorting a pair of input tensors is supported via key packing if the key
    // is F16, BF16 or F32 and the value is S16 or S32.
    if (sort_op.operand_count() == 2) {
      // TODO: b/470413500 - add F8 types support.
      if ((sort_key_type != F32 && sort_key_type != F16 &&
           sort_key_type != BF16) ||
          (sort_value_type != S32 && sort_value_type != S16)) {
        return std::nullopt;
      }
      int total_bits = primitive_util::BitWidth(sort_key_type) +
                       primitive_util::BitWidth(sort_value_type.value());
      sort_key_type = primitive_util::UnsignedIntegralTypeForBitWidth(
          primitive_util::BitWidth(sort_key_type));
      sort_value_type = primitive_util::UnsignedIntegralTypeForBitWidth(
          total_bits <= 32 ? 32 : 64);

      if (IsNumpySortMemoryExpensive(sort_op.operand(0)->shape(), sort_key_type,
                                     *sort_value_type)) {
        return std::nullopt;
      }
    } else if (sort_op.operand_count() == 1) {
      // Cub cannot sort the original keys directly, hence treat them as values
      // in a key-value pair sort.
      sort_value_type = sort_key_type;
      // The synthetic keys used for sorting are unsigned integers.
      sort_key_type = primitive_util::UnsignedIntegralTypeForBitWidth(
          primitive_util::BitWidth(sort_key_type));
    } else {
      return std::nullopt;
    }
  }
  return SortComputationAnalysis{
      sort_analysis->key_operand, sort_analysis->descending,
      sort_analysis->sort_order, sort_key_type, sort_value_type};
}

// Returns whether the sort operation is supported by CUB.
bool AreOperandTypesSupportedByCub(
    const SortComputationAnalysis& sort_analysis) {
  PrimitiveType key_type = sort_analysis.key_type;
  std::optional<PrimitiveType> value_type = sort_analysis.value_type;
  if (!value_type.has_value()) {
    switch (key_type) {
      case BF16:
      case F16:
      case F32:
      case F64:
      case S8:
      case S16:
      case S32:
      case S64:
      case U8:
      case U16:
      case U32:
      case U64:
        return true;
      default:
        return false;
    }
  }
  auto value_bitwidth = primitive_util::BitWidth(*value_type);
  switch (key_type) {
    case U8:
    case U16:
    case U32:
    case U64:
    case F32:
      return value_bitwidth == 16 || value_bitwidth == 32 ||
             value_bitwidth == 64;
    case S32:
      return value_bitwidth == 32;
    default:
      return false;
  }
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

// Packs keys and values for argsort with Numpy order.
// We pack the original key (casted to unsigned) and the value into a single
// packed pair. The packed pair will be the second operand of
// the sort (the payload).
// PackedPair = (OriginalKey << ValueBitWidth) | Value
std::pair<HloInstruction*, HloInstruction*> PackNumpySortPairs(
    HloSortInstruction* sort_op, HloInstruction* original_keys,
    HloInstruction* values, const SortComputationAnalysis& sort_analysis) {
  PrimitiveType original_key_type = original_keys->shape().element_type();
  PrimitiveType key_unsigned_type =
      primitive_util::UnsignedIntegralTypeForBitWidth(
          primitive_util::BitWidth(original_key_type));
  // 1. Synthesize Keys (F32 -> U32, F16/BF16 -> U16)
  HloInstruction* synth_keys =
      AddNumpySortKey(original_keys, key_unsigned_type, original_key_type);

  // 2. Values
  PrimitiveType value_type = values->shape().element_type();
  PrimitiveType value_unsigned_type =
      primitive_util::UnsignedIntegralTypeForBitWidth(
          primitive_util::BitWidth(value_type));
  HloInstruction* values_unsigned =
      sort_op->AddInstruction(HloInstruction::CreateBitcastConvert(
          ShapeUtil::ChangeElementType(values->shape(), value_unsigned_type),
          values));
  if (sort_analysis.descending) {
    values_unsigned = sort_op->AddInstruction(HloInstruction::CreateUnary(
        values_unsigned->shape(), HloOpcode::kNot, values_unsigned));
  }

  // 3. Original Keys (as Unsigned Key Type)
  HloInstruction* original_keys_unsigned =
      sort_op->AddInstruction(HloInstruction::CreateBitcastConvert(
          ShapeUtil::ChangeElementType(original_keys->shape(),
                                       key_unsigned_type),
          original_keys));

  // 4. Pack Pair: (OriginalKey << ValueBitWidth) | Value
  int total_bits = primitive_util::BitWidth(original_key_type) +
                   primitive_util::BitWidth(value_type);
  PrimitiveType packed_type = total_bits <= 32 ? U32 : U64;
  Shape packed_shape =
      ShapeUtil::ChangeElementType(synth_keys->shape(), packed_type);

  HloInstruction* values_packed = sort_op->AddInstruction(
      HloInstruction::CreateConvert(packed_shape, values_unsigned));
  HloInstruction* orig_keys_packed = sort_op->AddInstruction(
      HloInstruction::CreateConvert(packed_shape, original_keys_unsigned));

  int shift_amount = primitive_util::BitWidth(value_type);
  HloInstruction* constant_shift = sort_op->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(
          packed_type, static_cast<uint64_t>(shift_amount))));
  HloInstruction* broadcasted_shift = sort_op->AddInstruction(
      HloInstruction::CreateBroadcast(packed_shape, constant_shift, {}));

  HloInstruction* val_high = sort_op->AddInstruction(
      HloInstruction::CreateBinary(packed_shape, HloOpcode::kShiftLeft,
                                   orig_keys_packed, broadcasted_shift));
  HloInstruction* packed_pairs =
      sort_op->AddInstruction(HloInstruction::CreateBinary(
          packed_shape, HloOpcode::kOr, val_high, values_packed));

  return {synth_keys, packed_pairs};
}

// Unpacks the packed pair from argsort with Numpy order.
// PackedPair = (OriginalKey << ValueBitWidth) | Value
// Returns (OriginalKey, Value) if the key is the first operand,
// otherwise returns (Value, OriginalKey).
HloInstruction* UnpackNumpySortPairs(
    HloSortInstruction* sort_op, HloInstruction* custom_call,
    const SortComputationAnalysis& sort_analysis) {
  Shape packed_shape = custom_call->shape().tuple_shapes(1);
  HloInstruction* packed_pairs = sort_op->AddInstruction(
      HloInstruction::CreateGetTupleElement(packed_shape, custom_call, 1));

  Shape key_shape = sort_op->operand(sort_analysis.key_operand)->shape();
  Shape value_shape = sort_op->operand(1 - sort_analysis.key_operand)->shape();
  PrimitiveType packed_type = packed_shape.element_type();

  int shift_amount = primitive_util::BitWidth(value_shape.element_type());
  HloInstruction* constant_shift = sort_op->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(
          packed_type, static_cast<uint64_t>(shift_amount))));
  HloInstruction* broadcasted_shift = sort_op->AddInstruction(
      HloInstruction::CreateBroadcast(packed_shape, constant_shift, {}));

  // Extract Original Keys
  HloInstruction* original_keys_packed = sort_op->AddInstruction(
      HloInstruction::CreateBinary(packed_shape, HloOpcode::kShiftRightLogical,
                                   packed_pairs, broadcasted_shift));

  PrimitiveType original_key_type = key_shape.element_type();
  PrimitiveType key_unsigned_type =
      primitive_util::UnsignedIntegralTypeForBitWidth(
          primitive_util::BitWidth(original_key_type));

  HloInstruction* original_keys_unsigned =
      sort_op->AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::ChangeElementType(original_keys_packed->shape(),
                                       key_unsigned_type),
          original_keys_packed));
  HloInstruction* original_keys = sort_op->AddInstruction(
      HloInstruction::CreateBitcastConvert(key_shape, original_keys_unsigned));

  // Extract Values
  PrimitiveType value_type = value_shape.element_type();
  PrimitiveType value_unsigned_type =
      primitive_util::UnsignedIntegralTypeForBitWidth(
          primitive_util::BitWidth(value_type));
  HloInstruction* values_unsigned =
      sort_op->AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::ChangeElementType(packed_shape, value_unsigned_type),
          packed_pairs));
  if (sort_analysis.descending) {
    values_unsigned = sort_op->AddInstruction(HloInstruction::CreateUnary(
        values_unsigned->shape(), HloOpcode::kNot, values_unsigned));
  }
  HloInstruction* values = sort_op->AddInstruction(
      HloInstruction::CreateBitcastConvert(value_shape, values_unsigned));

  if (sort_analysis.key_operand == 0) {
    return sort_op->AddInstruction(
        HloInstruction::CreateTuple({original_keys, values}));
  }
  return sort_op->AddInstruction(
      HloInstruction::CreateTuple({values, original_keys}));
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
    if (auto* cuda_cc = device_description.gpu_compute_capability()
                            .cuda_compute_capability()) {
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
  if (!AreOperandTypesSupportedByCub(*sort_analysis)) {
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
  if (sort_analysis.sort_order == SortOrderType::kNumpyOrder &&
      sort_op->operand_count() == 1) {
    sorting_pairs = true;
    keys = AddNumpySortKey(sort_op->mutable_operand(sort_analysis.key_operand),
                           sort_analysis.key_type,
                           sort_analysis.value_type.value());
    values = sort_op->mutable_operand(sort_analysis.key_operand);
  }

  // Support for argsort (sort pairs) with Numpy order.
  // We pack the original key and the value into a single
  // packed pair. The packed pair will be the second operand of the sort.
  if (sort_analysis.sort_order == SortOrderType::kNumpyOrder &&
      sort_op->operand_count() == 2) {
    std::pair<HloInstruction*, HloInstruction*> packed = PackNumpySortPairs(
        sort_op, sort_op->mutable_operand(sort_analysis.key_operand),
        sort_op->mutable_operand(1 - sort_analysis.key_operand), sort_analysis);
    keys = packed.first;
    values = packed.second;
  }

  // Build the resulting shape for the custom call.
  std::vector<Shape> shapes{keys->shape()};
  std::vector<HloInstruction*> operands{keys};
  if (values != nullptr) {
    shapes.push_back(values->shape());
    operands.push_back(values);
  }
  // The last shape corresponds to the scratch buffer. In this pass we put 1 as
  // the scratch size, but later the actual size will be set by the
  // AssignCubScratchSize pass.
  shapes.push_back(ShapeUtil::MakeShape(U8, {/*scratch_size=*/1}));
  Shape call_shape = ShapeUtil::MakeTupleShape(absl::MakeSpan(shapes));

  // Build the custom call instruction.
  HloInstruction* custom_call =
      sort_op->AddInstruction(HloInstruction::CreateCustomCall(
          call_shape, absl::MakeSpan(operands),
          kCubDeviceRadixSortUnassignedScratchSizeTarget));

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
    if (sort_op->operand_count() == 1) {
      // Discard the synthetic keys generated for sorting in Numpy order.
      replacement =
          sort_op->AddInstruction(HloInstruction::CreateGetTupleElement(
              values->shape(), custom_call, 1));
    } else {
      replacement = UnpackNumpySortPairs(sort_op, custom_call, sort_analysis);
    }
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
absl::StatusOr<bool> SortRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(3, "SortRewriter::RunImpl(), before:\n" + module->ToString());
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  XLA_VLOG_LINES(3, "SortRewriter::RunImpl(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla::gpu
