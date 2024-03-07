/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/tests/test_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <utility>

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/transfer_manager.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

template <typename FloatT, typename GeneratorT>
void PopulateWithRandomFloatingPointData(Literal* literal,
                                         std::minstd_rand0* engine) {
  std::uniform_real_distribution<GeneratorT> generator(-0.1f, 0.2f);
  for (FloatT& value : literal->data<FloatT>()) {
    value = static_cast<FloatT>(generator(*engine));
  }
}

// Populates a floating point literal with random floating points sampled from a
// uniform-log distribution spanning approximately the entire range of the
// representable floating point.
template <typename FloatT>
void PopulateWithRandomFullRangeFloatingPointData(Literal* literal,
                                                  std::minstd_rand0* engine) {
  constexpr float kSpecialValueProbability = 1e-6;
  constexpr float kSpecialValues[] = {+0.F,
                                      -0.F,
                                      1.F,
                                      -1.F,
                                      std::numeric_limits<float>::infinity(),
                                      -std::numeric_limits<float>::infinity()};
  constexpr int kNumSpecialValues = sizeof(kSpecialValues) / sizeof(float);
  std::uniform_real_distribution<float> special_value_gen(0, 1);

  // Generates floating points with a log-uniform distribution. This causes the
  // exponent of the floating point to have a uniform distribution.
  const int min_exp = std::numeric_limits<FloatT>::min_exponent;
  const int max_exp = std::numeric_limits<FloatT>::max_exponent;
  std::uniform_real_distribution<double> generator(min_exp - 1, max_exp - 1);

  for (FloatT& value : literal->data<FloatT>()) {
    // Each special value has a kSpecialValueProbability chance to be generated
    // instead of sampling using the normal distributions.
    if (special_value_gen(*engine) <
        kSpecialValueProbability * kNumSpecialValues) {
      value =
          static_cast<FloatT>(kSpecialValues[(*engine)() % kNumSpecialValues]);
    } else {
      float sign = ((*engine)() % 2 == 0) ? 1 : -1;
      value = static_cast<FloatT>(pow(2, generator(*engine)) * sign);
    }
  }
}

template <typename FloatT>
void PopulateWithIntNext(Literal* literal) {
  using BitRepT = UnsignedIntegerTypeForSizeType<sizeof(FloatT)>;
  // Duplicates may be generated if we don't have enough bits.
  // Skip bfloat16 and float32 subnormals.
  const FloatT kFirstValue =
      std::is_same_v<FloatT, bfloat16> || sizeof(FloatT) >= sizeof(float)
          ? std::numeric_limits<FloatT>::min()
          : std::numeric_limits<FloatT>::denorm_min();
  // `current` keeps track of the next value we need to populate.
  auto current = literal->data<FloatT>().begin();
  auto end = literal->data<FloatT>().end();
  // `sign` keeps track of the sign of the next value.
  bool sign = false;
  while (current != end) {
    // We start populating values at zero and increase magnitude from there.
    *current = sign ? static_cast<FloatT>(-0.0f) : static_cast<FloatT>(0.0f);
    current++;
    // The next value is either the smallest denormal or normal.
    auto value = sign ? -kFirstValue : kFirstValue;
    // Fill the array with values of increasing magnitude until we hit a
    // non-finite value.
    while (current != end && Eigen::numext::isfinite(value)) {
      // Populate the value.
      *current = value;
      // Generate the next value by lexicographically increasing the bit
      // representation.
      const BitRepT next_value = Eigen::numext::bit_cast<BitRepT>(value) + 1;
      value = Eigen::numext::bit_cast<FloatT>(next_value);
      current++;
    }
    // We ran out of finite values, flip the sign and begin again.
    sign = !sign;
  }
}

template <typename FloatT>
void PopulateWithNoDuplicateData(Literal* literal, std::minstd_rand0* engine) {
  PopulateWithIntNext<FloatT>(literal);
  std::shuffle(literal->data<FloatT>().begin(), literal->data<FloatT>().end(),
               *engine);
}

template <typename FloatT>
void PopulateWithFloatingPointData(
    Literal* literal, std::minstd_rand0* engine, bool no_duplicates,
    bool use_large_range, std::optional<int64_t> max_bits_of_precision) {
  using ComputeT =
      std::conditional_t<sizeof(FloatT) < sizeof(float), float, FloatT>;
  CHECK(engine != nullptr);
  CHECK_EQ(literal->shape().element_type(),
           primitive_util::NativeToPrimitiveType<FloatT>());
  if (max_bits_of_precision.has_value()) {
    CHECK(!use_large_range) << "Cannot set both use_large_range and "
                               "max_bits_of_precision for floating points.";
    CHECK(!no_duplicates) << "Cannot set both no_duplicates and "
                             "max_bits_of_precision for floating points.";
    std::uniform_int_distribution<int64_t> generator(
        -(1 << *max_bits_of_precision), 1 << *max_bits_of_precision);
    for (FloatT& value : literal->data<FloatT>()) {
      int64_t temp = generator(*engine);
      // We want to generate floating point numbers to a fixed precision, while
      // keeping them between -1 and 1. This preserves their bits of precision
      // while keeping the numbers small.
      value = static_cast<FloatT>(temp * pow(2, -ceil(log2(abs(temp)))));
    }
  } else if (no_duplicates) {
    PopulateWithNoDuplicateData<FloatT>(literal, engine);
  } else if (use_large_range) {
    PopulateWithRandomFullRangeFloatingPointData<FloatT>(literal, engine);
  } else {
    PopulateWithRandomFloatingPointData<FloatT, ComputeT>(literal, engine);
  }
}

template <typename ComplexT>
void PopulateWithComplexData(Literal* result, std::minstd_rand0* engine,
                             bool no_duplicates, bool use_large_range) {
  using InnerFloatT = typename ComplexT::value_type;
  CHECK(engine != nullptr);
  CHECK_EQ(result->shape().element_type(),
           primitive_util::NativeToPrimitiveType<ComplexT>());
  Shape floating_point_shape = ShapeUtil::ChangeElementType(
      result->shape(), primitive_util::NativeToPrimitiveType<InnerFloatT>());
  Literal real_lit(floating_point_shape);
  Literal imaginary_lit(floating_point_shape);

  PopulateWithFloatingPointData<InnerFloatT>(
      &real_lit, engine, no_duplicates, use_large_range,
      /*max_bits_of_precision=*/std::nullopt);
  PopulateWithFloatingPointData<InnerFloatT>(
      &imaginary_lit, engine, no_duplicates, use_large_range,
      /*max_bits_of_precision=*/std::nullopt);

  absl::Span<const InnerFloatT> real_data = real_lit.data<InnerFloatT>();
  absl::Span<const InnerFloatT> imaginary_data =
      imaginary_lit.data<InnerFloatT>();
  absl::Span<ComplexT> result_data = result->data<ComplexT>();
  for (int i = 0; i < real_lit.data<InnerFloatT>().size(); i++) {
    result_data[i] = ComplexT(real_data[i], imaginary_data[i]);
  }
}

// uniform_int_distribution is not defined for 8-bit integers.
// Use 'short' for those types.
template <typename IntT>
using RngT = std::conditional_t<
    sizeof(IntT) < sizeof(uint16_t),
    std::conditional_t<std::numeric_limits<IntT>::is_signed, int16_t, uint16_t>,
    IntT>;

template <typename IntT>
void PopulateWithRandomIntegralDataWithBounds(Literal* literal,
                                              std::minstd_rand0* engine,
                                              bool no_duplicates, IntT min,
                                              IntT max) {
  CHECK(engine != nullptr);
  CHECK_EQ(literal->shape().element_type(),
           primitive_util::NativeToPrimitiveType<IntT>());
  if (no_duplicates &&
      ShapeUtil::ElementsIn(literal->shape()) < static_cast<int64_t>(max)) {
    std::iota(literal->data<IntT>().begin(), literal->data<IntT>().end(),
              static_cast<IntT>(0));
    std::shuffle(literal->data<IntT>().begin(), literal->data<IntT>().end(),
                 *engine);
  } else {
    std::uniform_int_distribution<RngT<IntT>> generator(
        static_cast<RngT<IntT>>(min), static_cast<RngT<IntT>>(max));
    for (IntT& value : literal->data<IntT>()) {
      value = static_cast<IntT>(generator(*engine));
    }
  }
}

// Similar to MakeFakeLiteral but takes a random number generator engine to
// enable reusing the engine across randomly generated literals.
// 'limit' is a optional pair that contains the min and the max values to be
// sample for integers (integer format only).
// 'is_sorted' sorts the sample data for integers (integer format only).
// 'no_duplicates' indicates that there should be no duplicate values in each
// generated array. This is uniqueness is best-effort only. Some types
// (half and bfloat16) are not supported and uniqueness cannot be guaranteed if
// the number of elements exceeds the number of different values supported by
// the type. (floating point format only)
// 'use_large_range' indicates the sampled data is from the full range of the
// floating point format. (floating point format only)
// 'max_bits_of_precision' sets the data to have the given number of bits or
// less (integer or floating point formats only).
absl::StatusOr<Literal> MakeFakeLiteralInternal(
    const Shape& shape, std::minstd_rand0* engine,
    std::optional<std::pair<int64_t, int64_t>> limit, bool is_sorted,
    bool no_duplicates, bool use_large_range,
    std::optional<int64_t> max_bits_of_precision) {
  if (shape.IsTuple()) {
    std::vector<Literal> elements;
    const auto& shape_tuple_shapes = shape.tuple_shapes();
    elements.reserve(shape_tuple_shapes.size());
    for (const Shape& element_shape : shape_tuple_shapes) {
      TF_ASSIGN_OR_RETURN(
          Literal element,
          MakeFakeLiteralInternal(element_shape, engine, limit, is_sorted,
                                  no_duplicates, use_large_range,
                                  max_bits_of_precision));
      elements.push_back(std::move(element));
    }
    return LiteralUtil::MakeTupleOwned(std::move(elements));
  }
  if (engine == nullptr) {
    return Literal::CreateFromShape(shape);
  }
  // Clear tiles/element size in shape's layout before using it for creating
  // literal.
  Shape new_shape = shape;
  new_shape.mutable_layout()->clear_tiles();
  new_shape.mutable_layout()->set_tail_padding_alignment_in_elements(1);
  new_shape.mutable_layout()->set_element_size_in_bits(0);
  Literal literal(new_shape);

  TF_RETURN_IF_ERROR(primitive_util::PrimitiveTypeSwitch<Status>(
      [&](auto primitive_type_constant) -> Status {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          if constexpr (primitive_util::IsFloatingPointType(
                            primitive_type_constant)) {
            PopulateWithFloatingPointData<NativeT>(
                &literal, engine, no_duplicates, use_large_range,
                max_bits_of_precision);
            return OkStatus();
          }
          if constexpr (primitive_type_constant == PRED) {
            std::uniform_int_distribution<int> generator(0, 1);
            TF_CHECK_OK(literal.Populate<bool>(
                [&](absl::Span<const int64_t> /*indices*/) {
                  return generator(*engine);
                }));
            return OkStatus();
          }
          if constexpr (primitive_util::IsIntegralType(
                            primitive_type_constant)) {
            NativeT max = std::numeric_limits<NativeT>::max();
            NativeT min = std::numeric_limits<NativeT>::lowest();
            if (limit.has_value()) {
              max = static_cast<NativeT>(limit->second);
              min = static_cast<NativeT>(limit->first);
            }
            if (max_bits_of_precision.has_value()) {
              max = std::min(max,
                             static_cast<NativeT>(1 << *max_bits_of_precision));
              if (primitive_util::IsSignedIntegralType(
                      primitive_type_constant)) {
                min = std::max(
                    min, static_cast<NativeT>(-(1 << *max_bits_of_precision)));
              }
            }
            PopulateWithRandomIntegralDataWithBounds<NativeT>(
                &literal, engine, /*no_duplicate*/ no_duplicates, min, max);
            if (is_sorted) {
              std::sort(literal.data<NativeT>().begin(),
                        literal.data<NativeT>().end());
            }
            return OkStatus();
          }
          if constexpr (primitive_util::IsComplexType(
                            primitive_type_constant)) {
            PopulateWithComplexData<NativeT>(&literal, engine, no_duplicates,
                                             use_large_range);
            return OkStatus();
          }
        }
        return Unimplemented(
            "Unsupported type for fake random literal generation with bounds: "
            "%s",
            ShapeUtil::HumanString(shape));
      },
      shape.element_type()));
  return std::move(literal);
}

enum class ConstantType { kUnknown, kZero, kOne };

// Return the constant type required by this computation, if known.
ConstantType GetInitValue(const HloComputation& computation) {
  // TODO(b/77635120): Add init values, for min, max, and their arg variants.
  const HloInstruction* const root = computation.root_instruction();
  if (computation.num_parameters() != 2 || root->operand_count() != 2 ||
      root->operand(0)->opcode() != HloOpcode::kParameter ||
      root->operand(1)->opcode() != HloOpcode::kParameter ||
      root->operand(0) == root->operand(1)) {
    return ConstantType::kUnknown;
  }

  switch (root->opcode()) {
    case HloOpcode::kAdd:
      return ConstantType::kZero;
    case HloOpcode::kMultiply:
      return ConstantType::kOne;
    default:
      return ConstantType::kUnknown;
  }
}

// Reduce, ReduceWindow, and SelectAndScatter ops may need a non-random
// initialization value.
bool NeedsInitValue(const HloUse& use) {
  const HloInstruction* const instruction = use.instruction;
  const HloOpcode opcode = instruction->opcode();
  const int64_t op_num = use.operand_number;
  return ((opcode == HloOpcode::kReduceWindow && op_num == 1) ||
          (opcode == HloOpcode::kSelectAndScatter && op_num == 2) ||
          (opcode == HloOpcode::kReduce &&
           op_num >= instruction->operand_count() / 2));
}

// Generate random values that are constrained to the input_shape minus the
// output_shape so as not to produce wrapping slices, for instance.
Literal MakeRandomIndex(int64_t index_bound, std::minstd_rand0* engine) {
  std::uniform_int_distribution<int32_t> generator(0, index_bound);
  return LiteralUtil::CreateR0<int32_t>(generator(*engine));
}

// Returns true if `dest' is reachable from `src' through data-formatting and
// custom call instructions within the same computation.
bool ReachableViaDataFormatting(const HloInstruction* src,
                                const HloInstruction* dest,
                                bool treat_gte_as_data_formatting) {
  if (src == dest) {
    return true;
  }
  switch (dest->opcode()) {
    case HloOpcode::kReshape:
    case HloOpcode::kTranspose:
    case HloOpcode::kCopy:
    case HloOpcode::kSlice:
      break;
    case HloOpcode::kCustomCall:
      if (dest->custom_call_target() == "AssumeGatherIndicesInBound") {
        break;
      }
      return false;
    // TODO(b/249417724): a workaround for tuple param.
    case HloOpcode::kGetTupleElement:
      if (treat_gte_as_data_formatting) {
        break;
      } else {
        return false;
      }
    default:
      return false;
  }
  for (const auto* operand : dest->operands()) {
    if (ReachableViaDataFormatting(src, operand,
                                   treat_gte_as_data_formatting)) {
      return true;
    }
  }
  return false;
}

// Use dataflow analysis on each parameter to see if there are uses that would
// be problematic when generating input data.  Returns the list of
// instructions that correspond to their uses.
//
// Should be paired with the CreateLiteralForConstrainedUses() function below.
std::vector<HloInstruction*> FindConstrainedUses(
    const HloDataflowAnalysis& dataflow, const HloInstruction& param,
    bool treat_gte_as_data_formatting) {
  std::vector<HloInstruction*> constrained_uses;
  for (const auto& pair : dataflow.GetInstructionValueSet(&param)) {
    const HloValue& value = dataflow.GetUniqueValueAt(&param, pair.first);
    for (const HloUse& use : value.GetUses()) {
      HloInstruction* instruction = use.instruction;
      const HloOpcode opcode = instruction->opcode();
      const int64_t op_num = use.operand_number;
      if ((opcode == HloOpcode::kDynamicSlice && op_num >= 1) ||
          (opcode == HloOpcode::kDynamicUpdateSlice && op_num >= 2)) {
        constrained_uses.push_back(instruction);
      } else if ((opcode == HloOpcode::kGather ||
                  opcode == HloOpcode::kScatter) &&
                 op_num == 1) {
        constrained_uses.push_back(instruction);
      } else if (opcode == HloOpcode::kFusion) {
        const HloInstruction* const to_analyze =
            instruction->fused_parameter(op_num);
        auto fused_uses = FindConstrainedUses(dataflow, *to_analyze,
                                              treat_gte_as_data_formatting);
        constrained_uses.insert(constrained_uses.end(), fused_uses.begin(),
                                fused_uses.end());
      } else if (NeedsInitValue(use)) {
        constrained_uses.push_back(instruction);
      } else if (opcode == HloOpcode::kConvert ||
                 opcode == HloOpcode::kReducePrecision) {
        auto converted_uses = FindConstrainedUses(dataflow, *instruction,
                                                  treat_gte_as_data_formatting);
        constrained_uses.insert(constrained_uses.end(), converted_uses.begin(),
                                converted_uses.end());
      } else if (opcode == HloOpcode::kSort &&
                 instruction->operand_count() >= 2 && op_num == 0) {
        // Operand 0 of sort is the array of keys used for key/value
        // (two-operand) kSort instructions. Since sort stability is not
        // guaranteed, constrain keys of key-value sort not to have
        // duplicates, since otherwise the value order may legitimately
        // differ.
        constrained_uses.push_back(instruction);
      }
    }
  }

  for (auto* instruction : param.parent()->instructions()) {
    const HloOpcode opcode = instruction->opcode();
    if (opcode == HloOpcode::kGather || opcode == HloOpcode::kScatter) {
      if (instruction->operand(1) == &param) {
        // Above already covers this case.
        continue;
      }
      if (ReachableViaDataFormatting(&param, instruction->operand(1),
                                     treat_gte_as_data_formatting)) {
        constrained_uses.push_back(instruction);
      }
    }
  }
  return constrained_uses;
}

// Given a parameter, generate a random Literal to use as input if there exist
// no constrained uses in the dataflow graph.  If such constraints exist,
// generate a constrained literal (either bounded in the case of indices, or
// zero in the case of init_values for reductions).
absl::StatusOr<Literal> CreateLiteralForConstrainedUses(
    const absl::Span<HloInstruction* const> constrained_uses,
    const HloInstruction& param, const Shape& param_shape,
    std::minstd_rand0* engine, bool use_large_range,
    std::optional<int64_t> max_bits_of_precision) {
  int64_t index_bound = INT64_MAX;
  bool no_duplicates = false;
  bool needs_constant = false;
  bool needs_sorted_indices = false;
  ConstantType constant_type = ConstantType::kUnknown;
  for (HloInstruction* use : constrained_uses) {
    switch (use->opcode()) {
      case HloOpcode::kDynamicSlice:
      case HloOpcode::kDynamicUpdateSlice: {
        const Shape& indexed_shape = use->operand(0)->shape();
        const Shape& slice_shape = use->opcode() == HloOpcode::kDynamicSlice
                                       ? use->shape()
                                       : use->operand(1)->shape();
        const int64_t first_index =
            Cast<HloDynamicIndexInstruction>(use)->first_index_operand_number();
        for (int64_t operand = first_index; operand < use->operand_count();
             ++operand) {
          if (use->operand(operand) == &param) {
            index_bound = std::min(
                index_bound,
                ShapeUtil::GetDimension(indexed_shape, operand - first_index) -
                    ShapeUtil::GetDimension(slice_shape,
                                            operand - first_index));
          }
        }
        break;
      }
      case HloOpcode::kGather:
      case HloOpcode::kScatter: {
        const Shape& operand_shape = use->operand(0)->shape();
        auto index_map = use->opcode() == HloOpcode::kGather
                             ? use->gather_dimension_numbers().start_index_map()
                             : use->scatter_dimension_numbers()
                                   .scatter_dims_to_operand_dims();
        for (const auto dim_in_operand : index_map) {
          index_bound = std::min(index_bound,
                                 operand_shape.dimensions(dim_in_operand) - 1);
        }
        if (use->opcode() == HloOpcode::kScatter) {
          needs_sorted_indices |=
              Cast<const HloScatterInstruction>(use)->indices_are_sorted();
        } else {
          needs_sorted_indices |=
              Cast<const HloGatherInstruction>(use)->indices_are_sorted();
        }
        break;
      }
      case HloOpcode::kReduce:
      case HloOpcode::kReduceWindow:
        needs_constant = true;
        constant_type = GetInitValue(*use->to_apply());
        break;

      case HloOpcode::kSelectAndScatter:
        needs_constant = true;
        constant_type = GetInitValue(*use->scatter());
        break;

      case HloOpcode::kSort:
        no_duplicates = true;
        break;

      default:
        return Unimplemented(
            "Constrained operand generation not implemented for %s.",
            use->ToString());
    }
  }
  int constraint_count = 0;
  constraint_count += no_duplicates ? 1 : 0;
  constraint_count += (index_bound != INT64_MAX) ? 1 : 0;
  constraint_count += needs_constant ? 1 : 0;
  if (constraint_count > 1) {
    return Unimplemented("Conflicting operand generation constraints.");
  }
  if (index_bound != INT64_MAX) {
    return MakeFakeLiteralInternal(param_shape, engine,
                                   std::pair<int64_t, int64_t>(0, index_bound),
                                   needs_sorted_indices, no_duplicates,
                                   use_large_range, max_bits_of_precision);
  } else if (needs_constant) {
    switch (constant_type) {
      case ConstantType::kZero:
        return LiteralUtil::Zero(param_shape.element_type());
      case ConstantType::kOne:
        return LiteralUtil::One(param_shape.element_type());
      case ConstantType::kUnknown:
        // We want the identity element for the computation, but we don't
        // really know what it is - so any value we generate will be just as
        // wrong.
        return MakeFakeLiteralInternal(
            param_shape, engine, /*limit=*/std::nullopt,
            /*is_sorted=*/needs_sorted_indices,
            /*no_duplicates=*/false, use_large_range, max_bits_of_precision);
    }
  } else {
    return MakeFakeLiteralInternal(param_shape, engine, /*limit=*/std::nullopt,
                                   /*is_sorted=*/needs_sorted_indices,
                                   no_duplicates, use_large_range,
                                   max_bits_of_precision);
  }
}

// Given a module entry parameter, use the dataflow analysis to see if a
// special case literal must be created, or if we can generate fake data.
absl::StatusOr<Literal> MakeConstrainedArgument(
    const HloDataflowAnalysis& dataflow, const HloInstruction& param,
    const Shape& param_shape, std::minstd_rand0* engine, bool use_large_range,
    bool treat_gte_as_data_formatting,
    std::optional<int64_t> max_bits_of_precision) {
  const auto constrained_uses =
      FindConstrainedUses(dataflow, param, treat_gte_as_data_formatting);
  return CreateLiteralForConstrainedUses(constrained_uses, param, param_shape,
                                         engine, use_large_range,
                                         max_bits_of_precision);
}

}  // namespace

absl::StatusOr<Literal> MakeFakeLiteral(const Shape& shape, bool pseudo_random,
                                        bool use_large_range) {
  auto engine = pseudo_random ? std::make_unique<std::minstd_rand0>() : nullptr;
  return MakeFakeLiteralInternal(shape, engine.get(), /*limit=*/std::nullopt,
                                 /*is_sorted=*/false,
                                 /*no_duplicates=*/false, use_large_range,
                                 /*max_bits_of_precision=*/std::nullopt);
}

absl::StatusOr<std::vector<Literal>> MakeFakeArguments(
    const HloModule* module, bool pseudo_random, bool use_large_range,
    bool treat_gte_as_data_formatting,
    std::optional<int64_t> max_bits_of_precision) {
  auto engine = pseudo_random ? std::make_unique<std::minstd_rand0>() : nullptr;
  return MakeFakeArguments(module, engine.get(), use_large_range,
                           treat_gte_as_data_formatting, max_bits_of_precision);
}

absl::StatusOr<std::vector<Literal>> MakeFakeArguments(
    const HloModule* module, std::minstd_rand0* engine, bool use_large_range,
    bool treat_gte_as_data_formatting,
    std::optional<int64_t> max_bits_of_precision) {
  TF_ASSIGN_OR_RETURN(auto dataflow, HloDataflowAnalysis::Run(*module));
  const auto params = module->entry_computation()->parameter_instructions();
  std::vector<Literal> arguments(params.size());
  for (int i = 0; i < params.size(); ++i) {
    const HloModuleConfig& module_config = module->config();
    const Shape& param_shape = (module_config.has_entry_computation_layout() &&
                                module_config.entry_computation_layout()
                                    .parameter_layout(i)
                                    .shape()
                                    .is_static())
                                   ? module_config.entry_computation_layout()
                                         .parameter_layout(i)
                                         .shape()
                                   : params[i]->shape();

    TF_ASSIGN_OR_RETURN(
        arguments[i],
        MakeConstrainedArgument(*dataflow, *params[i], param_shape, engine,
                                use_large_range, treat_gte_as_data_formatting,
                                max_bits_of_precision));
  }
  return std::move(arguments);
}

Status VerifyHloModule(HloModule* const module, bool layout_sensitive,
                       bool allow_mixed_precision) {
  return HloVerifier(/*layout_sensitive=*/layout_sensitive,
                     /*allow_mixed_precision=*/allow_mixed_precision)
      .Run(module)
      .status();
}

std::unique_ptr<HloDotInstruction> CreateCanonicalDot(const Shape& shape,
                                                      HloInstruction* lhs,
                                                      HloInstruction* rhs) {
  CHECK_LE(lhs->shape().rank(), 2);
  CHECK_LE(rhs->shape().rank(), 2);
  PrecisionConfig precision_config;
  precision_config.mutable_operand_precision()->Resize(
      2, PrecisionConfig::DEFAULT);
  DotDimensionNumbers dot_dimension_numbers;
  dot_dimension_numbers.add_lhs_contracting_dimensions(
      lhs->shape().rank() > 1 ? 1 : 0);
  dot_dimension_numbers.add_rhs_contracting_dimensions(0);
  return std::make_unique<HloDotInstruction>(
      shape, lhs, rhs, dot_dimension_numbers, precision_config);
}

bool IsMlirLoweringEnabled() { return false; }

}  // namespace xla
