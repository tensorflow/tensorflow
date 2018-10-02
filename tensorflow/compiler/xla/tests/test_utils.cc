/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <cmath>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"

namespace xla {

namespace {

template <typename FloatT, typename GeneratorT>
void PopulateWithRandomFloatingPointDataImpl(Literal* literal,
                                             std::minstd_rand0* engine,
                                             bool no_duplicates) {
  CHECK(engine != nullptr);
  CHECK_EQ(literal->shape().element_type(),
           primitive_util::NativeToPrimitiveType<FloatT>());
  if (no_duplicates) {
    // Duplicates may be generated if the number of elements in the literal
    // exceeds the number of positive values supported by the type.
    FloatT next_value = std::numeric_limits<FloatT>::min();
    for (FloatT& value : literal->data<FloatT>()) {
      value = next_value;
      next_value =
          std::nextafter(next_value, std::numeric_limits<FloatT>::max());
    }
    std::shuffle(literal->data<FloatT>().begin(), literal->data<FloatT>().end(),
                 *engine);
  } else {
    std::uniform_real_distribution<GeneratorT> generator(-0.1f, 0.2f);
    for (FloatT& value : literal->data<FloatT>()) {
      value = static_cast<FloatT>(generator(*engine));
    }
  }
}

template <typename FloatT>
void PopulateWithRandomFloatingPointData(Literal* literal,
                                         std::minstd_rand0* engine,
                                         bool no_duplicates) {
  CHECK(engine != nullptr);
  PopulateWithRandomFloatingPointDataImpl<FloatT, FloatT>(literal, engine,
                                                          no_duplicates);
}

template <>
void PopulateWithRandomFloatingPointData<half>(Literal* literal,
                                               std::minstd_rand0* engine,
                                               bool no_duplicates) {
  // no_duplicates is ignored for half types. Unique values can only be
  // generated for arrays with fewer than ~2**16 elements and no_duplicates is
  // best-effort anyway.
  CHECK(engine != nullptr);
  std::uniform_real_distribution<float> generator(-0.1f, 0.2f);
  for (half& value : literal->data<half>()) {
    value = static_cast<half>(generator(*engine));
  }
}

template <>
void PopulateWithRandomFloatingPointData<bfloat16>(Literal* literal,
                                                   std::minstd_rand0* engine,
                                                   bool no_duplicates) {
  // no_duplicates is ignored for bfloat types. Unique values can only be
  // generated for arrays with fewer than ~2**16 elements and no_duplicates is
  // best-effort anyway.
  CHECK(engine != nullptr);
  std::uniform_real_distribution<float> generator(-0.1f, 0.2f);
  for (bfloat16& value : literal->data<bfloat16>()) {
    value = static_cast<bfloat16>(generator(*engine));
  }
}

template <typename IntT>
void PopulateWithRandomIntegralData(Literal* literal, std::minstd_rand0* engine,
                                    bool no_duplicates) {
  CHECK(engine != nullptr);
  CHECK_EQ(literal->shape().element_type(),
           primitive_util::NativeToPrimitiveType<IntT>());
  if (no_duplicates && ShapeUtil::ElementsIn(literal->shape()) <
                           std::numeric_limits<IntT>::max()) {
    std::iota(literal->data<IntT>().begin(), literal->data<IntT>().end(), 0);
    std::shuffle(literal->data<IntT>().begin(), literal->data<IntT>().end(),
                 *engine);
  } else {
    std::uniform_int_distribution<IntT> generator(
        std::numeric_limits<IntT>::lowest(), std::numeric_limits<IntT>::max());
    for (IntT& value : literal->data<IntT>()) {
      value = generator(*engine);
    }
  }
}

// Similar to MakeFakeLiteral but takes a random number generator engine to
// enable reusing the engine across randomly generated literals. 'no_duplicates'
// indicates that there should be no duplicate values in each generated
// array. This is uniqueness is best-effort only. Some types (half and bfloat16)
// are not supported and uniqueness cannot be guaranteed if the number of
// elements exceeds the number of different values supported by the type.
StatusOr<Literal> MakeFakeLiteralInternal(const Shape& shape,
                                          std::minstd_rand0* engine,
                                          bool no_duplicates) {
  if (ShapeUtil::IsTuple(shape)) {
    std::vector<Literal> elements;
    for (const Shape& element_shape : shape.tuple_shapes()) {
      TF_ASSIGN_OR_RETURN(
          Literal element,
          MakeFakeLiteralInternal(element_shape, engine, no_duplicates));
      elements.push_back(std::move(element));
    }
    return LiteralUtil::MakeTupleOwned(std::move(elements));
  }
  if (engine == nullptr) {
    return Literal::CreateFromShape(shape);
  }
  Literal literal(shape);
  switch (shape.element_type()) {
    case BF16:
      PopulateWithRandomFloatingPointData<bfloat16>(&literal, engine,
                                                    no_duplicates);
      break;
    case F16:
      PopulateWithRandomFloatingPointData<half>(&literal, engine,
                                                no_duplicates);
      break;
    case F32:
      PopulateWithRandomFloatingPointData<float>(&literal, engine,
                                                 no_duplicates);
      break;
    case F64:
      PopulateWithRandomFloatingPointData<double>(&literal, engine,
                                                  no_duplicates);
      break;
    case S8:
      PopulateWithRandomIntegralData<int8>(&literal, engine, no_duplicates);
      break;
    case U8:
      PopulateWithRandomIntegralData<uint8>(&literal, engine, no_duplicates);
      break;
    case S16:
      PopulateWithRandomIntegralData<int16>(&literal, engine, no_duplicates);
      break;
    case U16:
      PopulateWithRandomIntegralData<uint16>(&literal, engine, no_duplicates);
      break;
    case S32:
      PopulateWithRandomIntegralData<int32>(&literal, engine, no_duplicates);
      break;
    case U32:
      PopulateWithRandomIntegralData<uint32>(&literal, engine, no_duplicates);
      break;
    case S64:
      PopulateWithRandomIntegralData<int64>(&literal, engine, no_duplicates);
      break;
    case U64:
      PopulateWithRandomIntegralData<uint64>(&literal, engine, no_duplicates);
      break;
    case PRED: {
      std::uniform_int_distribution<int> generator(0, 1);
      TF_CHECK_OK(
          literal.Populate<bool>([&](absl::Span<const int64> /*indices*/) {
            return generator(*engine);
          }));
      break;
    }
    // Token requires no data.
    case TOKEN:
      break;
    default:
      return Unimplemented("Unsupported type for fake literal generation: %s",
                           ShapeUtil::HumanString(shape));
  }
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
  const int64 op_num = use.operand_number;
  return ((opcode == HloOpcode::kReduceWindow && op_num == 1) ||
          (opcode == HloOpcode::kSelectAndScatter && op_num == 2) ||
          (opcode == HloOpcode::kReduce &&
           op_num >= instruction->operand_count() / 2));
}

// Generate random values that are constrained to the input_shape minus the
// output_shape so as not to produce wrapping slices, for instance.
Literal MakeRandomIndex(absl::Span<const int64> index_space,
                        std::minstd_rand0* engine) {
  std::vector<int32> start_indices(index_space.size());
  if (engine != nullptr) {
    for (int i = 0; i < index_space.size(); ++i) {
      std::uniform_int_distribution<int32> generator(0, index_space[i]);
      start_indices[i] = generator(*engine);
    }
  }
  return LiteralUtil::CreateR1<int32>(start_indices);
}

// Use dataflow analysis on each parameter to see if there are uses that would
// be problematic when generating input data.  Returns the list of instructions
// that correspond to their uses.
//
// Should be paired with the CreateLiteralForConstrainedUses() function below.
std::vector<HloInstruction*> FindConstrainedUses(
    const HloDataflowAnalysis& dataflow, const HloInstruction& param) {
  std::vector<HloInstruction*> constrained_uses;
  for (const auto& pair : dataflow.GetInstructionValueSet(&param)) {
    const HloValue& value = dataflow.GetUniqueValueAt(&param, pair.first);
    for (const HloUse& use : value.uses()) {
      HloInstruction* instruction = use.instruction;
      const HloOpcode opcode = instruction->opcode();
      const int64 op_num = use.operand_number;
      if ((opcode == HloOpcode::kDynamicSlice && op_num == 1) ||
          (opcode == HloOpcode::kDynamicUpdateSlice && op_num == 2)) {
        constrained_uses.push_back(instruction);
      } else if (opcode == HloOpcode::kFusion) {
        const HloInstruction* const to_analyze =
            instruction->fused_parameter(op_num);
        auto fused_uses = FindConstrainedUses(dataflow, *to_analyze);
        constrained_uses.insert(constrained_uses.end(), fused_uses.begin(),
                                fused_uses.end());
      } else if (NeedsInitValue(use)) {
        constrained_uses.push_back(instruction);
      } else if (opcode == HloOpcode::kConvert ||
                 opcode == HloOpcode::kReducePrecision) {
        auto converted_uses = FindConstrainedUses(dataflow, *instruction);
        constrained_uses.insert(constrained_uses.end(), converted_uses.begin(),
                                converted_uses.end());
      } else if (opcode == HloOpcode::kSort &&
                 instruction->operand_count() == 2 && op_num == 0) {
        // Operand 0 of sort is the array of keys used for key/value
        // (two-operand) kSort instructions.
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
StatusOr<Literal> CreateLiteralForConstrainedUses(
    const absl::Span<HloInstruction* const> constrained_uses,
    const HloInstruction& param, std::minstd_rand0* engine) {
  std::vector<int64> index_space;
  bool no_duplicates = false;
  bool needs_constant = false;
  ConstantType constant_type = ConstantType::kUnknown;
  for (HloInstruction* use : constrained_uses) {
    switch (use->opcode()) {
      case HloOpcode::kDynamicSlice:
      case HloOpcode::kDynamicUpdateSlice: {
        const Shape& indexed_shape = use->operand(0)->shape();
        const Shape& slice_shape = use->opcode() == HloOpcode::kDynamicSlice
                                       ? use->shape()
                                       : use->operand(1)->shape();
        const int64 rank = ShapeUtil::Rank(indexed_shape);
        if (!index_space.empty()) {
          TF_RET_CHECK(rank == index_space.size());
          for (int64 i = 0; i < rank; ++i) {
            index_space[i] = std::min(
                index_space[i], ShapeUtil::GetDimension(indexed_shape, i) -
                                    ShapeUtil::GetDimension(slice_shape, i));
          }
        } else {
          index_space.resize(rank);
          for (int64 i = 0; i < rank; ++i) {
            index_space[i] = ShapeUtil::GetDimension(indexed_shape, i) -
                             ShapeUtil::GetDimension(slice_shape, i);
          }
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
  constraint_count += !index_space.empty() ? 1 : 0;
  constraint_count += needs_constant ? 1 : 0;
  if (constraint_count > 1) {
    return Unimplemented("Conflicting operand generation constraints.");
  }
  if (!index_space.empty()) {
    return MakeRandomIndex(index_space, engine);
  } else if (needs_constant) {
    switch (constant_type) {
      case ConstantType::kZero:
        return LiteralUtil::Zero(param.shape().element_type());
      case ConstantType::kOne:
        return LiteralUtil::One(param.shape().element_type());
      case ConstantType::kUnknown:
        // We want the identity element for the computation, but we don't really
        // know what it is - so any value we generate will be just as wrong.
        return MakeFakeLiteralInternal(param.shape(), engine,
                                       /*no_duplicates=*/false);
    }
  } else {
    return MakeFakeLiteralInternal(param.shape(), engine, no_duplicates);
  }
}

// Given a module entry parameter, use the dataflow analysis to see if a
// special case literal must be created, or if we can generate fake data.
StatusOr<Literal> MakeConstrainedArgument(const HloDataflowAnalysis& dataflow,
                                          const HloInstruction& param,
                                          std::minstd_rand0* engine) {
  const auto constrained_uses = FindConstrainedUses(dataflow, param);
  return CreateLiteralForConstrainedUses(constrained_uses, param, engine);
}

}  // namespace

StatusOr<Literal> MakeFakeLiteral(const Shape& shape, bool pseudo_random) {
  auto engine =
      pseudo_random ? absl::make_unique<std::minstd_rand0>() : nullptr;
  return MakeFakeLiteralInternal(shape, engine.get(), /*no_duplicates=*/false);
}

StatusOr<std::vector<Literal>> MakeFakeArguments(HloModule* const module,
                                                 bool pseudo_random) {
  auto engine =
      pseudo_random ? absl::make_unique<std::minstd_rand0>() : nullptr;
  return MakeFakeArguments(module, engine.get());
}

StatusOr<std::vector<Literal>> MakeFakeArguments(HloModule* const module,
                                                 std::minstd_rand0* engine) {
  TF_ASSIGN_OR_RETURN(auto dataflow, HloDataflowAnalysis::Run(*module));
  const auto params = module->entry_computation()->parameter_instructions();
  std::vector<Literal> arguments(params.size());
  for (int i = 0; i < params.size(); ++i) {
    arguments[i] =
        MakeConstrainedArgument(*dataflow, *params[i], engine).ValueOrDie();
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
  CHECK_EQ(ShapeUtil::Rank(lhs->shape()), 2);
  CHECK_EQ(ShapeUtil::Rank(rhs->shape()), 2);
  PrecisionConfig precision_config;
  precision_config.mutable_operand_precision()->Resize(
      2, PrecisionConfig::DEFAULT);
  DotDimensionNumbers dot_dimension_numbers;
  dot_dimension_numbers.add_lhs_contracting_dimensions(1);
  dot_dimension_numbers.add_rhs_contracting_dimensions(0);
  return absl::make_unique<HloDotInstruction>(
      shape, lhs, rhs, dot_dimension_numbers, precision_config);
}
}  // namespace xla
