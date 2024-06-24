/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/client/lib/comparators.h"

#include <cmath>
#include <limits>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/test.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace {

class ComparatorsTest : public ClientLibraryTestBase {
 public:
  ComparatorsTest() : builder_(TestName()) {}
  XlaBuilder* builder() { return &builder_; }

 private:
  XlaBuilder builder_;
};

template <
    PrimitiveType type,
    typename T = typename primitive_util::PrimitiveTypeToNative<type>::type>
void BuildComparatorAndComparisons(ComparatorsTest* test,
                                   bool compare_less_than,
                                   absl::InlinedVector<bool, 10>* expected) {
  auto compare = compare_less_than
                     ? CreateScalarLtComputation({type}, test->builder())
                     : CreateScalarGtComputation({type}, test->builder());

  auto negative_nan = ConstantR0<T>(
      test->builder(), -T(std::numeric_limits<float>::quiet_NaN()));
  auto positive_nan = ConstantR0<T>(test->builder(),
                                    T(std::numeric_limits<float>::quiet_NaN()));
  auto negative_zero = ConstantR0<T>(test->builder(), T(-0.));
  auto positive_zero = ConstantR0<T>(test->builder(), T(0.));
  auto negative_infinity = MinValue(test->builder(), type);
  auto positive_infinity = MaxValue(test->builder(), type);

  // List the values in the expected sorting order from smallest to largest.
  std::vector<XlaOp> all_constants{negative_nan,      negative_infinity,
                                   negative_zero,     positive_zero,
                                   positive_infinity, positive_nan};

  // Do pairwise comparisons.
  std::vector<XlaOp> all_comparisons;
  all_comparisons.reserve(std::pow(all_constants.size(), 2));
  for (const XlaOp& lhs_constant : all_constants) {
    for (const XlaOp& rhs_constant : all_constants) {
      all_comparisons.push_back(Broadcast(
          Call(test->builder(), compare, {lhs_constant, rhs_constant}), {1}));
    }
  }

  // Concatenate the comparison results.
  ConcatInDim(test->builder(), all_comparisons, 0);

  // If we use less-than comparisons, we expect the comparison to result in true
  // if the lhs value to be compared appears earlier in 'all_constants' than the
  // rhs value. Likewise, if we use greater-than comparisons, we expect the
  // comparison to return true if the rhs value appears earlier in
  // 'all_constants' than the lhs value.
  expected->clear();
  for (int i = 0; i < all_constants.size(); ++i) {
    for (int j = 0; j < all_constants.size(); ++j) {
      expected->push_back(compare_less_than ? i < j : i > j);
    }
  }
}

XLA_TEST_F(ComparatorsTest, CompareLtBF16) {
  absl::InlinedVector<bool, 10> expected;
  BuildComparatorAndComparisons<BF16>(this, /*compare_less_than=*/true,
                                      &expected);
  ComputeAndCompareR1<bool>(builder(), expected, {});
}

XLA_TEST_F(ComparatorsTest, CompareGtBF16) {
  absl::InlinedVector<bool, 10> expected;
  BuildComparatorAndComparisons<BF16>(this, /*compare_less_than=*/false,
                                      &expected);
  ComputeAndCompareR1<bool>(builder(), expected, {});
}

XLA_TEST_F(ComparatorsTest, CompareLtF16) {
  absl::InlinedVector<bool, 10> expected;
  BuildComparatorAndComparisons<F16>(this, /*compare_less_than=*/true,
                                     &expected);
  ComputeAndCompareR1<bool>(builder(), expected, {});
}

XLA_TEST_F(ComparatorsTest, CompareGtF16) {
  absl::InlinedVector<bool, 10> expected;
  BuildComparatorAndComparisons<F16>(this, /*compare_less_than=*/false,
                                     &expected);
  ComputeAndCompareR1<bool>(builder(), expected, {});
}

XLA_TEST_F(ComparatorsTest, CompareLtF32) {
  absl::InlinedVector<bool, 10> expected;
  BuildComparatorAndComparisons<F32>(this, /*compare_less_than=*/true,
                                     &expected);
  ComputeAndCompareR1<bool>(builder(), expected, {});
}

XLA_TEST_F(ComparatorsTest, CompareGtF32) {
  absl::InlinedVector<bool, 10> expected;
  BuildComparatorAndComparisons<F32>(this, /*compare_less_than=*/false,
                                     &expected);
  ComputeAndCompareR1<bool>(builder(), expected, {});
}

XLA_TEST_F(ComparatorsTest, CompareLtF64) {
  absl::InlinedVector<bool, 10> expected;
  BuildComparatorAndComparisons<F64>(this, /*compare_less_than=*/true,
                                     &expected);
  ComputeAndCompareR1<bool>(builder(), expected, {});
}

XLA_TEST_F(ComparatorsTest, CompareGtF64) {
  absl::InlinedVector<bool, 10> expected;
  BuildComparatorAndComparisons<F64>(this, /*compare_less_than=*/false,
                                     &expected);
  ComputeAndCompareR1<bool>(builder(), expected, {});
}

const auto kCompareStr = HloOpcodeString(xla::HloOpcode::kCompare);
const auto kParameterStr = HloOpcodeString(xla::HloOpcode::kParameter);
const auto kSelectStr = HloOpcodeString(xla::HloOpcode::kSelect);

// Checks that `op` is a compare instruction with the given `direction` and
// whose inputs are parameter ops with the given numbers and `type`.
void ExpectCompareOp(
    const xla::HloInstructionProto op, xla::PrimitiveType type,
    absl::string_view direction, int parameter0_number, int parameter1_number,
    const tsl::protobuf::RepeatedPtrField<xla::HloInstructionProto>& all_ops) {
  EXPECT_EQ(op.opcode(), kCompareStr);

  const auto& operand0 = all_ops.at(op.operand_ids(0) - 1);
  EXPECT_EQ(operand0.opcode(), kParameterStr);
  EXPECT_EQ(operand0.parameter_number(), parameter0_number);
  EXPECT_EQ(operand0.shape().element_type(), type);

  const auto& operand1 = all_ops.at(op.operand_ids(1) - 1);
  EXPECT_EQ(operand1.opcode(), kParameterStr);
  EXPECT_EQ(operand1.parameter_number(), parameter1_number);
  EXPECT_EQ(operand1.shape().element_type(), type);
}

TEST(VariadicComparatorTest, OneOperandOneComparison) {
  XlaBuilder builder("test");
  XlaComputation comp = CreateScalarComparisonComputation(
      "computation", {U16}, {LtTotalOrder}, &builder);
  EXPECT_EQ(comp.proto().computations_size(), 1);
  EXPECT_EQ(comp.proto().computations(0).program_shape().parameters_size(), 2);

  const auto& instr = comp.proto().computations(0).instructions();
  const auto& root = instr.at(comp.proto().computations(0).root_id() - 1);
  ExpectCompareOp(root, U16, "LT", 0, 1, instr);
}

TEST(VariadicComparatorTest, TwoOperandsOneComparison) {
  XlaBuilder builder("test");
  XlaComputation comp = CreateScalarComparisonComputation(
      "computation", {U16, U32}, {LtTotalOrder, {}}, &builder);
  EXPECT_EQ(comp.proto().computations_size(), 1);
  EXPECT_EQ(comp.proto().computations(0).program_shape().parameters_size(), 4);

  const auto& instr = comp.proto().computations(0).instructions();
  const auto& root = instr.at(comp.proto().computations(0).root_id() - 1);
  ExpectCompareOp(root, U16, "LT", 0, 1, instr);
}

TEST(VariadicComparatorTest, TwoOperandsTwoComparisons) {
  XlaBuilder builder("test");
  XlaComputation comp = CreateScalarComparisonComputation(
      "computation", {U16, U32}, {LtTotalOrder, LtTotalOrder}, &builder);

  EXPECT_EQ(comp.proto().computations_size(), 1);
  EXPECT_EQ(comp.proto().computations(0).program_shape().parameters_size(), 4);

  const auto& instr = comp.proto().computations(0).instructions();
  const auto& root = instr.at(comp.proto().computations(0).root_id() - 1);
  EXPECT_EQ(root.opcode(), HloOpcodeString(xla::HloOpcode::kSelect));
  ExpectCompareOp(instr.at(root.operand_ids(0) - 1), U16, "EQ", 0, 1, instr);
  ExpectCompareOp(instr.at(root.operand_ids(1) - 1), U32, "LT", 2, 3, instr);
  ExpectCompareOp(instr.at(root.operand_ids(2) - 1), U16, "LT", 0, 1, instr);
}

}  // namespace
}  // namespace xla
