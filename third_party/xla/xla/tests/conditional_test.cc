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

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "xla/array2d.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
namespace {

constexpr ErrorSpec kErrorSpec{0.001};

class ConditionalOpTest : public ClientLibraryTestRunnerMixin<HloTestBase> {
 protected:
  void SetUp() override {
    ClientLibraryTestRunnerMixin<HloTestBase>::SetUp();
    mutable_debug_options()->set_xla_test_add_command_buffer_mode(true);
  }

  XlaComputation CreateR0ConstantComputation(float value) {
    XlaBuilder builder("Constant");
    Parameter(&builder, 0, empty_tuple_, "tuple");
    ConstantR0<float>(&builder, value);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return std::move(build_status).value();
  }

  XlaComputation CreateR0IdentityComputation() {
    XlaBuilder builder("Identity");
    Parameter(&builder, 0, r0f32_, "x");
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return std::move(build_status).value();
  }

  XlaComputation CreateCeilComputation(const Shape& shape) {
    XlaBuilder builder("Ceil");
    auto param = Parameter(&builder, 0, shape, "param");
    Ceil(param);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return std::move(build_status).value();
  }

  XlaComputation CreateR0CeilComputation() {
    return CreateCeilComputation(r0f32_);
  }

  XlaComputation CreateR1CeilComputation() {
    return CreateCeilComputation(r1s2f32_);
  }

  XlaComputation CreateFloorComputation(const Shape& shape) {
    XlaBuilder builder("Floor");
    auto param = Parameter(&builder, 0, shape, "param");
    Floor(param);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return std::move(build_status).value();
  }

  XlaComputation CreateR0FloorComputation() {
    return CreateFloorComputation(r0f32_);
  }

  XlaComputation CreateR1FloorComputation() {
    return CreateFloorComputation(r1s2f32_);
  }

  XlaComputation CreateTupleCeilComputation(const std::string& computation_name,
                                            const Shape& tuple_shape) {
    XlaBuilder builder(computation_name);
    auto tuple = Parameter(&builder, 0, tuple_shape, "tuple");
    auto x = GetTupleElement(tuple, 0);
    auto y = GetTupleElement(tuple, 1);
    auto x_ceil = Ceil(x);
    auto y_ceil = Ceil(y);
    Tuple(&builder, {x_ceil, y_ceil});
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return std::move(build_status).value();
  }

  XlaComputation CreateR0TupleCeilComputation() {
    return CreateTupleCeilComputation("CeilR0", tuple_2_r0f32_);
  }

  XlaComputation CreateR1TupleCeilComputation() {
    return CreateTupleCeilComputation("CeilR1", tuple_2_r1s2f32_);
  }

  XlaComputation CreateTupleFloorComputation(
      const std::string& computation_name, const Shape& tuple_shape) {
    XlaBuilder builder(computation_name);
    auto tuple = Parameter(&builder, 0, tuple_shape, "tuple");
    auto x = GetTupleElement(tuple, 0);
    auto y = GetTupleElement(tuple, 1);
    auto x_floor = Floor(x);
    auto y_floor = Floor(y);
    Tuple(&builder, {x_floor, y_floor});
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return std::move(build_status).value();
  }

  XlaComputation CreateR0TupleFloorComputation() {
    return CreateTupleFloorComputation("FloorR0", tuple_2_r0f32_);
  }

  XlaComputation CreateR1TupleFloorComputation() {
    return CreateTupleFloorComputation("FloorR1", tuple_2_r1s2f32_);
  }

  XlaComputation CreateTupleAddComputation(const std::string& computation_name,
                                           const Shape& tuple_shape) {
    XlaBuilder builder(computation_name);
    auto tuple = Parameter(&builder, 0, tuple_shape, "tuple");
    auto x = GetTupleElement(tuple, 0);
    auto y = GetTupleElement(tuple, 1);
    Add(x, y);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return std::move(build_status).value();
  }

  XlaComputation CreateR0TupleAddComputation() {
    return CreateTupleAddComputation("AddR0", tuple_2_r0f32_);
  }

  XlaComputation CreateR1TupleAddComputation() {
    return CreateTupleAddComputation("AddR1", tuple_2_r1s2f32_);
  }

  XlaComputation CreateTupleSubComputation(const std::string& computation_name,
                                           const Shape& tuple_shape) {
    XlaBuilder builder(computation_name);
    auto tuple = Parameter(&builder, 0, tuple_shape, "tuple");
    auto x = GetTupleElement(tuple, 0);
    auto y = GetTupleElement(tuple, 1);
    Sub(x, y);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return std::move(build_status).value();
  }

  XlaComputation CreateR0TupleSubComputation() {
    return CreateTupleSubComputation("SubR0", tuple_2_r0f32_);
  }

  XlaComputation CreateR1TupleSubComputation() {
    return CreateTupleSubComputation("SubR1", tuple_2_r1s2f32_);
  }

  Shape r0f32_ = ShapeUtil::MakeShape(F32, {});
  Shape r1s2f32_ = ShapeUtil::MakeShape(F32, {2});
  Shape tuple_2_r0f32_ = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {}), ShapeUtil::MakeShape(F32, {})});
  Shape tuple_2_r1s2f32_ = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {2}), ShapeUtil::MakeShape(F32, {2})});
  Shape empty_tuple_ = ShapeUtil::MakeTupleShape({});
};

// Test fixture to run indexed conditional (switch/case) tests with varying
// number of branches.
class CaseOpTest : public ConditionalOpTest,
                   public ::testing::WithParamInterface<int> {};

// Test true and false computations that do not take any parameters.
TEST_F(ConditionalOpTest, Parameters0) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(true, 0, "pred", &builder, &pred);
  auto operands = Tuple(&builder, {});
  auto true_computation = CreateR0ConstantComputation(56.0f);
  auto false_computation = CreateR0ConstantComputation(12.0f);
  Conditional(pred, operands, true_computation, operands, false_computation);

  ComputeAndCompareR0<float>(&builder, 56.0f, {&pred_arg}, kErrorSpec);
}

// Test branch computations that do not take any parameters.
TEST_P(CaseOpTest, Parameters0) {
  int num_branches = GetParam();
  for (int bi = -1; bi <= num_branches; ++bi) {
    SCOPED_TRACE(bi);
    XlaBuilder builder(TestName());
    XlaOp branch_index;
    auto branch_index_arg = CreateR0Parameter<int32_t>(
        bi, 0, "branch_index_arg", &builder, &branch_index);
    auto operand = Tuple(&builder, {});

    std::vector<XlaOp> operands(num_branches, operand);
    std::vector<XlaComputation> branches;
    branches.reserve(num_branches);
    std::vector<const XlaComputation*> branches_p(num_branches);
    for (int i = 0; i < num_branches; ++i) {
      branches.emplace_back(
          CreateR0ConstantComputation(static_cast<float>(i) * 10));
      branches_p[i] = &branches[i];
    }
    Conditional(branch_index, branches_p, operands);

    float expected = 10 * static_cast<float>((bi < 0 || bi >= num_branches)
                                                 ? num_branches - 1
                                                 : bi);
    ComputeAndCompareR0<float>(&builder, expected, {&branch_index_arg},
                               kErrorSpec);
  }
}

// Test true and false computations that take in 1 parameter.
TEST_F(ConditionalOpTest, Parameters1) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operand1 = ConstantR0<float>(&builder, 56.0f);
  auto operand2 = ConstantR0<float>(&builder, 12.0f);
  auto identity = CreateR0IdentityComputation();
  Conditional(pred, operand1, identity, operand2, identity);

  ComputeAndCompareR0<float>(&builder, 12.0f, {&pred_arg}, kErrorSpec);
}

// Test branch computations that take in 1 parameter.
TEST_P(CaseOpTest, Parameters1) {
  int num_branches = GetParam();
  for (int bi = -1; bi <= num_branches; ++bi) {
    SCOPED_TRACE(bi);
    XlaBuilder builder(TestName());
    XlaOp branch_index;
    auto branch_index_arg = CreateR0Parameter<int32_t>(
        bi, 0, "branch_index_arg", &builder, &branch_index);

    auto make_branch = [&builder, this](int i) {
      auto sb = builder.CreateSubBuilder(absl::StrCat("branch_", i));
      Add(ConstantR0<float>(sb.get(), static_cast<float>(i)),
          Parameter(sb.get(), 0, r0f32_, "p0"));
      return sb->BuildAndNoteError();
    };
    std::vector<XlaComputation> branches;
    branches.reserve(num_branches);
    std::vector<const XlaComputation*> branches_p(num_branches);
    std::vector<XlaOp> operands;
    operands.reserve(num_branches);
    std::vector<float> expecteds(num_branches);
    for (int i = 0; i < num_branches; ++i) {
      branches.emplace_back(make_branch(i));
      branches_p[i] = &branches[i];
      auto fi = static_cast<float>(i);
      operands.emplace_back(ConstantR0<float>(&builder, 10 * fi + 7));
      expecteds[i] = 10 * fi + 7 + fi;
    }

    Conditional(branch_index, branches_p, operands);
    float expected = (bi < 0 || bi >= num_branches)
                         ? expecteds[num_branches - 1]
                         : expecteds[bi];
    ComputeAndCompareR0<float>(&builder, expected, {&branch_index_arg},
                               kErrorSpec);
  }
}

// Test conditional with two different computations in the true and false cases
// that take in different arguments.
TEST_F(ConditionalOpTest, DiffComputationsDiffArgs) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operand1 = ConstantR0<float>(&builder, 56.4f);
  auto operand2 = ConstantR0<float>(&builder, 12.6f);
  Conditional(pred, operand1, CreateR0CeilComputation(), operand2,
              CreateR0FloorComputation());

  ComputeAndCompareR0<float>(&builder, 12.0f, {&pred_arg}, kErrorSpec);
}

// Test conditional with two different computations in the true and false cases
// that take in the same arguments.
TEST_F(ConditionalOpTest, DiffComputationsSameArg) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operand = ConstantR0<float>(&builder, 12.6f);
  Conditional(pred, operand, CreateR0CeilComputation(), operand,
              CreateR0FloorComputation());

  ComputeAndCompareR0<float>(&builder, 12.0f, {&pred_arg}, kErrorSpec);
}

// Test conditional with the same computation in the true and false cases but
// take in different arguments.
TEST_F(ConditionalOpTest, SameComputationDiffArgs) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operand1 = ConstantR0<float>(&builder, 56.4f);
  auto operand2 = ConstantR0<float>(&builder, 12.6f);
  auto floor = CreateR0FloorComputation();
  Conditional(pred, operand1, floor, operand2, floor);

  ComputeAndCompareR0<float>(&builder, 12.0f, {&pred_arg}, kErrorSpec);
}

// Test conditional with the same computation in the true and false cases that
// take in the same arguments.
TEST_F(ConditionalOpTest, SameComputationSameArg) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operand = ConstantR0<float>(&builder, 12.6f);
  auto floor = CreateR0FloorComputation();
  Conditional(pred, operand, floor, operand, floor);

  ComputeAndCompareR0<float>(&builder, 12.0f, {&pred_arg}, kErrorSpec);
}

// Test conditional with different instances of the same computation in the true
// and false cases.
TEST_F(ConditionalOpTest, SameComputationDiffInstances) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operand1 = ConstantR0<float>(&builder, 56.4f);
  auto operand2 = ConstantR0<float>(&builder, 12.6f);
  Conditional(pred, operand1, CreateR0FloorComputation(), operand2,
              CreateR0FloorComputation());

  ComputeAndCompareR0<float>(&builder, 12.0f, {&pred_arg}, kErrorSpec);
}

// Test the case when a call invokes a computation that contains a conditional.
TEST_F(ConditionalOpTest, ConditionalWithCall) {
  Shape r0bool = ShapeUtil::MakeShape(PRED, {});
  XlaBuilder inner_builder(TestName() + ".inner_conditional");
  auto pred_cond = Parameter(&inner_builder, 0, r0bool, "param0");
  auto true_operand = Parameter(&inner_builder, 1, r0f32_, "param1");
  auto false_operand = Parameter(&inner_builder, 2, r0f32_, "param2");
  Conditional(pred_cond, true_operand, CreateR0CeilComputation(), false_operand,
              CreateR0FloorComputation());
  auto inner_builder_result = inner_builder.Build().value();

  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operand1 = ConstantR0<float>(&builder, 56.4f);
  auto operand2 = ConstantR0<float>(&builder, 12.6f);
  Call(&builder, inner_builder_result, {pred, operand1, operand2});

  ComputeAndCompareR0<float>(&builder, 12.0f, {&pred_arg}, kErrorSpec);
}

// Test true and false computations that take in 2 parameters and predicate is
// true.
TEST_F(ConditionalOpTest, Parameters2TrueBranch) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(true, 0, "pred", &builder, &pred);
  auto operand1 = ConstantR0<float>(&builder, 56.0f);
  auto operand2 = ConstantR0<float>(&builder, 12.0f);
  auto operands = Tuple(&builder, {operand1, operand2});
  Conditional(pred, operands, CreateR0TupleAddComputation(), operands,
              CreateR0TupleSubComputation());

  ComputeAndCompareR0<float>(&builder, 68.0f, {&pred_arg}, kErrorSpec);
}

// Test true and false computations that take in 2 parameters and predicate is
// false.
TEST_F(ConditionalOpTest, Parameters2FalseBranch) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operand1 = ConstantR0<float>(&builder, 56.0f);
  auto operand2 = ConstantR0<float>(&builder, 12.0f);
  auto operands = Tuple(&builder, {operand1, operand2});
  Conditional(pred, operands, CreateR0TupleAddComputation(), operands,
              CreateR0TupleSubComputation());

  ComputeAndCompareR0<float>(&builder, 44.0f, {&pred_arg}, kErrorSpec);
}

// Test true and false computations that take in 2 array parameters and
// predicate is true.
TEST_F(ConditionalOpTest, Parameters2ArrayTrueBranch) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(true, 0, "pred", &builder, &pred);
  auto operand1 = ConstantR1<float>(&builder, {24.0f, 56.0f});
  auto operand2 = ConstantR1<float>(&builder, {10.0f, 11.0f});
  auto operands = Tuple(&builder, {operand1, operand2});
  Conditional(pred, operands, CreateR1TupleAddComputation(), operands,
              CreateR1TupleSubComputation());

  ComputeAndCompareR1<float>(&builder, {34.0f, 67.0f}, {&pred_arg}, kErrorSpec);
}

// Test branch computations that take in 2 array parameters.
TEST_P(CaseOpTest, Parameters2Array) {
  int num_branches = GetParam();
  for (int bi = -1; bi <= num_branches; ++bi) {
    SCOPED_TRACE(bi);
    XlaBuilder builder(TestName());
    XlaOp branch_index;
    auto branch_index_arg =
        CreateR0Parameter<int32_t>(bi, 0, "pred", &builder, &branch_index);
    auto operand1 = ConstantR1<float>(&builder, {24.0f, 56.0f});
    auto operand2 = ConstantR1<float>(&builder, {10.0f, 11.0f});
    auto operands = Tuple(&builder, {operand1, operand2});
    auto make_branch = [&builder, this](int i) {
      auto sb = builder.CreateSubBuilder(absl::StrCat("branch_", i));
      auto p = Parameter(sb.get(), 0, tuple_2_r1s2f32_, "p0");
      Add(Mul(ConstantR0<float>(sb.get(), static_cast<float>(i)),
              GetTupleElement(p, 0)),
          GetTupleElement(p, 1));
      return sb->BuildAndNoteError();
    };
    std::vector<XlaComputation> branches;
    branches.reserve(num_branches);
    std::vector<const XlaComputation*> branches_p(num_branches);
    for (int i = 0; i < num_branches; ++i) {
      branches.emplace_back(make_branch(i));
      branches_p[i] = &branches[i];
    }
    Conditional(branch_index, branches_p,
                std::vector<XlaOp>(num_branches, operands));
    auto modified_bi = static_cast<float>(
        (bi < 0 || bi >= num_branches) ? num_branches - 1 : bi);
    ComputeAndCompareR1<float>(
        &builder, {24.0f * modified_bi + 10, 56.0f * modified_bi + 11},
        {&branch_index_arg}, kErrorSpec);
  }
}

INSTANTIATE_TEST_SUITE_P(CaseOpTest_Instantiation, CaseOpTest,
                         ::testing::Values(1, 2, 3, 4, 5));

// Test true and false computations that take in 2 array parameters and
// predicate is false.
TEST_F(ConditionalOpTest, Parameters2ArrayFalseBranch) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operand1 = ConstantR1<float>(&builder, {24.0f, 56.0f});
  auto operand2 = ConstantR1<float>(&builder, {10.0f, 11.0f});
  auto operands = Tuple(&builder, {operand1, operand2});
  Conditional(pred, operands, CreateR1TupleAddComputation(), operands,
              CreateR1TupleSubComputation());

  ComputeAndCompareR1<float>(&builder, {14.0f, 45.0f}, {&pred_arg}, kErrorSpec);
}

// Test true and false computations that return a tuple of scalars.
TEST_F(ConditionalOpTest, ReturnTupleOfScalars) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operands = Tuple(&builder, {ConstantR0<float>(&builder, 12.2f),
                                   ConstantR0<float>(&builder, 25.6f)});
  Conditional(pred, operands, CreateR0TupleCeilComputation(), operands,
              CreateR0TupleFloorComputation());

  ComputeAndCompareLiteral(
      &builder,
      LiteralUtil::MakeTupleFromSlices({LiteralUtil::CreateR0<float>(12.0f),
                                        LiteralUtil::CreateR0<float>(25.0f)}),
      {&pred_arg}, kErrorSpec);
}

// Test true and false computations that return a tuple of arrays.
TEST_F(ConditionalOpTest, ReturnTupleOfArrays) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(true, 0, "pred", &builder, &pred);
  auto operands =
      Tuple(&builder, {ConstantR1<float>(&builder, {12.2f, 15.8f}),
                       ConstantR1<float>(&builder, {25.6f, 29.2f})});
  Conditional(pred, operands, CreateR1TupleCeilComputation(), operands,
              CreateR1TupleFloorComputation());

  ComputeAndCompareLiteral(&builder,
                           LiteralUtil::MakeTupleFromSlices(
                               {LiteralUtil::CreateR1<float>({13.0f, 16.0f}),
                                LiteralUtil::CreateR1<float>({26.0f, 30.0f})}),
                           {&pred_arg}, kErrorSpec);
}

// Test true and false computations that return a tuple of a predicate, a
// scalar, and an array.
TEST_F(ConditionalOpTest, ReturnTupleofPredicateScalarArray) {
  XlaBuilder true_builder(TestName() + ".true");
  {
    Parameter(&true_builder, 0, empty_tuple_, "tuple");
    auto true_pred = ConstantR0<bool>(&true_builder, true);
    auto true_scalar = ConstantR0<float>(&true_builder, 12.2f);
    auto true_array = ConstantR1<float>(&true_builder, {12.8f, 14.6f});
    Tuple(&true_builder, {true_pred, true_scalar, true_array});
  }
  auto true_builder_result = true_builder.Build();
  EXPECT_IS_OK(true_builder_result.status());

  XlaBuilder false_builder(TestName() + ".false");
  {
    Parameter(&false_builder, 0, empty_tuple_, "tuple");
    auto false_pred = ConstantR0<bool>(&false_builder, false);
    auto false_scalar = ConstantR0<float>(&false_builder, 25.6f);
    auto false_array = ConstantR1<float>(&false_builder, {26.4f, 32.6f});
    Tuple(&false_builder, {false_pred, false_scalar, false_array});
  }
  auto false_builder_result = false_builder.Build();
  EXPECT_IS_OK(false_builder_result.status());

  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(true, 0, "pred", &builder, &pred);
  auto operands = Tuple(&builder, {});
  Conditional(pred, operands, std::move(true_builder_result).value(), operands,
              std::move(false_builder_result).value());

  ComputeAndCompareLiteral(&builder,
                           LiteralUtil::MakeTupleFromSlices(
                               {LiteralUtil::CreateR0<bool>(true),
                                LiteralUtil::CreateR0<float>(12.2f),
                                LiteralUtil::CreateR1<float>({12.8f, 14.6f})}),
                           {&pred_arg}, kErrorSpec);
}

// Test true and false computations that return a nested tuple.
TEST_F(ConditionalOpTest, ReturnNestedTuple) {
  XlaBuilder true_builder(TestName() + ".true");
  {
    Parameter(&true_builder, 0, empty_tuple_, "tuple");
    auto true_constant1 = ConstantR0<float>(&true_builder, 12.2f);
    auto true_constant2 = ConstantR1<float>(&true_builder, {12.8f, 14.6f});
    auto true_constant3 = ConstantR1<float>(&true_builder, {25.4f, 29.8f});
    auto true_constant4 = ConstantR0<float>(&true_builder, 35.6f);
    Tuple(&true_builder,
          {Tuple(&true_builder, {true_constant1, true_constant2}),
           Tuple(&true_builder, {true_constant3, true_constant4})});
  }
  auto true_builder_result = true_builder.Build();
  EXPECT_IS_OK(true_builder_result.status());

  XlaBuilder false_builder(TestName() + ".false");
  {
    Parameter(&false_builder, 0, empty_tuple_, "tuple");
    auto false_constant1 = ConstantR0<float>(&false_builder, 46.6f);
    auto false_constant2 = ConstantR1<float>(&false_builder, {54.4f, 58.4f});
    auto false_constant3 = ConstantR1<float>(&false_builder, {62.1f, 67.4f});
    auto false_constant4 = ConstantR0<float>(&false_builder, 9.3f);
    Tuple(&false_builder,
          {Tuple(&false_builder, {false_constant1, false_constant2}),
           Tuple(&false_builder, {false_constant3, false_constant4})});
  }
  auto false_builder_result = false_builder.Build();
  EXPECT_IS_OK(false_builder_result.status());

  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operands = Tuple(&builder, {});
  Conditional(pred, operands, std::move(true_builder_result).value(), operands,
              std::move(false_builder_result).value());

  ComputeAndCompareLiteral(
      &builder,
      LiteralUtil::MakeTupleFromSlices(
          {LiteralUtil::MakeTupleFromSlices(
               {LiteralUtil::CreateR0<float>(46.6f),
                LiteralUtil::CreateR1<float>({54.4f, 58.4f})}),
           LiteralUtil::MakeTupleFromSlices(
               {LiteralUtil::CreateR1<float>({62.1f, 67.4f}),
                LiteralUtil::CreateR0<float>(9.3f)})}),
      {&pred_arg}, kErrorSpec);
}

// Test conditional that takes in scalar operands in the form of external
// params.
TEST_F(ConditionalOpTest, ScalarOperandsFromExternalParams) {
  Shape r0bool = ShapeUtil::MakeShape(PRED, {});
  XlaBuilder builder(TestName());

  XlaOp pred, operand1, operand2;
  auto pred_arg = CreateR0Parameter<bool>(true, 0, "pred", &builder, &pred);
  auto operand1_param =
      CreateR0Parameter<float>(56.3f, 1, "operand1", &builder, &operand1);
  auto operand2_param =
      CreateR0Parameter<float>(12.7f, 2, "operand2", &builder, &operand2);
  Conditional(pred, operand1, CreateR0CeilComputation(), operand2,
              CreateR0FloorComputation());

  ComputeAndCompareR0<float>(&builder, 57.0f,
                             {&pred_arg, &operand1_param, &operand2_param},
                             kErrorSpec);
}

// Test conditional that takes in array operands in the form of external params.
TEST_F(ConditionalOpTest, ArrayOperandsFromExternalParams) {
  Shape r0bool = ShapeUtil::MakeShape(PRED, {});
  XlaBuilder builder(TestName());

  XlaOp pred, operand1, operand2;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operand1_param = CreateR1Parameter<float>({24.3f, 56.7f}, 1, "operand1",
                                                 &builder, &operand1);
  auto operand2_param = CreateR1Parameter<float>({10.2f, 11.6f}, 2, "operand2",
                                                 &builder, &operand2);
  Conditional(pred, operand1, CreateR1CeilComputation(), operand2,
              CreateR1FloorComputation());

  ComputeAndCompareR1<float>(&builder, {10.0f, 11.0f},
                             {&pred_arg, &operand1_param, &operand2_param},
                             kErrorSpec);
}

// Test the case where one conditional is nested within another.
TEST_F(ConditionalOpTest, NestedConditionals) {
  XlaBuilder inner_builder(TestName() + ".inner_conditional");
  {
    Shape r0bool = ShapeUtil::MakeShape(PRED, {});
    Shape tuple_shape = ShapeUtil::MakeTupleShape({r0bool, r0f32_, r0f32_});
    auto param0 = Parameter(&inner_builder, 0, tuple_shape, "param0");
    auto pred_cond = GetTupleElement(param0, 0);
    auto true_operand = GetTupleElement(param0, 1);
    auto false_operand = GetTupleElement(param0, 2);
    Conditional(pred_cond, true_operand, CreateR0CeilComputation(),
                false_operand, CreateR0FloorComputation());
  }
  auto inner_builder_result = inner_builder.Build();
  EXPECT_IS_OK(inner_builder_result.status());

  XlaBuilder builder(TestName());
  XlaOp pred1, pred2;
  auto pred1_arg = CreateR0Parameter<bool>(true, 0, "pred1", &builder, &pred1);
  auto pred2_arg = CreateR0Parameter<bool>(false, 1, "pred2", &builder, &pred2);
  auto operand1 = ConstantR0<float>(&builder, 1.1f);
  auto operand2 = ConstantR0<float>(&builder, 12.2f);
  auto operand3 = ConstantR0<float>(&builder, 43.3f);
  auto tuple_operand = Tuple(&builder, {pred2, operand1, operand2});
  Conditional(pred1, tuple_operand, std::move(inner_builder_result).value(),
              operand3, CreateR0IdentityComputation());

  ComputeAndCompareR0<float>(&builder, 12.0f, {&pred1_arg, &pred2_arg},
                             kErrorSpec);
}

TEST_F(ConditionalOpTest, ConditionalInNestedComputation) {
  XlaBuilder inner_builder(TestName() + ".inner_conditional");
  {
    Shape r0bool = ShapeUtil::MakeShape(PRED, {});
    Shape tuple_shape = ShapeUtil::MakeTupleShape({r0bool, r0f32_, r0f32_});
    auto param0 = Parameter(&inner_builder, 0, tuple_shape, "param0");
    auto pred_cond = GetTupleElement(param0, 0);
    auto true_operand = GetTupleElement(param0, 1);
    auto false_operand = GetTupleElement(param0, 2);
    Conditional(pred_cond, true_operand, CreateR0CeilComputation(),
                false_operand, CreateR0FloorComputation());
  }
  auto inner_builder_result = inner_builder.Build();
  EXPECT_IS_OK(inner_builder_result.status());

  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operand1 = ConstantR0<float>(&builder, 1.1f);
  auto operand2 = ConstantR0<float>(&builder, 12.2f);
  auto tuple_operand = Tuple(&builder, {pred, operand1, operand2});
  Call(&builder, std::move(inner_builder_result).value(), {tuple_operand});

  ComputeAndCompareR0<float>(&builder, 12.0f, {&pred_arg}, kErrorSpec);
}

// Test a mismatch in the shape of the true operand and true computation.
TEST_F(ConditionalOpTest, ShapeMismatch) {
  XlaBuilder builder(TestName());
  auto pred = ConstantR0<bool>(&builder, true);
  auto operand1 = ConstantR0<float>(&builder, 56.0f);
  auto operand2 = ConstantR0<float>(&builder, 12.0f);
  auto operands = Tuple(&builder, {operand1, operand2});
  Conditional(pred, operands, CreateR1TupleAddComputation(), operands,
              CreateR0TupleSubComputation());

  auto result = builder.Build();
  EXPECT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(),
              ::testing::HasSubstr("operand 0 must match the shape of the "
                                   "only parameter of branch computation 0"));
}

TEST_F(ConditionalOpTest, SwappedInputsInSequentialConditionals) {
  Shape tuple_shape = ShapeUtil::MakeTupleShape({r0f32_, r0f32_});
  XlaComputation swapper;
  {
    XlaBuilder builder(TestName() + ".swapper");
    auto param0 = Parameter(&builder, 0, tuple_shape, "sp0");
    auto x = GetTupleElement(param0, 0);
    auto y = GetTupleElement(param0, 1);
    Tuple(&builder, {y, x});
    swapper = builder.Build().value();
  }
  XlaComputation forwarder;
  {
    XlaBuilder builder(TestName() + ".forwarder");
    auto param0 = Parameter(&builder, 0, tuple_shape, "fp0");
    auto x = GetTupleElement(param0, 0);
    auto y = GetTupleElement(param0, 1);
    Tuple(&builder, {x, y});
    forwarder = builder.Build().value();
  }
  XlaComputation main;
  {
    XlaBuilder builder(TestName() + ".main");
    auto param0 = Parameter(&builder, 0, tuple_shape, "mp0");
    auto x = GetTupleElement(param0, 0);
    auto y = GetTupleElement(param0, 1);
    auto lt_pred = Lt(x, y);
    auto res = Conditional(lt_pred, param0, forwarder, param0, swapper);
    auto ge_pred = Ge(x, y);
    Conditional(ge_pred, res, swapper, res, forwarder);
    main = builder.Build().value();
  }

  auto test_swap = [&](float a, float b) {
    XlaBuilder builder(TestName());
    XlaOp x, y;
    auto x_arg = CreateR0Parameter<float>(a, 0, "x", &builder, &x);
    auto y_arg = CreateR0Parameter<float>(b, 1, "y", &builder, &y);
    auto tuple_operand = Tuple(&builder, {x, y});
    Call(&builder, main, {tuple_operand});
    ComputeAndCompareLiteral(
        &builder,
        LiteralUtil::MakeTupleFromSlices(
            {LiteralUtil::CreateR0<float>(a), LiteralUtil::CreateR0<float>(b)}),
        {&x_arg, &y_arg}, kErrorSpec);
  };
  test_swap(3.11f, 9.4f);
  test_swap(11.24f, 5.55f);
}

// Test conditional that duplicates tuple elements in the then and else
// computations. This is a regression test for b/112550242.
TEST_F(ConditionalOpTest, DuplicateElementsConditional) {
  const Shape scalar = ShapeUtil::MakeShape(S32, {});
  const Shape tuple2 = ShapeUtil::MakeTupleShape({scalar, scalar});
  XlaComputation then_comp;
  {
    XlaBuilder builder(TestName() + ".then");
    auto p = Parameter(&builder, 0, tuple2, "then.p");
    auto e0 = GetTupleElement(p, 0);
    auto e1 = GetTupleElement(p, 1);
    Tuple(&builder, {e0, e1, e0});
    then_comp = builder.Build().value();
  }
  XlaComputation else_comp;
  {
    XlaBuilder builder(TestName() + ".else");
    auto p = Parameter(&builder, 0, tuple2, "else.p");
    auto e0 = GetTupleElement(p, 0);
    auto e1 = GetTupleElement(p, 1);
    Tuple(&builder, {e0, e1, e1});
    else_comp = builder.Build().value();
  }

  {
    // Pred is true case.
    std::vector<Literal> args;
    args.push_back(LiteralUtil::MakeTupleFromSlices(
        {LiteralUtil::CreateR0<int32_t>(123),
         LiteralUtil::CreateR0<int32_t>(-42)}));
    args.push_back(LiteralUtil::CreateR0<bool>(true));
    XlaBuilder builder(TestName() + ".main");
    auto p = Parameter(&builder, 0, tuple2, "p0");
    auto p_pred = Parameter(&builder, 1, ShapeUtil::MakeShape(PRED, {}), "p1");
    Conditional(p_pred, p, then_comp, p, else_comp);
    ComputeAndCompare(&builder, {&args[0], &args[1]});
  }
  {
    // Pred is false case.
    std::vector<Literal> args;
    args.push_back(LiteralUtil::MakeTupleFromSlices(
        {LiteralUtil::CreateR0<int32_t>(123),
         LiteralUtil::CreateR0<int32_t>(-42)}));
    args.push_back(LiteralUtil::CreateR0<bool>(false));
    XlaBuilder builder(TestName() + ".main");
    auto p = Parameter(&builder, 0, tuple2, "p0");
    auto p_pred = Parameter(&builder, 1, ShapeUtil::MakeShape(PRED, {}), "p1");
    Conditional(p_pred, p, then_comp, p, else_comp);
    ComputeAndCompare(&builder, {&args[0], &args[1]});
  }
}

using ConditionalOpHloTest = HloTestBase;

TEST_F(ConditionalOpHloTest, ParallelExecution) {
  // Test conditional works when an executable is executed in parallel.
  const char* const hlo_string = R"(
  HloModule m

  true_computation {
    param = f32[8,8] parameter(0)
    ROOT dot = f32[8,8] dot(param, param), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  }

  false_computation {
    param = f32[8,8] parameter(0)
    ROOT dot = f32[8,8] dot(param, param), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  }

  ENTRY entry_computation {
    p = pred[] parameter(0)
    x = f32[8,8] parameter(1)
    ROOT conditional = f32[8,8] conditional(p, x, x), true_computation=true_computation, false_computation=false_computation
  }
  )";

  // Create literal where even rows are 1.0 and odd rows are 0.0.
  Array2D<float> input_array(8, 8);
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      input_array(i, j) = (i % 2 == 0) ? 1.0f : 0.0f;
    }
  }
  Literal input_literal = LiteralUtil::CreateR2FromArray2D(input_array);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CreateExecutable(std::move(module), /*run_hlo_passes=*/true));

  TF_ASSERT_OK_AND_ASSIGN(
      Literal true_result,
      test_runner().ExecuteWithExecutable(
          executable.get(),
          {LiteralUtil::CreateR0<bool>(true), input_literal.Clone()}));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal false_result,
      test_runner().ExecuteWithExecutable(
          executable.get(),
          {LiteralUtil::CreateR0<bool>(false), input_literal.Clone()}));

  constexpr int kNumThreads = 50;
  std::vector<Literal> results(kNumThreads);
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(),
                                        "conditional_test_pool", kNumThreads);
    for (int i = 0; i < kNumThreads; ++i) {
      thread_pool.Schedule([this, i, &input_literal, &executable, &results]() {
        TF_ASSERT_OK_AND_ASSIGN(
            results[i],
            test_runner().ExecuteWithExecutable(
                executable.get(), {LiteralUtil::CreateR0<bool>(i % 2 == 1),
                                   input_literal.Clone()}));
      });
    }
  }
  // Threadpool destructor waits for all threads to finish
  for (int i = 0; i < kNumThreads; ++i) {
    if (i % 2 == 1) {
      ASSERT_EQ(results[i], true_result) << "i: " << i;
    } else {
      ASSERT_EQ(results[i], false_result) << "i: " << i;
    }
  }
}

}  // namespace
}  // namespace xla
