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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class ParamsTest : public ClientLibraryTestBase {};

XLA_TEST_F(ParamsTest, ConstantR0F32Param) {
  XlaBuilder builder(TestName());
  Literal param0_literal = LiteralUtil::CreateR0<float>(3.14159f);
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(param0_literal).ConsumeValueOrDie();

  Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "param0");

  ComputeAndCompareR0<float>(&builder, 3.14159f, {param0_data.get()},
                             ErrorSpec(0.0001f));
}

XLA_TEST_F(ParamsTest, ConstantR1S0F32Param) {
  XlaBuilder builder(TestName());
  Literal param0_literal = LiteralUtil::CreateR1<float>({});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(param0_literal).ConsumeValueOrDie();

  Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {0}), "param0");

  ComputeAndCompareR1<float>(&builder, {}, {param0_data.get()},
                             ErrorSpec(0.01f));
}

XLA_TEST_F(ParamsTest, ConstantR1S2F32Param) {
  XlaBuilder builder(TestName());
  Literal param0_literal = LiteralUtil::CreateR1<float>({3.14f, -100.25f});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(param0_literal).ConsumeValueOrDie();

  Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {2}), "param0");

  ComputeAndCompareR1<float>(&builder, {3.14f, -100.25f}, {param0_data.get()},
                             ErrorSpec(0.01f));
}

XLA_TEST_F(ParamsTest, ConstantR1U8Param) {
  XlaBuilder builder(TestName());
  std::string str("hello world");
  Literal param0_literal = LiteralUtil::CreateR1U8(str);
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(param0_literal).ConsumeValueOrDie();

  Parameter(&builder, 0,
            ShapeUtil::MakeShape(U8, {static_cast<int64_t>(str.size())}),
            "param0");

  ComputeAndCompareR1U8(&builder, str, {param0_data.get()});
}

XLA_TEST_F(ParamsTest, ConstantR2_3x0_F32Param) {
  XlaBuilder builder(TestName());
  Literal param0_literal =
      LiteralUtil::CreateR2FromArray2D<float>(Array2D<float>(3, 0));
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(param0_literal).ConsumeValueOrDie();

  Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {3, 0}), "param0");

  ComputeAndCompareR2<float>(&builder, Array2D<float>(3, 0),
                             {param0_data.get()}, ErrorSpec(0.01f));
}

XLA_TEST_F(ParamsTest, ConstantR2F32Param) {
  XlaBuilder builder(TestName());
  Literal param0_literal = LiteralUtil::CreateR2<float>(
      {{3.14f, -100.25f}, {7e8f, 7e-9f}, {30.3f, -100.0f}});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(param0_literal).ConsumeValueOrDie();

  Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {3, 2}), "param0");

  Array2D<float> expected_array(
      {{3.14f, -100.25f}, {7e8f, 7e-9f}, {30.3f, -100.0f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {param0_data.get()},
                             ErrorSpec(0.01f));
}

XLA_TEST_F(ParamsTest, TwoParameters) {
  XlaBuilder builder(TestName());

  Literal literal0 = LiteralUtil::CreateR1<float>({1, 2});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(literal0).ConsumeValueOrDie();
  auto param0 = Parameter(&builder, 0, literal0.shape(), "param0");

  Literal literal1 = LiteralUtil::CreateR1<float>({10, 20});
  std::unique_ptr<GlobalData> param1_data =
      client_->TransferToServer(literal1).ConsumeValueOrDie();
  auto param1 = Parameter(&builder, 1, literal1.shape(), "param1");

  // Use both parameters
  //
  // {1, 2} + {10, 20} = {11, 22}
  auto sum = Add(param0, param1);
  sum = Add(param0, param1);

  // Use only the second parameter again, to show that it can be used
  // twice and to make the computation asymmetric in the two
  // parameters to test that the parameters are not swapped.
  //
  // {11, 22} * {10, 20} = {110, 440}
  Mul(sum, param1);

  ComputeAndCompareR1<float>(&builder, {110, 440},
                             {param0_data.get(), param1_data.get()},
                             ErrorSpec(0.0001f));
}

XLA_TEST_F(ParamsTest, MissingParameter) {
  // Test that an error is returned when a computation with an incomplete set of
  // parameters (parameter numbers not contiguous from 0) is executed.
  Literal literal = LiteralUtil::CreateR0<float>(3.14159f);
  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(literal).ConsumeValueOrDie();

  XlaBuilder builder(TestName());
  Parameter(&builder, 2, ShapeUtil::MakeShape(F32, {}), "param2");
  auto computation_status = builder.Build();

  ASSERT_NE(computation_status.status(), Status::OK());
}

XLA_TEST_F(ParamsTest, UnusedParameter) {
  XlaBuilder builder(TestName());

  Literal literal0 = LiteralUtil::CreateR1<float>({1, 2});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(literal0).ConsumeValueOrDie();
  Parameter(&builder, 0, literal0.shape(), "param0");

  Literal literal1 = LiteralUtil::CreateR1<float>({10, 20});
  std::unique_ptr<GlobalData> param1_data =
      client_->TransferToServer(literal1).ConsumeValueOrDie();
  Parameter(&builder, 1, literal1.shape(), "param1");

  ComputeAndCompareR1<float>(&builder, {10, 20},
                             {param0_data.get(), param1_data.get()},
                             ErrorSpec(0.0001f));
}

XLA_TEST_F(ParamsTest, UnusedParametersInUnusedExpression) {
  // Build a computation with a couple unused parameters which are used in an
  // unused expression.
  XlaBuilder builder(TestName());

  Literal literal0 = LiteralUtil::CreateR1<float>({1, 2});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(literal0).ConsumeValueOrDie();

  Literal literal1 = LiteralUtil::CreateR1<float>({10, 20, 30});
  std::unique_ptr<GlobalData> param1_data =
      client_->TransferToServer(literal1).ConsumeValueOrDie();

  auto param0 = Parameter(&builder, 0, literal0.shape(), "param0");
  auto param1 = Parameter(&builder, 1, literal1.shape(), "param1");
  auto param2 = Parameter(&builder, 2, literal1.shape(), "param2");

  // This add is unused.
  Add(param1, param2);

  Neg(param0);

  ComputeAndCompareR1<float>(
      &builder, {-1, -2},
      {param0_data.get(), param1_data.get(), param1_data.get()},
      ErrorSpec(0.0001f));
}

XLA_TEST_F(ParamsTest, HundredLargeR1Parameters) {
  XlaBuilder builder(TestName());
  constexpr int size = 8 * 128 * 2;

  std::vector<float> init_value = {{0, 1}};
  init_value.resize(size);
  XlaOp sum_handle = ConstantR1<float>(&builder, init_value);
  std::vector<float> sum = {{0, 1}};
  sum.resize(size);

  std::vector<std::unique_ptr<GlobalData>> param_data_owner;

  constexpr int parameter_count = 100;
  for (int i = 0; i < parameter_count; ++i) {
    const float entry0 = i;
    const float entry1 = 2 * i;
    sum[0] += entry0;
    sum[1] += entry1;

    std::vector<float> sum_value = {{entry0, entry1}};
    sum_value.resize(size);
    Literal literal = LiteralUtil::CreateR1<float>(sum_value);
    param_data_owner.push_back(
        client_->TransferToServer(literal).ConsumeValueOrDie());
    XlaOp param = Parameter(&builder, i, literal.shape(), "param");
    sum_handle = Add(sum_handle, param);
  }

  std::vector<GlobalData*> param_data;
  param_data.reserve(param_data_owner.size());
  for (const std::unique_ptr<GlobalData>& data : param_data_owner) {
    param_data.push_back(data.get());
  }

  ComputeAndCompareR1<float>(&builder, sum, param_data, ErrorSpec(0.0001f));
}

// Only run the 3,000-parameter tests in opt mode to avoid test timeouts.
// Timeout last observed on 2017-11-20.
#ifdef NDEBUG

// TODO(b/65525254) Fails on GPU on 2017-09-10 because we try to reserve too
// much space in parameter memory for the kernel.
//
// TODO(b/65526061) Failed on CPU on 2017-09-10 due to timeout in LLVM
// compilation.
XLA_TEST_F(ParamsTest,
           DISABLED_ON_CPU(DISABLED_ON_GPU(ThreeThousandParameters))) {
  XlaBuilder builder(TestName());

  std::vector<std::unique_ptr<GlobalData>> param_data_owner;
  XlaOp sum_handle = ConstantR0<float>(&builder, 0.0f);
  float target = 0.0;
  constexpr int kParamCount = 3000;
  for (int i = 0; i < kParamCount; ++i) {
    target += i;
    Literal literal = LiteralUtil::CreateR0<float>(i);
    param_data_owner.push_back(
        std::move(client_->TransferToServer(literal)).ValueOrDie());
    XlaOp param = Parameter(&builder, i, literal.shape(), "param");
    sum_handle = Add(sum_handle, param);
  }

  std::vector<GlobalData*> param_data;
  param_data.reserve(param_data_owner.size());
  for (const std::unique_ptr<GlobalData>& data : param_data_owner) {
    param_data.push_back(data.get());
  }

  ComputeAndCompareR0<float>(&builder, target, param_data, ErrorSpec(0.0001f));
}

// TODO(b/65525254) Fails on GPU on 2017-09-10 because we try to reserve too
// much space in parameter memory for the kernel.
//
// TODO(b/65526061) Failed on CPU on 2017-09-10 due to timeout in LLVM
// compilation.
XLA_TEST_F(ParamsTest, DISABLED_ON_CPU(DISABLED_ON_GPU(
                           ThreeThousandParametersAndOutputElements))) {
  XlaBuilder builder(TestName());

  std::vector<std::unique_ptr<GlobalData>> param_data_owner;
  XlaOp sum_handle = ConstantR1<int32_t>(&builder, {0, 0});
  int32_t target = 0;
  constexpr int kParamCount = 3000;
  std::vector<XlaOp> params;
  param_data_owner.reserve(kParamCount);
  params.reserve(kParamCount);
  for (int i = 0; i < kParamCount; ++i) {
    target += i;
    Literal literal = LiteralUtil::CreateR1<int32_t>({i, i});
    param_data_owner.push_back(
        std::move(client_->TransferToServer(literal)).ValueOrDie());
    XlaOp param = Parameter(&builder, i, literal.shape(), "param");
    params.push_back(param);
    sum_handle = Add(sum_handle, param);
  }

  std::vector<XlaOp> outputs;
  outputs.reserve(kParamCount);
  for (int i = 0; i < kParamCount; ++i) {
    outputs.push_back(Add(params[i], sum_handle));
  }

  Tuple(&builder, outputs);

  std::vector<GlobalData*> param_data;
  param_data.reserve(param_data_owner.size());
  for (const std::unique_ptr<GlobalData>& data : param_data_owner) {
    param_data.push_back(data.get());
  }

  std::vector<Literal> elements;
  std::vector<const Literal*> ptrs;
  elements.reserve(kParamCount);
  for (int i = 0; i < kParamCount; ++i) {
    elements.push_back(
        LiteralUtil::CreateR1<int32_t>({target + i, target + i}));
    ptrs.push_back(&elements.back());
  }
  ComputeAndCompareTuple(&builder, LiteralUtil::MakeTuple(ptrs), param_data);
}

// Test large number of parameters flowing into a while-loop.
// Construct conceptually the following HLO graph:
//
// p0 = parameter(0)
// p1 = parameter(1)
// ...
// pN = parameter(N)
// result = while (false) {
//   p0 += (1, 1);
//   p1 += (1, 1);
//   ...
//   pN += (1, 1)
// }
// result = {p0, p1, ..., pN}
//
// TODO(b/70173746): Times out during compilation on GPU and CPU backends as of
// 2017-12-12.
XLA_TEST_F(ParamsTest,
           DISABLED_ON_CPU(DISABLED_ON_GPU(ManyParametersIntoWhileLoop))) {
  XlaBuilder builder(TestName());

  std::vector<std::unique_ptr<GlobalData>> param_data_owner;
  constexpr int kParamCount = 1900;
  std::vector<XlaOp> params;
  std::vector<Shape> parameter_shapes;
  param_data_owner.reserve(kParamCount);
  params.reserve(kParamCount);
  parameter_shapes.reserve(kParamCount);
  for (int i = 0; i < kParamCount; ++i) {
    Literal literal = LiteralUtil::CreateR1<int32_t>({i, i});
    param_data_owner.push_back(
        std::move(client_->TransferToServer(literal)).ValueOrDie());
    XlaOp param = Parameter(&builder, i, literal.shape(), "param");
    params.push_back(param);
    parameter_shapes.push_back(literal.shape());
  }

  // Add bool parameter for the loop condition. Use a parameter HLO instead of a
  // constant because DCE may eliminate the while-body otherwise.
  Literal bool_literal = LiteralUtil::CreateR0<bool>(false);
  param_data_owner.push_back(
      std::move(client_->TransferToServer(bool_literal)).ValueOrDie());
  XlaOp bool_param =
      Parameter(&builder, kParamCount, bool_literal.shape(), "bool_param");
  params.push_back(bool_param);
  parameter_shapes.push_back(bool_literal.shape());

  auto init = Tuple(&builder, params);

  // Create a computation for the condition: while(bool_param).
  Shape while_shape = ShapeUtil::MakeTupleShape(parameter_shapes);
  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto condition_parameter =
        Parameter(&builder, 0, while_shape, "condition_parameter");
    GetTupleElement(condition_parameter, kParamCount);
    condition = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the body.
  // Add {1, 1} to the each tuple element.
  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto body_parameter = Parameter(&builder, 0, while_shape, "body_parameter");
    std::vector<XlaOp> updates;
    updates.reserve(kParamCount + 1);
    for (int i = 0; i < kParamCount; ++i) {
      auto add = Add(GetTupleElement(body_parameter, i),
                     ConstantR1<int32_t>(&builder, {1, 1}));
      updates.push_back(add);
    }
    // Add bool parameter.
    updates.push_back(GetTupleElement(body_parameter, kParamCount));

    Tuple(&builder, updates);
    body = builder.Build().ConsumeValueOrDie();
  }

  auto loop = While(condition, body, init);

  std::vector<XlaOp> outputs;
  outputs.reserve(kParamCount);
  for (int i = 0; i < kParamCount; ++i) {
    outputs.push_back(GetTupleElement(loop, i));
  }
  Tuple(&builder, outputs);

  std::vector<GlobalData*> param_data;
  param_data.reserve(param_data_owner.size());
  for (const std::unique_ptr<GlobalData>& data : param_data_owner) {
    param_data.push_back(data.get());
  }

  std::vector<Literal> elements;
  std::vector<const Literal*> ptrs;
  elements.reserve(kParamCount);
  for (int i = 0; i < kParamCount; ++i) {
    elements.push_back(LiteralUtil::CreateR1<int32_t>({i, i}));
    ptrs.push_back(&elements.back());
  }
  ComputeAndCompareTuple(&builder, LiteralUtil::MakeTuple(ptrs), param_data);
}

#endif

XLA_TEST_F(ParamsTest, TupleOfR1ParametersAddedTogether) {
  XlaBuilder builder(TestName());

  Shape r1f32_3 = ShapeUtil::MakeShape(F32, {3});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({r1f32_3, r1f32_3});
  auto input = Parameter(&builder, 0, tuple_shape, "input");
  auto lhs = GetTupleElement(input, 0);
  auto rhs = GetTupleElement(input, 1);
  Add(lhs, rhs);

  std::unique_ptr<GlobalData> data =
      client_
          ->TransferToServer(LiteralUtil::MakeTupleFromSlices({
              LiteralUtil::CreateR1<float>({1, 2, 3}),
              LiteralUtil::CreateR1<float>({4, 5, 6}),
          }))
          .ConsumeValueOrDie();

  std::vector<GlobalData*> arguments = {data.get()};
  const std::vector<float> expected = {1 + 4, 2 + 5, 3 + 6};
  ComputeAndCompareR1<float>(&builder, expected, arguments, ErrorSpec(1e-5));
}

// Verifies that passing a 2x2 with {0, 1} layout returns the same value back
// when (transferred to the server and) passed through a parameter.
XLA_TEST_F(ParamsTest, R2_2x2_Layout_01) {
  Literal literal = LiteralUtil::CreateR2WithLayout<float>(
      {{1, 2}, {3, 4}}, LayoutUtil::MakeLayout({0, 1}));
  XlaBuilder builder(TestName());
  Parameter(&builder, 0, literal.shape(), "input");

  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(literal).ConsumeValueOrDie();
  ComputeAndCompareLiteral(&builder, literal, {data.get()}, ErrorSpec(1e-3));
}

// As above, but for {1, 0} layout.
XLA_TEST_F(ParamsTest, R2_2x2_Layout_10) {
  Literal literal = LiteralUtil::CreateR2WithLayout<float>(
      {{1, 3}, {2, 4}}, LayoutUtil::MakeLayout({1, 0}));
  XlaBuilder builder(TestName());
  Parameter(&builder, 0, literal.shape(), "input");

  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(literal).ConsumeValueOrDie();
  ComputeAndCompareLiteral(&builder, literal, {data.get()}, ErrorSpec(1e-3));
}

XLA_TEST_F(ParamsTest, R2_2x2_TryToPassReverseLayoutToParameter) {
  Literal literal = LiteralUtil::CreateR2<float>({
      {1, 3},
      {2, 4},
  });
  const Shape original = literal.shape();
  {
    // Reverse the layout present in original, and make that the layout of the
    // literal.
    std::vector<int64_t> original_layout(
        original.layout().minor_to_major().begin(),
        original.layout().minor_to_major().end());
    std::reverse(original_layout.begin(), original_layout.end());
    *literal.mutable_shape_do_not_use()->mutable_layout() =
        LayoutUtil::MakeLayout(original_layout);
    ASSERT_EQ(2, literal.Get<float>({0, 1}));
  }
  // Use the original shape in building the computation.
  XlaBuilder builder(TestName());
  auto input = Parameter(&builder, 0, original, "input");
  // Use the slice operator to get an off-diagonal element.
  Slice(input, {0, 1}, {1, 2}, {1, 1});

  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(literal).ConsumeValueOrDie();
  // Check that we got the off-diagonal value that we expected.
  Array2D<float> expected(1, 1);
  expected(0, 0) = 2;
  ComputeAndCompareR2(&builder, expected, {data.get()}, ErrorSpec(1e-3));
}

}  // namespace
}  // namespace xla
