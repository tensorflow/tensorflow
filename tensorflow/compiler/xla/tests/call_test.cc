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

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class CallOpTest : public ClientLibraryTestBase {
 protected:
  XlaComputation CreateR0F32IdentityComputation() {
    XlaBuilder builder("Identity");
    Parameter(&builder, 0, r0f32_, "x");
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return std::move(build_status).value();
  }

  XlaComputation CreateR1S0F32AdditionComputation() {
    XlaBuilder builder("Addition");
    auto x = Parameter(&builder, 0, r1s0f32_, "x");
    auto y = Parameter(&builder, 1, r1s0f32_, "y");
    Add(x, y);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return std::move(build_status).value();
  }

  XlaComputation CreateR1S2F32AdditionComputation() {
    XlaBuilder builder("Addition");
    auto x = Parameter(&builder, 0, r1s2f32_, "x");
    auto y = Parameter(&builder, 1, r1s2f32_, "y");
    Add(x, y);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return std::move(build_status).value();
  }

  XlaComputation CreateR0F32TupleComputation() {
    XlaBuilder builder("Tuple");
    Tuple(&builder, {Parameter(&builder, 0, r0f32_, "x")});
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return std::move(build_status).value();
  }

  Shape r0f32_ = ShapeUtil::MakeShape(F32, {});
  Shape r1s0f32_ = ShapeUtil::MakeShape(F32, {0});
  Shape r1s2f32_ = ShapeUtil::MakeShape(F32, {2});
};

XLA_TEST_F(CallOpTest, CallR0F32IdentityScalar) {
  XlaBuilder builder(TestName());
  XlaComputation callee = CreateR0F32IdentityComputation();
  auto constant = ConstantLiteral(&builder, LiteralUtil::CreateR0<float>(42.0));
  Call(&builder, callee, {constant});

  ComputeAndCompareR0<float>(&builder, 42.0, {}, ErrorSpec(0.01f));
}

XLA_TEST_F(CallOpTest, CallR1S0F32AddArray) {
  XlaBuilder builder(TestName());
  XlaComputation callee = CreateR1S0F32AdditionComputation();
  auto x = ConstantLiteral(&builder, LiteralUtil::CreateR1<float>({}));
  auto y = ConstantLiteral(&builder, LiteralUtil::CreateR1<float>({}));
  Call(&builder, callee, {x, y});

  ComputeAndCompareR1<float>(&builder, {}, {}, ErrorSpec(0.01f));
}

XLA_TEST_F(CallOpTest, CallR1S2F32AddArray) {
  XlaBuilder builder(TestName());
  XlaComputation callee = CreateR1S2F32AdditionComputation();
  auto x =
      ConstantLiteral(&builder, LiteralUtil::CreateR1<float>({1.0f, 2.0f}));
  auto y =
      ConstantLiteral(&builder, LiteralUtil::CreateR1<float>({2.0f, 3.0f}));
  Call(&builder, callee, {x, y});

  ComputeAndCompareR1<float>(&builder, {3.0f, 5.0f}, {}, ErrorSpec(0.01f));
}

XLA_TEST_F(CallOpTest, CallTreeTwoDeepBranchFactorThree) {
  XlaBuilder builder("inner");
  {
    auto x = Parameter(&builder, 0, r0f32_, "x");
    Add(x, ConstantR0<float>(&builder, 1.0));
  }
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation inner, builder.Build());

  XlaBuilder builder2("outer");
  {
    auto x = Parameter(&builder2, 0, r0f32_, "x");
    x = Call(&builder2, inner, {x});
    x = Call(&builder2, inner, {x});
    x = Call(&builder2, inner, {x});
  }
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation outer, builder2.Build());

  XlaBuilder builder3("outermost");
  {
    auto x = Parameter(&builder3, 0, r0f32_, "x");
    x = Call(&builder3, outer, {x});
    x = Call(&builder3, outer, {x});
    x = Call(&builder3, outer, {x});
  }

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GlobalData> start,
      client_->TransferToServer(LiteralUtil::CreateR0<float>(1.0f)));
  ComputeAndCompareR0<float>(&builder3, 10.0f, {start.get()}, ErrorSpec(0.0f));
}

XLA_TEST_F(CallOpTest, CallR0F32Tuple) {
  XlaBuilder builder(TestName());
  XlaComputation callee = CreateR0F32TupleComputation();
  auto elem = LiteralUtil::CreateR0<float>(42.0);
  auto tuple = LiteralUtil::MakeTuple({&elem});
  Call(&builder, callee, {ConstantLiteral(&builder, elem)});

  ComputeAndCompareTuple(&builder, tuple, {}, ErrorSpec(0.01f));
}

}  // namespace
}  // namespace xla
