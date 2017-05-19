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

#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
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
  Computation CreateR0F32IdentityComputation() {
    ComputationBuilder builder(client_, "Identity");
    builder.Parameter(0, r0f32_, "x");
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  Computation CreateR1S0F32AdditionComputation() {
    ComputationBuilder builder(client_, "Addition");
    auto x = builder.Parameter(0, r1s0f32_, "x");
    auto y = builder.Parameter(1, r1s0f32_, "y");
    builder.Add(x, y);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  Computation CreateR1S2F32AdditionComputation() {
    ComputationBuilder builder(client_, "Addition");
    auto x = builder.Parameter(0, r1s2f32_, "x");
    auto y = builder.Parameter(1, r1s2f32_, "y");
    builder.Add(x, y);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  Computation CreateR0F32TupleComputation() {
    ComputationBuilder builder(client_, "Tuple");
    builder.Tuple({builder.Parameter(0, r0f32_, "x")});
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  Shape r0f32_ = ShapeUtil::MakeShape(F32, {});
  Shape r1s0f32_ = ShapeUtil::MakeShape(F32, {0});
  Shape r1s2f32_ = ShapeUtil::MakeShape(F32, {2});
};

XLA_TEST_F(CallOpTest, DISABLED_ON_GPU(CallR0F32IdentityScalar)) {
  ComputationBuilder builder(client_, TestName());
  Computation callee = CreateR0F32IdentityComputation();
  auto constant = builder.ConstantLiteral(*LiteralUtil::CreateR0<float>(42.0));
  builder.Call(callee, {constant});

  ComputeAndCompareR0<float>(&builder, 42.0, {}, ErrorSpec(0.01f));
}

XLA_TEST_F(CallOpTest, DISABLED_ON_GPU(CallR1S0F32AddArray)) {
  ComputationBuilder builder(client_, TestName());
  Computation callee = CreateR1S0F32AdditionComputation();
  auto x = builder.ConstantLiteral(*LiteralUtil::CreateR1<float>({}));
  auto y = builder.ConstantLiteral(*LiteralUtil::CreateR1<float>({}));
  builder.Call(callee, {x, y});

  ComputeAndCompareR1<float>(&builder, {}, {}, ErrorSpec(0.01f));
}

XLA_TEST_F(CallOpTest, DISABLED_ON_GPU(CallR1S2F32AddArray)) {
  ComputationBuilder builder(client_, TestName());
  Computation callee = CreateR1S2F32AdditionComputation();
  auto x = builder.ConstantLiteral(*LiteralUtil::CreateR1<float>({1.0f, 2.0f}));
  auto y = builder.ConstantLiteral(*LiteralUtil::CreateR1<float>({2.0f, 3.0f}));
  builder.Call(callee, {x, y});

  ComputeAndCompareR1<float>(&builder, {3.0f, 5.0f}, {}, ErrorSpec(0.01f));
}

XLA_TEST_F(CallOpTest, DISABLED_ON_GPU(CallR0F32Tuple)) {
  ComputationBuilder builder(client_, TestName());
  Computation callee = CreateR0F32TupleComputation();
  auto elem = LiteralUtil::CreateR0<float>(42.0);
  auto tuple = LiteralUtil::MakeTuple({elem.get()});
  builder.Call(callee, {builder.ConstantLiteral(*elem)});

  ComputeAndCompareTuple(&builder, *tuple, {}, ErrorSpec(0.01f));
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendCpuCompilerFlags(&flag_list);
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }
  testing::InitGoogleTest(&argc, argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return 2;
  }
  return RUN_ALL_TESTS();
}
