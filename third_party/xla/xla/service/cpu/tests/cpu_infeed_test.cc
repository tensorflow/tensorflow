/* Copyright 2018 The OpenXLA Authors.

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

#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_runner_interface.h"
#ifndef _WIN32
#include <unistd.h>
#endif

#include <cstdint>
#include <memory>

#include "xla/error_spec.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class InfeedTest : public ClientLibraryTestRunnerMixin<
                       HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>> {
 protected:
  // Transfers the given literal to the infeed interface of the device, and
  // check if the returned data from Infeed HLO is same as the literal.
  void TestInfeedRoundTrip(const Literal& literal) {
    XlaBuilder builder(TestName());
    Infeed(&builder, literal.shape());
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> module,
        HloModuleFromXlaBuilder(&builder, execution_options()));
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<OpaqueExecutable> executable,
        CreateExecutable(std::move(module), /*run_hlo_passes=*/true));

    HloRunnerInterface::ReplicatedExecuteOptions options;
    options.num_devices = 1;
    options.infeed_steps = 1;
    options.infeed_values = {&literal};
    ASSERT_OK_AND_ASSIGN(
        std::vector<Literal> result,
        test_runner().ExecuteReplicated(
            [executable = executable.get()](int64_t) { return executable; },
            [](int64_t) { return 0; }, [](int64_t, int64_t) { return nullptr; },
            options, nullptr));
    CHECK_EQ(result.size(), 1);

    EXPECT_TRUE(LiteralTestUtil::Near(literal, result[0], kDefaultErrorSpec));
  }
};

TEST_F(InfeedTest, SingleInfeedR0Bool) {
  TestInfeedRoundTrip(LiteralUtil::CreateR0<bool>(true));
}

TEST_F(InfeedTest, SingleInfeedR1U32) {
  TestInfeedRoundTrip(LiteralUtil::CreateR1<uint32_t>({1, 2, 3}));
}

TEST_F(InfeedTest, SingleInfeedR2F32) {
  TestInfeedRoundTrip(LiteralUtil::CreateR2F32Linspace(0.0, 1.0, 128, 64));
}

TEST_F(InfeedTest, SingleInfeedR3F32) {
  TestInfeedRoundTrip(
      LiteralUtil::CreateR3({{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
                             {{1.1f, 2.1f, 3.1f}, {6.1f, 3.5f, 2.8f}}}));
}

TEST_F(InfeedTest, SingleInfeedR3F32DifferentLayout) {
  const Layout r3_dim0minor = LayoutUtil::MakeLayout({0, 1, 2});
  const Layout r3_dim0major = LayoutUtil::MakeLayout({2, 1, 0});

  ASSERT_NO_FATAL_FAILURE(TestInfeedRoundTrip(LiteralUtil::CreateR3WithLayout(
      {{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
       {{1.1f, 2.1f, 3.1f}, {6.1f, 3.5f, 2.8f}}},
      r3_dim0minor)));

  TestInfeedRoundTrip(LiteralUtil::CreateR3WithLayout(
      {{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
       {{1.1f, 2.1f, 3.1f}, {6.1f, 3.5f, 2.8f}}},
      r3_dim0major));
}

TEST_F(InfeedTest, SingleInfeedR4S32) {
  TestInfeedRoundTrip(LiteralUtil::CreateR4(
      {{{{1, -2}, {-4, 5}, {6, 7}}, {{8, 9}, {10, 11}, {12, 13}}},
       {{{10, 3}, {7, -2}, {3, 6}}, {{2, 5}, {-11, 5}, {-2, -5}}}}));
}

TEST_F(InfeedTest, SingleInfeedTuple) {
  TestInfeedRoundTrip(LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR1<uint32_t>({1, 2, 3}),
       LiteralUtil::CreateR0<bool>(false)}));
}

TEST_F(InfeedTest, SingleInfeedEmptyTuple) {
  TestInfeedRoundTrip(LiteralUtil::MakeTuple({}));
}

// Tests Infeed operation used in a while loop, as in the code below. The
// computation is launched asynchronously, and then infeed data is transferred.
//
// float acc = 0.0f;
// while (acc < 40.0f) {
//   acc += reduce_add(Infeed());
// }
// return acc;
TEST_F(InfeedTest, SingleInfeedInWhile) {
  XlaBuilder builder(TestName());
  const Shape infeed_shape = ShapeUtil::MakeShape(F32, {3});
  const Shape result_shape = ShapeUtil::MakeShape(F32, {});

  // Create a computation for the condition: repeat until (prev < 40.0f) holds.
  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    Gt(ConstantR0<float>(&builder, 40.0f), prev);
    condition = builder.Build().value();
  }
  // Create a computation for the body: add the reduced value of the Infeed
  // data to the result variable.
  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto infeed = Infeed(&builder, infeed_shape);
    auto addend = Reduce(infeed, ConstantR0<float>(&builder, 0.0f),
                         CreateScalarAddComputation(F32, &builder), {0});
    Add(prev, addend);
    body = builder.Build().value();
  }
  // Create a While node with computations for the condition and the body.
  auto init = ConstantR0<float>(&builder, 0.0f);
  While(condition, body, init);

  // Build and asynchronously launch the computation.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       HloModuleFromXlaBuilder(&builder, execution_options()));

  // Send 5 Infeed data of shape F32[3].
  const Literal infeed_data =
      LiteralUtil::MakeTupleOwned(LiteralUtil::CreateR1<float>({1, 2, 3}),
                                  LiteralUtil::CreateR1<float>({4, 5, 6}),
                                  LiteralUtil::CreateR1<float>({7, 8, 9}),
                                  LiteralUtil::CreateR1<float>({10, 11, 12}),
                                  LiteralUtil::CreateR1<float>({13, 14, 15}));

  HloRunnerInterface::ReplicatedExecuteOptions options;
  options.num_devices = 1;
  options.infeed_steps = 1;
  options.infeed_values = {&infeed_data};
  options.run_hlo_passes = true;
  ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> result_literals,
      test_runner().ExecuteReplicated(std::move(module), options));

  // Only the first 3 infeed data should be added.
  LiteralTestUtil::ExpectR0Near<float>(45.0f, result_literals[0],
                                       ErrorSpec{1e-7});
}

}  // namespace
}  // namespace xla
