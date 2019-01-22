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

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"

#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

#include "tensorflow/compiler/plugin/poplar/tests/test_utils.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

class InfeedTest : public ClientLibraryTestBase {
 protected:
  // Transfers the given literal to the infeed interface of the device, and
  // check if the returned data from Infeed HLO is same as the literal.
  void TestInfeedRoundTrip(const Literal& literal) {
    ASSERT_IS_OK(client_->TransferToInfeed(literal));

    XlaBuilder builder(TestName());
    Infeed(&builder, literal.shape());
    if (literal.shape().IsTuple()) {
      ComputeAndCompareTuple(&builder, literal, {});
    } else {
      ComputeAndCompareLiteral(&builder, literal, {});
    }
  }

  Status TestInfeedTransfer(const Literal& literal) {
    return client_->TransferToInfeed(literal);
  }

  ~InfeedTest() {
    auto* manager = GetXfeedManager(0);
    manager->Reset();
  }
};

TEST_F(InfeedTest, SingleInfeedR3F32Push) {
  auto literal =
      LiteralUtil::CreateR3({{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
                             {{1.1f, 2.1f, 3.1f}, {6.1f, 3.5f, 2.8f}}});
  ASSERT_IS_OK(TestInfeedTransfer(literal));
}

TEST_F(InfeedTest, SingleInfeedR3F32RoundTrip) {
  auto literal =
      LiteralUtil::CreateR3({{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
                             {{1.1f, 2.1f, 3.1f}, {6.1f, 3.5f, 2.8f}}});
  TestInfeedRoundTrip(literal);
}

// Tests Infeed operation used in a while loop, as in the code below. The
// computation is launched asynchronously, and then infeed data is transferred.
//
// float acc = 0.0f;
// while (acc < 40.0f) {
//   acc += reduce_add(Infeed());
// }
// return acc;

// NOTE(shauryas): copy paste from cpu_infeed_test.cc
TEST_F(InfeedTest, SingleInfeedInWhile) {
  XlaBuilder builder(TestName());
  const auto infeed_shape = ShapeUtil::MakeShape(F32, {3});
  const auto result_shape = ShapeUtil::MakeShape(F32, {});

  // Create a computation for the condition: repeat until (prev < 40.0f) holds.
  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    Gt(ConstantR0<float>(&builder, 40.0f), prev);
    condition = builder.Build().ConsumeValueOrDie();
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
    body = builder.Build().ConsumeValueOrDie();
  }
  // Create a While node with computations for the condition and the body.
  auto init = ConstantR0<float>(&builder, 0.0f);
  While(condition, body, init);

  // Build and asynchronously launch the computation.
  auto computation = builder.Build().ConsumeValueOrDie();
  std::unique_ptr<GlobalData> result;
  tensorflow::Thread* computation_thread =
      tensorflow::Env::Default()->StartThread(
          tensorflow::ThreadOptions{}, "computation_thread", [&] {
            result = client_->Execute(computation, {}, &execution_options_)
                         .ValueOrDie();
          });

  // Send 5 Infeed data of shape F32[3].
  ASSERT_IS_OK(
      client_->TransferToInfeed(LiteralUtil::CreateR1<float>({1, 2, 3})));
  ASSERT_IS_OK(
      client_->TransferToInfeed(LiteralUtil::CreateR1<float>({4, 5, 6})));
  ASSERT_IS_OK(
      client_->TransferToInfeed(LiteralUtil::CreateR1<float>({7, 8, 9})));
  ASSERT_IS_OK(
      client_->TransferToInfeed(LiteralUtil::CreateR1<float>({10, 11, 12})));
  ASSERT_IS_OK(
      client_->TransferToInfeed(LiteralUtil::CreateR1<float>({13, 14, 15})));

  delete computation_thread;  // Joins the thread.
  auto result_literal = client_->Transfer(*result).ConsumeValueOrDie();

  // Only the first 3 infeed data should be added.
  LiteralTestUtil::ExpectR0Near<float>(45.0f, result_literal, ErrorSpec{1e-7});
}

TEST_F(InfeedTest, SingleInfeedTuple) {
  TestInfeedRoundTrip(LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f}),
       LiteralUtil::CreateR1<float>({4.0f, 5.0f, 6.0f})}));
}

TEST_F(InfeedTest, SingleInfeedEmptyTuple) {
  TestInfeedRoundTrip(LiteralUtil::MakeTuple({}));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
