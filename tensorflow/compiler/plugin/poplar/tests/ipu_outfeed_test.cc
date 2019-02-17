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
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

#include "tensorflow/compiler/plugin/poplar/tests/test_utils.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/local_client_test_base.h"

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

class OutfeedTest : public ClientLibraryTestBase {};

TEST_F(OutfeedTest, OutfeedTuple) {
  XlaBuilder builder(TestName());
  auto input_literal = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f}),
       LiteralUtil::CreateR1<float>({4.0f, 5.0f, 6.0f})});
  auto shape = input_literal.shape();

  auto input = Parameter(&builder, 0, shape, "input");

  auto input_global =
      client_->TransferToServer(input_literal).ConsumeValueOrDie();
  XlaOp token = CreateToken(&builder);
  XlaOp outfeed = OutfeedWithToken(input, token, shape, "");

  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());
  std::unique_ptr<GlobalData> result;
  std::unique_ptr<tensorflow::Thread> thread(
      tensorflow::Env::Default()->StartThread(
          tensorflow::ThreadOptions(), "execute_thread", [&] {
            result = client_->Execute(computation, {input_global.get()})
                         .ConsumeValueOrDie();
          }));

  TF_ASSERT_OK_AND_ASSIGN(Literal result_literal1,
                          client_->TransferFromOutfeed(&shape));

  thread.reset();
  LiteralTestUtil::Equal(input_literal, result_literal1);
}

TEST_F(OutfeedTest, OutfeedTupleOfAdd) {
  XlaBuilder builder(TestName());

  auto input1_literal = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f}),
       LiteralUtil::CreateR1<float>({4.0f, 5.0f, 6.0f})});
  auto input2_literal = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR1<float>({1.1f, 2.1f, 3.1f}),
       LiteralUtil::CreateR1<float>({4.1f, 5.1f, 6.1f})});
  auto shape = input1_literal.shape();

  auto input1 = Parameter(&builder, 0, shape, "input1");
  auto input2 = Parameter(&builder, 1, shape, "input2");

  auto input1_global =
      client_->TransferToServer(input1_literal).ConsumeValueOrDie();
  auto input2_global =
      client_->TransferToServer(input2_literal).ConsumeValueOrDie();

  auto a1 = GetTupleElement(input1, 0);
  auto b1 = GetTupleElement(input1, 1);

  auto a2 = GetTupleElement(input2, 0);
  auto b2 = GetTupleElement(input2, 1);

  XlaOp add_result_a = Add(a1, a2);
  XlaOp add_result_b = Add(b1, b2);
  XlaOp tuple_input = Tuple(&builder, {add_result_a, add_result_b});
  Shape outfeed_shape = builder.GetShape(tuple_input).ValueOrDie();
  XlaOp token = CreateToken(&builder);
  XlaOp outfeed = OutfeedWithToken(tuple_input, token, outfeed_shape, "");

  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());
  std::unique_ptr<GlobalData> result;
  std::unique_ptr<tensorflow::Thread> thread(
      tensorflow::Env::Default()->StartThread(
          tensorflow::ThreadOptions(), "execute_thread", [&] {
            result = client_
                         ->Execute(computation,
                                   {input1_global.get(), input2_global.get()})
                         .ConsumeValueOrDie();
          }));

  TF_ASSERT_OK_AND_ASSIGN(Literal result_literal,
                          client_->TransferFromOutfeed(&shape));

  thread.reset();

  auto expected_result = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR1<float>({2.1f, 4.1f, 6.1f}),
       LiteralUtil::CreateR1<float>({8.1f, 10.1f, 12.1f})});
  LiteralTestUtil::NearOrEqual(expected_result, result_literal,
                               ErrorSpec{1e-6});
}

TEST_F(OutfeedTest, SingleOutfeed) {
  XlaBuilder builder(TestName());
  auto input1_literal = xla::LiteralUtil::CreateR1<float>({1.1f, 1.1f});
  auto input2_literal = xla::LiteralUtil::CreateR1<float>({1.2f, 1.2f});
  auto shape = input1_literal.shape();

  auto input1 = Parameter(&builder, 0, shape, "input1");
  auto input2 = Parameter(&builder, 1, shape, "input2");

  auto input1_global =
      client_->TransferToServer(input1_literal).ConsumeValueOrDie();
  auto input2_global =
      client_->TransferToServer(input2_literal).ConsumeValueOrDie();

  XlaOp add_result = Add(input1, input2);
  XlaOp outfeed_token = CreateToken(&builder);
  XlaOp outfeed = OutfeedWithToken(add_result, outfeed_token, shape, "");

  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());

  std::unique_ptr<GlobalData> result;
  std::unique_ptr<tensorflow::Thread> thread(
      tensorflow::Env::Default()->StartThread(
          tensorflow::ThreadOptions(), "execute_thread", [&] {
            result = client_
                         ->Execute(computation,
                                   {input1_global.get(), input2_global.get()})
                         .ConsumeValueOrDie();
          }));

  TF_ASSERT_OK_AND_ASSIGN(Literal result_literal,
                          client_->TransferFromOutfeed(&shape));

  thread.reset();
  LiteralTestUtil::ExpectR1Near<float>({2.3f, 2.3f}, result_literal,
                                       ErrorSpec{1e-6});
}

TEST_F(OutfeedTest, OutfeedInWhileLoop) {
  XlaBuilder builder(TestName());
  const auto counter_shape = ShapeUtil::MakeShape(S32, {});
  const auto scalar_shape = ShapeUtil::MakeShape(F32, {});
  const auto outfeed_shape = ShapeUtil::MakeShape(F32, {3});

  const auto tuple_shape =
      ShapeUtil::MakeTupleShape({counter_shape, scalar_shape, outfeed_shape});

  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto cond_tuple = Parameter(&builder, 0, tuple_shape, "cond_tuple");
    auto counter = GetTupleElement(cond_tuple, 0);
    Gt(ConstantR0<int32_t>(&builder, 5), counter);
    condition = builder.Build().ConsumeValueOrDie();
  }

  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto body_tuple = Parameter(&builder, 0, tuple_shape, "body_tuple");
    auto counter = GetTupleElement(body_tuple, 0);
    auto prev = GetTupleElement(body_tuple, 1);
    auto values = GetTupleElement(body_tuple, 2);
    auto addend = Reduce(values, ConstantR0<float>(&builder, 0.0f),
                         CreateScalarAddComputation(F32, &builder), {0});

    auto result = Add(prev, addend);
    auto added_values = Add(ConstantR0<float>(&builder, 1.0f), values);
    auto token = CreateToken(&builder);
    auto outfeed_token =
        OutfeedWithToken(added_values, token, outfeed_shape, "");
    // Outfeed token should be replaced with an empty tuple by the
    // root token replacer pass
    auto counter_inc = Add(counter, ConstantR0<int32>(&builder, 1));
    Tuple(&builder, {counter_inc, result, added_values});
    body = builder.Build().ConsumeValueOrDie();
  }

  auto init = ConstantR0<int32>(&builder, 0);
  auto accumulator = ConstantR0<float>(&builder, 0.0f);
  auto initial_values = ConstantR1<float>(&builder, {1.0f, 2.0f, 3.0f});
  auto token = CreateToken(&builder);
  auto loop_tuple = Tuple(&builder, {init, accumulator, initial_values});
  While(condition, body, loop_tuple);

  // Build and asynchronously launch the computation.
  auto computation = builder.Build().ConsumeValueOrDie();
  std::unique_ptr<GlobalData> result;
  tensorflow::Thread* computation_thread =
      tensorflow::Env::Default()->StartThread(
          tensorflow::ThreadOptions{}, "computation_thread", [&] {
            result = client_->Execute(computation, {}, &execution_options_)
                         .ValueOrDie();
          });

  TF_ASSERT_OK_AND_ASSIGN(Literal result_literal1,
                          client_->TransferFromOutfeed(&outfeed_shape));
  LiteralTestUtil::ExpectR1Near<float>({2.0f, 3.0f, 4.0f}, result_literal1,
                                       ErrorSpec{1e-6});

  TF_ASSERT_OK_AND_ASSIGN(Literal result_literal2,
                          client_->TransferFromOutfeed(&outfeed_shape));
  LiteralTestUtil::ExpectR1Near<float>({3.0f, 4.0f, 5.0f}, result_literal2,
                                       ErrorSpec{1e-6});

  TF_ASSERT_OK_AND_ASSIGN(Literal result_literal3,
                          client_->TransferFromOutfeed(&outfeed_shape));
  LiteralTestUtil::ExpectR1Near<float>({4.0f, 5.0f, 6.0f}, result_literal3,
                                       ErrorSpec{1e-6});

  TF_ASSERT_OK_AND_ASSIGN(Literal result_literal4,
                          client_->TransferFromOutfeed(&outfeed_shape));
  LiteralTestUtil::ExpectR1Near<float>({5.0f, 6.0f, 7.0f}, result_literal4,
                                       ErrorSpec{1e-6});

  TF_ASSERT_OK_AND_ASSIGN(Literal result_literal5,
                          client_->TransferFromOutfeed(&outfeed_shape));
  LiteralTestUtil::ExpectR1Near<float>({6.0f, 7.0f, 8.0f}, result_literal5,
                                       ErrorSpec{1e-6});

  delete computation_thread;  // Joins the thread.
  auto result_literal = client_->Transfer(*result).ConsumeValueOrDie();
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
