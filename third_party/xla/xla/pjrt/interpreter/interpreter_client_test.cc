/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/pjrt/interpreter/interpreter_client.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

TEST(InterpreterClientTest, EvaluateOnceShouldSucceed) {
  InterpreterClient client;
  const Shape shape = ShapeUtil::MakeShape(S32, {4});
  XlaBuilder builder("test");
  Add(Parameter(&builder, 0, shape, "parameter0"),
      ConstantR1(&builder, absl::Span<const int32_t>{1, 1, 1, 1}));
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> executable,
                          client.CompileAndLoad(computation, CompileOptions()));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> argument,
      client.BufferFromHostLiteral(
          LiteralUtil::CreateR1(absl::Span<const int32_t>{1, 2, 3, 4}),
          client.memory_spaces().front()));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> results,
      executable->Execute({{argument.get()}}, ExecuteOptions()));

  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results.front().size(), 1);
  Literal result_literal(shape);
  TF_ASSERT_OK(results.front().front()->ToLiteralSync(&result_literal));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      result_literal,
      LiteralUtil::CreateR1(absl::Span<const int32_t>{2, 3, 4, 5})));
}

TEST(InterpreterClientTest, EvaluateTwiceShouldSucceed) {
  InterpreterClient client;
  const Shape shape = ShapeUtil::MakeShape(S32, {4});
  XlaBuilder builder("test");
  Add(Parameter(&builder, 0, shape, "parameter0"),
      ConstantR1(&builder, absl::Span<const int32_t>{1, 1, 1, 1}));
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> executable,
                          client.CompileAndLoad(computation, CompileOptions()));

  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> results;
  for (const Literal& execution_argument :
       {LiteralUtil::CreateR1(absl::Span<const int32_t>{1, 2, 3, 4}),
        LiteralUtil::CreateR1(absl::Span<const int32_t>{4, 3, 2, 1})}) {
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<PjRtBuffer> argument_buffer,
        client.BufferFromHostLiteral(execution_argument,
                                     client.memory_spaces().front()));
    TF_ASSERT_OK_AND_ASSIGN(
        results.emplace_back(),
        executable->ExecuteSharded({argument_buffer.get()},
                                   client.addressable_devices().front(),
                                   ExecuteOptions()));
  }

  std::vector<Literal> expected_literals;
  expected_literals.push_back(
      LiteralUtil::CreateR1(absl::Span<const int32_t>{2, 3, 4, 5}));
  expected_literals.push_back(
      LiteralUtil::CreateR1(absl::Span<const int32_t>{5, 4, 3, 2}));

  ASSERT_EQ(results.size(), 2);
  Literal actual_literal(shape);
  for (int i = 0; i < results.size(); ++i) {
    const std::vector<std::unique_ptr<PjRtBuffer>>& actual_buffers = results[i];
    EXPECT_EQ(actual_buffers.size(), 1);
    TF_ASSERT_OK(actual_buffers.front()->ToLiteralSync(&actual_literal));
    EXPECT_TRUE(LiteralTestUtil::Equal(actual_literal, expected_literals[i]));
  }
}

TEST(InterpreterClientTest, EvaluateWithHloOutputCallbackSucceeds) {
  InterpreterClient client;
  const Shape shape = ShapeUtil::MakeShape(S32, {4});
  XlaBuilder builder("test");

  auto parameter = Parameter(&builder, 0, shape, "parameter0");
  auto constant = ConstantR1(&builder, absl::Span<const int32_t>{1, 1, 1, 1});

  FrontendAttributes frontend_attributes;
  (*frontend_attributes.mutable_map())["_xla_tag"] = "42";
  builder.SetFrontendAttributes(frontend_attributes);

  Add(parameter, constant);
  builder.ClearFrontendAttributes();

  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> executable,
                          client.CompileAndLoad(computation, CompileOptions()));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> argument,
      client.BufferFromHostLiteral(
          LiteralUtil::CreateR1(absl::Span<const int32_t>{1, 2, 3, 4}),
          client.memory_spaces().front()));

  bool callback_called = false;
  HloOutputCallback callback;
  callback.callback_id = 42;
  callback.num_operands = 0;
  callback.callback =
      [&](int64_t replica_id, int64_t partition_id,
          absl::Span<std::shared_ptr<const Literal> const> literals) {
        callback_called = true;
        ASSERT_EQ(literals.size(), 1);
        ASSERT_NE(literals[0], nullptr);
        EXPECT_TRUE(LiteralTestUtil::Equal(
            *literals[0],
            LiteralUtil::CreateR1(absl::Span<const int32_t>{2, 3, 4, 5})));
      };

  std::vector<HloOutputCallback> callbacks;
  callbacks.push_back(std::move(callback));
  ExecuteOptions options;
  options.hlo_output_callbacks = callbacks;

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> results,
      executable->Execute({{argument.get()}}, options));

  EXPECT_TRUE(callback_called);
}

}  // namespace
}  // namespace xla
