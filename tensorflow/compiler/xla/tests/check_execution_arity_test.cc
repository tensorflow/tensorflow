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

#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::ContainsRegex;

class CheckExecutionArityTest : public ClientLibraryTestBase {};

TEST_F(CheckExecutionArityTest, TwoParamComputationNumArguments) {
  XlaBuilder builder("add_two_params");
  auto param_literal = LiteralUtil::CreateR1<float>({1.1f, 2.2f});

  auto p0 = Parameter(&builder, 0, param_literal.shape(), "param0");
  auto p1 = Parameter(&builder, 1, param_literal.shape(), "param1");
  Add(p0, p1);

  auto param0_data =
      client_->TransferToServer(param_literal).ConsumeValueOrDie();
  auto param1_data =
      client_->TransferToServer(param_literal).ConsumeValueOrDie();

  auto computation_status = builder.Build();
  ASSERT_IS_OK(computation_status.status());
  auto computation = computation_status.ConsumeValueOrDie();

  // The arity of the UserComputation is 2 arguments. Execution will succeed
  // with 2 arguments, but fail with a different number.
  auto result_two_args = client_->Execute(
      computation, {param0_data.get(), param1_data.get()}, &execution_options_);
  ASSERT_IS_OK(result_two_args.status());

  auto result_one_arg =
      client_->Execute(computation, {param0_data.get()}, &execution_options_);
  ASSERT_FALSE(result_one_arg.ok());
  ASSERT_EQ(result_one_arg.status().code(),
            tensorflow::error::INVALID_ARGUMENT);
  ASSERT_THAT(result_one_arg.status().error_message(),
              ContainsRegex("takes 2"));

  auto result_zero_args =
      client_->Execute(computation, {}, &execution_options_);
  ASSERT_FALSE(result_zero_args.ok());
  ASSERT_EQ(result_zero_args.status().code(),
            tensorflow::error::INVALID_ARGUMENT);
  ASSERT_THAT(result_zero_args.status().error_message(),
              ContainsRegex("takes 2"));
}

XLA_TEST_F(CheckExecutionArityTest, CheckArgumentShapes) {
  XlaBuilder builder("add_two_params");

  auto p0 = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "param0");
  auto p1 = Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {4}), "param1");
  Mul(p0, p1);

  auto computation_status = builder.Build();
  ASSERT_IS_OK(computation_status.status());
  auto computation = computation_status.ConsumeValueOrDie();

  auto f32_literal = LiteralUtil::CreateR0<float>(1.1f);
  auto f32_data = client_->TransferToServer(f32_literal).ConsumeValueOrDie();
  auto f32_4_literal = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  auto f32_4_data =
      client_->TransferToServer(f32_4_literal).ConsumeValueOrDie();
  auto u8_4_literal = LiteralUtil::CreateR1U8("hola");
  auto u8_4_data = client_->TransferToServer(u8_4_literal).ConsumeValueOrDie();

  // Match
  auto status = client_->Execute(
      computation, {f32_data.get(), f32_4_data.get()}, &execution_options_);
  ASSERT_IS_OK(status.status());

  // Shape mismatch in parameter 0
  status = client_->Execute(computation, {f32_4_data.get(), f32_4_data.get()},
                            &execution_options_);
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.status().code(), tensorflow::error::INVALID_ARGUMENT);
  ASSERT_THAT(status.status().error_message(),
              ContainsRegex(
                  "Argument does not match shape of computation parameter 0"));

  // Shape mismatch in parameter 1 (rank)
  status = client_->Execute(computation, {f32_data.get(), f32_data.get()},
                            &execution_options_);
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.status().code(), tensorflow::error::INVALID_ARGUMENT);
  ASSERT_THAT(status.status().error_message(),
              ContainsRegex(
                  "Argument does not match shape of computation parameter 1"));

  // Shape mismatch in parameter 1 (element type)
  status = client_->Execute(computation, {f32_data.get(), u8_4_data.get()},
                            &execution_options_);
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.status().code(), tensorflow::error::INVALID_ARGUMENT);
  ASSERT_THAT(status.status().error_message(),
              ContainsRegex(
                  "Argument does not match shape of computation parameter 1"));
}

}  // namespace
}  // namespace xla
