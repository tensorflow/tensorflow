/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/tests/test_utils.h"

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/local_client_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

// A test fixture is used because we need a client for our computation builder.
class TestUtilsTest : public LocalClientTestBase {};

XLA_TEST_F(TestUtilsTest, UnusedParam) {
  XlaBuilder builder(TestName());
  // Make the reduction lambda.
  Shape single_float = ShapeUtil::MakeShape(F32, {});
  Parameter(&builder, 0, single_float, "unused");
  Parameter(&builder, 1, single_float, "used");
  auto computation_status = builder.Build();
  TF_ASSERT_OK(computation_status.status());

  // Make the reduction.
  Shape pair_float = ShapeUtil::MakeShape(F32, {2});
  Reduce(Parameter(&builder, 0, pair_float, "operand"),
         Parameter(&builder, 1, single_float, "init"),
         computation_status.ValueOrDie(), {0});
  computation_status = builder.Build();
  TF_ASSERT_OK(computation_status.status());

  auto executable_status = local_client_->Compile(
      computation_status.ValueOrDie(), {&pair_float, &single_float},
      ExecutableBuildOptions());
  TF_ASSERT_OK(executable_status.status());
  HloModule& module = const_cast<HloModule&>(
      executable_status.ValueOrDie()->executable()->module());
  TF_ASSERT_OK(MakeFakeArguments(&module).status());
}

XLA_TEST_F(TestUtilsTest, Token) {
  auto module = ParseHloString(
                    R"(HloModule outfeed_module

    ENTRY InfeedToOutfeed {
      token = token[] parameter(0)
      infeed = ((u32[3]{0}, pred[]), token[]) infeed(token)
      infeed.data = (u32[3]{0}, pred[]) get-tuple-element(infeed), index=0
      outfeed = token[] outfeed(infeed.data, token)
      ROOT infeed.1 = ((u32[3]{0}, pred[]), token[]) infeed(token)
      infeed.1.data = (u32[3]{0}, pred[]) get-tuple-element(infeed.1), index=0
      infeed.1.token = token[] get-tuple-element(infeed.1), index=1
      outfeed.1 = token[] outfeed(infeed.1.data, infeed.1.token)
    })")
                    .ValueOrDie();
  TF_ASSERT_OK(MakeFakeArguments(module.get()).status());
}

}  // namespace
}  // namespace xla
