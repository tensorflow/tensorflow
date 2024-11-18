// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/vendors/cc/litert_convert.h"

#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/litert_convert_test_util.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/litert_convert_types.h"
#include "tensorflow/lite/experimental/litert/vendors/examples/example_convert_types_impl.h"

namespace litert {
namespace {

using ::litert::example::ExampleCapability;
using ::litert::example::ExampleGraphContext;
using ::litert::example::ExampleLegalizer;
using ::litert::example::ExampleOp;
using ::litert::example::ExampleOpLegalization;
using ::litert::example::Type;
using ::litert::testing::ConversionTestContext;
using ::litert::testing::CreateExampleScopedFinalizingLegalization;
using ::litert::testing::ExampleScopedFinalizingLegalization;
using ::litert::testing::TestOpContext;
using ::testing::UnorderedElementsAreArray;

TEST(PartitionViaCapabilityTest, ExamplePartition) {
  auto model = testing::LoadTestFileModel("simple_multi_op.tflite");
  std::vector<LiteRtOp> selected_ops;
  ExampleLegalizer legalizer;
  legalizer.Register(ExampleOpLegalization<kLiteRtOpCodeTflMul>::Create());

  CompilerCapability<ExampleOp> capability = ExampleCapability;

  LITERT_ASSERT_STATUS_OK(PartitionViaCapabilities(
      legalizer, capability, model,
      [&selected_ops](auto litert_op) { selected_ops.push_back(litert_op); }));

  ASSERT_EQ(selected_ops.size(), 2);
  EXPECT_EQ(selected_ops[0]->op_code, kLiteRtOpCodeTflMul);
  EXPECT_EQ(selected_ops[1]->op_code, kLiteRtOpCodeTflMul);
}

TEST(FinalizingLegalizationTest, OpToMatch) {
  ExampleGraphContext graph_context;
  auto finalize_legalization =
      CreateExampleScopedFinalizingLegalization(graph_context, nullptr);

  EXPECT_EQ(finalize_legalization->OpToMatch(), kLiteRtOpCodeTflCustom);
}

TEST(FinalizingLegalizationTest, Legalize) {
  ConversionTestContext<kLiteRtOpCodeTflCustom> context;
  context.graph_context.infos.emplace_back();
  TestOpContext op(kLiteRtOpCodeTflCustom, {"input_1", "input_2"},
                   {"output_1"});
  for (auto& input : op.GetOp().Inputs()) {
    LITERT_ASSERT_STATUS_OK(context.scope_context.Push(input));
  }

  auto result = context.legalization.Legalize(op.GetOp());
  ASSERT_TRUE(result);

  auto simple_result = GetSimpleConversionResult(*result);
  ASSERT_TRUE(simple_result);

  ASSERT_EQ(context.graph_context.infos.size(), 1);
  ASSERT_EQ(context.graph_context.Cur().tensor_names.size(), 3);
  EXPECT_THAT(context.graph_context.Cur().tensor_names,
              UnorderedElementsAreArray({"input_1", "input_2", "output_1"}));

  ASSERT_EQ(context.graph_context.Cur().backend_ops.size(), 1);
  EXPECT_EQ(context.graph_context.Cur().backend_ops.front().code,
            static_cast<int>(simple_result->code));

  EXPECT_TRUE(context.scope_context.GetScope()->contains(
      op.GetOp().Outputs().front().Get()));
}

TEST(FinalizingLegalizationTest, InputNotInScopeFail) {
  ConversionTestContext<kLiteRtOpCodeTflCustom> context;
  context.graph_context.infos.emplace_back();
  TestOpContext op(kLiteRtOpCodeTflCustom, {"input_1", "input_2"},
                   {"output_1"});
  auto result = context.legalization.Legalize(op.GetOp());
  ASSERT_FALSE(result);
}

TEST(FinalizingLegalizationTest, NoMatch) {
  ConversionTestContext<kLiteRtOpCodeTflCustom> context;
  context.graph_context.infos.emplace_back();
  TestOpContext op(kLiteRtOpCodeTflMul, {"input_1", "input_2"}, {"output_1"});
  for (auto& input : op.GetOp().Inputs()) {
    LITERT_ASSERT_STATUS_OK(context.scope_context.Push(input));
  }
  auto result = context.legalization.Legalize(op.GetOp());
  ASSERT_TRUE(result);
  ASSERT_FALSE(LegalizationMatched(*result));
}

}  // namespace
}  // namespace litert
