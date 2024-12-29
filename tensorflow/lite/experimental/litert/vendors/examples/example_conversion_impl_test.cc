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

#include "tensorflow/lite/experimental/litert/vendors/examples/example_conversion_impl.h"

#include <array>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_graph.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/experimental/litert/test/test_macros.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/conversion.h"
#include "tensorflow/lite/experimental/litert/vendors/examples/example_ir.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace litert::example {
namespace {

using ::testing::ElementsAreArray;
using ::testing::HasSubstr;

TEST(ExampleConversionImplTest, ConvertTensor) {
  static constexpr std::array kDims = {2, 2};
  static constexpr absl::string_view kName = "foo";

  LiteRtTensorT litert_tensor;
  litert_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32,
                                             absl::MakeConstSpan(kDims)));
  litert_tensor.SetName(std::string(kName));

  ExampleTensorAllocator tensor_alloc;
  auto tensor_convert = MakeTensorConverter(tensor_alloc);

  auto& example_tensor = **tensor_convert(Tensor(&litert_tensor));
  EXPECT_EQ(example_tensor.type, ExampleTensorType::FLOAT);
  EXPECT_THAT(example_tensor.dims, ElementsAreArray(kDims));
  EXPECT_EQ(example_tensor.name, kName);
}

TEST(ExampleConversionImplTest, ExampleGraphBuilder) {
  ExampleTensor input;
  input.type = ExampleTensorType::FLOAT;
  input.dims = {2, 2};
  input.id = 1;

  ExampleTensor output;
  output.type = ExampleTensorType::INT;
  output.dims = {3, 3};
  output.id = 2;

  ExampleOp op;
  op.op_code = ExampleOpType::ADD;
  op.inputs = {1};
  op.outputs = {2};

  ExampleGraphBuilder builder;
  static constexpr absl::string_view kName = "FOO_GRAPH";

  builder.InitGraph(std::string(kName));
  LITERT_ASSERT_STATUS_OK(builder.RegisterTensor(input));
  LITERT_ASSERT_STATUS_OK(builder.RegisterOp(op));
  LITERT_ASSERT_STATUS_OK(builder.RegisterTensor(output));
  LITERT_ASSERT_STATUS_OK(builder.FinalizeGraph());

  const auto serialized = builder.Serialize();
  EXPECT_THAT(serialized, HasSubstr("1FLOAT[2, 2]"));
  EXPECT_THAT(serialized, HasSubstr("2INT[3, 3]"));
  EXPECT_THAT(serialized, HasSubstr("ADD(1)->(2)"));
  EXPECT_THAT(serialized, HasSubstr("FINALIZED"));
  EXPECT_THAT(serialized, HasSubstr(kName));
}

TEST(ExampleConversionImplTest, LegalizeAddSimpleResult) {
  static constexpr std::array kDims = {2, 2};

  LiteRtTensorT input1;
  input1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32,
                                      absl::MakeConstSpan(kDims)));
  input1.SetName("input1");

  LiteRtTensorT input2;
  input2.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32,
                                      absl::MakeConstSpan(kDims)));
  input2.SetName("input2");

  LiteRtTensorT output;
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32,
                                      absl::MakeConstSpan(kDims)));
  output.SetName("output");

  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  internal::AttachInput(&input1, op);
  internal::AttachInput(&input2, op);
  internal::AttachOutput(&output, op);

  tflite::AddOptionsT add_opts;
  add_opts.fused_activation_function = tflite::ActivationFunctionType_NONE;
  internal::TflOptions tfl_opts;
  tfl_opts.Set(std::move(add_opts));
  detail::SetTflOptions(op, std::move(tfl_opts));

  ExampleTensorAllocator tensor_alloc;
  ExampleOpAllocator op_alloc;

  ExampleLegalizeAdd legalize_add;
  EXPECT_EQ(legalize_add.OpToMatch(), kLiteRtOpCodeTflAdd);

  auto legalized =
      legalize_add.Legalize(Op(&op), MakeTensorConverter, MakeTensorConverter,
                            tensor_alloc, op_alloc);

  ASSERT_TRUE(legalized);

  auto simple_result = GetSimpleConversionResult(*legalized);
  ASSERT_TRUE(simple_result);
  auto& example_op = **simple_result;

  EXPECT_EQ(example_op.op_code, ExampleOpType::ADD);
  EXPECT_THAT(example_op.inputs, ElementsAreArray({0, 1}));
  EXPECT_THAT(example_op.input_names,
              ElementsAreArray({input1.Name(), input2.Name()}));
  EXPECT_THAT(example_op.outputs, ElementsAreArray({2}));
  EXPECT_THAT(example_op.output_names, ElementsAreArray({output.Name()}));
}

TEST(ExampleConversionImplTest, LegalizeAddGeneralResult) {
  static constexpr std::array kDims = {2, 2};
  LiteRtTensorT input1;
  input1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32,
                                      absl::MakeConstSpan(kDims)));
  input1.SetName("input1");

  LiteRtTensorT input2;
  input2.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32,
                                      absl::MakeConstSpan(kDims)));
  input2.SetName("input2");

  LiteRtTensorT output;
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32,
                                      absl::MakeConstSpan(kDims)));
  output.SetName("output");

  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  internal::AttachInput(&input1, op);
  internal::AttachInput(&input2, op);
  internal::AttachOutput(&output, op);

  tflite::AddOptionsT add_opts;
  add_opts.fused_activation_function = tflite::ActivationFunctionType_RELU;
  internal::TflOptions tfl_opts;
  tfl_opts.Set(std::move(add_opts));
  detail::SetTflOptions(op, std::move(tfl_opts));

  ExampleTensorAllocator tensor_alloc;
  ExampleOpAllocator op_alloc;

  auto legalize_add = ExampleLegalizeAdd::Make();
  EXPECT_EQ(legalize_add->OpToMatch(), kLiteRtOpCodeTflAdd);

  auto legalized =
      legalize_add->Legalize(Op(&op), MakeTensorConverter, MakeTensorConverter,
                             tensor_alloc, op_alloc);
  ASSERT_TRUE(legalized);

  auto gen_result = GetGeneralConversionResult(*legalized);
  ASSERT_TRUE(gen_result);

  ASSERT_EQ(gen_result->ops.size(), 2);
  EXPECT_EQ(gen_result->ops[0]->op_code, ExampleOpType::ADD);
  EXPECT_THAT(gen_result->ops[0]->inputs, ElementsAreArray({0, 1}));
  EXPECT_THAT(gen_result->ops[0]->input_names,
              ElementsAreArray({input1.Name(), input2.Name()}));
  EXPECT_THAT(gen_result->ops[0]->outputs, ElementsAreArray({3}));
  EXPECT_THAT(gen_result->ops[0]->output_names,
              ElementsAreArray({kIntermediateTensorName}));
  EXPECT_EQ(gen_result->ops[1]->op_code, ExampleOpType::RELU);
  EXPECT_THAT(gen_result->ops[1]->inputs, ElementsAreArray({3}));
  EXPECT_THAT(gen_result->ops[1]->input_names,
              ElementsAreArray({kIntermediateTensorName}));
  EXPECT_THAT(gen_result->ops[1]->outputs, ElementsAreArray({2}));
  EXPECT_THAT(gen_result->ops[1]->output_names,
              ElementsAreArray({output.Name()}));
  EXPECT_EQ(gen_result->intermediate_tensors.size(), 1);
  EXPECT_EQ(gen_result->intermediate_tensors.front()->id, 3);
  EXPECT_EQ(gen_result->intermediate_tensors.front()->name,
            kIntermediateTensorName);
}

}  // namespace

}  // namespace litert::example
