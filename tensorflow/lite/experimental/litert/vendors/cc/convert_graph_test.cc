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

#include "tensorflow/lite/experimental/litert/vendors/cc/convert_graph.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_graph.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/experimental/litert/test/test_macros.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/backend_ir.h"
#include "tensorflow/lite/experimental/litert/vendors/examples/example_conversion_impl.h"
#include "tensorflow/lite/experimental/litert/vendors/examples/example_ir.h"

namespace litert {
namespace {

using ::litert::example::ExampleOpAllocator;
using ::litert::example::ExampleOpType;
using ::litert::example::ExampleTensorAllocator;
using ::litert::example::ExampleTypes;
using ::litert::example::MakeAllLegalizations;
using ::litert::example::MakeTensorConverter;
using ::testing::AllOf;
using ::testing::ElementsAreArray;
using ::testing::Expectation;
using ::testing::ExpectationSet;
using ::testing::Field;
using ::testing::Return;

static constexpr std::array kDims = {2, 2};
static constexpr auto kElementType = kLiteRtElementTypeFloat32;
static constexpr absl::string_view kGraphName = "graph_name";

TensorType GetTestTensorType() {
  return MakeRankedTensorType(kElementType, absl::MakeConstSpan(kDims));
}

class MockGraphBuilder
    : public BackendGraphBuilder<ExampleTypes::Op, ExampleTypes::Tensor> {
 public:
  MOCK_METHOD(void, InitGraph, (std::string name), (override));
  MOCK_METHOD(LiteRtStatus, RegisterTensor, (ExampleTypes::Tensor & tensor),
              (override));
  MOCK_METHOD(LiteRtStatus, RegisterOp, (ExampleTypes::Op & op), (override));
  MOCK_METHOD(LiteRtStatus, FinalizeGraph, (), (override));
};

TEST(ConvertGraphTest, ConvertSingleSimpleConversion) {
  LiteRtSubgraphT subgraph;

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflMul);

  auto& input1 = subgraph.EmplaceTensor();
  input1.SetType(GetTestTensorType());
  input1.SetName("input1");

  auto& input2 = subgraph.EmplaceTensor();
  input2.SetType(GetTestTensorType());
  input2.SetName("input2");

  auto& output = subgraph.EmplaceTensor();
  output.SetType(GetTestTensorType());
  output.SetName("output");

  internal::AttachInput(&input1, op);
  internal::AttachInput(&input2, op);
  internal::AttachOutput(&output, op);

  subgraph.Inputs().push_back(&input1);
  subgraph.Inputs().push_back(&input2);
  subgraph.Outputs().push_back(&output);

  Subgraph litert_subgraph(&subgraph);

  ExampleOpAllocator op_alloc;
  ExampleTensorAllocator tensor_alloc;

  MockGraphBuilder builder;

  Expectation init_graph =
      EXPECT_CALL(builder, InitGraph(std::string(kGraphName))).Times(1);

  ExpectationSet reg_inputs;
  reg_inputs +=
      EXPECT_CALL(builder, RegisterTensor(Field(&ExampleTypes::Tensor::name,
                                                input1.Name())))
          .Times(1)
          .After(init_graph)
          .WillOnce(Return(kLiteRtStatusOk));
  reg_inputs +=
      EXPECT_CALL(builder, RegisterTensor(Field(&ExampleTypes::Tensor::name,
                                                input2.Name())))
          .Times(1)
          .After(init_graph)
          .WillOnce(Return(kLiteRtStatusOk));

  ExpectationSet reg_outputs;
  reg_outputs +=
      EXPECT_CALL(builder, RegisterTensor(Field(&ExampleTypes::Tensor::name,
                                                output.Name())))
          .Times(1)
          .After(init_graph)
          .WillOnce(Return(kLiteRtStatusOk));

  auto match_reg_op_args =
      AllOf(Field(&ExampleTypes::Op::op_code, ExampleOpType::MUL),
            Field(&ExampleTypes::Op::input_names,
                  ElementsAreArray({input1.Name(), input2.Name()})),
            Field(&ExampleTypes::Op::output_names,
                  ElementsAreArray({output.Name()})));

  Expectation reg_op = EXPECT_CALL(builder, RegisterOp(match_reg_op_args))
                           .Times(1)
                           .After(reg_inputs, reg_outputs)
                           .WillOnce(Return(kLiteRtStatusOk));

  Expectation finalize_graph = EXPECT_CALL(builder, FinalizeGraph())
                                   .Times(1)
                                   .After(reg_op)
                                   .WillOnce(Return(kLiteRtStatusOk));

  auto stat = ConvertGraph<ExampleTypes>(
      litert_subgraph, std::string(kGraphName), MakeTensorConverter,
      tensor_alloc, op_alloc, MakeAllLegalizations(), builder);

  LITERT_ASSERT_STATUS_OK(stat);
}

TEST(ConvertGraphTest, ConvertSingleGeneralConversion) {
  LiteRtSubgraphT subgraph;

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  tflite::AddOptionsT add_opts;
  add_opts.fused_activation_function = tflite::ActivationFunctionType_RELU;
  internal::TflOptions tfl_opts;
  tfl_opts.Set(std::move(add_opts));
  detail::SetTflOptions(op, std::move(tfl_opts));

  auto& input1 = subgraph.EmplaceTensor();
  input1.SetType(GetTestTensorType());
  input1.SetName("input1");

  auto& input2 = subgraph.EmplaceTensor();
  input2.SetType(GetTestTensorType());
  input2.SetName("input2");

  auto& output = subgraph.EmplaceTensor();
  output.SetType(GetTestTensorType());
  output.SetName("output");

  internal::AttachInput(&input1, op);
  internal::AttachInput(&input2, op);
  internal::AttachOutput(&output, op);

  subgraph.Inputs().push_back(&input1);
  subgraph.Inputs().push_back(&input2);
  subgraph.Outputs().push_back(&output);

  Subgraph litert_subgraph(&subgraph);

  ExampleOpAllocator op_alloc;
  ExampleTensorAllocator tensor_alloc;

  MockGraphBuilder builder;

  Expectation init_graph =
      EXPECT_CALL(builder, InitGraph(std::string(kGraphName))).Times(1);

  ExpectationSet reg_inputs;
  reg_inputs +=
      EXPECT_CALL(builder, RegisterTensor(Field(&ExampleTypes::Tensor::name,
                                                input1.Name())))
          .Times(1)
          .After(init_graph)
          .WillOnce(Return(kLiteRtStatusOk));
  reg_inputs +=
      EXPECT_CALL(builder, RegisterTensor(Field(&ExampleTypes::Tensor::name,
                                                input2.Name())))
          .Times(1)
          .After(init_graph)
          .WillOnce(Return(kLiteRtStatusOk));

  ExpectationSet reg_intermediates;
  reg_intermediates +=
      EXPECT_CALL(builder,
                  RegisterTensor(Field(&ExampleTypes::Tensor::name,
                                       example::kIntermediateTensorName)))
          .Times(1)
          .After(init_graph)
          .WillOnce(Return(kLiteRtStatusOk));

  ExpectationSet reg_outputs;
  reg_outputs +=
      EXPECT_CALL(builder, RegisterTensor(Field(&ExampleTypes::Tensor::name,
                                                output.Name())))
          .Times(1)
          .After(init_graph)
          .WillOnce(Return(kLiteRtStatusOk));

  auto match_reg_add_args =
      AllOf(Field(&ExampleTypes::Op::op_code, ExampleOpType::ADD),
            Field(&ExampleTypes::Op::input_names,
                  ElementsAreArray({input1.Name(), input2.Name()})),
            Field(&ExampleTypes::Op::output_names,
                  ElementsAreArray({example::kIntermediateTensorName})));

  Expectation reg_add = EXPECT_CALL(builder, RegisterOp(match_reg_add_args))
                            .Times(1)
                            .After(reg_inputs, reg_intermediates)
                            .WillOnce(Return(kLiteRtStatusOk));

  auto match_reg_relu_args =
      AllOf(Field(&ExampleTypes::Op::op_code, ExampleOpType::RELU),
            Field(&ExampleTypes::Op::input_names,
                  ElementsAreArray({example::kIntermediateTensorName})),
            Field(&ExampleTypes::Op::output_names,
                  ElementsAreArray({output.Name()})));

  Expectation reg_relu = EXPECT_CALL(builder, RegisterOp(match_reg_relu_args))
                             .Times(1)
                             .After(reg_add, reg_intermediates, reg_outputs)
                             .WillOnce(Return(kLiteRtStatusOk));

  Expectation finalize_graph = EXPECT_CALL(builder, FinalizeGraph())
                                   .Times(1)
                                   .After(reg_relu)
                                   .WillOnce(Return(kLiteRtStatusOk));

  auto stat = ConvertGraph<ExampleTypes>(
      litert_subgraph, std::string(kGraphName), MakeTensorConverter,
      tensor_alloc, op_alloc, MakeAllLegalizations(), builder);

  LITERT_ASSERT_STATUS_OK(stat);
}

TEST(ConvertGraphTest, ConvertMultipleOps) {
  LiteRtSubgraphT subgraph;

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflMul);

  auto& input1 = subgraph.EmplaceTensor();
  input1.SetType(GetTestTensorType());
  input1.SetName("input1");

  auto& input2 = subgraph.EmplaceTensor();
  input2.SetType(GetTestTensorType());
  input2.SetName("input2");

  auto& output1 = subgraph.EmplaceTensor();
  output1.SetType(GetTestTensorType());
  output1.SetName("output1");

  auto& cst = subgraph.EmplaceTensor();
  OwningBufferRef<uint8_t> weights(8);
  cst.Weights().SetFromBuf(weights);
  cst.SetName("cst");
  cst.SetType(GetTestTensorType());

  auto& op2 = subgraph.EmplaceOp();
  op2.SetOpCode(kLiteRtOpCodeTflAdd);

  auto& output2 = subgraph.EmplaceTensor();
  output2.SetType(GetTestTensorType());
  output2.SetName("output2");

  internal::AttachInput(&input1, op);
  internal::AttachInput(&input2, op);
  internal::AttachOutput(&output1, op);

  internal::AttachInput(&output1, op2);
  internal::AttachInput(&cst, op2);
  internal::AttachOutput(&output2, op2);

  subgraph.Inputs().push_back(&input1);
  subgraph.Inputs().push_back(&input2);
  subgraph.Outputs().push_back(&output2);

  Subgraph litert_subgraph(&subgraph);

  ExampleOpAllocator op_alloc;
  ExampleTensorAllocator tensor_alloc;

  MockGraphBuilder builder;

  Expectation init_graph =
      EXPECT_CALL(builder, InitGraph(std::string(kGraphName))).Times(1);

  ExpectationSet reg_inputs;
  reg_inputs +=
      EXPECT_CALL(builder, RegisterTensor(Field(&ExampleTypes::Tensor::name,
                                                input1.Name())))
          .Times(1)
          .After(init_graph)
          .WillOnce(Return(kLiteRtStatusOk));
  reg_inputs +=
      EXPECT_CALL(builder, RegisterTensor(Field(&ExampleTypes::Tensor::name,
                                                input2.Name())))
          .Times(1)
          .After(init_graph)
          .WillOnce(Return(kLiteRtStatusOk));

  Expectation reg_output1 =
      EXPECT_CALL(builder, RegisterTensor(Field(&ExampleTypes::Tensor::name,
                                                output1.Name())))
          .Times(1)
          .After(init_graph)
          .WillOnce(Return(kLiteRtStatusOk));

  Expectation reg_cst =
      EXPECT_CALL(builder, RegisterTensor(
                               Field(&ExampleTypes::Tensor::name, cst.Name())))
          .Times(1)
          .After(init_graph)
          .WillOnce(Return(kLiteRtStatusOk));

  Expectation reg_output2 =
      EXPECT_CALL(builder, RegisterTensor(Field(&ExampleTypes::Tensor::name,
                                                output2.Name())))
          .Times(1)
          .After(init_graph)
          .WillOnce(Return(kLiteRtStatusOk));

  auto match_reg_op1_args =
      AllOf(Field(&ExampleTypes::Op::op_code, ExampleOpType::MUL),
            Field(&ExampleTypes::Op::input_names,
                  ElementsAreArray({input1.Name(), input2.Name()})),
            Field(&ExampleTypes::Op::output_names,
                  ElementsAreArray({output1.Name()})));

  Expectation reg_op1 = EXPECT_CALL(builder, RegisterOp(match_reg_op1_args))
                            .Times(1)
                            .After(reg_inputs, reg_output1)
                            .WillOnce(Return(kLiteRtStatusOk));

  auto match_reg_op2_args =
      AllOf(Field(&ExampleTypes::Op::op_code, ExampleOpType::ADD),
            Field(&ExampleTypes::Op::input_names,
                  ElementsAreArray({output1.Name(), cst.Name()})),
            Field(&ExampleTypes::Op::output_names,
                  ElementsAreArray({output2.Name()})));

  Expectation reg_op2 = EXPECT_CALL(builder, RegisterOp(match_reg_op2_args))
                            .Times(1)
                            .After(reg_op1, reg_cst, reg_output2, reg_output1)
                            .WillOnce(Return(kLiteRtStatusOk));

  Expectation finalize_graph = EXPECT_CALL(builder, FinalizeGraph())
                                   .Times(1)
                                   .After(reg_op2)
                                   .WillOnce(Return(kLiteRtStatusOk));

  auto stat = ConvertGraph<ExampleTypes>(
      litert_subgraph, std::string(kGraphName), MakeTensorConverter,
      tensor_alloc, op_alloc, MakeAllLegalizations(), builder);

  LITERT_ASSERT_STATUS_OK(stat);
}

}  // namespace
}  // namespace litert
