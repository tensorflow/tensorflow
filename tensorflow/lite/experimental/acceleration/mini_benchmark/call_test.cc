/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <cstddef>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/call_register.h"
#include "tensorflow/lite/interpreter_test_util.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {

namespace {

class CallTest : public subgraph_test_util::ControlFlowOpTest {
 public:
  CallTest() { interpreter_ = std::make_unique<Interpreter>(&error_reporter_); }
  ~CallTest() override = default;
  void SetupTensor(Subgraph* subgraph, int tensor_index, TfLiteType type) {
    ASSERT_EQ(subgraph->SetTensorParametersReadWrite(tensor_index, type, "", 0,
                                                     nullptr, {}, false),
              kTfLiteOk);
  }
  void BuildCallSubgraph(Subgraph* subgraph, std::vector<uint8_t> params_buffer,
                         std::vector<int> inputs, std::vector<int> outputs,
                         int expected_node_index, bool single_node_subgraph) {
    if (single_node_subgraph) {
      int first_new_tensor_index;
      ASSERT_EQ(subgraph->AddTensors(inputs.size() + outputs.size(),
                                     &first_new_tensor_index),
                kTfLiteOk);
      ASSERT_EQ(first_new_tensor_index, 0);
      ASSERT_EQ(subgraph->SetInputs(inputs), kTfLiteOk);
      ASSERT_EQ(subgraph->SetOutputs(outputs), kTfLiteOk);
    }
    for (const int& idx : inputs) {
      SetupTensor(subgraph, idx, kTfLiteInt32);
    }
    for (const int& idx : outputs) {
      SetupTensor(subgraph, idx, kTfLiteInt32);
    }
    int node_index;
    subgraph->AddNodeWithParameters(
        inputs, outputs, {},
        reinterpret_cast<const char*>(params_buffer.data()),
        params_buffer.size(), nullptr, acceleration::ops::Register_CALL(),
        &node_index);
    ASSERT_EQ(node_index, expected_node_index);
  }
  void BuildCallSubgraph(Subgraph* subgraph, int index, int loop_count,
                         std::vector<int> inputs, std::vector<int> outputs,
                         int expected_node_index = 0,
                         bool single_node_subgraph = true) {
    flexbuffers::Builder fbb;
    fbb.Map([&] {
      fbb.Int("subgraph_index", index);
      fbb.Int("loop_count", loop_count);
    });
    fbb.Finish();
    BuildCallSubgraph(subgraph, fbb.GetBuffer(), inputs, outputs,
                      expected_node_index, single_node_subgraph);
  }

  void BuildGraphWithMultipleOutputs(Subgraph* subgraph) {
    const int kInput1 = 0;
    const int kInput2 = 1;
    const int kMulOutput = 2;
    const int kAddOutput = 3;
    const int kTensorCount = 4;
    // kInput1(0) --> +---+
    //                |MUL| --> kOutput(2)
    // kInput2(1) --> +---+
    //
    // kInput1(0) --> +---+
    //                |ADD| --> kOutput(3)
    // kInput2(1) --> +---+

    int first_new_tensor_index;
    ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
              kTfLiteOk);
    ASSERT_EQ(first_new_tensor_index, 0);
    ASSERT_EQ(subgraph->SetInputs({kInput1, kInput2}), kTfLiteOk);
    ASSERT_EQ(subgraph->SetOutputs({kMulOutput, kAddOutput}), kTfLiteOk);

    SetupTensor(subgraph, kInput1, kTfLiteInt32);
    SetupTensor(subgraph, kInput2, kTfLiteInt32);
    SetupTensor(subgraph, kMulOutput, kTfLiteInt32);
    SetupTensor(subgraph, kAddOutput, kTfLiteInt32);
    TfLiteMulParams* params_mul =
        reinterpret_cast<TfLiteMulParams*>(malloc(sizeof(TfLiteMulParams)));
    params_mul->activation = kTfLiteActNone;
    int node_index;
    subgraph->AddNodeWithParameters(
        {kInput1, kInput2}, {kMulOutput}, {}, nullptr, 0, params_mul,
        ::tflite::ops::builtin::Register_MUL(), &node_index);
    TfLiteAddParams* params_add =
        reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
    params_add->activation = kTfLiteActNone;
    subgraph->AddNodeWithParameters(
        {kInput1, kInput2}, {kAddOutput}, {}, nullptr, 0, params_add,
        ::tflite::ops::builtin::Register_ADD(), &node_index);
  }
  void BuildMultiNodeGraph(Subgraph* this_subgraph) {
    // kIn1(0)----------------
    //                       |
    //                       |        +----+
    //            +---+      -------->|    |              +---+
    // kIn2(1)--> |PAD|-->kOut1(4)--->|CALL|-->kOut2(5)-->|MUL|-->kOut3(6)
    // kIn3(2)--> |   |               |    |              |   |
    //            +---+               +----+         ---->|   |
    //                                               |    +---+
    //                                               |
    //                                               |
    // kIn4(3)----------------------------------------
    const int kInput1 = 0, kInput2 = 1, kInput3 = 2, kInput4 = 3;
    const int kOutput1 = 4, kOutput2 = 5, kOutput3 = 6;
    const int kTensorCount = 7;
    int first_new_tensor_index;
    ASSERT_EQ(this_subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
              kTfLiteOk);
    ASSERT_EQ(first_new_tensor_index, 0);
    std::vector<int> inputs = {kInput1, kInput2, kInput3, kInput4};
    std::vector<int> outputs = {kOutput3};
    ASSERT_EQ(this_subgraph->SetInputs(inputs), kTfLiteOk);
    ASSERT_EQ(this_subgraph->SetOutputs({kOutput3}), kTfLiteOk);
    for (int idx = 0; idx < kTensorCount; ++idx) {
      SetupTensor(this_subgraph, idx, kTfLiteInt32);
    }
    int expected_node_index = 0, node_index;
    // Node 1: Pad op.
    auto* pad_reg = ops::builtin::Register_PAD();
    pad_reg->builtin_code = kTfLiteBuiltinPad;
    this_subgraph->AddNodeWithParameters(
        {kInput2, kInput3}, {kOutput1}, {}, nullptr, 0,
        reinterpret_cast<TfLitePadParams*>(malloc(sizeof(TfLitePadParams))),
        pad_reg, &node_index);
    ASSERT_EQ(node_index, expected_node_index++);
    // Node 2: Call op, calls subgraph that contains Add op.
    AddSubgraphs(1);
    const int kLoopCount = 1;
    const int kSubgraphIndex = 1;
    builder_->BuildAddSubgraph(interpreter_->subgraph(1));
    CallTest::BuildCallSubgraph(this_subgraph, kSubgraphIndex, kLoopCount,
                                {kInput1, kOutput1}, {kOutput2},
                                expected_node_index++, false);
    // Node 3: Mul op.
    TfLiteMulParams* mul_params =
        reinterpret_cast<TfLiteMulParams*>(malloc(sizeof(TfLiteMulParams)));
    mul_params->activation = kTfLiteActNone;
    auto* mul_reg = ops::builtin::Register_MUL();
    mul_reg->builtin_code = kTfLiteBuiltinMul;
    this_subgraph->AddNodeWithParameters({kInput4, kOutput2}, {kOutput3}, {},
                                         nullptr, 0, mul_params, mul_reg,
                                         &node_index);
    ASSERT_EQ(node_index, expected_node_index++);
  }
  TestErrorReporter error_reporter_;
};

/** Tests the happy path for `call` op. **/
TEST_F(CallTest, SubgraphMultipleInputsSingleOutput) {
  std::vector<std::vector<int>> test_shapes = {
      {3, 2}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  // Will loop over and will be fed to the subgraph as {1,2}, {1,3}, {1,1,3},
  // {1,3,1,2}.
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    interpreter_ = std::make_unique<Interpreter>();
    AddSubgraphs(1);
    int loop_count = test_shapes[i][0];
    builder_->BuildMulSubgraph(interpreter_->subgraph(1));
    CallTest::BuildCallSubgraph(&interpreter_->primary_subgraph(), 1,
                                loop_count, {0, 1}, {2});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[0], test_shapes[i]);
    interpreter_->ResizeInputTensor(interpreter_->inputs()[1], test_shapes[i]);
    ASSERT_EQ(interpreter_->subgraph(1)->AllocateTensors(), kTfLiteOk);
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

    subgraph_test_util::FillIntTensor(
        interpreter_->tensor(interpreter_->inputs()[0]), {-1, 2, -3, 4, -5, 6});
    subgraph_test_util::FillIntTensor(
        interpreter_->tensor(interpreter_->inputs()[1]), {-1, 2, -3, 4, -5, 6});
    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

    TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
    subgraph_test_util::CheckIntTensor(output, test_shapes[i],
                                       {1, 4, 9, 16, 25, 36});
  }
}

TEST_F(CallTest, ShouldBeANoOpWhenLoopCountIsZero) {
  AddSubgraphs(1);
  int loop_count = 0;
  builder_->BuildMulSubgraph(interpreter_->subgraph(1));
  CallTest::BuildCallSubgraph(&interpreter_->primary_subgraph(), 1, loop_count,
                              {0, 1}, {2});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {0, 3});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {0, 3});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  subgraph_test_util::CheckIntTensor(output, {0, 3}, {});
}

TEST_F(CallTest, SubgraphWithFixedInputShapes) {
  AddSubgraphs(1);
  const int kLoopCount = 2;
  const int kBatchSizeSubgraph = 1;
  const int kFixedInputLen = 3;
  const std::vector<int> kCallOpInputShape = {kLoopCount, kFixedInputLen};
  const std::vector<int> kSubgraphInputShape = {kBatchSizeSubgraph,
                                                kFixedInputLen};

  Subgraph* subgraph = interpreter_->subgraph(1);
  builder_->BuildMulSubgraph(subgraph);
  CallTest::BuildCallSubgraph(&interpreter_->primary_subgraph(), 1, kLoopCount,
                              {0, 1}, {2});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], kCallOpInputShape);
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], kCallOpInputShape);

  subgraph->ResizeInputTensor(subgraph->inputs()[0], kSubgraphInputShape);
  subgraph->ResizeInputTensor(subgraph->inputs()[1], kSubgraphInputShape);

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  subgraph_test_util::FillIntTensor(
      interpreter_->tensor(interpreter_->inputs()[0]), {-1, 2, -3, 4, -5, 6});
  subgraph_test_util::FillIntTensor(
      interpreter_->tensor(interpreter_->inputs()[1]), {-1, 2, -3, 4, -5, 6});
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  subgraph_test_util::CheckIntTensor(output, kCallOpInputShape,
                                     {1, 4, 9, 16, 25, 36});
}

TEST_F(CallTest, SubgraphWithMultipleInputsAndOutputs) {
  std::vector<std::vector<int>> test_shapes = {
      {3, 2, 1}, {1, 2, 3}, {2, 1, 3}, {2, 3, 1, 1}, {2, 3}};
  for (size_t i = 0; i < test_shapes.size(); ++i) {
    interpreter_ = std::make_unique<Interpreter>();
    AddSubgraphs(1);
    int loop_count = test_shapes[i][0];
    CallTest::BuildGraphWithMultipleOutputs(interpreter_->subgraph(1));
    CallTest::BuildCallSubgraph(&interpreter_->primary_subgraph(), 1,
                                loop_count, {0, 1}, {2, 3});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[0], test_shapes[i]);
    interpreter_->ResizeInputTensor(interpreter_->inputs()[1], test_shapes[i]);
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

    subgraph_test_util::FillIntTensor(
        interpreter_->tensor(interpreter_->inputs()[0]), {-1, 2, -3, 4, -5, 6});
    subgraph_test_util::FillIntTensor(
        interpreter_->tensor(interpreter_->inputs()[1]), {-1, 2, -3, 4, -5, 6});
    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

    TfLiteTensor* output_mul = interpreter_->tensor(interpreter_->outputs()[0]);
    subgraph_test_util::CheckIntTensor(output_mul, test_shapes[i],
                                       {1, 4, 9, 16, 25, 36});
    TfLiteTensor* output_add = interpreter_->tensor(interpreter_->outputs()[1]);
    subgraph_test_util::CheckIntTensor(output_add, test_shapes[i],
                                       {-2, 4, -6, 8, -10, 12});
  }
}

TEST_F(CallTest, ShouldHandleInvalidParamsAndSetToDefault) {
  flexbuffers::Builder fbb;
  fbb.Vector([&]() {
    fbb.String("hi");
    fbb.String("hello");
  });
  fbb.Finish();
  AddSubgraphs(1);

  CallTest::BuildCallSubgraph(&interpreter_->primary_subgraph(),
                              fbb.GetBuffer(), {0}, {1}, 0, true);
  const int kNodeIndex = 0;
  const TfLiteNode* call_node = &interpreter_->primary_subgraph()
                                     .nodes_and_registration()[kNodeIndex]
                                     .first;
  tflite::acceleration::ops::TfLiteCallParams* op_data =
      reinterpret_cast<tflite::acceleration::ops::TfLiteCallParams*>(
          call_node->user_data);

  EXPECT_EQ(op_data->subgraph_index, 0);
  EXPECT_EQ(op_data->loop_count, 0);
}
TEST_F(CallTest, MultiNodeGraph) {
  CallTest::BuildMultiNodeGraph(&interpreter_->primary_subgraph());
  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1, 4, 4, 1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1, 2, 2, 1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {4, 2});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[3], {1, 4, 4, 1});

  ASSERT_EQ(interpreter_->subgraph(1)->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  subgraph_test_util::FillIntTensor(
      interpreter_->tensor(interpreter_->inputs()[0]), std::vector<int>(16, 1));
  subgraph_test_util::FillIntTensor(
      interpreter_->tensor(interpreter_->inputs()[1]), {1, 2, 3, 4});
  subgraph_test_util::FillIntTensor(
      interpreter_->tensor(interpreter_->inputs()[2]),
      {0, 0, 1, 1, 1, 1, 0, 0});
  subgraph_test_util::FillIntTensor(
      interpreter_->tensor(interpreter_->inputs()[3]), std::vector<int>(16, 2));

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  subgraph_test_util::CheckIntTensor(
      output, {1, 4, 4, 1}, {2, 2, 2, 2, 2, 4, 6, 2, 2, 8, 10, 2, 2, 2, 2, 2});
}

// Note: For the tests below the error messages returned by the error reporter
// are of the following format:
// "<filename>:<line number> <error message>. Node <number name> failed to
// prepare.\n"
// It's sufficient to test whether the string returned by error reporter
// contains the expected error message.
TEST_F(CallTest, ShouldFailWith0DInputs) {
  AddSubgraphs(1);
  int loop_count = 5;
  builder_->BuildMulSubgraph(interpreter_->subgraph(1));
  interpreter_->subgraph(1)->ResizeInputTensor(0, {});
  interpreter_->subgraph(1)->ResizeInputTensor(1, {});
  CallTest::BuildCallSubgraph(&interpreter_->primary_subgraph(), 1, loop_count,
                              {0, 1}, {2});

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteError);

  EXPECT_THAT(
      error_reporter_.error_messages(),
      testing::HasSubstr(
          "Dimensions of all of call node's inputs should be non-zero."));
}

TEST_F(CallTest, ShouldFailWhenLoopCountDoesNotMatchBatchSize) {
  AddSubgraphs(1);
  int loop_count = 7;
  builder_->BuildMulSubgraph(interpreter_->subgraph(1));
  CallTest::BuildCallSubgraph(&interpreter_->primary_subgraph(), 1, loop_count,
                              {0, 1}, {2});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {5, 3});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {5, 3});

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteError);
  EXPECT_THAT(
      error_reporter_.error_messages(),
      testing::HasSubstr("node_input->dims->data[0] != loop_count (5 != 7)"));
}

TEST_F(CallTest, ShouldFailForSubgraphWithIncompatibleInputShapes) {
  AddSubgraphs(1);
  const int kLoopCount = 5;
  const int kBatchSizeSubgraph = 1;
  std::vector<int> call_op_input = {kLoopCount, 3};
  std::vector<int> subgraph_input = {kBatchSizeSubgraph, 7};
  Subgraph* subgraph = interpreter_->subgraph(1);
  builder_->BuildMulSubgraph(subgraph);
  CallTest::BuildCallSubgraph(&interpreter_->primary_subgraph(), 1, kLoopCount,
                              {0, 1}, {2});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], call_op_input);
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], call_op_input);
  subgraph->ResizeInputTensor(subgraph->inputs()[0], subgraph_input);
  subgraph->ResizeInputTensor(subgraph->inputs()[1], subgraph_input);

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteError);

  EXPECT_THAT(
      error_reporter_.error_messages(),
      testing::HasSubstr("All dimensions except the batch size should match "
                         "for call node and the subgraph to invoke"));
}

TEST_F(CallTest, ShouldFailWhenSubgraphIndexMatchesInvokedSubgraph) {
  const int kPrimarySubgraphIndex = 0;
  CallTest::BuildCallSubgraph(&interpreter_->primary_subgraph(),
                              kPrimarySubgraphIndex, 1, {0}, {1});

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteError);

  EXPECT_THAT(
      error_reporter_.error_messages(),
      testing::HasSubstr(
          "Subgraph to invoke must be different from the invoking graph."));
}

TEST_F(CallTest, ShouldFailWithNegativeLoopCount) {
  AddSubgraphs(1);
  CallTest::BuildCallSubgraph(&interpreter_->primary_subgraph(), 1, -1, {0},
                              {1});

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteError);

  EXPECT_THAT(error_reporter_.error_messages(),
              testing::HasSubstr("Loop count must be positive."));
}

}  // namespace
}  // namespace tflite
