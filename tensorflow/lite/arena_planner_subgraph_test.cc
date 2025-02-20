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
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"

namespace tflite {
namespace {

TEST(ArenaPlannerSubgraphTest, TestSubgraphInputsAndOutputsNotShared) {
  Interpreter interpreter;
  subgraph_test_util::SubgraphBuilder builder;
  builder.BuildInplaceOpSubgraph(&interpreter.primary_subgraph());
  ASSERT_EQ(interpreter.ResizeInputTensor(interpreter.inputs()[0], {2}),
            kTfLiteOk);
  ASSERT_EQ(interpreter.ResizeInputTensor(interpreter.inputs()[1], {2}),
            kTfLiteOk);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  const TfLiteTensor* add_input0 = interpreter.tensor(interpreter.inputs()[0]);
  const TfLiteTensor* add_input1 = interpreter.tensor(interpreter.inputs()[1]);
  const int kIntermediateTensor0 = 2;
  const int kIntermediateTensor1 = 3;
  const TfLiteTensor* add_output = interpreter.tensor(kIntermediateTensor0);
  const TfLiteTensor* reshape_output = interpreter.tensor(kIntermediateTensor1);
  const TfLiteTensor* subgraph_output =
      interpreter.tensor(interpreter.outputs()[0]);
  // Neither input to ADD may be shared since they are subgraph inputs.
  EXPECT_NE(add_input0->data.data, add_output->data.data);
  EXPECT_NE(add_input1->data.data, add_output->data.data);
  // The output of ADD may be shared with the output of RESHAPE.
  EXPECT_EQ(add_output->data.data, reshape_output->data.data);
  // RESHAPE's output can't be shared as the output of the following op is a
  // subgraph output.
  EXPECT_NE(reshape_output->data.data, subgraph_output->data.data);
}

TEST(ArenaPlannerSubgraphTest, TestSharingBroadcastOps) {
  Interpreter interpreter;
  subgraph_test_util::SubgraphBuilder builder;
  builder.BuildBroadcastingSubgraph(&interpreter.primary_subgraph());
  ASSERT_EQ(interpreter.ResizeInputTensor(interpreter.inputs()[0], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter.ResizeInputTensor(interpreter.inputs()[1], {2}),
            kTfLiteOk);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  TfLiteTensor* subgraph_input0 = interpreter.tensor(interpreter.inputs()[0]);
  TfLiteTensor* subgraph_input1 = interpreter.tensor(interpreter.inputs()[1]);
  const int kIntermediateTensor0 = 2;
  const int kIntermediateTensor1 = 3;
  const int kIntermediateTensor2 = 4;
  const int kIntermediateTensor3 = 5;
  const TfLiteTensor* intermediate_tensor0 =
      interpreter.tensor(kIntermediateTensor0);
  const TfLiteTensor* intermediate_tensor1 =
      interpreter.tensor(kIntermediateTensor1);
  const TfLiteTensor* intermediate_tensor2 =
      interpreter.tensor(kIntermediateTensor2);
  const TfLiteTensor* intermediate_tensor3 =
      interpreter.tensor(kIntermediateTensor3);
  subgraph_test_util::FillIntTensor(subgraph_input0, {1});
  subgraph_test_util::FillIntTensor(subgraph_input1, {2, 2});
  const TfLiteTensor* subgraph_output =
      interpreter.tensor(interpreter.outputs()[0]);
  // Neither input to ADD may be shared since they are subgraph inputs.
  EXPECT_NE(subgraph_input0->data.data, intermediate_tensor0->data.data);
  EXPECT_NE(subgraph_input1->data.data, intermediate_tensor0->data.data);
  // The 2nd ADD op consumes intermediate_tensor0 twice. Since there is more
  // than one consumer, sharing is not possible.
  EXPECT_NE(intermediate_tensor0->data.data, intermediate_tensor1->data.data);
  // intermediate_tensor1 can be shared with intermediate_tensor2.
  EXPECT_EQ(intermediate_tensor1->data.data, intermediate_tensor2->data.data);
  // intermediate_tensor3 cannot be shared with a subgraph output.
  EXPECT_NE(intermediate_tensor3->data.data, subgraph_output->data.data);
  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);
  subgraph_test_util::CheckIntTensor(subgraph_output, {2}, {9, 9});
}

TEST(ArenaPlannerSubgraphTest, TestOffsetAddSharing) {
  Interpreter interpreter;
  subgraph_test_util::SubgraphBuilder builder;
  builder.BuildOffsetAddSharing(&interpreter.primary_subgraph());
  ASSERT_EQ(interpreter.ResizeInputTensor(interpreter.inputs()[0], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter.ResizeInputTensor(interpreter.inputs()[1], {3}),
            kTfLiteOk);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  TfLiteTensor* subgraph_input0 = interpreter.tensor(interpreter.inputs()[0]);
  TfLiteTensor* subgraph_input1 = interpreter.tensor(interpreter.inputs()[1]);
  const int kIntermediateTensor0 = 2;
  const int kIntermediateTensor1 = 3;
  const int kIntermediateTensor2 = 4;
  const TfLiteTensor* intermediate_tensor0 =
      interpreter.tensor(kIntermediateTensor0);
  const TfLiteTensor* intermediate_tensor1 =
      interpreter.tensor(kIntermediateTensor1);
  const TfLiteTensor* intermediate_tensor2 =
      interpreter.tensor(kIntermediateTensor2);
  subgraph_test_util::FillIntTensor(subgraph_input0, {1});
  subgraph_test_util::FillIntTensor(subgraph_input1, {3, 4, 5});
  const TfLiteTensor* subgraph_output =
      interpreter.tensor(interpreter.outputs()[0]);
  // Neither input to ADD may be shared since they are subgraph inputs.
  EXPECT_NE(subgraph_input0->data.data, intermediate_tensor0->data.data);
  EXPECT_NE(subgraph_input1->data.data, intermediate_tensor1->data.data);
  // ADD_OFFSET allows sharing of input1, but not input0.
  EXPECT_EQ(intermediate_tensor1->data.data, intermediate_tensor2->data.data);
  // intermediate_tensor2 cannot be shared with a subgraph output.
  EXPECT_NE(intermediate_tensor2->data.data, subgraph_output->data.data);
  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);
  subgraph_test_util::CheckIntTensor(subgraph_output, {3}, {10, 12, 11});
}

TEST(ArenaPlannerSubgraphTest, HWTensor) {
  Interpreter interpreter;
  subgraph_test_util::SubgraphBuilder builder;
  builder.BuildInplaceOpSubgraph(&interpreter.primary_subgraph());
  ASSERT_EQ(interpreter.ResizeInputTensor(interpreter.inputs()[0], {2}),
            kTfLiteOk);
  ASSERT_EQ(interpreter.ResizeInputTensor(interpreter.inputs()[1], {2}),
            kTfLiteOk);

  TfLiteTensor* add_input0 = interpreter.tensor(interpreter.inputs()[0]);
  const TfLiteTensor* add_input1 = interpreter.tensor(interpreter.inputs()[1]);
  const int kIntermediateTensor0 = 2;
  const int kIntermediateTensor1 = 3;
  const TfLiteTensor* add_output = interpreter.tensor(kIntermediateTensor0);
  const TfLiteTensor* reshape_output = interpreter.tensor(kIntermediateTensor1);
  TfLiteTensor* subgraph_output = interpreter.tensor(interpreter.outputs()[0]);

  add_input0->allocation_type = kTfLiteNonCpu;
  subgraph_output->allocation_type = kTfLiteNonCpu;
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  // Non-CPU tensors shouldn't be allocated.
  EXPECT_EQ(add_input0->data.data, nullptr);
  EXPECT_EQ(subgraph_output->data.data, nullptr);
  // CPU tensors should be allocated.
  EXPECT_NE(add_input1->data.data, nullptr);
  EXPECT_NE(add_output->data.data, nullptr);
  EXPECT_NE(reshape_output->data.data, nullptr);
}

}  // namespace
}  // namespace tflite
