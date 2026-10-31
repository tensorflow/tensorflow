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
#include "tensorflow/lite/optional_debug_tools.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>

#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {
// This is specific to the testdata/add.bin model used in the test.
void InitInputTensorData(Interpreter* interpreter) {
  ASSERT_EQ(interpreter->inputs().size(), 1);
  TfLiteTensor* t = interpreter->input_tensor(0);
  ASSERT_EQ(t->type, kTfLiteFloat32);
  float* data = static_cast<float*>(t->data.data);
  int num_elements = t->bytes / sizeof(float);
  std::fill(data, data + num_elements, 1.0f);
}
}  // namespace

TEST(OptionalDebugTools, PrintInterpreterState) {
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/add.bin");
  ASSERT_TRUE(model);

  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          *model, ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &interpreter),
      kTfLiteOk);

  // Ensure printing the interpreter state doesn't crash:
  //   * Before allocation
  //   * After allocation
  //   * After invocation
  PrintInterpreterState(interpreter.get());

  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  PrintInterpreterState(interpreter.get());

  InitInputTensorData(interpreter.get());
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  PrintInterpreterState(interpreter.get());
}

TEST(OptionalDebugTools, PrintInterpreterStateWithDelegate) {
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/add.bin");
  ASSERT_TRUE(model);

  // Create and instantiate an interpreter with a delegate.
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          *model, ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &interpreter),
      kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(xnnpack_delegate.get()),
            kTfLiteOk);

  InitInputTensorData(interpreter.get());
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

  // Ensure printing the interpreter state doesn't crash.
  PrintInterpreterState(interpreter.get());
}

TEST(OptionalDebugTools, GetNodeDelegationMetadata) {
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/add.bin");
  ASSERT_TRUE(model);

  // Create and instantiate an interpreter with a delegate.
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          *model, ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &interpreter),
      kTfLiteOk);

  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  auto metadata = GetNodeDelegationMetadata(*interpreter->subgraph(0));
  EXPECT_FALSE(metadata.has_delegate_applied);
  for (int i = 0; i < metadata.is_node_delegated.size(); ++i) {
    EXPECT_FALSE(metadata.is_node_delegated[i]);
    EXPECT_EQ(metadata.replaced_by_node[i], -1);
  }

  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(xnnpack_delegate.get()),
            kTfLiteOk);
  auto metadata_with_delegate =
      GetNodeDelegationMetadata(*interpreter->subgraph(0));

  // The first two nodes are delegated to the third node.
  EXPECT_TRUE(metadata_with_delegate.has_delegate_applied);
  EXPECT_EQ(metadata_with_delegate.is_node_delegated[0], true);
  EXPECT_EQ(metadata_with_delegate.replaced_by_node[0], 2);
  EXPECT_EQ(metadata_with_delegate.is_node_delegated[1], true);
  EXPECT_EQ(metadata_with_delegate.replaced_by_node[1], 2);
  EXPECT_EQ(metadata_with_delegate.is_node_delegated[2], false);
  EXPECT_EQ(metadata_with_delegate.replaced_by_node[2], -1);
}

// Positive control and bounds test for out-of-range delegated node indices.
TEST(OptionalDebugTools, GetNodeDelegationMetadataOutOfRangeIndexIgnored) {
  auto model =
      FlatBufferModel::BuildFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_TRUE(model);

  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          *model, ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &interpreter),
      kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(xnnpack_delegate.get()),
            kTfLiteOk);

  Subgraph* subgraph = interpreter->subgraph(0);
  ASSERT_EQ(subgraph->nodes_size(), 3);
  const auto* delegate_entry = subgraph->node_and_registration(2);
  ASSERT_NE(delegate_entry, nullptr);
  auto* params =
      static_cast<TfLiteDelegateParams*>(delegate_entry->first.builtin_data);
  ASSERT_NE(params, nullptr);
  ASSERT_NE(params->nodes_to_replace, nullptr);
  ASSERT_GE(params->nodes_to_replace->size, 2);

  // Positive control before testing out-of-range node IDs.
  SubgraphDelegationMetadata baseline = GetNodeDelegationMetadata(*subgraph);
  EXPECT_TRUE(baseline.has_delegate_applied);
  EXPECT_TRUE(baseline.is_node_delegated[0]);
  EXPECT_TRUE(baseline.is_node_delegated[1]);

  const int orig_nid0 = params->nodes_to_replace->data[0];
  const int orig_nid1 = params->nodes_to_replace->data[1];
  params->nodes_to_replace->data[0] = 64;
  params->nodes_to_replace->data[1] = -1;

  SubgraphDelegationMetadata guarded = GetNodeDelegationMetadata(*subgraph);
  EXPECT_TRUE(guarded.has_delegate_applied);
  EXPECT_FALSE(guarded.is_node_delegated[0]);
  EXPECT_EQ(guarded.replaced_by_node[0], -1);
  EXPECT_FALSE(guarded.is_node_delegated[1]);
  EXPECT_EQ(guarded.replaced_by_node[1], -1);
  PrintInterpreterState(interpreter.get());

  params->nodes_to_replace->data[0] = orig_nid0;
  params->nodes_to_replace->data[1] = orig_nid1;
}

TEST(OptionalDebugTools, PrintInterpreterStateOutOfRangeBuiltinCodeSafe) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(2), kTfLiteOk);
  TfLiteQuantizationParams quant = {};
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(0, kTfLiteFloat32, "in",
                                                     {1}, quant),
            kTfLiteOk);
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(1, kTfLiteFloat32, "out",
                                                     {1}, quant),
            kTfLiteOk);

  TfLiteRegistration neg_reg = {};
  neg_reg.builtin_code = -1;
  neg_reg.custom_name = nullptr;
  int neg_index = -1;
  ASSERT_EQ(interpreter.AddNodeWithParameters({0}, {1}, nullptr, 0, nullptr,
                                              &neg_reg, &neg_index),
            kTfLiteOk);

  TfLiteRegistration high_reg = {};
  high_reg.builtin_code = static_cast<int32_t>(BuiltinOperator_MAX) + 100;
  high_reg.custom_name = nullptr;
  int high_index = -1;
  ASSERT_EQ(interpreter.AddNodeWithParameters({0}, {1}, nullptr, 0, nullptr,
                                              &high_reg, &high_index),
            kTfLiteOk);

  PrintInterpreterState(&interpreter);
}

}  // namespace tflite
