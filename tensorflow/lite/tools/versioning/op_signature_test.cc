/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/versioning/op_signature.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/lite/model_builder.h"

namespace tflite {

TEST(GetOpSignature, FlatBufferModel) {
  const std::string& full_path =
      tensorflow::GetDataDependencyFilepath("tensorflow/lite/testdata/add.bin");
  auto fb_model = FlatBufferModel::BuildFromFile(full_path.data());
  ASSERT_TRUE(fb_model);
  auto model = fb_model->GetModel();
  auto subgraphs = model->subgraphs();
  const SubGraph* subgraph = subgraphs->Get(0);
  const Operator* op1 = subgraph->operators()->Get(0);
  const OperatorCode* op_code1 =
      model->operator_codes()->Get(op1->opcode_index());
  OpSignature op_sig = GetOpSignature(op_code1, op1, subgraph, model);
  EXPECT_EQ(op_sig.op, BuiltinOperator_ADD);
  EXPECT_EQ(op_sig.inputs[0].type, kTfLiteFloat32);
  EXPECT_EQ(op_sig.inputs[0].dims.size(), 4);
  EXPECT_FALSE(op_sig.inputs[0].is_const);
  EXPECT_EQ(op_sig.outputs[0].type, kTfLiteFloat32);
  EXPECT_FALSE(op_sig.outputs[0].is_const);
  EXPECT_EQ(op_sig.outputs[0].dims.size(), 4);
  EXPECT_NE(op_sig.builtin_data, nullptr);
  free(op_sig.builtin_data);

  const Operator* op2 = subgraph->operators()->Get(1);
  const OperatorCode* op_code2 =
      model->operator_codes()->Get(op2->opcode_index());
  op_sig = GetOpSignature(op_code2, op2, subgraph, model);
  EXPECT_EQ(op_sig.op, BuiltinOperator_ADD);
  EXPECT_EQ(op_sig.inputs[0].type, kTfLiteFloat32);
  EXPECT_EQ(op_sig.inputs[0].dims.size(), 4);
  EXPECT_FALSE(op_sig.inputs[0].is_const);
  EXPECT_EQ(op_sig.outputs[0].type, kTfLiteFloat32);
  EXPECT_FALSE(op_sig.outputs[0].is_const);
  EXPECT_EQ(op_sig.outputs[0].dims.size(), 4);
  EXPECT_NE(op_sig.builtin_data, nullptr);
  free(op_sig.builtin_data);
}

}  // namespace tflite
