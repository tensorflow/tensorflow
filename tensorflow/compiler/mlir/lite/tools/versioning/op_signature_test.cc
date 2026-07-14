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
#include "tensorflow/compiler/mlir/lite/tools/versioning/op_signature.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/compiler/mlir/lite/core/absl_error_model_builder.h"
#include "tensorflow/compiler/mlir/lite/core/c/tflite_types.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/core/platform/resource_loader.h"

namespace tflite {

TEST(GetOpSignature, FlatBufferModel) {
  const std::string& full_path = tensorflow::GetDataDependencyFilepath(
      "tensorflow/compiler/mlir/lite/testdata/add.bin");
  auto fb_model =
      mlir::TFL::FlatBufferModelAbslError::BuildFromFile(full_path.data());
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
  EXPECT_FALSE(op_sig.inputs[0].is_shape_dynamic);
  EXPECT_EQ(op_sig.outputs[0].type, kTfLiteFloat32);
  EXPECT_FALSE(op_sig.outputs[0].is_const);
  EXPECT_EQ(op_sig.outputs[0].dims.size(), 4);
  EXPECT_FALSE(op_sig.outputs[0].is_shape_dynamic);
  EXPECT_NE(op_sig.builtin_data, nullptr);
  EXPECT_EQ(op_sig.version, 1);
  free(op_sig.builtin_data);

  const Operator* op2 = subgraph->operators()->Get(1);
  const OperatorCode* op_code2 =
      model->operator_codes()->Get(op2->opcode_index());
  op_sig = GetOpSignature(op_code2, op2, subgraph, model);
  EXPECT_EQ(op_sig.op, BuiltinOperator_ADD);
  EXPECT_EQ(op_sig.inputs[0].type, kTfLiteFloat32);
  EXPECT_EQ(op_sig.inputs[0].dims.size(), 4);
  EXPECT_FALSE(op_sig.inputs[0].is_const);
  EXPECT_FALSE(op_sig.inputs[0].is_shape_dynamic);
  EXPECT_EQ(op_sig.outputs[0].type, kTfLiteFloat32);
  EXPECT_FALSE(op_sig.outputs[0].is_const);
  EXPECT_EQ(op_sig.outputs[0].dims.size(), 4);
  EXPECT_FALSE(op_sig.outputs[0].is_shape_dynamic);
  EXPECT_NE(op_sig.builtin_data, nullptr);
  EXPECT_EQ(op_sig.version, 1);
  free(op_sig.builtin_data);

  const std::string& full_path3 = tensorflow::GetDataDependencyFilepath(
      "tensorflow/compiler/mlir/lite/testdata/multi_signatures.bin");
  auto fb_model3 =
      mlir::TFL::FlatBufferModelAbslError::BuildFromFile(full_path3.data());
  ASSERT_TRUE(fb_model3);
  auto model3 = fb_model3->GetModel();
  auto subgraphs3 = model3->subgraphs();
  const SubGraph* subgraph3 = subgraphs3->Get(0);
  const Operator* op3 = subgraph3->operators()->Get(0);
  const OperatorCode* op_code3 =
      model3->operator_codes()->Get(op3->opcode_index());
  op_sig = GetOpSignature(op_code3, op3, subgraph3, model3);
  EXPECT_EQ(op_sig.op, BuiltinOperator_ADD);
  EXPECT_EQ(op_sig.inputs[0].type, kTfLiteFloat32);
  EXPECT_EQ(op_sig.inputs[0].dims.size(), 1);
  EXPECT_FALSE(op_sig.inputs[0].is_const);
  EXPECT_TRUE(op_sig.inputs[0].is_shape_dynamic);
  EXPECT_EQ(op_sig.outputs[0].type, kTfLiteFloat32);
  EXPECT_FALSE(op_sig.outputs[0].is_const);
  EXPECT_EQ(op_sig.outputs[0].dims.size(), 1);
  EXPECT_TRUE(op_sig.outputs[0].is_shape_dynamic);
  EXPECT_NE(op_sig.builtin_data, nullptr);
  EXPECT_EQ(op_sig.version, 1);
  free(op_sig.builtin_data);
}

}  // namespace tflite
