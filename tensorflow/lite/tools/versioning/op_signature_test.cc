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

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

// StubTfLiteContext is a TfLiteContext which has 3 nodes as the followings.
// dummyAdd -> target op -> dummyAdd
class StubTfLiteContext : public TfLiteContext {
 public:
  StubTfLiteContext(const int builtin_code, const int op_version,
                    const int num_inputs)
      : TfLiteContext({0}) {
    // Stub execution plan
    exec_plan_ = TfLiteIntArrayCreate(3);
    for (int i = 0; i < 3; ++i) exec_plan_->data[i] = i;

    int tensor_no = 0;
    std::memset(nodes_, 0, sizeof(nodes_));
    std::memset(registrations_, 0, sizeof(registrations_));

    // Node 0, dummyAdd
    nodes_[0].inputs = TfLiteIntArrayCreate(1);
    nodes_[0].inputs->data[0] = tensor_no++;
    nodes_[0].outputs = TfLiteIntArrayCreate(1);
    nodes_[0].outputs->data[0] = tensor_no;
    nodes_[0].builtin_data = nullptr;

    // Node 1, target op
    nodes_[1].inputs = TfLiteIntArrayCreate(num_inputs);
    for (int i = 0; i < num_inputs; i++) {
      nodes_[1].inputs->data[i] = tensor_no++;
    }
    nodes_[1].outputs = TfLiteIntArrayCreate(1);
    nodes_[1].outputs->data[0] = tensor_no;
    nodes_[1].builtin_data = malloc(1024);
    std::memset(nodes_[1].builtin_data, 0, 1024);

    // Node 2, dummyAdd
    nodes_[2].inputs = TfLiteIntArrayCreate(1);
    nodes_[2].inputs->data[0] = tensor_no++;
    nodes_[2].outputs = TfLiteIntArrayCreate(1);
    nodes_[2].outputs->data[0] = tensor_no++;
    nodes_[2].builtin_data = nullptr;

    // Creates tensors of 4d float32
    tensors_.resize(tensor_no);
    for (size_t i = 0; i < tensors_.size(); i++) {
      std::memset(&tensors_[i], 0, sizeof(tensors_[i]));
      tensors_[i].buffer_handle = kTfLiteNullBufferHandle;
      tensors_[i].type = kTfLiteFloat32;
      tensors_[i].dims = TfLiteIntArrayCreate(4);
      for (int d = 0; d < 4; d++) {
        tensors_[i].dims->data[d] = 1;
      }
    }
    tensors = tensors_.data();
    tensors_size = tensors_.size();

    // Creates registrations
    registrations_[0].builtin_code = kTfLiteBuiltinAdd;
    registrations_[1].builtin_code = builtin_code;
    registrations_[1].version = op_version;
    registrations_[2].builtin_code = kTfLiteBuiltinAdd;

    this->GetExecutionPlan = StubGetExecutionPlan;
    this->GetNodeAndRegistration = StubGetNodeAndRegistration;
  }
  ~StubTfLiteContext() {
    for (auto& node : nodes_) {
      TfLiteIntArrayFree(node.inputs);
      TfLiteIntArrayFree(node.outputs);
      if (node.builtin_data) {
        free(node.builtin_data);
      }
    }
    for (auto& tensor : tensors_) {
      TfLiteIntArrayFree(tensor.dims);
    }
    TfLiteIntArrayFree(exec_plan_);
  }

  TfLiteIntArray* exec_plan() const { return exec_plan_; }
  TfLiteNode* node() { return &nodes_[1]; }
  TfLiteRegistration* registration() { return &registrations_[1]; }
  TfLiteNode* node(int node_index) { return &nodes_[node_index]; }
  TfLiteRegistration* registration(int reg_index) {
    return &registrations_[reg_index];
  }
  TfLiteTensor* tensor(int tensor_index) { return &tensors_[tensor_index]; }

 private:
  static TfLiteStatus StubGetExecutionPlan(TfLiteContext* context,
                                           TfLiteIntArray** execution_plan) {
    StubTfLiteContext* stub = reinterpret_cast<StubTfLiteContext*>(context);
    *execution_plan = stub->exec_plan();
    return kTfLiteOk;
  }

  static TfLiteStatus StubGetNodeAndRegistration(
      TfLiteContext* context, int node_index, TfLiteNode** node,
      TfLiteRegistration** registration) {
    StubTfLiteContext* stub = reinterpret_cast<StubTfLiteContext*>(context);
    *node = stub->node(node_index);
    *registration = stub->registration(node_index);
    return kTfLiteOk;
  }

  TfLiteIntArray* exec_plan_;
  TfLiteNode nodes_[3];
  TfLiteRegistration registrations_[3];
  std::vector<TfLiteTensor> tensors_;
};

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
      "tensorflow/lite/testdata/multi_signatures.bin");
  auto fb_model3 = FlatBufferModel::BuildFromFile(full_path3.data());
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

TEST(GetOpSignature, TfLiteContext) {
  auto context = std::make_unique<StubTfLiteContext>(kTfLiteBuiltinAdd,
                                                     /*op_version=*/1,
                                                     /*num_inputs=*/4);
  OpSignature op_sig =
      GetOpSignature(context.get(), context->node(), context->registration());
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
}

}  // namespace tflite
