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
#include <unordered_map>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/optimize/node_info_delegate.h"

namespace tflite {
namespace optimize {
namespace calibration {
namespace {

class TestDelegateObserver : public DelegateObserver {
 public:
  explicit TestDelegateObserver(TfLiteStatus status_to_return)
      : status_to_return_(status_to_return) {}

  TfLiteStatus OnDelegatePrepareCalled(TfLiteContext* context) override {
    num_times_called_++;
    return status_to_return_;
  }
  int num_times_called() { return num_times_called_; }

 private:
  int num_times_called_ = 0;
  TfLiteStatus status_to_return_;
};

TEST(NodeInfoDelegateTest, DelegateObserverIsCalled) {
  TestDelegateObserver observer(kTfLiteOk);
  NodeInfoDelegateParams params;
  params.delegate_observer = &observer;
  auto model = FlatBufferModel::BuildFromFile(
      "third_party/tensorflow/lite/testdata/multi_add.bin");
  ASSERT_TRUE(model);
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(InterpreterBuilder(*model,
                               ops::builtin::BuiltinOpResolver{})(&interpreter),
            kTfLiteOk);
  ASSERT_TRUE(interpreter);
  EXPECT_EQ(0, observer.num_times_called());
  TfLiteDelegate delegate = CreateNodeInfoDelegate(&params);

  auto status = interpreter->ModifyGraphWithDelegate(&delegate);
  EXPECT_EQ(kTfLiteOk, status);
  EXPECT_EQ(1, observer.num_times_called());
}

TEST(NodeInfoDelegateTest, ObserverErrorCausesModifyGraphFailure) {
  // Observer returns error
  TestDelegateObserver observer(kTfLiteError);
  NodeInfoDelegateParams params;
  params.delegate_observer = &observer;
  auto model = FlatBufferModel::BuildFromFile(
      "third_party/tensorflow/lite/testdata/multi_add.bin");
  ASSERT_TRUE(model);
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(InterpreterBuilder(*model,
                               ops::builtin::BuiltinOpResolver{})(&interpreter),
            kTfLiteOk);
  ASSERT_TRUE(interpreter);
  TfLiteDelegate delegate = CreateNodeInfoDelegate(&params);

  auto status = interpreter->ModifyGraphWithDelegate(&delegate);
  EXPECT_EQ(kTfLiteError, status);
}

TEST(NodeInfoDelegateTest, NodeInfoDelegateObserver) {
  auto model = FlatBufferModel::BuildFromFile(
      "third_party/tensorflow/lite/testdata/multi_add.bin");
  ASSERT_TRUE(model);

  std::unordered_map<int, OperatorInfo> index_to_opinfo;
  auto primary_subgraph = model->GetModel()->subgraphs()->Get(0);
  auto operators = primary_subgraph->operators();
  auto subgraph_tensors = primary_subgraph->tensors();
  for (size_t i = 0; i < operators->size(); i++) {
    OperatorInfo info;
    auto op_inputs = operators->Get(i)->inputs();
    auto op_outputs = operators->Get(i)->outputs();
    info.inputs = std::vector<int>(op_inputs->begin(), op_inputs->end());
    info.outputs = std::vector<int>(op_outputs->begin(), op_outputs->end());
    index_to_opinfo[i] = info;
  }

  std::unordered_map<const TfLiteNode*, OperatorInfo> node_to_opinfo;
  NodeInfoDelegateObserver observer(index_to_opinfo, &node_to_opinfo);
  NodeInfoDelegateParams params;
  params.delegate_observer = &observer;
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(InterpreterBuilder(*model,
                               ops::builtin::BuiltinOpResolver{})(&interpreter),
            kTfLiteOk);
  ASSERT_TRUE(interpreter);

  TfLiteDelegate delegate = CreateNodeInfoDelegate(&params);

  auto status = interpreter->ModifyGraphWithDelegate(&delegate);
  EXPECT_EQ(kTfLiteOk, status);
  EXPECT_EQ(index_to_opinfo.size(), node_to_opinfo.size());
  EXPECT_EQ(interpreter->nodes_size(), node_to_opinfo.size());

  for (const auto& node_and_opinfo : node_to_opinfo) {
    const TfLiteNode* tflite_node = node_and_opinfo.first;
    const OperatorInfo& info = node_and_opinfo.second;
    ASSERT_EQ(tflite_node->inputs->size, info.inputs.size());
    ASSERT_EQ(tflite_node->outputs->size, info.outputs.size());

    for (size_t input_index = 0; input_index < info.inputs.size();
         input_index++) {
      const TfLiteTensor* tflite_tensor =
          interpreter->tensor(tflite_node->inputs->data[input_index]);
      EXPECT_EQ(tflite_tensor->name,
                subgraph_tensors->Get(info.inputs[input_index])->name()->str());
    }

    for (size_t output_index = 0; output_index < info.outputs.size();
         output_index++) {
      const TfLiteTensor* tflite_tensor =
          interpreter->tensor(tflite_node->outputs->data[output_index]);
      EXPECT_EQ(
          tflite_tensor->name,
          subgraph_tensors->Get(info.outputs[output_index])->name()->str());
    }
  }
}

}  // namespace
}  // namespace calibration
}  // namespace optimize
}  // namespace tflite

int main(int argc, char** argv) {
  // On Linux, add: FLAGS_logtostderr = true;
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
