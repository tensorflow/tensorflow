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

#include "tensorflow/lite/profiling/profile_summarizer.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/profiling/buffered_profiler.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace profiling {

namespace {

const char* kOpName = "SimpleOpEval";

TfLiteStatus SimpleOpEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, /*index=*/0, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, /*index=*/1, &input2));

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, /*index=*/0, &output));

  int32_t* output_data = output->data.i32;
  *output_data = *(input1->data.i32) + *(input2->data.i32);
  return kTfLiteOk;
}

const char* SimpleOpProfilingString(const TfLiteContext* context,
                                    const TfLiteNode* node) {
  return "Profile";
}

TfLiteRegistration* RegisterSimpleOp() {
  static TfLiteRegistration registration = {
      nullptr,        nullptr, nullptr,
      SimpleOpEval,   nullptr, tflite::BuiltinOperator_CUSTOM,
      "SimpleOpEval", 1};
  return &registration;
}

TfLiteRegistration* RegisterSimpleOpWithProfilingDetails() {
  static TfLiteRegistration registration = {nullptr,
                                            nullptr,
                                            nullptr,
                                            SimpleOpEval,
                                            SimpleOpProfilingString,
                                            tflite::BuiltinOperator_CUSTOM,
                                            kOpName,
                                            1};
  return &registration;
}

class SimpleOpModel : public SingleOpModel {
 public:
  void Init(const std::function<TfLiteRegistration*()>& registration);
  tflite::Interpreter* GetInterpreter() { return interpreter_.get(); }
  void SetInputs(int32_t x, int32_t y) {
    PopulateTensor(inputs_[0], {x});
    PopulateTensor(inputs_[1], {y});
  }
  int32_t GetOutput() { return ExtractVector<int32_t>(output_)[0]; }

 private:
  int inputs_[2];
  int output_;
};

void SimpleOpModel::Init(
    const std::function<TfLiteRegistration*()>& registration) {
  inputs_[0] = AddInput({TensorType_INT32, {1}});
  inputs_[1] = AddInput({TensorType_INT32, {1}});
  output_ = AddOutput({TensorType_INT32, {}});
  SetCustomOp(kOpName, {}, registration);
  BuildInterpreter({GetShape(inputs_[0]), GetShape(inputs_[1])});
}

TEST(ProfileSummarizerTest, Empty) {
  ProfileSummarizer summarizer;
  std::string output = summarizer.GetOutputString();
  EXPECT_GT(output.size(), 0);
}

TEST(ProfileSummarizerTest, Interpreter) {
  BufferedProfiler profiler(1024);
  SimpleOpModel m;
  m.Init(RegisterSimpleOp);
  auto interpreter = m.GetInterpreter();
  interpreter->SetProfiler(&profiler);
  profiler.StartProfiling();
  m.SetInputs(1, 2);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  // 3 = 1 + 2
  EXPECT_EQ(m.GetOutput(), 3);
  profiler.StopProfiling();
  ProfileSummarizer summarizer;
  auto events = profiler.GetProfileEvents();
  EXPECT_EQ(2, events.size());
  summarizer.ProcessProfiles(profiler.GetProfileEvents(), *interpreter);
  auto output = summarizer.GetOutputString();
  // TODO(shashishekhar): Add a better test here.
  ASSERT_TRUE(output.find("SimpleOpEval") != std::string::npos) << output;
  ASSERT_TRUE(output.find("Invoke") == std::string::npos) << output;  // NOLINT
}

TEST(ProfileSummarizerTest, InterpreterPlusProfilingDetails) {
  BufferedProfiler profiler(1024);
  SimpleOpModel m;
  m.Init(RegisterSimpleOpWithProfilingDetails);
  auto interpreter = m.GetInterpreter();
  interpreter->SetProfiler(&profiler);
  profiler.StartProfiling();
  m.SetInputs(1, 2);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  // 3 = 1 + 2
  EXPECT_EQ(m.GetOutput(), 3);
  profiler.StopProfiling();
  ProfileSummarizer summarizer;
  auto events = profiler.GetProfileEvents();
  EXPECT_EQ(2, events.size());
  summarizer.ProcessProfiles(profiler.GetProfileEvents(), *interpreter);
  auto output = summarizer.GetOutputString();
  // TODO(shashishekhar): Add a better test here.
  ASSERT_TRUE(output.find("SimpleOpEval/Profile") != std::string::npos)
      << output;
}

// A simple test that performs `ADD` if condition is true, and `MUL` otherwise.
// The computation is: `cond ? a + b : a * b`.
class ProfileSummarizerIfOpTest : public subgraph_test_util::ControlFlowOpTest {
 protected:
  void SetUp() override {
    AddSubgraphs(2);
    builder_->BuildAddSubgraph(interpreter_->subgraph(1));
    builder_->BuildMulSubgraph(interpreter_->subgraph(2));
    builder_->BuildIfSubgraph(&interpreter_->primary_subgraph());

    interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {2});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1, 2});
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

    subgraph_test_util::FillIntTensor(
        interpreter_->tensor(interpreter_->inputs()[1]), {5, 7});
    subgraph_test_util::FillIntTensor(
        interpreter_->tensor(interpreter_->inputs()[2]), {1, 2});
  }
};

TEST_F(ProfileSummarizerIfOpTest, TestIfTrue) {
  BufferedProfiler profiler(1024);
  interpreter_->SetProfiler(&profiler);

  interpreter_->typed_input_tensor<bool>(0)[0] = true;
  profiler.StartProfiling();
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  profiler.StopProfiling();
  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  subgraph_test_util::CheckIntTensor(output, {1, 2}, {6, 9});

  auto events = profiler.GetProfileEvents();
  EXPECT_EQ(5, events.size());
  int event_count_of_subgraph_zero = std::count_if(
      events.begin(), events.end(),
      [](auto event) { return event->extra_event_metadata == 0; });
  int event_count_of_subgraph_one = std::count_if(
      events.begin(), events.end(),
      [](auto event) { return event->extra_event_metadata == 1; });
  int event_count_of_subgraph_two = std::count_if(
      events.begin(), events.end(),
      [](auto event) { return event->extra_event_metadata == 2; });
  EXPECT_EQ(2, event_count_of_subgraph_zero);
  EXPECT_EQ(3, event_count_of_subgraph_one);
  EXPECT_EQ(0, event_count_of_subgraph_two);
}

TEST_F(ProfileSummarizerIfOpTest, TestIfFalse) {
  BufferedProfiler profiler(1024);
  interpreter_->SetProfiler(&profiler);

  interpreter_->typed_input_tensor<bool>(0)[0] = false;
  profiler.StartProfiling();
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  profiler.StopProfiling();
  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  subgraph_test_util::CheckIntTensor(output, {1, 2}, {5, 14});

  auto events = profiler.GetProfileEvents();
  EXPECT_EQ(5, events.size());
  int event_count_of_subgraph_zero = std::count_if(
      events.begin(), events.end(),
      [](auto event) { return event->extra_event_metadata == 0; });
  int event_count_of_subgraph_one = std::count_if(
      events.begin(), events.end(),
      [](auto event) { return event->extra_event_metadata == 1; });
  int event_count_of_subgraph_two = std::count_if(
      events.begin(), events.end(),
      [](auto event) { return event->extra_event_metadata == 2; });
  EXPECT_EQ(2, event_count_of_subgraph_zero);
  EXPECT_EQ(0, event_count_of_subgraph_one);
  EXPECT_EQ(3, event_count_of_subgraph_two);
}

}  // namespace
}  // namespace profiling
}  // namespace tflite
