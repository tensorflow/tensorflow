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

#include "tensorflow/lite/experimental/micro/micro_interpreter.h"

#include "tensorflow/lite/experimental/micro/test_helpers.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"

namespace tflite {
namespace {
void* MockInit(TfLiteContext* context, const char* buffer, size_t length) {
  // We don't support delegate in TFL micro. This is a weak check to test if
  // context struct being zero-initialized.
  TF_LITE_MICRO_EXPECT_EQ(nullptr,
                          context->ReplaceNodeSubsetsWithDelegateKernels);
  // Do nothing.
  return nullptr;
}

void MockFree(TfLiteContext* context, void* buffer) {
  // Do nothing.
}

TfLiteStatus MockPrepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus MockInvoke(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
  const int32_t* input_data = input->data.i32;
  const TfLiteTensor* weight = &context->tensors[node->inputs->data[1]];
  const uint8_t* weight_data = weight->data.uint8;
  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
  int32_t* output_data = output->data.i32;
  output_data[0] = input_data[0] + weight_data[0];
  return kTfLiteOk;
}

class MockOpResolver : public OpResolver {
 public:
  const TfLiteRegistration* FindOp(BuiltinOperator op,
                                   int version) const override {
    return nullptr;
  }
  const TfLiteRegistration* FindOp(const char* op, int version) const override {
    if (strcmp(op, "mock_custom") == 0) {
      static TfLiteRegistration r = {MockInit, MockFree, MockPrepare,
                                     MockInvoke};
      return &r;
    } else {
      return nullptr;
    }
  }
};

}  // namespace
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestInterpreter) {
  const tflite::Model* model = tflite::testing::GetMockModel();
  TF_LITE_MICRO_EXPECT_NE(nullptr, model);
  tflite::MockOpResolver mock_resolver;
  constexpr size_t allocator_buffer_size = 1024;
  uint8_t allocator_buffer[allocator_buffer_size];
  tflite::MicroInterpreter interpreter(model, mock_resolver, allocator_buffer,
                                       allocator_buffer_size,
                                       micro_test::reporter);
  interpreter.AllocateTensors();
  TF_LITE_MICRO_EXPECT_EQ(1, interpreter.inputs_size());
  TF_LITE_MICRO_EXPECT_EQ(1, interpreter.outputs_size());

  TfLiteTensor* input = interpreter.input(0);
  TF_LITE_MICRO_EXPECT_NE(nullptr, input);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, input->type);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(4, input->bytes);
  TF_LITE_MICRO_EXPECT_NE(nullptr, input->data.i32);
  input->data.i32[0] = 21;

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, interpreter.Invoke());

  TfLiteTensor* output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_NE(nullptr, output);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, output->type);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(4, output->bytes);
  TF_LITE_MICRO_EXPECT_NE(nullptr, output->data.i32);
  TF_LITE_MICRO_EXPECT_EQ(42, output->data.i32[0]);
}

TF_LITE_MICRO_TEST(TestInterpreterProvideInputBuffer) {
  const tflite::Model* model = tflite::testing::GetMockModel();
  TF_LITE_MICRO_EXPECT_NE(nullptr, model);
  tflite::MockOpResolver mock_resolver;
  int32_t input_buffer = 21;
  constexpr size_t allocator_buffer_size = 1024;
  uint8_t allocator_buffer[allocator_buffer_size];
  tflite::MicroInterpreter interpreter(model, mock_resolver, allocator_buffer,
                                       allocator_buffer_size,
                                       micro_test::reporter);
  interpreter.RegisterPreallocatedInput(
      reinterpret_cast<uint8_t*>(&input_buffer), 0);
  interpreter.AllocateTensors();
  TF_LITE_MICRO_EXPECT_EQ(1, interpreter.inputs_size());
  TF_LITE_MICRO_EXPECT_EQ(1, interpreter.outputs_size());

  TfLiteTensor* input = interpreter.input(0);
  TF_LITE_MICRO_EXPECT_NE(nullptr, input);
  TF_LITE_MICRO_EXPECT_EQ(reinterpret_cast<uint8_t*>(&input_buffer),
                          reinterpret_cast<uint8_t*>(input->data.raw));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, input->type);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(4, input->bytes);
  TF_LITE_MICRO_EXPECT_NE(nullptr, input->data.i32);
  TF_LITE_MICRO_EXPECT_EQ(21, *input->data.i32);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, interpreter.Invoke());

  TfLiteTensor* output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_NE(nullptr, output);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, output->type);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(4, output->bytes);
  TF_LITE_MICRO_EXPECT_NE(nullptr, output->data.i32);
  TF_LITE_MICRO_EXPECT_EQ(42, output->data.i32[0]);
}

TF_LITE_MICRO_TESTS_END
