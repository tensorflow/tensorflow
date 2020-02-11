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

#include "tensorflow/lite/micro/micro_interpreter.h"

#include "tensorflow/lite/micro/micro_optional_debug_tools.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

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
  output_data[0] =
      0;  // Catch output tensor sharing memory with an input tensor
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
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  TF_LITE_MICRO_EXPECT_NE(nullptr, model);
  tflite::MockOpResolver mock_resolver;
  constexpr size_t allocator_buffer_size = 1024;
  uint8_t allocator_buffer[allocator_buffer_size];
  tflite::MicroInterpreter interpreter(model, mock_resolver, allocator_buffer,
                                       allocator_buffer_size,
                                       micro_test::reporter);
  TF_LITE_MICRO_EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  TF_LITE_MICRO_EXPECT_EQ(1, interpreter.inputs_size());
  TF_LITE_MICRO_EXPECT_EQ(2, interpreter.outputs_size());

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

  output = interpreter.output(1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, output);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, output->type);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(4, output->bytes);
  TF_LITE_MICRO_EXPECT_NE(nullptr, output->data.i32);
  TF_LITE_MICRO_EXPECT_EQ(42, output->data.i32[0]);

  // Just to make sure that this method works.
  tflite::PrintInterpreterState(&interpreter);
}

TF_LITE_MICRO_TEST(TestVariableTensorReset) {
  const tflite::Model* model = tflite::testing::GetComplexMockModel();
  TF_LITE_MICRO_EXPECT_NE(nullptr, model);

  tflite::MockOpResolver mock_resolver;
  constexpr size_t allocator_buffer_size = 2048;
  uint8_t allocator_buffer[allocator_buffer_size];
  tflite::MicroInterpreter interpreter(model, mock_resolver, allocator_buffer,
                                       allocator_buffer_size,
                                       micro_test::reporter);
  TF_LITE_MICRO_EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  TF_LITE_MICRO_EXPECT_EQ(1, interpreter.inputs_size());
  TF_LITE_MICRO_EXPECT_EQ(1, interpreter.outputs_size());

  // Assign hard-code values:
  for (size_t i = 0; i < interpreter.tensors_size(); ++i) {
    TfLiteTensor* cur_tensor = interpreter.tensor(i);
    int buffer_length = tflite::ElementCount(*cur_tensor->dims);
    // Assign all buffers to non-zero values. Variable tensors will be assigned
    // 2 here and will be verified that they have been reset after the API call.
    int buffer_value = cur_tensor->is_variable ? 2 : 1;
    switch (cur_tensor->type) {
      case kTfLiteInt32: {
        int32_t* buffer = tflite::GetTensorData<int32_t>(cur_tensor);
        for (int j = 0; j < buffer_length; ++j) {
          buffer[j] = static_cast<int32_t>(buffer_value);
        }
        break;
      }
      case kTfLiteUInt8: {
        uint8_t* buffer = tflite::GetTensorData<uint8_t>(cur_tensor);
        for (int j = 0; j < buffer_length; ++j) {
          buffer[j] = static_cast<uint8_t>(buffer_value);
        }
        break;
      }
      default:
        TF_LITE_MICRO_FAIL("Unsupported dtype");
    }
  }

  interpreter.ResetVariableTensors();

  // Ensure only variable tensors have been reset to zero:
  for (size_t i = 0; i < interpreter.tensors_size(); ++i) {
    TfLiteTensor* cur_tensor = interpreter.tensor(i);
    int buffer_length = tflite::ElementCount(*cur_tensor->dims);
    // Variable tensors should be zero (not the value assigned in the for loop
    // above).
    int buffer_value = cur_tensor->is_variable ? 0 : 1;
    switch (cur_tensor->type) {
      case kTfLiteInt32: {
        int32_t* buffer = tflite::GetTensorData<int32_t>(cur_tensor);
        for (int j = 0; j < buffer_length; ++j) {
          TF_LITE_MICRO_EXPECT_EQ(buffer_value, buffer[j]);
        }
        break;
      }
      case kTfLiteUInt8: {
        uint8_t* buffer = tflite::GetTensorData<uint8_t>(cur_tensor);
        for (int j = 0; j < buffer_length; ++j) {
          TF_LITE_MICRO_EXPECT_EQ(buffer_value, buffer[j]);
        }
        break;
      }
      default:
        TF_LITE_MICRO_FAIL("Unsupported dtype");
    }
  }
}

// The interpreter initialization requires multiple steps and this test case
// ensures that simply creating and destructing an interpreter object is ok.
// b/147830765 has one example of a change that caused trouble for this simple
// case.
TF_LITE_MICRO_TEST(TestIncompleteInitialization) {
  const tflite::Model* model = tflite::testing::GetComplexMockModel();
  TF_LITE_MICRO_EXPECT_NE(nullptr, model);

  tflite::MockOpResolver mock_resolver;
  constexpr size_t allocator_buffer_size = 2048;
  uint8_t allocator_buffer[allocator_buffer_size];
  tflite::MicroInterpreter interpreter(model, mock_resolver, allocator_buffer,
                                       allocator_buffer_size,
                                       micro_test::reporter);
}

TF_LITE_MICRO_TESTS_END
