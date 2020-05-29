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

#include <cstdint>

#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_optional_debug_tools.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace {

// A simple operator that returns the median of the input with the number of
// times the kernel was invoked. The implementation below is deliberately
// complicated, just to demonstrate how kernel memory planning works.
class SimpleStatefulOp {
  static constexpr int kBufferNotAllocated = 0;
  // Inputs:
  static constexpr int kInputTensor = 0;
  // Outputs:
  static constexpr int kMedianTensor = 0;
  static constexpr int kInvokeCount = 1;
  struct OpData {
    int invoke_count = 0;
    int sorting_buffer = kBufferNotAllocated;
  };

 public:
  static const TfLiteRegistration* getRegistration() {
    static TfLiteRegistration r = {Init, /* free= */ nullptr, Prepare, Invoke};
    return &r;
  }

  static void* Init(TfLiteContext* context, const char* buffer, size_t length) {
    TF_LITE_MICRO_EXPECT_EQ(nullptr, context->RequestScratchBufferInArena);
    TF_LITE_MICRO_EXPECT_EQ(nullptr, context->AllocateBufferForEval);
    TF_LITE_MICRO_EXPECT_EQ(nullptr, context->GetScratchBuffer);

    void* raw;
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, context->AllocatePersistentBuffer(
                                           context, sizeof(OpData), &raw));
    OpData* data = reinterpret_cast<OpData*>(raw);
    *data = {};
    return raw;
  }

  static TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
    OpData* data = reinterpret_cast<OpData*>(node->user_data);

    // Make sure that the input is in uint8 with at least 1 data entry.
    const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    if (input->type != kTfLiteUInt8) return kTfLiteError;
    if (NumElements(input->dims) == 0) return kTfLiteError;

    // Allocate a temporary buffer with the same size of input for sorting.
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, sizeof(uint8_t) * NumElements(input->dims),
        &data->sorting_buffer));
    return kTfLiteOk;
  }

  static TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node) {
    OpData* data = reinterpret_cast<OpData*>(node->user_data);
    data->invoke_count += 1;

    const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    const uint8_t* input_data = GetTensorData<uint8_t>(input);
    int size = NumElements(input->dims);

    uint8_t* sorting_buffer = reinterpret_cast<uint8_t*>(
        context->GetScratchBuffer(context, data->sorting_buffer));
    // Copy inputs data to the sorting buffer. We don't want to mutate the input
    // tensor as it might be used by a another node.
    for (int i = 0; i < size; i++) {
      sorting_buffer[i] = input_data[i];
    }

    // In place insertion sort on `sorting_buffer`.
    for (int i = 1; i < size; i++) {
      for (int j = i; j > 0 && sorting_buffer[j] < sorting_buffer[j - 1]; j--) {
        std::swap(sorting_buffer[j], sorting_buffer[j - 1]);
      }
    }

    TfLiteTensor* median = GetOutput(context, node, kMedianTensor);
    uint8_t* median_data = GetTensorData<uint8_t>(median);
    TfLiteTensor* invoke_count = GetOutput(context, node, kInvokeCount);
    int32_t* invoke_count_data = GetTensorData<int32_t>(invoke_count);

    median_data[0] = sorting_buffer[size / 2];
    invoke_count_data[0] = data->invoke_count;
    return kTfLiteOk;
  }
};

bool freed = false;

class MockCustom {
 public:
  static const TfLiteRegistration* getRegistration() {
    static TfLiteRegistration r = {Init, Free, Prepare, Invoke};
    return &r;
  }

  static void* Init(TfLiteContext* context, const char* buffer, size_t length) {
    // We don't support delegate in TFL micro. This is a weak check to test if
    // context struct being zero-initialized.
    TF_LITE_MICRO_EXPECT_EQ(nullptr,
                            context->ReplaceNodeSubsetsWithDelegateKernels);
    // Do nothing.
    return nullptr;
  }

  static void Free(TfLiteContext* context, void* buffer) { freed = true; }

  static TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
    return kTfLiteOk;
  }

  static TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node) {
    const TfLiteTensor* input = GetInput(context, node, 0);
    const int32_t* input_data = input->data.i32;
    const TfLiteTensor* weight = GetInput(context, node, 1);
    const uint8_t* weight_data = weight->data.uint8;
    TfLiteTensor* output = GetOutput(context, node, 0);
    int32_t* output_data = output->data.i32;
    output_data[0] =
        0;  // Catch output tensor sharing memory with an input tensor
    output_data[0] = input_data[0] + weight_data[0];
    return kTfLiteOk;
  }
};

class MockOpResolver : public MicroOpResolver {
 public:
  const TfLiteRegistration* FindOp(BuiltinOperator op,
                                   int version) const override {
    return nullptr;
  }
  const TfLiteRegistration* FindOp(const char* op, int version) const override {
    if (strcmp(op, "mock_custom") == 0) {
      return MockCustom::getRegistration();
    } else if (strcmp(op, "simple_stateful_op") == 0) {
      return SimpleStatefulOp::getRegistration();
    } else {
      return nullptr;
    }
  }

  MicroOpResolver::BuiltinParseFunction GetOpDataParser(
      tflite::BuiltinOperator) const override {
    // TODO(b/149408647): Figure out an alternative so that we do not have any
    // references to ParseOpData in the micro code and the signature for
    // MicroOpResolver::BuiltinParseFunction can be changed to be different from
    // ParseOpData.
    return ParseOpData;
  }

  TfLiteStatus AddBuiltin(tflite::BuiltinOperator op,
                          TfLiteRegistration* registration,
                          int version) override {
    // This function is currently not used in the tests.
    return kTfLiteError;
  }
};

}  // namespace
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestInterpreter) {
  tflite::freed = false;
  const tflite::Model* model = tflite::testing::GetSimpleMockModel();
  TF_LITE_MICRO_EXPECT_NE(nullptr, model);
  tflite::MockOpResolver mock_resolver;
  constexpr size_t allocator_buffer_size =
      928 /* optimal arena size at the time of writting. */ +
      16 /* alignment */ + 100 /* some headroom */;
  uint8_t allocator_buffer[allocator_buffer_size];

  // Create a new scope so that we can test the destructor.
  {
    tflite::MicroInterpreter interpreter(model, mock_resolver, allocator_buffer,
                                         allocator_buffer_size,
                                         micro_test::reporter);
    TF_LITE_MICRO_EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
    TF_LITE_MICRO_EXPECT_LE(interpreter.arena_used_bytes(), 928 + 100);
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

  TF_LITE_MICRO_EXPECT_EQ(tflite::freed, true);
}

TF_LITE_MICRO_TEST(TestKernelMemoryPlanning) {
  const tflite::Model* model = tflite::testing::GetSimpleStatefulModel();
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
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(3, input->dims->data[0]);
  input->data.uint8[0] = 2;
  input->data.uint8[1] = 3;
  input->data.uint8[2] = 1;

  uint8_t expected_median = 2;

  {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, interpreter.Invoke());
    TfLiteTensor* median = interpreter.output(0);
    TF_LITE_MICRO_EXPECT_EQ(expected_median, median->data.uint8[0]);
    TfLiteTensor* invoke_count = interpreter.output(1);
    TF_LITE_MICRO_EXPECT_EQ(1, invoke_count->data.i32[0]);
  }

  {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, interpreter.Invoke());
    TfLiteTensor* median = interpreter.output(0);
    TF_LITE_MICRO_EXPECT_EQ(expected_median, median->data.uint8[0]);
    TfLiteTensor* invoke_count = interpreter.output(1);
    TF_LITE_MICRO_EXPECT_EQ(2, invoke_count->data.i32[0]);
  }
}

TF_LITE_MICRO_TEST(TestVariableTensorReset) {
  const tflite::Model* model = tflite::testing::GetComplexMockModel();
  TF_LITE_MICRO_EXPECT_NE(nullptr, model);

  tflite::MockOpResolver mock_resolver;
  constexpr size_t allocator_buffer_size =
      2096 /* optimal arena size at the time of writting. */ +
      16 /* alignment */ + 100 /* some headroom */;
  uint8_t allocator_buffer[allocator_buffer_size];
  tflite::MicroInterpreter interpreter(model, mock_resolver, allocator_buffer,
                                       allocator_buffer_size,
                                       micro_test::reporter);
  TF_LITE_MICRO_EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  TF_LITE_MICRO_EXPECT_LE(interpreter.arena_used_bytes(), 2096 + 100);
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
