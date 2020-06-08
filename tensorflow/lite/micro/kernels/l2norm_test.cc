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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"


namespace tflite {
namespace testing {
namespace {

// used to set the quantization parameters for the int8 and uint8 tests
constexpr float kInputMin = -2.0;
constexpr float kInputMax = 2.0;
constexpr float kOutputMin = -1.0;
constexpr float kOutputMax = 127.0 / 128.0;


void QuantizeInputData(const float input_data[], int length,
                       uint8_t* quantized_data) {
  for (int i=0; i < 6; i++) {
    quantized_data[i] = tflite::testing::F2Q(input_data[i],
                                             tflite::testing::kInputMin,
                                             tflite::testing::kInputMax);
  }
}

void QuantizeInputData(const float input_data[], int length,
                       int8_t* quantized_data) {
  for (int i=0; i < 6; i++) {
    quantized_data[i] = tflite::testing::F2QS(input_data[i],
                                             tflite::testing::kInputMin,
                                             tflite::testing::kInputMax);
  }
}

TfLiteTensor CreateL2NormTensor(const float* data, TfLiteIntArray* dims,
                              const char* name, bool is_input) {
  return CreateFloatTensor(data, dims, name);
}

TfLiteTensor CreateL2NormTensor(const uint8* data, TfLiteIntArray* dims,
                              const char* name, bool is_input) {
  TfLiteTensor tensor;

  if (is_input) {
    tensor = CreateQuantizedTensor(data, dims, name, kInputMin, kInputMax);
  } else {
    tensor = CreateQuantizedTensor(data, dims, name, kOutputMin, kOutputMax);
  }

  tensor.quantization.type = kTfLiteAffineQuantization;
  return tensor;
}

TfLiteTensor CreateL2NormTensor(const int8* data, TfLiteIntArray* dims,
                              const char* name, bool is_input) {
  TfLiteTensor tensor;

  if (is_input) {
    tensor = CreateQuantizedTensor(data, dims, name, kInputMin, kInputMax);
  } else {
    tensor = CreateQuantizedTensor(data, dims, name, kOutputMin, kOutputMax);
  }

  tensor.quantization.type = kTfLiteAffineQuantization;
  return tensor;
}

template <typename T>
inline float Dequantize(const T data, float scale, int32_t zero_point) {
  return scale * (data - zero_point);
}

template<typename T>
void TestL2Normalization(const int* input_dims_data,
                               const T* input_data,
                               const float* expected_output_data,
                               T* output_data, float variance) {
  TfLiteIntArray* dims = IntArrayFromInts(input_dims_data);

  const int output_dims_count = ElementCount(*dims);

  constexpr int tensors_size = 2;
  TfLiteTensor tensors[tensors_size] = {
      CreateL2NormTensor(input_data, dims, "input_tensor", true),
      CreateL2NormTensor(output_data, dims, "output_tensor", false),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);
  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_L2_NORMALIZATION);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteL2NormParams builtin_data = {
    .activation = kTfLiteActNone,
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  int temporaries_array_data[] = {0};
  TfLiteIntArray* temporaries_array = IntArrayFromInts(temporaries_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = nullptr;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));

  // Compare the results from dequantization and expected outputs, and make
  // sure the difference is within a threshold.
  if (tensors[1].quantization.type != kTfLiteNoQuantization) {
    TfLiteTensor* output_tensor = &tensors[1];
    int32_t zero_point = output_tensor->params.zero_point;
    float scale = output_tensor->params.scale;

    for (int i = 0; i < output_dims_count; ++i) {
      float output_val = Dequantize(output_data[i], scale, zero_point);

      TF_LITE_MICRO_EXPECT_LE(expected_output_data[i] - variance, output_val);
      TF_LITE_MICRO_EXPECT_GE(expected_output_data[i] + variance, output_val);
    }
  } else {
    for (int i = 0; i < output_dims_count; ++i) {
      float output_val = static_cast<float>(output_data[i]);
      TF_LITE_MICRO_EXPECT_LE(expected_output_data[i] - variance, output_val);
      TF_LITE_MICRO_EXPECT_GE(expected_output_data[i] + variance, output_val);
    }
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite


TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SimpleFloatTest) {
  const int input_dims[] = {4, 1, 1, 1, 6};
  constexpr int data_length = 6;
  const float input_data[data_length] = {
    -1.1, 0.6, 0.7, 1.2, -0.7, 0.1
  };
  const float expected_output_data[data_length] = {
    -0.55, 0.3, 0.35, 0.6, -0.35, 0.05
  };
  float output_data[data_length];

  tflite::testing::TestL2Normalization<float>(input_dims, input_data,
    expected_output_data, output_data, 0);
}

TF_LITE_MICRO_TEST(ZerosVectorFloatTest) {
  const int input_dims[] = {4, 1, 1, 1, 6};
  constexpr int data_length = 6;
  const float input_data[data_length] = {0, 0, 0, 0, 0, 0};
  const float expected_output_data[data_length] = {0, 0, 0, 0, 0, 0};
  float output_data[data_length];

  tflite::testing::TestL2Normalization<float>(input_dims, input_data,
    expected_output_data, output_data, 0);
}

TF_LITE_MICRO_TEST(SimpleFloatWithRankLessThanFourTest) {
  const int input_dims[] = {4, 1, 1, 1, 6};
  constexpr int data_length = 6;
  const float input_data[data_length] = {
    -1.1, 0.6, 0.7, 1.2, -0.7, 0.1
  };
  const float expected_output_data[data_length] = {
    -0.55, 0.3, 0.35, 0.6, -0.35, 0.05
  };
  float output_data[data_length];

  tflite::testing::TestL2Normalization<float>(input_dims, input_data,
    expected_output_data, output_data, 0);
}

TF_LITE_MICRO_TEST(MultipleBatchFloatTest) {
  const int input_dims[] = {4, 3, 1, 1, 6};
  constexpr int data_length = 18;
  const float input_data[data_length] = {
    -1.1, 0.6, 0.7, 1.2, -0.7, 0.1,  // batch 1
    -1.1, 0.6, 0.7, 1.2, -0.7, 0.1,  // batch 2
    -1.1, 0.6, 0.7, 1.2, -0.7, 0.1,  // batch 3
  };
  const float expected_output_data[data_length] = {
    -0.55, 0.3, 0.35, 0.6, -0.35, 0.05,  // batch 1
    -0.55, 0.3, 0.35, 0.6, -0.35, 0.05,  // batch 2
    -0.55, 0.3, 0.35, 0.6, -0.35, 0.05,  // batch 3
  };
  float output_data[data_length];

  tflite::testing::TestL2Normalization<float>(input_dims, input_data,
    expected_output_data, output_data, 0);
}

TF_LITE_MICRO_TEST(ZerosVectorUint8Test) {
  const int input_dims[] = {4, 1, 1, 1, 6};
  constexpr int data_length = 6;
  const float input_data[data_length] = {0};
  const float expected_output_data[data_length] = {0};
  uint8_t quantized_input[data_length];
  uint8_t output_data[data_length];

  tflite::testing::QuantizeInputData(input_data, data_length, quantized_input);

  tflite::testing::TestL2Normalization<uint8_t>(input_dims, quantized_input,
    expected_output_data, output_data, .1);
}

TF_LITE_MICRO_TEST(SimpleUint8Test) {
  const int input_dims[] = {4, 1, 1, 1, 6};
  constexpr int data_length = 6;
  float input_data[data_length] = {
    -1.1, 0.6, 0.7, 1.2, -0.7, 0.1
  };
  float expected_output[data_length] = {
    -0.55, 0.3, 0.35, 0.6, -0.35, 0.05
  };
  uint8_t quantized_input[data_length];
  uint8_t output_data[data_length];

  tflite::testing::QuantizeInputData(input_data, data_length, quantized_input);

  tflite::testing::TestL2Normalization<uint8_t>(input_dims, quantized_input,
    expected_output, output_data, .1);
}

TF_LITE_MICRO_TEST(SimpleInt8Test) {
  const int input_dims[] = {4, 1, 1, 1, 6};
  constexpr int data_length = 6;
  float input_data[data_length] = {
    -1.1, 0.6, 0.7, 1.2, -0.7, 0.1
  };
  float expected_output[data_length] = {
    -0.55, 0.3, 0.35, 0.6, -0.35, 0.05
  };
  int8_t quantized_input[data_length];
  int8_t output_data[data_length];

  tflite::testing::QuantizeInputData(input_data, data_length, quantized_input);

  tflite::testing::TestL2Normalization<int8_t>(input_dims, quantized_input,
    expected_output, output_data, .1);
}

TF_LITE_MICRO_TEST(ZerosVectorInt8Test) {
  const int input_dims[] = {4, 1, 1, 1, 6};
  constexpr int data_length = 6;
  const float input_data[data_length] = {0};
  const float expected_output_data[data_length] = {0};
  int8_t quantized_input[data_length];
  int8_t output_data[data_length];

  tflite::testing::QuantizeInputData(input_data, data_length, quantized_input);

  tflite::testing::TestL2Normalization<int8_t>(input_dims, quantized_input,
    expected_output_data, output_data, .1);
}

TF_LITE_MICRO_TEST(MultipleBatchUint8Test) {
  const int input_dims[] = {4, 1, 1, 1, 6};
  constexpr int data_length = 18;
  float input_data[data_length] = {
    -1.1, 0.6, 0.7, 1.2, -0.7, 0.1,  // batch 1
    -1.1, 0.6, 0.7, 1.2, -0.7, 0.1,  // batch 2
    -1.1, 0.6, 0.7, 1.2, -0.7, 0.1,  // batch 3
  };
  float expected_output[data_length] = {
    -0.55, 0.3, 0.35, 0.6, -0.35, 0.05,  // batch 1
    -0.55, 0.3, 0.35, 0.6, -0.35, 0.05,  // batch 2
    -0.55, 0.3, 0.35, 0.6, -0.35, 0.05,  // batch 3
  };
  uint8_t quantized_input[data_length];
  uint8_t output_data[data_length];

  tflite::testing::QuantizeInputData(input_data, data_length, quantized_input);

  tflite::testing::TestL2Normalization<uint8_t>(input_dims, quantized_input,
    expected_output, output_data, .1);
}

TF_LITE_MICRO_TEST(MultipleBatchInt8Test) {
  const int input_dims[] = {4, 1, 1, 1, 6};
  constexpr int data_length = 18;
  float input_data[data_length] = {
    -1.1, 0.6, 0.7, 1.2, -0.7, 0.1,  // batch 1
    -1.1, 0.6, 0.7, 1.2, -0.7, 0.1,  // batch 2
    -1.1, 0.6, 0.7, 1.2, -0.7, 0.1,  // batch 3
  };
  float expected_output[data_length] = {
    -0.55, 0.3, 0.35, 0.6, -0.35, 0.05,  // batch 1
    -0.55, 0.3, 0.35, 0.6, -0.35, 0.05,  // batch 2
    -0.55, 0.3, 0.35, 0.6, -0.35, 0.05,  // batch 3
  };
  int8_t quantized_input[data_length];
  int8_t output_data[data_length];

  tflite::testing::QuantizeInputData(input_data, data_length, quantized_input);

  tflite::testing::TestL2Normalization<int8_t>(input_dims, quantized_input,
    expected_output, output_data, .1);
}

TF_LITE_MICRO_TESTS_END
