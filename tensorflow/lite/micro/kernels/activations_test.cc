/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

void TestReluFloat(const int* input_dims_data, const float* input_data,
                   const int* output_dims_data, const float* golden,
                   float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims),
      CreateFloatTensor(output_data, output_dims),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_RELU);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  const char* init_data = nullptr;
  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = nullptr;
  node.user_data = user_data;
  node.builtin_data = nullptr;
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;
  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  if (registration->free) {
    registration->free(&context, user_data);
  }
  for (int i = 0; i < output_elements_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output_data[i], 1e-5f);
  }
}

void TestRelu6Float(const int* input_dims_data, const float* input_data,
                    const int* output_dims_data, const float* golden,
                    float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims),
      CreateFloatTensor(output_data, output_dims),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_RELU6);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  const char* init_data = nullptr;
  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = nullptr;
  node.user_data = user_data;
  node.builtin_data = nullptr;
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;
  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  if (registration->free) {
    registration->free(&context, user_data);
  }
  for (int i = 0; i < output_elements_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output_data[i], 1e-5f);
  }
}

void TestReluUint8(const int* input_dims_data, const float* input_data,
                   uint8_t* input_data_quantized, const float input_scale,
                   const int input_zero_point, const float* golden,
                   uint8_t* golden_quantized, const int* output_dims_data,
                   const float output_scale, const int output_zero_point,
                   uint8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_data_quantized, input_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_RELU);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  const char* init_data = nullptr;
  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = nullptr;
  node.user_data = user_data;
  node.builtin_data = nullptr;
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;
  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  if (registration->free) {
    registration->free(&context, user_data);
  }

  AsymmetricQuantize(golden, golden_quantized, output_elements_count,
                     output_scale, output_zero_point);

  for (int i = 0; i < output_elements_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(golden_quantized[i], output_data[i]);
  }
}

void TestRelu6Uint8(const int* input_dims_data, const float* input_data,
                    uint8_t* input_data_quantized, const float input_scale,
                    const int input_zero_point, const float* golden,
                    uint8_t* golden_quantized, const int* output_dims_data,
                    const float output_scale, const int output_zero_point,
                    uint8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_data_quantized, input_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_RELU6);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  const char* init_data = nullptr;
  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = nullptr;
  node.user_data = user_data;
  node.builtin_data = nullptr;
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;
  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  if (registration->free) {
    registration->free(&context, user_data);
  }

  AsymmetricQuantize(golden, golden_quantized, output_elements_count,
                     output_scale, output_zero_point);

  for (int i = 0; i < output_elements_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(golden_quantized[i], output_data[i]);
  }
}

void TestReluInt8(const int* input_dims_data, const float* input_data,
                  int8_t* input_data_quantized, const float input_scale,
                  const int input_zero_point, const float* golden,
                  int8_t* golden_quantized, const int* output_dims_data,
                  const float output_scale, const int output_zero_point,
                  int8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);
  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_data_quantized, input_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_RELU);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  const char* init_data = nullptr;
  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = nullptr;
  node.user_data = user_data;
  node.builtin_data = nullptr;
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;
  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));

  if (registration->free) {
    registration->free(&context, user_data);
  }

  AsymmetricQuantize(golden, golden_quantized, output_elements_count,
                     output_scale, output_zero_point);

  for (int i = 0; i < output_elements_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(golden_quantized[i], output_data[i]);
  }
}

void TestRelu6Int8(const int* input_dims_data, const float* input_data,
                   int8_t* input_data_quantized, const float input_scale,
                   const int input_zero_point, const float* golden,
                   int8_t* golden_quantized, const int* output_dims_data,
                   const float output_scale, const int output_zero_point,
                   int8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);
  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_data_quantized, input_dims,
                            input_scale, input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_RELU6);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  const char* init_data = nullptr;
  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }
  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = nullptr;
  node.user_data = user_data;
  node.builtin_data = nullptr;
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;
  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));

  if (registration->free) {
    registration->free(&context, user_data);
  }

  AsymmetricQuantize(golden, golden_quantized, output_elements_count,
                     output_scale, output_zero_point);

  for (int i = 0; i < output_elements_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(golden_quantized[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SimpleReluTestFloat) {
  const int output_elements_count = 10;
  const int input_shape[] = {2, 1, 5};
  const float input_data[] = {
      1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, -4.0, -5.0,
  };
  const float golden[] = {1.0, 2.0, 3.0, 4.0, 5.0, 0, 0, 0, 0, 0};
  const int output_shape[] = {2, 1, 5};
  float output_data[output_elements_count];
  tflite::testing::TestReluFloat(input_shape, input_data, output_shape, golden,
                                 output_data);
}

TF_LITE_MICRO_TEST(SimpleRelu6TestFloat) {
  const int output_elements_count = 10;
  float output_data[output_elements_count];
  const int input_shape[] = {2, 1, 5};
  const float input_data[] = {4.0,  5.0,  6.0,  7.0,  8.0,
                              -4.0, -5.0, -6.0, -7.0, -8.0};
  const int output_shape[] = {2, 1, 5};
  const float golden[] = {
      4.0, 5.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  };

  tflite::testing::TestRelu6Float(input_shape, input_data, output_shape, golden,
                                  output_data);
}

TF_LITE_MICRO_TEST(SimpleReluTestUint8) {
  const int elements_count = 10;

  const int input_shape[] = {2, 1, 5};
  const float input_data[] = {1, 2, 3, 4, 5, -1, -2, -3, -4, -5};
  uint8_t input_quantized[elements_count];
  const int output_shape[] = {2, 1, 5};
  const float golden[] = {1, 2, 3, 4, 5, 0, 0, 0, 0, 0};
  uint8_t golden_quantized[elements_count];
  uint8_t output_data[elements_count];

  const float input_scale = 0.5f;
  const int input_zero_point = 127;
  const float output_scale = 0.5f;
  const int output_zero_point = 127;

  tflite::testing::TestReluUint8(input_shape, input_data, input_quantized,
                                 input_scale, input_zero_point, golden,
                                 golden_quantized, output_shape, output_scale,
                                 output_zero_point, output_data);
}

TF_LITE_MICRO_TEST(SimpleRelu6TestUint8) {
  const int elements_count = 10;

  const int input_shape[] = {2, 1, 5};
  const float input_data[] = {4, 5, 6, 7, 8, -1, -2, -3, -4, -5};
  uint8_t input_quantized[elements_count];
  const int output_shape[] = {2, 1, 5};
  const float golden[] = {4, 5, 6, 6, 6, 0, 0, 0, 0, 0};
  uint8_t golden_quantized[elements_count];
  uint8_t output_data[elements_count];

  const float input_scale = 0.5f;
  const int input_zero_point = 127;
  const float output_scale = 0.5f;
  const int output_zero_point = 127;

  tflite::testing::TestRelu6Uint8(input_shape, input_data, input_quantized,
                                  input_scale, input_zero_point, golden,
                                  golden_quantized, output_shape, output_scale,
                                  output_zero_point, output_data);
}

TF_LITE_MICRO_TEST(SimpleReluTestInt8) {
  const int elements_count = 10;

  const int input_shape[] = {2, 1, 5};
  const float input_data[] = {1, 2, 3, 4, 5, -1, -2, -3, -4, -5};
  int8_t input_quantized[elements_count];
  const int output_shape[] = {2, 1, 5};
  const float golden[] = {1, 2, 3, 4, 5, 0, 0, 0, 0, 0};
  int8_t golden_quantized[elements_count];
  int8_t output_data[elements_count];

  const float input_scale = 0.5f;
  const int input_zero_point = 0;
  const float output_scale = 0.5f;
  const int output_zero_point = 0;

  tflite::testing::TestReluInt8(input_shape, input_data, input_quantized,
                                input_scale, input_zero_point, golden,
                                golden_quantized, output_shape, output_scale,
                                output_zero_point, output_data);
}

TF_LITE_MICRO_TEST(SimpleRelu6TestInt8) {
  const int elements_count = 10;

  const int input_shape[] = {2, 1, 5};
  const float input_data[] = {4, 5, 6, 7, 8, -1, -2, -3, -4, -5};
  int8_t input_quantized[elements_count];
  const int output_shape[] = {2, 1, 5};
  const float golden[] = {4, 5, 6, 6, 6, 0, 0, 0, 0, 0};
  int8_t golden_quantized[elements_count];
  int8_t output_data[elements_count];

  const float input_scale = 0.5f;
  const int input_zero_point = 127;
  const float output_scale = 0.5f;
  const int output_zero_point = 127;

  tflite::testing::TestRelu6Int8(input_shape, input_data, input_quantized,
                                 input_scale, input_zero_point, golden,
                                 golden_quantized, output_shape, output_scale,
                                 output_zero_point, output_data);
}

TF_LITE_MICRO_TESTS_END
