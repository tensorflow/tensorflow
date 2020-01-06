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

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

// Shapes and values for mixed broadcast tests.
const int broadcast_output_dims_count = 36;
const int broadcast_num_shapes = 4;

const int broadcast_input1_shape[] = {4, 2, 3, 1, 2};
const float broadcast_input1_values[] = {-0.3, 2.3, 0.9,  0.5, 0.8, -1.1,
                                         1.2,  2.8, -1.6, 0.0, 0.7, -2.2};
const float broadcast_input2_values[] = {0.2, 0.3, -0.4, 0.5, 1.0, 0.9};
const float
    broadcast_goldens[broadcast_num_shapes][broadcast_output_dims_count] = {
        {-0.1, 2.6,  -0.7, 2.8,  0.7,  3.2,  1.1, 0.8,  0.5, 1.0,  1.9, 1.4,
         1.0,  -0.8, 0.4,  -0.6, 1.8,  -0.2, 1.4, 3.1,  0.8, 3.3,  2.2, 3.7,
         -1.4, 0.3,  -2.0, 0.5,  -0.6, 0.9,  0.9, -1.9, 0.3, -1.7, 1.7, -1.3},
        {-0.1, 2.6, 0.5, 1.0, 1.8, -0.2, 1.4, 3.1, -2.0, 0.5, 1.7, -1.3},
        {-0.1, 2.5,  0.0,  2.6,  -0.7, 1.9,  1.1, 0.7,  1.2, 0.8,  0.5, 0.1,
         1.0,  -0.9, 1.1,  -0.8, 0.4,  -1.5, 1.7, 3.3,  2.2, 3.8,  2.1, 3.7,
         -1.1, 0.5,  -0.6, 1.0,  -0.7, 0.9,  1.2, -1.7, 1.7, -1.2, 1.6, -1.3},
        {-0.1, 2.5, 1.2, 0.8, 0.4, -1.5, 1.7, 3.3, -0.6, 1.0, 1.6, -1.3},
};

const int broadcast_max_shape_size = 5;
const int broadcast_input2_shapes[broadcast_num_shapes]
                                 [broadcast_max_shape_size] = {
                                     {4, 1, 1, 3, 2},
                                     {4, 1, 3, 1, 2},
                                     {4, 2, 1, 3, 1},
                                     {4, 2, 3, 1, 1},
};
const int broadcast_output_shapes[broadcast_num_shapes]
                                 [broadcast_max_shape_size] = {
                                     {4, 2, 3, 3, 2},
                                     {4, 2, 3, 1, 2},
                                     {4, 2, 3, 3, 2},
                                     {4, 2, 3, 1, 2},
};

template <typename T>
void ValidateAddGoldens(TfLiteTensor* tensors, int tensors_size,
                        const T* golden, T* output, int output_size,
                        TfLiteFusedActivation activation,
                        float tolerance = 1e-5) {
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(::tflite::BuiltinOperator_ADD, 1);

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteAddParams builtin_data;
  builtin_data.activation = activation;

  const char* init_data = reinterpret_cast<const char*>(&builtin_data);
  const size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  int temporaries_array_data[] = {0};
  TfLiteIntArray* temporaries_array = IntArrayFromInts(temporaries_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
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

  for (int i = 0; i < output_size; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output[i], tolerance);
  }
}

void TestAddFloat(const int* input1_dims_data, const float* input1_data,
                  const int* input2_dims_data, const float* input2_data,
                  const int* output_dims_data, const float* expected_output,
                  TfLiteFusedActivation activation, float* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input1_data, input1_dims, "input1_tensor"),
      CreateFloatTensor(input2_data, input2_dims, "input2_tensor"),
      CreateFloatTensor(output_data, output_dims, "output_tensor"),
  };

  ValidateAddGoldens(tensors, tensors_size, expected_output, output_data,
                     ElementCount(*output_dims), activation);
}

template <typename T>
void TestAddQuantized(const int* input1_dims_data, const float* input1_data,
                      T* input1_quantized, float input1_scale,
                      int input1_zero_point, const int* input2_dims_data,
                      const float* input2_data, T* input2_quantized,
                      float input2_scale, int input2_zero_point,
                      const int* output_dims_data, const float* golden,
                      T* golden_quantized, float output_scale,
                      int output_zero_point, TfLiteFusedActivation activation,
                      T* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      tflite::testing::CreateQuantizedTensor(
          input1_data, input1_quantized, input1_dims, input1_scale,
          input1_zero_point, "input1_tensor"),
      tflite::testing::CreateQuantizedTensor(
          input2_data, input2_quantized, input2_dims, input2_scale,
          input2_zero_point, "input2_tensor"),
      tflite::testing::CreateQuantizedTensor(output_data, output_dims,
                                             output_scale, output_zero_point,
                                             "output_tensor"),
  };
  tflite::AsymmetricQuantize(golden, golden_quantized,
                             ElementCount(*output_dims), output_scale,
                             output_zero_point);

  ValidateAddGoldens(tensors, tensors_size, golden_quantized, output_data,
                     ElementCount(*output_dims), activation);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloatAddNoActivation) {
  const int output_dims_count = 4;
  const int inout_shape[] = {4, 1, 2, 2, 1};
  const float input1_values[] = {-2.0, 0.2, 0.7, 0.8};
  const float input2_values[] = {0.1, 0.2, 0.3, 0.5};
  const float golden_values[] = {-1.9, 0.4, 1.0, 1.3};
  float output_data[output_dims_count];
  tflite::testing::TestAddFloat(inout_shape, input1_values, inout_shape,
                                input2_values, inout_shape, golden_values,
                                kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(FloatAddActivationRelu1) {
  const int output_dims_count = 4;
  const int inout_shape[] = {4, 1, 2, 2, 1};
  const float input1_values[] = {-2.0, 0.2, 0.7, 0.8};
  const float input2_values[] = {0.1, 0.2, 0.3, 0.5};
  const float golden_values[] = {-1.0, 0.4, 1.0, 1.0};

  float output_data[output_dims_count];
  tflite::testing::TestAddFloat(inout_shape, input1_values, inout_shape,
                                input2_values, inout_shape, golden_values,
                                kTfLiteActRelu1, output_data);
}

TF_LITE_MICRO_TEST(FloatAddVariousInputShapes) {
  const int output_dims_count = 6;
  float output_data[output_dims_count];

  const float input1_values[] = {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0};
  const float input2_values[] = {0.1, 0.2, 0.3, 0.5, 1.1, 0.1};
  const float expected_output[] = {-1.9, 0.4, 1.0, 1.3, 2.2, 2.1};

  constexpr int num_shapes = 4;
  constexpr int max_shape_size = 5;
  const int test_shapes[num_shapes][max_shape_size] = {
      {1, 6},
      {2, 2, 3},
      {3, 2, 1, 3},
      {4, 1, 3, 1, 2},
  };

  for (int i = 0; i < num_shapes; ++i) {
    tflite::testing::TestAddFloat(test_shapes[i], input1_values, test_shapes[i],
                                  input2_values, test_shapes[i],
                                  expected_output, kTfLiteActNone, output_data);
  }
}

TF_LITE_MICRO_TEST(FloatAddWithScalarBroadcast) {
  const int output_dims_count = 6;
  float output_data[output_dims_count];

  const float input1_values[] = {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0};
  const int input2_shape[] = {0};
  const float input2_values[] = {0.1};
  const float expected_output[] = {-1.9, 0.3, 0.8, 0.9, 1.2, 2.1};

  constexpr int num_shapes = 4;
  constexpr int max_shape_size = 5;
  const int test_shapes[num_shapes][max_shape_size] = {
      {1, 6},
      {2, 2, 3},
      {3, 2, 1, 3},
      {4, 1, 3, 1, 2},
  };

  for (int i = 0; i < num_shapes; ++i) {
    tflite::testing::TestAddFloat(test_shapes[i], input1_values, input2_shape,
                                  input2_values, test_shapes[i],
                                  expected_output, kTfLiteActNone, output_data);
  }
}

TF_LITE_MICRO_TEST(QuantizedAddNoActivationUint8) {
  const float scales[] = {0.25, 0.5, 1.0};
  const int zero_points[] = {125, 129, 135};
  const int output_dims_count = 4;
  const int inout_shape[] = {4, 1, 2, 2, 1};
  const float input1_values[] = {-2.01, -1.01, -0.01, 0.98};
  const float input2_values[] = {1.01, 1.99, 2.99, 4.02};
  const float golden_values[] = {-1, 1, 3, 5};

  uint8_t input1_quantized[output_dims_count];
  uint8_t input2_quantized[output_dims_count];
  uint8_t golden_quantized[output_dims_count];
  uint8_t output[output_dims_count];

  tflite::testing::TestAddQuantized(
      inout_shape, input1_values, input1_quantized, scales[0], zero_points[0],
      inout_shape, input2_values, input2_quantized, scales[1], zero_points[1],
      inout_shape, golden_values, golden_quantized, scales[2], zero_points[2],
      kTfLiteActNone, output);
}

TF_LITE_MICRO_TEST(QuantizedAddNoActivationInt8) {
  const float scales[] = {0.25, 0.5, 1.0};
  const int zero_points[] = {-10, 4, 13};
  const int output_dims_count = 4;
  const int inout_shape[] = {4, 1, 2, 2, 1};
  const float input1_values[] = {-2.01, -1.01, -0.01, 0.98};
  const float input2_values[] = {1.01, 1.99, 2.99, 4.02};
  const float golden_values[] = {-1, 1, 3, 5};

  int8_t input1_quantized[output_dims_count];
  int8_t input2_quantized[output_dims_count];
  int8_t golden_quantized[output_dims_count];
  int8_t output[output_dims_count];

  tflite::testing::TestAddQuantized(
      inout_shape, input1_values, input1_quantized, scales[0], zero_points[0],
      inout_shape, input2_values, input2_quantized, scales[1], zero_points[1],
      inout_shape, golden_values, golden_quantized, scales[2], zero_points[2],
      kTfLiteActNone, output);
}

TF_LITE_MICRO_TEST(QuantizedAddActivationRelu1Uint8) {
  const float scales[] = {0.25, 0.5, 1.0};
  const int zero_points[] = {125, 129, 135};
  const int output_dims_count = 4;
  const int inout_shape[] = {4, 1, 2, 2, 1};
  const float input1_values[] = {-2.01, -1.01, -0.01, 0.98};
  const float input2_values[] = {1.01, 1.99, 2.99, 4.02};
  const float golden_values[] = {-1, 1, 1, 1};

  uint8_t input1_quantized[output_dims_count];
  uint8_t input2_quantized[output_dims_count];
  uint8_t golden_quantized[output_dims_count];
  uint8_t output[output_dims_count];

  tflite::testing::TestAddQuantized(
      inout_shape, input1_values, input1_quantized, scales[0], zero_points[0],
      inout_shape, input2_values, input2_quantized, scales[1], zero_points[1],
      inout_shape, golden_values, golden_quantized, scales[2], zero_points[2],
      kTfLiteActRelu1, output);
}

TF_LITE_MICRO_TEST(QuantizedAddActivationRelu1Int8) {
  const float scales[] = {0.25, 0.5, 1.0};
  const int zero_points[] = {-10, 4, 13};
  const int output_dims_count = 4;
  const int inout_shape[] = {4, 1, 2, 2, 1};
  const float input1_values[] = {-2.01, -1.01, -0.01, 0.98};
  const float input2_values[] = {1.01, 1.99, 2.99, 4.02};
  const float golden_values[] = {-1, 1, 1, 1};

  int8_t input1_quantized[output_dims_count];
  int8_t input2_quantized[output_dims_count];
  int8_t golden_quantized[output_dims_count];
  int8_t output[output_dims_count];

  tflite::testing::TestAddQuantized(
      inout_shape, input1_values, input1_quantized, scales[0], zero_points[0],
      inout_shape, input2_values, input2_quantized, scales[1], zero_points[1],
      inout_shape, golden_values, golden_quantized, scales[2], zero_points[2],
      kTfLiteActRelu1, output);
}

TF_LITE_MICRO_TEST(QuantizedAddVariousInputShapesUint8) {
  const float scales[] = {0.1, 0.05, 0.1};
  const int zero_points[] = {120, 130, 139};
  const int output_dims_count = 6;

  constexpr int num_shapes = 4;
  constexpr int max_shape_size = 5;
  const int test_shapes[num_shapes][max_shape_size] = {
      {1, 6},
      {2, 2, 3},
      {3, 2, 1, 3},
      {4, 1, 3, 1, 2},
  };

  const float input1_values[] = {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0};
  const float input2_values[] = {0.1, 0.2, 0.3, 0.5, 1.1, 0.1};
  const float golden_values[] = {-1.9, 0.4, 1.0, 1.3, 2.2, 2.1};

  uint8_t input1_quantized[output_dims_count];
  uint8_t input2_quantized[output_dims_count];
  uint8_t golden_quantized[output_dims_count];
  uint8_t output[output_dims_count];

  for (int i = 0; i < num_shapes; i++) {
    tflite::testing::TestAddQuantized(
        test_shapes[i], input1_values, input1_quantized, scales[0],
        zero_points[0], test_shapes[i], input2_values, input2_quantized,
        scales[1], zero_points[1], test_shapes[i], golden_values,
        golden_quantized, scales[2], zero_points[2], kTfLiteActNone, output);
  }
}

TF_LITE_MICRO_TEST(QuantizedAddVariousInputShapesInt8) {
  const float scales[] = {0.1, 0.05, 0.1};
  const int zero_points[] = {-9, 5, 14};
  const int output_dims_count = 6;

  constexpr int num_shapes = 4;
  constexpr int max_shape_size = 5;
  const int test_shapes[num_shapes][max_shape_size] = {
      {1, 6},
      {2, 2, 3},
      {3, 2, 1, 3},
      {4, 1, 3, 1, 2},
  };

  const float input1_values[] = {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0};
  const float input2_values[] = {0.1, 0.2, 0.3, 0.5, 1.1, 0.1};
  const float golden_values[] = {-1.9, 0.4, 1.0, 1.3, 2.2, 2.1};

  int8_t input1_quantized[output_dims_count];
  int8_t input2_quantized[output_dims_count];
  int8_t golden_quantized[output_dims_count];
  int8_t output[output_dims_count];

  for (int i = 0; i < num_shapes; i++) {
    tflite::testing::TestAddQuantized(
        test_shapes[i], input1_values, input1_quantized, scales[0],
        zero_points[0], test_shapes[i], input2_values, input2_quantized,
        scales[1], zero_points[1], test_shapes[i], golden_values,
        golden_quantized, scales[2], zero_points[2], kTfLiteActNone, output);
  }
}

TF_LITE_MICRO_TEST(QuantizedAddWithScalarBroadcastUint8) {
  const int output_dims_count = 6;

  const float input1_values[] = {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0};
  const int input2_shape[] = {0};
  const float input2_values[] = {0.1};
  const float golden[] = {-1.9, 0.3, 0.8, 0.9, 1.2, 2.1};

  constexpr int num_shapes = 4;
  constexpr int max_shape_size = 5;
  const int test_shapes[num_shapes][max_shape_size] = {
      {1, 6},
      {2, 2, 3},
      {3, 2, 1, 3},
      {4, 1, 3, 1, 2},
  };

  const float scales[] = {0.1, 0.1, 0.1};
  const int zero_points[] = {120, 131, 139};

  uint8_t input1_quantized[output_dims_count];
  uint8_t input2_quantized[output_dims_count];
  uint8_t golden_quantized[output_dims_count];
  uint8_t output[output_dims_count];

  for (int i = 0; i < num_shapes; ++i) {
    tflite::testing::TestAddQuantized(
        test_shapes[i], input1_values, input1_quantized, scales[0],
        zero_points[0], input2_shape, input2_values, input2_quantized,
        scales[1], zero_points[1], test_shapes[i], golden, golden_quantized,
        scales[2], zero_points[2], kTfLiteActNone, output);
  }
}
TF_LITE_MICRO_TEST(QuantizedAddWithScalarBroadcastFloat) {
  const float scales[] = {0.1, 0.05, 0.1};
  const int zero_points[] = {127, 131, 139};
  uint8_t input1_quantized[tflite::testing::broadcast_output_dims_count];
  uint8_t input2_quantized[tflite::testing::broadcast_output_dims_count];
  uint8_t golden_quantized[tflite::testing::broadcast_output_dims_count];
  uint8_t output[tflite::testing::broadcast_output_dims_count];
  float output_float[tflite::testing::broadcast_output_dims_count];

  for (int i = 0; i < tflite::testing::broadcast_num_shapes; ++i) {
    tflite::testing::TestAddFloat(tflite::testing::broadcast_input1_shape,
                                  tflite::testing::broadcast_input1_values,
                                  tflite::testing::broadcast_input2_shapes[i],
                                  tflite::testing::broadcast_input2_values,
                                  tflite::testing::broadcast_output_shapes[i],
                                  tflite::testing::broadcast_goldens[i],
                                  kTfLiteActNone, output_float);
  }
}

TF_LITE_MICRO_TEST(QuantizedAddWithScalarBroadcastInt8) {
  const int output_dims_count = 6;

  const float input1_values[] = {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0};
  const int input2_shape[] = {0};
  const float input2_values[] = {0.1};
  const float golden[] = {-1.9, 0.3, 0.8, 0.9, 1.2, 2.1};

  constexpr int num_shapes = 4;
  constexpr int max_shape_size = 5;
  const int test_shapes[num_shapes][max_shape_size] = {
      {1, 6},
      {2, 2, 3},
      {3, 2, 1, 3},
      {4, 1, 3, 1, 2},
  };

  const float scales[] = {0.1, 0.05, 0.05};
  const int zero_points[] = {-8, 4, 12};

  int8_t input1_quantized[output_dims_count];
  int8_t input2_quantized[output_dims_count];
  int8_t golden_quantized[output_dims_count];
  int8_t output[output_dims_count];

  for (int i = 0; i < num_shapes; ++i) {
    tflite::testing::TestAddQuantized(
        test_shapes[i], input1_values, input1_quantized, scales[0],
        zero_points[0], input2_shape, input2_values, input2_quantized,
        scales[1], zero_points[1], test_shapes[i], golden, golden_quantized,
        scales[2], zero_points[2], kTfLiteActNone, output);
  }
}

TF_LITE_MICRO_TEST(QuantizedAddWithMixedBroadcastUint8) {
  const float scales[] = {0.1, 0.05, 0.1};
  const int zero_points[] = {127, 131, 139};
  uint8_t input1_quantized[tflite::testing::broadcast_output_dims_count];
  uint8_t input2_quantized[tflite::testing::broadcast_output_dims_count];
  uint8_t golden_quantized[tflite::testing::broadcast_output_dims_count];
  uint8_t output[tflite::testing::broadcast_output_dims_count];
  float output_float[tflite::testing::broadcast_output_dims_count];

  for (int i = 0; i < tflite::testing::broadcast_num_shapes; ++i) {
    tflite::testing::TestAddQuantized(
        tflite::testing::broadcast_input1_shape,
        tflite::testing::broadcast_input1_values, input1_quantized, scales[0],
        zero_points[0], tflite::testing::broadcast_input2_shapes[i],
        tflite::testing::broadcast_input2_values, input2_quantized, scales[1],
        zero_points[1], tflite::testing::broadcast_output_shapes[i],
        tflite::testing::broadcast_goldens[i], golden_quantized, scales[2],
        zero_points[2], kTfLiteActNone, output);
  }
}

TF_LITE_MICRO_TEST(QuantizedAddWithMixedBroadcastInt8) {
  const float scales[] = {0.1, 0.05, 0.1};
  const int zero_points[] = {-10, -5, 7};
  int8_t input1_quantized[tflite::testing::broadcast_output_dims_count];
  int8_t input2_quantized[tflite::testing::broadcast_output_dims_count];
  int8_t golden_quantized[tflite::testing::broadcast_output_dims_count];
  int8_t output[tflite::testing::broadcast_output_dims_count];
  float output_float[tflite::testing::broadcast_output_dims_count];

  for (int i = 0; i < tflite::testing::broadcast_num_shapes; ++i) {
    tflite::testing::TestAddQuantized(
        tflite::testing::broadcast_input1_shape,
        tflite::testing::broadcast_input1_values, input1_quantized, scales[0],
        zero_points[0], tflite::testing::broadcast_input2_shapes[i],
        tflite::testing::broadcast_input2_values, input2_quantized, scales[1],
        zero_points[1], tflite::testing::broadcast_output_shapes[i],
        tflite::testing::broadcast_goldens[i], golden_quantized, scales[2],
        zero_points[2], kTfLiteActNone, output);
  }
}

TF_LITE_MICRO_TESTS_END
