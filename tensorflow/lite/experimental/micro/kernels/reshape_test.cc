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
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/simple_tensor_allocator.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"
#include "tensorflow/lite/experimental/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

inline TfLiteTensor CreateInt32Tensor(std::initializer_list<int32_t> data,
                                      TfLiteIntArray* dims, const char* name) {
  TfLiteTensor result;
  result.type = kTfLiteInt32;
  result.data.i32 = const_cast<int32_t*>(data.begin());
  result.dims = dims;
  result.params = {};
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(int32_t);
  result.allocation = nullptr;
  result.name = name;
  result.is_variable = true;
  return result;
}

inline TfLiteTensor CreateInt32ConstTensor(std::initializer_list<int32_t> data,
                                           TfLiteIntArray* dims,
                                           const char* name) {
  auto result = CreateInt32Tensor(data, dims, name);
  result.is_variable = false;
  return result;
}

TfLiteReshapeParams create_params(int* shape_data) {
  TfLiteReshapeParams op_params = {};
  op_params.num_dimensions = shape_data[0];
  for (int i = 0; i < shape_data[0]; ++i)
    op_params.shape[i] = shape_data[i + 1];
  return op_params;
}

// If expected output is empty, the test is expected to fail.
void TestReshapeImpl(TfLiteTensor* input_tensor, TfLiteTensor* shape_tensor,
                     TfLiteTensor* output_tensor, int expected_output_size,
                     const float* expected_output,
                     std::initializer_list<int> expected_dims) {
  TfLiteContext context;
  TfLiteTensor tensors[3];
  if (shape_tensor == nullptr) {
    constexpr int inputs_size = 1;
    constexpr int outputs_size = 1;
    constexpr int tensors_size = inputs_size + outputs_size;
    tensors[0] = *input_tensor;
    tensors[1] = *output_tensor,
    PopulateContext(tensors, tensors_size, &context);
  } else {
    constexpr int inputs_size = 2;
    constexpr int outputs_size = 1;
    constexpr int tensors_size = inputs_size + outputs_size;
    tensors[0] = *input_tensor;
    tensors[1] = *shape_tensor;
    tensors[2] = *output_tensor;
    PopulateContext(tensors, tensors_size, &context);
  }

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_RESHAPE, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);
  TfLiteReshapeParams builtin_data =
      create_params(reinterpret_cast<int*>(output_tensor->dims));
  const char* init_data = reinterpret_cast<const char*>(&builtin_data);
  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }
  TfLiteIntArray* temporaries_array = IntArrayFromInitializer({0});
  TfLiteNode node;
  if (shape_tensor == nullptr) {
    node.inputs = IntArrayFromInitializer({1, 0});
    node.outputs = IntArrayFromInitializer({1, 1});
  } else {
    node.inputs = IntArrayFromInitializer({2, 0, 1});
    node.outputs = IntArrayFromInitializer({1, 2});
  }
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
  if (expected_output_size == 0) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                            registration->invoke(&context, &node));
    return;
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  if (registration->free) {
    registration->free(&context, user_data);
  }
  const int output_dims_count = ElementCount(*output_tensor->dims);
  switch (output_tensor->type) {
    case kTfLiteFloat32:
      for (int i = 0; i < expected_output_size; ++i) {
        TF_LITE_MICRO_EXPECT_NEAR(expected_output[i], output_tensor->data.f[i],
                                  1e-5f);
      }
      break;
    case kTfLiteUInt8:
      for (int i = 0; i < expected_output_size; ++i) {
        TF_LITE_MICRO_EXPECT_NEAR(expected_output[i],
                                  output_tensor->data.uint8[i], 1e-5f);
      }
      break;
    case kTfLiteInt8:
      for (int i = 0; i < expected_output_size; ++i) {
        TF_LITE_MICRO_EXPECT_NEAR(expected_output[i],
                                  output_tensor->data.int8[i], 1e-5f);
      }
      break;
    default:
      break;
  }
  TF_LITE_MICRO_EXPECT_EQ(expected_dims.size(), output_tensor->dims->size);
  for (int i = 0; i < expected_dims.size(); ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_dims.begin()[i],
                              output_tensor->dims->data[i], 1e-5f);
  }
}

void TestReshapeTyped(TfLiteTensor* input_tensor,
                      std::initializer_list<int> shape_dims_data,
                      std::initializer_list<int32_t> shape_data,
                      int* output_dims_data, TfLiteTensor* output_tensor,
                      int expected_output_size, const float* expected_output,
                      std::initializer_list<int> expected_dims) {
  TestReshapeImpl(input_tensor, nullptr, output_tensor, expected_output_size,
                  expected_output, expected_dims);
  TfLiteIntArray* shape_dims = IntArrayFromInitializer(shape_dims_data);
  auto shape_tensor = CreateInt32Tensor(shape_data, shape_dims, "shape_tensor");
  TestReshapeImpl(input_tensor, &shape_tensor, output_tensor,
                  expected_output_size, expected_output, expected_dims);
  auto shape_const_tensor =
      CreateInt32ConstTensor(shape_data, shape_dims, "shape_tensor");
  TestReshapeImpl(input_tensor, &shape_const_tensor, output_tensor,
                  expected_output_size, expected_output, expected_dims);
}

void TestReshape(std::initializer_list<int> input_dims_data,
                 std::initializer_list<float> input_data,
                 std::initializer_list<int> shape_dims_data,
                 std::initializer_list<int32_t> shape_data,
                 int* output_dims_data, float* output_data,
                 std::initializer_list<float> expected_output,
                 std::initializer_list<int> expected_dims) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  int expected_output_size = expected_output.size();
  // Testing float input.
  auto input_tensor = CreateFloatTensor(input_data, input_dims, "input_tensor");
  auto output_tensor =
      CreateFloatTensor(output_data, output_dims, "input_tensor");
  TestReshapeTyped(&input_tensor, shape_dims_data, shape_data, output_dims_data,
                   &output_tensor, expected_output_size,
                   expected_output.begin(), expected_dims);
  // Testing uint8 input.
  float expected_uint8[16], expected_int8[16];
  uint8_t input_uint8[16], output_uint8[16];
  int8_t input_int8[16], output_int8[16];
  float input_min = 0;
  float input_max = 15.9375;
  for (int i = 0; i < input_data.size(); ++i) {
    input_uint8[i] = F2Q(input_data.begin()[i], input_min, input_max);
  }
  for (int i = 0; i < expected_output.size(); ++i) {
    expected_uint8[i] = F2Q(expected_output.begin()[i], input_min, input_max);
  }
  input_tensor = CreateQuantizedTensor(input_uint8, input_dims, "input_tensor",
                                       input_min, input_max);
  output_tensor = CreateQuantizedTensor(output_uint8, output_dims,
                                        "input_tensor", input_min, input_max);
  TestReshapeTyped(&input_tensor, shape_dims_data, shape_data, output_dims_data,
                   &output_tensor, expected_output_size, expected_uint8,
                   expected_dims);
}
}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(MismatchedDimensions) {
  float output_data[8];
  int output_dims[3] = {2, 2, 1};
  tflite::testing::TestReshape({4, 1, 2, 4, 1},  // input_dims
                               {3},              // input_data
                               {1, 2},           // shape_dims
                               {2, 1},           // shape_data
                               output_dims,      // output_dims
                               output_data, {},  // expected_output
                               {}                // expected_dims
  );
}

TF_LITE_MICRO_TEST(TooManyDimensions) {
  float output_data[2];
  int output_dims[10] = {9, 1, 1, 1, 1, 1, 1, 1, 1, 2};
  tflite::testing::TestReshape({9, 1, 1, 2, 1, 1, 1, 1, 1, 1},  // input_dims
                               {3, 2},                          // input_data
                               {1, 9},                          // shape_dims
                               {1, 1, 1, 1, 1, 1, 1, 1, 2},     // shape_data
                               output_dims,                     // output_dims
                               output_data, {3, 2},         // expected_output
                               {1, 1, 1, 1, 1, 1, 1, 1, 2}  // expected_dims
  );
}

// Number of dimensions > 8 is accepted in micro since it does not use
// TfLiteReshapeParams.
TF_LITE_MICRO_TEST(TooManySpecialDimensions) {
  float output_data[8];
  int output_dims[5] = {4, -1, -1, 2, 4};
  tflite::testing::TestReshape({4, 1, 2, 4, 1},  // input_dims
                               {3},              // input_data
                               {1, 4},           // shape_dims
                               {-1, -1, 2, 4},   // shape_data
                               output_dims,      // output_dims
                               output_data, {},  // expected_output
                               {}                // expected_dims
  );
}

// Create the model with a 2x2 shape. Processing still works because the new
// shape ends up being hardcoded as a flat vector.
TF_LITE_MICRO_TEST(InvalidShape) {
  using tflite::testing::CreateFloatTensor;
  using tflite::testing::IntArrayFromInitializer;
  using tflite::testing::IntArrayFromInts;
  TfLiteIntArray* input_dims = IntArrayFromInitializer({3, 1, 2, 2});
  auto input_data = {3.0f};
  auto input_tensor = CreateFloatTensor(input_data, input_dims, "input_tensor");
  float output_data[4];
  int output_dims_data[6] = {2, 2, 1, 2, 2, 1};
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  auto output_tensor =
      CreateFloatTensor(output_data, output_dims, "input_tensor");
  tflite::testing::TestReshapeImpl(&input_tensor,          // input_tensor
                                   nullptr,                // shape_tensor
                                   &output_tensor, 0, {},  // expected_output
                                   {}                      // expected_dims
  );
}

TF_LITE_MICRO_TEST(RegularShapes) {
  float output_data[8];
  int output_dims[4] = {3, 2, 2, 2};
  tflite::testing::TestReshape({4, 1, 2, 4, 1},           // input_dims
                               {1, 2, 3, 4, 5, 6, 7, 8},  // input_data
                               {1, 3},                    // shape_dims
                               {2, 2, 2},                 // shape_data
                               output_dims,               // output_dims
                               output_data,
                               {1, 2, 3, 4, 5, 6, 7, 8},  // expected_output
                               {2, 2, 2}                  // expected_dims
  );
}

TF_LITE_MICRO_TEST(WithStretchDimension) {
  float output_data[8];
  int output_dims[4] = {3, 2, 1, -1};
  tflite::testing::TestReshape({4, 1, 2, 4, 1},           // input_dims
                               {1, 2, 3, 4, 5, 6, 7, 8},  // input_data
                               {1, 3},                    // shape_dims
                               {2, 1, -1},                // shape_data
                               output_dims,               // output_dims
                               output_data,
                               {1, 2, 3, 4, 5, 6, 7, 8},  // expected_output
                               {2, 1, 4}                  // expected_dims
  );
}

// Shape is specified as '[]', which is the modern way to represent scalar
// input and output.
TF_LITE_MICRO_TEST(ScalarOutput) {
  float output_data[1];
  int output_dims[1] = {0};
  tflite::testing::TestReshape({1, 1},            // input_dims
                               {3},               // input_data
                               {0},               // shape_dims
                               {},                // shape_data
                               output_dims,       // output_dims
                               output_data, {3},  // expected_output
                               {}                 // expected_dims
  );
}

// Some old models specify '[0]' as the new shape, indicating that both input
// and output are scalars.
TF_LITE_MICRO_TEST(LegacyScalarOutput) {
  using tflite::testing::CreateFloatTensor;
  using tflite::testing::IntArrayFromInitializer;
  using tflite::testing::IntArrayFromInts;
  TfLiteIntArray* input_dims = IntArrayFromInitializer({1, 1});
  auto input_data = {3.0f};
  auto input_tensor = CreateFloatTensor(input_data, input_dims, "input_tensor");
  float output_data[1];
  int output_dims_data[2] = {1, 0};
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  auto output_tensor =
      CreateFloatTensor(output_data, output_dims, "input_tensor");
  TfLiteIntArray* shape_dims = tflite::testing::IntArrayFromInitializer({1, 0});
  auto shape_tensor =
      tflite::testing::CreateInt32Tensor({0}, shape_dims, "shape_tensor");
  tflite::testing::TestReshapeImpl(&input_tensor,          // input_tensor
                                   &shape_tensor,          // shape_tensor
                                   &output_tensor, 0, {},  // expected_output
                                   {}                      // expected_dims
  );
  auto shape_const_tensor =
      tflite::testing::CreateInt32ConstTensor({0}, shape_dims, "shape_tensor");
  tflite::testing::TestReshapeImpl(&input_tensor,          // input_tensor
                                   &shape_const_tensor,    // shape_tensor
                                   &output_tensor, 0, {},  // expected_output
                                   {}                      // expected_dims
  );
  float expected_ouput[1] = {3};
  tflite::testing::TestReshapeImpl(&input_tensor,  // input_tensor
                                   nullptr,        // shape_tensor
                                   &output_tensor, 1,
                                   expected_ouput,  // expected_output
                                   {}               // expected_dims
  );
}

TF_LITE_MICRO_TESTS_END
