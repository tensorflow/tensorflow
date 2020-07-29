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
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"


namespace tflite {
namespace testing {
namespace {

using uint8 = std::uint8_t;
using int32 = std::int32_t;

TfLiteTensor TestCreateTensor(const float* data, TfLiteIntArray* dims) {
  return CreateFloatTensor(data, dims);
}

TfLiteTensor TestCreateTensor(const uint8* data, TfLiteIntArray* dims) {
  return CreateQuantizedTensor(data, dims, 0, 255);
}

TfLiteTensor TestCreateTensor(const int8* data, TfLiteIntArray* dims) {
  return CreateQuantizedTensor(data, dims, -128, 127);
}

// Input data expects a 4-D tensor of [batch, height, width, channels]
// Output data should match input datas batch and channels
// Expected sizes should be a 1-D tensor with 2 elements: new_height & new_width
template <typename T>
void TestResizeNearestNeighbor(const int* input_dims_data, const T* input_data,
                               const int32* expected_size_data,
                               const T* expected_output_data,
                               const int* output_dims_data, T* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);

  int expected_size_dims_data[] = {2, 1, 2};
  TfLiteIntArray* expected_size_dims =
      IntArrayFromInts(expected_size_dims_data);

  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  const int output_dims_count = ElementCount(*output_dims);

  constexpr int tensors_size = 3;
  TfLiteTensor tensors[tensors_size] = {
      TestCreateTensor(input_data, input_dims),
      CreateInt32Tensor(expected_size_data, expected_size_dims),
      TestCreateTensor(output_data, output_dims),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);
  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteResizeNearestNeighborParams builtin_data = {
    .align_corners = false
  };

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
  node.user_data = nullptr;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));

  // compare results
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite


TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(HorizontalResize) {
  const int input_dims[] = {4, 1, 1, 2, 1};
  const float input_data[] = {3, 6};
  const int32 expected_size_data[] = {1, 3};
  const float expected_output_data[] = {3, 3, 6};
  const int output_dims[] = {4, 1, 1, 3, 1};
  float output_data[3];

  tflite::testing::TestResizeNearestNeighbor<float>(input_dims, input_data,
    expected_size_data, expected_output_data, output_dims, output_data);
}
TF_LITE_MICRO_TEST(HorizontalResizeUInt8) {
  const int input_dims[] = {4, 1, 1, 2, 1};
  const uint8 input_data[] = {3, 6};
  const int32 expected_size_data[] = {1, 3};
  const uint8 expected_output_data[] = {3, 3, 6};
  const int output_dims[] = {4, 1, 1, 3, 1};
  uint8 output_data[3];

  tflite::testing::TestResizeNearestNeighbor<uint8>(input_dims, input_data,
    expected_size_data, expected_output_data, output_dims, output_data);
}
TF_LITE_MICRO_TEST(HorizontalResizeInt8) {
  const int input_dims[] = {4, 1, 1, 2, 1};
  const int8 input_data[] = {-3, 6};
  const int32 expected_size_data[] = {1, 3};
  const int8 expected_output_data[] = {-3, -3, 6};
  const int output_dims[] = {4, 1, 1, 3, 1};
  int8 output_data[3];

  tflite::testing::TestResizeNearestNeighbor<int8>(input_dims, input_data,
    expected_size_data, expected_output_data, output_dims, output_data);
}
TF_LITE_MICRO_TEST(VerticalResize) {
  const int input_dims[] = {4, 1, 2, 1, 1};
  const float input_data[] = {3, 9};
  const int32 expected_size_data[] = {3, 1};
  const float expected_output_data[] = {3, 3, 9};
  const int output_dims[] = {4, 1, 3, 1, 1};
  float output_data[3];

  tflite::testing::TestResizeNearestNeighbor<float>(input_dims, input_data,
    expected_size_data, expected_output_data, output_dims, output_data);
}
TF_LITE_MICRO_TEST(VerticalResizeUInt8) {
  const int input_dims[] = {4, 1, 2, 1, 1};
  const uint8 input_data[] = {3, 9};
  const int32 expected_size_data[] = {3, 1};
  const uint8 expected_output_data[] = {3, 3, 9};
  const int output_dims[] = {4, 1, 3, 1, 1};
  uint8 output_data[3];

  tflite::testing::TestResizeNearestNeighbor<uint8>(input_dims, input_data,
    expected_size_data, expected_output_data, output_dims, output_data);
}
TF_LITE_MICRO_TEST(VerticalResizeInt8) {
  const int input_dims[] = {4, 1, 2, 1, 1};
  const int8 input_data[] = {3, -9};
  const int32 expected_size_data[] = {3, 1};
  const int8 expected_output_data[] = {3, 3, -9};
  const int output_dims[] = {4, 1, 3, 1, 1};
  int8 output_data[3];

  tflite::testing::TestResizeNearestNeighbor<int8>(input_dims, input_data,
    expected_size_data, expected_output_data, output_dims, output_data);
}
TF_LITE_MICRO_TEST(TwoDimensionalResize) {
  const int input_dims[] = {4, 1, 2, 2, 1};
  const float input_data[] = {3, 6,   //
                              9, 12,  //
                             };
  const int32 expected_size_data[] = {3, 3};
  const float expected_output_data[] = {3, 3, 6,  //
                                        3, 3, 6,  //
                                        9, 9, 12  //
                                       };

  const int output_dims[] = {4, 1, 3, 3, 1};
  float output_data[9];

  tflite::testing::TestResizeNearestNeighbor<float>(input_dims, input_data,
    expected_size_data, expected_output_data, output_dims, output_data);
}
TF_LITE_MICRO_TEST(TwoDimensionalResizeUInt8) {
  const int input_dims[] = {4, 1, 2, 2, 1};
  const uint8 input_data[] = {3, 6,  //
                              9, 12  //
                             };
  const int32 expected_size_data[] = {3, 3};
  const uint8 expected_output_data[] = {3, 3, 6,  //
                                        3, 3, 6,  //
                                        9, 9, 12  //
                                       };
  const int output_dims[] = {4, 1, 3, 3, 1};
  uint8 output_data[9];

  tflite::testing::TestResizeNearestNeighbor<uint8>(input_dims, input_data,
    expected_size_data, expected_output_data, output_dims, output_data);
}
TF_LITE_MICRO_TEST(TwoDimensionalResizeInt8) {
  const int input_dims[] = {4, 1, 2, 2, 1};
  const int8 input_data[] = {3, -6,  //
                             9, 12,  //
                            };
  const int32 expected_size_data[] = {3, 3};
  const int8 expected_output_data[] = {3, 3, -6,  //
                                       3, 3, -6,  //
                                       9, 9, 12,  //
                                      };
  const int output_dims[] = {4, 1, 3, 3, 1};
  int8 output_data[9];

  tflite::testing::TestResizeNearestNeighbor<int8>(input_dims, input_data,
    expected_size_data, expected_output_data, output_dims, output_data);
}
TF_LITE_MICRO_TEST(TwoDimensionalResizeWithTwoBatches) {
  const int input_dims[] = {4, 2, 2, 2, 1};
  const float input_data[] = {3, 6,   //
                              9, 12,  //
                              4, 10,  //
                              10, 16  //
                             };
  const int32 expected_size_data[] = {3, 3};
  const float expected_output_data[] = {3, 3, 6,     //
                                        3, 3, 6,     //
                                        9, 9, 12,    //
                                        4, 4, 10,    //
                                        4, 4, 10,    //
                                        10, 10, 16,  //
                                       };
  const int output_dims[] = {4, 2, 3, 3, 1};
  float output_data[18];

  tflite::testing::TestResizeNearestNeighbor<float>(input_dims, input_data,
    expected_size_data, expected_output_data, output_dims, output_data);
}
TF_LITE_MICRO_TEST(TwoDimensionalResizeWithTwoBatchesUInt8) {
  const int input_dims[] = {4, 2, 2, 2, 1};
  const uint8 input_data[] = {3, 6,   //
                              9, 12,  //
                              4, 10,  //
                              10, 16  //
                             };
  const int32 expected_size_data[] = {3, 3};
  const uint8 expected_output_data[] = {3, 3, 6,     //
                                        3, 3, 6,     //
                                        9, 9, 12,    //
                                        4, 4, 10,    //
                                        4, 4, 10,    //
                                        10, 10, 16,  //
                                       };
  const int output_dims[] = {4, 2, 3, 3, 1};
  uint8 output_data[18];

  tflite::testing::TestResizeNearestNeighbor<uint8>(input_dims, input_data,
    expected_size_data, expected_output_data, output_dims, output_data);
}
TF_LITE_MICRO_TEST(TwoDimensionalResizeWithTwoBatchesInt8) {
  const int input_dims[] = {4, 2, 2, 2, 1};
  const int8 input_data[] = {3, 6,    //
                             9, -12,  //
                             -4, 10,  //
                             10, 16   //
                            };
  const int32 expected_size_data[] = {3, 3};
  const int8 expected_output_data[] = {3, 3, 6,     //
                                       3, 3, 6,     //
                                       9, 9, -12,   //
                                       -4, -4, 10,  //
                                       -4, -4, 10,  //
                                       10, 10, 16,  //
                                      };
  const int output_dims[] = {4, 2, 3, 3, 1};
  int8 output_data[18];

  tflite::testing::TestResizeNearestNeighbor<int8>(input_dims, input_data,
    expected_size_data, expected_output_data, output_dims, output_data);
}
TF_LITE_MICRO_TEST(ThreeDimensionalResize) {
  const int input_dims[] = {4, 1, 2, 2, 2};
  const float input_data[] = {3, 4, 6, 10,    //
                              9, 10, 12, 16,  //
                             };
  const int32 expected_size_data[] = {3, 3};
  const float expected_output_data[] = {3, 4, 3, 4, 6, 10,     //
                                        3, 4, 3, 4, 6, 10,     //
                                        9, 10, 9, 10, 12, 16,  //
                                     };
  const int output_dims[] = {4, 1, 3, 3, 2};
  float output_data[18];

  tflite::testing::TestResizeNearestNeighbor<float>(input_dims, input_data,
    expected_size_data, expected_output_data, output_dims, output_data);
}
TF_LITE_MICRO_TEST(ThreeDimensionalResizeUInt8) {
  const int input_dims[] = {4, 1, 2, 2, 2};
  const uint8 input_data[] = {3, 4, 6, 10,     //
                              10, 12, 14, 16,  //
                             };
  const int32 expected_size_data[] = {3, 3};
  const uint8 expected_output_data[] = {3, 4, 3, 4, 6, 10,       //
                                        3, 4, 3, 4, 6, 10,       //
                                        10, 12, 10, 12, 14, 16,  //
                                     };
  const int output_dims[] = {4, 1, 3, 3, 2};
  uint8 output_data[18];

  tflite::testing::TestResizeNearestNeighbor<uint8>(input_dims, input_data,
    expected_size_data, expected_output_data, output_dims, output_data);
}
TF_LITE_MICRO_TEST(ThreeDimensionalResizeInt8) {
  const int input_dims[] = {4, 1, 2, 2, 2};
  const int8 input_data[] = {3, 4, -6, 10,    //
                             10, 12, -14, 16,  //
                            };
  const int32 expected_size_data[] = {3, 3};
  const int8 expected_output_data[] = {3, 4, 3, 4, -6, 10,       //
                                        3, 4, 3, 4, -6, 10,       //
                                        10, 12, 10, 12, -14, 16,  //
                                     };
  const int output_dims[] = {4, 1, 3, 3, 2};
  int8 output_data[18];

  tflite::testing::TestResizeNearestNeighbor<int8>(input_dims, input_data,
    expected_size_data, expected_output_data, output_dims, output_data);
}

TF_LITE_MICRO_TESTS_END
