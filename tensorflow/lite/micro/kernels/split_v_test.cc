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
#include "tensorflow/lite/micro/debug_log.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {

template <int N>
struct OutputTensors {
  float* data[N];
  int* dims[N];
  float* expected_output_data[N];
};
template <int N>
void TestSplitVFloat(const int* input_dims_data, const float* input_data,
                     const int* axis_dims_data, const int32_t* axis_data,
                     const int* split_dims_data, const int32_t* split_data,
                     const OutputTensors<N>& output_tensors) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* axis_dims = IntArrayFromInts(axis_dims_data);
  TfLiteIntArray* split_dims = IntArrayFromInts(split_dims_data);
  TfLiteIntArray* output_dims[N];
  for (int i = 0; i < N; i++)
    output_dims[i] = IntArrayFromInts(output_tensors.dims[i]);

  // Place a unique value in the uninitialized output buffer.
  for (int i = 0; i < N; i++) {
    int dim_count = ElementCount(*output_dims[i]);
    for (int j = 0; j < dim_count; j++) {
      (output_tensors.data[i])[j] = 23;
    }
  }
  constexpr int input_size = 1;
  constexpr int axis_size = 1;
  constexpr int split_size = 1;
  constexpr int output_size = N;

  constexpr int tensors_size =
      input_size + output_size + axis_size + split_size;

  // first input tensor is data
  // second is size_splits
  // third is axis
  // then come outputs

  TfLiteTensor tensors[tensors_size];
  tensors[0] = CreateTensor(input_data, input_dims);
  tensors[1] = CreateTensor(split_data, split_dims);
  tensors[2] = CreateTensor(axis_data, axis_dims);

  // add output tensors
  for (int i = 0; i < N; i++)
    tensors[3 + i] = CreateTensor(output_tensors.data[i], output_dims[i]);

  tensors[2].allocation_type = kTfLiteMmapRo;
  tensors[1].allocation_type = kTfLiteMmapRo;

  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[N + 1];
  outputs_array_data[0] = N;
  for (int i = 0; i < N; i++) outputs_array_data[i + 1] = i + 3;
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration =
      tflite::ops::micro::Register_SPLIT_V();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < N; i++) {
    int dim_count = ElementCount(*output_dims[i]);
    for (int j = 0; j < dim_count; j++) {
      TF_LITE_MICRO_EXPECT_NEAR((output_tensors.expected_output_data[i])[j],
                                (output_tensors.data[i])[j], 1e-5f);
    }
  }
}

}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SPLIT_V_ThreeOutputs) {
  constexpr int output1_dims_count = 3;
  constexpr int output2_dims_count = 3;
  constexpr int output3_dims_count = 6;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  float output3_data[output3_dims_count];
  int input_shape[] = {2, 4, 3};
  float input_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int axis_shape[] = {1, 1};
  int32_t axis_values[] = {0};
  int split_shape[] = {1, 3};
  int32_t split_values[] = {1, 1, 2};
  int output1_shape[] = {2, 1, 3};
  float output1_values[] = {1, 2, 3};
  int output2_shape[] = {2, 1, 3};
  float output2_values[] = {4, 5, 6};
  int output3_shape[] = {2, 2, 3};
  float output3_values[] = {7, 8, 9, 10, 11, 12};

  tflite::testing::OutputTensors<3> output_tensors;
  output_tensors.data[0] = output1_data;
  output_tensors.data[1] = output2_data;
  output_tensors.data[2] = output3_data;

  output_tensors.dims[0] = output1_shape;
  output_tensors.dims[1] = output2_shape;
  output_tensors.dims[2] = output3_shape;

  output_tensors.expected_output_data[0] = output1_values;
  output_tensors.expected_output_data[1] = output2_values;
  output_tensors.expected_output_data[2] = output3_values;

  tflite::testing::TestSplitVFloat(input_shape, input_values, axis_shape,
                                   axis_values, split_shape, split_values,
                                   output_tensors);
}

TF_LITE_MICRO_TEST(SPLIT_V_FourDimensionalFloatAxis0) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];

  int input_shape[] = {4, 2, 2, 2, 2};
  float input_values[] = {1, 2,  3,  4,  5,  6,  7,  8,
                          9, 10, 11, 12, 13, 14, 15, 16};
  int axis_shape[] = {1, 1};
  int32_t axis_values[] = {0};
  int split_shape[] = {1, 2};
  int32_t split_values[] = {1, 1};
  int output1_shape[] = {4, 1, 2, 2, 2};
  float output1_values[] = {1, 2, 3, 4, 5, 6, 7, 8};
  int output2_shape[] = {4, 1, 2, 2, 2};
  float output2_values[] = {9, 10, 11, 12, 13, 14, 15, 16};

  tflite::testing::OutputTensors<2> output_tensors;

  output_tensors.data[0] = output1_data;
  output_tensors.data[1] = output2_data;

  output_tensors.dims[0] = output1_shape;
  output_tensors.dims[1] = output2_shape;

  output_tensors.expected_output_data[0] = output1_values;
  output_tensors.expected_output_data[1] = output2_values;

  tflite::testing::TestSplitVFloat(input_shape, input_values, axis_shape,
                                   axis_values, split_shape, split_values,
                                   output_tensors);
}

TF_LITE_MICRO_TEST(SPLIT_V_FourDimensionalFloatAxis1) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];

  int input_shape[] = {4, 2, 2, 2, 2};
  float input_values[] = {1, 2,  3,  4,  5,  6,  7,  8,
                          9, 10, 11, 12, 13, 14, 15, 16};
  int axis_shape[] = {1, 1};
  int32_t axis_values[] = {1};
  int split_shape[] = {1, 2};
  int32_t split_values[] = {1, 1};
  int output1_shape[] = {4, 2, 1, 2, 2};
  float output1_values[] = {1, 2, 3, 4, 9, 10, 11, 12};
  int output2_shape[] = {4, 2, 1, 2, 2};
  float output2_values[] = {5, 6, 7, 8, 13, 14, 15, 16};

  tflite::testing::OutputTensors<2> output_tensors;

  output_tensors.data[0] = output1_data;
  output_tensors.data[1] = output2_data;

  output_tensors.dims[0] = output1_shape;
  output_tensors.dims[1] = output2_shape;

  output_tensors.expected_output_data[0] = output1_values;
  output_tensors.expected_output_data[1] = output2_values;

  tflite::testing::TestSplitVFloat(input_shape, input_values, axis_shape,
                                   axis_values, split_shape, split_values,
                                   output_tensors);
}

TF_LITE_MICRO_TEST(SPLIT_VFourDimensionalFloatAxis2) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];

  int input_shape[] = {4, 2, 2, 2, 2};
  float input_values[] = {1, 2,  3,  4,  5,  6,  7,  8,
                          9, 10, 11, 12, 13, 14, 15, 16};
  int axis_shape[] = {1, 1};
  int32_t axis_values[] = {2};
  int split_shape[] = {1, 2};
  int32_t split_values[] = {1, 1};
  int output1_shape[] = {4, 2, 2, 1, 2};
  float output1_values[] = {1, 2, 5, 6, 9, 10, 13, 14};
  int output2_shape[] = {4, 2, 2, 1, 2};
  float output2_values[] = {3, 4, 7, 8, 11, 12, 15, 16};

  tflite::testing::OutputTensors<2> output_tensors;

  output_tensors.data[0] = output1_data;
  output_tensors.data[1] = output2_data;

  output_tensors.dims[0] = output1_shape;
  output_tensors.dims[1] = output2_shape;

  output_tensors.expected_output_data[0] = output1_values;
  output_tensors.expected_output_data[1] = output2_values;

  tflite::testing::TestSplitVFloat(input_shape, input_values, axis_shape,
                                   axis_values, split_shape, split_values,
                                   output_tensors);
}

TF_LITE_MICRO_TEST(SPLIT_V_FourDimensionalFloatAxis3) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  int input_shape[] = {4, 2, 2, 2, 2};
  float input_values[] = {1, 2,  3,  4,  5,  6,  7,  8,
                          9, 10, 11, 12, 13, 14, 15, 16};
  int axis_shape[] = {1, 1};
  int32_t axis_values[] = {3};
  int split_shape[] = {1, 2};
  int32_t split_values[] = {1, 1};
  int output1_shape[] = {4, 2, 2, 2, 1};
  float output1_values[] = {1, 3, 5, 7, 9, 11, 13, 15};
  int output2_shape[] = {4, 2, 2, 2, 1};
  float output2_values[] = {2, 4, 6, 8, 10, 12, 14, 16};

  tflite::testing::OutputTensors<2> output_tensors;

  output_tensors.data[0] = output1_data;
  output_tensors.data[1] = output2_data;

  output_tensors.dims[0] = output1_shape;
  output_tensors.dims[1] = output2_shape;

  output_tensors.expected_output_data[0] = output1_values;
  output_tensors.expected_output_data[1] = output2_values;

  tflite::testing::TestSplitVFloat(input_shape, input_values, axis_shape,
                                   axis_values, split_shape, split_values,
                                   output_tensors);
}

TF_LITE_MICRO_TEST(SPLIT_V_FourDimensionalFloatNegativeAxis) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];

  int input_shape[] = {4, 2, 2, 2, 2};
  float input_values[] = {1, 2,  3,  4,  5,  6,  7,  8,
                          9, 10, 11, 12, 13, 14, 15, 16};
  int axis_shape[] = {1, 1};
  int32_t axis_values[] = {-4};
  int split_shape[] = {1, 2};
  int32_t split_values[] = {1, 1};
  int output1_shape[] = {4, 1, 2, 2, 2};
  float output1_values[] = {1, 2, 3, 4, 5, 6, 7, 8};
  int output2_shape[] = {4, 1, 2, 2, 2};
  float output2_values[] = {9, 10, 11, 12, 13, 14, 15, 16};

  tflite::testing::OutputTensors<2> output_tensors;

  output_tensors.data[0] = output1_data;
  output_tensors.data[1] = output2_data;

  output_tensors.dims[0] = output1_shape;
  output_tensors.dims[1] = output2_shape;

  output_tensors.expected_output_data[0] = output1_values;
  output_tensors.expected_output_data[1] = output2_values;

  tflite::testing::TestSplitVFloat(input_shape, input_values, axis_shape,
                                   axis_values, split_shape, split_values,
                                   output_tensors);
}

TF_LITE_MICRO_TEST(SPLIT_V_OneDimensionalFloatAxis0) {
  constexpr int output1_dims_count = 1;
  constexpr int output2_dims_count = 1;
  constexpr int output3_dims_count = 1;
  constexpr int output4_dims_count = 1;
  constexpr int output5_dims_count = 1;
  constexpr int output6_dims_count = 1;
  constexpr int output7_dims_count = 1;
  constexpr int output8_dims_count = 1;

  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  float output3_data[output3_dims_count];
  float output4_data[output4_dims_count];
  float output5_data[output5_dims_count];
  float output6_data[output6_dims_count];
  float output7_data[output7_dims_count];
  float output8_data[output8_dims_count];
  int input_shape[] = {1, 8};
  float input_values[] = {1, 2, 3, 4, 5, 6, 7, 8};
  int axis_shape[] = {1, 1};
  int32_t axis_value[] = {0};
  int split_size_shape[] = {1, 8};
  int32_t split[] = {1, 1, 1, 1, 1, 1, 1, 1};
  int output1_shape[] = {1, 1};
  float output1_values[] = {1};
  int output2_shape[] = {1, 1};
  float output2_values[] = {2};

  int output3_shape[] = {1, 1};
  float output3_values[] = {3};
  int output4_shape[] = {1, 1};
  float output4_values[] = {4};

  int output5_shape[] = {1, 1};
  float output5_values[] = {5};
  int output6_shape[] = {1, 1};
  float output6_values[] = {6};

  int output7_shape[] = {1, 1};
  float output7_values[] = {7};
  int output8_shape[] = {1, 1};
  float output8_values[] = {8};

  tflite::testing::OutputTensors<8> output_tensors;

  output_tensors.data[0] = output1_data;
  output_tensors.data[1] = output2_data;
  output_tensors.data[2] = output3_data;
  output_tensors.data[3] = output4_data;
  output_tensors.data[4] = output5_data;
  output_tensors.data[5] = output6_data;
  output_tensors.data[6] = output7_data;
  output_tensors.data[7] = output8_data;

  output_tensors.dims[0] = output1_shape;
  output_tensors.dims[1] = output2_shape;
  output_tensors.dims[2] = output3_shape;
  output_tensors.dims[3] = output4_shape;
  output_tensors.dims[4] = output5_shape;
  output_tensors.dims[5] = output6_shape;
  output_tensors.dims[6] = output7_shape;
  output_tensors.dims[7] = output8_shape;

  output_tensors.expected_output_data[0] = output1_values;
  output_tensors.expected_output_data[1] = output2_values;
  output_tensors.expected_output_data[2] = output3_values;
  output_tensors.expected_output_data[3] = output4_values;
  output_tensors.expected_output_data[4] = output5_values;
  output_tensors.expected_output_data[5] = output6_values;
  output_tensors.expected_output_data[6] = output7_values;
  output_tensors.expected_output_data[7] = output8_values;

  tflite::testing::TestSplitVFloat(input_shape, input_values, axis_shape,
                                   axis_value, split_size_shape, split,
                                   output_tensors);
}

TF_LITE_MICRO_TEST(SPLIT_V_OneDimensionalFloatTest2) {
  constexpr int output1_dims_count = 1;
  constexpr int output2_dims_count = 1;
  constexpr int output3_dims_count = 1;
  constexpr int output4_dims_count = 1;
  constexpr int output5_dims_count = 1;
  constexpr int output6_dims_count = 1;
  constexpr int output7_dims_count = 2;

  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  float output3_data[output3_dims_count];
  float output4_data[output4_dims_count];
  float output5_data[output5_dims_count];
  float output6_data[output6_dims_count];
  float output7_data[output7_dims_count];

  int input_shape[] = {1, 8};
  float input_values[] = {1, 2, 3, 4, 5, 6, 7, 8};
  int axis_shape[] = {1, 1};
  int32_t axis_value[] = {0};
  int split_size_shape[] = {1, 8};
  int32_t split[] = {1, 1, 1, 1, 1, 1, 2, -1};
  int output1_shape[] = {1, 1};
  float output1_values[] = {1};
  int output2_shape[] = {1, 1};
  float output2_values[] = {2};

  int output3_shape[] = {1, 1};
  float output3_values[] = {3};
  int output4_shape[] = {1, 1};
  float output4_values[] = {4};

  int output5_shape[] = {1, 1};
  float output5_values[] = {5};
  int output6_shape[] = {1, 1};
  float output6_values[] = {6};

  int output7_shape[] = {1, 2};
  float output7_values[] = {7, 8};
  int output8_shape[] = {1, 0};
  float output8_values[1] = {};

  tflite::testing::OutputTensors<8> output_tensors;

  output_tensors.data[0] = output1_data;
  output_tensors.data[1] = output2_data;
  output_tensors.data[2] = output3_data;
  output_tensors.data[3] = output4_data;
  output_tensors.data[4] = output5_data;
  output_tensors.data[5] = output6_data;
  output_tensors.data[6] = output7_data;
  output_tensors.data[7] = NULL;

  output_tensors.dims[0] = output1_shape;
  output_tensors.dims[1] = output2_shape;
  output_tensors.dims[2] = output3_shape;
  output_tensors.dims[3] = output4_shape;
  output_tensors.dims[4] = output5_shape;
  output_tensors.dims[5] = output6_shape;
  output_tensors.dims[6] = output7_shape;
  output_tensors.dims[7] = output8_shape;

  output_tensors.expected_output_data[0] = output1_values;
  output_tensors.expected_output_data[1] = output2_values;
  output_tensors.expected_output_data[2] = output3_values;
  output_tensors.expected_output_data[3] = output4_values;
  output_tensors.expected_output_data[4] = output5_values;
  output_tensors.expected_output_data[5] = output6_values;
  output_tensors.expected_output_data[6] = output7_values;
  output_tensors.expected_output_data[7] = output8_values;

  tflite::testing::TestSplitVFloat(input_shape, input_values, axis_shape,
                                   axis_value, split_size_shape, split,
                                   output_tensors);
}

TF_LITE_MICRO_TESTS_END
