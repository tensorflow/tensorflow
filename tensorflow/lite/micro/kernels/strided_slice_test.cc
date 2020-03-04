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
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

template <typename input_type = int32_t,
          TfLiteType tensor_input_type = kTfLiteInt32>
inline TfLiteTensor CreateTensor(const input_type* data, TfLiteIntArray* dims,
                                 const char* name, bool is_variable = false) {
  TfLiteTensor result;
  result.type = tensor_input_type;
  result.data.raw = reinterpret_cast<char*>(const_cast<input_type*>(data));
  result.dims = dims;
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(input_type);
  result.allocation = nullptr;
  result.name = name;
  result.is_variable = is_variable;
  return result;
}

template <typename input_type = int32_t,
          TfLiteType tensor_input_type = kTfLiteInt32>
inline TfLiteTensor CreateTensor(std::initializer_list<input_type> data,
                                 TfLiteIntArray* dims, const char* name,
                                 bool is_variable = false) {
  return CreateTensor<input_type, tensor_input_type>(data.begin(), dims, name,
                                                     is_variable);
}

template <typename input_type = float,
          TfLiteType tensor_input_type = kTfLiteFloat32>
void TestStrideSlide(std::initializer_list<int> input_shape,
                     std::initializer_list<int> begin_shape,
                     std::initializer_list<int> end_shape,
                     std::initializer_list<int> strides_shape, int begin_mask,
                     int end_mask, int ellipsis_mask, int new_axis_mask,
                     int shrink_axis_mask,
                     std::initializer_list<input_type> input_data,
                     std::initializer_list<int32_t> begin_data,
                     std::initializer_list<int32_t> end_data,
                     std::initializer_list<int32_t> strides_data,
                     std::initializer_list<int> output_shape,
                     input_type* output_data,
                     std::initializer_list<int> expected_output,
                     bool expect_prepare_err, bool expect_invoke_err,
                     int num_invoke = 1) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_shape);
  TfLiteIntArray* begin_dims = IntArrayFromInitializer(begin_shape);
  TfLiteIntArray* end_dims = IntArrayFromInitializer(end_shape);
  TfLiteIntArray* strides_dims = IntArrayFromInitializer(strides_shape);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_shape);
  constexpr int inputs_size = 4;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor<input_type, tensor_input_type>(input_data, input_dims,
                                                  "input_tensor"),
      CreateTensor<int32_t, kTfLiteInt32>(begin_data, begin_dims,
                                          "begin_tensor"),
      CreateTensor<int32_t, kTfLiteInt32>(end_data, end_dims, "end_tensor"),
      CreateTensor<int32_t, kTfLiteInt32>(strides_data, strides_dims,
                                          "stride_tensor"),
      CreateTensor<input_type, tensor_input_type>(output_data, output_dims,
                                                  "output_tensor"),
  };
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_STRIDED_SLICE, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);
  TfLiteStridedSliceParams builtin_data = {begin_mask, end_mask, ellipsis_mask,
                                           new_axis_mask, shrink_axis_mask};
  const char* init_data = reinterpret_cast<const char*>(&builtin_data);
  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }

  int inputs_array_data[] = {4, 0, 1, 2, 3};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 4};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = nullptr;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;
  if (registration->prepare) {
    if (expect_prepare_err) {
      TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                              registration->prepare(&context, &node));
      return;
    }
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  if (expect_invoke_err) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                            registration->invoke(&context, &node));
    return;
  }
  for (int i = 0; i < num_invoke; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  }
  if (registration->free) {
    registration->free(&context, user_data);
  }
  auto* output_tensor = &context.tensors[node.outputs->data[0]];
  for (int i = 0; i < expected_output.size(); ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output.begin()[i], output_data[i],
                              1e-5f);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN
using tflite::testing::TestStrideSlide;

TF_LITE_MICRO_TEST(UnsupportedInputSize) {
  float output_data[4];
  TestStrideSlide<float>({5, 2, 2, 2, 2, 2},  // input_shape
                         {1, 5},              //  begin_shape
                         {1, 5},              // end_shape
                         {1, 5},              //  strides_shape
                         0,                   // begin_mask
                         0,                   // end_mask
                         0,                   // ellipsis_mask
                         0,                   // new_axis_mask
                         0,                   // shrink_axis_mask
                         {},                  // input_data
                         {},                  // begin_data
                         {},                  // end_data
                         {},                  // strides_data
                         {0},                 // output_shape
                         output_data,         // output_data
                         {},                  // expected_output
                         true,                // expect_prepare_err
                         false                // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In1D) {
  float output_data[4];
  TestStrideSlide<float>({1, 4},        // input_shape
                         {1, 1},        //  begin_shape
                         {1, 1},        // end_shape
                         {1, 1},        //  strides_shape
                         0,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         0,             // shrink_axis_mask
                         {1, 2, 3, 4},  // input_data
                         {1},           // begin_data
                         {3},           // end_data
                         {1},           // strides_data
                         {1, 2},        // output_shape
                         output_data,   // output_data
                         {2, 3},        // expected_output
                         false,         // expect_prepare_err
                         false          // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In1D_EmptyOutput) {
  float output_data[4];
  TestStrideSlide<float>({1, 4},        // input_shape
                         {1, 1},        //  begin_shape
                         {1, 1},        // end_shape
                         {1, 1},        //  strides_shape
                         0,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         0,             // shrink_axis_mask
                         {1, 2, 3, 4},  // input_data
                         {10},          // begin_data
                         {3},           // end_data
                         {1},           // strides_data
                         {1, 0},        // output_shape
                         output_data,   // output_data
                         {},            // expected_output
                         false,         // expect_prepare_err
                         false          // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In1D_NegativeBegin) {
  float output_data[4];
  TestStrideSlide<float>({1, 4},        // input_shape
                         {1, 1},        //  begin_shape
                         {1, 1},        // end_shape
                         {1, 1},        //  strides_shape
                         0,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         0,             // shrink_axis_mask
                         {1, 2, 3, 4},  // input_data
                         {-3},          // begin_data
                         {3},           // end_data
                         {1},           // strides_data
                         {1, 2},        // output_shape
                         output_data,   // output_data
                         {2, 3},        // expected_output
                         false,         // expect_prepare_err
                         false          // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In1D_OutOfRangeBegin) {
  float output_data[4];
  TestStrideSlide<float>({1, 4},        // input_shape
                         {1, 1},        //  begin_shape
                         {1, 1},        // end_shape
                         {1, 1},        //  strides_shape
                         0,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         0,             // shrink_axis_mask
                         {1, 2, 3, 4},  // input_data
                         {-5},          // begin_data
                         {3},           // end_data
                         {1},           // strides_data
                         {1, 3},        // output_shape
                         output_data,   // output_data
                         {1, 2, 3},     // expected_output
                         false,         // expect_prepare_err
                         false          // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In1D_NegativeEnd) {
  float output_data[4];
  TestStrideSlide<float>({1, 4},        // input_shape
                         {1, 1},        //  begin_shape
                         {1, 1},        // end_shape
                         {1, 1},        //  strides_shape
                         0,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         0,             // shrink_axis_mask
                         {1, 2, 3, 4},  // input_data
                         {1},           // begin_data
                         {-2},          // end_data
                         {1},           // strides_data
                         {1, 1},        // output_shape
                         output_data,   // output_data
                         {2},           // expected_output
                         false,         // expect_prepare_err
                         false          // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In1D_OutOfRangeEnd) {
  float output_data[4];
  TestStrideSlide<float>({1, 4},        // input_shape
                         {1, 1},        //  begin_shape
                         {1, 1},        // end_shape
                         {1, 1},        //  strides_shape
                         0,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         0,             // shrink_axis_mask
                         {1, 2, 3, 4},  // input_data
                         {-3},          // begin_data
                         {5},           // end_data
                         {1},           // strides_data
                         {1, 3},        // output_shape
                         output_data,   // output_data
                         {2, 3, 4},     // expected_output
                         false,         // expect_prepare_err
                         false          // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In1D_BeginMask) {
  float output_data[4];
  TestStrideSlide<float>({1, 4},        // input_shape
                         {1, 1},        //  begin_shape
                         {1, 1},        // end_shape
                         {1, 1},        //  strides_shape
                         1,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         0,             // shrink_axis_mask
                         {1, 2, 3, 4},  // input_data
                         {1},           // begin_data
                         {3},           // end_data
                         {1},           // strides_data
                         {1, 3},        // output_shape
                         output_data,   // output_data
                         {1, 2, 3},     // expected_output
                         false,         // expect_prepare_err
                         false          // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In1D_NegativeBeginNegativeStride) {
  float output_data[4];
  TestStrideSlide<float>({1, 4},        // input_shape
                         {1, 1},        //  begin_shape
                         {1, 1},        // end_shape
                         {1, 1},        //  strides_shape
                         0,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         0,             // shrink_axis_mask
                         {1, 2, 3, 4},  // input_data
                         {-2},          // begin_data
                         {-3},          // end_data
                         {-1},          // strides_data
                         {1, 1},        // output_shape
                         output_data,   // output_data
                         {3},           // expected_output
                         false,         // expect_prepare_err
                         false          // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In1D_OutOfRangeBeginNegativeStride) {
  float output_data[4];
  TestStrideSlide<float>({1, 4},        // input_shape
                         {1, 1},        //  begin_shape
                         {1, 1},        // end_shape
                         {1, 1},        //  strides_shape
                         0,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         0,             // shrink_axis_mask
                         {1, 2, 3, 4},  // input_data
                         {5},           // begin_data
                         {2},           // end_data
                         {-1},          // strides_data
                         {1, 1},        // output_shape
                         output_data,   // output_data
                         {4},           // expected_output
                         false,         // expect_prepare_err
                         false          // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In1D_NegativeEndNegativeStride) {
  float output_data[4];
  TestStrideSlide<float>({1, 4},        // input_shape
                         {1, 1},        //  begin_shape
                         {1, 1},        // end_shape
                         {1, 1},        //  strides_shape
                         0,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         0,             // shrink_axis_mask
                         {1, 2, 3, 4},  // input_data
                         {2},           // begin_data
                         {-4},          // end_data
                         {-1},          // strides_data
                         {1, 2},        // output_shape
                         output_data,   // output_data
                         {3, 2},        // expected_output
                         false,         // expect_prepare_err
                         false          // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In1D_OutOfRangeEndNegativeStride) {
  float output_data[4];
  TestStrideSlide<float>({1, 4},        // input_shape
                         {1, 1},        //  begin_shape
                         {1, 1},        // end_shape
                         {1, 1},        //  strides_shape
                         0,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         0,             // shrink_axis_mask
                         {1, 2, 3, 4},  // input_data
                         {-3},          // begin_data
                         {-5},          // end_data
                         {-1},          // strides_data
                         {1, 2},        // output_shape
                         output_data,   // output_data
                         {2, 1},        // expected_output
                         false,         // expect_prepare_err
                         false          // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In1D_EndMask) {
  float output_data[4];
  TestStrideSlide<float>({1, 4},        // input_shape
                         {1, 1},        //  begin_shape
                         {1, 1},        // end_shape
                         {1, 1},        //  strides_shape
                         0,             // begin_mask
                         1,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         0,             // shrink_axis_mask
                         {1, 2, 3, 4},  // input_data
                         {1},           // begin_data
                         {3},           // end_data
                         {1},           // strides_data
                         {1, 3},        // output_shape
                         output_data,   // output_data
                         {2, 3, 4},     // expected_output
                         false,         // expect_prepare_err
                         false          // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In1D_NegStride) {
  float output_data[4];
  TestStrideSlide<float>({1, 3},       // input_shape
                         {1, 1},       //  begin_shape
                         {1, 1},       // end_shape
                         {1, 1},       //  strides_shape
                         0,            // begin_mask
                         0,            // end_mask
                         0,            // ellipsis_mask
                         0,            // new_axis_mask
                         0,            // shrink_axis_mask
                         {1, 2, 3},    // input_data
                         {-1},         // begin_data
                         {-4},         // end_data
                         {-1},         // strides_data
                         {1, 3},       // output_shape
                         output_data,  // output_data
                         {3, 2, 1},    // expected_output
                         false,        // expect_prepare_err
                         false         // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In1D_EvenLenStride2) {
  float output_data[4];
  TestStrideSlide<float>({1, 2},       // input_shape
                         {1, 1},       //  begin_shape
                         {1, 1},       // end_shape
                         {1, 1},       //  strides_shape
                         0,            // begin_mask
                         0,            // end_mask
                         0,            // ellipsis_mask
                         0,            // new_axis_mask
                         0,            // shrink_axis_mask
                         {1, 2},       // input_data
                         {0},          // begin_data
                         {4},          // end_data
                         {2},          // strides_data
                         {1, 1},       // output_shape
                         output_data,  // output_data
                         {1},          // expected_output
                         false,        // expect_prepare_err
                         false         // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In1D_OddLenStride2) {
  float output_data[4];
  TestStrideSlide<float>({1, 3},       // input_shape
                         {1, 1},       //  begin_shape
                         {1, 1},       // end_shape
                         {1, 1},       //  strides_shape
                         0,            // begin_mask
                         0,            // end_mask
                         0,            // ellipsis_mask
                         0,            // new_axis_mask
                         0,            // shrink_axis_mask
                         {1, 2, 3},    // input_data
                         {0},          // begin_data
                         {3},          // end_data
                         {2},          // strides_data
                         {1, 2},       // output_shape
                         output_data,  // output_data
                         {1, 3},       // expected_output
                         false,        // expect_prepare_err
                         false         // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In2D_Identity) {
  float output_data[8];
  TestStrideSlide<float>({2, 2, 3},           // input_shape
                         {1, 2},              //  begin_shape
                         {1, 2},              // end_shape
                         {1, 2},              //  strides_shape
                         0,                   // begin_mask
                         0,                   // end_mask
                         0,                   // ellipsis_mask
                         0,                   // new_axis_mask
                         0,                   // shrink_axis_mask
                         {1, 2, 3, 4, 5, 6},  // input_data
                         {0, 0},              // begin_data
                         {2, 3},              // end_data
                         {1, 1},              // strides_data
                         {2, 2, 3},           // output_shape
                         output_data,         // output_data
                         {1, 2, 3, 4, 5, 6},  // expected_output
                         false,               // expect_prepare_err
                         false                // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In2D) {
  float output_data[8];
  TestStrideSlide<float>({2, 2, 3},           // input_shape
                         {1, 2},              //  begin_shape
                         {1, 2},              // end_shape
                         {1, 2},              //  strides_shape
                         0,                   // begin_mask
                         0,                   // end_mask
                         0,                   // ellipsis_mask
                         0,                   // new_axis_mask
                         0,                   // shrink_axis_mask
                         {1, 2, 3, 4, 5, 6},  // input_data
                         {1, 0},              // begin_data
                         {2, 2},              // end_data
                         {1, 1},              // strides_data
                         {2, 1, 2},           // output_shape
                         output_data,         // output_data
                         {4, 5},              // expected_output
                         false,               // expect_prepare_err
                         false                // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In2D_Stride2) {
  float output_data[8];
  TestStrideSlide<float>({2, 2, 3},           // input_shape
                         {1, 2},              //  begin_shape
                         {1, 2},              // end_shape
                         {1, 2},              //  strides_shape
                         0,                   // begin_mask
                         0,                   // end_mask
                         0,                   // ellipsis_mask
                         0,                   // new_axis_mask
                         0,                   // shrink_axis_mask
                         {1, 2, 3, 4, 5, 6},  // input_data
                         {0, 0},              // begin_data
                         {2, 3},              // end_data
                         {2, 2},              // strides_data
                         {2, 1, 2},           // output_shape
                         output_data,         // output_data
                         {1, 3},              // expected_output
                         false,               // expect_prepare_err
                         false                // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In2D_NegStride) {
  float output_data[8];
  TestStrideSlide<float>({2, 2, 3},           // input_shape
                         {1, 2},              //  begin_shape
                         {1, 2},              // end_shape
                         {1, 2},              //  strides_shape
                         0,                   // begin_mask
                         0,                   // end_mask
                         0,                   // ellipsis_mask
                         0,                   // new_axis_mask
                         0,                   // shrink_axis_mask
                         {1, 2, 3, 4, 5, 6},  // input_data
                         {1, -1},             // begin_data
                         {2, -4},             // end_data
                         {2, -1},             // strides_data
                         {2, 1, 3},           // output_shape
                         output_data,         // output_data
                         {6, 5, 4},           // expected_output
                         false,               // expect_prepare_err
                         false                // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In2D_BeginMask) {
  float output_data[8];
  TestStrideSlide<float>({2, 2, 3},           // input_shape
                         {1, 2},              //  begin_shape
                         {1, 2},              // end_shape
                         {1, 2},              //  strides_shape
                         1,                   // begin_mask
                         0,                   // end_mask
                         0,                   // ellipsis_mask
                         0,                   // new_axis_mask
                         0,                   // shrink_axis_mask
                         {1, 2, 3, 4, 5, 6},  // input_data
                         {1, 0},              // begin_data
                         {2, 2},              // end_data
                         {1, 1},              // strides_data
                         {2, 2, 2},           // output_shape
                         output_data,         // output_data
                         {1, 2, 4, 5},        // expected_output
                         false,               // expect_prepare_err
                         false                // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In2D_EndMask) {
  float output_data[8];
  TestStrideSlide<float>({2, 2, 3},           // input_shape
                         {1, 2},              //  begin_shape
                         {1, 2},              // end_shape
                         {1, 2},              //  strides_shape
                         0,                   // begin_mask
                         2,                   // end_mask
                         0,                   // ellipsis_mask
                         0,                   // new_axis_mask
                         0,                   // shrink_axis_mask
                         {1, 2, 3, 4, 5, 6},  // input_data
                         {1, 0},              // begin_data
                         {2, 2},              // end_data
                         {1, 1},              // strides_data
                         {2, 1, 3},           // output_shape
                         output_data,         // output_data
                         {4, 5, 6},           // expected_output
                         false,               // expect_prepare_err
                         false                // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In2D_NegStrideBeginMask) {
  float output_data[8];
  TestStrideSlide<float>({2, 2, 3},           // input_shape
                         {1, 2},              //  begin_shape
                         {1, 2},              // end_shape
                         {1, 2},              //  strides_shape
                         2,                   // begin_mask
                         0,                   // end_mask
                         0,                   // ellipsis_mask
                         0,                   // new_axis_mask
                         0,                   // shrink_axis_mask
                         {1, 2, 3, 4, 5, 6},  // input_data
                         {1, -2},             // begin_data
                         {2, -4},             // end_data
                         {1, -1},             // strides_data
                         {2, 1, 3},           // output_shape
                         output_data,         // output_data
                         {6, 5, 4},           // expected_output
                         false,               // expect_prepare_err
                         false                // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In2D_NegStrideEndMask) {
  float output_data[8];
  TestStrideSlide<float>({2, 2, 3},           // input_shape
                         {1, 2},              //  begin_shape
                         {1, 2},              // end_shape
                         {1, 2},              //  strides_shape
                         0,                   // begin_mask
                         2,                   // end_mask
                         0,                   // ellipsis_mask
                         0,                   // new_axis_mask
                         0,                   // shrink_axis_mask
                         {1, 2, 3, 4, 5, 6},  // input_data
                         {1, -2},             // begin_data
                         {2, -3},             // end_data
                         {1, -1},             // strides_data
                         {2, 1, 2},           // output_shape
                         output_data,         // output_data
                         {5, 4},              // expected_output
                         false,               // expect_prepare_err
                         false                // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In3D_Identity) {
  float output_data[16];
  TestStrideSlide<float>(
      {3, 2, 3, 2},                             // input_shape
      {1, 3},                                   //  begin_shape
      {1, 3},                                   // end_shape
      {1, 3},                                   //  strides_shape
      0,                                        // begin_mask
      0,                                        // end_mask
      0,                                        // ellipsis_mask
      0,                                        // new_axis_mask
      0,                                        // shrink_axis_mask
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},  // input_data
      {0, 0, 0},                                // begin_data
      {2, 3, 2},                                // end_data
      {1, 1, 1},                                // strides_data
      {3, 2, 3, 2},                             // output_shape
      output_data,                              // output_data
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},  // expected_output
      false,                                    // expect_prepare_err
      false                                     // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In3D_NegStride) {
  float output_data[16];
  TestStrideSlide<float>(
      {3, 2, 3, 2},                             // input_shape
      {1, 3},                                   //  begin_shape
      {1, 3},                                   // end_shape
      {1, 3},                                   //  strides_shape
      0,                                        // begin_mask
      0,                                        // end_mask
      0,                                        // ellipsis_mask
      0,                                        // new_axis_mask
      0,                                        // shrink_axis_mask
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},  // input_data
      {-1, -1, -1},                             // begin_data
      {-3, -4, -3},                             // end_data
      {-1, -1, -1},                             // strides_data
      {3, 2, 3, 2},                             // output_shape
      output_data,                              // output_data
      {12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},  // expected_output
      false,                                    // expect_prepare_err
      false                                     // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In3D_Strided2) {
  float output_data[16];
  TestStrideSlide<float>({3, 2, 3, 2},  // input_shape
                         {1, 3},        //  begin_shape
                         {1, 3},        // end_shape
                         {1, 3},        //  strides_shape
                         0,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         0,             // shrink_axis_mask
                         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},  // input_data
                         {0, 0, 0},                                // begin_data
                         {2, 3, 2},                                // end_data
                         {2, 2, 2},     // strides_data
                         {3, 1, 2, 1},  // output_shape
                         output_data,   // output_data
                         {1, 5},        // expected_output
                         false,         // expect_prepare_err
                         false          // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In1D_ShrinkAxisMask1) {
  float output_data[4];
  TestStrideSlide<float>({1, 4},        // input_shape
                         {1, 1},        //  begin_shape
                         {1, 1},        // end_shape
                         {1, 1},        //  strides_shape
                         0,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         1,             // shrink_axis_mask
                         {1, 2, 3, 4},  // input_data
                         {1},           // begin_data
                         {2},           // end_data
                         {1},           // strides_data
                         {0},           // output_shape
                         output_data,   // output_data
                         {2},           // expected_output
                         false,         // expect_prepare_err
                         false          // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In1D_ShrinkAxisMask1_NegativeSlice) {
  float output_data[4];
  TestStrideSlide<float>({1, 4},        // input_shape
                         {1, 1},        //  begin_shape
                         {1, 1},        // end_shape
                         {1, 1},        //  strides_shape
                         0,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         1,             // shrink_axis_mask
                         {0, 1, 2, 3},  // input_data
                         {-1},          // begin_data
                         {0},           // end_data
                         {1},           // strides_data
                         {0},           // output_shape
                         output_data,   // output_data
                         {3},           // expected_output
                         false,         // expect_prepare_err
                         false          // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In2D_ShrinkAxis3_NegativeSlice) {
  float output_data[4];
  TestStrideSlide<float>({2, 4, 1},     // input_shape
                         {1, 2},        //  begin_shape
                         {1, 2},        // end_shape
                         {1, 2},        //  strides_shape
                         0,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         3,             // shrink_axis_mask
                         {0, 1, 2, 3},  // input_data
                         {-2, -1},      // begin_data
                         {-1, 0},       // end_data
                         {1, 1},        // strides_data
                         {0},           // output_shape
                         output_data,   // output_data
                         {2},           // expected_output
                         false,         // expect_prepare_err
                         false          // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In2D_ShrinkAxis2_BeginEndAxis1_NegativeSlice) {
  float output_data[4];
  TestStrideSlide<float>({2, 4, 1},     // input_shape
                         {1, 2},        //  begin_shape
                         {1, 2},        // end_shape
                         {1, 2},        //  strides_shape
                         1,             // begin_mask
                         1,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         2,             // shrink_axis_mask
                         {0, 1, 2, 3},  // input_data
                         {0, -1},       // begin_data
                         {0, 0},        // end_data
                         {1, 1},        // strides_data
                         {1, 4},        // output_shape
                         output_data,   // output_data
                         {0, 1, 2, 3},  // expected_output
                         false,         // expect_prepare_err
                         false          // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In1D_BeginMaskShrinkAxisMask1) {
  float output_data[4];
  TestStrideSlide<float>({1, 4},        // input_shape
                         {1, 1},        //  begin_shape
                         {1, 1},        // end_shape
                         {1, 1},        //  strides_shape
                         1,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         1,             // shrink_axis_mask
                         {1, 2, 3, 4},  // input_data
                         {1},           // begin_data
                         {1},           // end_data
                         {1},           // strides_data
                         {0},           // output_shape
                         output_data,   // output_data
                         {1},           // expected_output
                         false,         // expect_prepare_err
                         false          // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In2D_ShrinkAxisMask1) {
  float output_data[6];
  TestStrideSlide<float>({2, 2, 3},           // input_shape
                         {1, 2},              //  begin_shape
                         {1, 2},              // end_shape
                         {1, 2},              //  strides_shape
                         0,                   // begin_mask
                         0,                   // end_mask
                         0,                   // ellipsis_mask
                         0,                   // new_axis_mask
                         1,                   // shrink_axis_mask
                         {1, 2, 3, 4, 5, 6},  // input_data
                         {0, 0},              // begin_data
                         {1, 3},              // end_data
                         {1, 1},              // strides_data
                         {1, 3},              // output_shape
                         output_data,         // output_data
                         {1, 2, 3},           // expected_output
                         false,               // expect_prepare_err
                         false                // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In2D_ShrinkAxisMask2) {
  float output_data[6];
  TestStrideSlide<float>({2, 2, 3},           // input_shape
                         {1, 2},              //  begin_shape
                         {1, 2},              // end_shape
                         {1, 2},              //  strides_shape
                         0,                   // begin_mask
                         0,                   // end_mask
                         0,                   // ellipsis_mask
                         0,                   // new_axis_mask
                         2,                   // shrink_axis_mask
                         {1, 2, 3, 4, 5, 6},  // input_data
                         {0, 0},              // begin_data
                         {2, 1},              // end_data
                         {1, 1},              // strides_data
                         {1, 2},              // output_shape
                         output_data,         // output_data
                         {1, 4},              // expected_output
                         false,               // expect_prepare_err
                         false                // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In2D_ShrinkAxisMask3) {
  float output_data[6];
  TestStrideSlide<float>({2, 2, 3},           // input_shape
                         {1, 2},              //  begin_shape
                         {1, 2},              // end_shape
                         {1, 2},              //  strides_shape
                         0,                   // begin_mask
                         0,                   // end_mask
                         0,                   // ellipsis_mask
                         0,                   // new_axis_mask
                         3,                   // shrink_axis_mask
                         {1, 2, 3, 4, 5, 6},  // input_data
                         {0, 0},              // begin_data
                         {1, 1},              // end_data
                         {1, 1},              // strides_data
                         {0},                 // output_shape
                         output_data,         // output_data
                         {1},                 // expected_output
                         false,               // expect_prepare_err
                         false                // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In3D_IdentityShrinkAxis1) {
  float output_data[16];
  TestStrideSlide<float>({3, 2, 3, 2},  // input_shape
                         {1, 3},        //  begin_shape
                         {1, 3},        // end_shape
                         {1, 3},        //  strides_shape
                         0,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         1,             // shrink_axis_mask
                         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},  // input_data
                         {0, 0, 0},                                // begin_data
                         {1, 3, 2},                                // end_data
                         {1, 1, 1},           // strides_data
                         {2, 3, 2},           // output_shape
                         output_data,         // output_data
                         {1, 2, 3, 4, 5, 6},  // expected_output
                         false,               // expect_prepare_err
                         false                // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In3D_IdentityShrinkAxis2) {
  float output_data[16];
  TestStrideSlide<float>({3, 2, 3, 2},  // input_shape
                         {1, 3},        //  begin_shape
                         {1, 3},        // end_shape
                         {1, 3},        //  strides_shape
                         0,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         2,             // shrink_axis_mask
                         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},  // input_data
                         {0, 0, 0},                                // begin_data
                         {2, 1, 2},                                // end_data
                         {1, 1, 1},     // strides_data
                         {2, 2, 2},     // output_shape
                         output_data,   // output_data
                         {1, 2, 7, 8},  // expected_output
                         false,         // expect_prepare_err
                         false          // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In3D_IdentityShrinkAxis3) {
  float output_data[16];
  TestStrideSlide<float>({3, 2, 3, 2},  // input_shape
                         {1, 3},        //  begin_shape
                         {1, 3},        // end_shape
                         {1, 3},        //  strides_shape
                         0,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         3,             // shrink_axis_mask
                         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},  // input_data
                         {0, 0, 0},                                // begin_data
                         {1, 1, 2},                                // end_data
                         {1, 1, 1},    // strides_data
                         {1, 2},       // output_shape
                         output_data,  // output_data
                         {1, 2},       // expected_output
                         false,        // expect_prepare_err
                         false         // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In3D_IdentityShrinkAxis4) {
  float output_data[16];
  TestStrideSlide<float>({3, 2, 3, 2},  // input_shape
                         {1, 3},        //  begin_shape
                         {1, 3},        // end_shape
                         {1, 3},        //  strides_shape
                         0,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         4,             // shrink_axis_mask
                         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},  // input_data
                         {0, 0, 0},                                // begin_data
                         {2, 3, 1},                                // end_data
                         {1, 1, 1},            // strides_data
                         {2, 2, 3},            // output_shape
                         output_data,          // output_data
                         {1, 3, 5, 7, 9, 11},  // expected_output
                         false,                // expect_prepare_err
                         false                 // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In3D_IdentityShrinkAxis5) {
  float output_data[16];
  TestStrideSlide<float>({3, 2, 3, 2},  // input_shape
                         {1, 3},        //  begin_shape
                         {1, 3},        // end_shape
                         {1, 3},        //  strides_shape
                         0,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         5,             // shrink_axis_mask
                         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},  // input_data
                         {0, 0, 0},                                // begin_data
                         {1, 3, 1},                                // end_data
                         {1, 1, 1},    // strides_data
                         {1, 3},       // output_shape
                         output_data,  // output_data
                         {1, 3, 5},    // expected_output
                         false,        // expect_prepare_err
                         false         // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In3D_IdentityShrinkAxis6) {
  float output_data[16];
  TestStrideSlide<float>({3, 2, 3, 2},  // input_shape
                         {1, 3},        //  begin_shape
                         {1, 3},        // end_shape
                         {1, 3},        //  strides_shape
                         0,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         6,             // shrink_axis_mask
                         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},  // input_data
                         {0, 0, 0},                                // begin_data
                         {2, 1, 1},                                // end_data
                         {1, 1, 1},    // strides_data
                         {1, 2},       // output_shape
                         output_data,  // output_data
                         {1, 7},       // expected_output
                         false,        // expect_prepare_err
                         false         // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In3D_IdentityShrinkAxis7) {
  float output_data[16];
  TestStrideSlide<float>({3, 2, 3, 2},  // input_shape
                         {1, 3},        //  begin_shape
                         {1, 3},        // end_shape
                         {1, 3},        //  strides_shape
                         0,             // begin_mask
                         0,             // end_mask
                         0,             // ellipsis_mask
                         0,             // new_axis_mask
                         7,             // shrink_axis_mask
                         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},  // input_data
                         {0, 0, 0},                                // begin_data
                         {1, 1, 1},                                // end_data
                         {1, 1, 1},    // strides_data
                         {0},          // output_shape
                         output_data,  // output_data
                         {1},          // expected_output
                         false,        // expect_prepare_err
                         false         // expect_invoke_err
  );
}

// This tests catches a very subtle bug that was fixed by cl/188403234.
TF_LITE_MICRO_TEST(RunTwice) {
  float output_data[6];
  TestStrideSlide<float>({2, 2, 3},           // input_shape
                         {1, 2},              //  begin_shape
                         {1, 2},              // end_shape
                         {1, 2},              //  strides_shape
                         1,                   // begin_mask
                         0,                   // end_mask
                         0,                   // ellipsis_mask
                         0,                   // new_axis_mask
                         0,                   // shrink_axis_mask
                         {1, 2, 3, 4, 5, 6},  // input_data
                         {1, 0},              // begin_data
                         {2, 2},              // end_data
                         {1, 1},              // strides_data
                         {2, 2, 2},           // output_shape
                         output_data,         // output_data
                         {1, 2, 4, 5},        // expected_output
                         false,               // expect_prepare_err
                         false,               // expect_invoke_err
                         2                    // num_invoke
  );
}

TF_LITE_MICRO_TEST(In3D_IdentityShrinkAxis1Uint8) {
  uint8_t output_data[12];
  TestStrideSlide<uint8_t, kTfLiteUInt8>(
      {3, 2, 3, 2},                             // input_shape
      {1, 3},                                   //  begin_shape
      {1, 3},                                   // end_shape
      {1, 3},                                   //  strides_shape
      0,                                        // begin_mask
      0,                                        // end_mask
      0,                                        // ellipsis_mask
      0,                                        // new_axis_mask
      1,                                        // shrink_axis_mask
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},  // input_data
      {0, 0, 0},                                // begin_data
      {1, 3, 2},                                // end_data
      {1, 1, 1},                                // strides_data
      {2, 3, 2},                                // output_shape
      output_data,                              // output_data
      {1, 2, 3, 4, 5, 6},                       // expected_output
      false,                                    // expect_prepare_err
      false                                     // expect_invoke_err
  );
}

TF_LITE_MICRO_TEST(In3D_IdentityShrinkAxis1int8) {
  int8_t output_data[12];
  TestStrideSlide<int8_t, kTfLiteInt8>(
      {3, 2, 3, 2},                             // input_shape
      {1, 3},                                   //  begin_shape
      {1, 3},                                   // end_shape
      {1, 3},                                   //  strides_shape
      0,                                        // begin_mask
      0,                                        // end_mask
      0,                                        // ellipsis_mask
      0,                                        // new_axis_mask
      1,                                        // shrink_axis_mask
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},  // input_data
      {0, 0, 0},                                // begin_data
      {1, 3, 2},                                // end_data
      {1, 1, 1},                                // strides_data
      {2, 3, 2},                                // output_shape
      output_data,                              // output_data
      {1, 2, 3, 4, 5, 6},                       // expected_output
      false,                                    // expect_prepare_err
      false                                     // expect_invoke_err
  );
}

TF_LITE_MICRO_TESTS_END
