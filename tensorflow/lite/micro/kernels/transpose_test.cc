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
#include <stdint.h>

#include <initializer_list>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"
#include "tensorflow/lite/kernels/internal/reference/transpose.h"

namespace tflite {
namespace testing {
namespace {

// template <typename T = float>
// void ValidateTransposeGoldens(TfLiteTensor* tensors, int tensors_size, TfLiteIntArray* inputs_array,
//     TfLiteIntArray* outputs_array, const T* expected_output,
//     const size_t expected_output_len, const int* expected_dims,
//     const size_t expected_dims_len, bool expect_failure) {

//   const TfLiteRegistration registration =
//       tflite::ops::micro::Register_TRANSPOSE();

//   micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
//                              outputs_array,
//                              /*builtin_data=*/nullptr, micro_test::reporter);

//   if (expect_failure) {
//     TF_LITE_MICRO_EXPECT_NE(kTfLiteOk, runner.InitAndPrepare());
//     return;
//   }

//   TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
//   TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

//   TfLiteTensor* output_tensor = &tensors[outputs_array->data[0]];
//   const T* output_data = GetTensorData<T>(output_tensor);
//   for (size_t i = 0; i < expected_output_len; ++i) {
//     TF_LITE_MICRO_EXPECT_NEAR(expected_output[i], output_data[i], 1e-5f);
//   }
//   TF_LITE_MICRO_EXPECT_EQ(expected_dims_len,
//                           static_cast<size_t>(output_tensor->dims->size));
//   for (size_t i = 0; i < expected_dims_len; ++i) {
//     TF_LITE_MICRO_EXPECT_EQ(expected_dims[i], output_tensor->dims->data[i]);
//   }
// }

// template <typename T = float>
// void TestTransposeWithShape(TfLiteTensor* input_tensor, 
//                     TfLiteTensor* perm_tensor, 
//                     TfLiteTensor* output_tensor, 
//                     const T* expected_output,
//                     const size_t expected_output_len,
//                     const int* expected_dims,
//                     const size_t expected_dims_len, 
//                     bool expect_failure) {

//   constexpr int inputs_size = 2;
//   constexpr int outputs_size = 1;
//   constexpr int tensors_size = inputs_size + outputs_size;
//   TfLiteTensor tensors[tensors_size];
//   tensors[0] = *input_tensor;
//   tensors[1] = *perm_tensor;
//   tensors[2] = *output_tensor;

//   int inputs_data[] = {2, 0, 1};
//   TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_data);
//   int outputs_data[] = {1, 2};
//   TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_data);

//   ValidateTransposeGoldens(tensors, tensors_size, 
//                         inputs_array, outputs_array,
//                         expected_output, expected_output_len, 
//                         expected_dims, expected_dims_len, 
//                         expect_failure);

// }

// template <typename T = float, TfLiteType tensor_type = kTfLiteFloat32>
// void TestTranspose(const int* input_dims_data, const T* input_data,
//                    const int* perm_dims_data, const int32_t* perm_data,
//                    int* output_dims_data, T* output_data,
//                    const T* expected_output, const size_t expected_output_len,
//                    const int* expected_dims, const size_t expected_dims_len,
//                    bool expect_failure = false) {

//   TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
//   TfLiteIntArray* perm_dims = IntArrayFromInts(perm_dims_data);
//   TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

//   TfLiteTensor input_tensor =
//       CreateTensor<T, tensor_type>(input_data, input_dims);
//   TfLiteTensor perm_tensor =
//       CreateTensor<int32_t, kTfLiteInt32>(perm_data, perm_dims);
//   TfLiteTensor output_tensor =
//       CreateTensor<T, tensor_type>(output_data, output_dims);

//   TestTransposeWithShape(&input_tensor, &perm_tensor, &output_tensor,
//                           expected_output, expected_output_len, expected_dims,
//                           expected_dims_len, expect_failure);
// }

template <typename T>
inline RuntimeShape GetTensorShape(std::vector<T> data) {
  return RuntimeShape(data.size(), data.data());
}

template <typename T>
void RunTestPermutation(const std::vector<int>& shape,
                        const std::vector<int>& perms,
                        std::vector<T>* input_transposed) {
  // Count elements and allocate output.
  int count = 1;
  for (auto factor : shape) count *= factor;
  input_transposed->resize(count);

  // Create the dummy data
  std::vector<T> input(count);
  for (unsigned int i = 0; i < input.size(); i++) {
    input[i] = i;
  }

  // Make input and output shapes.
  const RuntimeShape input_shape = GetTensorShape(shape);
  RuntimeShape output_shape(perms.size());
  for (unsigned int i = 0; i < perms.size(); i++) {
    output_shape.SetDim(i, input_shape.Dims(perms[i]));
  }

  TransposeParams params;
  params.perm_count = perms.size();
  for (unsigned int i = 0; i < perms.size(); ++i) {
    params.perm[i] = perms[i];
  }

  tflite::reference_ops::Transpose<T>(params, 
      input_shape, input.data(), 
      output_shape, input_transposed->data());
}

}  // namespace
}  // namespace testing
}  // namespace tflite

#define TF_LITE_MICRO_ARRAY_COMP_EQ(_a,_b)              \
    {                                                   \
      TF_LITE_MICRO_EXPECT_EQ(_a.size(),_b.size());     \
      for (unsigned int _e = 0; _e < _a.size(); _e++) { \
        TF_LITE_MICRO_EXPECT_EQ(_a[_e], _b[_e]);        \
      }                                                 \
    }

#define TF_LITE_MICRO_ARRAY_COMP_NE(_a,_b)              \
    {                                                   \
      bool size_eq = _a.size() == _b.size();            \
      bool cont_eq = true;                              \
      if (size_eq) {                                    \
        for (unsigned int _e = 0; _e < _a.size(); _e++) \
          cont_eq &= _a[_e] == _b[_e];                  \
      }                                                 \
      if (size_eq & cont_eq) {                          \
        TF_LITE_MICRO_FAIL("Arrays are equal");         \
      }                                                 \
    }

template <typename T>
void TransposeTestTestRefOps1D() {
  // Basic 1D identity.
  std::vector<T> out;
  tflite::testing::RunTestPermutation<T>({3}, {0}, &out);
  std::vector<T> expected({0, 1, 2});

  TF_LITE_MICRO_ARRAY_COMP_EQ(out, expected);
}

template <typename T>
void TransposeTestTestRefOps2D() {
  std::vector<T> out;
  // Basic 2D.
  tflite::testing::RunTestPermutation<T>({3, 2}, {1, 0}, &out);
  TF_LITE_MICRO_ARRAY_COMP_EQ(out, std::vector<T>({0, 2, 4, 1, 3, 5}));
  // Identity.
  tflite::testing::RunTestPermutation<T>({3, 2}, {0, 1}, &out);
  TF_LITE_MICRO_ARRAY_COMP_EQ(out, std::vector<T>({0, 1, 2, 3, 4, 5}));
}

template <typename T>
void TransposeTestTestRefOps3D() {
  std::vector<T> out;    
  {
    std::vector<T> ref({0, 4, 8,  12, 16, 20, 1, 5, 9,  13, 17, 21,
                          2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23});
    tflite::testing::RunTestPermutation<T>(/*shape=*/{2, 3, 4}, /*perms=*/{2, 0, 1}, &out);  
    TF_LITE_MICRO_ARRAY_COMP_EQ(out, ref);
  }

  // Test 3 dimensional identity transform
  {
    tflite::testing::RunTestPermutation<T>(/*shape=*/{2, 3, 4}, /*perms=*/{0, 1, 2}, &out);
    std::vector<T> ref(out.size());
    for (unsigned int k = 0; k < ref.size(); k++) ref[k] = k;
    TF_LITE_MICRO_ARRAY_COMP_EQ(out, ref);
  }

  /**
   * Additional tests that mimic first case, but with different perm.
   */
  {
    std::vector<T> ref({0, 12, 1, 13, 2, 14, 3, 15, 4,  16, 5,  17,
                            6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23});
    tflite::testing::RunTestPermutation<T>(/*shape=*/{2, 3, 4}, /*perms=*/{1, 2, 0}, &out);
    TF_LITE_MICRO_ARRAY_COMP_EQ(out, ref);
  }

  {
    std::vector<T> ref({0,  4,  8,  1,  5,  9,  2,  6,  10, 3,  7,  11,
                            12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23});
    tflite::testing::RunTestPermutation<T>(/*shape=*/{2, 3, 4}, /*perms=*/{0, 2, 1}, &out);
    TF_LITE_MICRO_ARRAY_COMP_EQ(out, ref);
  }

  {
    std::vector<T> ref({0,  1,  2,  3,  12, 13, 14, 15, 4,  5,  6,  7,
                            16, 17, 18, 19, 8,  9,  10, 11, 20, 21, 22, 23});
    tflite::testing::RunTestPermutation<T>(/*shape=*/{2, 3, 4}, /*perms=*/{1, 0, 2}, &out);
    TF_LITE_MICRO_ARRAY_COMP_EQ(out, ref);
  }

  {
    std::vector<T> ref({0, 12, 4, 16, 8,  20, 1, 13, 5, 17, 9,  21,
                            2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23});
    tflite::testing::RunTestPermutation<T>(/*shape=*/{2, 3, 4}, /*perms=*/{2, 1, 0}, &out);
    TF_LITE_MICRO_ARRAY_COMP_EQ(out, ref);
  }
}

template <typename T>
void TransposeTestTestRefOps3D_OneInDimension() {
  std::vector<T> out;
  // Shape with 1 as first dim -> transposed.
  {
    std::vector<T> ref({0, 3, 1, 4, 2, 5});
    tflite::testing::RunTestPermutation<T>(/*shape=*/{1, 2, 3}, /*perms=*/{2, 0, 1}, &out);
    TF_LITE_MICRO_ARRAY_COMP_EQ(out, ref);
  }
  // Shape with 1 as first dim -> identity.
  {
    std::vector<T> ref({0, 1, 2, 3, 4, 5});
    tflite::testing::RunTestPermutation<T>(/*shape=*/{1, 2, 3}, /*perms=*/{1, 2, 0}, &out);
    TF_LITE_MICRO_ARRAY_COMP_EQ(out, ref);
  }
  // Shape with 1 as third dim -> transposed.
  {
    std::vector<T> ref({0, 3, 1, 4, 2, 5});
    tflite::testing::RunTestPermutation<T>(/*shape=*/{2, 3, 1}, /*perms=*/{1, 2, 0}, &out);
    TF_LITE_MICRO_ARRAY_COMP_EQ(out, ref);
  }
  // Shape with 1 as third dim -> identity.
  {
    std::vector<T> ref({0, 1, 2, 3, 4, 5});
    tflite::testing::RunTestPermutation<T>(/*shape=*/{2, 3, 1}, /*perms=*/{2, 0, 1}, &out);
    TF_LITE_MICRO_ARRAY_COMP_EQ(out, ref);
  }
}

template <typename T>
void TransposeTestTestRefOps4D() {
  std::vector<T> out;
  // Basic 4d.
  tflite::testing::RunTestPermutation<T>({2, 3, 4, 5}, {2, 0, 1, 3}, &out);
  TF_LITE_MICRO_ARRAY_COMP_EQ(
      out,
      std::vector<T>(
          {0,  1,  2,  3,  4,  20, 21, 22, 23, 24, 40,  41,  42,  43,  44,
           60, 61, 62, 63, 64, 80, 81, 82, 83, 84, 100, 101, 102, 103, 104,
           5,  6,  7,  8,  9,  25, 26, 27, 28, 29, 45,  46,  47,  48,  49,
           65, 66, 67, 68, 69, 85, 86, 87, 88, 89, 105, 106, 107, 108, 109,
           10, 11, 12, 13, 14, 30, 31, 32, 33, 34, 50,  51,  52,  53,  54,
           70, 71, 72, 73, 74, 90, 91, 92, 93, 94, 110, 111, 112, 113, 114,
           15, 16, 17, 18, 19, 35, 36, 37, 38, 39, 55,  56,  57,  58,  59,
           75, 76, 77, 78, 79, 95, 96, 97, 98, 99, 115, 116, 117, 118, 119}));
  tflite::testing::RunTestPermutation<T>({2, 3, 4, 5}, {0, 1, 2, 3}, &out);
  // Basic identity.
  std::vector<T> ref(out.size());
  for (unsigned int k = 0; k < ref.size(); k++) ref[k] = k;
  TF_LITE_MICRO_ARRAY_COMP_EQ(out, ref);
};

TF_LITE_MICRO_TESTS_BEGIN

// TF_LITE_MICRO_TEST(MustFail) {
//   TF_LITE_MICRO_FAIL("Boom");
// }

// Safety test to ensure the array tests 
// are passing successfully
TF_LITE_MICRO_TEST(ARRAY_COMP_ShouldSucceed) {
  std::vector<float> a({0, 1, 2, 3, 4, 5});
  std::vector<float> b({0, 1, 2, 3, 4, 5});

  TF_LITE_MICRO_ARRAY_COMP_EQ(a,b);
}

// Safety test to ensure the array tests 
// are failing as expected
TF_LITE_MICRO_TEST(ARRAY_COMP_ShouldFail) {
  std::vector<float> a({0, 1, 2, 3, 4, 6});
  std::vector<float> b({0, 1, 2, 3, 4, 5});
  std::vector<float> c({0, 1, 2, 3, 4});

  TF_LITE_MICRO_ARRAY_COMP_NE(a, b);
  TF_LITE_MICRO_ARRAY_COMP_NE(b, c);
}

TF_LITE_MICRO_TEST(TestRefOps1D) { TransposeTestTestRefOps1D<float>(); }

TF_LITE_MICRO_TEST(TestRefOps2DFloat) { TransposeTestTestRefOps2D<float>(); }
TF_LITE_MICRO_TEST(TestRefOps2DInt8) { TransposeTestTestRefOps2D<int8_t>(); }
TF_LITE_MICRO_TEST(TestRefOps2DUInt8) { TransposeTestTestRefOps2D<uint8_t>(); }

TF_LITE_MICRO_TEST(TestRefOps3DFloat) { TransposeTestTestRefOps3D<float>(); }
TF_LITE_MICRO_TEST(TestRefOps3DInt8) { TransposeTestTestRefOps3D<int8_t>(); }
TF_LITE_MICRO_TEST(TestRefOps3DUInt8) { TransposeTestTestRefOps3D<uint8_t>(); }

TF_LITE_MICRO_TEST(TestRefOps3D_OneInDimensionFloat) { TransposeTestTestRefOps3D_OneInDimension<float>(); }
TF_LITE_MICRO_TEST(TestRefOps3D_OneInDimensionInt8) { TransposeTestTestRefOps3D_OneInDimension<int8_t>(); }
TF_LITE_MICRO_TEST(TestRefOps3D_OneInDimensionUInt8) { TransposeTestTestRefOps3D_OneInDimension<uint8_t>(); }

TF_LITE_MICRO_TEST(TestRefOps4DFloat) { TransposeTestTestRefOps4D<float>(); }
TF_LITE_MICRO_TEST(TestRefOps4DInt8) { TransposeTestTestRefOps4D<int8_t>(); }
TF_LITE_MICRO_TEST(TestRefOps4DInt16) { TransposeTestTestRefOps4D<int16_t>(); }


// TF_LITE_MICRO_TEST(TransposeCreateTensorPerm) {
//   const int perm_dims_data[] = { 1 };
//   const int32_t perm_int32[] = { 1 };

//   TfLiteIntArray* perm_dims = tflite::testing::IntArrayFromInts(perm_dims_data);
//   TfLiteTensor perm_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(perm_dims, perm_int32);

//   TF_LITE_MICRO_EXPECT_EQ(perm_tensor.dims.data[0], 1);
// }

// TF_LITE_MICRO_TEST(TransposeBasic1DIdentityShouldSucceed) {

//   float output_data_float[32];
//   int8_t output_data_int8[32];
//   uint8_t output_data_uint8[32];

//   const int input_dims[] = { 3 };
//   const float input_float[] = { 0, 1, 2 };
//   const int8_t input_int8[] = { 0, 1, 2 };
//   const uint8_t input_uint8[] = { 0, 1, 2 };

//   const int perm_dims[] = { 1 };
//   const int32_t perm_int32[] = { 1 };

//   int output_dims[] = { 3 };

//   const int golden_output_len = 3;
//   const float golden_output_float[] = { 0, 1, 2 };;
//   const int8_t golden_output_int8[] = { 0, 1, 2 };;
//   const uint8_t golden_output_uint8[] = { 0, 1, 2 };;

//   const int golden_dims_len = 1;
//   const int golden_dims[] = { 3 };

//   tflite::testing::TestTranspose(input_dims, input_float, perm_dims, perm_int32,
//                                  output_dims, output_data_float,
//                                  golden_output_float, golden_output_len,
//                                  golden_dims, golden_dims_len);

//   tflite::testing::TestTranspose<int8_t, kTfLiteInt8>(
//             input_dims, input_int8, perm_dims, perm_int32,
//             output_dims, output_data_int8,
//             golden_output_int8, golden_output_len,
//             golden_dims, golden_dims_len);

//   tflite::testing::TestTranspose<uint8_t, kTfLiteUInt8>(
//             input_dims, input_uint8, perm_dims, perm_int32,
//             output_dims, output_data_uint8,
//             golden_output_uint8, golden_output_len,
//             golden_dims, golden_dims_len);

// }

// TF_LITE_MICRO_TEST(TransposeBasic2DShouldSucceed) {

//   float output_data_float[32];
//   int8_t output_data_int8[32];
//   uint8_t output_data_uint8[32];

//   const int input_dims[] = { 3, 2 };
//   const float input_float[] = { 0, 1, 2, 3, 4, 5 };
//   const int8_t input_int8[] = { 0, 1, 2, 3, 4, 5 };
//   const uint8_t input_uint8[] = { 0, 1, 2, 3, 4, 5 };

//   const int perm_dims[] = { 1 };
//   const int32_t perm_int32[] = { 1, 0 };

//    int output_dims[] = { 2, 3 };

//   const int golden_output_len = 6;
//   const float golden_output_float[] = { 0, 2, 4, 1, 3, 5 };
//   const int8_t golden_output_int8[] = { 0, 2, 4, 1, 3, 5 };
//   const uint8_t golden_output_uint8[] = { 0, 2, 4, 1, 3, 5 };

//   const int golden_dims_len = 1;
//   const int golden_dims[] = { 2, 3 };

//   tflite::testing::TestTranspose<float, kTfLiteFloat32>(
//             input_dims, input_float, perm_dims, perm_int32,
//             output_dims, output_data_float,
//             golden_output_float, golden_output_len,
//             golden_dims, golden_dims_len);

//   tflite::testing::TestTranspose<int8_t, kTfLiteInt8>(
//             input_dims, input_int8, perm_dims, perm_int32,
//             output_dims, output_data_int8,
//             golden_output_int8, golden_output_len,
//             golden_dims, golden_dims_len);

//   tflite::testing::TestTranspose<uint8_t, kTfLiteUInt8>(
//             input_dims, input_uint8, perm_dims, perm_int32,
//             output_dims, output_data_uint8,
//             golden_output_uint8, golden_output_len,
//             golden_dims, golden_dims_len);
// }

// TF_LITE_MICRO_TEST(TransposeBasic3D) {

//   float output_data_float[32];
//   int8_t output_data_int8[32];
//   uint8_t output_data_uint8[32];

//   const int input_dims[] = { 1, 2, 3 };
//   const float input_float[] = { 0, 1, 2, 3, 4, 5 };
//   const int8_t input_int8[] = { 0, 1, 2, 3, 4, 5 };
//   const uint8_t input_uint8[] = { 0, 1, 2, 3, 4, 5 };

//   const int perm_dims[] = { 3 };
//   const int32_t perm_int32[] = { 2, 0, 1 };

//    int output_dims[] = { 2, 3 };

//   const int golden_output_len = 6;
//   const float golden_output_float[] = { 0, 2, 4, 1, 3, 5 };
//   const int8_t golden_output_int8[] = { 0, 2, 4, 1, 3, 5 };
//   const uint8_t golden_output_uint8[] = { 0, 2, 4, 1, 3, 5 };

//   const int golden_dims_len = 1;
//   const int golden_dims[] = { 2, 3 };

//   tflite::testing::TestTranspose<float, kTfLiteFloat32>(
//             input_dims, input_float, perm_dims, perm_int32,
//             output_dims, output_data_float,
//             golden_output_float, golden_output_len,
//             golden_dims, golden_dims_len);

//   tflite::testing::TestTranspose<int8_t, kTfLiteInt8>(
//             input_dims, input_int8, perm_dims, perm_int32,
//             output_dims, output_data_int8,
//             golden_output_int8, golden_output_len,
//             golden_dims, golden_dims_len);

//   tflite::testing::TestTranspose<uint8_t, kTfLiteUInt8>(
//             input_dims, input_uint8, perm_dims, perm_int32,
//             output_dims, output_data_uint8,
//             golden_output_uint8, golden_output_len,
//             golden_dims, golden_dims_len);

// }


// TF_LITE_MICRO_TEST(TransposeBasic4DShouldSucceed) {

//   float output_data_float[64];
//   int8_t output_data_int8[64];
//   uint8_t output_data_uint8[64];

//   const int input_dims[] = { 2, 1, 5, 4 };
//   const float input_float[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40 };
//   const int8_t input_int8[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40 };
//   const uint8_t input_uint8[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40 };

//   const int perm_dims[] = { 1 };
//   const int32_t perm_int32[] = { 3, 2, 1, 0 };

//   int output_dims[] = { 4 };

//   const int golden_output_len = 40;

//   const float golden_output_float[] = { 1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19, 4, 8, 12, 16, 20, 21, 25, 29, 33, 37, 22, 26, 30, 34, 38, 23, 27, 31, 35, 39, 24, 28, 32, 36, 40 };
//   const int8_t golden_output_int8[] = { 1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19, 4, 8, 12, 16, 20, 21, 25, 29, 33, 37, 22, 26, 30, 34, 38, 23, 27, 31, 35, 39, 24, 28, 32, 36, 40 };
//   const uint8_t golden_output_uint8[] = { 1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19, 4, 8, 12, 16, 20, 21, 25, 29, 33, 37, 22, 26, 30, 34, 38, 23, 27, 31, 35, 39, 24, 28, 32, 36, 40 };

//   const int golden_dims_len = 4;
//   const int golden_dims[] = { 2, 4, 5, 1 };

//   tflite::testing::TestTranspose(
//             input_dims, input_float, perm_dims, perm_int32,
//             output_dims, output_data_float,
//             golden_output_float, golden_output_len,
//             golden_dims, golden_dims_len);

//   tflite::testing::TestTranspose<int8_t, kTfLiteInt8>(
//             input_dims, input_int8, perm_dims, perm_int32,
//             output_dims, output_data_int8,
//             golden_output_int8, golden_output_len,
//             golden_dims, golden_dims_len);

//   tflite::testing::TestTranspose<uint8_t, kTfLiteUInt8>(
//             input_dims, input_uint8, perm_dims, perm_int32,
//             output_dims, output_data_uint8,
//             golden_output_uint8, golden_output_len,
//             golden_dims, golden_dims_len);

// }

TF_LITE_MICRO_TESTS_END

#undef TF_LITE_MICRO_ARRAY_COMP_EQ
#undef TF_LITE_MICRO_ARRAY_COMP_NE 
