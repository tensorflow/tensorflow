/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

enum class TestType {
  kConst = 0,
  kDynamic = 1,
};

class Conv3dTransposeOpModel : public SingleOpModel {
 public:
  Conv3dTransposeOpModel(
      std::initializer_list<int> output_shape_data, const TensorData& filter,
      const TensorData& input, const TensorData& bias, const TensorData& output,
      TestType test_type, Padding padding = Padding_VALID,
      int32_t stride_depth = 1, int32_t stride_width = 1,
      int32_t stride_height = 1,
      ActivationFunctionType activation = ActivationFunctionType_NONE,
      int32_t dilation_depth = 1, int32_t dilation_width = 1,
      int32_t dilation_height = 1) {
    if (test_type == TestType::kDynamic) {
      output_shape_ = AddInput({TensorType_INT32, {5}});
    } else {
      output_shape_ = AddConstInput(TensorType_INT32, output_shape_data, {5});
    }
    filter_ = AddInput(filter);
    input_ = AddInput(input);
    bias_ = AddInput(bias);
    output_ = AddOutput(output);
    SetBuiltinOp(
        BuiltinOperator_CONV_3D_TRANSPOSE, BuiltinOptions_Conv3DOptions,
        CreateConv3DOptions(builder_, padding, stride_depth, stride_width,
                            stride_height, activation, dilation_depth,
                            dilation_width, dilation_height)
            .Union());
    BuildInterpreter({GetShape(output_shape_), GetShape(filter_),
                      GetShape(input_), GetShape(bias_)});

    if (test_type == TestType::kDynamic) {
      PopulateTensor(output_shape_, output_shape_data);
    }
  }

  Conv3dTransposeOpModel(
      std::initializer_list<int> output_shape_data, const TensorData& filter,
      const TensorData& input, const TensorData& output, TestType test_type,
      Padding padding = Padding_VALID, int32_t stride_depth = 1,
      int32_t stride_width = 1, int32_t stride_height = 1,
      ActivationFunctionType activation = ActivationFunctionType_NONE,
      int32_t dilation_depth = 1, int32_t dilation_width = 1,
      int32_t dilation_height = 1) {
    if (test_type == TestType::kDynamic) {
      output_shape_ = AddInput({TensorType_INT32, {5}});
    } else {
      output_shape_ = AddConstInput(TensorType_INT32, output_shape_data, {5});
    }
    filter_ = AddInput(filter);
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(
        BuiltinOperator_CONV_3D_TRANSPOSE, BuiltinOptions_Conv3DOptions,
        CreateConv3DOptions(builder_, padding, stride_depth, stride_width,
                            stride_height, activation, dilation_depth,
                            dilation_width, dilation_height)
            .Union());
    BuildInterpreter(
        {GetShape(output_shape_), GetShape(filter_), GetShape(input_)});
    if (test_type == TestType::kDynamic) {
      PopulateTensor(output_shape_, output_shape_data);
    }
  }

  void SetFilter(std::vector<float> f) { PopulateTensor(filter_, f); }

  void SetBias(std::initializer_list<float> f) { PopulateTensor(bias_, f); }

  void SetInput(std::vector<float> data) { PopulateTensor(input_, data); }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int output_shape_;
  int input_;
  int filter_;
  int bias_;
  int output_;
};

template <typename T>
std::vector<T> CreateRangeVector(int N) {
  std::vector<T> result;
  for (int i = 0; i < N; ++i) result.push_back(i);
  return result;
}

class Conv3dTransposeOpTest : public ::testing::TestWithParam<TestType> {};

TEST_P(Conv3dTransposeOpTest, InvalidInputDimsTest) {
  EXPECT_DEATH_IF_SUPPORTED(
      Conv3dTransposeOpModel m(
          {1, 2, 3, 4, 5}, {TensorType_FLOAT32, {2, 2, 4, 1}},
          {TensorType_FLOAT32, {3, 2, 2, 1}}, {TensorType_FLOAT32, {}},
          Conv3dTransposeOpTest::GetParam()),
      "input->dims->size != 5");
}

TEST_P(Conv3dTransposeOpTest, InvalidFilterDimsTest) {
  EXPECT_DEATH_IF_SUPPORTED(
      Conv3dTransposeOpModel m(
          {1, 2, 3, 4, 5}, {TensorType_FLOAT32, {2, 2, 4, 1}},
          {TensorType_FLOAT32, {1, 3, 2, 2, 1}}, {TensorType_FLOAT32, {}},
          Conv3dTransposeOpTest::GetParam()),
      "filter->dims->size != 5");
}

TEST_P(Conv3dTransposeOpTest, MismatchChannelSizeTest) {
  EXPECT_DEATH_IF_SUPPORTED(
      Conv3dTransposeOpModel m(
          {1, 2, 3, 4, 5}, {TensorType_FLOAT32, {1, 2, 2, 4, 1}},
          {TensorType_FLOAT32, {1, 3, 2, 2, 2}}, {TensorType_FLOAT32, {}},
          Conv3dTransposeOpTest::GetParam()),
      "SizeOfDimension.input, 4. != SizeOfDimension.filter, 4.");
}

TEST_P(Conv3dTransposeOpTest, MismatchBiasSizeTest) {
  EXPECT_DEATH_IF_SUPPORTED(
      Conv3dTransposeOpModel m(
          {1, 2, 3, 4, 5}, {TensorType_FLOAT32, {1, 3, 2, 2, 2}},
          {TensorType_FLOAT32, {1, 2, 2, 4, 2}}, {TensorType_FLOAT32, {3}},
          {TensorType_FLOAT32, {}}, Conv3dTransposeOpTest::GetParam()),
      "NumElements.bias. != SizeOfDimension.filter, 3.");
}

TEST_P(Conv3dTransposeOpTest, SimpleFloat32Test) {
  Conv3dTransposeOpModel m(
      {1, 3, 3, 5, 2}, {TensorType_FLOAT32, {2, 2, 2, 2, 2}},
      {TensorType_FLOAT32, {1, 2, 2, 4, 2}}, {TensorType_FLOAT32, {}},
      Conv3dTransposeOpTest::GetParam());

  m.SetInput(CreateRangeVector<float>(32));
  m.SetFilter({-1, -1, -1, -1, -1, 1, -1, 1, -1, 1,  1,  1, 1, 1,  -1, -1,
               1,  -1, 1,  1,  1,  1, -1, 1, -1, -1, -1, 1, 1, -1, 1,  -1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 3, 3, 5, 2));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(
          {-1,  -1,  -4,  -4,  -8,  -8,  -12, -12, 1,   1,   -16, -16, -18,
           -16, -18, -20, -18, -24, 14,  -12, 1,   17,  18,  4,   22,  4,
           26,  4,   29,  -29, -34, -32, -36, -30, -36, -30, -36, -30, 14,
           2,   -50, 2,   -8,  -26, -8,  -26, -8,  -26, 74,  -44, -16, 50,
           28,  4,   28,  4,   28,  4,   60,  -62, -1,  33,  32,  38,  36,
           42,  40,  46,  45,  1,   -34, 50,  10,  54,  10,  58,  10,  62,
           60,  0,   -49, 1,   -54, 0,   -58, 0,   -62, 0,   -1,  -1}));
}

TEST_P(Conv3dTransposeOpTest, PaddingValidTest) {
  Conv3dTransposeOpModel m(
      {1, 4, 5, 6, 2}, {TensorType_FLOAT32, {2, 2, 2, 2, 2}},
      {TensorType_FLOAT32, {1, 3, 4, 5, 2}}, {TensorType_FLOAT32, {}},
      Conv3dTransposeOpTest::GetParam());

  m.SetInput(CreateRangeVector<float>(120));
  m.SetFilter({-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,  1, -1, -1,
               1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, -1, -1, 1, -1, 1, 1,  1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 4, 5, 6, 2));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(
          {-1,   -1,   -6,   -6,   -14,  -14,  -22,  -22,  -30,  -30,  -17,
           -17,  -22,  -20,  -50,  -46,  -58,  -58,  -66,  -70,  -74,  -82,
           -20,  -54,  -62,  -40,  -90,  -106, -98,  -118, -106, -130, -114,
           -142, -20,  -94,  -102, -60,  -130, -166, -138, -178, -146, -190,
           -154, -202, -20,  -134, -61,  1,    -4,   -60,  -4,   -64,  -4,
           -68,  -4,   -72,  77,   -77,  -80,  -80,  -160, -164, -164, -172,
           -168, -180, -172, -188, -96,  -96,  -162, -98,  -188, -282, -196,
           -290, -204, -298, -212, -306, -18,  -196, -202, -118, -228, -322,
           -236, -330, -244, -338, -252, -346, -18,  -216, -242, -138, -268,
           -362, -276, -370, -284, -378, -292, -386, -18,  -236, -202, 2,
           -68,  -78,  -72,  -78,  -76,  -78,  -80,  -78,  158,  -80,  -80,
           -160, -240, -324, -244, -332, -248, -340, -252, -348, -176, -176,
           -322, -178, -348, -442, -356, -450, -364, -458, -372, -466, -18,
           -276, -362, -198, -388, -482, -396, -490, -404, -498, -412, -506,
           -18,  -296, -402, -218, -428, -522, -436, -530, -444, -538, -452,
           -546, -18,  -316, -362, 2,    -148, -78,  -152, -78,  -156, -78,
           -160, -78,  238,  -80,  161,  1,    166,  2,    170,  2,    174,
           2,    178,  2,    1,    1,    20,   2,    22,   164,  22,   168,
           22,   172,  22,   176,  2,    178,  20,   2,    22,   184,  22,
           188,  22,   192,  22,   196,  2,    198,  20,   2,    22,   204,
           22,   208,  22,   212,  22,   216,  2,    218,  -221, 1,    -224,
           222,  -228, 226,  -232, 230,  -236, 234,  1,    237}));
}

TEST_P(Conv3dTransposeOpTest, PaddingSameTest) {
  Conv3dTransposeOpModel m(
      {1, 3, 4, 5, 2}, {TensorType_FLOAT32, {2, 2, 2, 2, 2}},
      {TensorType_FLOAT32, {1, 3, 4, 5, 2}}, {TensorType_FLOAT32, {}},
      Conv3dTransposeOpTest::GetParam(), Padding_SAME);

  m.SetInput(CreateRangeVector<float>(120));
  m.SetFilter({1,  -1, 1,  -1, 1,  -1, -1, 1, 1, -1, -1, 1, 1,  -1, -1, 1,
               -1, 1,  -1, 1,  -1, -1, -1, 1, 1, 1,  1,  1, -1, 1,  -1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 3, 4, 5, 2));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(
          {-1,  -1,  -2,  0,   -2,  0,   -2,  0,   -2,  0,   -2,  0,   -4,  2,
           -4,  2,   -4,  2,   -4,  2,   -2,  0,   -4,  2,   -4,  2,   -4,  2,
           -4,  2,   -2,  0,   -4,  2,   -4,  2,   -4,  2,   -4,  2,   0,   0,
           -2,  2,   -6,  2,   -10, 2,   -14, 2,   0,   2,   -18, 10,  -18, 14,
           -18, 18,  -18, 22,  20,  22,  -18, 30,  -18, 34,  -18, 38,  -18, 42,
           40,  42,  -18, 50,  -18, 54,  -18, 58,  -18, 62,  0,   0,   -82, 2,
           -86, 2,   -90, 2,   -94, 2,   80,  82,  -18, 90,  -18, 94,  -18, 98,
           -18, 102, 100, 102, -18, 110, -18, 114, -18, 118, -18, 122, 120, 122,
           -18, 130, -18, 134, -18, 138, -18, 142}));
}

TEST_P(Conv3dTransposeOpTest, PaddingValidComplexTest) {
  Conv3dTransposeOpModel m(
      {2, 4, 3, 2, 2}, {TensorType_FLOAT32, {2, 2, 2, 2, 2}},
      {TensorType_FLOAT32, {2, 3, 2, 1, 2}}, {TensorType_FLOAT32, {}},
      Conv3dTransposeOpTest::GetParam(), Padding_VALID);

  m.SetInput(CreateRangeVector<float>(24));
  m.SetFilter({1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1,
               1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 4, 3, 2, 2));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(
          {-1, 1,   1, -1, -2, 4,   2, 0,  -1, -5,  1, 5,  -2, 10,  2, -2,
           -4, 8,   4, 8,  -2, -18, 2, 18, -2, 26,  2, -2, -4, 8,   4, 24,
           -2, -34, 2, 34, -1, 17,  1, -1, -2, 4,   2, 16, -1, -21, 1, 21,
           -1, 25,  1, -1, -2, 4,   2, 24, -1, -29, 1, 29, -2, 58,  2, -2,
           -4, 8,   4, 56, -2, -66, 2, 66, -2, 74,  2, -2, -4, 8,   4, 72,
           -2, -82, 2, 82, -1, 41,  1, -1, -2, 4,   2, 40, -1, -45, 1, 45}));
}

TEST_P(Conv3dTransposeOpTest, StrideTest) {
  Conv3dTransposeOpModel m(
      {2, 4, 3, 2, 2}, {TensorType_FLOAT32, {2, 2, 2, 2, 2}},
      {TensorType_FLOAT32, {2, 2, 2, 1, 2}}, {TensorType_FLOAT32, {}},
      Conv3dTransposeOpTest::GetParam(), Padding_VALID,
      /*stride_depth=*/2,
      /*stride_width=*/1, /*stride_height=*/1);

  m.SetInput(CreateRangeVector<float>(16));
  m.SetFilter({1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1,
               1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 4, 3, 2, 2));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(
          {-1, 1,   1, -1, -2, 4,   2, 0,  -1, -5,  1, 5,  -1, 1,   1, -1,
           -2, 4,   2, 0,  -1, -5,  1, 5,  -1, 9,   1, -1, -2, 4,   2, 8,
           -1, -13, 1, 13, -1, 9,   1, -1, -2, 4,   2, 8,  -1, -13, 1, 13,
           -1, 17,  1, -1, -2, 4,   2, 16, -1, -21, 1, 21, -1, 17,  1, -1,
           -2, 4,   2, 16, -1, -21, 1, 21, -1, 25,  1, -1, -2, 4,   2, 24,
           -1, -29, 1, 29, -1, 25,  1, -1, -2, 4,   2, 24, -1, -29, 1, 29}));
}

TEST_P(Conv3dTransposeOpTest, StrideAndPaddingSameTest) {
  Conv3dTransposeOpModel m(
      {2, 4, 2, 1, 2}, {TensorType_FLOAT32, {2, 2, 2, 2, 2}},
      {TensorType_FLOAT32, {2, 2, 2, 1, 2}}, {TensorType_FLOAT32, {}},
      Conv3dTransposeOpTest::GetParam(), Padding_SAME,
      /*stride_depth=*/2,
      /*stride_width=*/1, /*stride_height=*/1);

  m.SetInput(CreateRangeVector<float>(16));
  m.SetFilter({1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1,
               1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 4, 2, 1, 2));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({-1, 1,  -2, 4,  -1, 1,  -2, 4,  -1, 9,  -2,
                                4,  -1, 9,  -2, 4,  -1, 17, -2, 4,  -1, 17,
                                -2, 4,  -1, 25, -2, 4,  -1, 25, -2, 4}));
}

TEST_P(Conv3dTransposeOpTest, DilationTest) {
  Conv3dTransposeOpModel m(
      {1, 3, 3, 2, 2}, {TensorType_FLOAT32, {1, 2, 2, 2, 1}},
      {TensorType_FLOAT32, {1, 3, 1, 1, 1}}, {TensorType_FLOAT32, {}},
      Conv3dTransposeOpTest::GetParam(), Padding_VALID,
      /*stride_depth=*/1,
      /*stride_width=*/1, /*stride_height=*/1,
      /*activation=*/ActivationFunctionType_NONE,
      /*dilation_depth=*/1, /*dilation_width=*/1,
      /*dilation_height=*/2);

  m.SetInput(CreateRangeVector<float>(3));
  m.SetFilter({1, -1, 1, 1, -1, 1, 1, -1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 3, 3, 2, 2));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0, 0,
                                1, -1, 1, 1, 0, 0, 0, 0, -1, 1, 1, -1,
                                2, -2, 2, 2, 0, 0, 0, 0, -2, 2, 2, -2}));
}

TEST_P(Conv3dTransposeOpTest, BiasTest) {
  Conv3dTransposeOpModel m({2, 4, 3, 2, 2},
                           {TensorType_FLOAT32, {2, 2, 2, 2, 2}},
                           {TensorType_FLOAT32, {2, 3, 2, 1, 2}},
                           {TensorType_FLOAT32, {2}}, {TensorType_FLOAT32, {}},
                           Conv3dTransposeOpTest::GetParam(), Padding_VALID);

  m.SetInput(CreateRangeVector<float>(24));
  m.SetFilter({1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1,
               1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1});
  m.SetBias({1, 2});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 4, 3, 2, 2));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(
          {0,  3,   2, 1,  -1, 6,   3, 2,  0,  -3,  2, 7,  -1, 12,  3, 0,
           -3, 10,  5, 10, -1, -16, 3, 20, -1, 28,  3, 0,  -3, 10,  5, 26,
           -1, -32, 3, 36, 0,  19,  2, 1,  -1, 6,   3, 18, 0,  -19, 2, 23,
           0,  27,  2, 1,  -1, 6,   3, 26, 0,  -27, 2, 31, -1, 60,  3, 0,
           -3, 10,  5, 58, -1, -64, 3, 68, -1, 76,  3, 0,  -3, 10,  5, 74,
           -1, -80, 3, 84, 0,  43,  2, 1,  -1, 6,   3, 42, 0,  -43, 2, 47}));
}

INSTANTIATE_TEST_SUITE_P(Conv3dTransposeOpTest, Conv3dTransposeOpTest,
                         ::testing::Values(TestType::kConst,
                                           TestType::kDynamic));

}  // namespace
}  // namespace tflite
