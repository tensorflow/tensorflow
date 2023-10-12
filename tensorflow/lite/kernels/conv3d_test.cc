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

class Conv3dOpModel : public SingleOpModel {
 public:
  Conv3dOpModel(const TensorData& input, const TensorData& filter,
                const TensorData& bias, const TensorData& output,
                Padding padding = Padding_VALID, int32_t stride_depth = 1,
                int32_t stride_width = 1, int32_t stride_height = 1,
                ActivationFunctionType activation = ActivationFunctionType_NONE,
                int32_t dilation_depth = 1, int32_t dilation_width = 1,
                int32_t dilation_height = 1) {
    input_ = AddInput(input);
    filter_ = AddInput(filter);
    bias_ = AddInput(bias);
    output_ = AddOutput(output);
    SetBuiltinOp(
        BuiltinOperator_CONV_3D, BuiltinOptions_Conv3DOptions,
        CreateConv3DOptions(builder_, padding, stride_depth, stride_width,
                            stride_height, activation, dilation_depth,
                            dilation_width, dilation_height)
            .Union());
    BuildInterpreter({GetShape(input_), GetShape(filter_), GetShape(bias_)});
  }

  Conv3dOpModel(const TensorData& input, const TensorData& filter,
                const TensorData& output, Padding padding = Padding_VALID,
                int32_t stride_depth = 1, int32_t stride_width = 1,
                int32_t stride_height = 1,
                ActivationFunctionType activation = ActivationFunctionType_NONE,
                int32_t dilation_depth = 1, int32_t dilation_width = 1,
                int32_t dilation_height = 1) {
    input_ = AddInput(input);
    filter_ = AddInput(filter);
    output_ = AddOutput(output);
    SetBuiltinOp(
        BuiltinOperator_CONV_3D, BuiltinOptions_Conv3DOptions,
        CreateConv3DOptions(builder_, padding, stride_depth, stride_width,
                            stride_height, activation, dilation_depth,
                            dilation_width, dilation_height)
            .Union());
    BuildInterpreter({GetShape(input_), GetShape(filter_)});
  }

  void SetFilter(std::vector<float> f) { PopulateTensor(filter_, f); }

  void SetBias(std::initializer_list<float> f) { PopulateTensor(bias_, f); }

  void SetInput(std::vector<float> data) { PopulateTensor(input_, data); }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
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

TEST(Conv3dOpModel, InvalidInputDimsTest) {
  EXPECT_DEATH_IF_SUPPORTED(Conv3dOpModel m({TensorType_FLOAT32, {2, 2, 4, 1}},
                                            {TensorType_FLOAT32, {3, 2, 2, 1}},
                                            {TensorType_FLOAT32, {}}),
                            "input->dims->size != 5");
}

TEST(Conv3dOpModel, InvalidFilterDimsTest) {
  EXPECT_DEATH_IF_SUPPORTED(
      Conv3dOpModel m({TensorType_FLOAT32, {1, 2, 2, 4, 1}},
                      {TensorType_FLOAT32, {3, 2, 2, 1}},
                      {TensorType_FLOAT32, {}}),
      "filter->dims->size != 5");
}

TEST(Conv3dOpModel, MismatchChannelSizeTest) {
  EXPECT_DEATH_IF_SUPPORTED(
      Conv3dOpModel m({TensorType_FLOAT32, {1, 2, 2, 4, 1}},
                      {TensorType_FLOAT32, {1, 3, 2, 2, 2}},
                      {TensorType_FLOAT32, {}}),
      "input->dims->data.4. != filter->dims->data.3.");
}

TEST(Conv3dOpModel, MismatchBiasSizeTest) {
  EXPECT_DEATH_IF_SUPPORTED(
      Conv3dOpModel m({TensorType_FLOAT32, {1, 2, 2, 4, 2}},
                      {TensorType_FLOAT32, {1, 3, 2, 2, 1}},
                      {TensorType_FLOAT32, {2}}, {TensorType_FLOAT32, {}}),
      "NumElements.bias. != SizeOfDimension.filter, 4.");
}

TEST(Conv3dOpModel, SimpleFloat32Test) {
  Conv3dOpModel m({TensorType_FLOAT32, {1, 2, 2, 4, 2}},
                  {TensorType_FLOAT32, {2, 2, 2, 2, 2}},
                  {TensorType_FLOAT32, {}});

  m.SetInput(CreateRangeVector<float>(32));
  m.SetFilter({-1, -1, -1, -1, -1, 1, -1, 1, -1, 1,  1,  1, 1, 1,  -1, -1,
               1,  -1, 1,  1,  1,  1, -1, 1, -1, -1, -1, 1, 1, -1, 1,  -1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 1, 1, 3, 2));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({30, 6, 26, 10, 22, 14}));
}

TEST(Conv3dOpModel, PaddingValidTest) {
  Conv3dOpModel m({TensorType_FLOAT32, {1, 3, 4, 5, 2}},
                  {TensorType_FLOAT32, {2, 2, 2, 2, 2}},
                  {TensorType_FLOAT32, {}});

  m.SetInput(CreateRangeVector<float>(120));
  m.SetFilter({-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,  1, -1, -1,
               1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, -1, -1, 1, -1, 1, 1,  1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 2, 3, 4, 2));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({-214, 266, -234, 270, -254, 274, -274, 278, -314, 286,
                        -334, 290, -354, 294, -374, 298, -414, 306, -434, 310,
                        -454, 314, -474, 318, -614, 346, -634, 350, -654, 354,
                        -674, 358, -714, 366, -734, 370, -754, 374, -774, 378,
                        -814, 386, -834, 390, -854, 394, -874, 398}));
}

TEST(Conv3dOpModel, PaddingSameTest) {
  Conv3dOpModel m({TensorType_FLOAT32, {1, 3, 4, 5, 2}},
                  {TensorType_FLOAT32, {2, 2, 2, 2, 2}},
                  {TensorType_FLOAT32, {}}, Padding_SAME);

  m.SetInput(CreateRangeVector<float>(120));
  m.SetFilter({1,  -1, 1,  -1, 1,  -1, -1, 1, 1, -1, -1, 1, 1,  -1, -1, 1,
               -1, 1,  -1, 1,  -1, -1, -1, 1, 1, 1,  1,  1, -1, 1,  -1, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 3, 4, 5, 2));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(
          {-172, 290,  -176, 298,  -180, 306,  -184, 314,  36,   198,  -192,
           330,  -196, 338,  -200, 346,  -204, 354,  56,   218,  -212, 370,
           -216, 378,  -220, 386,  -224, 394,  76,   238,  -226, 82,   -230,
           82,   -234, 82,   -238, 82,   -80,  80,   -252, 450,  -256, 458,
           -260, 466,  -264, 474,  116,  278,  -272, 490,  -276, 498,  -280,
           506,  -284, 514,  136,  298,  -292, 530,  -296, 538,  -300, 546,
           -304, 554,  156,  318,  -306, 82,   -310, 82,   -314, 82,   -318,
           82,   -80,  80,   158,  -158, 162,  -162, 166,  -166, 170,  -170,
           176,  -176, 178,  -178, 182,  -182, 186,  -186, 190,  -190, 196,
           -196, 198,  -198, 202,  -202, 206,  -206, 210,  -210, 216,  -216,
           220,  -220, 224,  -224, 228,  -228, 232,  -232, 237,  -237}));
}

TEST(Conv3dOpModel, StrideTest) {
  Conv3dOpModel m({TensorType_FLOAT32, {2, 2, 3, 4, 2}},
                  {TensorType_FLOAT32, {2, 2, 2, 2, 2}},
                  {TensorType_FLOAT32, {}}, Padding_VALID, /*stride_depth=*/2,
                  /*stride_width=*/2, /*stride_height=*/2);

  m.SetInput(CreateRangeVector<float>(96));
  m.SetFilter({1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1,
               1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 1, 1, 2, 2));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({52, 8, 68, 8, 244, 8, 260, 8}));
}

TEST(Conv3dOpModel, StrideAndPaddingSameTest) {
  Conv3dOpModel m({TensorType_FLOAT32, {2, 2, 3, 4, 2}},
                  {TensorType_FLOAT32, {2, 2, 2, 2, 2}},
                  {TensorType_FLOAT32, {}}, Padding_SAME, /*stride_depth=*/2,
                  /*stride_width=*/2, /*stride_height=*/2);

  m.SetInput(CreateRangeVector<float>(96));
  m.SetFilter({-1, 1, -1, 1,  1,  1,  1,  1,  -1, 1, -1, -1, -1, 1,  1,  1,
               1,  1, -1, -1, -1, -1, -1, -1, 1,  1, 1,  -1, -1, -1, -1, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 1, 2, 2, 2));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({-70, -28, -86, -12, -82, -16, -90, -8, -262,
                                164, -278, 180, -178, 80, -186, 88}));
}

TEST(Conv3dOpModel, DilationTest) {
  Conv3dOpModel m({TensorType_FLOAT32, {2, 2, 3, 4, 2}},
                  {TensorType_FLOAT32, {2, 2, 2, 2, 2}},
                  {TensorType_FLOAT32, {}}, Padding_VALID, /*stride_depth=*/1,
                  /*stride_width=*/1, /*stride_height=*/1,
                  /*activation=*/ActivationFunctionType_NONE,
                  /*dilation_depth=*/1, /*dilation_width=*/1,
                  /*dilation_height=*/2);

  m.SetInput(CreateRangeVector<float>(96));
  m.SetFilter(CreateRangeVector<float>(32));
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 1, 1, 3, 2));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({7248, 7592, 7728, 8104, 8208, 8616, 18768,
                                19880, 19248, 20392, 19728, 20904}));
}

TEST(Conv3dOpModel, BiasTest) {
  Conv3dOpModel m({TensorType_FLOAT32, {2, 2, 3, 4, 2}},
                  {TensorType_FLOAT32, {2, 2, 2, 2, 2}},
                  {TensorType_FLOAT32, {2}}, {TensorType_FLOAT32, {}},
                  Padding_VALID, /*stride_depth=*/2,
                  /*stride_width=*/2, /*stride_height=*/2);

  m.SetInput(CreateRangeVector<float>(96));
  m.SetFilter({1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1,
               1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1});
  m.SetBias({1, 2});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 1, 1, 2, 2));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({53, 10, 69, 10, 245, 10, 261, 10}));
}

TEST(Conv3dOpModel, NoIm2ColTensorTest) {
  Conv3dOpModel m({TensorType_FLOAT32, {1, 2, 2, 2, 4}},
                  {TensorType_FLOAT32, {1, 1, 1, 4, 4}},
                  {TensorType_FLOAT32, {}}, Padding_VALID);

  m.SetInput(CreateRangeVector<float>(32));
  m.SetFilter(CreateRangeVector<float>(16));
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 2, 2, 2, 4));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({56,  62,  68,  74,  152, 174, 196, 218, 248, 286, 324,
                        362, 344, 398, 452, 506, 440, 510, 580, 650, 536, 622,
                        708, 794, 632, 734, 836, 938, 728, 846, 964, 1082}));
}

}  // namespace
}  // namespace tflite
