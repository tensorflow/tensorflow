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
#include <vector>

#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/perception/perception_ops.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace ops {
namespace custom {

namespace {

using testing::ElementsAreArray;

class DenseImageWarpOpModel : public SingleOpModel {
 public:
  DenseImageWarpOpModel(const TensorData& input, const TensorData& flow,
                        const TensorData& output) {
    input_ = AddInput(input);
    flow_ = AddInput(flow);
    output_ = AddOutput(output);

    std::vector<uint8_t> custom_option;
    SetCustomOp("DenseImageWarp", custom_option, RegisterDenseImageWarp);
    BuildInterpreter({GetShape(input_), GetShape(flow_)});
  }

  void SetInput(const std::vector<float>& data) {
    PopulateTensor(input_, data);
  }
  void SetFlow(const std::vector<float>& data) { PopulateTensor(flow_, data); }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int flow_;
  int output_;
};

TEST(DenseImageWarpOpTest, MismatchedSizeTest) {
  EXPECT_DEATH_IF_SUPPORTED(
      DenseImageWarpOpModel model(
          /*input=*/{TensorType_FLOAT32, {1, 4, 4, 1}},
          /*flow=*/{TensorType_FLOAT32, {1, 4, 2, 2}},
          /*output=*/{TensorType_FLOAT32, {}});
      , "input_shape.Dims.2. != flow_shape.Dims.2. .4 != 2.");
}

TEST(DenseImageWarpOpTest, WrongFlowSizeTest) {
  EXPECT_DEATH_IF_SUPPORTED(DenseImageWarpOpModel model(
                                /*input=*/{TensorType_FLOAT32, {1, 4, 4, 1}},
                                /*flow=*/{TensorType_FLOAT32, {1, 4, 4, 1}},
                                /*output=*/{TensorType_FLOAT32, {}});
                            , "The last dimension of flow tensor must be 2.");
}

TEST(DenseImageWarpOpTest, SimpleTest) {
  DenseImageWarpOpModel model(
      /*input=*/{TensorType_FLOAT32, {1, 4, 4, 1}},
      /*flow=*/{TensorType_FLOAT32, {1, 4, 4, 2}},
      /*output=*/{TensorType_FLOAT32, {}});
  model.SetInput({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  model.SetFlow({4, 10, 6,  10, 4, 2, 6, 6,  10, -4, 2,  -2, 6,  8, 6, 0,
                 2, -2, 10, 6,  4, 4, 2, -4, -4, 10, -4, -4, -2, 6, 4, 6});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({0, 0, 0, 0, 3, 3, 0, 3, 2, 0,
                                                   0, 3, 12, 15, 12, 0}));
}

TEST(DenseImageWarpOpTest, RoundTest) {
  DenseImageWarpOpModel model(
      /*input=*/{TensorType_FLOAT32, {1, 4, 4, 1}},
      /*flow=*/{TensorType_FLOAT32, {1, 4, 4, 2}},
      /*output=*/{TensorType_FLOAT32, {}});
  model.SetInput({0.2, 1.5, 2.4, 3.5, 4.6, 5.1, 6.3, 7.2, 8.5, 9.6, 10.9, 11.6,
                  12.8, 13.2, 14.4, 15.5});
  model.SetFlow({4, 10, 6,  10, 4, 2, 6, 6,  10, -4, 2,  -2, 6,  8, 6, 0,
                 2, -2, 10, 6,  4, 4, 2, -4, -4, 10, -4, -4, -2, 6, 4, 6});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({0.2, 0.2, 0.2, 0.2, 3.5, 3.5, 0.2, 3.5, 2.4,
                                0.2, 0.2, 3.5, 12.8, 15.5, 12.8, 0.2}));
}

TEST(DenseImageWarpOpTest, WithBatchandChannelTest) {
  DenseImageWarpOpModel model(
      /*input=*/{TensorType_FLOAT32, {2, 4, 4, 3}},
      /*flow=*/{TensorType_FLOAT32, {2, 4, 4, 2}},
      /*output=*/{TensorType_FLOAT32, {}});

  std::vector<float> input_data;
  for (int i = 0; i < 96; ++i) input_data.push_back(i);
  model.SetInput(input_data);
  model.SetFlow({2, -2, 10, 6,  4, 4, 2, -4, -4, 10, -4, -4, -2, 6, 4, 6,
                 4, 10, 6,  10, 4, 2, 6, 6,  10, -4, 2,  -2, 6,  8, 6, 0,
                 2, -2, 10, 6,  4, 4, 2, -4, -4, 10, -4, -4, -2, 6, 4, 6,
                 4, 10, 6,  10, 4, 2, 6, 6,  10, -4, 2,  -2, 6,  8, 6, 0});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 4, 4, 3}));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({6,  7,  8,  0,  1,  2,  0,  1,  2,  9,  10, 11, 36, 37,
                        38, 45, 46, 47, 36, 37, 38, 0,  1,  2,  0,  1,  2,  0,
                        1,  2,  0,  1,  2,  0,  1,  2,  9,  10, 11, 21, 22, 23,
                        0,  1,  2,  9,  10, 11, 54, 55, 56, 48, 49, 50, 48, 49,
                        50, 57, 58, 59, 84, 85, 86, 93, 94, 95, 84, 85, 86, 48,
                        49, 50, 48, 49, 50, 48, 49, 50, 48, 49, 50, 48, 49, 50,
                        57, 58, 59, 69, 70, 71, 48, 49, 50, 57, 58, 59}));
}
}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
