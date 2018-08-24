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
#include <vector>
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;

template <typename T>
class UnpackOpModel : public SingleOpModel {
 public:
  UnpackOpModel(const TensorData& input, int axis) {
    CHECK_LE(axis, input.shape.size());
    const int num_outputs = input.shape[axis];
    input_ = AddInput(input);
    for (int i = 0; i < num_outputs; ++i) {
      outputs_.push_back(AddOutput(input.type));
    }
    SetBuiltinOp(BuiltinOperator_UNPACK, BuiltinOptions_UnpackOptions,
                 CreatePackOptions(builder_, num_outputs, axis).Union());
    BuildInterpreter({GetShape(input_)});
  }

  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(input_, data);
  }

  std::vector<std::vector<T>> GetOutputDatas() {
    std::vector<std::vector<T>> output_datas;
    for (const int output : outputs_) {
      std::cerr << "the output is " << output << std::endl;
      output_datas.push_back(ExtractVector<T>(output));
    }
    return output_datas;
  }

  std::vector<std::vector<int>> GetOutputShapes() {
    std::vector<std::vector<int>> output_shapes;
    for (const int output : outputs_) {
      output_shapes.push_back(GetTensorShape(output));
    }
    return output_shapes;
  }

 private:
  int input_;
  std::vector<int> outputs_;
};

// float32 tests.
TEST(UnpackOpTest, FloatThreeOutputs) {
  UnpackOpModel<float> model({TensorType_FLOAT32, {3, 2}}, 0);
  model.SetInput({1, 2, 3, 4, 5, 6});
  model.Invoke();

  // Check outputs shapes.
  const std::vector<std::vector<int>>& output_shapes = model.GetOutputShapes();
  EXPECT_EQ(output_shapes.size(), 3);
  EXPECT_THAT(output_shapes[0], ElementsAre(2));
  EXPECT_THAT(output_shapes[1], ElementsAre(2));
  EXPECT_THAT(output_shapes[2], ElementsAre(2));

  // Check outputs values.
  const std::vector<std::vector<float>>& output_datas = model.GetOutputDatas();
  EXPECT_EQ(output_datas.size(), 3);
  EXPECT_THAT(output_datas[0], ElementsAre(1, 2));
  EXPECT_THAT(output_datas[1], ElementsAre(3, 4));
  EXPECT_THAT(output_datas[2], ElementsAre(5, 6));
}

TEST(UnpackOpTest, FloatThreeOutputsAxisOne) {
  UnpackOpModel<float> model({TensorType_FLOAT32, {3, 2}}, 1);
  model.SetInput({1, 2, 3, 4, 5, 6});
  model.Invoke();

  // Check outputs shapes.
  const std::vector<std::vector<int>>& output_shapes = model.GetOutputShapes();
  EXPECT_EQ(output_shapes.size(), 2);
  EXPECT_THAT(output_shapes[0], ElementsAre(3));
  EXPECT_THAT(output_shapes[1], ElementsAre(3));

  // Check outputs values.
  const std::vector<std::vector<float>>& output_datas = model.GetOutputDatas();
  EXPECT_EQ(output_datas.size(), 2);
  EXPECT_THAT(output_datas[0], ElementsAre(1, 3, 5));
  EXPECT_THAT(output_datas[1], ElementsAre(2, 4, 6));
}

TEST(UnpackOpTest, FloatOneOutput) {
  UnpackOpModel<float> model({TensorType_FLOAT32, {1, 6}}, 0);
  model.SetInput({1, 2, 3, 4, 5, 6});
  model.Invoke();

  // Check outputs shapes.
  const std::vector<std::vector<int>>& output_shapes = model.GetOutputShapes();
  EXPECT_EQ(output_shapes.size(), 1);
  EXPECT_THAT(output_shapes[0], ElementsAre(6));

  // Check outputs values.
  const std::vector<std::vector<float>>& output_datas = model.GetOutputDatas();
  EXPECT_EQ(output_datas.size(), 1);
  EXPECT_THAT(output_datas[0], ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(UnpackOpTest, FloatThreeDimensionsOutputs) {
  UnpackOpModel<float> model({TensorType_FLOAT32, {2, 2, 2}}, 2);
  model.SetInput({1, 2, 3, 4, 5, 6, 7, 8});
  model.Invoke();

  // Check outputs shapes.
  const std::vector<std::vector<int>>& output_shapes = model.GetOutputShapes();
  EXPECT_EQ(output_shapes.size(), 2);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 2));
  EXPECT_THAT(output_shapes[1], ElementsAre(2, 2));

  // Check outputs values.
  const std::vector<std::vector<float>>& output_datas = model.GetOutputDatas();
  EXPECT_EQ(output_datas.size(), 2);
  EXPECT_THAT(output_datas[0], ElementsAre(1, 3, 5, 7));
  EXPECT_THAT(output_datas[1], ElementsAre(2, 4, 6, 8));
}

// int32 tests.
TEST(UnpackOpTest, IntThreeOutputs) {
  UnpackOpModel<int32_t> model({TensorType_INT32, {3, 2}}, 0);
  model.SetInput({1, 2, 3, 4, 5, 6});
  model.Invoke();

  // Check outputs shapes.
  const std::vector<std::vector<int>>& output_shapes = model.GetOutputShapes();
  EXPECT_EQ(output_shapes.size(), 3);
  EXPECT_THAT(output_shapes[0], ElementsAre(2));
  EXPECT_THAT(output_shapes[1], ElementsAre(2));
  EXPECT_THAT(output_shapes[2], ElementsAre(2));

  // Check outputs values.
  const std::vector<std::vector<int32_t>>& output_datas =
      model.GetOutputDatas();
  EXPECT_EQ(output_datas.size(), 3);
  EXPECT_THAT(output_datas[0], ElementsAre(1, 2));
  EXPECT_THAT(output_datas[1], ElementsAre(3, 4));
  EXPECT_THAT(output_datas[2], ElementsAre(5, 6));
}

TEST(UnpackOpTest, IntThreeOutputsAxisOne) {
  UnpackOpModel<int32_t> model({TensorType_INT32, {3, 2}}, 1);
  model.SetInput({1, 2, 3, 4, 5, 6});
  model.Invoke();

  // Check outputs shapes.
  const std::vector<std::vector<int>>& output_shapes = model.GetOutputShapes();
  EXPECT_EQ(output_shapes.size(), 2);
  EXPECT_THAT(output_shapes[0], ElementsAre(3));
  EXPECT_THAT(output_shapes[1], ElementsAre(3));

  // Check outputs values.
  const std::vector<std::vector<int32_t>>& output_datas =
      model.GetOutputDatas();
  EXPECT_EQ(output_datas.size(), 2);
  EXPECT_THAT(output_datas[0], ElementsAre(1, 3, 5));
  EXPECT_THAT(output_datas[1], ElementsAre(2, 4, 6));
}

TEST(UnpackOpTest, IntOneOutput) {
  UnpackOpModel<int32_t> model({TensorType_INT32, {1, 6}}, 0);
  model.SetInput({1, 2, 3, 4, 5, 6});
  model.Invoke();

  // Check outputs shapes.
  const std::vector<std::vector<int>>& output_shapes = model.GetOutputShapes();
  EXPECT_EQ(output_shapes.size(), 1);
  EXPECT_THAT(output_shapes[0], ElementsAre(6));

  // Check outputs values.
  const std::vector<std::vector<int32_t>>& output_datas =
      model.GetOutputDatas();
  EXPECT_EQ(output_datas.size(), 1);
  EXPECT_THAT(output_datas[0], ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(UnpackOpTest, IntThreeDimensionsOutputs) {
  UnpackOpModel<int32_t> model({TensorType_INT32, {2, 2, 2}}, 2);
  model.SetInput({1, 2, 3, 4, 5, 6, 7, 8});
  model.Invoke();

  // Check outputs shapes.
  const std::vector<std::vector<int>>& output_shapes = model.GetOutputShapes();
  EXPECT_EQ(output_shapes.size(), 2);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 2));
  EXPECT_THAT(output_shapes[1], ElementsAre(2, 2));

  // Check outputs values.
  const std::vector<std::vector<int32_t>>& output_datas =
      model.GetOutputDatas();
  EXPECT_EQ(output_datas.size(), 2);
  EXPECT_THAT(output_datas[0], ElementsAre(1, 3, 5, 7));
  EXPECT_THAT(output_datas[1], ElementsAre(2, 4, 6, 8));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
