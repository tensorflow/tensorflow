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
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename T>
class ArgBaseOpModel : public SingleOpModel {
 public:
  ArgBaseOpModel(std::initializer_list<int> input_shape, TensorType input_type,
                 TensorType output_type, TensorType index_output_type) {
    input_ = AddInput(input_type);
    axis_ = AddInput(TensorType_INT32);
    output_ = AddOutput(output_type);
  }

  int input() { return input_; }
  int axis() { return axis_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int axis_;
  int output_;
};

template <typename T>
class ArgMaxOpModel : public ArgBaseOpModel<T> {
 public:
  ArgMaxOpModel(std::initializer_list<int> input_shape, TensorType input_type,
                TensorType output_type, TensorType index_output_type)
      : ArgBaseOpModel<T>(input_shape, input_type, output_type,
                          index_output_type) {
    ArgBaseOpModel<T>::SetBuiltinOp(
        BuiltinOperator_ARG_MAX, BuiltinOptions_ArgMaxOptions,
        CreateArgMaxOptions(ArgBaseOpModel<T>::builder_, index_output_type)
            .Union());
    ArgBaseOpModel<T>::BuildInterpreter({input_shape, {1, 1, 1, 1}});
  }
};

template <typename T>
class ArgMinOpModel : public ArgBaseOpModel<T> {
 public:
  ArgMinOpModel(std::initializer_list<int> input_shape, TensorType input_type,
                TensorType output_type, TensorType index_output_type)
      : ArgBaseOpModel<T>(input_shape, input_type, output_type,
                          index_output_type) {
    ArgBaseOpModel<T>::SetBuiltinOp(
        BuiltinOperator_ARG_MIN, BuiltinOptions_ArgMinOptions,
        CreateArgMinOptions(ArgBaseOpModel<T>::builder_, index_output_type)
            .Union());
    ArgBaseOpModel<T>::BuildInterpreter({input_shape, {1, 1, 1, 1}});
  }
};

TEST(ArgMaxOpTest, GetMaxArgFloat) {
  ArgMaxOpModel<int32_t> model({1, 1, 1, 4}, TensorType_FLOAT32,
                               TensorType_INT32, TensorType_INT32);
  model.PopulateTensor<float>(model.input(), {0.1, 0.9, 0.7, 0.3});
  model.PopulateTensor<int>(model.axis(), {3});
  model.Invoke();

  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1, 1}));
}

TEST(ArgMaxOpTest, GetMaxArgInt) {
  ArgMaxOpModel<int32_t> model({1, 1, 1, 4}, TensorType_INT32, TensorType_INT32,
                               TensorType_INT32);
  model.PopulateTensor<int>(model.input(), {1, 9, 7, 3});
  model.PopulateTensor<int>(model.axis(), {3});
  model.Invoke();

  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1, 1}));
}

TEST(ArgMaxOpTest, GetMaxArgMulDimensions) {
  ArgMaxOpModel<int32_t> model({1, 1, 2, 4}, TensorType_INT32, TensorType_INT32,
                               TensorType_INT32);
  model.PopulateTensor<int>(model.input(), {1, 2, 7, 8, 1, 9, 7, 3});
  model.PopulateTensor<int>(model.axis(), {3});
  model.Invoke();

  EXPECT_THAT(model.GetOutput(), ElementsAreArray({3, 1}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 2, 1}));
}

TEST(ArgMaxOpTest, GetMaxArgNegativeAxis) {
  ArgMaxOpModel<int32_t> model({1, 1, 2, 4}, TensorType_INT32, TensorType_INT32,
                               TensorType_INT32);
  model.PopulateTensor<int>(model.input(), {1, 2, 7, 8, 1, 9, 7, 3});
  model.PopulateTensor<int>(model.axis(), {-2});
  model.Invoke();

  EXPECT_THAT(model.GetOutput(), ElementsAreArray({0, 1, 0, 0}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1, 4}));
}

TEST(ArgMaxOpTest, GetMaxArgOutput64) {
  ArgMaxOpModel<int64_t> model({1, 1, 2, 4}, TensorType_INT32, TensorType_INT64,
                               TensorType_INT64);
  model.PopulateTensor<int>(model.input(), {10, 2, 7, 8, 1, 9, 7, 3});
  model.PopulateTensor<int>(model.axis(), {3});
  model.Invoke();

  EXPECT_THAT(model.GetOutput(), ElementsAreArray({0, 1}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 2, 1}));
}

TEST(ArgMinOpTest, GetMinArgFloat) {
  ArgMinOpModel<int32_t> model({1, 1, 1, 4}, TensorType_FLOAT32,
                               TensorType_INT32, TensorType_INT32);
  model.PopulateTensor<float>(model.input(), {0.1, 0.9, 0.7, 0.3});
  model.PopulateTensor<int>(model.axis(), {3});
  model.Invoke();

  EXPECT_THAT(model.GetOutput(), ElementsAreArray({0}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1, 1}));
}

TEST(ArgMinOpTest, GetMinArgInt) {
  ArgMinOpModel<int32_t> model({1, 1, 1, 4}, TensorType_INT32, TensorType_INT32,
                               TensorType_INT32);
  model.PopulateTensor<int>(model.input(), {1, 9, 7, 3});
  model.PopulateTensor<int>(model.axis(), {3});
  model.Invoke();

  EXPECT_THAT(model.GetOutput(), ElementsAreArray({0}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1, 1}));
}

TEST(ArgMinOpTest, GetMinArgMulDimensions) {
  ArgMinOpModel<int32_t> model({1, 1, 2, 4}, TensorType_INT32, TensorType_INT32,
                               TensorType_INT32);
  model.PopulateTensor<int>(model.input(), {1, 2, 7, 8, 1, 9, 7, 3});
  model.PopulateTensor<int>(model.axis(), {3});
  model.Invoke();

  EXPECT_THAT(model.GetOutput(), ElementsAreArray({0, 0}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 2, 1}));
}

TEST(ArgMinOpTest, GetMinArgNegativeAxis) {
  ArgMinOpModel<int32_t> model({1, 1, 2, 4}, TensorType_INT32, TensorType_INT32,
                               TensorType_INT32);
  model.PopulateTensor<int>(model.input(), {1, 2, 7, 8, 1, 9, 7, 3});
  model.PopulateTensor<int>(model.axis(), {-2});
  model.Invoke();

  EXPECT_THAT(model.GetOutput(), ElementsAreArray({0, 0, 0, 1}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1, 4}));
}

TEST(ArgMinOpTest, GetMinArgOutput64) {
  ArgMinOpModel<int64_t> model({1, 1, 2, 4}, TensorType_INT32, TensorType_INT64,
                               TensorType_INT64);
  model.PopulateTensor<int>(model.input(), {10, 2, 7, 8, 1, 9, 7, 3});
  model.PopulateTensor<int>(model.axis(), {3});
  model.Invoke();

  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 0}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 2, 1}));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
