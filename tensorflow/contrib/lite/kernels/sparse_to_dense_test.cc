
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
#include <cstdarg>
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename T>
class SparseToDenseOpModel : public SingleOpModel {
 public:
  SparseToDenseOpModel(std::initializer_list<int> indices_shape,
                       std::initializer_list<int> output_shape_shape,
                       std::initializer_list<int> values_shape, T default_value,
                       TensorType tensor_index_type,
                       TensorType tensor_input_type) {
    indices_ = AddInput(tensor_index_type);
    output_shape_ = AddInput(TensorType_INT32);
    values_ = AddInput(tensor_input_type);
    default_value_ = AddInput(tensor_input_type);
    output_ = AddOutput(tensor_input_type);

    SetBuiltinOp(BuiltinOperator_SPARSE_TO_DENSE,
                 BuiltinOptions_SparseToDenseOptions,
                 CreateSparseToDenseOptions(builder_, false).Union());
    BuildInterpreter({indices_shape, output_shape_shape, values_shape, {1}});

    PopulateTensor<T>(default_value_, {default_value});
  }

  int indices() { return indices_; }
  int output_shape() { return output_shape_; }
  int values() { return values_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int indices_;
  int output_shape_;
  int values_;
  int default_value_;
  int output_;
};

TEST(SparseToDenseOpModelTest, ZeroDimensionTest) {
  SparseToDenseOpModel<float> m({1}, {1}, {1}, 0, TensorType_INT32,
                                TensorType_FLOAT32);
  m.PopulateTensor<int32_t>(m.indices(), {3});
  m.PopulateTensor<int32_t>(m.output_shape(), {5});
  m.PopulateTensor<float>(m.values(), {7});
  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0, 7, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({5}));
}

TEST(SparseToDenseOpModelTest, OneDimensionTest) {
  SparseToDenseOpModel<float> m({3}, {1}, {3}, 0, TensorType_INT32,
                                TensorType_FLOAT32);
  m.PopulateTensor<int32_t>(m.indices(), {1, 3, 5});
  m.PopulateTensor<int32_t>(m.output_shape(), {7});
  m.PopulateTensor<float>(m.values(), {2, 4, 6});
  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 2, 0, 4, 0, 6, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({7}));
}

TEST(SparseToDenseOpModelTest, TwoDimensionsTest) {
  SparseToDenseOpModel<float> m({3, 3}, {3}, {3}, 0, TensorType_INT32,
                                TensorType_FLOAT32);
  m.PopulateTensor<int32_t>(m.indices(), {0, 0, 0, 1, 2, 1, 2, 0, 1});
  m.PopulateTensor<int32_t>(m.output_shape(), {3, 3, 3});
  m.PopulateTensor<float>(m.values(), {2, 4, 6});
  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 4, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 3, 3}));
}

TEST(SparseToDenseOpModelTest, DefaultValueTest) {
  SparseToDenseOpModel<float> m({3, 3}, {3}, {3}, -1, TensorType_INT32,
                                TensorType_FLOAT32);
  m.PopulateTensor<int32_t>(m.indices(), {0, 0, 0, 1, 2, 1, 2, 0, 1});
  m.PopulateTensor<int32_t>(m.output_shape(), {3, 3, 3});
  m.PopulateTensor<float>(m.values(), {2, 4, 6});
  m.Invoke();

  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        -1, -1, 4,  -1, -1, 6,  -1, -1, -1, -1, -1, -1, -1}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 3, 3}));
}

TEST(SparseToDenseOpModelTest, IntegerValueTest) {
  SparseToDenseOpModel<int32_t> m({3, 3}, {3}, {3}, -1, TensorType_INT32,
                                  TensorType_INT32);
  m.PopulateTensor<int32_t>(m.indices(), {0, 0, 0, 1, 2, 1, 2, 0, 1});
  m.PopulateTensor<int32_t>(m.output_shape(), {3, 3, 3});
  m.PopulateTensor<int32_t>(m.values(), {2, 4, 6});
  m.Invoke();

  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        -1, -1, 4,  -1, -1, 6,  -1, -1, -1, -1, -1, -1, -1}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 3, 3}));
}

TEST(SparseToDenseOpModelTest, Int64IndexTest) {
  SparseToDenseOpModel<float> m({3, 3}, {3}, {3}, -1, TensorType_INT64,
                                TensorType_FLOAT32);
  m.PopulateTensor<int64_t>(m.indices(), {0, 0, 0, 1, 2, 1, 2, 0, 1});
  m.PopulateTensor<int32_t>(m.output_shape(), {3, 3, 3});
  m.PopulateTensor<float>(m.values(), {2, 4, 6});
  m.Invoke();

  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        -1, -1, 4,  -1, -1, 6,  -1, -1, -1, -1, -1, -1, -1}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 3, 3}));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
