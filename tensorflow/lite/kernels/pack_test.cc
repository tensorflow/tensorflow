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
#include <stdint.h>

#include <initializer_list>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

template <typename T>
class PackOpModel : public SingleOpModel {
 public:
  PackOpModel(const TensorData& input_template, int axis, int values_count,
              bool constant_tensors,
              const std::vector<std::vector<T>>& input_data) {
    std::vector<std::vector<int>> all_input_shapes;
    for (int i = 0; i < values_count; ++i) {
      all_input_shapes.push_back(input_template.shape);
      if (constant_tensors) {
        AddConstInput(input_template, input_data[i]);
      } else {
        AddInput(input_template);
      }
    }
    output_ = AddOutput({input_template.type, /*shape=*/{}, input_template.min,
                         input_template.max});
    SetBuiltinOp(BuiltinOperator_PACK, BuiltinOptions_PackOptions,
                 CreatePackOptions(builder_, values_count, axis).Union());
    BuildInterpreter(all_input_shapes, /*use_simple_allocator=*/false);
    if (!constant_tensors) {
      for (int i = 0; i < values_count; ++i) {
        PopulateTensor(i, input_data[i]);
      }
    }
  }

  void SetInput(int index, std::initializer_list<T> data) {
    PopulateTensor(index, data);
  }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int output_;
};

template <typename InputType>
struct PackOpTestInt : public ::testing::Test {
  using TypeToTest = InputType;
  TensorType tensor_type = GetTensorType<InputType>();
};

using TestTypes =
    testing::Types<int8_t, uint8_t, int16_t, int32_t, int64_t, float>;
TYPED_TEST_CASE(PackOpTestInt, TestTypes);

TYPED_TEST(PackOpTestInt, ThreeInputs) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    PackOpModel<typename TestFixture::TypeToTest> model(
        {TestFixture::tensor_type, {2}}, 0, 3, constant_tensors,
        {{1, 4}, {2, 5}, {3, 6}});
    ASSERT_EQ(model.Invoke(), kTfLiteOk);
    EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 2));
    EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 4, 2, 5, 3, 6}));
  }
}

TYPED_TEST(PackOpTestInt, ThreeInputsDifferentAxis) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    PackOpModel<typename TestFixture::TypeToTest> model(
        {TestFixture::tensor_type, {2}}, 1, 3, constant_tensors,
        {{1, 4}, {2, 5}, {3, 6}});
    ASSERT_EQ(model.Invoke(), kTfLiteOk);
    EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
    EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
  }
}
TYPED_TEST(PackOpTestInt, ThreeInputsNegativeAxis) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    PackOpModel<typename TestFixture::TypeToTest> model(
        {TestFixture::tensor_type, {2}}, -1, 3, constant_tensors,
        {{1, 4}, {2, 5}, {3, 6}});
    ASSERT_EQ(model.Invoke(), kTfLiteOk);
    EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
    EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
  }
}
TYPED_TEST(PackOpTestInt, MultilDimensions) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    PackOpModel<typename TestFixture::TypeToTest> model(
        {TestFixture::tensor_type, {2, 3}}, 1, 2, constant_tensors,
        {{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}});
    ASSERT_EQ(model.Invoke(), kTfLiteOk);
    EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 2, 3));
    EXPECT_THAT(model.GetOutput(),
                ElementsAreArray({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));
  }
}

TYPED_TEST(PackOpTestInt, FiveDimensions) {
  for (bool constant_tensors : {true, false}) {
    if (SingleOpModel::GetForceUseNnapi() && constant_tensors) {
      // NNAPI does not support graphs with all constant inputs.
      continue;
    }
    PackOpModel<typename TestFixture::TypeToTest> model(
        {TestFixture::tensor_type, {2, 2, 2, 2}}, 1, 2, constant_tensors,
        {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
         {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}});
    ASSERT_EQ(model.Invoke(), kTfLiteOk);
    EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 2, 2, 2, 2));
    EXPECT_THAT(model.GetOutput(),
                ElementsAreArray({1,  2,  3,  4,  5,  6,  7,  8,  17, 18, 19,
                                  20, 21, 22, 23, 24, 9,  10, 11, 12, 13, 14,
                                  15, 16, 25, 26, 27, 28, 29, 30, 31, 32}));
  }
}
}  // namespace
}  // namespace tflite
