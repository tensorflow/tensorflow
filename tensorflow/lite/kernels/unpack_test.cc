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
#include <iostream>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename T>
class UnpackOpModel : public SingleOpModel {
 public:
  UnpackOpModel(const TensorData& input, int axis) {
    if (axis < 0) {
      axis += input.shape.size();
    }
    const int num_outputs = input.shape[axis];
    input_ = AddInput(input);
    for (int i = 0; i < num_outputs; ++i) {
      outputs_.push_back(AddOutput(input.type));
    }
    SetBuiltinOp(BuiltinOperator_UNPACK, BuiltinOptions_UnpackOptions,
                 CreateUnpackOptions(builder_, num_outputs, axis).Union());
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

template <typename T>
void Check(int axis, const std::initializer_list<int>& input_shape,
           const std::initializer_list<T>& input_data,
           const std::vector<std::vector<int>>& exp_output_shape,
           const std::vector<std::vector<T>>& exp_output_data,
           const TensorType& type = TensorType_FLOAT32) {
  UnpackOpModel<T> m({type, input_shape}, axis);
  m.SetInput(input_data);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  // Check outputs shapes.
  EXPECT_THAT(m.GetOutputShapes(), ElementsAreArray(exp_output_shape));

  // Check outputs values.
  EXPECT_THAT(m.GetOutputDatas(), ElementsAreArray(exp_output_data));
}

template <typename InputType>
struct UnpackOpTest : public ::testing::Test {
  using TypeToTest = InputType;
  TensorType TENSOR_TYPE =
      (std::is_same<InputType, int16_t>::value
           ? TensorType_INT16
           : (std::is_same<InputType, uint8_t>::value
                  ? TensorType_UINT8
                  : (std::is_same<InputType, int8_t>::value
                         ? TensorType_INT8
                         : (std::is_same<InputType, int32_t>::value
                                ? TensorType_INT32
                                : TensorType_FLOAT32))));
};

using TestTypes = testing::Types<float, int32_t, int8_t, uint8_t, int16_t>;
TYPED_TEST_CASE(UnpackOpTest, TestTypes);

TYPED_TEST(UnpackOpTest, ThreeOutputs) {
  Check<typename TestFixture::TypeToTest>(
      /*axis=*/0, /*input_shape=*/{3, 2},
      /*input_data=*/{1, 2, 3, 4, 5, 6},
      /*exp_output_shape=*/{{2}, {2}, {2}},
      /*exp_output_data=*/{{1, 2}, {3, 4}, {5, 6}}, TestFixture::TENSOR_TYPE);
}

TYPED_TEST(UnpackOpTest, ThreeOutputsAxisOne) {
  Check<typename TestFixture::TypeToTest>(
      /*axis=*/1, /*input_shape=*/{3, 2},
      /*input_data=*/{1, 2, 3, 4, 5, 6},
      /*exp_output_shape=*/{{3}, {3}},
      /*exp_output_data=*/{{1, 3, 5}, {2, 4, 6}}, TestFixture::TENSOR_TYPE);
}

TYPED_TEST(UnpackOpTest, ThreeOutputsNegativeAxisOne) {
  Check<typename TestFixture::TypeToTest>(
      /*axis=*/-1, /*input_shape=*/{3, 2},
      /*input_data=*/{1, 2, 3, 4, 5, 6},
      /*exp_output_shape=*/{{3}, {3}},
      /*exp_output_data=*/{{1, 3, 5}, {2, 4, 6}}, TestFixture::TENSOR_TYPE);
}

TYPED_TEST(UnpackOpTest, OneOutput) {
  Check<typename TestFixture::TypeToTest>(
      /*axis=*/0, /*input_shape=*/{1, 6},
      /*input_data=*/{1, 2, 3, 4, 5, 6},
      /*exp_output_shape=*/{{6}},
      /*exp_output_data=*/{{1, 2, 3, 4, 5, 6}}, TestFixture::TENSOR_TYPE);
}

TYPED_TEST(UnpackOpTest, ThreeDimensionsOutputs) {
  Check<typename TestFixture::TypeToTest>(
      /*axis=*/2, /*input_shape=*/{2, 2, 2},
      /*input_data=*/{1, 2, 3, 4, 5, 6, 7, 8},
      /*exp_output_shape=*/{{2, 2}, {2, 2}},
      /*exp_output_data=*/{{1, 3, 5, 7}, {2, 4, 6, 8}},
      TestFixture::TENSOR_TYPE);
}

TYPED_TEST(UnpackOpTest, FiveDimensionsOutputs) {
  Check<typename TestFixture::TypeToTest>(
      /*axis=*/2, /*input_shape=*/{2, 2, 2, 2, 1},
      /*input_data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      /*exp_output_shape=*/{{2, 2, 2, 1}, {2, 2, 2, 1}},
      /*exp_output_data=*/
      {{1, 2, 5, 6, 9, 10, 13, 14}, {3, 4, 7, 8, 11, 12, 15, 16}},
      /*type=*/TestFixture::TENSOR_TYPE);
}

TYPED_TEST(UnpackOpTest, VectorToScalar) {
  Check<typename TestFixture::TypeToTest>(
      /*axis=*/0, /*input_shape=*/{5},
      /*input_data=*/{1, 2, 3, 4, 5},
      /*exp_output_shape=*/{{}, {}, {}, {}, {}},
      /*exp_output_data=*/{{1}, {2}, {3}, {4}, {5}}, TestFixture::TENSOR_TYPE);
}

// bool tests.
TEST(UnpackOpTestBool, BoolThreeOutputs) {
  Check<bool>(
      /*axis=*/0, /*input_shape=*/{3, 2},
      /*input_data=*/{true, false, true, false, true, false},
      /*exp_output_shape=*/{{2}, {2}, {2}},
      /*exp_output_data=*/{{true, false}, {true, false}, {true, false}},
      /*type=*/TensorType_BOOL);
}

TEST(UnpackOpTestBool, BoolThreeOutputsAxisOne) {
  Check<bool>(
      /*axis=*/1, /*input_shape=*/{3, 2},
      /*input_data=*/{true, false, true, false, true, false},
      /*exp_output_shape=*/{{3}, {3}},
      /*exp_output_data=*/{{true, true, true}, {false, false, false}},
      /*type=*/TensorType_BOOL);
}

TEST(UnpackOpTestBool, BoolThreeOutputsNegativeAxisOne) {
  Check<bool>(
      /*axis=*/-1, /*input_shape=*/{3, 2},
      /*input_data=*/{true, false, true, false, true, false},
      /*exp_output_shape=*/{{3}, {3}},
      /*exp_output_data=*/{{true, true, true}, {false, false, false}},
      /*type=*/TensorType_BOOL);
}

TEST(UnpackOpTestBool, BoolThreeOutputsNegativeAxisTwo) {
  Check<bool>(
      /*axis=*/-2, /*input_shape=*/{3, 2},
      /*input_data=*/{true, false, true, false, true, false},
      /*exp_output_shape=*/{{2}, {2}, {2}},
      /*exp_output_data=*/{{true, false}, {true, false}, {true, false}},
      /*type=*/TensorType_BOOL);
}

TEST(UnpackOpTestBool, BoolOneOutput) {
  Check<bool>(
      /*axis=*/0, /*input_shape=*/{1, 6},
      /*input_data=*/{true, false, true, false, true, false},
      /*exp_output_shape=*/{{6}},
      /*exp_output_data=*/{{true, false, true, false, true, false}},
      /*type=*/TensorType_BOOL);
}

TEST(UnpackOpTestBool, BoolThreeDimensionsOutputs) {
  Check<bool>(
      /*axis=*/2, /*input_shape=*/{2, 2, 2},
      /*input_data=*/{true, false, true, false, true, false, true, false},
      /*exp_output_shape=*/{{2, 2}, {2, 2}},
      /*exp_output_data=*/
      {{true, true, true, true}, {false, false, false, false}},
      /*type=*/TensorType_BOOL);
}

TEST(UnpackOpTest, BoolFiveDimensionsOutputs) {
  Check<bool>(
      /*axis=*/2, /*input_shape=*/{2, 2, 2, 2, 1},
      /*input_data=*/
      {true, false, true, false, true, false, true, false, true, true, true,
       true, true, true, true, true},
      /*exp_output_shape=*/{{2, 2, 2, 1}, {2, 2, 2, 1}},
      /*exp_output_data=*/
      {{true, false, true, false, true, true, true, true},
       {true, false, true, false, true, true, true, true}},
      /*type=*/TensorType_BOOL);
}

TEST(UnpackOpTestBool, BoolVectorToScalar) {
  Check<bool>(/*axis=*/0, /*input_shape=*/{5},
              /*input_data=*/{true, false, true, false, true},
              /*exp_output_shape=*/{{}, {}, {}, {}, {}},
              /*exp_output_data=*/{{true}, {false}, {true}, {false}, {true}},
              /*type=*/TensorType_BOOL);
}

}  // namespace
}  // namespace tflite
