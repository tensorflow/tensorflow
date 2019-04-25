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
  m.Invoke();

  // Check outputs shapes.
  EXPECT_THAT(m.GetOutputShapes(), ElementsAreArray(exp_output_shape));

  // Check outputs values.
  EXPECT_THAT(m.GetOutputDatas(), ElementsAreArray(exp_output_data));
}

// float32 tests.
TEST(UnpackOpTest, FloatThreeOutputs) {
  Check<float>(/*axis=*/0, /*input_shape=*/{3, 2},
               /*input_data=*/{1, 2, 3, 4, 5, 6},
               /*expected_output_shape=*/{{2}, {2}, {2}},
               /*expected_output_data=*/{{1, 2}, {3, 4}, {5, 6}});
}

TEST(UnpackOpTest, FloatThreeOutputsAxisOne) {
  Check<float>(/*axis=*/1, /*input_shape=*/{3, 2},
               /*input_data=*/{1, 2, 3, 4, 5, 6},
               /*expected_output_shape=*/{{3}, {3}},
               /*expected_output_data=*/{{1, 3, 5}, {2, 4, 6}});
}

TEST(UnpackOpTest, FloatThreeOutputsNegativeAxisOne) {
  Check<float>(/*axis=*/-1, /*input_shape=*/{3, 2},
               /*input_data=*/{1, 2, 3, 4, 5, 6},
               /*expected_output_shape=*/{{3}, {3}},
               /*expected_output_data=*/{{1, 3, 5}, {2, 4, 6}});
}

TEST(UnpackOpTest, FloatThreeOutputsNegativeAxisTwo) {
  Check<float>(/*axis=*/-2, /*input_shape=*/{3, 2},
               /*input_data=*/{1, 2, 3, 4, 5, 6},
               /*expected_output_shape=*/{{2}, {2}, {2}},
               /*expected_output_data=*/{{1, 2}, {3, 4}, {5, 6}});
}

TEST(UnpackOpTest, FloatOneOutput) {
  Check<float>(/*axis=*/0, /*input_shape=*/{1, 6},
               /*input_data=*/{1, 2, 3, 4, 5, 6},
               /*expected_output_shape=*/{{6}},
               /*expected_output_data=*/{{1, 2, 3, 4, 5, 6}});
}

TEST(UnpackOpTest, FloatThreeDimensionsOutputs) {
  Check<float>(/*axis=*/2, /*input_shape=*/{2, 2, 2},
               /*input_data=*/{1, 2, 3, 4, 5, 6, 7, 8},
               /*expected_output_shape=*/{{2, 2}, {2, 2}},
               /*expected_output_data=*/{{1, 3, 5, 7}, {2, 4, 6, 8}});
}

// int32 tests.
TEST(UnpackOpTest, IntThreeOutputs) {
  Check<int32_t>(/*axis=*/0, /*input_shape=*/{3, 2},
                 /*input_data=*/{1, 2, 3, 4, 5, 6},
                 /*expected_output_shape=*/{{2}, {2}, {2}},
                 /*expected_output_data=*/{{1, 2}, {3, 4}, {5, 6}},
                 /*type=*/TensorType_INT32);
}

TEST(UnpackOpTest, IntThreeOutputsAxisOne) {
  Check<int32_t>(/*axis=*/1, /*input_shape=*/{3, 2},
                 /*input_data=*/{1, 2, 3, 4, 5, 6},
                 /*expected_output_shape=*/{{3}, {3}},
                 /*expected_output_data=*/{{1, 3, 5}, {2, 4, 6}},
                 /*type=*/TensorType_INT32);
}

TEST(UnpackOpTest, IntOneOutput) {
  Check<int32_t>(/*axis=*/0, /*input_shape=*/{1, 6},
                 /*input_data=*/{1, 2, 3, 4, 5, 6},
                 /*expected_output_shape=*/{{6}},
                 /*expected_output_data=*/{{1, 2, 3, 4, 5, 6}},
                 /*type=*/TensorType_INT32);
}

TEST(UnpackOpTest, IntThreeDimensionsOutputs) {
  Check<int32_t>(/*axis=*/2, /*input_shape=*/{2, 2, 2},
                 /*input_data=*/{1, 2, 3, 4, 5, 6, 7, 8},
                 /*expected_output_shape=*/{{2, 2}, {2, 2}},
                 /*expected_output_data=*/{{1, 3, 5, 7}, {2, 4, 6, 8}},
                 /*type=*/TensorType_INT32);
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
