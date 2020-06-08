/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BaseAddNOpModel : public SingleOpModel {
 public:
  BaseAddNOpModel(const std::vector<TensorData>& inputs,
                  const TensorData& output) {
    int num_inputs = inputs.size();
    std::vector<std::vector<int>> input_shapes;

    for (int i = 0; i < num_inputs; ++i) {
      inputs_.push_back(AddInput(inputs[i]));
      input_shapes.push_back(GetShape(inputs_[i]));
    }

    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_ADD_N, BuiltinOptions_AddNOptions,
                 CreateAddNOptions(builder_).Union());
    BuildInterpreter(input_shapes);
  }

  int input(int i) { return inputs_[i]; }

 protected:
  std::vector<int> inputs_;
  int output_;
};

class FloatAddNOpModel : public BaseAddNOpModel {
 public:
  using BaseAddNOpModel::BaseAddNOpModel;

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

class IntegerAddNOpModel : public BaseAddNOpModel {
 public:
  using BaseAddNOpModel::BaseAddNOpModel;

  std::vector<int32_t> GetOutput() { return ExtractVector<int32_t>(output_); }
};

TEST(FloatAddNOpModel, AddMultipleTensors) {
  FloatAddNOpModel m({{TensorType_FLOAT32, {1, 2, 2, 1}},
                      {TensorType_FLOAT32, {1, 2, 2, 1}},
                      {TensorType_FLOAT32, {1, 2, 2, 1}}},
                     {TensorType_FLOAT32, {}});
  m.PopulateTensor<float>(m.input(0), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input(1), {0.1, 0.2, 0.3, 0.5});
  m.PopulateTensor<float>(m.input(2), {0.5, 0.1, 0.1, 0.2});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1.4, 0.5, 1.1, 1.5}));
}

TEST(IntegerAddNOpModel, AddMultipleTensors) {
  IntegerAddNOpModel m({{TensorType_INT32, {1, 2, 2, 1}},
                        {TensorType_INT32, {1, 2, 2, 1}},
                        {TensorType_INT32, {1, 2, 2, 1}}},
                       {TensorType_INT32, {}});
  m.PopulateTensor<int32_t>(m.input(0), {-20, 2, 7, 8});
  m.PopulateTensor<int32_t>(m.input(1), {1, 2, 3, 5});
  m.PopulateTensor<int32_t>(m.input(2), {10, -5, 1, -2});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-9, -1, 11, 11}));
}

}  // namespace
}  // namespace tflite
