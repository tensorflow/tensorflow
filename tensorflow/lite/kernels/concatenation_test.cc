/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BaseConcatenationOpModel : public SingleOpModel {
 public:
  // TODO(ahentz): Also test different activation types, axis, input
  // dimensions.
  BaseConcatenationOpModel() {}
  BaseConcatenationOpModel(const TensorData& input_template, int axis,
                           int num_inputs) {
    std::vector<std::vector<int>> all_input_shapes;
    for (int i = 0; i < num_inputs; ++i) {
      all_input_shapes.push_back(input_template.shape);
      AddInput(input_template);
    }
    output_ = AddOutput({input_template.type, /*shape=*/{}, input_template.min,
                         input_template.max});
    SetBuiltinOp(
        BuiltinOperator_CONCATENATION, BuiltinOptions_ConcatenationOptions,
        CreateConcatenationOptions(builder_, axis, ActivationFunctionType_NONE)
            .Union());
    BuildInterpreter(all_input_shapes);
  }

 protected:
  int output_;
};

class ConcatenationOpModel : public BaseConcatenationOpModel {
 public:
  using BaseConcatenationOpModel::BaseConcatenationOpModel;
  void SetInput(int index, std::initializer_list<float> data) {
    PopulateTensor(index, data);
  }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

class QuantizedConcatenationOpModel : public BaseConcatenationOpModel {
 public:
  using BaseConcatenationOpModel::BaseConcatenationOpModel;
  QuantizedConcatenationOpModel(const std::vector<TensorData>& input_template,
                                int axis, int num_inputs,
                                const TensorData& output_template) {
    std::vector<std::vector<int>> all_input_shapes;
    CHECK_EQ(input_template.size(), num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      all_input_shapes.push_back(input_template[i].shape);
      AddInput(input_template[i]);
    }
    output_ = AddOutput({output_template.type, /*shape=*/{},
                         output_template.min, output_template.max});
    SetBuiltinOp(
        BuiltinOperator_CONCATENATION, BuiltinOptions_ConcatenationOptions,
        CreateConcatenationOptions(builder_, axis, ActivationFunctionType_NONE)
            .Union());
    BuildInterpreter(all_input_shapes);
  }
  void SetInput(int index, std::initializer_list<float> data) {
    QuantizeAndPopulate<uint8_t>(index, data);
  }
  std::vector<uint8_t> GetOutput() { return ExtractVector<uint8_t>(output_); }
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }
};

TEST(ConcatenationOpTest, ThreeDimensionalOneInput) {
  ConcatenationOpModel m0({TensorType_FLOAT32, {2, 1, 2}}, /*axis=*/1,
                          /*num_inputs=*/1);
  m0.SetInput(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.Invoke();
  EXPECT_THAT(m0.GetOutput(), ElementsAreArray({1, 3, 4, 7}));
}

TEST(ConcatenationOpTest, OneTrivialInput) {
  ConcatenationOpModel m0({TensorType_FLOAT32, {1}}, /*axis=*/0,
                          /*num_inputs=*/1);
  m0.SetInput(0, {5.0f});
  m0.Invoke();
  EXPECT_THAT(m0.GetOutput(), ::testing::ElementsAre(5));
}

TEST(ConcatenationOpTest, TwoDimensionalOneInput) {
  ConcatenationOpModel m0({TensorType_FLOAT32, {2, 3}}, /*axis=*/0,
                          /*num_inputs=*/1);
  m0.SetInput(0, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  m0.Invoke();
  EXPECT_THAT(m0.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(ConcatenationOpTest, TwoInputsTwoAxesNegativeAxes) {
  // We will concatenate two tensors along different dimensions.
  auto tensor0 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto tensor1 = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  ConcatenationOpModel m0({TensorType_FLOAT32, {2, 3}}, /*axis=*/0,
                          /*num_inputs=*/2);
  m0.SetInput(0, tensor0);
  m0.SetInput(1, tensor1);
  m0.Invoke();
  EXPECT_THAT(m0.GetOutput(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));

  ConcatenationOpModel m0_negative({TensorType_FLOAT32, {2, 3}}, /*axis=*/-2,
                                   /*num_inputs=*/2);
  m0_negative.SetInput(0, tensor0);
  m0_negative.SetInput(1, tensor1);
  m0_negative.Invoke();
  EXPECT_THAT(m0_negative.GetOutput(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));

  ConcatenationOpModel m1({TensorType_FLOAT32, {2, 3}}, /*axis=*/1,
                          /*num_inputs=*/2);
  m1.SetInput(0, tensor0);
  m1.SetInput(1, tensor1);
  m1.Invoke();
  EXPECT_THAT(m1.GetOutput(),
              ElementsAreArray({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));

  ConcatenationOpModel m1_negative({TensorType_FLOAT32, {2, 3}}, /*axis=*/-1,
                                   /*num_inputs=*/2);
  m1_negative.SetInput(0, tensor0);
  m1_negative.SetInput(1, tensor1);
  m1_negative.Invoke();
  EXPECT_THAT(m1_negative.GetOutput(),
              ElementsAreArray({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));
}

TEST(ConcatenationOpTest, FourInputs) {
  ConcatenationOpModel m0({TensorType_FLOAT32, {2, 1, 2}}, /*axis=*/2,
                          /*num_inputs=*/4);
  m0.SetInput(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.SetInput(1, {1.1f, 3.1f, 4.1f, 7.1f});
  m0.SetInput(2, {1.2f, 3.2f, 4.2f, 7.2f});
  m0.SetInput(3, {1.3f, 3.3f, 4.3f, 7.3f});
  m0.Invoke();
  EXPECT_THAT(m0.GetOutput(),
              ElementsAreArray({
                  1.0f, 3.0f, 1.1f, 3.1f, 1.2f, 3.2f, 1.3f, 3.3f,  //
                  4.0f, 7.0f, 4.1f, 7.1f, 4.2f, 7.2f, 4.3f, 7.3f,  //
              }));
}

TEST(ConcatenationOpTest, FourInputsQuantized) {
  QuantizedConcatenationOpModel m0({TensorType_UINT8, {2, 1, 2}, -12.7, 12.8},
                                   /*axis=*/2,
                                   /*num_inputs=*/4);

  m0.SetInput(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.SetInput(1, {1.1f, 3.1f, 4.1f, 7.1f});
  m0.SetInput(2, {1.2f, 3.2f, 4.2f, 7.2f});
  m0.SetInput(3, {1.3f, 3.3f, 4.3f, 7.3f});
  m0.Invoke();
  EXPECT_THAT(m0.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({
                  1.0f, 3.0f, 1.1f, 3.1f, 1.2f, 3.2f, 1.3f, 3.3f,  //
                  4.0f, 7.0f, 4.1f, 7.1f, 4.2f, 7.2f, 4.3f, 7.3f,  //
              })));
  EXPECT_THAT(m0.GetOutput(), ElementsAreArray({
                                  137, 157, 138, 158, 139, 159, 140, 160,  //
                                  167, 197, 168, 198, 169, 199, 170, 200,  //
                              }));
}

TEST(ConcatenationOpTest, FourInputsQuantizedMixedRange) {
  QuantizedConcatenationOpModel m0({{TensorType_UINT8, {2, 1, 2}, -10.7, 10.8},
                                    {TensorType_UINT8, {2, 1, 2}, 0, 12.8},
                                    {TensorType_UINT8, {2, 1, 2}, -11, 11.8},
                                    {TensorType_UINT8, {2, 1, 2}, 0, 7.4}},
                                   /*axis=*/2, /*num_inputs=*/4,
                                   {TensorType_UINT8, {2, 1, 2}, -12.7, 12.8});

  m0.SetInput(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.SetInput(1, {1.1f, 3.1f, 4.1f, 7.1f});
  m0.SetInput(2, {1.2f, 3.2f, 4.2f, 7.2f});
  m0.SetInput(3, {1.3f, 3.3f, 4.3f, 7.3f});
  m0.Invoke();
  EXPECT_THAT(m0.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({
                  1.0f, 3.0f, 1.1f, 3.1f, 1.2f, 3.2f, 1.3f, 3.3f,  //
                  4.0f, 7.0f, 4.1f, 7.1f, 4.2f, 7.2f, 4.3f, 7.3f,  //
              })));
  EXPECT_THAT(m0.GetOutput(), ElementsAreArray({
                                  137, 157, 138, 158, 139, 159, 140, 160,  //
                                  167, 197, 168, 198, 169, 199, 170, 200,  //
                              }));
}

TEST(ConcatenationOpTest, FourInputsQuantizedMixedRangeClampingLogic) {
  QuantizedConcatenationOpModel m0({{TensorType_UINT8, {2, 1, 2}, -10.7, 10.8},
                                    {TensorType_UINT8, {2, 1, 2}, 0, 12.8},
                                    {TensorType_UINT8, {2, 1, 2}, -11, 11.8},
                                    {TensorType_UINT8, {2, 1, 2}, 0, 7.4}},
                                   /*axis=*/2, /*num_inputs=*/4,
                                   {TensorType_UINT8, {2, 1, 2}, -1., 1.});

  m0.SetInput(0, {1.0f, -3.0f, -4.0f, -7.0f});
  m0.SetInput(1, {1.1f, 3.1f, 4.1f, 7.1f});
  m0.SetInput(2, {1.2f, -3.2f, -4.2f, 7.2f});
  m0.SetInput(3, {1.3f, 3.3f, 4.3f, 7.3f});
  m0.Invoke();
  EXPECT_THAT(m0.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f,   //
                      -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,  //
                  },
                  4e-3)));
  EXPECT_THAT(m0.GetOutput(), ElementsAreArray({
                                  255, 0, 255, 255, 255, 0, 255, 255,  //
                                  0, 0, 255, 255, 0, 255, 255, 255,    //
                              }));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
