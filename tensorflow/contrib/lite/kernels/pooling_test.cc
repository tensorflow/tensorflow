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
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BasePoolingOpModel : public SingleOpModel {
 public:
  // TODO(ahentz): Also test different activation types, bias, padding types,
  // stride values.
  BasePoolingOpModel(BuiltinOperator type, const TensorData& input,
                     int filter_width, int filter_height,
                     const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);

    SetBuiltinOp(
        type, BuiltinOptions_Pool2DOptions,
        CreatePool2DOptions(builder_, Padding_VALID, 2, 2, filter_width,
                            filter_height, ActivationFunctionType_NONE)
            .Union());

    BuildInterpreter({GetShape(input_)});
  }

 protected:
  int input_;
  int output_;
};

class FloatPoolingOpModel : public BasePoolingOpModel {
 public:
  using BasePoolingOpModel::BasePoolingOpModel;

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

class QuantizedPoolingOpModel : public BasePoolingOpModel {
 public:
  using BasePoolingOpModel::BasePoolingOpModel;

  void SetInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<uint8_t>(input_, data);
  }

  std::vector<uint8_t> GetOutput() { return ExtractVector<uint8_t>(output_); }
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }
};

TEST(FloatPoolingOpTest, AveragePool) {
  FloatPoolingOpModel m(BuiltinOperator_AVERAGE_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}});
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2.75, 5.75}));
}

TEST(QuantizedPoolingOpTest, AveragePool) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  QuantizedPoolingOpModel m(
      BuiltinOperator_AVERAGE_POOL_2D,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, 0, 15.9375},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_UINT8, {}, 0, 15.9375});
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({2.75, 5.75})));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({44, 92}));
}

TEST(FloatPoolingOpTest, MaxPool) {
  FloatPoolingOpModel m(BuiltinOperator_MAX_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}});
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({6, 10}));
}

TEST(QuantizedPoolingOpTest, MaxPool) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  QuantizedPoolingOpModel m(
      BuiltinOperator_MAX_POOL_2D,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, 0, 15.9375},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_UINT8, {}, 0, 15.9375});
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({6, 10})));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({96, 160}));
}

TEST(FloatPoolingOpTest, L2Pool) {
  FloatPoolingOpModel m(BuiltinOperator_L2_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}});
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3.5, 6.5}));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
