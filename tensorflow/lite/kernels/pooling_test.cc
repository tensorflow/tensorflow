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

class BasePoolingOpModel : public SingleOpModel {
 public:
  BasePoolingOpModel(
      BuiltinOperator type, const TensorData& input, int filter_width,
      int filter_height, const TensorData& output,
      Padding padding = Padding_VALID, int stride_w = 2, int stride_h = 2,
      ActivationFunctionType activation = ActivationFunctionType_NONE) {
    input_ = AddInput(input);
    output_ = AddOutput(output);

    SetBuiltinOp(type, BuiltinOptions_Pool2DOptions,
                 CreatePool2DOptions(builder_, padding, stride_w, stride_h,
                                     filter_width, filter_height, activation)
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

  void SetInput(const std::vector<float>& data) {
    QuantizeAndPopulate<uint8_t>(input_, data);
  }

  std::vector<uint8_t> GetOutput() { return ExtractVector<uint8_t>(output_); }
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }
};

class SymmetricQuantizedPoolingOpModel : public BasePoolingOpModel {
 public:
  using BasePoolingOpModel::BasePoolingOpModel;

  void SetInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<int8_t>(input_, data);
  }

  void SetInput(const std::vector<float>& data) {
    QuantizeAndPopulate<int8_t>(input_, data);
  }

  std::vector<int8_t> GetOutput() { return ExtractVector<int8_t>(output_); }
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<int8_t>(ExtractVector<int8_t>(output_), GetScale(output_),
                              GetZeroPoint(output_));
  }
};

class SymmetricQuantizedPoolingOpModel16 : public BasePoolingOpModel {
 public:
  using BasePoolingOpModel::BasePoolingOpModel;

  void SetInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<int16_t>(input_, data);
  }

  void SetInput(const std::vector<float>& data) {
    QuantizeAndPopulate<int16_t>(input_, data);
  }

  std::vector<int16_t> GetOutput() { return ExtractVector<int16_t>(output_); }
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<int16_t>(ExtractVector<int16_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }
};

// Replicate each entry in a vector n times along depth (innermost dimension).
// The values are incremented by delta, creating ramps offset by each input
// value. This is used to create simple and predicatable variation.
std::vector<float> ReplicateDepthRamp(const std::vector<float>& image_plane,
                                      int n, float delta) {
  const int size = image_plane.size();
  std::vector<float> ramped_data(n * size);
  // The input is treated as a 1-D even if logically it is multi-dimensional.
  for (int input_index = 0; input_index < size; ++input_index) {
    for (int depth = 0; depth < n; ++depth) {
      ramped_data[n * input_index + depth] =
          image_plane[input_index] + depth * delta;
    }
  }

  return ramped_data;
}

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

TEST(FloatPoolingOpTest, AveragePoolActivationRelu) {
  FloatPoolingOpModel m(BuiltinOperator_AVERAGE_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}}, Padding_VALID, 2,
                        2, ActivationFunctionType_RELU);
  m.SetInput({
      0, -6, 2, 4,   //
      3, 2, -10, 7,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0.0, 0.75}));
}

TEST(FloatPoolingOpTest, AveragePoolActivationRelu1) {
  FloatPoolingOpModel m(BuiltinOperator_AVERAGE_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}}, Padding_VALID, 2,
                        2, ActivationFunctionType_RELU_N1_TO_1);
  m.SetInput({
      0, -6, 2, 4,     //
      -3, -2, -10, 7,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1.0, 0.75}));

  m.SetInput({
      0, -6, -2, -4,   //
      -3, -2, 10, -7,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1.0, -0.75}));
}

TEST(FloatPoolingOpTest, AveragePoolActivationRelu6) {
  FloatPoolingOpModel m(BuiltinOperator_AVERAGE_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}}, Padding_VALID, 2,
                        2, ActivationFunctionType_RELU6);
  m.SetInput({
      0, -6, 12, 4,   //
      -3, -2, 10, 7,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0.0, 6.0}));

  m.SetInput({
      0, 6, 12, 4,  //
      3, 2, 10, 7,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2.75, 6.0}));
}

TEST(FloatPoolingOpTest, AveragePoolPaddingSameStride1) {
  FloatPoolingOpModel m(BuiltinOperator_AVERAGE_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}}, Padding_SAME, 1,
                        1);
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({2.75, 5.0, 5.75, 5.5, 2.5, 6.0, 8.5, 7.0}));
}

TEST(FloatPoolingOpTest, AveragePoolPaddingValidStride1) {
  FloatPoolingOpModel m(BuiltinOperator_AVERAGE_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}}, Padding_VALID, 1,
                        1);
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2.75, 5.0, 5.75}));
}

TEST(QuantizedPoolingOpTest, AveragePool) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[0, 15.9375] --> [Scale{0.0625}, zero_point{0}]
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

TEST(QuantizedPoolingOpTest, AveragePoolActivationRelu) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[-15.9375, 15.9375] --> [Scale{0.125}, zero_point{128}]
  QuantizedPoolingOpModel m(
      BuiltinOperator_AVERAGE_POOL_2D,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, -15.9375, 15.9375},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_UINT8, {}, -15.9375, 15.9375}, Padding_VALID, 2, 2,
      ActivationFunctionType_RELU);
  m.SetInput({
      0, -6, 2, 4,   //
      3, 2, -10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({0.0, 0.75})));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({128, 134}));
}

TEST(QuantizedPoolingOpTest, AveragePoolActivationRelu1) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[-15.9375, 15.9375] --> [Scale{0.125}, zero_point{128}]
  QuantizedPoolingOpModel m(
      BuiltinOperator_AVERAGE_POOL_2D,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, -15.9375, 15.9375},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_UINT8, {}, -15.9375, 15.9375}, Padding_VALID, 2, 2,
      ActivationFunctionType_RELU_N1_TO_1);
  m.SetInput({
      0, -6, 2, 4,     //
      -3, -2, -10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({-1.0, 0.75})));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({120, 134}));

  m.SetInput({
      0, -6, -2, -4,   //
      -3, -2, 10, -7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({-1.0, -0.75})));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({120, 122}));
}

TEST(QuantizedPoolingOpTest, AveragePoolActivationRelu6) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[-15.9375, 15.9375] --> [Scale{0.125}, zero_point{128}]
  QuantizedPoolingOpModel m(
      BuiltinOperator_AVERAGE_POOL_2D,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, -15.9375, 15.9375},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_UINT8, {}, -15.9375, 15.9375}, Padding_VALID, 2, 2,
      ActivationFunctionType_RELU6);
  m.SetInput({
      0, -6, 12, 4,   //
      -3, -2, 10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({0.0, 6.0})));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({128, 176}));

  m.SetInput({
      0, 6, 12, 4,  //
      3, 2, 10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({2.75, 6.0})));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({150, 176}));
}

TEST(QuantizedPoolingOpTest, AveragePoolPaddingSameStride1) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[0, 15.9375] --> [Scale{0.0625}, zero_point{0}]
  QuantizedPoolingOpModel m(
      BuiltinOperator_AVERAGE_POOL_2D,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, 0, 15.9375},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_UINT8, {}, 0, 15.9375}, Padding_SAME, 1, 1);
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(
                  ArrayFloatNear({2.75, 5.0, 5.75, 5.5, 2.5, 6.0, 8.5, 7.0})));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({44, 80, 92, 88, 40, 96, 136, 112}));
}

TEST(QuantizedPoolingOpTest, AveragePoolPaddingValidStride1) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[0, 15.9375] --> [Scale{0.0625}, zero_point{0}]
  QuantizedPoolingOpModel m(
      BuiltinOperator_AVERAGE_POOL_2D,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, 0, 15.9375},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_UINT8, {}, 0, 15.9375}, Padding_VALID, 1, 1);
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({2.75, 5.0, 5.75})));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({44, 80, 92}));
}
// Send in a white image, expect a white pixel.
TEST(QuantizedPoolingOpTest, AveragePoolImageSize16) {
  int image_size = 16;
  QuantizedPoolingOpModel m(
      BuiltinOperator_AVERAGE_POOL_2D,
      /*input=*/{TensorType_UINT8, {1, image_size, image_size, 1}, 0, 16},
      /*filter_width=*/image_size,
      /*filter_height=*/image_size,
      /*output=*/{TensorType_UINT8, {}, 0, 16});

  std::vector<float> input(image_size * image_size, 16.f);
  m.SetInput(input);
  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ::testing::ElementsAre(255));
  EXPECT_THAT(m.GetDequantizedOutput(), ElementsAreArray(ArrayFloatNear({16})));
}

TEST(QuantizedPoolingOpTest, AveragePoolLargeDepth) {
  // Test with a larger depth that is not a multiple of the tranche size, or of
  // any register-oriented multiples such as 8 and 16.
  constexpr int depth = 1999;  // Prime number.
  QuantizedPoolingOpModel m(
      BuiltinOperator_AVERAGE_POOL_2D,
      /*input=*/{TensorType_UINT8, {1, 2, 4, depth}, 0, 15.9375},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_UINT8, {}, 0, 15.9375});

  std::vector<float> input_image_plane({
      0.f, 6.f, 2.f, 4.f,   //
      3.f, 2.f, 10.f, 7.f,  //
  });
  std::vector<float> output_image_plane({2.75f, 5.75f});

  m.SetInput(ReplicateDepthRamp(input_image_plane, depth, 1.f / 512.f));
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(
                  ReplicateDepthRamp(output_image_plane, depth, 1.f / 512.f),
                  1. / 32.f)));
}

// Test quantized AveragePool with int16 input and output. The input is the same
// as the uint8 test QuantizedPoolingOpTest.AveragePool but with a scale of
// 1/4096 rather than 1/16.
TEST(QuantizedPoolingOpTest, SymmetricAveragePool16) {
  const float ulp = 1.f / 4096.f;
  SymmetricQuantizedPoolingOpModel16 m(
      BuiltinOperator_AVERAGE_POOL_2D,
      /*input=*/{TensorType_INT16, {1, 2, 4, 1}, 0, 16 - ulp},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_INT16, {}, 0, 16 - ulp});
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({2.75, 5.75})));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({(44 - 128) * 256, (92 - 128) * 256}));
}

// Test quantized AveragePool with int8 input and output. The input is the same
// as the uint8 test QuantizedPoolingOpTest.AveragePool. The float output is
// identical to uint8 test and quantized output is identical to uint8 test with
// a 128 shift.
TEST(QuantizedPoolingOpTest, SymmetricAveragePool) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[0, 15.9375] --> [Scale{0.0625}, zero_point{-128}]
  SymmetricQuantizedPoolingOpModel m(
      BuiltinOperator_AVERAGE_POOL_2D,
      /*input=*/{TensorType_INT8, {1, 2, 4, 1}, 0, 15.9375},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_INT8, {}, 0, 15.9375});
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({2.75, 5.75})));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({44 - 128, 92 - 128}));
}
// Test quantized AveragePool with int8 input and output. The input is the same
// as the uint8 test QuantizedPoolingOpTest.AveragePool. The float output is
// identical to uint8 test and quantized output is identical to uint8 test with
// a 128 shift.
TEST(QuantizedPoolingOpTest, SymmetricAveragePoolActivationRelu) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[-15.9375, 15.8130] --> [Scale{0.124512}, zero_point{0}]
  SymmetricQuantizedPoolingOpModel m(
      BuiltinOperator_AVERAGE_POOL_2D,
      /*input=*/{TensorType_INT8, {1, 2, 4, 1}, -15.9375, 15.8130},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_INT8, {}, -15.9375, 15.8130}, Padding_VALID, 2, 2,
      ActivationFunctionType_RELU);
  m.SetInput({
      0, -6, 2, 4,   //
      3, 2, -10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({0.0, 0.75}, 0.0030)));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({128 - 128, 134 - 128}));
}

TEST(QuantizedPoolingOpTest, SymmetricAveragePoolActivationRelu1) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[-15.9375, 15.8130] --> [Scale{0.124512}, zero_point{0}]
  SymmetricQuantizedPoolingOpModel m(
      BuiltinOperator_AVERAGE_POOL_2D,
      /*input=*/{TensorType_INT8, {1, 2, 4, 1}, -15.9375, 15.8130},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_INT8, {}, -15.9375, 15.8130}, Padding_VALID, 2, 2,
      ActivationFunctionType_RELU_N1_TO_1);
  m.SetInput({
      0, -6, 2, 4,     //
      -3, -2, -10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({-1.0, 0.75}, 0.0040)));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({120 - 128, 134 - 128}));

  m.SetInput({
      0, -6, -2, -4,   //
      -3, -2, 10, -7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({-1.0, -0.75}, 0.0040)));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({120 - 128, 122 - 128}));
}
// Test quantized AveragePool with int8 input and output. The input is the same
// as the uint8 test QuantizedPoolingOpTest.AveragePool. The float output is
// identical to uint8 test and quantized output is identical to uint8 test with
// a 128 shift.
TEST(QuantizedPoolingOpTest, SymmetricAveragePoolActivationRelu6) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[-15.9375, 15.8130] --> [Scale{0.124512}, zero_point{0}]
  SymmetricQuantizedPoolingOpModel m(
      BuiltinOperator_AVERAGE_POOL_2D,
      /*input=*/{TensorType_INT8, {1, 2, 4, 1}, -15.9375, 15.8130},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_INT8, {}, -15.9375, 15.8130}, Padding_VALID, 2, 2,
      ActivationFunctionType_RELU6);
  m.SetInput({
      0, -6, 12, 4,   //
      -3, -2, 10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({0.0, 6.0}, 0.025)));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({128 - 128, 176 - 128}));

  m.SetInput({
      0, 6, 12, 4,  //
      3, 2, 10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({2.75, 6.0}, 0.025)));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({150 - 128, 176 - 128}));
}

TEST(QuantizedPoolingOpTest, SymmetricAveragePoolPaddingSameStride1) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[0, 15.9375] --> [Scale{0.0625}, zero_point{-128}]
  SymmetricQuantizedPoolingOpModel m(
      BuiltinOperator_AVERAGE_POOL_2D,
      /*input=*/{TensorType_INT8, {1, 2, 4, 1}, 0, 15.9375},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_INT8, {}, 0, 15.9375}, Padding_SAME, 1, 1);
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(
                  ArrayFloatNear({2.75, 5.0, 5.75, 5.5, 2.5, 6.0, 8.5, 7.0})));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({44 - 128, 80 - 128, 92 - 128, 88 - 128,
                                40 - 128, 96 - 128, 136 - 128, 112 - 128}));
}

TEST(QuantizedPoolingOpTest, SymmetricAveragePoolPaddingValidStride1) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[0, 15.9375] --> [Scale{0.0625}, zero_point{-128}]
  SymmetricQuantizedPoolingOpModel m(
      BuiltinOperator_AVERAGE_POOL_2D,
      /*input=*/{TensorType_INT8, {1, 2, 4, 1}, 0, 15.9375},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_INT8, {}, 0, 15.9375}, Padding_VALID, 1, 1);
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({2.75, 5.0, 5.75})));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({44 - 128, 80 - 128, 92 - 128}));
}

// This is not accelerated because the filter window is too large
// Send in a white image and expect a white pixel.
TEST(QuantizedPoolingOpTest, AveragePoolImageSize17) {
  int image_size = 17;
  QuantizedPoolingOpModel m(
      BuiltinOperator_AVERAGE_POOL_2D,
      /*input=*/{TensorType_UINT8, {1, image_size, image_size, 1}, 0, 16},
      /*filter_width=*/image_size,
      /*filter_height=*/image_size,
      /*output=*/{TensorType_UINT8, {}, 0, 16});

  std::vector<float> input(image_size * image_size, 16.f);
  m.SetInput(input);
  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ::testing::ElementsAre(255));
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

TEST(FloatPoolingOpTest, MaxPoolActivationRelu) {
  FloatPoolingOpModel m(BuiltinOperator_MAX_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}}, Padding_VALID, 2,
                        2, ActivationFunctionType_RELU);
  m.SetInput({
      -1, -6, 2, 4,     //
      -3, -2, 10.5, 7,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0.0, 10.5}));
}

TEST(FloatPoolingOpTest, MaxPoolActivationRelu1) {
  FloatPoolingOpModel m(BuiltinOperator_MAX_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}}, Padding_VALID, 2,
                        2, ActivationFunctionType_RELU_N1_TO_1);
  m.SetInput({
      -2.75, -6, 0.2, 0.4,  //
      -3, -2, -0.3, 0.7,    //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1.0, 0.7}));

  m.SetInput({
      -2.75, -6, -2, -4,  //
      -3, -2, 10, -7,     //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1.0, 1.0}));
}

TEST(FloatPoolingOpTest, MaxPoolActivationRelu6) {
  FloatPoolingOpModel m(BuiltinOperator_MAX_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}}, Padding_VALID, 2,
                        2, ActivationFunctionType_RELU6);
  m.SetInput({
      -1.5, -6, 12, 4,  //
      -3, -2, 10, 7,    //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0.0, 6.0}));

  m.SetInput({
      0, 4.5, 12, 4,  //
      3, 2, 10, 7,    //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({4.5, 6.0}));
}

TEST(FloatPoolingOpTest, MaxPoolPaddingSameStride1) {
  FloatPoolingOpModel m(BuiltinOperator_MAX_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}}, Padding_SAME, 1,
                        1);
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({6, 10, 10, 7, 3, 10, 10, 7}));
}

TEST(FloatPoolingOpTest, MaxPoolPaddingValidStride1) {
  FloatPoolingOpModel m(BuiltinOperator_MAX_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}}, Padding_VALID, 1,
                        1);
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({6, 10, 10}));
}

TEST(QuantizedUInt8PoolingOpTest, MaxPool) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[0, 15.9375] --> [Scale{0.0625}, zero_point{0}]
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

TEST(QuantizedUInt8PoolingOpTest, MaxPoolActivationRelu) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[-15.9375, 15.9375] --> [Scale{0.125}, zero_point{128}]
  QuantizedPoolingOpModel m(
      BuiltinOperator_MAX_POOL_2D,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, -15.9375, 15.9375},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_UINT8, {}, -15.9375, 15.9375}, Padding_VALID, 2, 2,
      ActivationFunctionType_RELU);
  m.SetInput({
      -1.5, -6, 2, 4,  //
      -3, -2, 10, 7,   //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({0, 10})));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({128, 208}));
}

TEST(QuantizedUInt8PoolingOpTest, MaxPoolActivationRelu1) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[-15.9375, 15.9375] --> [Scale{0.125}, zero_point{128}]
  QuantizedPoolingOpModel m(
      BuiltinOperator_MAX_POOL_2D,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, -15.9375, 15.9375},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_UINT8, {}, -15.9375, 15.9375}, Padding_VALID, 2, 2,
      ActivationFunctionType_RELU_N1_TO_1);
  m.SetInput({
      -1.7, -6, 2, 4,  //
      -3, -2, -10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({-1.0, 1.0})));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({120, 136}));

  m.SetInput({
      0, -6, -0.2, -0.4,    //
      -3, -2, 0.75, -0.99,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({0.0, 0.75})));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({128, 134}));
}

TEST(QuantizedUInt8PoolingOpTest, MaxPoolActivationRelu6) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[-15.9375, 15.9375] --> [Scale{0.125}, zero_point{128}]
  QuantizedPoolingOpModel m(
      BuiltinOperator_MAX_POOL_2D,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, -15.9375, 15.9375},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_UINT8, {}, -15.9375, 15.9375}, Padding_VALID, 2, 2,
      ActivationFunctionType_RELU6);
  m.SetInput({
      0, -6, 12, 4,   //
      -3, -2, 10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({0.0, 6.0})));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({128, 176}));

  m.SetInput({
      0, 4.5, 12, 4,  //
      3, 2, 10, 7,    //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({4.5, 6.0})));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({164, 176}));
}

TEST(QuantizedUInt8PoolingOpTest, MaxPoolPaddingSameStride1) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[0, 15.9375] --> [Scale{0.0625}, zero_point{0}]
  QuantizedPoolingOpModel m(
      BuiltinOperator_MAX_POOL_2D,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, 0, 15.9375},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_UINT8, {}, 0, 15.9375}, Padding_SAME, 1, 1);
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({6, 10, 10, 7, 3, 10, 10, 7})));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({96, 160, 160, 112, 48, 160, 160, 112}));
}

TEST(QuantizedUInt8PoolingOpTest, MaxPoolPaddingValidStride1) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[0, 15.9375] --> [Scale{0.0625}, zero_point{0}]
  QuantizedPoolingOpModel m(
      BuiltinOperator_MAX_POOL_2D,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, 0, 15.9375},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_UINT8, {}, 0, 15.9375}, Padding_VALID, 1, 1);
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({6, 10, 10})));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({96, 160, 160}));
}

TEST(QuantizedPoolingOpTest, MaxPoolLargeDepth) {
  // Test with a larger depth that is not a multiple of the tranche size, or of
  // any register-oriented multiples such as 8 and 16.
  constexpr int depth = 1999;  // Prime number.
  QuantizedPoolingOpModel m(
      BuiltinOperator_MAX_POOL_2D,
      /*input=*/{TensorType_UINT8, {1, 2, 4, depth}, 0, 15.9375},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_UINT8, {}, 0, 15.9375});

  std::vector<float> input_image_plane({
      0.f, 6.f, 2.f, 4.f,   //
      3.f, 2.f, 10.f, 7.f,  //
  });
  std::vector<float> output_image_plane({6.f, 10.f});

  m.SetInput(ReplicateDepthRamp(input_image_plane, depth, 1.f / 512.f));
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(
                  ReplicateDepthRamp(output_image_plane, depth, 1.f / 512.f),
                  1. / 32.f)));
}

TEST(QuantizedInt8PoolingOpTest, MaxPool) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[0, 15.9375] --> [Scale{0.0625}, zero_point{-128}]
  SymmetricQuantizedPoolingOpModel m(
      BuiltinOperator_MAX_POOL_2D,
      /*input=*/{TensorType_INT8, {1, 2, 4, 1}, 0, 15.9375},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_INT8, {}, 0, 15.9375});
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({6, 10})));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({96 - 128, 160 - 128}));
}

TEST(QuantizedInt8PoolingOpTest16, MaxPool) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[0, 16-(1/4096)] --> [Scale{(1/4096)}, zero_point{-32768}]
  const float ulp = 1.f / 4096.f;
  SymmetricQuantizedPoolingOpModel16 m(
      BuiltinOperator_MAX_POOL_2D,
      /*input=*/{TensorType_INT16, {1, 2, 4, 1}, 0, 16 - ulp},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_INT16, {}, 0, 16 - ulp});
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({6, 10})));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({(96 - 128) * 256, (160 - 128) * 256}));
}

TEST(QuantizedInt8PoolingOpTest, MaxPoolActivationRelu) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[-15.9375, 15.8130] --> [Scale{0.124512}, zero_point{0}]
  SymmetricQuantizedPoolingOpModel m(
      BuiltinOperator_MAX_POOL_2D,
      /*input=*/{TensorType_INT8, {1, 2, 4, 1}, -15.9375, 15.8130},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_INT8, {}, -15.9375, 15.8130}, Padding_VALID, 2, 2,
      ActivationFunctionType_RELU);
  m.SetInput({
      -1.5, -6, 2, 4,  //
      -3, -2, 10, 7,   //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({0, 10}, 0.04)));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({128 - 128, 208 - 128}));
}

TEST(QuantizedInt8PoolingOpTest, MaxPoolActivationRelu1) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[-15.9375, 15.8130] --> [Scale{0.124512}, zero_point{0}]
  SymmetricQuantizedPoolingOpModel m(
      BuiltinOperator_MAX_POOL_2D,
      /*input=*/{TensorType_INT8, {1, 2, 4, 1}, -15.9375, 15.8130},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_INT8, {}, -15.9375, 15.8130}, Padding_VALID, 2, 2,
      ActivationFunctionType_RELU_N1_TO_1);
  m.SetInput({
      -1.7, -6, 2, 4,  //
      -3, -2, -10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({-1.0, 1.0}, 0.004)));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({120 - 128, 136 - 128}));

  m.SetInput({
      0, -6, -0.2, -0.4,    //
      -3, -2, 0.75, -0.99,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({0.0, 0.75}, 0.004)));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({128 - 128, 134 - 128}));
}

TEST(QuantizedInt8PoolingOpTest, MaxPoolActivationRelu6) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[-15.9375, 15.8130] --> [Scale{0.124512}, zero_point{0}]
  SymmetricQuantizedPoolingOpModel m(
      BuiltinOperator_MAX_POOL_2D,
      /*input=*/{TensorType_INT8, {1, 2, 4, 1}, -15.9375, 15.8130},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_INT8, {}, -15.9375, 15.8130}, Padding_VALID, 2, 2,
      ActivationFunctionType_RELU6);
  m.SetInput({
      0, -6, 12, 4,   //
      -3, -2, 10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({0.0, 6.0}, 0.025)));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({128 - 128, 176 - 128}));

  m.SetInput({
      0, 4.5, 12, 4,  //
      3, 2, 10, 7,    //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({4.5, 6.0}, 0.025)));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({164 - 128, 176 - 128}));
}

TEST(QuantizedInt8PoolingOpTest, MaxPoolPaddingSameStride1) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[0, 15.9375] --> [Scale{0.0625}, zero_point{-128}]
  SymmetricQuantizedPoolingOpModel m(
      BuiltinOperator_MAX_POOL_2D,
      /*input=*/{TensorType_INT8, {1, 2, 4, 1}, 0, 15.9375},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_INT8, {}, 0, 15.9375}, Padding_SAME, 1, 1);
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({6, 10, 10, 7, 3, 10, 10, 7})));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({96 - 128, 160 - 128, 160 - 128, 112 - 128,
                                48 - 128, 160 - 128, 160 - 128, 112 - 128}));
}

TEST(QuantizedInt8PoolingOpTest, MaxPoolPaddingValidStride1) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[0, 15.9375] --> [Scale{0.0625}, zero_point{-128}]
  SymmetricQuantizedPoolingOpModel m(
      BuiltinOperator_MAX_POOL_2D,
      /*input=*/{TensorType_INT8, {1, 2, 4, 1}, 0, 15.9375},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_INT8, {}, 0, 15.9375}, Padding_VALID, 1, 1);
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({6, 10, 10})));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({96 - 128, 160 - 128, 160 - 128}));
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

TEST(FloatPoolingOpTest, L2PoolActivationRelu) {
  FloatPoolingOpModel m(BuiltinOperator_L2_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}}, Padding_VALID, 2,
                        2, ActivationFunctionType_RELU);
  m.SetInput({
      -1, -6, 2, 4,   //
      -3, -2, 10, 7,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({3.53553, 6.5})));
}

TEST(FloatPoolingOpTest, L2PoolActivationRelu1) {
  FloatPoolingOpModel m(BuiltinOperator_L2_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}}, Padding_VALID, 2,
                        2, ActivationFunctionType_RELU_N1_TO_1);
  m.SetInput({
      -0.1, -0.6, 2, 4,   //
      -0.3, -0.2, 10, 7,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({0.353553, 1.0})));
}

TEST(FloatPoolingOpTest, L2PoolActivationRelu6) {
  FloatPoolingOpModel m(BuiltinOperator_L2_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}}, Padding_VALID, 2,
                        2, ActivationFunctionType_RELU6);
  m.SetInput({
      -0.1, -0.6, 2, 4,   //
      -0.3, -0.2, 10, 7,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({0.353553, 6.0})));
}

TEST(FloatPoolingOpTest, L2PoolPaddingSame) {
  FloatPoolingOpModel m(BuiltinOperator_L2_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}}, Padding_SAME);
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3.5, 6.5}));
}

TEST(FloatPoolingOpTest, L2PoolPaddingSameSlide1) {
  FloatPoolingOpModel m(BuiltinOperator_L2_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}}, Padding_SAME, 1,
                        1);
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {3.5, 6.0, 6.5, 5.70088, 2.54951, 7.2111, 8.63134, 7.0},
                  /*max_abs_error=*/1e-4)));
}

TEST(FloatPoolingOpTest, L2PoolPaddingValidSlide1) {
  FloatPoolingOpModel m(BuiltinOperator_L2_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}}, Padding_VALID, 1,
                        1);
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3.5, 6.0, 6.5}));
}

}  // namespace
}  // namespace tflite
