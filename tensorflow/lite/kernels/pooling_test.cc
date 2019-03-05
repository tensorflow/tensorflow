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

// Test quantized AveragePool with int8 input and output. The input is the same
// as the uint8 test QuantizedPoolingOpTest.AveragePool. The float output is
// identical to uint8 test and quantized output is identical to uint8 test with
// a 128 shift.
TEST(QuantizedPoolingOpTest, SymmetricAveragePool) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
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

// Send in a white image, expect something other than a white pixel, due to
// overflow.
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

  // Ordinarily we would see '255' here. However, the optimized version of
  // AveragePool uses a uint16 accumulator which causes it to overflow for
  // images this large.
  EXPECT_THAT(m.GetOutput(), ::testing::ElementsAre(28));
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

TEST(QuantizedUInt8PoolingOpTest, MaxPool) {
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
  SymmetricQuantizedPoolingOpModel m(
      BuiltinOperator_MAX_POOL_2D,
      /*input=*/{TensorType_INT8, {1, 2, 4, 1}, 0, 15.9375},
      /*filter_width=*/2, /*filter_height=*/2,
      /*output=*/{TensorType_INT8, {}, 0, 15.9375});
  m.SetInput({
      0, -6, 2, 4,   //
      3, 2, -10, 7,  //
  });
  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({3, 7})));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-80, -16}));
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
