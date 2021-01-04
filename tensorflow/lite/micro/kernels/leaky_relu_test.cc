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
#include <gtest/gtest.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BaseActivationsOpModel : public SingleOpModel {
 public:
  // A dedicated constructor for LeakyRelu, which does some options.
  BaseActivationsOpModel(TensorData input, float alpha) {
    input_ = AddInput(input);
    // The output scale and input scale might be different.
    if (input.type == TensorType_UINT8 || input.type == TensorType_INT8 ||
        input.type == TensorType_INT16) {
      auto output_min = (input.min >= 0) ? input.min : input.min * alpha;
      auto output_max = (input.max >= 0) ? input.max : input.max * alpha;
      if (input.type == TensorType_INT16) {
        output_ = AddOutput({TensorType_INT16,
                             {},
                             0,
                             0,
                             output_max / (std::numeric_limits<int16_t>::max()),
                             0});
      } else {
        output_ = AddOutput({input.type, {}, output_min, output_max});
      }
    } else {
      output_ = AddOutput({input.type, {}});
    }
    SetBuiltinOp(BuiltinOperator_LEAKY_RELU, BuiltinOptions_LeakyReluOptions,
                 CreateLeakyReluOptions(builder_, alpha).Union());
    BuildInterpreter({GetShape(input_)});
  }

 protected:
  int input_;
  int output_;
};

// Our fixed-point math function implementations have roughly 12 bits of
// accuracy, when specialized to 16-bit fixed-point arithmetic.
// That is purely an implementation compromise, it would have been possible
// to get closer to 16 bits of accuracy but that would be more expensive,
// and not needed for our purposes as ultimately the output is either
// immediately down-quantized to 8 bits, or will typically be at the output
// of the surrounding LSTM cell.
// So we can require roughly 2^-12 accuracy when the output is 16-bit, and
// we can more or less expect the full 2^-8 accuracy when the output is 8-bit.
//
// However, the representable output interval is often [-1, 1]  (it has to be
// for tanh, and even for logistic, when we implement it in fixed-point, we
// typically have to do so on such a symmetric interval, e.g. ARM NEON only
// has signed fixed-point arithmetic (SQRDMULH)).  As the width of [-1, 1]
// is 2, our representable values are often diluted by a factor of 2, whence
// the factor of 2 below.
const float kQuantizedTolerance = 2 * (1. / 256);
const float kQuantizedToleranceInt16 = 2 * (1. / 4096);

class QuantizedActivationsOpModel : public BaseActivationsOpModel {
 public:
  using BaseActivationsOpModel::BaseActivationsOpModel;

  template <typename T>
  void SetInput(const std::vector<float>& data) {
    QuantizeAndPopulate<T>(input_, data);
  }
  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }
};

TEST(QuantizedActivationsOpTest, LeakyReluUint8) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedActivationsOpModel m(
      /*input=*/{TensorType_UINT8, {2, 3}, 8 * kMin, 8 * kMax}, 0.5);

  m.SetInput<uint8_t>({
      0.0f, 1.0f, 3.0f,    // Row 1
      1.0f, -1.0f, -2.0f,  // Row 2
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.0f, 1.0f, 3.0f,    // Row 1
                      1.0f, -0.5f, -1.0f,  // Row 2
                  },
                  kQuantizedTolerance * 8)));
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedActivationsOpTestLeakyRelu() {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<integer_dtype>::max() /
      static_cast<float>(std::numeric_limits<integer_dtype>::max() + 1);

  QuantizedActivationsOpModel m(
      /*input=*/{tensor_type, {5, 5}, 5 * kMin, 5 * kMax}, 0.1);

  m.SetInput<integer_dtype>({
      -5.0f, -4.6f, -4.2f, -3.8f, -3.4f,  // Row 1
      -3.0f, -2.6f, -2.2f, -1.8f, -1.4f,  // Row 2
      -1.0f, -0.6f, -0.2f, 0.2f,  0.6f,   // Row 3
      1.0f,  1.4f,  1.8f,  2.2f,  2.6f,   // Row 4
      3.0f,  3.4f,  3.8f,  4.2f,  4.6f,   // Row 5
  });
  m.Invoke();

  float kTestQuantizedTolerance = tensor_type == TensorType_INT16
                                      ? kQuantizedToleranceInt16
                                      : kQuantizedTolerance * 5;

  EXPECT_THAT(m.GetDequantizedOutput<integer_dtype>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      -0.50f, -0.46f, -0.42f, -0.38f, -0.34f,  // Row 1
                      -0.30f, -0.26f, -0.22f, -0.18f, -0.14f,  // Row 2
                      -0.10f, -0.06f, -0.02f, 0.20f,  0.60f,   // Row 3
                      1.00f,  1.40f,  1.80f,  2.20f,  2.60f,   // Row 4
                      3.00f,  3.40f,  3.80f,  4.20f,  4.60f,   // Row 5
                  },
                  kTestQuantizedTolerance)));
}

TEST(QuantizedActivationsOpTest, LeakyReluInt8) {
  QuantizedActivationsOpTestLeakyRelu<TensorType_INT8, int8_t>();
}

TEST(QuantizedActivationsOpTest, LeakyReluInt16) {
  QuantizedActivationsOpTestLeakyRelu<TensorType_INT16, int16_t>();
}

class LeakyReluOpModel : public SingleOpModel {
 public:
  LeakyReluOpModel(const TensorData& input, float alpha) {
    input_ = AddInput(input);
    output_ = AddOutput(input);
    SetBuiltinOp(BuiltinOperator_LEAKY_RELU, BuiltinOptions_LeakyReluOptions,
                 CreateLeakyReluOptions(builder_, alpha).Union());
    BuildInterpreter({GetShape(input_)});
  }
  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 protected:
  int input_;
  int output_;
};

TEST(FloatActivationsOpTest, LeakyRelu) {
  LeakyReluOpModel m({TensorType_FLOAT32, {2, 3}}, 0.5f);

  m.SetInput({
      0.0f, 1.0f, 3.0f,    // Row 1
      1.0f, -1.0f, -2.0f,  // Row 2
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0.0f, 1.0f, 3.0f,    // Row 1
                                 1.0f, -0.5f, -1.0f,  // Row 2
                             }));
}

}  // namespace
}  // namespace tflite
