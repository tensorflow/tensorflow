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

#include "absl/memory/memory.h"
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
  // Most activations don't take any options, so this constructor works for
  // them.
  BaseActivationsOpModel(BuiltinOperator type, TensorData input) {
    input_ = AddInput(input);
    if (input.type == TensorType_UINT8) {
      output_ = AddOutput({input.type, {}, 0, 0, 1. / 256});
    } else if (input.type == TensorType_INT8) {
      output_ = AddOutput({input.type, {}, 0, 0, 1. / 256, -128});
    } else {
      output_ = AddOutput({input.type, {}});
    }
    SetBuiltinOp(type, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(input_)});
  }

  BaseActivationsOpModel(TfLiteRegistration* registration, BuiltinOperator type,
                         TensorData input) {
    input_ = AddInput(input);
    if (input.type == TensorType_UINT8) {
      output_ = AddOutput({input.type, {}, 0, 0, 1. / 256});
    } else if (input.type == TensorType_INT8) {
      output_ = AddOutput({input.type, {}, 0, 0, 1. / 256, -128});
    } else {
      output_ = AddOutput({input.type, {}});
    }
    SetBuiltinOp(type, BuiltinOptions_NONE, 0);
    resolver_ = absl::make_unique<SingleOpResolver>(type, registration);
    BuildInterpreter({GetShape(input_)});
  }

  // A dedicated constructor for SOFTMAX, which does some options.
  BaseActivationsOpModel(float softmax_beta, TensorData input,
                         TensorType output_type) {
    input_ = AddInput(input);
    if (output_type == TensorType_UINT8) {
      output_ = AddOutput({TensorType_UINT8, {}, 0, 0, 1. / 256});
    } else if (output_type == TensorType_INT8) {
      output_ = AddOutput({TensorType_INT8, {}, 0, 0, 1. / 256, -128});
    } else if (input.type == TensorType_INT16 &&
               output_type == TensorType_INT16) {
      output_ = AddOutput({TensorType_INT16,
                           {},
                           0,
                           0,
                           1.0f / (std::numeric_limits<int16_t>::max() + 1),
                           0});
    } else if (input.type != TensorType_INT16 &&
               output_type == TensorType_INT16) {
      output_ = AddOutput({TensorType_INT16, {}, 0, 0, 1. / 32768, -16384});
    } else {
      output_ = AddOutput({output_type, {}});
    }
    SetBuiltinOp(BuiltinOperator_SOFTMAX, BuiltinOptions_SoftmaxOptions,
                 CreateSoftmaxOptions(builder_, softmax_beta).Union());
    BuildInterpreter({GetShape(input_)});
  }

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

  BaseActivationsOpModel(BuiltinOperator type, const TensorData& input,
                         const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(type, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(input_)});
  }

  BaseActivationsOpModel(TfLiteRegistration* registration, BuiltinOperator type,
                         const TensorData& input, const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(type, BuiltinOptions_NONE, 0);
    resolver_ = absl::make_unique<SingleOpResolver>(type, registration);
    BuildInterpreter({GetShape(input_)});
  }

 protected:
  int input_;
  int output_;
};

class FloatActivationsOpModel : public BaseActivationsOpModel {
 public:
  using BaseActivationsOpModel::BaseActivationsOpModel;

  void SetInput(const std::vector<float>& data) {
    PopulateTensor(input_, data);
  }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
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

TEST(FloatActivationsOpTest, Elu) {
  FloatActivationsOpModel m(BuiltinOperator_ELU,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, -4,     //
      3, -2, 10, -0.1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 0.0, -0.997521, 2.0, -0.981684,    //
                                 3.0, -0.864665, 10.0, -0.0951626,  //
                             })));
}

TEST(QuantizedActivationsOpTest, EluInt8) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedActivationsOpModel model(
      BuiltinOperator_ELU,
      /*input=*/{TensorType_INT8, {1, 2, 4, 1}, 8 * kMin, 8 * kMax},
      /*output=*/{TensorType_INT8, {1, 2, 4, 1}, 8 * kMin, 8 * kMax});

  model.SetInput<int8_t>({
      0, -6, 2, -4,    //
      3, -2, 6, -0.1,  //
  });

  model.Invoke();
  EXPECT_THAT(model.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0, -1.0, 2.0, -1,          //
                      3.0, -0.875, 6.0, -0.125,  //
                  },
                  kQuantizedTolerance)));
}

}  // namespace
}  // namespace tflite
