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
#include <stdint.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {

namespace ops {
namespace builtin {

// Tanh kernel registrations.
TfLiteRegistration* Register_TANH_REF();
TfLiteRegistration* Register_TANH_GENERIC_OPT();
TfLiteRegistration* Register_TANH_FIXED_POINT_OPT();

// Logistic kernel registrations.
TfLiteRegistration* Register_LOGISTIC_REF();
TfLiteRegistration* Register_LOGISTIC_GENERIC_OPT();
TfLiteRegistration* Register_LOGISTIC_FIXED_POINT_OPT();

// LogSoftmax kernel registrations.
TfLiteRegistration* Register_LOG_SOFTMAX_REF();
TfLiteRegistration* Register_LOG_SOFTMAX();

// Softmax kernel registrations.
TfLiteRegistration* Register_SOFTMAX_REF();
TfLiteRegistration* Register_SOFTMAX();

// PRelu kernel registrations.
TfLiteRegistration* Register_PRELU_REF();
TfLiteRegistration* Register_PRELU();

// LeakyRelu kernel registrations.
TfLiteRegistration* Register_LEAKY_RELU_REF();
TfLiteRegistration* Register_LEAKY_RELU();

}  // namespace builtin
}  // namespace ops

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
    resolver_ = std::make_unique<SingleOpResolver>(type, registration);
    BuildInterpreter({GetShape(input_)});
  }

  // A dedicated constructor for SOFTMAX, which does some options.
  BaseActivationsOpModel(TfLiteRegistration* registration, float softmax_beta,
                         TensorData input, TensorType output_type) {
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
      output_ = AddOutput({TensorType_INT16, {}, 0, 0, 1. / 65536, -32768});
    } else {
      output_ = AddOutput({output_type, {}});
    }
    SetBuiltinOp(BuiltinOperator_SOFTMAX, BuiltinOptions_SoftmaxOptions,
                 CreateSoftmaxOptions(builder_, softmax_beta).Union());
    resolver_ = std::make_unique<SingleOpResolver>(BuiltinOperator_SOFTMAX,
                                                   registration);
    BuildInterpreter({GetShape(input_)});
  }

  // A dedicated constructor for LeakyRelu, which does some options.
  BaseActivationsOpModel(TfLiteRegistration* registration, TensorData input,
                         float alpha) {
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
    resolver_ = std::make_unique<SingleOpResolver>(BuiltinOperator_LEAKY_RELU,
                                                   registration);
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
    resolver_ = std::make_unique<SingleOpResolver>(type, registration);
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

const auto kTanhKernelMap = new std::map<string, TfLiteRegistration*>({
    {"Reference", ops::builtin::Register_TANH_REF()},
    {"GenericOptimized", ops::builtin::Register_TANH_GENERIC_OPT()},
    {"FixedPointOptimized", ops::builtin::Register_TANH_FIXED_POINT_OPT()},
});

class TanhOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kTanhKernelMap;
  }
};

const auto kLogisticKernelMap = new std::map<string, TfLiteRegistration*>({
    {"Reference", ops::builtin::Register_LOGISTIC_REF()},
    {"GenericOptimized", ops::builtin::Register_LOGISTIC_GENERIC_OPT()},
    {"FixedPointOptimized", ops::builtin::Register_LOGISTIC_FIXED_POINT_OPT()},
});

class LogisticOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kLogisticKernelMap;
  }
};

const auto kLogSoftmaxKernelMap = new std::map<string, TfLiteRegistration*>({
    {"Reference", ops::builtin::Register_LOG_SOFTMAX_REF()},
    {"GenericOptimized", ops::builtin::Register_LOG_SOFTMAX()},
});

class LogSoftmaxOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kLogSoftmaxKernelMap;
  }
};

const auto kSoftmaxKernelMap = new std::map<string, TfLiteRegistration*>({
    {"Reference", ops::builtin::Register_SOFTMAX_REF()},
    {"GenericOptimized", ops::builtin::Register_SOFTMAX()},
});

class SoftmaxOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kSoftmaxKernelMap;
  }
};

TEST(FloatActivationsOpTest, Elu) {
  FloatActivationsOpModel m(BuiltinOperator_ELU,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, -4,     //
      3, -2, 10, -0.1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0, -1.0, 2.0, -1,          //
                      3.0, -0.875, 6.0, -0.125,  //
                  },
                  kQuantizedTolerance)));
}

TEST(FloatActivationsOpTest, Relu) {
  FloatActivationsOpModel m(BuiltinOperator_RELU,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0, 0, 2, 4,   //
                                 3, 0, 10, 1,  //
                             }));
}

TEST(FloatActivationsOpTest, Relu0To1) {
  FloatActivationsOpModel m(BuiltinOperator_RELU_0_TO_1,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0.0, -0.6, 0.2, -0.4,  //
      0.3, -2.0, 1.1, -0.1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0.0, 0.0, 0.2, 0.0,  //
                                 0.3, 0.0, 1.0, 0.0,  //
                             }));
}

TEST(FloatActivationsOpTest, Relu1) {
  FloatActivationsOpModel m(BuiltinOperator_RELU_N1_TO_1,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0.0, -0.6, 0.2, -0.4,  //
      0.3, -2.0, 1.1, -0.1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0.0, -0.6, 0.2, -0.4,  //
                                 0.3, -1.0, 1.0, -0.1,  //
                             }));
}

TEST(FloatActivationsOpTest, Relu6) {
  FloatActivationsOpModel m(BuiltinOperator_RELU6,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0, 0, 2, 4,  //
                                 3, 0, 6, 1,  //
                             }));
}

void GenerateUniformRandomVector(int size, float min, float max,
                                 std::minstd_rand* random_engine,
                                 std::vector<float>* result) {
  // Never use std::uniform_*_distribution in tests, it's
  // implementation-defined. Likewise, don't use std::default_random_engine,
  // implementation-defined. Implementation-defined is bad because it means that
  // any toolchain update or new platform may run into test failures.
  // std::minstd_rand is a standard instantiation of
  // std::linear_congruential_engine, the cheapest generator in c++11 stdlib,
  // it's good enough here.
  result->resize(size);
  for (int i = 0; i < size; i++) {
    // We don't care whether the `max` value may ever be produced exactly.
    // It may actually be thanks to rounding, as std::minstd_rand::modulus
    // is 2^31 - 1 is greater than the inverse float epsilon.
    float random_value_scaled_0_1 =
        (*random_engine)() *
        (1.0f / static_cast<float>(std::minstd_rand::modulus));
    (*result)[i] = min + (max - min) * random_value_scaled_0_1;
  }
}

void EvalTestReferenceHardSwish(int size, const std::vector<float>& input,
                                std::vector<float>* result) {
  result->resize(size);
  for (int i = 0; i < size; i++) {
    const float in = input[i];
    (*result)[i] = in * std::min(6.0f, std::max(0.0f, in + 3)) * (1.0f / 6.0f);
  }
}

void TestFloatHardSwish(int size, std::minstd_rand* random_engine) {
  std::vector<float> float_input_values;
  const float kMin = -10.0f;
  const float kMax = 10.0f;
  GenerateUniformRandomVector(size, kMin, kMax, random_engine,
                              &float_input_values);
  std::vector<float> float_ref_output_values;
  EvalTestReferenceHardSwish(size, float_input_values,
                             &float_ref_output_values);
  FloatActivationsOpModel m(BuiltinOperator_HARD_SWISH,
                            /*input=*/{TensorType_FLOAT32, {1, 1, 1, size}},
                            /*output=*/{TensorType_FLOAT32, {1, 1, 1, size}});
  m.SetInput(float_input_values);

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(float_ref_output_values)));
}

template <typename QuantizedType>
void TestQuantizedHardSwish(TensorType tensor_type, int size, float input_min,
                            float input_max, float output_min, float output_max,
                            std::minstd_rand* random_engine) {
  std::vector<float> float_input_values;
  GenerateUniformRandomVector(size, input_min, input_max, random_engine,
                              &float_input_values);
  std::vector<float> float_ref_output_values;
  EvalTestReferenceHardSwish(size, float_input_values,
                             &float_ref_output_values);
  for (float& val : float_ref_output_values) {
    val = std::min(output_max, std::max(output_min, val));
  }
  QuantizedActivationsOpModel m(
      BuiltinOperator_HARD_SWISH,
      /*input=*/{tensor_type, {1, 1, 1, size}, input_min, input_max},
      /*output=*/{tensor_type, {1, 1, 1, size}, output_min, output_max});
  m.SetInput<QuantizedType>(float_input_values);

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const std::vector<float>& dequantized_output =
      m.GetDequantizedOutput<QuantizedType>();
  // The numerical error for any 8bit quantized function is at least one half
  // times the quantization step: 0.5 * (kOutMax - kOutMin) / 256.
  // To that we add again the quantization step (kOutMax - kOutMin) / 256
  // to allow for an off-by-one rounding error.
  const float kTolerance =
      std::max(input_max - input_min, output_max - output_min) * (1.5f / 256.f);
  EXPECT_THAT(dequantized_output, ElementsAreArray(ArrayFloatNear(
                                      float_ref_output_values, kTolerance)));
}

template <typename QuantizedType>
void TestQuantizedHardSwishBias(TensorType tensor_type, float input_min,
                                float input_max, float output_min,
                                float output_max, float tolerated_bias) {
  const float quantized_type_range =
      static_cast<float>(std::numeric_limits<QuantizedType>::max()) -
      static_cast<float>(std::numeric_limits<QuantizedType>::min());
  const float input_scale = (input_max - input_min) / quantized_type_range;
  const float output_scale = (output_max - output_min) / quantized_type_range;
  const float max_scale = std::max(output_scale, input_scale);

  // In this bias-focused test case, no need for randomly generated input
  // values.
  ASSERT_LE(input_min, -3.0f);
  ASSERT_GE(input_max, 3.0f);
  const int quantized_input_negative_three =
      std::round(std::numeric_limits<QuantizedType>::min() +
                 (-3.0f - input_min) / input_scale);
  const int quantized_input_positive_three =
      std::round(std::numeric_limits<QuantizedType>::min() +
                 (3.0f - input_min) / input_scale);
  std::vector<float> float_input_values;
  for (int i = quantized_input_negative_three;
       i <= quantized_input_positive_three; i++) {
    float_input_values.push_back(
        input_min +
        (i - std::numeric_limits<QuantizedType>::min()) * input_scale);
  }
  const int size = float_input_values.size();
  std::vector<float> float_ref_output_values;
  EvalTestReferenceHardSwish(size, float_input_values,
                             &float_ref_output_values);
  for (float& val : float_ref_output_values) {
    val = std::min(output_max, std::max(output_min, val));
  }
  QuantizedActivationsOpModel m(
      BuiltinOperator_HARD_SWISH,
      /*input=*/{tensor_type, {1, 1, 1, size}, input_min, input_max},
      /*output=*/{tensor_type, {1, 1, 1, size}, output_min, output_max});
  m.SetInput<QuantizedType>(float_input_values);

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const std::vector<float>& dequantized_output =
      m.GetDequantizedOutput<QuantizedType>();

  float sum_diff = 0;
  for (int i = 0; i < size; i++) {
    sum_diff += dequantized_output[i] - float_ref_output_values[i];
  }
  const float bias = sum_diff / (size * max_scale);
  EXPECT_LE(std::abs(bias), tolerated_bias);
}

TEST(FloatActivationsOpTest, HardSwish) {
  std::minstd_rand random_engine;
  for (int size : {1, 2, 3, 4, 10, 20, 30, 40, 100}) {
    TestFloatHardSwish(size, &random_engine);
  }
}

TEST(QuantizedActivationsOpTest, HardSwish) {
  std::minstd_rand random_engine;
  std::vector<std::pair<float, float>> minmax_pairs{
      {0.f, 1.f}, {-2.f, 1.f}, {-5.f, 10.f}, {-40.f, 60.f}};
  for (const auto& input_minmax : minmax_pairs) {
    for (const auto& output_minmax : minmax_pairs) {
      float input_min = input_minmax.first;
      float input_max = input_minmax.second;
      float output_min = output_minmax.first;
      float output_max = output_minmax.second;
      for (int size : {1, 3, 10, 100}) {
        TestQuantizedHardSwish<uint8_t>(TensorType_UINT8, size, input_min,
                                        input_max, output_min, output_max,
                                        &random_engine);
        TestQuantizedHardSwish<int8_t>(TensorType_INT8, size, input_min,
                                       input_max, output_min, output_max,
                                       &random_engine);
      }
    }
  }
}

// See the comment in the reference implementation of quantized HardSwish:
// A numerical issue significantly affecting ImageNet classification accuracy
// with MobileNet v3 is only observable at the scale of HardSwish unit tests
// if we monitor specifically bias. This testcase is extracted from one of the
// HardSwish nodes in that MobileNet v3 that exhibited this issue.
TEST(QuantizedActivationsOpTest, HardSwishBias) {
  TestQuantizedHardSwishBias<uint8_t>(TensorType_UINT8, -11.654928f, 25.036512f,
                                      -0.3905796f, 24.50887f, 0.035);
}

TEST_P(TanhOpTest, Tanh) {
  FloatActivationsOpModel m(GetRegistration(), BuiltinOperator_TANH,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 0, -0.9999877, 0.9640275, 0.999329,    //
                                 0.99505475, -0.9640275, 1, 0.7615941,  //
                             })));
}

TEST(QuantizedActivationsOpTest, Relu6Uint8) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedActivationsOpModel m(
      BuiltinOperator_RELU6,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, 8 * kMin, 8 * kMax},
      /*output=*/{TensorType_UINT8, {1, 2, 4, 1}, 8 * kMin, 8 * kMax});
  m.SetInput<uint8_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0, 0, 2, 4,  //
                      3, 0, 6, 1,  //
                  },
                  kQuantizedTolerance)));
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAreArray({128, 128, 160, 192, 176, 128, 224, 144}));
}

const auto kLeakyReluKernelMap = new std::map<string, TfLiteRegistration*>({
    {"Reference", ops::builtin::Register_LEAKY_RELU_REF()},
    {"GenericOptimized", ops::builtin::Register_LEAKY_RELU()},
});

class LeakyReluOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kLeakyReluKernelMap;
  }
};

TEST_P(LeakyReluOpTest, LeakyReluUint8) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedActivationsOpModel m(
      GetRegistration(),
      /*input=*/{TensorType_UINT8, {2, 3}, 8 * kMin, 8 * kMax}, 0.5);

  m.SetInput<uint8_t>({
      0.0f, 1.0f, 3.0f,    // Row 1
      1.0f, -1.0f, -2.0f,  // Row 2
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.0f, 1.0f, 3.0f,    // Row 1
                      1.0f, -0.5f, -1.0f,  // Row 2
                  },
                  kQuantizedTolerance * 8)));
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedActivationsOpTestLeakyRelu(TfLiteRegistration* registration) {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<integer_dtype>::max() /
      static_cast<float>(std::numeric_limits<integer_dtype>::max() + 1);

  QuantizedActivationsOpModel m(
      registration,
      /*input=*/{tensor_type, {5, 5}, 5 * kMin, 5 * kMax}, 0.1);

  m.SetInput<integer_dtype>({
      -5.0f, -4.6f, -4.2f, -3.8f, -3.4f,  // Row 1
      -3.0f, -2.6f, -2.2f, -1.8f, -1.4f,  // Row 2
      -1.0f, -0.6f, -0.2f, 0.2f,  0.6f,   // Row 3
      1.0f,  1.4f,  1.8f,  2.2f,  2.6f,   // Row 4
      3.0f,  3.4f,  3.8f,  4.2f,  4.6f,   // Row 5
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

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

TEST_P(LeakyReluOpTest, LeakyReluInt8) {
  QuantizedActivationsOpTestLeakyRelu<TensorType_INT8, int8_t>(
      GetRegistration());
}

TEST_P(LeakyReluOpTest, LeakyReluInt16) {
  QuantizedActivationsOpTestLeakyRelu<TensorType_INT16, int16_t>(
      GetRegistration());
}

TEST(QuantizedActivationsOpTest, Relu0To1Int8) {
  const float kMin = 0;
  const float kMax = 1;
  QuantizedActivationsOpModel m(
      BuiltinOperator_RELU_0_TO_1,
      /*input=*/{TensorType_INT8, {1, 2, 4, 1}, 2 * kMin, kMax},
      /*output=*/{TensorType_INT8, {1, 2, 4, 1}, 2 * kMin, kMax});

  m.SetInput<int8_t>({
      0.0, -0.6, 0.2, -0.4,  //
      0.3, -2.0, 1.1, -0.1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(), ElementsAreArray(ArrayFloatNear(
                                                    {
                                                        0.0, 0.0, 0.2, 0.0,  //
                                                        0.3, 0.0, 1.0, 0.0,  //
                                                    },
                                                    kQuantizedTolerance)));
}

TEST(QuantizedActivationsOpTest, Relu1Int8) {
  const float kMin = -1;
  const float kMax = 1;
  QuantizedActivationsOpModel m(
      BuiltinOperator_RELU_N1_TO_1,
      /*input=*/{TensorType_INT8, {1, 2, 4, 1}, 2 * kMin, kMax},
      /*output=*/{TensorType_INT8, {1, 2, 4, 1}, 2 * kMin, kMax});

  m.SetInput<int8_t>({
      0.0, -0.6, 0.2, -0.4,  //
      0.3, -2.0, 1.1, -0.1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.0, -0.6, 0.2, -0.4,  //
                      0.3, -1.0, 1.0, -0.1,  //
                  },
                  kQuantizedTolerance)));
}

TEST(QuantizedActivationsOpTest, Relu0To1UInt8) {
  const float kMin = 0;
  const float kMax = 1;
  QuantizedActivationsOpModel m(
      BuiltinOperator_RELU_0_TO_1,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, 2 * kMin, kMax},
      /*output=*/{TensorType_UINT8, {1, 2, 4, 1}, 2 * kMin, kMax});

  m.SetInput<uint8_t>({
      0.0, -0.6, 0.2, -0.4,  //
      0.3, -2.0, 1.1, -0.1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.0, 0.0, 0.2, 0.0,  //
                      0.3, 0.0, 1.0, 0.0,  //
                  },
                  kQuantizedTolerance)));
}

TEST(QuantizedActivationsOpTest, Relu1UInt8) {
  const float kMin = -1;
  const float kMax = 1;
  QuantizedActivationsOpModel m(
      BuiltinOperator_RELU_N1_TO_1,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, 2 * kMin, kMax},
      /*output=*/{TensorType_UINT8, {1, 2, 4, 1}, 2 * kMin, kMax});

  m.SetInput<uint8_t>({
      0.0, -0.6, 0.2, -0.4,  //
      0.3, -2.0, 1.1, -0.1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.0, -0.6, 0.2, -0.4,  //
                      0.3, -1.0, 1.0, -0.1,  //
                  },
                  kQuantizedTolerance)));
}

TEST(QuantizedActivationsOpTest, Relu6Int8) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedActivationsOpModel m(
      BuiltinOperator_RELU6,
      /*input=*/{TensorType_INT8, {1, 2, 4, 1}, 8 * kMin, 8 * kMax},
      /*output=*/{TensorType_INT8, {1, 2, 4, 1}, 8 * kMin, 8 * kMax});
  m.SetInput<int8_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(), ElementsAreArray(ArrayFloatNear(
                                                    {
                                                        0, 0, 2, 4,  //
                                                        3, 0, 6, 1,  //
                                                    },
                                                    kQuantizedTolerance)));
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({0, 0, 32, 64, 48, 0, 96, 16}));
}

TEST(QuantizedActivationsOpTest, Relu6Int16) {
  const float kMin = -1;
  const float kMax = 32767.f / 32768.f;
  QuantizedActivationsOpModel m(
      BuiltinOperator_RELU6,
      /*input=*/{TensorType_INT16, {1, 2, 4, 1}, 8 * kMin, 8 * kMax},
      /*output=*/{TensorType_INT16, {1, 2, 4, 1}, 8 * kMin, 8 * kMax});
  m.SetInput<int16_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0, 0, 2, 4,  //
                      3, 0, 6, 1,  //
                  },
                  kQuantizedToleranceInt16)));
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAreArray({0, 0, 8192, 16384, 12288, 0, 24576, 4096}));
}

TEST(QuantizedActivationsOpTest, ReluUint8) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedActivationsOpModel m(
      BuiltinOperator_RELU,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, 8 * kMin, 8 * kMax},
      /*output=*/{TensorType_UINT8, {1, 2, 4, 1}, 8 * kMin, 8 * kMax});
  m.SetInput<uint8_t>({
      0, -6, 2, 4,  //
      3, -2, 7, 1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0, 0, 2, 4,  //
                      3, 0, 7, 1,  //
                  },
                  kQuantizedTolerance)));
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAreArray({128, 128, 160, 192, 176, 128, 240, 144}));
}

TEST(QuantizedActivationsOpTest, ReluInt8) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedActivationsOpModel m(
      BuiltinOperator_RELU,
      /*input=*/{TensorType_INT8, {1, 2, 4, 1}, 8 * kMin, 8 * kMax},
      /*output=*/{TensorType_INT8, {1, 2, 4, 1}, 8 * kMin, 8 * kMax});
  m.SetInput<int8_t>({
      0, -6, 2, 4,  //
      3, -2, 7, 1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(), ElementsAreArray(ArrayFloatNear(
                                                    {
                                                        0, 0, 2, 4,  //
                                                        3, 0, 7, 1,  //
                                                    },
                                                    kQuantizedTolerance)));
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({0, 0, 32, 64, 48, 0, 112, 16}));
}

TEST(QuantizedActivationsOpTest, ReluInt16) {
  const float kMin = -1;
  const float kMax = 32767.f / 32768.f;
  QuantizedActivationsOpModel m(
      BuiltinOperator_RELU,
      /*input=*/{TensorType_INT16, {1, 2, 4, 1}, 8 * kMin, 8 * kMax},
      /*output=*/{TensorType_INT16, {1, 2, 4, 1}, 8 * kMin, 8 * kMax});
  m.SetInput<int16_t>({
      0, -6, 2, 4,  //
      3, -2, 7, 1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0, 0, 2, 4,  //
                      3, 0, 7, 1,  //
                  },
                  kQuantizedToleranceInt16)));
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAreArray({0, 0, 8192, 16384, 12288, 0, 28672, 4096}));
}

TEST_P(TanhOpTest, TanhUint8) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  const float kTanhTolerance = 0.014f;
  QuantizedActivationsOpModel m(
      GetRegistration(), BuiltinOperator_TANH,
      /*input=*/{TensorType_UINT8, {89}, 8 * kMin, 8 * kMax},
      /*output=*/{TensorType_UINT8, {89}, kMin, kMax});
  // 64+16+8+1 elements, from -8 to 8.
  m.SetInput<uint8_t>(
      {-8.0000000000, -7.8181818182, -7.6363636364, -7.4545454545,
       -7.2727272727, -7.0909090909, -6.9090909091, -6.7272727273,
       -6.5454545455, -6.3636363636, -6.1818181818, -6.0000000000,
       -5.8181818182, -5.6363636364, -5.4545454545, -5.2727272727,
       -5.0909090909, -4.9090909091, -4.7272727273, -4.5454545455,
       -4.3636363636, -4.1818181818, -4.0000000000, -3.8181818182,
       -3.6363636364, -3.4545454545, -3.2727272727, -3.0909090909,
       -2.9090909091, -2.7272727273, -2.5454545455, -2.3636363636,
       -2.1818181818, -2.0000000000, -1.8181818182, -1.6363636364,
       -1.4545454545, -1.2727272727, -1.0909090909, -0.9090909091,
       -0.7272727273, -0.5454545455, -0.3636363636, -0.1818181818,
       0.0000000000,  0.1818181818,  0.3636363636,  0.5454545455,
       0.7272727273,  0.9090909091,  1.0909090909,  1.2727272727,
       1.4545454545,  1.6363636364,  1.8181818182,  2.0000000000,
       2.1818181818,  2.3636363636,  2.5454545455,  2.7272727273,
       2.9090909091,  3.0909090909,  3.2727272727,  3.4545454545,
       3.6363636364,  3.8181818182,  4.0000000000,  4.1818181818,
       4.3636363636,  4.5454545455,  4.7272727273,  4.9090909091,
       5.0909090909,  5.2727272727,  5.4545454545,  5.6363636364,
       5.8181818182,  6.0000000000,  6.1818181818,  6.3636363636,
       6.5454545455,  6.7272727273,  6.9090909091,  7.0909090909,
       7.2727272727,  7.4545454545,  7.6363636364,  7.8181818182,
       8.0000000000});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {-0.9999997749, -0.9999996762, -0.9999995342, -0.9999993300,
                   -0.9999990361, -0.9999986134, -0.9999980053, -0.9999971306,
                   -0.9999958722, -0.9999940619, -0.9999914578, -0.9999877117,
                   -0.9999823226, -0.9999745703, -0.9999634183, -0.9999473758,
                   -0.9999242982, -0.9998911009, -0.9998433469, -0.9997746542,
                   -0.9996758446, -0.9995337191, -0.9993292997, -0.9990353053,
                   -0.9986125310, -0.9980046622, -0.9971308601, -0.9958751909,
                   -0.9940716137, -0.9914827859, -0.9877703933, -0.9824541388,
                   -0.9748561217, -0.9640275801, -0.9486568273, -0.9269625051,
                   -0.8965880154, -0.8545351057, -0.7972097087, -0.7206956332,
                   -0.6213939966, -0.4971057414, -0.3484130125, -0.1798408185,
                   0.0000000000,  0.1798408185,  0.3484130125,  0.4971057414,
                   0.6213939966,  0.7206956332,  0.7972097087,  0.8545351057,
                   0.8965880154,  0.9269625051,  0.9486568273,  0.9640275801,
                   0.9748561217,  0.9824541388,  0.9877703933,  0.9914827859,
                   0.9940716137,  0.9958751909,  0.9971308601,  0.9980046622,
                   0.9986125310,  0.9990353053,  0.9993292997,  0.9995337191,
                   0.9996758446,  0.9997746542,  0.9998433469,  0.9998911009,
                   0.9999242982,  0.9999473758,  0.9999634183,  0.9999745703,
                   0.9999823226,  0.9999877117,  0.9999914578,  0.9999940619,
                   0.9999958722,  0.9999971306,  0.9999980053,  0.9999986134,
                   0.9999990361,  0.9999993300,  0.9999995342,  0.9999996762,
                   0.9999997749},
                  kTanhTolerance)));
}

TEST_P(TanhOpTest, TanhInt8) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  const float kTanhTolerance = 0.014f;
  QuantizedActivationsOpModel m(
      GetRegistration(), BuiltinOperator_TANH,
      /*input=*/{TensorType_INT8, {89}, 8 * kMin, 8 * kMax},
      /*output=*/{TensorType_INT8, {89}, kMin, kMax});
  // 64+16+8+1 elements, from -8 to 8.
  m.SetInput<int8_t>(
      {-8.0000000000, -7.8181818182, -7.6363636364, -7.4545454545,
       -7.2727272727, -7.0909090909, -6.9090909091, -6.7272727273,
       -6.5454545455, -6.3636363636, -6.1818181818, -6.0000000000,
       -5.8181818182, -5.6363636364, -5.4545454545, -5.2727272727,
       -5.0909090909, -4.9090909091, -4.7272727273, -4.5454545455,
       -4.3636363636, -4.1818181818, -4.0000000000, -3.8181818182,
       -3.6363636364, -3.4545454545, -3.2727272727, -3.0909090909,
       -2.9090909091, -2.7272727273, -2.5454545455, -2.3636363636,
       -2.1818181818, -2.0000000000, -1.8181818182, -1.6363636364,
       -1.4545454545, -1.2727272727, -1.0909090909, -0.9090909091,
       -0.7272727273, -0.5454545455, -0.3636363636, -0.1818181818,
       0.0000000000,  0.1818181818,  0.3636363636,  0.5454545455,
       0.7272727273,  0.9090909091,  1.0909090909,  1.2727272727,
       1.4545454545,  1.6363636364,  1.8181818182,  2.0000000000,
       2.1818181818,  2.3636363636,  2.5454545455,  2.7272727273,
       2.9090909091,  3.0909090909,  3.2727272727,  3.4545454545,
       3.6363636364,  3.8181818182,  4.0000000000,  4.1818181818,
       4.3636363636,  4.5454545455,  4.7272727273,  4.9090909091,
       5.0909090909,  5.2727272727,  5.4545454545,  5.6363636364,
       5.8181818182,  6.0000000000,  6.1818181818,  6.3636363636,
       6.5454545455,  6.7272727273,  6.9090909091,  7.0909090909,
       7.2727272727,  7.4545454545,  7.6363636364,  7.8181818182,
       8.0000000000});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {-0.9999997749, -0.9999996762, -0.9999995342, -0.9999993300,
                   -0.9999990361, -0.9999986134, -0.9999980053, -0.9999971306,
                   -0.9999958722, -0.9999940619, -0.9999914578, -0.9999877117,
                   -0.9999823226, -0.9999745703, -0.9999634183, -0.9999473758,
                   -0.9999242982, -0.9998911009, -0.9998433469, -0.9997746542,
                   -0.9996758446, -0.9995337191, -0.9993292997, -0.9990353053,
                   -0.9986125310, -0.9980046622, -0.9971308601, -0.9958751909,
                   -0.9940716137, -0.9914827859, -0.9877703933, -0.9824541388,
                   -0.9748561217, -0.9640275801, -0.9486568273, -0.9269625051,
                   -0.8965880154, -0.8545351057, -0.7972097087, -0.7206956332,
                   -0.6213939966, -0.4971057414, -0.3484130125, -0.1798408185,
                   0.0000000000,  0.1798408185,  0.3484130125,  0.4971057414,
                   0.6213939966,  0.7206956332,  0.7972097087,  0.8545351057,
                   0.8965880154,  0.9269625051,  0.9486568273,  0.9640275801,
                   0.9748561217,  0.9824541388,  0.9877703933,  0.9914827859,
                   0.9940716137,  0.9958751909,  0.9971308601,  0.9980046622,
                   0.9986125310,  0.9990353053,  0.9993292997,  0.9995337191,
                   0.9996758446,  0.9997746542,  0.9998433469,  0.9998911009,
                   0.9999242982,  0.9999473758,  0.9999634183,  0.9999745703,
                   0.9999823226,  0.9999877117,  0.9999914578,  0.9999940619,
                   0.9999958722,  0.9999971306,  0.9999980053,  0.9999986134,
                   0.9999990361,  0.9999993300,  0.9999995342,  0.9999996762,
                   0.9999997749},
                  kTanhTolerance)));
}

TEST_P(TanhOpTest, TanhInt16) {
  const float kMin = -1;
  const float kMax = 32767.f / 32768.f;
  QuantizedActivationsOpModel m(
      GetRegistration(), BuiltinOperator_TANH,
      /*input=*/{TensorType_INT16, {177}, 16 * kMin, 16 * kMax},
      /*output=*/{TensorType_INT16, {177}, kMin, kMax});
  m.SetInput<int16_t>(
      {-20.0000000000, -19.7727272727, -19.5454545455, -19.3181818182,
       -19.0909090909, -18.8636363636, -18.6363636364, -18.4090909091,
       -18.1818181818, -17.9545454545, -17.7272727273, -17.5000000000,
       -17.2727272727, -17.0454545455, -16.8181818182, -16.5909090909,
       -16.3636363636, -16.1363636364, -15.9090909091, -15.6818181818,
       -15.4545454545, -15.2272727273, -15.0000000000, -14.7727272727,
       -14.5454545455, -14.3181818182, -14.0909090909, -13.8636363636,
       -13.6363636364, -13.4090909091, -13.1818181818, -12.9545454545,
       -12.7272727273, -12.5000000000, -12.2727272727, -12.0454545455,
       -11.8181818182, -11.5909090909, -11.3636363636, -11.1363636364,
       -10.9090909091, -10.6818181818, -10.4545454545, -10.2272727273,
       -10.0000000000, -9.7727272727,  -9.5454545455,  -9.3181818182,
       -9.0909090909,  -8.8636363636,  -8.6363636364,  -8.4090909091,
       -8.1818181818,  -7.9545454545,  -7.7272727273,  -7.5000000000,
       -7.2727272727,  -7.0454545455,  -6.8181818182,  -6.5909090909,
       -6.3636363636,  -6.1363636364,  -5.9090909091,  -5.6818181818,
       -5.4545454545,  -5.2272727273,  -5.0000000000,  -4.7727272727,
       -4.5454545455,  -4.3181818182,  -4.0909090909,  -3.8636363636,
       -3.6363636364,  -3.4090909091,  -3.1818181818,  -2.9545454545,
       -2.7272727273,  -2.5000000000,  -2.2727272727,  -2.0454545455,
       -1.8181818182,  -1.5909090909,  -1.3636363636,  -1.1363636364,
       -0.9090909091,  -0.6818181818,  -0.4545454545,  -0.2272727273,
       0.0000000000,   0.2272727273,   0.4545454545,   0.6818181818,
       0.9090909091,   1.1363636364,   1.3636363636,   1.5909090909,
       1.8181818182,   2.0454545455,   2.2727272727,   2.5000000000,
       2.7272727273,   2.9545454545,   3.1818181818,   3.4090909091,
       3.6363636364,   3.8636363636,   4.0909090909,   4.3181818182,
       4.5454545455,   4.7727272727,   5.0000000000,   5.2272727273,
       5.4545454545,   5.6818181818,   5.9090909091,   6.1363636364,
       6.3636363636,   6.5909090909,   6.8181818182,   7.0454545455,
       7.2727272727,   7.5000000000,   7.7272727273,   7.9545454545,
       8.1818181818,   8.4090909091,   8.6363636364,   8.8636363636,
       9.0909090909,   9.3181818182,   9.5454545455,   9.7727272727,
       10.0000000000,  10.2272727273,  10.4545454545,  10.6818181818,
       10.9090909091,  11.1363636364,  11.3636363636,  11.5909090909,
       11.8181818182,  12.0454545455,  12.2727272727,  12.5000000000,
       12.7272727273,  12.9545454545,  13.1818181818,  13.4090909091,
       13.6363636364,  13.8636363636,  14.0909090909,  14.3181818182,
       14.5454545455,  14.7727272727,  15.0000000000,  15.2272727273,
       15.4545454545,  15.6818181818,  15.9090909091,  16.1363636364,
       16.3636363636,  16.5909090909,  16.8181818182,  17.0454545455,
       17.2727272727,  17.5000000000,  17.7272727273,  17.9545454545,
       18.1818181818,  18.4090909091,  18.6363636364,  18.8636363636,
       19.0909090909,  19.3181818182,  19.5454545455,  19.7727272727,
       20.0000000000});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {-1.0000000000, -1.0000000000, -1.0000000000, -1.0000000000,
                   -1.0000000000, -1.0000000000, -1.0000000000, -1.0000000000,
                   -1.0000000000, -1.0000000000, -1.0000000000, -1.0000000000,
                   -1.0000000000, -1.0000000000, -1.0000000000, -1.0000000000,
                   -1.0000000000, -1.0000000000, -1.0000000000, -1.0000000000,
                   -1.0000000000, -1.0000000000, -1.0000000000, -1.0000000000,
                   -1.0000000000, -1.0000000000, -1.0000000000, -1.0000000000,
                   -1.0000000000, -1.0000000000, -1.0000000000, -1.0000000000,
                   -1.0000000000, -1.0000000000, -1.0000000000, -0.9999999999,
                   -0.9999999999, -0.9999999998, -0.9999999997, -0.9999999996,
                   -0.9999999993, -0.9999999989, -0.9999999983, -0.9999999974,
                   -0.9999999959, -0.9999999935, -0.9999999898, -0.9999999839,
                   -0.9999999746, -0.9999999600, -0.9999999370, -0.9999999007,
                   -0.9999998435, -0.9999997535, -0.9999996117, -0.9999993882,
                   -0.9999990361, -0.9999984815, -0.9999976076, -0.9999962309,
                   -0.9999940619, -0.9999906449, -0.9999852614, -0.9999767801,
                   -0.9999634183, -0.9999423677, -0.9999092043, -0.9998569589,
                   -0.9997746542, -0.9996450004, -0.9994407705, -0.9991190997,
                   -0.9986125310, -0.9978149744, -0.9965597488, -0.9945853915,
                   -0.9914827859, -0.9866142982, -0.9789923110, -0.9671021386,
                   -0.9486568273, -0.9202886021, -0.8772337852, -0.8131859906,
                   -0.7206956332, -0.5927001330, -0.4256281972, -0.2234388228,
                   0.0000000000,  0.2234388228,  0.4256281972,  0.5927001330,
                   0.7206956332,  0.8131859906,  0.8772337852,  0.9202886021,
                   0.9486568273,  0.9671021386,  0.9789923110,  0.9866142982,
                   0.9914827859,  0.9945853915,  0.9965597488,  0.9978149744,
                   0.9986125310,  0.9991190997,  0.9994407705,  0.9996450004,
                   0.9997746542,  0.9998569589,  0.9999092043,  0.9999423677,
                   0.9999634183,  0.9999767801,  0.9999852614,  0.9999906449,
                   0.9999940619,  0.9999962309,  0.9999976076,  0.9999984815,
                   0.9999990361,  0.9999993882,  0.9999996117,  0.9999997535,
                   0.9999998435,  0.9999999007,  0.9999999370,  0.9999999600,
                   0.9999999746,  0.9999999839,  0.9999999898,  0.9999999935,
                   0.9999999959,  0.9999999974,  0.9999999983,  0.9999999989,
                   0.9999999993,  0.9999999996,  0.9999999997,  0.9999999998,
                   0.9999999999,  0.9999999999,  1.0000000000,  1.0000000000,
                   1.0000000000,  1.0000000000,  1.0000000000,  1.0000000000,
                   1.0000000000,  1.0000000000,  1.0000000000,  1.0000000000,
                   1.0000000000,  1.0000000000,  1.0000000000,  1.0000000000,
                   1.0000000000,  1.0000000000,  1.0000000000,  1.0000000000,
                   1.0000000000,  1.0000000000,  1.0000000000,  1.0000000000,
                   1.0000000000,  1.0000000000,  1.0000000000,  1.0000000000,
                   1.0000000000,  1.0000000000,  1.0000000000,  1.0000000000,
                   1.0000000000,  1.0000000000,  1.0000000000,  1.0000000000,
                   1.0000000000},
                  kQuantizedToleranceInt16)));
}

TEST_P(TanhOpTest, TanhInt16General) {
  const float kMin = -1;
  const float kMax = 32767.f / 32768.f;
  QuantizedActivationsOpModel m(
      GetRegistration(), BuiltinOperator_TANH,
      /*input=*/{TensorType_INT16, {10}, 11 * kMin, 11 * kMax},
      /*output=*/{TensorType_INT16, {10}, kMin, kMax});
  m.SetInput<int16_t>({-10, -4, 1, 0.5, 0.25,  //
                       0, -0.1, 6, 7.0909090909, 8});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {-1.0, -0.999329, 0.761594, 0.462117, 0.244919,  //
                   0.0, -0.099668, 0.999988, 0.999999, 1.0},
                  kQuantizedToleranceInt16)));
}

TEST_P(LogisticOpTest, Sigmoid) {
  FloatActivationsOpModel m(GetRegistration(), BuiltinOperator_LOGISTIC,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 0.5, 0.002473, 0.880797, 0.982014,       //
                                 0.952574, 0.119203, 0.999955, 0.731059,  //
                             })));
}

TEST_P(LogisticOpTest, SigmoidUint8) {
  QuantizedActivationsOpModel m(GetRegistration(), BuiltinOperator_LOGISTIC,
                                /*input=*/{TensorType_UINT8, {89}, -10, 10});
  // 64+16+8+1 elements, from -10 to 10
  m.SetInput<uint8_t>(
      {-10.0000000000, -9.7727272727, -9.5454545455, -9.3181818182,
       -9.0909090909,  -8.8636363636, -8.6363636364, -8.4090909091,
       -8.1818181818,  -7.9545454545, -7.7272727273, -7.5000000000,
       -7.2727272727,  -7.0454545455, -6.8181818182, -6.5909090909,
       -6.3636363636,  -6.1363636364, -5.9090909091, -5.6818181818,
       -5.4545454545,  -5.2272727273, -5.0000000000, -4.7727272727,
       -4.5454545455,  -4.3181818182, -4.0909090909, -3.8636363636,
       -3.6363636364,  -3.4090909091, -3.1818181818, -2.9545454545,
       -2.7272727273,  -2.5000000000, -2.2727272727, -2.0454545455,
       -1.8181818182,  -1.5909090909, -1.3636363636, -1.1363636364,
       -0.9090909091,  -0.6818181818, -0.4545454545, -0.2272727273,
       0.0000000000,   0.2272727273,  0.4545454545,  0.6818181818,
       0.9090909091,   1.1363636364,  1.3636363636,  1.5909090909,
       1.8181818182,   2.0454545455,  2.2727272727,  2.5000000000,
       2.7272727273,   2.9545454545,  3.1818181818,  3.4090909091,
       3.6363636364,   3.8636363636,  4.0909090909,  4.3181818182,
       4.5454545455,   4.7727272727,  5.0000000000,  5.2272727273,
       5.4545454545,   5.6818181818,  5.9090909091,  6.1363636364,
       6.3636363636,   6.5909090909,  6.8181818182,  7.0454545455,
       7.2727272727,   7.5000000000,  7.7272727273,  7.9545454545,
       8.1818181818,   8.4090909091,  8.6363636364,  8.8636363636,
       9.0909090909,   9.3181818182,  9.5454545455,  9.7727272727,
       10.0000000000});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput<uint8_t>(),
      ElementsAreArray(ArrayFloatNear(
          {0.0000453979, 0.0000569815, 0.0000715205, 0.0000897689, 0.0001126729,
           0.0001414198, 0.0001774998, 0.0002227827, 0.0002796147, 0.0003509396,
           0.0004404502, 0.0005527786, 0.0006937345, 0.0008706021, 0.0010925128,
           0.0013709094, 0.0017201256, 0.0021581065, 0.0027073042, 0.0033957870,
           0.0042586071, 0.0053394826, 0.0066928509, 0.0083863576, 0.0105038445,
           0.0131488902, 0.0164489307, 0.0205599431, 0.0256715863, 0.0320125562,
           0.0398556989, 0.0495221198, 0.0613831074, 0.0758581800, 0.0934070047,
           0.1145124805, 0.1396521834, 0.1692560327, 0.2036499335, 0.2429886272,
           0.2871859014, 0.3358556241, 0.3882805886, 0.4434251301, 0.5000000000,
           0.5565748699, 0.6117194114, 0.6641443759, 0.7128140986, 0.7570113728,
           0.7963500665, 0.8307439673, 0.8603478166, 0.8854875195, 0.9065929953,
           0.9241418200, 0.9386168926, 0.9504778802, 0.9601443011, 0.9679874438,
           0.9743284137, 0.9794400569, 0.9835510693, 0.9868511098, 0.9894961555,
           0.9916136424, 0.9933071491, 0.9946605174, 0.9957413929, 0.9966042130,
           0.9972926958, 0.9978418935, 0.9982798744, 0.9986290906, 0.9989074872,
           0.9991293979, 0.9993062655, 0.9994472214, 0.9995595498, 0.9996490604,
           0.9997203853, 0.9997772173, 0.9998225002, 0.9998585802, 0.9998873271,
           0.9999102311, 0.9999284795, 0.9999430185, 0.9999546021},
          kQuantizedTolerance)));
}

TEST_P(LogisticOpTest, SigmoidInt8) {
  QuantizedActivationsOpModel m(GetRegistration(), BuiltinOperator_LOGISTIC,
                                /*input=*/{TensorType_INT8, {89}, -10, 10});
  // 64+16+8+1 elements, from -10 to 10
  m.SetInput<int8_t>(
      {-10.0000000000, -9.7727272727, -9.5454545455, -9.3181818182,
       -9.0909090909,  -8.8636363636, -8.6363636364, -8.4090909091,
       -8.1818181818,  -7.9545454545, -7.7272727273, -7.5000000000,
       -7.2727272727,  -7.0454545455, -6.8181818182, -6.5909090909,
       -6.3636363636,  -6.1363636364, -5.9090909091, -5.6818181818,
       -5.4545454545,  -5.2272727273, -5.0000000000, -4.7727272727,
       -4.5454545455,  -4.3181818182, -4.0909090909, -3.8636363636,
       -3.6363636364,  -3.4090909091, -3.1818181818, -2.9545454545,
       -2.7272727273,  -2.5000000000, -2.2727272727, -2.0454545455,
       -1.8181818182,  -1.5909090909, -1.3636363636, -1.1363636364,
       -0.9090909091,  -0.6818181818, -0.4545454545, -0.2272727273,
       0.0000000000,   0.2272727273,  0.4545454545,  0.6818181818,
       0.9090909091,   1.1363636364,  1.3636363636,  1.5909090909,
       1.8181818182,   2.0454545455,  2.2727272727,  2.5000000000,
       2.7272727273,   2.9545454545,  3.1818181818,  3.4090909091,
       3.6363636364,   3.8636363636,  4.0909090909,  4.3181818182,
       4.5454545455,   4.7727272727,  5.0000000000,  5.2272727273,
       5.4545454545,   5.6818181818,  5.9090909091,  6.1363636364,
       6.3636363636,   6.5909090909,  6.8181818182,  7.0454545455,
       7.2727272727,   7.5000000000,  7.7272727273,  7.9545454545,
       8.1818181818,   8.4090909091,  8.6363636364,  8.8636363636,
       9.0909090909,   9.3181818182,  9.5454545455,  9.7727272727,
       10.0000000000});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput<int8_t>(),
      ElementsAreArray(ArrayFloatNear(
          {0.0000453979, 0.0000569815, 0.0000715205, 0.0000897689, 0.0001126729,
           0.0001414198, 0.0001774998, 0.0002227827, 0.0002796147, 0.0003509396,
           0.0004404502, 0.0005527786, 0.0006937345, 0.0008706021, 0.0010925128,
           0.0013709094, 0.0017201256, 0.0021581065, 0.0027073042, 0.0033957870,
           0.0042586071, 0.0053394826, 0.0066928509, 0.0083863576, 0.0105038445,
           0.0131488902, 0.0164489307, 0.0205599431, 0.0256715863, 0.0320125562,
           0.0398556989, 0.0495221198, 0.0613831074, 0.0758581800, 0.0934070047,
           0.1145124805, 0.1396521834, 0.1692560327, 0.2036499335, 0.2429886272,
           0.2871859014, 0.3358556241, 0.3882805886, 0.4434251301, 0.5000000000,
           0.5565748699, 0.6117194114, 0.6641443759, 0.7128140986, 0.7570113728,
           0.7963500665, 0.8307439673, 0.8603478166, 0.8854875195, 0.9065929953,
           0.9241418200, 0.9386168926, 0.9504778802, 0.9601443011, 0.9679874438,
           0.9743284137, 0.9794400569, 0.9835510693, 0.9868511098, 0.9894961555,
           0.9916136424, 0.9933071491, 0.9946605174, 0.9957413929, 0.9966042130,
           0.9972926958, 0.9978418935, 0.9982798744, 0.9986290906, 0.9989074872,
           0.9991293979, 0.9993062655, 0.9994472214, 0.9995595498, 0.9996490604,
           0.9997203853, 0.9997772173, 0.9998225002, 0.9998585802, 0.9998873271,
           0.9999102311, 0.9999284795, 0.9999430185, 0.9999546021},
          kQuantizedTolerance)));
}

TEST_P(LogisticOpTest, SigmoidInt16) {
  const float kMin = -1;
  const float kMax = 32767.f / 32768.f;
  QuantizedActivationsOpModel m(
      GetRegistration(), BuiltinOperator_LOGISTIC,
      /*input=*/{TensorType_INT16, {177}, 16 * kMin, 16 * kMax},
      /*output=*/{TensorType_INT16, {177}, kMin, kMax});
  m.SetInput<int16_t>(
      {-20.0000000000, -19.7727272727, -19.5454545455, -19.3181818182,
       -19.0909090909, -18.8636363636, -18.6363636364, -18.4090909091,
       -18.1818181818, -17.9545454545, -17.7272727273, -17.5000000000,
       -17.2727272727, -17.0454545455, -16.8181818182, -16.5909090909,
       -16.3636363636, -16.1363636364, -15.9090909091, -15.6818181818,
       -15.4545454545, -15.2272727273, -15.0000000000, -14.7727272727,
       -14.5454545455, -14.3181818182, -14.0909090909, -13.8636363636,
       -13.6363636364, -13.4090909091, -13.1818181818, -12.9545454545,
       -12.7272727273, -12.5000000000, -12.2727272727, -12.0454545455,
       -11.8181818182, -11.5909090909, -11.3636363636, -11.1363636364,
       -10.9090909091, -10.6818181818, -10.4545454545, -10.2272727273,
       -10.0000000000, -9.7727272727,  -9.5454545455,  -9.3181818182,
       -9.0909090909,  -8.8636363636,  -8.6363636364,  -8.4090909091,
       -8.1818181818,  -7.9545454545,  -7.7272727273,  -7.5000000000,
       -7.2727272727,  -7.0454545455,  -6.8181818182,  -6.5909090909,
       -6.3636363636,  -6.1363636364,  -5.9090909091,  -5.6818181818,
       -5.4545454545,  -5.2272727273,  -5.0000000000,  -4.7727272727,
       -4.5454545455,  -4.3181818182,  -4.0909090909,  -3.8636363636,
       -3.6363636364,  -3.4090909091,  -3.1818181818,  -2.9545454545,
       -2.7272727273,  -2.5000000000,  -2.2727272727,  -2.0454545455,
       -1.8181818182,  -1.5909090909,  -1.3636363636,  -1.1363636364,
       -0.9090909091,  -0.6818181818,  -0.4545454545,  -0.2272727273,
       0.0000000000,   0.2272727273,   0.4545454545,   0.6818181818,
       0.9090909091,   1.1363636364,   1.3636363636,   1.5909090909,
       1.8181818182,   2.0454545455,   2.2727272727,   2.5000000000,
       2.7272727273,   2.9545454545,   3.1818181818,   3.4090909091,
       3.6363636364,   3.8636363636,   4.0909090909,   4.3181818182,
       4.5454545455,   4.7727272727,   5.0000000000,   5.2272727273,
       5.4545454545,   5.6818181818,   5.9090909091,   6.1363636364,
       6.3636363636,   6.5909090909,   6.8181818182,   7.0454545455,
       7.2727272727,   7.5000000000,   7.7272727273,   7.9545454545,
       8.1818181818,   8.4090909091,   8.6363636364,   8.8636363636,
       9.0909090909,   9.3181818182,   9.5454545455,   9.7727272727,
       10.0000000000,  10.2272727273,  10.4545454545,  10.6818181818,
       10.9090909091,  11.1363636364,  11.3636363636,  11.5909090909,
       11.8181818182,  12.0454545455,  12.2727272727,  12.5000000000,
       12.7272727273,  12.9545454545,  13.1818181818,  13.4090909091,
       13.6363636364,  13.8636363636,  14.0909090909,  14.3181818182,
       14.5454545455,  14.7727272727,  15.0000000000,  15.2272727273,
       15.4545454545,  15.6818181818,  15.9090909091,  16.1363636364,
       16.3636363636,  16.5909090909,  16.8181818182,  17.0454545455,
       17.2727272727,  17.5000000000,  17.7272727273,  17.9545454545,
       18.1818181818,  18.4090909091,  18.6363636364,  18.8636363636,
       19.0909090909,  19.3181818182,  19.5454545455,  19.7727272727,
       20.0000000000});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput<int16_t>(),
      ElementsAreArray(ArrayFloatNear(
          {0.0000000021, 0.0000000026, 0.0000000032, 0.0000000041, 0.0000000051,
           0.0000000064, 0.0000000081, 0.0000000101, 0.0000000127, 0.0000000159,
           0.0000000200, 0.0000000251, 0.0000000315, 0.0000000396, 0.0000000497,
           0.0000000623, 0.0000000782, 0.0000000982, 0.0000001232, 0.0000001547,
           0.0000001942, 0.0000002437, 0.0000003059, 0.0000003840, 0.0000004819,
           0.0000006049, 0.0000007593, 0.0000009530, 0.0000011962, 0.0000015014,
           0.0000018846, 0.0000023654, 0.0000029690, 0.0000037266, 0.0000046776,
           0.0000058711, 0.0000073693, 0.0000092497, 0.0000116100, 0.0000145724,
           0.0000182909, 0.0000229581, 0.0000288162, 0.0000361690, 0.0000453979,
           0.0000569815, 0.0000715205, 0.0000897689, 0.0001126729, 0.0001414198,
           0.0001774998, 0.0002227827, 0.0002796147, 0.0003509396, 0.0004404502,
           0.0005527786, 0.0006937345, 0.0008706021, 0.0010925128, 0.0013709094,
           0.0017201256, 0.0021581065, 0.0027073042, 0.0033957870, 0.0042586071,
           0.0053394826, 0.0066928509, 0.0083863576, 0.0105038445, 0.0131488902,
           0.0164489307, 0.0205599431, 0.0256715863, 0.0320125562, 0.0398556989,
           0.0495221198, 0.0613831074, 0.0758581800, 0.0934070047, 0.1145124805,
           0.1396521834, 0.1692560327, 0.2036499335, 0.2429886272, 0.2871859014,
           0.3358556241, 0.3882805886, 0.4434251301, 0.5000000000, 0.5565748699,
           0.6117194114, 0.6641443759, 0.7128140986, 0.7570113728, 0.7963500665,
           0.8307439673, 0.8603478166, 0.8854875195, 0.9065929953, 0.9241418200,
           0.9386168926, 0.9504778802, 0.9601443011, 0.9679874438, 0.9743284137,
           0.9794400569, 0.9835510693, 0.9868511098, 0.9894961555, 0.9916136424,
           0.9933071491, 0.9946605174, 0.9957413929, 0.9966042130, 0.9972926958,
           0.9978418935, 0.9982798744, 0.9986290906, 0.9989074872, 0.9991293979,
           0.9993062655, 0.9994472214, 0.9995595498, 0.9996490604, 0.9997203853,
           0.9997772173, 0.9998225002, 0.9998585802, 0.9998873271, 0.9999102311,
           0.9999284795, 0.9999430185, 0.9999546021, 0.9999638310, 0.9999711838,
           0.9999770419, 0.9999817091, 0.9999854276, 0.9999883900, 0.9999907503,
           0.9999926307, 0.9999941289, 0.9999953224, 0.9999962734, 0.9999970310,
           0.9999976346, 0.9999981154, 0.9999984986, 0.9999988038, 0.9999990470,
           0.9999992407, 0.9999993951, 0.9999995181, 0.9999996160, 0.9999996941,
           0.9999997563, 0.9999998058, 0.9999998453, 0.9999998768, 0.9999999018,
           0.9999999218, 0.9999999377, 0.9999999503, 0.9999999604, 0.9999999685,
           0.9999999749, 0.9999999800, 0.9999999841, 0.9999999873, 0.9999999899,
           0.9999999919, 0.9999999936, 0.9999999949, 0.9999999959, 0.9999999968,
           0.9999999974, 0.9999999979},
          kQuantizedToleranceInt16)));
}

TEST_P(LogisticOpTest, SigmoidInt16General) {
  const float kMin = -1;
  const float kMax = 32767.f / 32768.f;
  QuantizedActivationsOpModel m(
      GetRegistration(), BuiltinOperator_LOGISTIC,
      /*input=*/{TensorType_INT16, {12}, 13 * kMin, 13 * kMax},
      /*output=*/{TensorType_INT16, {12}, kMin, kMax});
  m.SetInput<int16_t>({
      0, -6, 2, 4, 0.1, 12,    //
      3, -2, 10, 1, 0.25, -12  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {0.5, 0.002473, 0.880797, 0.982014, 0.524979, 0.999994,  //
                   0.952574, 0.119203, 0.999955, 0.731059, 0.562177, 0},
                  kQuantizedToleranceInt16)));
}

TEST_P(SoftmaxOpTest, Softmax4D) {
  FloatActivationsOpModel m(GetRegistration(), 0.1f,
                            {TensorType_FLOAT32, {1, 2, 1, 4}},
                            TensorType_FLOAT32);
  m.SetInput({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 .23463, .12877, .28658, .35003,  //
                                 .22528, .13664, .45365, .18443,  //
                             })));

  // Same input, but a different shape.
  FloatActivationsOpModel m2(GetRegistration(), 0.1f,
                             {TensorType_FLOAT32, {4, 1, 1, 2}},
                             TensorType_FLOAT32);
  m2.SetInput({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  ASSERT_EQ(m2.Invoke(), kTfLiteOk);
  EXPECT_THAT(m2.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                  0.645656, 0.354344,  //
                                  0.450166, 0.549834,  //
                                  0.622459, 0.377541,  //
                                  0.710949, 0.28905,   //
                              })));
}

TEST_P(SoftmaxOpTest, Softmax4DUint8) {
  QuantizedActivationsOpModel m(GetRegistration(), 0.1f,
                                {TensorType_UINT8, {1, 2, 1, 4}, -10, 10},
                                TensorType_UINT8);
  m.SetInput<uint8_t>({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(GetRegistration(), 0.1f,
                                 {TensorType_UINT8, {4, 1, 1, 2}, -10, 10},
                                 TensorType_UINT8);
  m2.SetInput<uint8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  ASSERT_EQ(m2.Invoke(), kTfLiteOk);
  EXPECT_THAT(m2.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.645656, 0.354344,  //
                      0.450166, 0.549834,  //
                      0.622459, 0.377541,  //
                      0.710949, 0.28905,   //
                  },
                  kQuantizedTolerance)));
}

TEST_P(SoftmaxOpTest, Softmax4DUint8Int16) {
  QuantizedActivationsOpModel m(GetRegistration(), 0.1f,
                                {TensorType_UINT8, {1, 2, 1, 4}, -10, 10},
                                TensorType_INT16);
  m.SetInput<uint8_t>({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(GetRegistration(), 0.1f,
                                 {TensorType_UINT8, {4, 1, 1, 2}, -10, 10},
                                 TensorType_INT16);
  m2.SetInput<uint8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  ASSERT_EQ(m2.Invoke(), kTfLiteOk);
  EXPECT_THAT(m2.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.645656, 0.354344,  //
                      0.450166, 0.549834,  //
                      0.622459, 0.377541,  //
                      0.710949, 0.28905,   //
                  },
                  kQuantizedTolerance)));
}

// Test quantized softmax with int8 input and output. With the same input as in
// QuantizedActivationsOpTest.Softmax1D, the dequantized output is identical.
TEST_P(SoftmaxOpTest, Softmax1DInt8) {
  QuantizedActivationsOpModel m(
      GetRegistration(), 0.1, {TensorType_INT8, {8}, -10, 10}, TensorType_INT8);
  m.SetInput<int8_t>({0, -6, 2, 4, 3, -2, 10, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput<int8_t>(),
      ElementsAreArray(ArrayFloatNear({0.09766, 0.05469, 0.12109, 0.14453,
                                       0.13281, 0.07813, 0.26563, 0.10938},
                                      kQuantizedTolerance)));
}

// Test quantized softmax with int16 input and output. With the same input as in
// QuantizedActivationsOpTest.Softmax2D, the dequantized output is identical.
TEST_P(SoftmaxOpTest, Softmax1DInt16) {
  const float kMin = -1;
  const float kMax = 32767.f / 32768.f;
  QuantizedActivationsOpModel m(
      GetRegistration(), 1,
      /*input=*/{TensorType_INT16, {3}, 3 * kMin, 3 * kMax},
      /*output_type-*/ TensorType_INT16);
  m.SetInput<int16_t>({1, 2, 3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput<int16_t>(),
      ElementsAreArray(ArrayFloatNear({0.0900269, 0.2447285, 0.66524096},
                                      kQuantizedToleranceInt16)));
}

TEST_P(SoftmaxOpTest, Softmax1DInt16ZeroElement) {
  const float kMin = -1;
  const float kMax = 32767.f / 32768.f;
  QuantizedActivationsOpModel m(
      GetRegistration(), 0.1,
      /*input=*/{TensorType_INT16, {1}, 1 * kMin, 1 * kMax}, TensorType_INT16);
  m.SetInput<int16_t>({0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear({1}, kQuantizedToleranceInt16)));
}

TEST_P(SoftmaxOpTest, Softmax2DInt16) {
  const float kMin = -1;
  const float kMax = 32767.f / 32768.f;
  QuantizedActivationsOpModel m(
      GetRegistration(), 0.1,
      /*input=*/{TensorType_INT16, {2, 4}, 10 * kMin, 10 * kMax},
      TensorType_INT16);
  m.SetInput<int16_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedToleranceInt16)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(
      GetRegistration(), 0.1,
      /*input=*/{TensorType_INT16, {4, 2}, 10 * kMin, 10 * kMax},
      TensorType_INT16);
  m2.SetInput<int16_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  ASSERT_EQ(m2.Invoke(), kTfLiteOk);
  EXPECT_THAT(m2.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.645656, 0.354344,  //
                      0.450166, 0.549834,  //
                      0.622459, 0.377541,  //
                      0.710949, 0.28905,   //
                  },
                  kQuantizedToleranceInt16)));
}

TEST_P(SoftmaxOpTest, Softmax3DInt16) {
  const float kMin = -1;
  const float kMax = 32767.f / 32768.f;
  QuantizedActivationsOpModel m(
      GetRegistration(), 1,
      /*input=*/{TensorType_INT16, {1, 2, 4}, 10 * kMin, 10 * kMax},
      TensorType_INT16);
  m.SetInput<int16_t>({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .0158756, .000039, .1173, .866779,   //
                      .00091, .0000061, .998959, .000123,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(
      GetRegistration(), 1,
      /*input=*/{TensorType_INT16, {4, 1, 2}, 10 * kMin, 10 * kMax},
      TensorType_INT16);
  m2.SetInput<int16_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  ASSERT_EQ(m2.Invoke(), kTfLiteOk);
  EXPECT_THAT(m2.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.997527, 0.0024726,       //
                      0.11920292, 0.88079707,    //
                      0.99330715, 0.00669285,    //
                      0.999876605, 0.000123395,  //
                  },
                  kQuantizedTolerance)));
}

// Test quantized softmax with int16 input and output. With the same input as in
// QuantizedActivationsOpTest.Softmax4D, the dequantized output is identical.
TEST_P(SoftmaxOpTest, Softmax4DInt16) {
  const float kMin = -1;
  const float kMax = 32767.f / 32768.f;
  QuantizedActivationsOpModel m(
      GetRegistration(), 0.1,
      /*input=*/{TensorType_INT16, {1, 2, 1, 4}, 10 * kMin, 10 * kMax},
      TensorType_INT16);
  m.SetInput<int16_t>({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedToleranceInt16)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(
      GetRegistration(), 0.1,
      /*input=*/{TensorType_INT16, {4, 1, 1, 2}, 10 * kMin, 10 * kMax},
      TensorType_INT16);
  m2.SetInput<int16_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  ASSERT_EQ(m2.Invoke(), kTfLiteOk);
  EXPECT_THAT(m2.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.645656, 0.354344,  //
                      0.450166, 0.549834,  //
                      0.622459, 0.377541,  //
                      0.710949, 0.28905,   //
                  },
                  kQuantizedToleranceInt16)));
}

// Test quantized softmax with int8 input and int16 output. With the same input
// as in QuantizedActivationsOpTest.Softmax1D, the dequantized output is
// identical.
TEST_P(SoftmaxOpTest, Softmax1DInt8Int16) {
  QuantizedActivationsOpModel m(GetRegistration(), 0.1f,
                                {TensorType_INT8, {8}, -10, 10},
                                TensorType_INT16);
  m.SetInput<int8_t>({0, -6, 2, 4, 3, -2, 10, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput<int16_t>(),
      ElementsAreArray(ArrayFloatNear({0.09766, 0.05469, 0.12109, 0.14453,
                                       0.13281, 0.07813, 0.26563, 0.10938},
                                      kQuantizedTolerance)));
}

// Test quantized softmax with int8 input and output. With the same input as in
// QuantizedActivationsOpTest.Softmax2D, the dequantized output is identical.
TEST_P(SoftmaxOpTest, Softmax2DInt8) {
  QuantizedActivationsOpModel m(GetRegistration(), 0.1f,
                                {TensorType_INT8, {2, 4}, -10, 10},
                                TensorType_INT8);
  m.SetInput<int8_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(GetRegistration(), 0.1f,
                                 {TensorType_INT8, {4, 2}, -10, 10},
                                 TensorType_INT8);
  m2.SetInput<int8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  ASSERT_EQ(m2.Invoke(), kTfLiteOk);
  EXPECT_THAT(m2.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.645656, 0.354344,  //
                      0.450166, 0.549834,  //
                      0.622459, 0.377541,  //
                      0.710949, 0.28905,   //
                  },
                  kQuantizedTolerance)));
}

// Test quantized softmax with int8 input and int16 output. With the same input
// as in QuantizedActivationsOpTest.Softmax2D, the dequantized output is
// identical.
TEST_P(SoftmaxOpTest, Softmax2DInt8Int16) {
  QuantizedActivationsOpModel m(GetRegistration(), 0.1f,
                                {TensorType_INT8, {2, 4}, -10, 10},
                                TensorType_INT16);
  m.SetInput<int8_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(GetRegistration(), 0.1f,
                                 {TensorType_INT8, {4, 2}, -10, 10},
                                 TensorType_INT16);
  m2.SetInput<int8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  ASSERT_EQ(m2.Invoke(), kTfLiteOk);
  EXPECT_THAT(m2.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.645656, 0.354344,  //
                      0.450166, 0.549834,  //
                      0.622459, 0.377541,  //
                      0.710949, 0.28905,   //
                  },
                  kQuantizedTolerance)));
}

// Test quantized softmax with int8 input and output. With the same input as in
// QuantizedActivationsOpTest.Softmax3D, the dequantized output is identical.
TEST_P(SoftmaxOpTest, Softmax3DInt8) {
  QuantizedActivationsOpModel m(GetRegistration(), 0.1f,
                                {TensorType_INT8, {1, 2, 4}, -10, 10},
                                TensorType_INT8);
  m.SetInput<int8_t>({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(GetRegistration(), 0.1f,
                                 {TensorType_INT8, {4, 1, 2}, -10, 10},
                                 TensorType_INT8);
  m2.SetInput<int8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  ASSERT_EQ(m2.Invoke(), kTfLiteOk);
  EXPECT_THAT(m2.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.645656, 0.354344,  //
                      0.450166, 0.549834,  //
                      0.622459, 0.377541,  //
                      0.710949, 0.28905,   //
                  },
                  kQuantizedTolerance)));
}

// Test quantized softmax with int8 input and output. With the same input as in
// QuantizedActivationsOpTest.Softmax3D, the dequantized output is identical.
TEST_P(SoftmaxOpTest, Softmax3DInt8Int16) {
  QuantizedActivationsOpModel m(GetRegistration(), 0.1f,
                                {TensorType_INT8, {1, 2, 4}, -10, 10},
                                TensorType_INT16);
  m.SetInput<int8_t>({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(GetRegistration(), 0.1f,
                                 {TensorType_INT8, {4, 1, 2}, -10, 10},
                                 TensorType_INT16);
  m2.SetInput<int8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  ASSERT_EQ(m2.Invoke(), kTfLiteOk);
  EXPECT_THAT(m2.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.645656, 0.354344,  //
                      0.450166, 0.549834,  //
                      0.622459, 0.377541,  //
                      0.710949, 0.28905,   //
                  },
                  kQuantizedTolerance)));
}

// Test quantized softmax with int8 input and output. With the same input as in
// QuantizedActivationsOpTest.Softmax4D, the dequantized output is identical.
TEST_P(SoftmaxOpTest, Softmax4DInt8) {
  QuantizedActivationsOpModel m(GetRegistration(), 0.1f,
                                {TensorType_INT8, {1, 2, 1, 4}, -10, 10},
                                TensorType_INT8);
  m.SetInput<int8_t>({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({
                                         -68, -95, -54, -38,  //
                                         -70, -93, -12, -81,  //
                                     }));
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(GetRegistration(), 0.1f,
                                 {TensorType_INT8, {4, 1, 1, 2}, -10, 10},
                                 TensorType_INT8);
  m2.SetInput<int8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  ASSERT_EQ(m2.Invoke(), kTfLiteOk);
  EXPECT_THAT(m2.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.645656, 0.354344,  //
                      0.450166, 0.549834,  //
                      0.622459, 0.377541,  //
                      0.710949, 0.28905,   //
                  },
                  kQuantizedTolerance)));
}

// Test quantized softmax with int8 input and output. With the same input as in
// QuantizedActivationsOpTest.Softmax4D, the dequantized output is identical.
TEST_P(SoftmaxOpTest, Softmax4DInt8Int16) {
  QuantizedActivationsOpModel m(GetRegistration(), 0.1f,
                                {TensorType_INT8, {1, 2, 1, 4}, -10, 10},
                                TensorType_INT16);
  m.SetInput<int8_t>({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(GetRegistration(), 0.1f,
                                 {TensorType_INT8, {4, 1, 1, 2}, -10, 10},
                                 TensorType_INT16);
  m2.SetInput<int8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  ASSERT_EQ(m2.Invoke(), kTfLiteOk);
  EXPECT_THAT(m2.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.645656, 0.354344,  //
                      0.450166, 0.549834,  //
                      0.622459, 0.377541,  //
                      0.710949, 0.28905,   //
                  },
                  kQuantizedTolerance)));
}

TEST_P(SoftmaxOpTest, Softmax3D) {
  FloatActivationsOpModel m(GetRegistration(), 0.1f,
                            {TensorType_FLOAT32, {1, 2, 4}},
                            TensorType_FLOAT32);
  m.SetInput({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 .23463, .12877, .28658, .35003,  //
                                 .22528, .13664, .45365, .18443,  //
                             })));

  // Same input, but a different shape.
  FloatActivationsOpModel m2(GetRegistration(), 0.1f,
                             {TensorType_FLOAT32, {4, 1, 2}},
                             TensorType_FLOAT32);
  m2.SetInput({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  ASSERT_EQ(m2.Invoke(), kTfLiteOk);
  EXPECT_THAT(m2.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                  0.645656, 0.354344,  //
                                  0.450166, 0.549834,  //
                                  0.622459, 0.377541,  //
                                  0.710949, 0.28905,   //
                              })));
}

TEST_P(SoftmaxOpTest, Softmax3DUint8) {
  QuantizedActivationsOpModel m(GetRegistration(), 0.1f,
                                {TensorType_UINT8, {1, 2, 4}, -10, 10},
                                TensorType_UINT8);
  m.SetInput<uint8_t>({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(GetRegistration(), 0.1f,
                                 {TensorType_UINT8, {4, 1, 2}, -10, 10},
                                 TensorType_UINT8);
  m2.SetInput<uint8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  ASSERT_EQ(m2.Invoke(), kTfLiteOk);
  EXPECT_THAT(m2.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.645656, 0.354344,  //
                      0.450166, 0.549834,  //
                      0.622459, 0.377541,  //
                      0.710949, 0.28905,   //
                  },
                  kQuantizedTolerance)));
}

TEST_P(SoftmaxOpTest, Softmax3DUint8Int16) {
  QuantizedActivationsOpModel m(GetRegistration(), 0.1f,
                                {TensorType_UINT8, {1, 2, 4}, -10, 10},
                                TensorType_INT16);
  m.SetInput<uint8_t>({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(GetRegistration(), 0.1f,
                                 {TensorType_UINT8, {4, 1, 2}, -10, 10},
                                 TensorType_INT16);
  m2.SetInput<uint8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  ASSERT_EQ(m2.Invoke(), kTfLiteOk);
  EXPECT_THAT(m2.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.645656, 0.354344,  //
                      0.450166, 0.549834,  //
                      0.622459, 0.377541,  //
                      0.710949, 0.28905,   //
                  },
                  kQuantizedTolerance)));
}

TEST_P(SoftmaxOpTest, Softmax1D) {
  FloatActivationsOpModel m(GetRegistration(), 0.1f, {TensorType_FLOAT32, {8}},
                            TensorType_FLOAT32);
  m.SetInput({0, -6, 2, 4, 3, -2, 10, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          {.09752, .05352, .11911, .14548, .13164, .07984, .26509, .10778})));
}

TEST_P(SoftmaxOpTest, Softmax1DMax) {
  FloatActivationsOpModel m(GetRegistration(), 0.1f, {TensorType_FLOAT32, {8}},
                            TensorType_FLOAT32);
  m.SetInput({std::numeric_limits<float>::max(), -6, 2, 4, 3, -2, 10, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({1, 0, 0, 0, 0, 0, 0, 0})));
}

TEST_P(SoftmaxOpTest, Softmax1DInf) {
  FloatActivationsOpModel m(GetRegistration(), 0.1f, {TensorType_FLOAT32, {8}},
                            TensorType_FLOAT32);
  m.SetInput({std::numeric_limits<float>::infinity(), -6, 2, 4, 3, -2, 10, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  auto output = m.GetOutput();
  for (int i = 0; i < 8; ++i) {
    EXPECT_TRUE(std::isnan(output[i]));
  }
}

TEST_P(SoftmaxOpTest, Softmax1DUint8) {
  QuantizedActivationsOpModel m(GetRegistration(), 0.1f,
                                {TensorType_UINT8, {8}, -10, 10},
                                TensorType_UINT8);
  m.SetInput<uint8_t>({0, -6, 2, 4, 3, -2, 10, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput<uint8_t>(),
      ElementsAreArray(ArrayFloatNear({0.09766, 0.05469, 0.12109, 0.14453,
                                       0.13281, 0.07813, 0.26563, 0.10938},
                                      kQuantizedTolerance)));
}

TEST_P(SoftmaxOpTest, Softmax1DUint8Int16) {
  QuantizedActivationsOpModel m(GetRegistration(), 0.1f,
                                {TensorType_UINT8, {8}, -10, 10},
                                TensorType_INT16);
  m.SetInput<uint8_t>({0, -6, 2, 4, 3, -2, 10, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput<int16_t>(),
      ElementsAreArray(ArrayFloatNear({0.09766, 0.05469, 0.12109, 0.14453,
                                       0.13281, 0.07813, 0.26563, 0.10938},
                                      kQuantizedTolerance)));
}

TEST_P(SoftmaxOpTest, Softmax2D) {
  FloatActivationsOpModel m(GetRegistration(), 0.1f,
                            {TensorType_FLOAT32, {2, 4}}, TensorType_FLOAT32);
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 .23463, .12877, .28658, .35003,  //
                                 .22528, .13664, .45365, .18443,  //
                             })));

  // Same input, but a different shape.
  FloatActivationsOpModel m2(GetRegistration(), 0.1f,
                             {TensorType_FLOAT32, {4, 2}}, TensorType_FLOAT32);
  m2.SetInput({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  ASSERT_EQ(m2.Invoke(), kTfLiteOk);
  EXPECT_THAT(m2.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                  0.645656, 0.354344,  //
                                  0.450166, 0.549834,  //
                                  0.622459, 0.377541,  //
                                  0.710949, 0.28905,   //
                              })));
}

TEST_P(SoftmaxOpTest, Softmax2DMultithreading) {
  FloatActivationsOpModel m(GetRegistration(), 0.1f,
                            {TensorType_FLOAT32, {16, 4}}, TensorType_FLOAT32);
  m.SetInput({
      0, -6, 2,  4,  //  Thread 1.
      3, -2, 10, 1,  //
      0, -6, 2,  4,  //
      3, -2, 10, 1,  //
      0, -6, 2,  4,  //
      3, -2, 10, 1,  //
      0, -6, 2,  4,  //
      3, -2, 10, 1,  //
      3, -2, 10, 1,  //  Thread 2.
      0, -6, 2,  4,  //
      3, -2, 10, 1,  //
      0, -6, 2,  4,  //
      3, -2, 10, 1,  //
      0, -6, 2,  4,  //
      3, -2, 10, 1,  //
      0, -6, 2,  4,  //
  });
  m.SetNumThreads(2);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 .23463, .12877, .28658, .35003,  //
                                 .22528, .13664, .45365, .18443,  //
                                 .23463, .12877, .28658, .35003,  //
                                 .22528, .13664, .45365, .18443,  //
                                 .23463, .12877, .28658, .35003,  //
                                 .22528, .13664, .45365, .18443,  //
                                 .23463, .12877, .28658, .35003,  //
                                 .22528, .13664, .45365, .18443,  //
                                 .22528, .13664, .45365, .18443,  //
                                 .23463, .12877, .28658, .35003,  //
                                 .22528, .13664, .45365, .18443,  //
                                 .23463, .12877, .28658, .35003,  //
                                 .22528, .13664, .45365, .18443,  //
                                 .23463, .12877, .28658, .35003,  //
                                 .22528, .13664, .45365, .18443,  //
                                 .23463, .12877, .28658, .35003,  //
                             })));

  // Same input, but a different shape.
  FloatActivationsOpModel m2(GetRegistration(), 0.1f,
                             {TensorType_FLOAT32, {16, 2}}, TensorType_FLOAT32);
  m2.SetInput({
      0,  -6,  // Thread 1
      2,  4,   //
      3,  -2,  //
      10, 1,   //
      0,  -6,  //
      2,  4,   //
      3,  -2,  //
      10, 1,   //
      10, 1,   // Thread 2
      3,  -2,  //
      2,  4,   //
      0,  -6,  //
      10, 1,   //
      3,  -2,  //
      2,  4,   //
      0,  -6,  //
  });
  m2.SetNumThreads(2);
  ASSERT_EQ(m2.Invoke(), kTfLiteOk);
  EXPECT_THAT(m2.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                  0.645656, 0.354344,  //
                                  0.450166, 0.549834,  //
                                  0.622459, 0.377541,  //
                                  0.710949, 0.28905,   //
                                  0.645656, 0.354344,  //
                                  0.450166, 0.549834,  //
                                  0.622459, 0.377541,  //
                                  0.710949, 0.28905,   //
                                  0.710949, 0.28905,   //
                                  0.622459, 0.377541,  //
                                  0.450166, 0.549834,  //
                                  0.645656, 0.354344,  //
                                  0.710949, 0.28905,   //
                                  0.622459, 0.377541,  //
                                  0.450166, 0.549834,  //
                                  0.645656, 0.354344,  //
                              })));
}

TEST_P(SoftmaxOpTest, Softmax2DUint8) {
  QuantizedActivationsOpModel m(GetRegistration(), 0.1f,
                                {TensorType_UINT8, {2, 4}, -10, 10},
                                TensorType_UINT8);
  m.SetInput<uint8_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(GetRegistration(), 0.1f,
                                 {TensorType_UINT8, {4, 2}, -10, 10},
                                 TensorType_UINT8);
  m2.SetInput<uint8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  ASSERT_EQ(m2.Invoke(), kTfLiteOk);
  EXPECT_THAT(m2.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.645656, 0.354344,  //
                      0.450166, 0.549834,  //
                      0.622459, 0.377541,  //
                      0.710949, 0.28905,   //
                  },
                  kQuantizedTolerance)));
}

TEST_P(SoftmaxOpTest, Softmax2DUint8Int16) {
  QuantizedActivationsOpModel m(GetRegistration(), 0.1f,
                                {TensorType_UINT8, {2, 4}, -10, 10},
                                TensorType_INT16);
  m.SetInput<uint8_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(GetRegistration(), 0.1f,
                                 {TensorType_UINT8, {4, 2}, -10, 10},
                                 TensorType_INT16);
  m2.SetInput<uint8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  ASSERT_EQ(m2.Invoke(), kTfLiteOk);
  EXPECT_THAT(m2.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.645656, 0.354344,  //
                      0.450166, 0.549834,  //
                      0.622459, 0.377541,  //
                      0.710949, 0.28905,   //
                  },
                  kQuantizedTolerance)));
}

// This contains the same test values as the Softmax test, but reference answer
// generated via the following snippet of python:
//   logits1 = tf.constant([[0, -6, 2, 4],[3, -2, 10, 1]], dtype=tf.float32)
//   logits2 = tf.constant([[0,-6],[2,4],[3,-2],[10,1]], dtype=tf.float32)
//   lsm1 = tf.nn.log_softmax(logits1)
//   lsm2 = tf.nn.log_softmax(logits2)
//   with tf.Session() as sess:
//     print('lsm1', sess.run(lsm1))
//     print('lsm2', sess.run(lsm2))

TEST_P(LogSoftmaxOpTest, LogSoftmax) {
  FloatActivationsOpModel m(GetRegistration(), BuiltinOperator_LOG_SOFTMAX,
                            /*input=*/{TensorType_FLOAT32, {2, 4}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 -4.14297, -10.14297, -2.14297, -.142971,    //
                                 -7.00104, -12.00104, -.00104087, -9.00104,  //
                             })));

  // Same input, but a different shape.
  FloatActivationsOpModel m2(GetRegistration(), BuiltinOperator_LOG_SOFTMAX,
                             /*input=*/{TensorType_FLOAT32, {4, 2}});
  m2.SetInput({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  ASSERT_EQ(m2.Invoke(), kTfLiteOk);
  EXPECT_THAT(m2.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                  -.00247565, -6.00247,   //
                                  -2.12692, -.126928,     //
                                  -.00671534, -5.00671,   //
                                  -.000123374, -9.00012,  //
                              })));
}

TEST_P(LogSoftmaxOpTest, LogSoftmaxUint8) {
  const float kLogSoftmaxQuantizedTolerance = 16 / 256.0;
  // Corresponds to input scale of 20/255.
  QuantizedActivationsOpModel m(
      GetRegistration(), BuiltinOperator_LOG_SOFTMAX,
      /*input=*/{TensorType_UINT8, {2, 4}, -10, 10},
      /*output=*/{TensorType_UINT8, {}, 0, 0, 16. / 256, 255});
  m.SetInput<uint8_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      -4.14297, -10.14297, -2.14297, -.142971,    //
                      -7.00104, -12.00104, -.00104087, -9.00104,  //
                  },
                  kLogSoftmaxQuantizedTolerance)));
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAreArray({189, 93, 221, 253, 142, 63, 255, 111}));
}

TEST_P(LogSoftmaxOpTest, LogSoftmaxInt8) {
  const float kLogSoftmaxQuantizedTolerance = 0.06355;
  QuantizedActivationsOpModel m(
      GetRegistration(), BuiltinOperator_LOG_SOFTMAX,
      /*input=*/{TensorType_INT8, {2, 4}, -10, 10},
      /*output=*/{TensorType_INT8, {}, 0, 0, 16. / 256, 127});
  m.SetInput<int8_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      -4.14297, -10.14297, -2.14297, -.142971,    //
                      -7.00104, -12.00104, -.00104087, -9.00104,  //
                  },
                  kLogSoftmaxQuantizedTolerance)));
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({
                                         61, -36, 93, 125,   //
                                         15, -65, 127, -16,  //
                                     }));
}

TEST(QuantizedActivationsOpTest, LogSoftmaxInt8LargeNegativeNumber) {
  const float kLogSoftmaxQuantizedTolerance = 0.06355;
  QuantizedActivationsOpModel m(
      BuiltinOperator_LOG_SOFTMAX,
      /*input=*/{TensorType_INT8, {2, 4}, -10, 10},
      /*output=*/{TensorType_INT8, {}, 0, 0, 16. / 256, 127});
  m.SetInput<int8_t>({
      -9.9, -9.9, 0, 0,  //
      7.8, -2, 2, 1,     //
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput<int8_t>(),
      ElementsAreArray(ArrayFloatNear(
          {-10.5625, -10.5625, -0.6875, -0.6875, -0.004, -9.8125, -5.75, -6.75},
          kLogSoftmaxQuantizedTolerance)));
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({
                                         -42, -42, 116, 116,  //
                                         127, -30, 35, 19,    //
                                     }));
}

const auto kPReluKernelMap = new std::map<string, TfLiteRegistration*>({
    {"Reference", ops::builtin::Register_PRELU_REF()},
    {"GenericOptimized", ops::builtin::Register_PRELU()},
});

// A base class of PRelu op model. It provides the constructor for
// FloatPReluOpModel and QuantizedPReluOpModel.
class BasePReluOpModel : public SingleOpModel {
 public:
  BasePReluOpModel(const TensorData& input, const TensorData& alpha) {
    input_ = AddInput(input);
    alpha_ = AddInput(alpha);
    output_ = AddOutput({input.type, input.shape, input.min, input.max});
    SetBuiltinOp(BuiltinOperator_PRELU, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(input_), GetShape(alpha_)});
  }

 protected:
  int input_;
  int alpha_;
  int output_;
};

// The FloatPReluOpModel class handles float input and output.
class FloatPReluOpModel : public BasePReluOpModel {
 public:
  using BasePReluOpModel::BasePReluOpModel;

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }
  void SetAlpha(std::initializer_list<float> data) {
    PopulateTensor(alpha_, data);
  }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

// The QuantizedPReluOpModel class handles quantized input and output.
class QuantizedPReluOpModel : public BasePReluOpModel {
 public:
  using BasePReluOpModel::BasePReluOpModel;

  template <typename T>
  void SetInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<T>(input_, data);
  }
  template <typename T>
  void SetAlpha(std::initializer_list<float> data) {
    QuantizeAndPopulate<T>(alpha_, data);
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

class PReluOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kPReluKernelMap;
  }
};

TEST_P(PReluOpTest, PReluFloat32) {
  FloatPReluOpModel m({TensorType_FLOAT32, {1, 2, 2, 3}},
                      {TensorType_FLOAT32, {1, 1, 3}});

  m.SetInput({
      0.0f, 0.0f, 0.0f,     // Row 1, Column 1
      1.0f, 1.0f, 1.0f,     // Row 1, Column 2
      -1.0f, -1.0f, -1.0f,  // Row 2, Column 1
      -2.0f, -2.0f, -2.0f,  // Row 2, Column 2
  });
  m.SetAlpha({0.0f, 1.0f, 2.0f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0.0f, 0.0f, 0.0f,    // Row 1, Column 1
                                 1.0f, 1.0f, 1.0f,    // Row 1, Column 2
                                 0.0f, -1.0f, -2.0f,  // Row 2, Column 1
                                 0.0f, -2.0f, -4.0f,  // Row 2, Column 2
                             }));
}

TEST_P(PReluOpTest, PReluFloat32SameShapes) {
  FloatPReluOpModel m({TensorType_FLOAT32, {1, 2, 2, 3}},
                      {TensorType_FLOAT32, {1, 2, 2, 3}});

  m.SetInput({
      0.0f, 0.0f, 0.0f,     // Row 1, Column 1
      1.0f, 1.0f, 1.0f,     // Row 1, Column 2
      -1.0f, -1.0f, -1.0f,  // Row 2, Column 1
      -2.0f, -2.0f, -2.0f,  // Row 2, Column 2
  });
  m.SetAlpha({
      0.0f, 1.0f, 2.0f,  // Row 1, Column 1
      0.0f, 1.0f, 2.0f,  // Row 1, Column 2
      0.0f, 1.0f, 2.0f,  // Row 2, Column 1
      0.0f, 1.0f, 2.0f,  // Row 2, Column 2
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0.0f, 0.0f, 0.0f,    // Row 1, Column 1
                                 1.0f, 1.0f, 1.0f,    // Row 1, Column 2
                                 0.0f, -1.0f, -2.0f,  // Row 2, Column 1
                                 0.0f, -2.0f, -4.0f,  // Row 2, Column 2
                             }));
}

TEST_P(PReluOpTest, PReluUInt8) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedPReluOpModel m({TensorType_UINT8, {1, 2, 2, 3}, kMin, kMax},
                          {TensorType_UINT8, {1, 1, 3}, kMin, kMax});
  m.SetInput<uint8_t>({
      0.0f, 0.0f, 0.0f,        // Row 1, Column 1
      0.5f, 0.5f, 0.5f,        // Row 1, Column 2
      -1.0f, -1.0f, -1.0f,     // Row 2, Column 1
      -0.25f, -0.25f, -0.25f,  // Row 2, Column 2
  });
  m.SetAlpha<uint8_t>({0.0f, 0.5f, -0.5f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.0f, 0.0f, 0.0f,       // Row 1, Column 1
                      0.5f, 0.5f, 0.5f,       // Row 1, Column 2
                      0.0f, -0.5f, 0.5f,      // Row 2, Column 1
                      0.0f, -0.125f, 0.125f,  // Row 2, Column 2
                  },
                  kQuantizedTolerance)));
  EXPECT_THAT(m.GetOutput<uint8_t>(), ElementsAreArray({
                                          128, 128, 128,  // Row 1, Column 1
                                          192, 192, 192,  // Row 1, Column 2
                                          128, 64, 192,   // Row 2, Column 1
                                          128, 112, 144,  // Row 2, Column 2
                                      }));
}

TEST_P(PReluOpTest, PReluUInt8SameShapes) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedPReluOpModel m({TensorType_UINT8, {1, 2, 2, 3}, kMin, kMax},
                          {TensorType_UINT8, {1, 2, 2, 3}, kMin, kMax});
  m.SetInput<uint8_t>({
      0.0f, 0.0f, 0.0f,        // Row 1, Column 1
      0.5f, 0.5f, 0.5f,        // Row 1, Column 2
      -1.0f, -1.0f, -1.0f,     // Row 2, Column 1
      -0.25f, -0.25f, -0.25f,  // Row 2, Column 2
  });
  m.SetAlpha<uint8_t>({
      0.0f, 0.5f, -0.5f,  // Row 1, Column 1
      0.0f, 0.5f, -0.5f,  // Row 1, Column 2
      0.0f, 0.5f, -0.5f,  // Row 2, Column 1
      0.0f, 0.5f, -0.5f,  // Row 2, Column 2
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.0f, 0.0f, 0.0f,       // Row 1, Column 1
                      0.5f, 0.5f, 0.5f,       // Row 1, Column 2
                      0.0f, -0.5f, 0.5f,      // Row 2, Column 1
                      0.0f, -0.125f, 0.125f,  // Row 2, Column 2
                  },
                  kQuantizedTolerance)));
  EXPECT_THAT(m.GetOutput<uint8_t>(), ElementsAreArray({
                                          128, 128, 128,  // Row 1, Column 1
                                          192, 192, 192,  // Row 1, Column 2
                                          128, 64, 192,   // Row 2, Column 1
                                          128, 112, 144,  // Row 2, Column 2
                                      }));
}

TEST_P(PReluOpTest, PReluInt8) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedPReluOpModel m({TensorType_INT8, {1, 2, 2, 3}, kMin, kMax},
                          {TensorType_INT8, {1, 1, 3}, kMin, kMax});
  m.SetInput<int8_t>({
      0.0f, 0.0f, 0.0f,        // Row 1, Column 1
      0.5f, 0.5f, 0.5f,        // Row 1, Column 2
      -1.0f, -1.0f, -1.0f,     // Row 2, Column 1
      -0.25f, -0.25f, -0.25f,  // Row 2, Column 2
  });
  m.SetAlpha<int8_t>({0.0f, 0.5f, -0.5f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.0f, 0.0f, 0.0f,       // Row 1, Column 1
                      0.5f, 0.5f, 0.5f,       // Row 1, Column 2
                      0.0f, -0.5f, 0.5f,      // Row 2, Column 1
                      0.0f, -0.125f, 0.125f,  // Row 2, Column 2
                  },
                  kQuantizedTolerance)));
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({
                                         0, 0, 0,     // Row 1, Column 1
                                         64, 64, 64,  // Row 1, Column 2
                                         0, -64, 64,  // Row 2, Column 1
                                         0, -16, 16,  // Row 2, Column 2
                                     }));
}

TEST_P(PReluOpTest, PReluInt8SameShapes) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedPReluOpModel m({TensorType_INT8, {1, 2, 2, 3}, kMin, kMax},
                          {TensorType_INT8, {1, 1, 3}, kMin, kMax});
  m.SetInput<int8_t>({
      0.0f, 0.0f, 0.0f,        // Row 1, Column 1
      0.5f, 0.5f, 0.5f,        // Row 1, Column 2
      -1.0f, -1.0f, -1.0f,     // Row 2, Column 1
      -0.25f, -0.25f, -0.25f,  // Row 2, Column 2
  });
  m.SetAlpha<int8_t>({0.0f, 0.5f, -0.5f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.0f, 0.0f, 0.0f,       // Row 1, Column 1
                      0.5f, 0.5f, 0.5f,       // Row 1, Column 2
                      0.0f, -0.5f, 0.5f,      // Row 2, Column 1
                      0.0f, -0.125f, 0.125f,  // Row 2, Column 2
                  },
                  kQuantizedTolerance)));
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({
                                         0, 0, 0,     // Row 1, Column 1
                                         64, 64, 64,  // Row 1, Column 2
                                         0, -64, 64,  // Row 2, Column 1
                                         0, -16, 16,  // Row 2, Column 2
                                     }));
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0.0f, 1.0f, 3.0f,    // Row 1
                                 1.0f, -0.5f, -1.0f,  // Row 2
                             }));
}

class GeluOpModel : public SingleOpModel {
 public:
  GeluOpModel(const TensorData& input, bool approximate) {
    input_ = AddInput(input);
    output_ = AddOutput(input);
    SetBuiltinOp(BuiltinOperator_GELU, BuiltinOptions_GeluOptions,
                 CreateGeluOptions(builder_, approximate).Union());
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

class BaseGeluOpModel : public SingleOpModel {
 public:
  BaseGeluOpModel(const TensorData& input, bool approximate) {
    input_ = AddInput(input);
    approximate_ = approximate;
    output_ = AddOutput({input.type, input.shape, input.min, input.max});
    SetBuiltinOp(BuiltinOperator_GELU, BuiltinOptions_GeluOptions,
                 CreateGeluOptions(builder_, approximate).Union());
    BuildInterpreter({GetShape(input_)});
  }

 protected:
  int input_;
  bool approximate_;
  int output_;
};

// The FloatGeluOpModel class handles float input and output.
class FloatGeluOpModel : public BaseGeluOpModel {
 public:
  using BaseGeluOpModel::BaseGeluOpModel;

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

// The QuantizedGeluOpModel class handles quantized input and output.
class QuantizedGeluOpModel : public BaseGeluOpModel {
 public:
  using BaseGeluOpModel::BaseGeluOpModel;

  template <typename T>
  void SetInput(std::initializer_list<float> data) {
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

TEST(FloatActivationsOpTest, Gelu) {
  FloatGeluOpModel m({TensorType_FLOAT32, {2, 3}}, /*approximate=*/false);

  m.SetInput({
      0.0f, 1.0f, 3.0f,    // Row 1
      1.0f, -1.0f, -2.0f,  // Row 2
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 0.0f, 0.841345f, 2.99595f,           // Row 1
                                 0.841345f, -0.158655f, -0.0455003f,  // Row 2
                             })));
}

TEST(FloatActivationsOpTest, GeluApproximate) {
  FloatGeluOpModel m({TensorType_FLOAT32, {2, 3}}, /*approximate=*/true);
  // The OpenCL delegate always uses the accurate version so use a higher
  // tolerance for validation.
  constexpr float kEpsilon = 1e-3;

  m.SetInput({
      0.0f, 1.0f, 3.0f,    // Row 1
      1.0f, -1.0f, -2.0f,  // Row 2
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.0f, 0.841192f, 2.99636f,           // Row 1
                      0.841192f, -0.158808f, -0.0454023f,  // Row 2
                  },
                  kEpsilon)));
}

TEST(QuantizedGeluOpTest, GeluInt8) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedGeluOpModel m({TensorType_INT8, {2, 3}, 3 * kMin, 3 * kMax},
                         /*approximate=*/false);
  m.SetInput<int8_t>({
      0.0f, 1.0f, 3.0f,    // Row 1
      1.0f, -1.0f, -2.0f,  // Row 2
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({
                  0.f, 0.84375f, 2.97656f,          // Row 1
                  0.84375f, -0.164062f, -0.046875f  // Row 2
              })));
}

TEST(QuantizedGeluOpTest, GeluInt8Approximate) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedGeluOpModel m({TensorType_INT8, {2, 3}, 3 * kMin, 3 * kMax},
                         /*approximate=*/true);
  m.SetInput<int8_t>({
      0.0f, 1.0f, 3.0f,    // Row 1
      1.0f, -1.0f, -2.0f,  // Row 2
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({
                  0.f, 0.84375f, 2.97656f,          // Row 1
                  0.84375f, -0.164062f, -0.046875f  // Row 2
              })));
}
TEST(QuantizedGeluOpTest, GeluUInt8) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedGeluOpModel m({TensorType_UINT8, {2, 3}, 3 * kMin, 3 * kMax},
                         /*approximate=*/false);
  m.SetInput<uint8_t>({
      0.0f, 1.0f, 3.0f,    // Row 1
      1.0f, -1.0f, -2.0f,  // Row 2
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({
                  0.f, 0.84375f, 2.97656f,          // Row 1
                  0.84375f, -0.164062f, -0.046875f  // Row 2
              })));
}

TEST(QuantizedGeluOpTest, GeluUInt8Approximate) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedGeluOpModel m({TensorType_UINT8, {2, 3}, 3 * kMin, 3 * kMax},
                         /*approximate=*/true);
  m.SetInput<uint8_t>({
      0.0f, 1.0f, 3.0f,    // Row 1
      1.0f, -1.0f, -2.0f,  // Row 2
  });
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({
                  0.f, 0.84375f, 2.97656f,          // Row 1
                  0.84375f, -0.164062f, -0.046875f  // Row 2
              })));
}

INSTANTIATE_TEST_SUITE_P(
    TanhOpTest, TanhOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kTanhKernelMap)));

INSTANTIATE_TEST_SUITE_P(
    LogisticOpTest, LogisticOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kLogisticKernelMap)));

INSTANTIATE_TEST_SUITE_P(
    LogSoftmaxOpTest, LogSoftmaxOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kLogSoftmaxKernelMap)));

INSTANTIATE_TEST_SUITE_P(
    SoftmaxOpTest, SoftmaxOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kSoftmaxKernelMap)));

INSTANTIATE_TEST_SUITE_P(
    PReluOpTest, PReluOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kPReluKernelMap)));

INSTANTIATE_TEST_SUITE_P(
    LeakyReluOpTest, LeakyReluOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kLeakyReluKernelMap)));

}  // namespace
}  // namespace tflite
