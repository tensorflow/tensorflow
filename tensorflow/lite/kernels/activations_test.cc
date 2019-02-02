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

class BaseActivationsOpModel : public SingleOpModel {
 public:
  // Most activations don't take any options, so this constructor works for
  // them.
  BaseActivationsOpModel(BuiltinOperator type, TensorData input) {
    input_ = AddInput(input);
    if (input.type == TensorType_UINT8) {
      output_ = AddOutput({input.type, {}, 0, 0, 1. / 256});
    } else {
      output_ = AddOutput({input.type, {}});
    }
    SetBuiltinOp(type, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(input_)});
  }

  // A dedicated constructor for SOFTMAX, which does some options.
  BaseActivationsOpModel(float softmax_beta, TensorData input) {
    input_ = AddInput(input);
    if (input.type == TensorType_UINT8) {
      output_ = AddOutput({input.type, {}, 0, 0, 1. / 256});
    } else if (input.type == TensorType_INT8) {
      output_ = AddOutput({TensorType_INT8, {}, 0, 0, 1. / 256, -128});
    } else {
      output_ = AddOutput({input.type, {}});
    }
    SetBuiltinOp(BuiltinOperator_SOFTMAX, BuiltinOptions_SoftmaxOptions,
                 CreateSoftmaxOptions(builder_, softmax_beta).Union());
    BuildInterpreter({GetShape(input_)});
  }

  BaseActivationsOpModel(BuiltinOperator type, const TensorData& input,
                         const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(type, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(input_)});
  }

 protected:
  int input_;
  int output_;
};

class FloatActivationsOpModel : public BaseActivationsOpModel {
 public:
  using BaseActivationsOpModel::BaseActivationsOpModel;

  void SetInput(std::initializer_list<float> data) {
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

TEST(FloatActivationsOpTest, Relu) {
  FloatActivationsOpModel m(BuiltinOperator_RELU,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0, 0, 2, 4,   //
                                 3, 0, 10, 1,  //
                             }));
}

TEST(FloatActivationsOpTest, Relu1) {
  FloatActivationsOpModel m(BuiltinOperator_RELU_N1_TO_1,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0.0, -0.6, 0.2, -0.4,  //
      0.3, -2.0, 1.1, -0.1,  //
  });
  m.Invoke();
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
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0, 0, 2, 4,  //
                                 3, 0, 6, 1,  //
                             }));
}

TEST(FloatActivationsOpTest, Tanh) {
  FloatActivationsOpModel m(BuiltinOperator_TANH,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 0, -0.9999877, 0.9640275, 0.999329,    //
                                 0.99505475, -0.9640275, 1, 0.7615941,  //
                             })));
}

TEST(QuantizedActivationsOpTest, Relu6) {
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
  m.Invoke();
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

TEST(QuantizedActivationsOpTest, Tanh) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedActivationsOpModel m(
      BuiltinOperator_TANH,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, 8 * kMin, 8 * kMax},
      /*output=*/{TensorType_UINT8, {1, 2, 4, 1}, kMin, kMax});
  m.SetInput<uint8_t>({
      0, -6, 2, 4,   //
      -4, -2, 8, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.0, -0.999987, 0.964027, 0.999329,     //
                      -0.999329, -0.96402, 0.99999, 0.76159,  //
                  },
                  kQuantizedTolerance)));
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAreArray({128, 0, 251, 255, 0, 5, 255, 225}));
}

TEST(QuantizedActivationsOpTest, TanhInt16) {
  const float kMin = -1;
  const float kMax = 32767.f / 32768.f;
  QuantizedActivationsOpModel m(
      BuiltinOperator_TANH,
      /*input=*/{TensorType_INT16, {1, 2, 4, 1}, 8 * kMin, 8 * kMax},
      /*output=*/{TensorType_INT16, {1, 2, 4, 1}, kMin, kMax});
  m.SetInput<int16_t>({
      0, -6, 2, 4,   //
      -4, -2, 8, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.0, -0.999987, 0.964027, 0.999329,     //
                      -0.999329, -0.96402, 0.99999, 0.76159,  //
                  },
                  kQuantizedToleranceInt16)));
}

TEST(FloatActivationsOpTest, Sigmoid) {
  FloatActivationsOpModel m(BuiltinOperator_LOGISTIC,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 0.5, 0.002473, 0.880797, 0.982014,       //
                                 0.952574, 0.119203, 0.999955, 0.731059,  //
                             })));
}

TEST(QuantizedActivationsOpTest, Sigmoid) {
  QuantizedActivationsOpModel m(
      BuiltinOperator_LOGISTIC,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, -10, 10});
  m.SetInput<uint8_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.5, 0.002473, 0.880797, 0.982014,       //
                      0.952574, 0.119203, 0.999955, 0.731059,  //
                  },
                  kQuantizedTolerance)));
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAreArray({128, 1, 227, 251, 244, 32, 255, 188}));
}

TEST(QuantizedActivationsOpTest, SigmoidInt16) {
  const float kMin = -1;
  const float kMax = 32767.f / 32768.f;
  QuantizedActivationsOpModel m(
      BuiltinOperator_LOGISTIC,
      /*input=*/{TensorType_INT16, {1, 2, 4, 1}, 8 * kMin, 8 * kMax},
      /*output=*/{TensorType_INT16, {1, 2, 4, 1}, kMin, kMax});
  m.SetInput<int16_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.5, 0.002473, 0.880797, 0.982014,       //
                      0.952574, 0.119203, 0.999955, 0.731059,  //
                  },
                  kQuantizedToleranceInt16)));
}

TEST(FloatActivationsOpTest, Softmax4D) {
  FloatActivationsOpModel m(0.1,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 1, 4}});
  m.SetInput({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 .23463, .12877, .28658, .35003,  //
                                 .22528, .13664, .45365, .18443,  //
                             })));

  // Same input, but a different shape.
  FloatActivationsOpModel m2(0.1,
                             /*input=*/{TensorType_FLOAT32, {4, 1, 1, 2}});
  m2.SetInput({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
  EXPECT_THAT(m2.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                  0.645656, 0.354344,  //
                                  0.450166, 0.549834,  //
                                  0.622459, 0.377541,  //
                                  0.710949, 0.28905,   //
                              })));
}

TEST(QuantizedActivationsOpTest, Softmax4DUint8) {
  QuantizedActivationsOpModel m(
      0.1,
      /*input=*/{TensorType_UINT8, {1, 2, 1, 4}, -10, 10});
  m.SetInput<uint8_t>({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(
      0.1,
      /*input=*/{TensorType_UINT8, {4, 1, 1, 2}, -10, 10});
  m2.SetInput<uint8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
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

// Test quantized softmax with int8 input and output. With the same input as in
// QuantizedActivationsOpTest.Softmax1D, the dequantized output is identical.
TEST(QuantizedActivationsOpTest, Softmax1DInt8) {
  QuantizedActivationsOpModel m(0.1,
                                /*input=*/{TensorType_INT8, {8}, -10, 10});
  m.SetInput<int8_t>({0, -6, 2, 4, 3, -2, 10, 1});
  m.Invoke();
  EXPECT_THAT(
      m.GetDequantizedOutput<int8_t>(),
      ElementsAreArray(ArrayFloatNear({0.09766, 0.05469, 0.12109, 0.14453,
                                       0.13281, 0.07813, 0.26563, 0.10938},
                                      kQuantizedTolerance)));
}

// Test quantized softmax with int8 input and output. With the same input as in
// QuantizedActivationsOpTest.Softmax2D, the dequantized output is identical.
TEST(QuantizedActivationsOpTest, Softmax2DInt8) {
  QuantizedActivationsOpModel m(0.1,
                                /*input=*/{TensorType_INT8, {2, 4}, -10, 10});
  m.SetInput<int8_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(0.1,
                                 /*input=*/{TensorType_INT8, {4, 2}, -10, 10});
  m2.SetInput<int8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
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
TEST(QuantizedActivationsOpTest, Softmax3DInt8) {
  QuantizedActivationsOpModel m(
      0.1,
      /*input=*/{TensorType_INT8, {1, 2, 4}, -10, 10});
  m.SetInput<int8_t>({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(
      0.1,
      /*input=*/{TensorType_INT8, {4, 1, 2}, -10, 10});
  m2.SetInput<int8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
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
TEST(QuantizedActivationsOpTest, Softmax4DInt8) {
  QuantizedActivationsOpModel m(
      0.1,
      /*input=*/{TensorType_INT8, {1, 2, 1, 4}, -10, 10});
  m.SetInput<int8_t>({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  m.Invoke();
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
  QuantizedActivationsOpModel m2(
      0.1,
      /*input=*/{TensorType_INT8, {4, 1, 1, 2}, -10, 10});
  m2.SetInput<int8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
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

TEST(FloatActivationsOpTest, Softmax3D) {
  FloatActivationsOpModel m(0.1,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4}});
  m.SetInput({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 .23463, .12877, .28658, .35003,  //
                                 .22528, .13664, .45365, .18443,  //
                             })));

  // Same input, but a different shape.
  FloatActivationsOpModel m2(0.1,
                             /*input=*/{TensorType_FLOAT32, {4, 1, 2}});
  m2.SetInput({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
  EXPECT_THAT(m2.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                  0.645656, 0.354344,  //
                                  0.450166, 0.549834,  //
                                  0.622459, 0.377541,  //
                                  0.710949, 0.28905,   //
                              })));
}

TEST(QuantizedActivationsOpTest, Softmax3DUint8) {
  QuantizedActivationsOpModel m(
      0.1,
      /*input=*/{TensorType_UINT8, {1, 2, 4}, -10, 10});
  m.SetInput<uint8_t>({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(
      0.1,
      /*input=*/{TensorType_UINT8, {4, 1, 2}, -10, 10});
  m2.SetInput<uint8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
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

TEST(FloatActivationsOpTest, Softmax1D) {
  FloatActivationsOpModel m(0.1,
                            /*input=*/{TensorType_FLOAT32, {8}});
  m.SetInput({0, -6, 2, 4, 3, -2, 10, 1});
  m.Invoke();
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          {.09752, .05352, .11911, .14548, .13164, .07984, .26509, .10778})));
}

TEST(QuantizedActivationsOpTest, Softmax1DUint8) {
  QuantizedActivationsOpModel m(0.1,
                                /*input=*/{TensorType_UINT8, {8}, -10, 10});
  m.SetInput<uint8_t>({0, -6, 2, 4, 3, -2, 10, 1});
  m.Invoke();
  EXPECT_THAT(
      m.GetDequantizedOutput<uint8_t>(),
      ElementsAreArray(ArrayFloatNear({0.09766, 0.05469, 0.12109, 0.14453,
                                       0.13281, 0.07813, 0.26563, 0.10938},
                                      kQuantizedTolerance)));
}

TEST(FloatActivationsOpTest, Softmax2D) {
  FloatActivationsOpModel m(0.1,
                            /*input=*/{TensorType_FLOAT32, {2, 4}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 .23463, .12877, .28658, .35003,  //
                                 .22528, .13664, .45365, .18443,  //
                             })));

  // Same input, but a different shape.
  FloatActivationsOpModel m2(0.1,
                             /*input=*/{TensorType_FLOAT32, {4, 2}});
  m2.SetInput({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
  EXPECT_THAT(m2.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                  0.645656, 0.354344,  //
                                  0.450166, 0.549834,  //
                                  0.622459, 0.377541,  //
                                  0.710949, 0.28905,   //
                              })));
}

TEST(QuantizedActivationsOpTest, Softmax2DUint8) {
  QuantizedActivationsOpModel m(0.1,
                                /*input=*/{TensorType_UINT8, {2, 4}, -10, 10});
  m.SetInput<uint8_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kQuantizedTolerance)));

  // Same input, but a different shape.
  QuantizedActivationsOpModel m2(0.1,
                                 /*input=*/{TensorType_UINT8, {4, 2}, -10, 10});
  m2.SetInput<uint8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
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

// This contains the same test values as the Softmax test, but reference answer
// generated via the following snippet of python:
//   logits1 = tf.constant([[0, -6, 2, 4],[3, -2, 10, 1]], dtype=tf.float32)
//   logits2 = tf.constant([[0,-6],[2,4],[3,-2],[10,1]], dtype=tf.float32)
//   lsm1 = tf.nn.log_softmax(logits1)
//   lsm2 = tf.nn.log_softmax(logits2)
//   with tf.Session() as sess:
//     print('lsm1', sess.run(lsm1))
//     print('lsm2', sess.run(lsm2))

TEST(FloatActivationsOpTest, LogSoftmax) {
  FloatActivationsOpModel m(BuiltinOperator_LOG_SOFTMAX,
                            /*input=*/{TensorType_FLOAT32, {2, 4}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 -4.14297, -10.14297, -2.14297, -.142971,    //
                                 -7.00104, -12.00104, -.00104087, -9.00104,  //
                             })));

  // Same input, but a different shape.
  FloatActivationsOpModel m2(BuiltinOperator_LOG_SOFTMAX,
                             /*input=*/{TensorType_FLOAT32, {4, 2}});
  m2.SetInput({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m2.Invoke();
  EXPECT_THAT(m2.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                  -.00247565, -6.00247,   //
                                  -2.12692, -.126928,     //
                                  -.00671534, -5.00671,   //
                                  -.000123374, -9.00012,  //
                              })));
}

TEST(QuantizedActivationsOpTest, LogSoftmax) {
  const float kLogSoftmaxQuantizedTolerance = 16 / 256.0;
  QuantizedActivationsOpModel m(
      BuiltinOperator_LOG_SOFTMAX,
      /*input=*/{TensorType_UINT8, {2, 4}, -10, 10},
      /*output=*/{TensorType_UINT8, {}, 0, 0, 16. / 256, 255});
  m.SetInput<uint8_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
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

TEST(FloatActivationsOpTest, PRelu) {
  FloatPReluOpModel m({TensorType_FLOAT32, {1, 2, 2, 3}},
                      {TensorType_FLOAT32, {1, 1, 3}});

  m.SetInput({
      0.0f, 0.0f, 0.0f,     // Row 1, Column 1
      1.0f, 1.0f, 1.0f,     // Row 1, Column 2
      -1.0f, -1.0f, -1.0f,  // Row 2, Column 1
      -2.0f, -2.0f, -2.0f,  // Row 1, Column 2
  });
  m.SetAlpha({0.0f, 1.0f, 2.0f});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0.0f, 0.0f, 0.0f,    // Row 1, Column 1
                                 1.0f, 1.0f, 1.0f,    // Row 1, Column 2
                                 0.0f, -1.0f, -2.0f,  // Row 2, Column 1
                                 0.0f, -2.0f, -4.0f,  // Row 1, Column 2
                             }));
}

TEST(QuantizedActivationsOpTest, PRelu) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  QuantizedPReluOpModel m({TensorType_UINT8, {1, 2, 2, 3}, kMin, kMax},
                          {TensorType_UINT8, {1, 1, 3}, kMin, kMax});
  m.SetInput<uint8_t>({
      0.0f, 0.0f, 0.0f,        // Row 1, Column 1
      0.5f, 0.5f, 0.5f,        // Row 1, Column 2
      -1.0f, -1.0f, -1.0f,     // Row 2, Column 1
      -0.25f, -0.25f, -0.25f,  // Row 1, Column 2
  });
  m.SetAlpha<uint8_t>({0.0f, 0.5f, -0.5f});
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.0f, 0.0f, 0.0f,       // Row 1, Column 1
                      0.5f, 0.5f, 0.5f,       // Row 1, Column 2
                      0.0f, -0.5f, 0.5f,      // Row 2, Column 1
                      0.0f, -0.125f, 0.125f,  // Row 1, Column 2
                  },
                  kQuantizedTolerance)));
  EXPECT_THAT(m.GetOutput<uint8_t>(), ElementsAreArray({
                                          128, 128, 128,  // Row 1, Column 1
                                          192, 192, 192,  // Row 1, Column 2
                                          128, 64, 192,   // Row 2, Column 1
                                          128, 112, 144,  // Row 1, Column 2
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
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0.0f, 1.0f, 3.0f,    // Row 1
                                 1.0f, -0.5f, -1.0f,  // Row 2
                             }));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
