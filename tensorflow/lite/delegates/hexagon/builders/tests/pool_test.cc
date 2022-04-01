/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <random>

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
using testing::ElementsAreArray;

class PoolingOpModel : public SingleOpModelWithHexagon {
 public:
  explicit PoolingOpModel(BuiltinOperator type, const TensorData& input,
                          int filter_width, int filter_height,
                          const TensorData& output,
                          tflite::Padding padding = Padding_VALID) {
    input_ = AddInput(input);
    output_ = AddOutput(output);

    SetBuiltinOp(type, BuiltinOptions_Pool2DOptions,
                 CreatePool2DOptions(builder_, padding, /*stride_w=*/2,
                                     /*stride_h=*/2, filter_width,
                                     filter_height, ActivationFunctionType_NONE)
                     .Union());

    BuildInterpreter({GetShape(input_)});
  }

  template <typename T>
  void SetInput(const std::vector<float>& data) {
    QuantizeAndPopulate<T>(input_, data);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

 private:
  int input_;
  int output_;
};

TEST(QuantizedPoolingOpTest, AveragePool) {
  PoolingOpModel m(BuiltinOperator_AVERAGE_POOL_2D,
                   /*input=*/{TensorType_UINT8, {1, 16, 8, 1}, 0, 10},
                   /*filter_width=*/8, /*filter_height=*/8,
                   /*output=*/{TensorType_UINT8, {}, 0, 10});
  m.SetInput<uint8_t>({
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
  });
  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {4.58824, 4.58824, 4.90196, 4.58824, 4.27451})));
}

TEST(QuantizedPoolingOpTest, AveragePool_Int8) {
  PoolingOpModel m(BuiltinOperator_AVERAGE_POOL_2D,
                   /*input=*/{TensorType_INT8, {1, 16, 8, 1}, 0, 10},
                   /*filter_width=*/8, /*filter_height=*/8,
                   /*output=*/{TensorType_INT8, {}, 0, 10});
  m.SetInput<int8_t>({
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
  });

  // Reference data.
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<int8_t>();

  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output)));
}

TEST(QuantizedUInt8PoolingOpTest, MaxPool) {
  // Choose the input ranges carefully so that the dequantized output matches
  // the results of the float model above.
  // Input Range[0, 15.9375] --> [Scale{0.0625}, zero_point{0}]
  PoolingOpModel m(BuiltinOperator_MAX_POOL_2D,
                   /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, 0, 15.9375},
                   /*filter_width=*/2, /*filter_height=*/2,
                   /*output=*/{TensorType_UINT8, {}, 0, 15.9375}, Padding_SAME);
  m.SetInput<uint8_t>({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  // Reference data.
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();

  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output)));
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

TEST(QuantizedUInt8PoolingOpTest, MaxPool_Valid_Large_Filter) {
  const int ksize = 15;
  PoolingOpModel m(BuiltinOperator_MAX_POOL_2D,
                   /*input=*/{TensorType_UINT8, {1, ksize, ksize, 512}, 0, 30},
                   /*filter_width=*/ksize, /*filter_height=*/ksize,
                   /*output=*/{TensorType_UINT8, {}, 0, 30}, Padding_VALID);

  std::minstd_rand random_engine;
  std::vector<float> input;
  GenerateUniformRandomVector(ksize * ksize * 512, 0, 30, &random_engine,
                              &input);

  m.SetInput<uint8_t>(input);

  // Reference data.
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();

  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output)));
}

}  // namespace tflite
