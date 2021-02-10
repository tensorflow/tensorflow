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

#include <limits>
#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

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

template <TensorType tensor_type, typename integer_dtype>
void QuantizedActivationsOpTestLeakyRelu() {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<integer_dtype>::max() /
      static_cast<float>(std::numeric_limits<integer_dtype>::max() + 1);
#ifdef notdef
  QuantizedActivationsOpModel m(
      /*input=*/{tensor_type, {5, 5}, 5 * kMin, 5 * kMax}, 0.1);

  m.SetInput<integer_dtype>({
      -5.0f, -4.6f, -4.2f, -3.8f, -3.4f,  // Row 1
      -3.0f, -2.6f, -2.2f, -1.8f, -1.4f,  // Row 2
      -1.0f, -0.6f, -0.2f, 0.2f,  0.6f,   // Row 3
      1.0f,  1.4f,  1.8f,  2.2f,  2.6f,   // Row 4
      3.0f,  3.4f,  3.8f,  4.2f,  4.6f,   // Row 5
  });

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
#endif  // notdef
}

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(QuantizedActivationsOpTestLeakyReluUint8) {
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
#ifdef notdef
  QuantizedActivationsOpModel m(
      /*input=*/{TensorType_UINT8, {2, 3}, 8 * kMin, 8 * kMax}, 0.5);

  m.SetInput<uint8_t>({
      0.0f, 1.0f, 3.0f,    // Row 1
      1.0f, -1.0f, -2.0f,  // Row 2
  });
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.0f, 1.0f, 3.0f,    // Row 1
                      1.0f, -0.5f, -1.0f,  // Row 2
                  },
                  kQuantizedTolerance * 8)));
#endif  // notdef
}

TF_LITE_MICRO_TEST(QuantizedActivationsOpTestLeakyReluInt8) {
  QuantizedActivationsOpTestLeakyRelu<TensorType_INT8, int8_t>();
}

TF_LITE_MICRO_TEST(QuantizedActivationsOpTestLeakyReluInt16) {
  QuantizedActivationsOpTestLeakyRelu<TensorType_INT16, int16_t>();
}

TF_LITE_MICRO_TEST(FloatActivationsOpTestLeakyRelu) {
#ifdef notdef
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
#endif  // notdef
}

TF_LITE_MICRO_TESTS_END

}  // namespace
}  // namespace testing
}  // namespace tflite
