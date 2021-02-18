/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifdef notdef
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
#endif  // notdef

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

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloatActivationsOpTestElu) {
#ifdef notdef
  FloatActivationsOpModel m(BuiltinOperator_ELU,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, -4,     //
      3, -2, 10, -0.1,  //
  });
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 0.0, -0.997521, 2.0, -0.981684,    //
                                 3.0, -0.864665, 10.0, -0.0951626,  //
                             })));
#endif  // notdef
}

TF_LITE_MICRO_TEST(QuantizedActivationsOpTestEluInt8) {
#ifdef notdef
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
#endif  // notdef
}

TF_LITE_MICRO_TESTS_END

}  // namespace
}  // namespace testing
}  // namespace tflite
