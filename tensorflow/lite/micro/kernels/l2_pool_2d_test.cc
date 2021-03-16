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
class BasePoolingOpModel : public SingleOpModel {
 public:
  BasePoolingOpModel(
      BuiltinOperator type, const TensorData& input, int filter_width,
      int filter_height, const TensorData& output,
      Padding padding = Padding_VALID, int stride_w = 2, int stride_h = 2,
      ActivationFunctionType activation = ActivationFunctionType_NONE) {
    SetBuiltinOp(type, BuiltinOptions_Pool2DOptions,
                 CreatePool2DOptions(builder_, padding, stride_w, stride_h,
                                     filter_width, filter_height, activation)
                     .Union());
  }
};
#endif  // notdef

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloatPoolingOpTestL2Pool) {
#ifdef notdef
  FloatPoolingOpModel m(BuiltinOperator_L2_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}});
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3.5, 6.5}));
#endif  // notdef
}

TF_LITE_MICRO_TEST(FloatPoolingOpTestL2PoolActivationRelu) {
#ifdef notdef
  FloatPoolingOpModel m(BuiltinOperator_L2_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}}, Padding_VALID, 2,
                        2, ActivationFunctionType_RELU);
  m.SetInput({
      -1, -6, 2, 4,   //
      -3, -2, 10, 7,  //
  });
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({3.53553, 6.5})));
#endif  // notdef
}

TF_LITE_MICRO_TEST(FloatPoolingOpTestL2PoolActivationRelu1) {
#ifdef notdef
  FloatPoolingOpModel m(BuiltinOperator_L2_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}}, Padding_VALID, 2,
                        2, ActivationFunctionType_RELU_N1_TO_1);
  m.SetInput({
      -0.1, -0.6, 2, 4,   //
      -0.3, -0.2, 10, 7,  //
  });
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({0.353553, 1.0})));
#endif  // notdef
}

TF_LITE_MICRO_TEST(FloatPoolingOpTestL2PoolActivationRelu6) {
#ifdef notdef
  FloatPoolingOpModel m(BuiltinOperator_L2_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}}, Padding_VALID, 2,
                        2, ActivationFunctionType_RELU6);
  m.SetInput({
      -0.1, -0.6, 2, 4,   //
      -0.3, -0.2, 10, 7,  //
  });
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({0.353553, 6.0})));
#endif  // notdef
}

TF_LITE_MICRO_TEST(FloatPoolingOpTestL2PoolPaddingSame) {
#ifdef notdef
  FloatPoolingOpModel m(BuiltinOperator_L2_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}}, Padding_SAME);
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3.5, 6.5}));
#endif  // notdef
}

TF_LITE_MICRO_TEST(FloatPoolingOpTestL2PoolPaddingSameStride1) {
#ifdef notdef
  FloatPoolingOpModel m(BuiltinOperator_L2_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}}, Padding_SAME, 1,
                        1);
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {3.5, 6.0, 6.5, 5.70088, 2.54951, 7.2111, 8.63134, 7.0},
                  /*max_abs_error=*/1e-4)));
#endif  // notdef
}

TF_LITE_MICRO_TEST(FloatPoolingOpTestL2PoolPaddingValidStride1) {
#ifdef notdef
  FloatPoolingOpModel m(BuiltinOperator_L2_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}}, Padding_VALID, 1,
                        1);
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3.5, 6.0, 6.5}));
#endif  // notdef
}

TF_LITE_MICRO_TESTS_END
