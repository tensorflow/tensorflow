/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
namespace {}

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloorModSimple) {
#ifdef notdef
  FloorMod<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                          {TensorType_INT32, {1, 2, 2, 1}},
                          {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {10, 9, 11, 3});
  model.PopulateTensor<int32_t>(model.input2(), {2, 2, 3, 4});
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
#endif  // notdef
}

TF_LITE_MICRO_TEST(FloorModNegativeValue) {
#ifdef notdef
  FloorMod<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                          {TensorType_INT32, {1, 2, 2, 1}},
                          {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {10, -9, -11, 7});
  model.PopulateTensor<int32_t>(model.input2(), {2, 2, -3, -4});
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, -2, -1));
#endif  // notdef
}

TF_LITE_MICRO_TEST(FloorModBroadcast) {
#ifdef notdef
  FloorMod<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                          {TensorType_INT32, {1}}, {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {10, -9, -11, 7});
  model.PopulateTensor<int32_t>(model.input2(), {-3});
  EXPECT_THAT(model.GetOutput(), ElementsAre(-2, 0, -2, -2));
#endif  // notdef
}

TF_LITE_MICRO_TEST(FloorModInt64WithBroadcast) {
#ifdef notdef
  FloorMod<int64_t> model({TensorType_INT64, {1, 2, 2, 1}},
                          {TensorType_INT64, {1}}, {TensorType_INT64, {}});
  model.PopulateTensor<int64_t>(model.input1(), {10, -9, -11, (1LL << 34) + 9});
  model.PopulateTensor<int64_t>(model.input2(), {-(1LL << 33)});
  EXPECT_THAT(model.GetOutput(),
              ElementsAre(-8589934582, -9, -11, -8589934583));
#endif  // notdef
}

TF_LITE_MICRO_TEST(FloorModFloatSimple) {
#ifdef notdef
  FloorMod<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                        {TensorType_FLOAT32, {1, 2, 2, 1}},
                        {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {10, 9, 11, 3});
  model.PopulateTensor<float>(model.input2(), {2, 2, 3, 4});
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
#endif  // notdef
}

TF_LITE_MICRO_TEST(FloorModFloatNegativeValue) {
#ifdef notdef
  FloorMod<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                        {TensorType_FLOAT32, {1, 2, 2, 1}},
                        {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {10, -9, -11, 7});
  model.PopulateTensor<float>(model.input2(), {2, 2, -3, -4});
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, -2, -1));
#endif  // notdef
}

TF_LITE_MICRO_TEST(FloorModFloatBroadcast) {
#ifdef notdef
  FloorMod<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                        {TensorType_FLOAT32, {1}}, {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {10, -9, -11, 7});
  model.PopulateTensor<float>(model.input2(), {-3});
  EXPECT_THAT(model.GetOutput(), ElementsAre(-2, 0, -2, -2));
#endif  // notdef
}

TF_LITE_MICRO_TESTS_END

}  // namespace testing
}  // namespace tflite
