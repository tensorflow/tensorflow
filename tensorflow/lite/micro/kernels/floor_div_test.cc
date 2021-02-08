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

#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloorDivModelSimple) {
#ifdef notdef
  FloorDivModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                               {TensorType_INT32, {1, 2, 2, 1}},
                               {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {10, 9, 11, 3});
  model.PopulateTensor<int32_t>(model.input2(), {2, 2, 3, 4});
  EXPECT_THAT(model.GetOutput(), ElementsAre(5, 4, 3, 0));
#endif
}

TF_LITE_MICRO_TEST(FloorDivModelNegativeValue) {
#ifdef notdef
  FloorDivModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                               {TensorType_INT32, {1, 2, 2, 1}},
                               {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {10, -9, -11, 7});
  model.PopulateTensor<int32_t>(model.input2(), {2, 2, -3, -4});
  EXPECT_THAT(model.GetOutput(), ElementsAre(5, -5, 3, -2));
#endif
}

TF_LITE_MICRO_TEST(FloorDivModelBroadcastFloorDiv) {
#ifdef notdef
  FloorDivModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                               {TensorType_INT32, {1}}, {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {10, -9, -11, 7});
  model.PopulateTensor<int32_t>(model.input2(), {-3});
  EXPECT_THAT(model.GetOutput(), ElementsAre(-4, 3, 3, -3));
#endif
}

TF_LITE_MICRO_TEST(FloorDivModelSimpleFloat) {
#ifdef notdef
  FloorDivModel<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                             {TensorType_FLOAT32, {1, 2, 2, 1}},
                             {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {10.05, 9.09, 11.9, 3.01});
  model.PopulateTensor<float>(model.input2(), {2.05, 2.03, 3.03, 4.03});
  EXPECT_THAT(model.GetOutput(), ElementsAre(4.0, 4.0, 3.0, 0.0));
#endif
}

TF_LITE_MICRO_TEST(FloorDivModelNegativeValueFloat) {
#ifdef notdef
  FloorDivModel<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                             {TensorType_FLOAT32, {1, 2, 2, 1}},
                             {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {10.03, -9.9, -11.0, 7.0});
  model.PopulateTensor<float>(model.input2(), {2.0, 2.3, -3.0, -4.1});
  EXPECT_THAT(model.GetOutput(), ElementsAre(5.0, -5.0, 3.0, -2.0));
#endif
}

TF_LITE_MICRO_TEST(FloorDivModelBroadcastFloorDivFloat) {
#ifdef notdef
  FloorDivModel<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                             {TensorType_FLOAT32, {1}},
                             {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {10.03, -9.9, -11.0, 7.0});
  model.PopulateTensor<float>(model.input2(), {-3.3});
  EXPECT_THAT(model.GetOutput(), ElementsAre(-4.0, 2.0, 3.0, -3.0));
#endif
}

TF_LITE_MICRO_TESTS_END

}  // namespace
}  // namespace testing
}  // namespace tflite
