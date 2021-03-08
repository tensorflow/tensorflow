/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
namespace {}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloatAddNOpAddMultipleTensors) {
#ifdef notdef
  FloatAddNOpModel m({{TensorType_FLOAT32, {1, 2, 2, 1}},
                      {TensorType_FLOAT32, {1, 2, 2, 1}},
                      {TensorType_FLOAT32, {1, 2, 2, 1}}},
                     {TensorType_FLOAT32, {}});
  m.PopulateTensor<float>(m.input(0), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input(1), {0.1, 0.2, 0.3, 0.5});
  m.PopulateTensor<float>(m.input(2), {0.5, 0.1, 0.1, 0.2});
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1.4, 0.5, 1.1, 1.5}));
#endif  // notdef
}

TF_LITE_MICRO_TEST(IntegerAddNOpAddMultipleTensors) {
#ifdef notdef
  IntegerAddNOpModel m({{TensorType_INT32, {1, 2, 2, 1}},
                        {TensorType_INT32, {1, 2, 2, 1}},
                        {TensorType_INT32, {1, 2, 2, 1}}},
                       {TensorType_INT32, {}});
  m.PopulateTensor<int32_t>(m.input(0), {-20, 2, 7, 8});
  m.PopulateTensor<int32_t>(m.input(1), {1, 2, 3, 5});
  m.PopulateTensor<int32_t>(m.input(2), {10, -5, 1, -2});
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-9, -1, 11, 11}));
#endif  // notdef
}

TF_LITE_MICRO_TESTS_END
