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

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(CumsumOpTestSimpleIntTest) {
#ifdef notdef
  CumsumOpModel<int32_t> m({TensorType_INT32, {2, 4}}, {TensorType_INT32, {}},
                           false, false);

  m.PopulateTensor<int>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {1});

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({1, 3, 6, 10, 5, 11, 18, 26}));
#endif  // notdef
}

TF_LITE_MICRO_TEST(CumsumOpTestSimpleIntAxis0Test) {
#ifdef notdef
  CumsumOpModel<int32_t> m({TensorType_INT32, {2, 4}}, {TensorType_INT32, {}},
                           false, false);

  m.PopulateTensor<int>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {0});

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({1, 2, 3, 4, 6, 8, 10, 12}));
#endif  // notdef
}

TF_LITE_MICRO_TEST(CumsumOpTestSimple1DIntTest) {
#ifdef notdef
  CumsumOpModel<int32_t> m({TensorType_INT32, {8}}, {TensorType_INT32, {}},
                           false, false);

  m.PopulateTensor<int>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {0});

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({1, 3, 6, 10, 15, 21, 28, 36}));
#endif  // notdef
}

TF_LITE_MICRO_TEST(CumsumOpTestSimpleIntReverseTest) {
#ifdef notdef
  CumsumOpModel<int32_t> m({TensorType_INT32, {2, 4}}, {TensorType_INT32, {}},
                           false, true);

  m.PopulateTensor<int>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {1});

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({10, 9, 7, 4, 26, 21, 15, 8}));
#endif  // notdef
}

TF_LITE_MICRO_TEST(CumsumOpTestSimpleIntExclusiveTest) {
#ifdef notdef
  CumsumOpModel<int32_t> m({TensorType_INT32, {2, 4}}, {TensorType_INT32, {}},
                           true, false);

  m.PopulateTensor<int>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {1});

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({0, 1, 3, 6, 0, 5, 11, 18}));
#endif  // notdef
}

TF_LITE_MICRO_TEST(CumsumOpTestSimpleFloatTest) {
#ifdef notdef
  CumsumOpModel<float> m({TensorType_FLOAT32, {2, 4}}, {TensorType_FLOAT32, {}},
                         false, false);

  m.PopulateTensor<float>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {1});

  EXPECT_THAT(m.GetOutput(), testing::ElementsAreArray(
                                 ArrayFloatNear({1, 3, 6, 10, 5, 11, 18, 26})));
#endif  // notdef
}

TF_LITE_MICRO_TESTS_END

}  // namespace
}  // namespace testing
}  // namespace tflite
