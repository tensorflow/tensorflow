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

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace ops {
namespace builtin {

namespace {

template <typename T>
class CumsumOpModel : public SingleOpModel {
 public:
  CumsumOpModel(const TensorData& input, const TensorData& output,
                bool exclusive, bool reverse) {
    input_ = AddInput(input);
    axis_ = AddInput({TensorType_INT32, {1}});

    output_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_CUMSUM, BuiltinOptions_CumsumOptions,
                 CreateCumsumOptions(builder_, exclusive, reverse).Union());

    BuildInterpreter({GetShape(input_), GetShape(axis_)});
  }

  int input() { return input_; }
  int axis() { return axis_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }

 private:
  int input_;
  int axis_;
  int output_;
};

TEST(CumsumOpTest, SimpleIntTest) {
  CumsumOpModel<int32_t> m({TensorType_INT32, {2, 4}}, {TensorType_INT32, {}},
                           false, false);

  m.PopulateTensor<int>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {1});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({1, 3, 6, 10, 5, 11, 18, 26}));
}

TEST(CumsumOpTest, SimpleInt64Test) {
  CumsumOpModel<int64_t> m({TensorType_INT64, {2, 4}}, {TensorType_INT64, {}},
                           false, false);

  m.PopulateTensor<int64_t>(
      m.input(), {100000000001l, 100000000002l, 100000000003l, 100000000004l,
                  100000000005l, 100000000006l, 100000000007l, 100000000008l});
  m.PopulateTensor<int>(m.axis(), {1});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), testing::ElementsAreArray(
                                 {100000000001l, 200000000003l, 300000000006l,
                                  400000000010l, 100000000005l, 200000000011l,
                                  300000000018l, 400000000026l}));
}

TEST(CumsumOpTest, SimpleIntAxis0Test) {
  CumsumOpModel<int32_t> m({TensorType_INT32, {2, 4}}, {TensorType_INT32, {}},
                           false, false);

  m.PopulateTensor<int>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {0});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({1, 2, 3, 4, 6, 8, 10, 12}));
}

TEST(CumsumOpTest, Simple1DIntTest) {
  CumsumOpModel<int32_t> m({TensorType_INT32, {8}}, {TensorType_INT32, {}},
                           false, false);

  m.PopulateTensor<int>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {0});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({1, 3, 6, 10, 15, 21, 28, 36}));
}

TEST(CumsumOpTest, SimpleIntReverseTest) {
  CumsumOpModel<int32_t> m({TensorType_INT32, {2, 4}}, {TensorType_INT32, {}},
                           false, true);

  m.PopulateTensor<int>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {1});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({10, 9, 7, 4, 26, 21, 15, 8}));
}

TEST(CumsumOpTest, SimpleIntExclusiveTest) {
  CumsumOpModel<int32_t> m({TensorType_INT32, {2, 4}}, {TensorType_INT32, {}},
                           true, false);

  m.PopulateTensor<int>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {1});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({0, 1, 3, 6, 0, 5, 11, 18}));
}

TEST(CumsumOpTest, SimpleFloatTest) {
  CumsumOpModel<float> m({TensorType_FLOAT32, {2, 4}}, {TensorType_FLOAT32, {}},
                         false, false);

  m.PopulateTensor<float>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {1});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), testing::ElementsAreArray(
                                 ArrayFloatNear({1, 3, 6, 10, 5, 11, 18, 26})));
}

}  // namespace
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
