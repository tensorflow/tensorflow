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
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/custom_ops_register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace ops {
namespace custom {

TfLiteRegistration* Register_CUMSUM();

namespace {

template <typename T>
class CumsumOpModel : public SingleOpModel {
 public:
  CumsumOpModel(const TensorData& input, const TensorData& output,
                bool exclusive, bool reverse) {
    input_ = AddInput(input);
    axis_ = AddInput({TensorType_INT32, {1}});

    output_ = AddOutput(output);

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Bool("exclusive", exclusive);
      fbb.Bool("reverse", reverse);
    });
    fbb.Finish();
    SetCustomOp("Cumsum", fbb.GetBuffer(), Register_CUMSUM);

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

  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({1, 3, 6, 10, 5, 11, 18, 26}));
}

TEST(CumsumOpTest, SimpleIntAxis0Test) {
  CumsumOpModel<int32_t> m({TensorType_INT32, {2, 4}}, {TensorType_INT32, {}},
                           false, false);

  m.PopulateTensor<int>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {0});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({1, 2, 3, 4, 6, 8, 10, 12}));
}

TEST(CumsumOpTest, Simple1DIntTest) {
  CumsumOpModel<int32_t> m({TensorType_INT32, {8}}, {TensorType_INT32, {}},
                           false, false);

  m.PopulateTensor<int>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {0});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({1, 3, 6, 10, 15, 21, 28, 36}));
}

TEST(CumsumOpTest, SimpleIntReverseTest) {
  CumsumOpModel<int32_t> m({TensorType_INT32, {2, 4}}, {TensorType_INT32, {}},
                           false, true);

  m.PopulateTensor<int>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {1});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({10, 9, 7, 4, 26, 21, 15, 8}));
}

TEST(CumsumOpTest, SimpleIntExclusiveTest) {
  CumsumOpModel<int32_t> m({TensorType_INT32, {2, 4}}, {TensorType_INT32, {}},
                           true, false);

  m.PopulateTensor<int>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {1});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({0, 1, 3, 6, 0, 5, 11, 18}));
}

TEST(CumsumOpTest, SimpleFloatTest) {
  CumsumOpModel<float> m({TensorType_FLOAT32, {2, 4}}, {TensorType_FLOAT32, {}},
                         false, false);

  m.PopulateTensor<float>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {1});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), testing::ElementsAreArray(
                                 ArrayFloatNear({1, 3, 6, 10, 5, 11, 18, 26})));
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
