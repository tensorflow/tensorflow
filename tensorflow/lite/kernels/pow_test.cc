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
#include <math.h>
#include <stdint.h>

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/internal/test_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

template <typename T>
class PowOpModel : public SingleOpModel {
 public:
  PowOpModel(const TensorData& input1, const TensorData& input2,
             const TensorData& output, bool allocate = true) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_POW, BuiltinOptions_PowOptions,
                 CreatePowOptions(builder_).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)},
                     /*num_threads=*/-1, /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/true,
                     /*allocate_and_delegate=*/allocate);
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input1_;
  int input2_;
  int output_;
};

template <typename T>
class FloatPowTest : public ::testing::Test {};

using FloatPowTestTypes = ::testing::Types<float, half, Eigen::bfloat16>;
TYPED_TEST_SUITE(FloatPowTest, FloatPowTestTypes);

TEST(PowOpModel, Simple) {
  PowOpModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {12, 2, 7, 8});
  model.PopulateTensor<int32_t>(model.input2(), {1, 2, 3, 1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(12, 4, 343, 8));
}

TEST(PowOpModel, NegativeAndZeroValue) {
  PowOpModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {0, 2, -7, 8});
  model.PopulateTensor<int32_t>(model.input2(), {1, 2, 3, 0});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 4, -343, 1));
}

TYPED_TEST(FloatPowTest, Float) {
  using T = TypeParam;
  PowOpModel<T> model({GetTensorType<T>(), {1, 2, 2, 1}},
                      {GetTensorType<T>(), {1, 2, 2, 1}},
                      {GetTensorType<T>(), {}}, /*allocate=*/false);
  TFLITE_ALLOCATE_AND_CHECK(T, &model);
  model.template PopulateTensor<T>(model.input1(), {0.3, 0.4, 0.7, 5.8});
  model.template PopulateTensor<T>(model.input2(), {0.5, 2.7, 3.1, 3.2});
  TFLITE_INVOKE_AND_CHECK(T, &model);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(
                  ArrayFloatNear({0.5477226, 0.08424846, 0.33098164, 277.313},
                                 /*max_abs_err=*/1e-3,
                                 /*fp16_max_abs_err=*/1e-2)));
}

TYPED_TEST(FloatPowTest, NegativeFloatTest) {
  using T = TypeParam;
  PowOpModel<T> model({GetTensorType<T>(), {1, 2, 2, 1}},
                      {GetTensorType<T>(), {1, 2, 2, 1}},
                      {GetTensorType<T>(), {}}, /*allocate=*/false);
  TFLITE_ALLOCATE_AND_CHECK(T, &model);
  model.template PopulateTensor<T>(model.input1(), {0.3, 0.4, 0.7, 5.8});
  model.template PopulateTensor<T>(model.input2(), {0.5, -2.7, 3.1, -3.2});
  TFLITE_INVOKE_AND_CHECK(T, &model);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(
                  ArrayFloatNear({0.5477226, 11.869653, 0.33098164, 0.003606},
                                 /*max_abs_err=*/1e-3,
                                 /*fp16_max_abs_err=*/1e-2)));
}

TEST(PowOpModel, BroadcastTest) {
  PowOpModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {1}}, {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {12, 2, 7, 8});
  model.PopulateTensor<int32_t>(model.input2(), {4});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(20736, 16, 2401, 4096));
}

TYPED_TEST(FloatPowTest, BroadcastFloatTest) {
  using T = TypeParam;
  PowOpModel<T> model({GetTensorType<T>(), {1, 2, 2, 1}},
                      {GetTensorType<T>(), {1}}, {GetTensorType<T>(), {}},
                      /*allocate=*/false);
  TFLITE_ALLOCATE_AND_CHECK(T, &model);
  model.template PopulateTensor<T>(model.input1(), {12, 2, 7, 8});
  model.template PopulateTensor<T>(model.input2(), {4});
  TFLITE_INVOKE_AND_CHECK(T, &model);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(ArrayFloatNear({20736, 16, 2401, 4096},
                                              NumericLimits<T>::epsilon())));
}

template <typename T>
void CalculateTrueResults(const std::vector<T>& input_data, T exponent,
                          int flat_size, std::vector<T>* output_data) {
  for (int i = 0; i < flat_size; ++i) {
    output_data->at(i) = static_cast<T>(std::pow(
        static_cast<float>(input_data[i]), static_cast<float>(exponent)));
  }
}

TYPED_TEST(FloatPowTest, FloatSingleIntegerExponentTest) {
  using T = TypeParam;
  PowOpModel<T> model({GetTensorType<T>(), {1, 2, 2, 1}},
                      {GetTensorType<T>(), {1}}, {GetTensorType<T>(), {}},
                      /*allocate=*/false);
  TFLITE_ALLOCATE_AND_CHECK(T, &model);
  const int input_size = 1 * 2 * 2 * 1;
  for (int i = 1; i < 20; ++i) {
    std::vector<float> input_data(input_size);
    for (int index = 0; index < input_size; ++index) {
      // For exponent is float case, if base < 0, we will result in nan, so
      // we only populate positive base.
      input_data[index] = UniformRandomFloat(0, 1.5);
    }
    model.template PopulateTensor<T>(model.input1(), ToVector<T>(input_data));
    float exponent = static_cast<float>(i);
    // Random deviate exponent, e.g., 1.99999 or 2.00001.
    exponent += UniformRandomInt(-1, 1) * 1e-5;
    model.template PopulateTensor<T>(model.input2(), ToVector<T>({exponent}));
    TFLITE_INVOKE_AND_CHECK(T, &model);

    std::vector<T> expected_output(input_size);
    CalculateTrueResults<T>(ToVector<T>(input_data), static_cast<T>(exponent),
                            input_size, &expected_output);
    EXPECT_THAT(model.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                       ToVector<float>(expected_output),
                                       /*max_abs_err=*/1e-2,
                                       /*fp16_max_abs_err=*/1e-2)));
  }
}

TEST(PowOpModel, IntSingleIntegerExponentTest) {
  PowOpModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {1}}, {TensorType_INT32, {}});
  const int input_size = 1 * 2 * 2 * 1;
  for (int i = 1; i < 20; ++i) {
    std::vector<int32_t> input_data(input_size);
    for (int index = 0; index < input_size; ++index) {
      input_data[index] = UniformRandomInt(-2, -2);
    }
    model.PopulateTensor<int32_t>(model.input1(), input_data);
    int exponent = i;
    model.PopulateTensor<int32_t>(model.input2(), {exponent});
    ASSERT_EQ(model.Invoke(), kTfLiteOk);
    EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
    std::vector<int32_t> output_data(input_size);
    CalculateTrueResults(input_data, exponent, input_size, &output_data);
    EXPECT_THAT(model.GetOutput(), ElementsAreArray(output_data));
  }
}

}  // namespace
}  // namespace tflite
