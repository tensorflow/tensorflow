/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <complex>

#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class CastOpModel : public SingleOpModel {
 public:
  CastOpModel(const TensorData& input, const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_CAST, BuiltinOptions_CastOptions,
                 CreateCastOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }

  int input() const { return input_; }
  int output() const { return output_; }

 protected:
  int input_;
  int output_;
};

TEST(CastOpModel, CastIntToFloat) {
  CastOpModel m({TensorType_INT64, {2, 3}}, {TensorType_FLOAT32, {2, 3}});
  m.PopulateTensor<int64_t>(m.input(), {100, 200, 300, 400, 500, 600});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray({100.f, 200.f, 300.f, 400.f, 500.f, 600.f}));
}

TEST(CastOpModel, CastFloatToInt) {
  CastOpModel m({TensorType_FLOAT32, {3, 2}}, {TensorType_INT32, {3, 2}});
  m.PopulateTensor<float>(m.input(), {100.f, 20.f, 3.f, 0.4f, 0.999f, 1.1f});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<int>(m.output()),
              ElementsAreArray({100, 20, 3, 0, 0, 1}));
}

TEST(CastOpModel, CastFloatToBool) {
  CastOpModel m({TensorType_FLOAT32, {3, 2}}, {TensorType_BOOL, {3, 2}});
  m.PopulateTensor<float>(m.input(), {100.f, -1.0f, 0.f, 0.4f, 0.999f, 1.1f});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<bool>(m.output()),
              ElementsAreArray({true, true, false, true, true, true}));
}

TEST(CastOpModel, CastBoolToFloat) {
  CastOpModel m({TensorType_BOOL, {3, 2}}, {TensorType_FLOAT32, {3, 2}});
  m.PopulateTensor<bool>(m.input(), {true, true, false, true, false, true});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray({1.f, 1.0f, 0.f, 1.0f, 0.0f, 1.0f}));
}

TEST(CastOpModel, CastComplex64ToFloat) {
  CastOpModel m({TensorType_COMPLEX64, {2, 3}}, {TensorType_FLOAT32, {2, 3}});
  m.PopulateTensor<std::complex<float>>(
      m.input(),
      {std::complex<float>(1.0f, 11.0f), std::complex<float>(2.0f, 12.0f),
       std::complex<float>(3.0f, 13.0f), std::complex<float>(4.0f, 14.0f),
       std::complex<float>(5.0f, 15.0f), std::complex<float>(6.0f, 16.0f)});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
}

TEST(CastOpModel, CastFloatToComplex64) {
  CastOpModel m({TensorType_FLOAT32, {2, 3}}, {TensorType_COMPLEX64, {2, 3}});
  m.PopulateTensor<float>(m.input(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  m.Invoke();
  EXPECT_THAT(
      m.ExtractVector<std::complex<float>>(m.output()),
      ElementsAreArray(
          {std::complex<float>(1.0f, 0.0f), std::complex<float>(2.0f, 0.0f),
           std::complex<float>(3.0f, 0.0f), std::complex<float>(4.0f, 0.0f),
           std::complex<float>(5.0f, 0.0f), std::complex<float>(6.0f, 0.0f)}));
}

TEST(CastOpModel, CastComplex64ToInt) {
  CastOpModel m({TensorType_COMPLEX64, {2, 3}}, {TensorType_INT32, {2, 3}});
  m.PopulateTensor<std::complex<float>>(
      m.input(),
      {std::complex<float>(1.0f, 11.0f), std::complex<float>(2.0f, 12.0f),
       std::complex<float>(3.0f, 13.0f), std::complex<float>(4.0f, 14.0f),
       std::complex<float>(5.0f, 15.0f), std::complex<float>(6.0f, 16.0f)});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<int>(m.output()),
              ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(CastOpModel, CastIntToComplex64) {
  CastOpModel m({TensorType_INT32, {2, 3}}, {TensorType_COMPLEX64, {2, 3}});
  m.PopulateTensor<int>(m.input(), {1, 2, 3, 4, 5, 6});
  m.Invoke();
  EXPECT_THAT(
      m.ExtractVector<std::complex<float>>(m.output()),
      ElementsAreArray(
          {std::complex<float>(1.0f, 0.0f), std::complex<float>(2.0f, 0.0f),
           std::complex<float>(3.0f, 0.0f), std::complex<float>(4.0f, 0.0f),
           std::complex<float>(5.0f, 0.0f), std::complex<float>(6.0f, 0.0f)}));
}

TEST(CastOpModel, CastComplex64ToComplex64) {
  CastOpModel m({TensorType_COMPLEX64, {2, 3}}, {TensorType_COMPLEX64, {2, 3}});
  m.PopulateTensor<std::complex<float>>(
      m.input(),
      {std::complex<float>(1.0f, 11.0f), std::complex<float>(2.0f, 12.0f),
       std::complex<float>(3.0f, 13.0f), std::complex<float>(4.0f, 14.0f),
       std::complex<float>(5.0f, 15.0f), std::complex<float>(6.0f, 16.0f)});
  m.Invoke();
  EXPECT_THAT(
      m.ExtractVector<std::complex<float>>(m.output()),
      ElementsAreArray(
          {std::complex<float>(1.0f, 11.0f), std::complex<float>(2.0f, 12.0f),
           std::complex<float>(3.0f, 13.0f), std::complex<float>(4.0f, 14.0f),
           std::complex<float>(5.0f, 15.0f),
           std::complex<float>(6.0f, 16.0f)}));
}

}  // namespace
}  // namespace tflite
int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
