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

#include <stdint.h>

#include <complex>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/custom_ops_register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace ops {
namespace custom {

namespace {

using ::testing::ElementsAreArray;

class Irfft2dOpModel : public SingleOpModel {
 public:
  Irfft2dOpModel(const TensorData& input, const TensorData& fft_lengths) {
    input_ = AddInput(input);
    fft_lengths_ = AddInput(fft_lengths);
    TensorType output_type = TensorType_FLOAT32;
    output_ = AddOutput({output_type, {}});

    const std::vector<uint8_t> custom_option;
    SetCustomOp("Irfft2d", custom_option, Register_IRFFT2D);
    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }
  int fft_lengths() { return fft_lengths_; }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int fft_lengths_;
  int output_;
};

TEST(Irfft2dOpTest, FftLengthMatchesInputSize) {
  Irfft2dOpModel model({TensorType_COMPLEX64, {4, 3}}, {TensorType_INT32, {2}});
  // clang-format off
  model.PopulateTensor<std::complex<float>>(model.input(), {
    {75, 0},  {-6, -1}, {9, 0},  {-10, 5},  {-3, 2}, {-6, 11},
    {-15, 0}, {-2, 13}, {-5, 0}, {-10, -5}, {3, -6}, {-6, -11}
  });
  // clang-format on
  model.PopulateTensor<int32_t>(model.fft_lengths(), {4, 4});
  model.Invoke();

  float expected_result[16] = {1, 2, 3, 4, 3, 8, 6, 3, 5, 2, 7, 6, 9, 5, 8, 3};
  EXPECT_THAT(model.GetOutput(), ElementsAreArray(expected_result));
}

TEST(Irfft2dOpTest, FftLengthSmallerThanInputSize) {
  Irfft2dOpModel model({TensorType_COMPLEX64, {4, 3}}, {TensorType_INT32, {2}});
  // clang-format off
  model.PopulateTensor<std::complex<float>>(model.input(), {
    {75, 0},  {-6, -1}, {9, 0},  {-10, 5},  {-3, 2}, {-6, 11},
    {-15, 0}, {-2, 13}, {-5, 0}, {-10, -5}, {3, -6}, {-6, -11}
  });
  // clang-format on
  model.PopulateTensor<int32_t>(model.fft_lengths(), {2, 2});
  model.Invoke();

  float expected_result[4] = {14, 18.5, 20.5, 22};
  EXPECT_THAT(model.GetOutput(), ElementsAreArray(expected_result));
}

TEST(Irfft2dOpTest, FftLengthGreaterThanInputSize) {
  Irfft2dOpModel model({TensorType_COMPLEX64, {4, 3}}, {TensorType_INT32, {2}});
  // clang-format off
  model.PopulateTensor<std::complex<float>>(model.input(), {
    {75, 0},  {-6, -1}, {9, 0},  {-10, 5},  {-3, 2}, {-6, 11},
    {-15, 0}, {-2, 13}, {-5, 0}, {-10, -5}, {3, -6}, {-6, -11}
  });
  // clang-format on
  model.PopulateTensor<int32_t>(model.fft_lengths(), {4, 8});
  model.Invoke();

  // clang-format off
  float expected_result[32] = {
    0.25, 0.54289322, 1.25, 1.25, 1.25, 1.95710678, 2.25, 1.25,
    1.25, 2.85355339, 4.25, 3.91421356, 2.75, 2.14644661, 1.75, 1.08578644,
    3., 1.43933983, 0.5, 2.14644661, 4., 3.56066017, 2.5, 2.85355339,
    5.625, 3.65533009, 1.375, 3.3017767, 5.125, 2.59466991, 0.375, 2.9482233
  };
  // clang-format on
  EXPECT_THAT(model.GetOutput(), ElementsAreArray(expected_result));
}

TEST(Irfft2dOpTest, InputDimsGreaterThan2) {
  Irfft2dOpModel model({TensorType_COMPLEX64, {2, 2, 3}},
                       {TensorType_INT32, {2}});
  // clang-format off
  model.PopulateTensor<std::complex<float>>(model.input(), {
    {30., 0.}, {-5, -3.}, { -4., 0.},
    {-10., 0.}, {1., 7.}, {  0., 0.},
    {58., 0.}, {-18., 6.}, { 26., 0.},
    {-18., 0.}, { 14., 2.}, {-18., 0.}
  });
  // clang-format on
  model.PopulateTensor<int32_t>(model.fft_lengths(), {2, 4});
  model.Invoke();

  float expected_result[16] = {1., 2., 3., 4., 3., 8., 6.,  3.,
                               5., 2., 7., 6., 7., 3., 23., 5.};
  EXPECT_THAT(model.GetOutput(), ElementsAreArray(expected_result));
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
