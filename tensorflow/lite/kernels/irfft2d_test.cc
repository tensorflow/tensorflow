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

#include <gtest/gtest.h>
#include <stdint.h>

#include <complex>
#include <vector>

#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace ops {
namespace builtin {

namespace {

using std::complex;
using ::testing::ElementsAreArray;

class Irfft2dOpModel : public SingleOpModel {
 public:
  Irfft2dOpModel(const TensorData& input, const TensorData& fft_lengths) {
    input_ = AddInput(input);
    fft_lengths_ = AddInput(fft_lengths);
    TensorType output_type = TensorType_COMPLEX64;
    output_ = AddOutput({output_type, {}});

    SetBuiltinOp(BuiltinOperator_RFFT2D, BuiltinOptions_Rfft2dOptions,
                 CreateRfft2dOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }
  int fft_lengths() { return fft_lengths_; }

  std::vector<complex<float>> GetOutput() {
    return ExtractVector<complex<float>>(output_);
  }
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

TEST(Irfft2dOpTest, DISABLED_FftLengthSmallerThanInputSize) {
  Irfft2dOpModel model({TensorType_COMPLEX64, {4, 3}}, {TensorType_INT32, {2}});
  // clang-format off
  model.PopulateTensor<std::complex<float>>(model.input(), {
    {75, 0},  {-6, -1}, {9, 0},  {-10, 5},  {-3, 2}, {-6, 11},
    {-15, 0}, {-2, 13}, {-5, 0}, {-10, -5}, {3, -6}, {-6, -11}
  });
  // clang-format on
  model.PopulateTensor<int32_t>(model.fft_lengths(), {4, 4});
  model.Invoke();

  float expected_result[20] = {1, 2, 3, 4, 0, 3, 8, 6, 3, 0,
                               5, 2, 7, 6, 0, 9, 5, 8, 3, 0};
  EXPECT_THAT(model.GetOutput(), ElementsAreArray(expected_result));
}

TEST(Irfft2dOpTest, DISABLED_FftLengthGreaterThanInputSize) {
  Irfft2dOpModel model({TensorType_COMPLEX64, {4, 5}}, {TensorType_INT32, {2}});
  // clang-format off
  model.PopulateTensor<std::complex<float>>(model.input(), {
    {50, 0}, {8.29289341, -33.6776695}, {-7, 1}, {9.70710659, -1.67766953}, {0, 0},
    {-10, -20}, {-16.3639603, -1.12132037}, {-5, 1}, {-7.19238806, -2.05025244}, {-6, 2},
    {10, 0}, {-4.7781744, -6.12132025}, {-1, 11}, {10.7781744, 1.87867963}, {4, 0},
    {-10, 20}, {11.1923885, 11.9497471}, {5, -5}, {-3.63603902, -3.12132025}, {-6, -2}
  });
  // clang-format on
  model.PopulateTensor<int32_t>(model.fft_lengths(), {4, 8});
  model.Invoke();

  float expected_result[12] = {1, 2, 3, 4, 3, 8, 6, 3, 5, 2, 7, 6};
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
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
