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
#include <cstdarg>

#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {

namespace ops {
namespace builtin {

TfLiteRegistration* Register_CONVOLUTION_REF();
TfLiteRegistration* Register_CONVOLUTION_GENERIC_OPT();
TfLiteRegistration* Register_CONVOLUTION_MULTITHREADED_OPT();
TfLiteRegistration* Register_CONVOLUTION_CBLAS_OPT();

}  // namespace builtin
}  // namespace ops

namespace {

using ::testing::ElementsAreArray;

class BaseConvolutionOpModel : public SingleOpModel {
 public:
  // TODO(ahentz): Also test different activation types, bias, padding types,
  // stride values.
  BaseConvolutionOpModel(
      TfLiteRegistration* registration, const TensorData& input,
      const TensorData& filter, const TensorData& output, int stride_width = 2,
      int stride_height = 2, enum Padding padding = Padding_VALID,
      enum ActivationFunctionType activation = ActivationFunctionType_NONE,
      int dilation_width_factor = 1, int dilation_height_factor = 1) {
    input_ = AddInput(input);
    filter_ = AddInput(filter);

    int bias_size = GetShape(filter_)[0];
    if (input.type == TensorType_FLOAT32) {
      bias_ = AddInput({TensorType_FLOAT32, {bias_size}});
    } else {
      // This is a quantized version. The scale of 'bias' depends on the scales
      // of input and filter. Supposedly this is correctly set during quantized
      // training.
      auto bias_scale = GetScale(input_) * GetScale(filter_);
      TensorData bias{TensorType_INT32, {bias_size}, 0, 0, bias_scale};
      bias_ = AddInput(bias);
    }

    output_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_CONV_2D, BuiltinOptions_Conv2DOptions,
                 CreateConv2DOptions(
                     builder_, padding, stride_width, stride_height, activation,
                     dilation_width_factor, dilation_height_factor)
                     .Union());

    resolver_ = absl::make_unique<SingleOpResolver>(BuiltinOperator_CONV_2D,
                                                    registration);
    BuildInterpreter({GetShape(input_), GetShape(filter_), GetShape(bias_)});
  }

 protected:
  int input_;
  int filter_;
  int bias_;
  int output_;
};

class ConvolutionOpModel : public BaseConvolutionOpModel {
 public:
  using BaseConvolutionOpModel::BaseConvolutionOpModel;

  void SetFilter(std::initializer_list<float> f) { PopulateTensor(filter_, f); }

  void SetBias(std::initializer_list<float> f) { PopulateTensor(bias_, f); }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

const auto kKernelMap = new std::map<string, TfLiteRegistration*>({
    {"Reference", ops::builtin::Register_CONVOLUTION_REF()},
    {"GenericOptimized", ops::builtin::Register_CONVOLUTION_GENERIC_OPT()},
    {"MultithreadedOptimized",
     ops::builtin::Register_CONVOLUTION_MULTITHREADED_OPT()},
    {"CblasOptimized", ops::builtin::Register_CONVOLUTION_CBLAS_OPT()},
});

class ConvolutionOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kKernelMap;
  }
};

TEST_P(ConvolutionOpTest, SimpleTestFloat32) {
  ConvolutionOpModel m(GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 1}},
                       {TensorType_FLOAT32, {3, 2, 2, 1}},
                       {TensorType_FLOAT32, {}});

  m.SetInput({
      // First batch
      1, 1, 1, 1,  // row = 1
      2, 2, 2, 2,  // row = 2
      // Second batch
      1, 2, 3, 4,  // row = 1
      1, 2, 3, 4,  // row = 2
  });
  m.SetFilter({
      1, 2, 3, 4,    // first 2x2 filter
      -1, 1, -1, 1,  // second 2x2 filter
      -1, -1, 1, 1,  // third 2x2 filter
  });
  m.SetBias({1, 2, 3});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 18, 2, 5,  // first batch, left
                                 18, 2, 5,  // first batch, right
                                 17, 4, 3,  // second batch, left
                                 37, 4, 3,  // second batch, right
                             }));
}

// This test's output is equivalent to the SimpleTestFloat32
// because we break each input into two channels, each with half of the value,
// while keeping the filters for each channel equivalent.
//
// 2 * (A/2) * B = A * B, where the left side is this new test.
TEST_P(ConvolutionOpTest, SimpleTestFloat32WithChannels) {
  ConvolutionOpModel m(GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 2}},
                       {TensorType_FLOAT32, {3, 2, 2, 2}},
                       {TensorType_FLOAT32, {}});

  m.SetInput({
      // First batch
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  // row = 1
      1, 1, 1, 1, 1, 1, 1, 1,                  // row = 2
      // Second batch
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2,  // row = 1
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2   // row = 2
  });
  m.SetFilter({
      1,  1,  2,  2,  3,  3,  4, 4,  // first 2x2 filter
      -1, -1, 1,  1,  -1, -1, 1, 1,  // second 2x2 filter
      -1, -1, -1, -1, 1,  1,  1, 1   // third 2x2 filter
  });
  m.SetBias({1, 2, 3});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 18, 2, 5,  // first batch, left
                                 18, 2, 5,  // first batch, right
                                 17, 4, 3,  // second batch, left
                                 37, 4, 3,  // second batch, right
                             }));
}

TEST_P(ConvolutionOpTest, InputAndFilterSameWidthHeight) {
  ConvolutionOpModel m(GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 1}},
                       {TensorType_FLOAT32, {1, 2, 4, 1}},
                       {TensorType_FLOAT32, {}});

  m.SetInput({
      // First batch
      1, 1, 1, 1,  // row = 1
      2, 2, 2, 2,  // row = 2
      // Second batch
      1, 2, 3, 4,  // row = 1
      1, 2, 3, 4,  // row = 2
  });
  m.SetFilter({
      1, 2, 3, 4,    // row = 1
      -1, -1, 1, 1,  // row = 2
  });
  m.SetBias({0});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({10, 34}));
}

TEST_P(ConvolutionOpTest, PointwiseFloat32) {
  ConvolutionOpModel m(GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 2}},
                       {TensorType_FLOAT32, {1, 1, 1, 2}},
                       {TensorType_FLOAT32, {}}, 1, 1);

  m.SetInput({
      // First batch
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  // row = 1
      1, 1, 1, 1, 1, 1, 1, 1,                  // row = 2
      // Second batch
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2,  // row = 1
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2   // row = 2
  });

  m.SetFilter({
      1, 2,  // first filter
  });
  m.SetBias({0});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 // First batch
                                 1.5, 1.5, 1.5, 1.5,  // row = 1
                                 3., 3., 3., 3.,      // row = 2
                                 // Second batch
                                 1.5, 3., 4.5, 6.,  // row = 1
                                 1.5, 3., 4.5, 6.,  // row = 2
                             }));
}

// TODO(alanchiao): this passes locally, but fails on continuous build system.
// Re-enable when root cause found.
TEST_P(ConvolutionOpTest, DISABLED_PointwiseMultifilterFloat32) {
  ConvolutionOpModel m(GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 2}},
                       {TensorType_FLOAT32, {2, 1, 1, 2}},
                       {TensorType_FLOAT32, {}}, 1, 1);

  m.SetInput({
      // First batch
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  // row = 1
      1, 1, 1, 1, 1, 1, 1, 1,                  // row = 2
      // Second batch
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2,  // row = 1
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2   // row = 2
  });

  m.SetFilter({
      1, 2,  // first filter
      2, 3,  // second filter
  });
  m.SetBias({0});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({
                  1.5, 2.5, 1.5, 2.5, 1.5, 2.5, 1.5, 2.5, 3., 5.,  3.,
                  5.,  3.,  5.,  3.,  5.,  1.5, 2.5, 3.,  5., 4.5, 7.5,
                  6.,  10., 1.5, 2.5, 3.,  5.,  4.5, 7.5, 6., 10.,
              }));
}

TEST_P(ConvolutionOpTest, SimpleTestFloat32WithAnisotropicStrides) {
  ConvolutionOpModel m(GetRegistration(), {TensorType_FLOAT32, {1, 3, 6, 1}},
                       {TensorType_FLOAT32, {1, 2, 2, 1}},
                       {TensorType_FLOAT32, {}},
                       /*stride_width=*/3, /*stride_height=*/1);
  m.SetInput({
      3, 2, 1, -1, -2, -3,  //
      4, 3, 2, -2, -3, -4,  //
      5, 4, 3, -3, -4, -5,  //
  });
  m.SetFilter({
      1, 2,  //
      3, 4,  //
  });
  m.SetBias({-1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 30, -24,  //
                                 40, -34,  //
                             }));
}

TEST_P(ConvolutionOpTest, HandCalculatedFloat32) {
  const int depth = 1;
  const int image_width = 4;
  const int image_height = 3;
  const int image_batch_count = 1;
  const int filter_size = 3;
  const int filter_count = 1;
  const int stride_width = 1;
  const int stride_height = 1;
  const Padding padding = Padding_SAME;
  ConvolutionOpModel m(
      GetRegistration(),
      {TensorType_FLOAT32,
       {image_batch_count, image_height, image_width, depth}},
      {TensorType_FLOAT32, {depth, filter_size, filter_size, filter_count}},
      {TensorType_FLOAT32, {}}, stride_width, stride_height, padding);

  // The image matrix is:
  // |  1 |  2 |  3 |  4 |
  // |  5 |  6 |  7 |  8 |
  // |  9 | 10 | 11 | 12 |
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  // The filter matrix is:
  // | 1 | 4 | 7 |
  // | 2 | 5 | 8 |
  // | 3 | 6 | 9 |
  m.SetFilter({1, 4, 7, 2, 5, 8, 3, 6, 9});
  // No bias for this test.
  m.SetBias({0});

  m.Invoke();
  // We're sliding the 3x3 filter across the 3x4 image, with accesses outside
  // the input set to zero because we're using the 'SAME' padding mode.
  // The calculations behind the expected output are:
  // (1*0)+(4*0)+(7*0)+(2*0)+(5*1)+(8*2)+(3*0)+(6*5)+(9*6)=105
  // (1*0)+(4*0)+(7*0)+(2*1)+(5*2)+(8*3)+(3*5)+(6*6)+(9*7)=150
  // (1*0)+(4*0)+(7*0)+(2*2)+(5*3)+(8*4)+(3*6)+(6*7)+(9*8)=183
  // (1*0)+(4*0)+(7*0)+(2*3)+(5*4)+(8*0)+(3*7)+(6*8)+(9*0)=95
  // (1*0)+(4*1)+(7*2)+(2*0)+(5*5)+(8*6)+(3*0)+(6*9)+(9*10)=235
  // (1*1)+(4*2)+(7*3)+(2*5)+(5*6)+(8*7)+(3*9)+(6*10)+(9*11)=312
  // (1*2)+(4*3)+(7*4)+(2*6)+(5*7)+(8*8)+(3*10)+(6*11)+(9*12)=357
  // (1*3)+(4*4)+(7*0)+(2*7)+(5*8)+(8*0)+(3*11)+(6*12)+(9*0)=178
  // (1*0)+(4*5)+(7*6)+(2*0)+(5*9)+(8*10)+(3*0)+(6*0)+(9*0)=187
  // (1*5)+(4*6)+(7*7)+(2*9)+(5*10)+(8*11)+(3*0)+(6*0)+(9*0)=234
  // (1*6)+(4*7)+(7*8)+(2*10)+(5*11)+(8*12)+(3*0)+(6*0)+(9*0)=261
  // (1*7)+(4*11)+(7*0)+(2*8)+(5*12)+(8*0)+(3*0)+(6*0)+(9*0)=121
  // This means we should end up with this matrix:
  // |  105  |  150  |  183  |   95  |
  // |  235  |  312  |  357  |  178  |
  // |  187  |  234  |  261  |  121  |
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({105, 150, 183, 95, 235, 312, 357,
                                               178, 187, 234, 261, 121}));
}

TEST_P(ConvolutionOpTest, HandCalculatedWithBiasFloat32) {
  const int depth = 1;
  const int image_width = 4;
  const int image_height = 3;
  const int image_batch_count = 1;
  const int filter_size = 3;
  const int filter_count = 1;
  const int stride_width = 1;
  const int stride_height = 1;
  const Padding padding = Padding_SAME;
  ConvolutionOpModel m(
      GetRegistration(),
      {TensorType_FLOAT32,
       {image_batch_count, image_height, image_width, depth}},
      {TensorType_FLOAT32, {depth, filter_size, filter_size, filter_count}},
      {TensorType_FLOAT32, {}}, stride_width, stride_height, padding);

  // The image matrix is:
  // |  1 |  2 |  3 |  4 |
  // |  5 |  6 |  7 |  8 |
  // |  9 | 10 | 11 | 12 |
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  // The filter matrix is:
  // | 1 | 4 | 7 |
  // | 2 | 5 | 8 |
  // | 3 | 6 | 9 |
  m.SetFilter({1, 4, 7, 2, 5, 8, 3, 6, 9});
  // Bias is | 10 |.
  m.SetBias({10});

  m.Invoke();
  // We're sliding the 3x3 filter across the 3x4 image, with accesses outside
  // the input set to zero because we're using the 'SAME' padding mode.
  // The calculations behind the expected output are:
  // (1*0)+(4*0)+(7*0)+(2*0)+(5*1)+(8*2)+(3*0)+(6*5)+(9*6)+10=115
  // (1*0)+(4*0)+(7*0)+(2*1)+(5*2)+(8*3)+(3*5)+(6*6)+(9*7)+10=160
  // (1*0)+(4*0)+(7*0)+(2*2)+(5*3)+(8*4)+(3*6)+(6*7)+(9*8)+10=193
  // (1*0)+(4*0)+(7*0)+(2*3)+(5*4)+(8*0)+(3*7)+(6*8)+(9*0)+10=105
  // (1*0)+(4*1)+(7*2)+(2*0)+(5*5)+(8*6)+(3*0)+(6*9)+(9*10)+10=245
  // (1*1)+(4*2)+(7*3)+(2*5)+(5*6)+(8*7)+(3*9)+(6*10)+(9*11)+10=322
  // (1*2)+(4*3)+(7*4)+(2*6)+(5*7)+(8*8)+(3*10)+(6*11)+(9*12)+10=367
  // (1*3)+(4*4)+(7*0)+(2*7)+(5*8)+(8*0)+(3*11)+(6*12)+(9*0)+10=188
  // (1*0)+(4*5)+(7*6)+(2*0)+(5*9)+(8*10)+(3*0)+(6*0)+(9*0)+10=197
  // (1*5)+(4*6)+(7*7)+(2*9)+(5*10)+(8*11)+(3*0)+(6*0)+(9*0)+10=244
  // (1*6)+(4*7)+(7*8)+(2*10)+(5*11)+(8*12)+(3*0)+(6*0)+(9*0)+10=271
  // (1*7)+(4*11)+(7*0)+(2*8)+(5*12)+(8*0)+(3*0)+(6*0)+(9*0)+10=131
  // This means we should end up with this matrix:
  // |  115  |  160  |  193  |  105  |
  // |  245  |  322  |  367  |  188  |
  // |  197  |  244  |  271  |  131  |
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({115, 160, 193, 105, 245, 322,
                                               367, 188, 197, 244, 271, 131}));
}

TEST_P(ConvolutionOpTest, HandCalculatedWithReluFloat32) {
  const int depth = 1;
  const int image_width = 4;
  const int image_height = 3;
  const int image_batch_count = 1;
  const int filter_size = 3;
  const int filter_count = 1;
  const int stride_width = 1;
  const int stride_height = 1;
  const Padding padding = Padding_SAME;
  ConvolutionOpModel m(
      GetRegistration(),
      {TensorType_FLOAT32,
       {image_batch_count, image_height, image_width, depth}},
      {TensorType_FLOAT32, {depth, filter_size, filter_size, filter_count}},
      {TensorType_FLOAT32, {}}, stride_width, stride_height, padding,
      ActivationFunctionType_RELU);

  // The image matrix is:
  // |  1 |  2 |  3 |  4 |
  // |  5 |  6 |  7 |  8 |
  // |  9 | 10 | 11 | 12 |
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  // The filter matrix is:
  // | 1 | 4 | 7 |
  // | 2 | 5 | 8 |
  // | 3 | 6 | 9 |
  m.SetFilter({1, 4, 7, 2, 5, 8, 3, 6, 9});
  // Bias is | -200 |.
  m.SetBias({-200});

  m.Invoke();
  // We're sliding the 3x3 filter across the 3x4 image, with accesses outside
  // the input set to zero because we're using the 'SAME' padding mode.
  // The calculations behind the expected output are:
  // (1*0)+(4*0)+(7*0)+(2*0)+(5*1)+(8*2)+(3*0)+(6*5)+(9*6)-200=-95
  // (1*0)+(4*0)+(7*0)+(2*1)+(5*2)+(8*3)+(3*5)+(6*6)+(9*7)-200=-50
  // (1*0)+(4*0)+(7*0)+(2*2)+(5*3)+(8*4)+(3*6)+(6*7)+(9*8)-200=-17
  // (1*0)+(4*0)+(7*0)+(2*3)+(5*4)+(8*0)+(3*7)+(6*8)+(9*0)-200=-105
  // (1*0)+(4*1)+(7*2)+(2*0)+(5*5)+(8*6)+(3*0)+(6*9)+(9*10)-200=35
  // (1*1)+(4*2)+(7*3)+(2*5)+(5*6)+(8*7)+(3*9)+(6*10)+(9*11)-200=112
  // (1*2)+(4*3)+(7*4)+(2*6)+(5*7)+(8*8)+(3*10)+(6*11)+(9*12)-200=157
  // (1*3)+(4*4)+(7*0)+(2*7)+(5*8)+(8*0)+(3*11)+(6*12)+(9*0)-200=-22
  // (1*0)+(4*5)+(7*6)+(2*0)+(5*9)+(8*10)+(3*0)+(6*0)+(9*0)-200=-13
  // (1*5)+(4*6)+(7*7)+(2*9)+(5*10)+(8*11)+(3*0)+(6*0)+(9*0)-200=34
  // (1*6)+(4*7)+(7*8)+(2*10)+(5*11)+(8*12)+(3*0)+(6*0)+(9*0)-200=61
  // (1*7)+(4*11)+(7*0)+(2*8)+(5*12)+(8*0)+(3*0)+(6*0)+(9*0)-200=-79
  // All negative values are gated to zero by the Relu activation function.
  // This means we should end up with this matrix:
  // |   0 |   0 |   0 |   0 |
  // |  35 | 112 | 157 |   0 |
  // |   0 |  34 |  61 |   0 |
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 0, 0, 0, 35, 112, 157, 0, 0, 34, 61, 0}));
}

TEST_P(ConvolutionOpTest, HandCalculatedValidFloat32) {
  const int depth = 1;
  const int image_width = 4;
  const int image_height = 3;
  const int image_batch_count = 1;
  const int filter_size = 3;
  const int filter_count = 1;
  const int stride_width = 1;
  const int stride_height = 1;
  const Padding padding = Padding_VALID;
  ConvolutionOpModel m(
      GetRegistration(),
      {TensorType_FLOAT32,
       {image_batch_count, image_height, image_width, depth}},
      {TensorType_FLOAT32, {depth, filter_size, filter_size, filter_count}},
      {TensorType_FLOAT32, {}}, stride_width, stride_height, padding);

  // The image matrix is:
  // |  1 |  2 |  3 |  4 |
  // |  5 |  6 |  7 |  8 |
  // |  9 | 10 | 11 | 12 |
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  // The filter matrix is:
  // | 1 | 4 | 7 |
  // | 2 | 5 | 8 |
  // | 3 | 6 | 9 |
  m.SetFilter({1, 4, 7, 2, 5, 8, 3, 6, 9});
  // No bias for this test.
  m.SetBias({0});

  m.Invoke();
  // We're sliding the 3x3 filter across the 3x4 image, with no accesses outside
  // the input because we're using the 'VALID' padding mode, giving a 2x1
  // output.
  // The calculations behind the expected output are:
  // (1*1)+(4*2)+(7*3)+(2*5)+(5*6)+(8*7)+(3*9)+(6*10)+(9*11)=312
  // (1*2)+(4*3)+(7*4)+(2*6)+(5*7)+(8*8)+(3*10)+(6*11)+(9*12)=357
  // This means we should end up with this matrix:
  // |  312  |  357  |
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({312, 357}));
}

TEST_P(ConvolutionOpTest, SimpleTestFloatWithDilation) {
  const int depth = 1;
  const int image_width = 9;
  const int image_height = 9;
  const int image_batch_count = 1;
  const int filter_size = 3;
  const int filter_count = 1;
  const int stride_width = 1;
  const int stride_height = 1;
  const int dilation_width_factor = 3;
  const int dilation_height_factor = 3;
  const Padding padding = Padding_VALID;
  ConvolutionOpModel m(
      GetRegistration(),
      {TensorType_FLOAT32,
       {image_batch_count, image_height, image_width, depth}},
      {TensorType_FLOAT32, {depth, filter_size, filter_size, filter_count}},
      {TensorType_FLOAT32, {}}, stride_width, stride_height, padding,
      ActivationFunctionType_NONE, dilation_width_factor,
      dilation_height_factor);

  // The image matrix is:
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // clang-format off
  m.SetInput({0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0});
  // clang-format on
  // The filter matrix is:
  // | 1 | 2 | 3 |
  // | 4 | 5 | 6 |
  // | 7 | 8 | 9 |
  m.SetFilter({1, 2, 3, 4, 5, 6, 7, 8, 9});
  // No bias for this test.
  m.SetBias({0});
  m.Invoke();

  // Since the dilation rate is 3 this will reduce the size of the output from
  // 10x10 to 3x3 of all 5s. Specifically:
  // | 5 | 5 | 5 |
  // | 5 | 5 | 5 |
  // | 5 | 5 | 5 |
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 5, 5, 5, 5, 5, 5, 5, 5}));
}

class QuantizedConvolutionOpModel : public BaseConvolutionOpModel {
 public:
  using BaseConvolutionOpModel::BaseConvolutionOpModel;

  void SetInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<uint8_t>(input_, data);
  }

  void SetFilter(std::initializer_list<float> data) {
    QuantizeAndPopulate<uint8_t>(filter_, data);
  }

  void SetBias(std::initializer_list<float> data) {
    QuantizeAndPopulate<int32_t>(bias_, data);
  }

  std::vector<uint8_t> GetOutput() { return ExtractVector<uint8_t>(output_); }
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }
};

// In this tests we set the input and output scales so that the results
// match exactly the 'non-quantized' version.
TEST_P(ConvolutionOpTest, SimpleTestQuantized) {
  QuantizedConvolutionOpModel m(GetRegistration(),
                                {TensorType_UINT8, {2, 2, 4, 1}, -63.5, 64},
                                {TensorType_UINT8, {3, 2, 2, 1}, -63.5, 64},
                                {TensorType_UINT8, {}, -127, 128});
  m.SetInput({
      // First batch
      1, 1, 1, 1,  // row = 1
      2, 2, 2, 2,  // row = 2
      // Second batch
      1, 2, 3, 4,  // row = 1
      1, 2, 3, 4,  // row = 2
  });
  m.SetFilter({
      1, 2, 3, 4,    // first 2x2 filter
      -1, 1, -1, 1,  // second 2x2 filter
      -1, -1, 1, 1,  // third 2x2 filter
  });
  m.SetBias({1, 2, 3});

  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      18, 2, 5,  // first batch, left
                      18, 2, 5,  // first batch, right
                      17, 4, 3,  // second batch, left
                      37, 4, 3,  // second batch, right
                  },
                  1e-5)));
  // For good  measure, let's also verify the quantized values:
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 145, 129, 132,  //
                                 145, 129, 132,  //
                                 144, 131, 130,  //
                                 164, 131, 130,  //
                             }));
}

TEST_P(ConvolutionOpTest, SimpleTestQuantizedOutputMultiplierGreaterThan1) {
  // output_multiplier = 1.0118
  QuantizedConvolutionOpModel quant_op(
      GetRegistration(), {TensorType_UINT8, {2, 2, 4, 1}, -128.5, 128},
      {TensorType_UINT8, {3, 2, 2, 1}, -128.5, 128},
      {TensorType_UINT8, {}, -127, 128});
  ConvolutionOpModel float_op(
      GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 1}},
      {TensorType_FLOAT32, {3, 2, 2, 1}}, {TensorType_FLOAT32, {}});
  std::initializer_list<float> input = {
      // First batch
      1, 1, 1, 1,  // row = 1
      2, 2, 2, 2,  // row = 2
      // Second batch
      1, 2, 3, 4,  // row = 1
      1, 2, 3, 4,  // row = 2
  };
  std::initializer_list<float> filter = {
      1,  2,  3,  4,  // first 2x2 filter
      -1, 1,  -1, 1,  // second 2x2 filter
      -1, -1, 1,  1,  // third 2x2 filter
  };
  std::initializer_list<float> bias = {1, 2, 3};

  quant_op.SetInput(input);
  quant_op.SetFilter(filter);
  quant_op.SetBias(bias);
  quant_op.Invoke();

  float_op.SetInput(input);
  float_op.SetFilter(filter);
  float_op.SetBias(bias);
  float_op.Invoke();

  EXPECT_THAT(quant_op.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(float_op.GetOutput(), 1)));
}

TEST_P(ConvolutionOpTest, SimpleTestQuantizedWithAnisotropicStrides) {
  QuantizedConvolutionOpModel m(GetRegistration(),
                                {TensorType_UINT8, {1, 3, 6, 1}, -63.5, 64},
                                {TensorType_UINT8, {1, 2, 2, 1}, -63.5, 64},
                                {TensorType_UINT8, {}, -127, 128},
                                /*stride_width=*/3, /*stride_height=*/1);
  m.SetInput({
      3, 2, 1, -1, -2, -3,  //
      4, 3, 2, -2, -3, -4,  //
      5, 4, 3, -3, -4, -5,  //
  });
  m.SetFilter({
      1, 2,  //
      3, 4,  //
  });
  m.SetBias({-1});
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput(), ElementsAreArray(ArrayFloatNear({
                                            30, -24,  //
                                            40, -34,  //
                                        })));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 157, 103,  //
                                 167, 93,   //
                             }));
}

TEST_P(ConvolutionOpTest, SimpleTestQuantizedWithDilation) {
  const int depth = 1;
  const int image_width = 9;
  const int image_height = 9;
  const int image_batch_count = 1;
  const int filter_size = 3;
  const int filter_count = 1;
  const int stride_width = 1;
  const int stride_height = 1;
  const int dilation_width_factor = 3;
  const int dilation_height_factor = 3;
  const Padding padding = Padding_VALID;
  QuantizedConvolutionOpModel m(
      GetRegistration(),
      {TensorType_UINT8,
       {image_batch_count, image_height, image_width, depth},
       0,
       255},
      {TensorType_UINT8,
       {depth, filter_size, filter_size, filter_count},
       0,
       255},
      {TensorType_UINT8, {}, 0, 255}, stride_width, stride_height, padding,
      ActivationFunctionType_NONE, dilation_width_factor,
      dilation_height_factor);

  // The image matrix is:
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  // clang-format off
  m.SetInput({0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0});
  // clang-format on
  // The filter matrix is:
  // | 1 | 2 | 3 |
  // | 4 | 5 | 6 |
  // | 7 | 8 | 9 |
  m.SetFilter({1, 2, 3, 4, 5, 6, 7, 8, 9});
  // No bias for this test.
  m.SetBias({0});
  m.Invoke();

  // Since the dilation rate is 3 this will reduce the size of the output from
  // 10x10 to 3x3 of all 5s. Specifically:
  // | 5 | 5 | 5 |
  // | 5 | 5 | 5 |
  // | 5 | 5 | 5 |
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray({5, 5, 5, 5, 5, 5, 5, 5, 5}));
}

class HybridConvolutionOpModel : public BaseConvolutionOpModel {
 public:
  using BaseConvolutionOpModel::BaseConvolutionOpModel;

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  void SetFilter(std::initializer_list<float> f) {
    SymmetricQuantizeAndPopulate(filter_, f);
  }

  void SetBias(std::initializer_list<float> data) {
    PopulateTensor(bias_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

TEST_P(ConvolutionOpTest, SimpleTestHybrid) {
  HybridConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 1}},
      {TensorType_UINT8, {3, 2, 2, 1}}, {TensorType_FLOAT32, {}});

  m.SetInput({
      // First batch
      1, 1, 1, 1,  // row = 1
      2, 2, 2, 2,  // row = 2
      // Second batch
      1, 2, 3, 4,  // row = 1
      1, 2, 3, 4,  // row = 2
  });
  m.SetFilter({
      1, 2, 3, 4,    // first 2x2 filter
      -1, 1, -1, 1,  // second 2x2 filter
      -1, -1, 1, 1,  // third 2x2 filter
  });
  m.SetBias({1, 2, 3});

  m.Invoke();

  // Example: we get 17.1577 instead of 17.
  //
  // Second batch:
  // 1 2 3 4  -> 32 64 95 127 with scale factor 127/4.
  // 1 2 3 4     32 64 95 127
  //
  // First filter:
  // 1 2  -> 32 64  with scale factor of 127/4.
  // 3 4     95 127
  //
  // The left half of the input gives us 16288. Multiply by (4/127)^2 for
  // dequantization and adding 1 for the bias gives us the result. and adding
  // the bias gives us the result.
  //
  // The optimized kernel converts the input into this matrix via Im2Col
  //
  // 1 1 2 2
  // 1 1 2 2
  // 1 2 1 2
  // 3 4 3 4
  //
  // and multiplies it with the filter directly.
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     18, 2, 5,  // first batch, left
                                     18, 2, 5,  // first batch, right
                                     17, 4, 3,  // second batch, left
                                     37, 4, 3,  // second batch, right
                                 },
                                 0.16)));
}

// This test's output is equivalent to the SimpleTestHybrid
// because we break each input into two channels, each with half of the value,
// while keeping the filters for each channel equivalent.
//
// 2 * (A/2) * B = A * B, where the left side is this new test.
TEST_P(ConvolutionOpTest, SimpleTestHybridWithChannels) {
  HybridConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 2}},
      {TensorType_UINT8, {3, 2, 2, 2}}, {TensorType_FLOAT32, {}});

  m.SetInput({
      // First batch
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  // row = 1
      1, 1, 1, 1, 1, 1, 1, 1,                  // row = 2
      // Second batch
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2,  // row = 1
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2   // row = 2
  });
  m.SetFilter({
      1,  1,  2,  2,  3,  3,  4, 4,  // first 2x2 filter
      -1, -1, 1,  1,  -1, -1, 1, 1,  // second 2x2 filter
      -1, -1, -1, -1, 1,  1,  1, 1   // third 2x2 filter
  });
  m.SetBias({1, 2, 3});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     18, 2, 5,  // first batch, left
                                     18, 2, 5,  // first batch, right
                                     17, 4, 3,  // second batch, left
                                     37, 4, 3,  // second batch, right
                                 },
                                 0.16)));
}

TEST_P(ConvolutionOpTest, PointwiseHybrid) {
  HybridConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 2}},
      {TensorType_UINT8, {1, 1, 1, 2}}, {TensorType_FLOAT32, {}}, 1, 1);

  m.SetInput({
      // First batch
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  // row = 1
      1, 1, 1, 1, 1, 1, 1, 1,                  // row = 2
      // Second batch
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2,  // row = 1
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2   // row = 2
  });

  m.SetFilter({
      1, 2,  // first filter
  });
  m.SetBias({0});

  m.Invoke();

  // Example: we get 3.03156 instead of 3.
  //
  // Second batch:
  // 0.5 0.5 1 1 1.5 1.5 2 2  -> 32 32 64 64 95 95 127 127 with scale factor
  // 127/2. We care about the two 64's.
  //
  // Filter:
  // 64 127 with scale factor of 127/2.
  //
  // (64 * 64 + 64 * 127) * (2/127)^2 gives us the expected result.
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.5, 1.5, 1.5, 1.5,  // first batch, row = 1
                      3., 3., 3., 3.,      // first batch, row = 2
                      1.5, 3., 4.5, 6.,    // second batch, row = 1
                      1.5, 3., 4.5, 6.,    // second batch, row = 2
                  },
                  0.0316)));
}

// TODO(alanchiao): this passes locally, but fails on continuous build system.
// Re-enable when root cause found.
TEST_P(ConvolutionOpTest, DISABLED_PointwiseMultifilterHybrid) {
  HybridConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 2}},
      {TensorType_UINT8, {2, 1, 1, 2}}, {TensorType_FLOAT32, {}}, 1, 1);

  m.SetInput({
      // First batch
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  // row = 1
      1, 1, 1, 1, 1, 1, 1, 1,                  // row = 2
      // Second batch
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2,  // row = 1
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2   // row = 2
  });

  m.SetFilter({
      1, 2,  // first filter
      2, 3,  // second filter
  });
  m.SetBias({0});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.5, 2.5, 1.5, 2.5, 1.5, 2.5, 1.5, 2.5, 3., 5.,  3.,
                      5.,  3.,  5.,  3.,  5.,  1.5, 2.5, 3.,  5., 4.5, 7.5,
                      6.,  10., 1.5, 2.5, 3.,  5.,  4.5, 7.5, 6., 10.,
                  },
                  0.0474)));
}

INSTANTIATE_TEST_CASE_P(
    ConvolutionOpTest, ConvolutionOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kKernelMap)));

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
