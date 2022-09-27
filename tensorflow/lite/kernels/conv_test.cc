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
#include <stddef.h>
#include <stdint.h>

#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {

namespace ops {
namespace builtin {

TfLiteRegistration* Register_CONV_2D_UINT8();
TfLiteRegistration* Register_CONVOLUTION_REF();
TfLiteRegistration* Register_CONVOLUTION_GENERIC_OPT();
TfLiteRegistration* Register_CONVOLUTION_MULTITHREADED_OPT();
TfLiteRegistration* Register_CONVOLUTION_CBLAS_OPT();

}  // namespace builtin
}  // namespace ops

namespace {

using ::testing::ElementsAreArray;

template <typename FilterType>
class BaseConvolutionOpModel : public SingleOpModel {
 public:
  BaseConvolutionOpModel(
      TfLiteRegistration* registration, const TensorData& input,
      const TensorData& filter, const TensorData& output, int stride_width = 2,
      int stride_height = 2, enum Padding padding = Padding_VALID,
      enum ActivationFunctionType activation = ActivationFunctionType_NONE,
      int dilation_width_factor = 1, int dilation_height_factor = 1,
      int num_threads = -1,
      std::initializer_list<FilterType> filter_data = {}) {
    input_ = AddInput(input);

    if (filter_data.size()) {
      filter_ = AddConstInput(filter, filter_data);
    } else {
      filter_ = AddInput(filter);
    }

    int bias_size = GetShape(filter_)[0];
    if (input.type == TensorType_FLOAT32) {
      bias_ = AddInput({TensorType_FLOAT32, {bias_size}});
    } else {
      // This is a quantized version. The scale of 'bias' depends on the scales
      // of input and filter. Supposedly this is correctly set during quantized
      // training.
      if (filter.per_channel_quantization) {
        // per channel quantization.
        std::vector<float> bias_scale(
            filter.per_channel_quantization_scales.size());
        std::vector<int64_t> bias_zero_points(
            filter.per_channel_quantization_scales.size());
        for (size_t i = 0; i < filter.per_channel_quantization_scales.size();
             ++i) {
          bias_scale[i] =
              input.scale * filter.per_channel_quantization_scales[i];
          bias_zero_points[i] = 0;
        }
        tflite::TensorType bias_type = TensorType_INT32;
        if (input.type == TensorType_INT16) {
          // In case of 16-bit, the bias type is set to be int 64.
          bias_type = TensorType_INT64;
        }
        TensorData bias{bias_type,
                        {bias_size},
                        /*min=*/0,
                        /*max=*/0,
                        /*scale=*/0,
                        /*zero_point=*/0,
                        true,
                        /*per_channel_quantization_scales=*/bias_scale,
                        /*per_channel_quantization_offsets=*/bias_zero_points,
                        /*channel_index==*/0};
        bias_ = AddInput(bias);
      } else {
        // per tensor quantization.
        auto bias_scale = GetScale(input_) * GetScale(filter_);
        TensorData bias{TensorType_INT32, {bias_size}, 0, 0, bias_scale};
        bias_ = AddInput(bias);
      }
    }

    output_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_CONV_2D, BuiltinOptions_Conv2DOptions,
                 CreateConv2DOptions(
                     builder_, padding, stride_width, stride_height, activation,
                     dilation_width_factor, dilation_height_factor)
                     .Union());

    resolver_ = std::make_unique<SingleOpResolver>(BuiltinOperator_CONV_2D,
                                                   registration);
    BuildInterpreter({GetShape(input_), GetShape(filter_), GetShape(bias_)},
                     num_threads, /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/true);
  }

 protected:
  int input_;
  int filter_;
  int bias_;
  int output_;
};

class ConvolutionOpModel : public BaseConvolutionOpModel<float> {
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
#ifndef TFLITE_WITH_RUY
    {"MultithreadedOptimized",
     ops::builtin::Register_CONVOLUTION_MULTITHREADED_OPT()},
#endif
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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 18, 2, 5,  // first batch, left
                                 18, 2, 5,  // first batch, right
                                 17, 4, 3,  // second batch, left
                                 37, 4, 3,  // second batch, right
                             }));
}

TEST_P(ConvolutionOpTest, SimpleTestFloat32SingleThreaded) {
  ConvolutionOpModel m(GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 1}},
                       {TensorType_FLOAT32, {3, 2, 2, 1}},
                       {TensorType_FLOAT32, {}}, 2, 2, Padding_VALID,
                       ActivationFunctionType_NONE, 1, 1, /*num_threads=*/1);

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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 18, 2, 5,  // first batch, left
                                 18, 2, 5,  // first batch, right
                                 17, 4, 3,  // second batch, left
                                 37, 4, 3,  // second batch, right
                             }));
}

TEST_P(ConvolutionOpTest, SimpleTestFloat32WithChannelsGrouped) {
  ConvolutionOpModel m(GetRegistration(), {TensorType_FLOAT32, {2, 2, 2, 2}},
                       {TensorType_FLOAT32, {2, 2, 2, 1}},  // 2 groups.
                       {TensorType_FLOAT32, {}});

  m.SetInput({
      // First batch
      1, 1, 1, 1,  // row = 1
      2, 2, 2, 2,  // row = 2
      // Second batch
      3, 3, 3, 3,  // row = 1
      4, 4, 4, 4   // row = 2
  });
  m.SetFilter({
      1, 1, 1, 1,      // first 2x2 filter
      -1, -1, -1, -1,  // second 2x2 filter
  });
  m.SetBias({1, 2});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 7, -4,    // first batch
                                 15, -12,  // second batch
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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({10, 34}));
}

TEST_P(ConvolutionOpTest, ActivationReluN1Test) {
  ConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 1}},
      {TensorType_FLOAT32, {3, 2, 2, 1}}, {TensorType_FLOAT32, {}},
      /*stride_width=*/2,
      /*stride_height=*/2,
      /*Padding=*/Padding_VALID,
      /*ActivationFunctionType=*/ActivationFunctionType_RELU_N1_TO_1);

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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 1, 1, 1,  // first batch, left
                                 1, 1, 1,  // first batch, right
                                 1, 1, 1,  // second batch, left
                                 1, 1, 1,  // second batch, right
                             }));
}

TEST_P(ConvolutionOpTest, ActivationRelu6Test) {
  ConvolutionOpModel m(GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 1}},
                       {TensorType_FLOAT32, {3, 2, 2, 1}},
                       {TensorType_FLOAT32, {}},
                       /*stride_width=*/2,
                       /*stride_height=*/2,
                       /*Padding=*/Padding_VALID,
                       /*ActivationFunctionType=*/ActivationFunctionType_RELU6);

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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 6, 2, 5,  // first batch, left
                                 6, 2, 5,  // first batch, right
                                 6, 4, 3,  // second batch, left
                                 6, 4, 3,  // second batch, right
                             }));
}

TEST_P(ConvolutionOpTest, StrideTest) {
  ConvolutionOpModel m(GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 1}},
                       {TensorType_FLOAT32, {3, 2, 2, 1}},
                       {TensorType_FLOAT32, {}},
                       /*stride_width=*/1,
                       /*stride_height=*/1,
                       /*Padding=*/Padding_VALID,
                       /*ActivationFunctionType=*/ActivationFunctionType_NONE);

  m.SetInput({
      // First batch
      1, 1, 1, 1,  // row = 1
      2, 2, 3, 2,  // row = 2
      // Second batch
      1, 2, 3, 4,  // row = 1
      1, 2, 4, 4,  // row = 2
  });
  m.SetFilter({
      1, 2, 3, 4,    // first 2x2 filter
      -1, 1, -1, 1,  // second 2x2 filter
      -1, -1, 1, 1,  // third 2x2 filter
  });
  m.SetBias({1, 2, 3});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 18, 2, 5,  // first batch, left
                                 22, 3, 6,  // first batch, middle
                                 21, 1, 6,  // first batch, right
                                 17, 4, 3,  // second batch, left
                                 31, 5, 4,  // second batch, middle
                                 40, 3, 4,  // second batch, right
                             }));
}

TEST_P(ConvolutionOpTest, PaddingTest) {
  ConvolutionOpModel m(GetRegistration(), {TensorType_FLOAT32, {1, 2, 4, 1}},
                       {TensorType_FLOAT32, {3, 2, 2, 1}},
                       {TensorType_FLOAT32, {}},
                       /*stride_width=*/1,
                       /*stride_height=*/1,
                       /*Padding=*/Padding_SAME,
                       /*ActivationFunctionType=*/ActivationFunctionType_NONE);

  m.SetInput({
      1, 1, 1, 1,  // row = 1
      2, 2, 3, 2,  // row = 2
  });
  m.SetFilter({
      1, 2, 3, 4,    // first 2x2 filter
      -1, 1, -1, 1,  // second 2x2 filter
      -1, -1, 1, 1,  // third 2x2 filter
  });
  m.SetBias({1, 2, 3});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 18, 2,  5,   // first row, left
                                 22, 3,  6,   //
                                 21, 1,  6,   //
                                 8,  -1, 4,   // first row, right
                                 7,  2,  -1,  // second row, left
                                 9,  3,  -2,  //
                                 8,  1,  -2,  //
                                 3,  0,  1,   // second row, right
                             }));
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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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

  // Add an additional test for the multi-threaded case, ensuring stability
  // under different thread counts.
  if (GetParam() == "MultithreadedOptimized") {
    for (int i = 1; i < 4; ++i) {
      m.SetNumThreads(i);
      ASSERT_EQ(m.Invoke(), kTfLiteOk);
      EXPECT_THAT(m.GetOutput(),
                  ElementsAreArray({105, 150, 183, 95, 235, 312, 357, 178, 187,
                                    234, 261, 121}));
    }
  }

  // Change the filter to ensure non-const filter behavior is correct.
  m.SetFilter({2, 4, 7, 2, 5, 8, 3, 6, 9});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({105, 150, 183, 95, 235, 313, 359,
                                               181, 187, 239, 267, 128}));
}

// TODO(b/157263074): Ideally using a const filter would be a parameterization
// of the test, so we ensure full test coverage with all the different
// types and backends.
TEST_P(ConvolutionOpTest, HandCalculatedFloat32WithConstFilter) {
  const int depth = 1;
  const int image_width = 4;
  const int image_height = 3;
  const int image_batch_count = 1;
  const int filter_size = 3;
  const int filter_count = 1;
  const int stride_width = 1;
  const int stride_height = 1;
  const Padding padding = Padding_SAME;
  // The filter matrix is:
  // | 1 | 4 | 7 |
  // | 2 | 5 | 8 |
  // | 3 | 6 | 9 |
  const std::initializer_list<float> filter_data = {1, 4, 7, 2, 5, 8, 3, 6, 9};
  ConvolutionOpModel m(
      GetRegistration(),
      {TensorType_FLOAT32,
       {image_batch_count, image_height, image_width, depth}},
      {TensorType_FLOAT32, {depth, filter_size, filter_size, filter_count}},
      {TensorType_FLOAT32, {}}, stride_width, stride_height, padding,
      ActivationFunctionType_NONE,
      /*dilation_width_factor=*/1,
      /*dilation_height_factor=*/1,
      /*num_threads=*/-1, filter_data);

  // The image matrix is:
  // |  1 |  2 |  3 |  4 |
  // |  5 |  6 |  7 |  8 |
  // |  9 | 10 | 11 | 12 |
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  // No bias for this test.
  m.SetBias({0});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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

  // Add an additional test for the multi-threaded case, ensuring stability
  // under different thread counts.
  if (GetParam() == "MultithreadedOptimized") {
    for (int i = 1; i < 4; ++i) {
      m.SetNumThreads(i);
      ASSERT_EQ(m.Invoke(), kTfLiteOk);
      EXPECT_THAT(m.GetOutput(),
                  ElementsAreArray({105, 150, 183, 95, 235, 312, 357, 178, 187,
                                    234, 261, 121}));
    }
  }
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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // Since the dilation rate is 3 this will reduce the size of the output from
  // 10x10 to 3x3 of all 5s. Specifically:
  // | 5 | 5 | 5 |
  // | 5 | 5 | 5 |
  // | 5 | 5 | 5 |
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 5, 5, 5, 5, 5, 5, 5, 5}));
}

class QuantizedConvolutionOpModel : public BaseConvolutionOpModel<uint8_t> {
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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

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

TEST_P(ConvolutionOpTest, SimpleTestQuantizedGrouped) {
  QuantizedConvolutionOpModel m(GetRegistration(),
                                {TensorType_UINT8, {2, 2, 2, 2}, -63.5, 64},
                                {TensorType_UINT8, {2, 2, 2, 1}, -63.5, 64},
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
  });
  m.SetBias({1, 2});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput(), ElementsAreArray(ArrayFloatNear(
                                            {
                                                18, 2,  // first batch
                                                23, 6   // second batch
                                            },
                                            1e-5)));
  // For good  measure, let's also verify the quantized values:
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 145, 129,  //
                                 150, 133,  //
                             }));
}

// Smoke test to ensure slightly irregular shapes safely partition into
// multi-threaded tasks. See also b/128996474.
TEST_P(ConvolutionOpTest, SimpleTestLargeIrregularQuantized) {
  QuantizedConvolutionOpModel m(
      GetRegistration(), {TensorType_UINT8, {1, 1, 1, 1024}, -127, 128},
      {TensorType_UINT8, {1001, 1, 1, 1024}, -127, 128},
      {TensorType_UINT8, {1, 1, 1, 1001}, -127, 128});
  m.QuantizeAndPopulate<uint8_t>(0 /*input*/, std::vector<float>(1024, 0));
  m.QuantizeAndPopulate<uint8_t>(1 /*filter*/,
                                 std::vector<float>(1001 * 1024, 0));
  m.QuantizeAndPopulate<int32_t>(2 /*bias*/, std::vector<float>(1001, 1));

  m.SetNumThreads(1);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  m.SetNumThreads(2);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  m.SetNumThreads(3);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(std::vector<uint8_t>(1001, 1)));
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
  ASSERT_EQ(quant_op.Invoke(), kTfLiteOk);

  float_op.SetInput(input);
  float_op.SetFilter(filter);
  float_op.SetBias(bias);
  ASSERT_EQ(float_op.Invoke(), kTfLiteOk);

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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
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
       -128,
       127},
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // Since the dilation rate is 3 this will reduce the size of the output from
  // 10x10 to 3x3 of all 5s. Specifically:
  // | 5 | 5 | 5 |
  // | 5 | 5 | 5 |
  // | 5 | 5 | 5 |
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray({5, 5, 5, 5, 5, 5, 5, 5, 5}));
}

class HybridConvolutionOpModel : public BaseConvolutionOpModel<int8_t> {
 public:
  using BaseConvolutionOpModel::BaseConvolutionOpModel;

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  void SetFilter(std::initializer_list<float> f) {
    SymmetricQuantizeAndPopulate(filter_, f);
  }

  void SetSignedFilter(std::initializer_list<float> f) {
    SignedSymmetricQuantizeAndPopulate(filter_, f);
  }

  void SetBias(std::initializer_list<float> data) {
    PopulateTensor(bias_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

TEST_P(ConvolutionOpTest, SimpleTestHybridUint8) {
  HybridConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 1}},
      {TensorType_UINT8, {3, 2, 2, 1}, 0, 0, 4.0 / 127.0, 0},
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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

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

TEST_P(ConvolutionOpTest, SimpleTestHybridUint8WithDilation) {
  const int stride_width = 1;
  const int stride_height = 1;
  const Padding padding = Padding_VALID;
  const int dilation_width_factor = 2;
  const int dilation_height_factor = 1;

  HybridConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {2, 1, 2, 1}},
      {TensorType_UINT8, {3, 1, 1, 1}, 0, 0, 4.0 / 127.0, 0},
      {TensorType_FLOAT32, {}}, stride_width, stride_height, padding,
      ActivationFunctionType_NONE, dilation_width_factor,
      dilation_height_factor);

  m.SetInput({
      1,
      1,
      1,
      1,
      2,
      2,
      2,
      2,
  });
  m.SetFilter({1, 2, 3});
  m.SetBias({1, 2, 3});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {2, 4, 6, 2, 4, 6, 2, 4, 6, 2, 4, 6}, 0.16)));
}

// This test's output is equivalent to the SimpleTestHybrid
// because we break each input into two channels, each with half of the value,
// while keeping the filters for each channel equivalent.
//
// 2 * (A/2) * B = A * B, where the left side is this new test.
TEST_P(ConvolutionOpTest, SimpleTestHybridWithChannelsUint8) {
  HybridConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 2}},
      {TensorType_UINT8, {3, 2, 2, 2}, 0, 0, 4.0 / 127.0, 0},
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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     18, 2, 5,  // first batch, left
                                     18, 2, 5,  // first batch, right
                                     17, 4, 3,  // second batch, left
                                     37, 4, 3,  // second batch, right
                                 },
                                 0.16)));
}

TEST_P(ConvolutionOpTest, SimpleTestHybridWithChannelsUint8Grouped) {
  HybridConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 2}},
      {TensorType_UINT8, {4, 2, 2, 2}, 0, 0, 4.0 / 127.0, 0},
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
      1,  1,  2,  2,  3,  3,  4,  4,  // first 2x2 filter
      -1, -1, 1,  1,  -1, -1, 1,  1,  // second 2x2 filter
      -1, -1, -1, -1, 1,  1,  1,  1,  // third 2x2 filter
      1,  1,  1,  1,  -1, -1, -1, -1  // forth 2x2 filter
  });
  m.SetBias({1, 2, 3, 4});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     18, 2, 5, 2,  // first batch, left
                                     18, 2, 5, 2,  // first batch, right
                                     17, 4, 3, 4,  // second batch, left
                                     37, 4, 3, 4,  // second batch, right
                                 },
                                 0.16)));
}

TEST_P(ConvolutionOpTest, PointwiseHybridUint8) {
  HybridConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 2}},
      {TensorType_UINT8, {1, 1, 1, 2}, 0, 0, 2.0 / 127.0, 0},
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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

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

TEST_P(ConvolutionOpTest, SimpleTestHybridInt8) {
  HybridConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 1}},
      {TensorType_INT8, {3, 2, 2, 1}, 0, 0, 4.0 / 127.0, 0},
      {TensorType_FLOAT32, {}});

  m.SetInput({
      // First batch
      1, 1, 1, 1,  // row = 1
      2, 2, 2, 2,  // row = 2
      // Second batch
      1, 2, 3, 4,  // row = 1
      1, 2, 3, 4,  // row = 2
  });
  m.SetSignedFilter({
      1, 2, 3, 4,    // first 2x2 filter
      -1, 1, -1, 1,  // second 2x2 filter
      -1, -1, 1, 1,  // third 2x2 filter
  });
  m.SetBias({1, 2, 3});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

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

TEST_P(ConvolutionOpTest, SimpleTestHybridInt8WithDilation) {
  const int stride_width = 1;
  const int stride_height = 1;
  const Padding padding = Padding_VALID;
  const int dilation_width_factor = 2;
  const int dilation_height_factor = 1;

  HybridConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 1}},
      {TensorType_INT8, {3, 2, 2, 1}, 0, 0, 4.0 / 127.0, 0},
      {TensorType_FLOAT32, {}}, stride_width, stride_height, padding,
      ActivationFunctionType_NONE, dilation_width_factor,
      dilation_height_factor);

  m.SetInput({
      // First batch
      1, 1, 1, 1,  // row = 1
      2, 2, 2, 2,  // row = 2
      // Second batch
      1, 2, 3, 4,  // row = 1
      1, 2, 3, 4,  // row = 2
  });
  m.SetSignedFilter({
      1, 2, 3, 4,    // first 2x2 filter
      -1, 1, -1, 1,  // second 2x2 filter
      -1, -1, 1, 1,  // third 2x2 filter
  });
  m.SetBias({1, 2, 3});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

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
  // 1 3 1 3
  // 2 4 2 4
  //
  // and multiplies it with the filter directly.
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     18, 2, 5,  // first batch, left
                                     18, 2, 5,  // first batch, right
                                     23, 6, 3,  // second batch, left
                                     33, 6, 3,  // second batch, right
                                 },
                                 0.16)));
}

TEST_P(ConvolutionOpTest, SimpleTestHybridInt8Big) {
  // A bigger variant of the simple hybrid test to ensure coverage on
  // optimized paths that are only enabled at larger matrix sizes.
  HybridConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 1}},
      {TensorType_INT8, {8, 2, 2, 1}, 0, 0, 4.0 / 127.0, 0},
      {TensorType_FLOAT32, {}});

  m.SetInput({
      // First batch
      1, 1, 1, 1,  // row = 1
      2, 2, 2, 2,  // row = 2
      // Second batch
      1, 2, 3, 4,  // row = 1
      1, 2, 3, 4,  // row = 2
  });
  m.SetSignedFilter({
      1,  2,  3,  4,   // first 2x2 filter
      -1, 1,  -1, 1,   // second 2x2 filter
      -1, -1, 1,  1,   // third 2x2 filter
      1,  1,  3,  3,   // fourth 2x2 filter
      -1, -1, 3,  3,   // fifth 2x2 filter
      4,  3,  2,  1,   // sixth 2x2 filter
      2,  1,  1,  2,   // seventh 2x2 filter
      1,  -1, 2,  -2,  // eighth 2x2 filter
  });
  m.SetBias({1, 2, 3, 4, 5, 6, 7, 8});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      18, 2, 5, 18, 15, 19, 16, 8,  // first batch, left
                      18, 2, 5, 18, 15, 19, 16, 8,  // first batch, right
                      17, 4, 3, 16, 11, 20, 16, 5,  // second batch, left
                      37, 4, 3, 32, 19, 40, 28, 5   // second batch, right
                  },
                  0.17)));
}

// This test's output is equivalent to the SimpleTestHybrid
// because we break each input into two channels, each with half of the value,
// while keeping the filters for each channel equivalent.
//
// 2 * (A/2) * B = A * B, where the left side is this new test.
TEST_P(ConvolutionOpTest, SimpleTestHybridWithChannelsInt8) {
  HybridConvolutionOpModel m(GetRegistration(),
                             {TensorType_FLOAT32, {2, 2, 4, 2}},
                             {TensorType_INT8,
                              {3, 2, 2, 2},
                              0,
                              0,
                              0,
                              0,
                              /*per_channel_quantization=*/true,
                              /*per_channel_quantization_scales=*/
                              {4.0 / 127.0, 4.0 / 127.0, 4.0 / 127.0},
                              /*per_channel_quantization_offsets=*/{0, 0, 0},
                              /*channel_index=*/0},
                             {TensorType_FLOAT32, {}});

  m.SetInput({
      // First batch
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  // row = 1
      1, 1, 1, 1, 1, 1, 1, 1,                  // row = 2
      // Second batch
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2,  // row = 1
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2   // row = 2
  });
  m.SetSignedFilter({
      1,  1,  2,  2,  3,  3,  4, 4,  // first 2x2 filter
      -1, -1, 1,  1,  -1, -1, 1, 1,  // second 2x2 filter
      -1, -1, -1, -1, 1,  1,  1, 1   // third 2x2 filter
  });
  m.SetBias({1, 2, 3});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     18, 2, 5,  // first batch, left
                                     18, 2, 5,  // first batch, right
                                     17, 4, 3,  // second batch, left
                                     37, 4, 3,  // second batch, right
                                 },
                                 0.16)));
}

TEST_P(ConvolutionOpTest, PointwiseHybridInt8) {
  HybridConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 2}},
      {TensorType_INT8, {1, 1, 1, 2}, 0, 0, 2.0 / 127.0, 0},
      {TensorType_FLOAT32, {}}, 1, 1);

  m.SetInput({
      // First batch
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  // row = 1
      1, 1, 1, 1, 1, 1, 1, 1,                  // row = 2
      // Second batch
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2,  // row = 1
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2   // row = 2
  });

  m.SetSignedFilter({
      1, 2,  // first filter
  });
  m.SetBias({0});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.5, 2.5, 1.5, 2.5, 1.5, 2.5, 1.5, 2.5, 3., 5.,  3.,
                      5.,  3.,  5.,  3.,  5.,  1.5, 2.5, 3.,  5., 4.5, 7.5,
                      6.,  10., 1.5, 2.5, 3.,  5.,  4.5, 7.5, 6., 10.,
                  },
                  0.0474)));
}

class PerChannelQuantizedConvolutionOpModel
    : public BaseConvolutionOpModel<int8_t> {
 public:
  using BaseConvolutionOpModel::BaseConvolutionOpModel;

  void SetInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<int8_t>(input_, data);
  }

  void SetFilter(std::initializer_list<float> data) {
    PerChannelSymmetricQuantizeAndPopulate(filter_, data);
  }

  void SetBias(std::initializer_list<float> data) {
    PerChannelQuantizeBias(bias_, data);
  }

  std::vector<int8_t> GetOutput() { return ExtractVector<int8_t>(output_); }
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<int8_t>(ExtractVector<int8_t>(output_), GetScale(output_),
                              GetZeroPoint(output_));
  }
};

#ifdef GTEST_HAS_DEATH_TEST
TEST_P(ConvolutionOpTest, AsymmetricPerchannelQuantization) {
  EXPECT_DEATH(PerChannelQuantizedConvolutionOpModel m(
                   GetRegistration(),
                   {TensorType_INT8, {1, 2, 3, 2}, -63.5, 64, 0.5, -1},
                   {TensorType_INT8,
                    // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
                    {2, 2, 2, 2},
                    0,
                    0,
                    0,
                    0,
                    /*per_channel_quantization=*/true,
                    /*per_channel_quantization_scales=*/{1},
                    /*per_channel_quantization_offsets=*/{1},
                    /*channel_index=*/0},
                   {TensorType_INT8, {}, -63.5, 64, 0.5, -1},
                   /*stride_width=*/1, /*stride_height=*/1),
               "Cannot allocate tensors");
}
#endif

TEST_P(ConvolutionOpTest, SimplePerTensorTest) {
  PerChannelQuantizedConvolutionOpModel m(
      GetRegistration(), {TensorType_INT8, {1, 2, 3, 2}, -63.5, 64, 0.5, -1},
      {TensorType_INT8,
       // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
       {2, 2, 2, 2},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{1},
       /*per_channel_quantization_offsets=*/{0},
       /*channel_index=*/0},
      {TensorType_INT8, {}, -63.5, 64, 0.5, -1},
      /*stride_width=*/1, /*stride_height=*/1);
  m.SetInput({
      // [1 * 2 * 3 * 2] as [batch, y, x, input_channel]
      3, 2,    // batch = 0, y = 0, x = 0
      1, -1,   // batch = 0, y = 0, x = 1
      -2, -3,  // batch = 0, y = 0, x = 2
      4, 3,    // batch = 0, y = 1, x = 0
      2, -2,   // batch = 0, y = 1, x = 1
      -3, -4,  // batch = 0, y = 1, x = 2
  });
  m.SetFilter(
      // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
      {
          1, 2,  // out channel = 0, y = 0, x = 0
          3, 4,  // out channel = 0, y = 0, x = 1
          3, 4,  // out channel = 0, y = 1, x = 0
          5, 6,  // out channel = 0, y = 1, x = 1
          7, 8,  // out channel = 1, y = 0, x = 0
          5, 6,  // out channel = 1, y = 0, x = 1
          3, 4,  // out channel = 1, y = 1, x = 0
          1, 2,  // out channel = 1, y = 1, x = 1
      });
  m.SetBias({3, -2});

  // Invoke and verify output.
  // output has dimension [1 * 1 * 2 * 2] as [batch, y, x, output_channel]
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({31, 56, -57, -44})));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({61, 111, -115, -89}));
}

TEST_P(ConvolutionOpTest, SimplePerChannelTest) {
  PerChannelQuantizedConvolutionOpModel m(
      GetRegistration(), {TensorType_INT8, {1, 2, 3, 2}, -63.5, 64, 0.5, -1},
      {TensorType_INT8,
       // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
       {2, 2, 2, 2},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{1, 2},
       /*per_channel_quantization_offsets=*/{0, 0},
       /*channel_index=*/0},
      {TensorType_INT8, {}, -63.5, 64, 0.5, -1},
      /*stride_width=*/1, /*stride_height=*/1);
  m.SetInput({
      // [1 * 2 * 3 * 2] as [batch, y, x, input_channel]
      3, 2,    // batch = 0, y = 0, x = 0
      1, -1,   // batch = 0, y = 0, x = 1
      -2, -3,  // batch = 0, y = 0, x = 2
      4, 3,    // batch = 0, y = 1, x = 0
      2, -2,   // batch = 0, y = 1, x = 1
      -3, -4,  // batch = 0, y = 1, x = 2
  });
  m.SetFilter(
      // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
      {
          1, 2,  // out channel = 0, y = 0, x = 0
          3, 4,  // out channel = 0, y = 0, x = 1
          3, 4,  // out channel = 0, y = 1, x = 0
          5, 6,  // out channel = 0, y = 1, x = 1
          7, 8,  // out channel = 1, y = 0, x = 0
          5, 6,  // out channel = 1, y = 0, x = 1
          3, 4,  // out channel = 1, y = 1, x = 0
          1, 2,  // out channel = 1, y = 1, x = 1
      });
  m.SetBias({3, -2});

  // Invoke and verify output.
  // output has dimension [1 * 1 * 2 * 2] as [batch, y, x, output_channel]
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({31, 64, -57, -46})));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({61, 127, -115, -93}));
}

class HybridPerChannelConvolutionOpModel
    : public BaseConvolutionOpModel<int8_t> {
 public:
  using BaseConvolutionOpModel::BaseConvolutionOpModel;

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  void SetSignedFilter(std::initializer_list<float> data) {
    PerChannelSymmetricQuantizeAndPopulate(filter_, data);
  }

  void SetBias(std::initializer_list<float> data) {
    PopulateTensor(bias_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

  template <typename T>
  std::vector<T> GetFilter() {
    return ExtractVector<T>(filter_);
  }
};

TEST_P(ConvolutionOpTest, SimpleTestHybridPerChannel) {
  float scale = 4.0 / 127.0;
  float scale2 = 1.0 / 127.0;
  HybridPerChannelConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 2}},
      {TensorType_INT8,
       {3, 2, 2, 2},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{scale, scale2, scale2},
       /*per_channel_quantization_offsets=*/{0, 0, 0},
       /*channel_index=*/0},
      {TensorType_FLOAT32, {}});

  m.SetInput({
      // First batch
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  // row = 1
      1, 1, 1, 1, 1, 1, 1, 1,                  // row = 2
      // Second batch
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2,  // row = 1
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2   // row = 2
  });
  m.SetSignedFilter({
      1,  1,  2,  2,  3,  3,  4, 4,  // first 2x2 filter
      -1, -1, 1,  1,  -1, -1, 1, 1,  // second 2x2 filter
      -1, -1, -1, -1, 1,  1,  1, 1   // third 2x2 filter
  });
  m.SetBias({1, 2, 3});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     18, 2, 5,  // first batch, left
                                     18, 2, 5,  // first batch, right
                                     17, 4, 3,  // second batch, left
                                     37, 4, 3,  // second batch, right
                                 },
                                 0.16)));
}

TEST_P(ConvolutionOpTest, SimpleTestHybridPerChannelGrouped) {
  float scale = 4.0 / 127.0;
  float scale2 = 1.0 / 127.0;
  HybridPerChannelConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 2}},
      {TensorType_INT8,
       {4, 2, 2, 2},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{scale, scale2, scale2, scale2},
       /*per_channel_quantization_offsets=*/{0, 0, 0, 0},
       /*channel_index=*/0},
      {TensorType_FLOAT32, {}});

  m.SetInput({
      // First batch
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  // row = 1
      1, 1, 1, 1, 1, 1, 1, 1,                  // row = 2
      // Second batch
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2,  // row = 1
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2   // row = 2
  });
  m.SetSignedFilter({
      1,  1,  2,  2,  3,  3,  4,  4,  // first 2x2 filter
      -1, -1, 1,  1,  -1, -1, 1,  1,  // second 2x2 filter
      -1, -1, -1, -1, 1,  1,  1,  1,  // third 2x2 filter
      1,  1,  1,  1,  -1, -1, -1, -1  // forth 2x2 filter
  });
  m.SetBias({1, 2, 3, 4});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     18, 2, 5, 2,  //
                                     18, 2, 5, 2,  //
                                     17, 4, 3, 4,  //
                                     37, 4, 3, 4,  //
                                 },
                                 0.16)));
}

TEST_P(ConvolutionOpTest, SimpleTestHybridWithPaddingPerChannel) {
  // Test uses the right zero points for padding if needed.
  const int stride_width = 1;
  const int stride_height = 2;
  float scale = 4.0 / 127.0;
  float scale2 = 1.0 / 127.0;
  HybridPerChannelConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 2}},
      {TensorType_INT8,
       {3, 2, 2, 2},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{scale, scale2, scale2},
       /*per_channel_quantization_offsets=*/{0, 0, 0},
       /*channel_index=*/0},
      {TensorType_FLOAT32, {}}, stride_width, stride_height, Padding_SAME);

  m.SetInput({
      // First batch
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  // row = 1
      1, 1, 1, 1, 1, 1, 1, 1,                  // row = 2
      // Second batch
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2,  // row = 1
      0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2   // row = 2
  });
  m.SetSignedFilter({
      1,  1,  2,  2,  3,  3,  4, 4,  // first 2x2 filter
      -1, -1, 1,  1,  -1, -1, 1, 1,  // second 2x2 filter
      -1, -1, -1, -1, 1,  1,  1, 1   // third 2x2 filter
  });
  m.SetBias({1, 2, 3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {18, 2, 5, 18, 2, 5, 18, 2, 5, 8,  -1, 4,
                                  17, 4, 3, 27, 4, 3, 37, 4, 3, 17, -6, 3},
                                 0.16)));
}

TEST_P(ConvolutionOpTest, SimpleTestHybridWithDilationPerChannel) {
  const int stride_width = 1;
  const int stride_height = 1;
  const Padding padding = Padding_VALID;
  const int dilation_width_factor = 2;
  const int dilation_height_factor = 1;

  float scale = 4.0 / 127.0;
  float scale2 = 1.0 / 127.0;
  HybridPerChannelConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {2, 2, 4, 1}},
      {TensorType_INT8,
       {3, 2, 2, 1},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{scale, scale2, scale2},
       /*per_channel_quantization_offsets=*/{0, 0, 0},
       /*channel_index=*/0},
      {TensorType_FLOAT32, {}}, stride_width, stride_height, padding,
      ActivationFunctionType_NONE, dilation_width_factor,
      dilation_height_factor);

  m.SetInput({
      // First batch
      1, 1, 1, 1,  // row = 1
      2, 2, 2, 2,  // row = 2
      // Second batch
      1, 2, 3, 4,  // row = 1
      1, 2, 3, 4,  // row = 2
  });
  m.SetSignedFilter({
      1, 2, 3, 4,    // first 2x2 filter
      -1, 1, -1, 1,  // second 2x2 filter
      -1, -1, 1, 1,  // third 2x2 filter
  });
  m.SetBias({1, 2, 3});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     18, 2, 5,  // first batch, left
                                     18, 2, 5,  // first batch, right
                                     23, 6, 3,  // second batch, left
                                     33, 6, 3,  // second batch, right
                                 },
                                 0.16)));
}

const auto kQuantizedKernelMap = new std::map<string, TfLiteRegistration*>({
    {"GenericOptimized", ops::builtin::Register_CONV_2D_UINT8()},
});

class QuantizedConvolutionOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kQuantizedKernelMap;
  }
};

// Simple test to ensure that the explicit quantized op registration behaves
// properly.
TEST_P(QuantizedConvolutionOpTest, SimpleTestExplicitQuantizedOp) {
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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

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

INSTANTIATE_TEST_SUITE_P(
    ConvolutionOpTest, ConvolutionOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kKernelMap)));

INSTANTIATE_TEST_SUITE_P(
    QuantizedConvolutionOpTest, QuantizedConvolutionOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kQuantizedKernelMap)));

}  // namespace
}  // namespace tflite
