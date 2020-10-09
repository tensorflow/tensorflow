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

#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/test_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {

namespace ops {
namespace builtin {

TfLiteRegistration* Register_DEPTHWISE_CONVOLUTION_REF();
TfLiteRegistration* Register_DEPTHWISE_CONVOLUTION_GENERIC_OPT();
TfLiteRegistration* Register_DEPTHWISE_CONVOLUTION_NEON_OPT();

}  // namespace builtin
}  // namespace ops

namespace {

using ::testing::ElementsAreArray;

class BaseDepthwiseConvolutionOpModel : public SingleOpModel {
 public:
  BaseDepthwiseConvolutionOpModel(
      TfLiteRegistration* registration, const TensorData& input,
      const TensorData& filter, const TensorData& output, Padding padding_type,
      int dilation_factor = 1, int stride_width = 1, int stride_height = 1,
      ActivationFunctionType fused_activation_function =
          ActivationFunctionType_NONE) {
    input_ = AddInput(input);
    filter_ = AddInput(filter);

    int bias_size = GetShape(filter_)[3];
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
        TensorData bias{TensorType_INT32,
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

    int input_depth = GetShape(input_)[3];
    int output_depth = GetShape(filter_)[3];
    int depth_mul = output_depth / input_depth;

    SetBuiltinOp(
        BuiltinOperator_DEPTHWISE_CONV_2D,
        BuiltinOptions_DepthwiseConv2DOptions,
        CreateDepthwiseConv2DOptions(
            builder_, padding_type, stride_width, stride_height, depth_mul,
            fused_activation_function, dilation_factor, dilation_factor)
            .Union());

    resolver_ = absl::make_unique<SingleOpResolver>(
        BuiltinOperator_DEPTHWISE_CONV_2D, registration);

    BuildInterpreter({GetShape(input_), GetShape(filter_), GetShape(bias_)});
  }

 protected:
  int input_;
  int filter_;
  int bias_;
  int output_;
};

class PerChannelHybridDepthwiseConvolutionOpModel
    : public BaseDepthwiseConvolutionOpModel {
 public:
  using BaseDepthwiseConvolutionOpModel::BaseDepthwiseConvolutionOpModel;

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  void SetFilter(std::initializer_list<float> data) {
    PerChannelSymmetricQuantizeAndPopulate(filter_, data);
  }

  void SetBias(std::initializer_list<float> data) {
    PopulateTensor(bias_, data);
  }

  void SetInput(const std::vector<float>& data) {
    PopulateTensor(input_, data);
  }

  void SetFilter(const std::vector<float>& data) {
    PerChannelSymmetricQuantizeAndPopulate(filter_, data);
  }

  void SetBias(const std::vector<float>& data) { PopulateTensor(bias_, data); }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

const auto kKernelMap = new std::map<string, TfLiteRegistration*>({
    {"Reference", ops::builtin::Register_DEPTHWISE_CONVOLUTION_REF()},
    {"GenericOptimized",
     ops::builtin::Register_DEPTHWISE_CONVOLUTION_GENERIC_OPT()},
    {"NeonOptimized", ops::builtin::Register_DEPTHWISE_CONVOLUTION_NEON_OPT()},
});

class PerChannelHybridDepthwiseConvolutionOptimizedOpTest
    : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kKernelMap;
  }
};

class PerChannelHybridDepthwiseConvolutionOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kKernelMap;
  }
};

void RandomTest(int b, int h, int w, int c, int fs, bool padding, int sw) {
  const float element_max = 1.0;
  const int input_size = b * h * w * c;
  const int filter_size = 1 * fs * fs * c;
  const int bias_size = c;
  std::vector<float> input_data(input_size);
  std::vector<float> filter_data(filter_size);
  std::vector<float> bias_data(bias_size);
  for (int i = 0; i < input_size; ++i) {
    input_data[i] = UniformRandomFloat(-element_max, element_max);
  }
  for (int i = 0; i < filter_size; ++i) {
    filter_data[i] = UniformRandomFloat(-element_max, element_max);
  }
  for (int i = 0; i < bias_size; ++i) {
    bias_data[i] = UniformRandomFloat(-element_max, element_max);
  }
  const TensorData input({TensorType_FLOAT32, {b, h, w, c}});
  const TensorData output({TensorType_FLOAT32, {}});
  std::vector<float> scales;
  std::vector<int64_t> offsets;
  for (int i = 0; i < c; i++) {
    scales.push_back(1.0 / 127.0);
    offsets.push_back(0.0);
  }
  const TensorData filter({TensorType_INT8,
                           {1, fs, fs, c},
                           0,
                           0,
                           0,
                           0,
                           /*per_channel_quantization=*/true,
                           scales,
                           offsets,
                           3});
  PerChannelHybridDepthwiseConvolutionOpModel hybrid_generic(
      ops::builtin::Register_DEPTHWISE_CONVOLUTION_REF(), input, filter, output,
      padding ? Padding_SAME : Padding_VALID,
      /* dilation_factor = */ 1,
      /* stride_width = */ sw,
      /* stride_height = */ sw);
  hybrid_generic.SetInput(input_data);
  hybrid_generic.SetFilter(filter_data);
  hybrid_generic.SetBias(bias_data);
  hybrid_generic.Invoke();
  std::vector<float> hybrid_generic_output = hybrid_generic.GetOutput();
  PerChannelHybridDepthwiseConvolutionOpModel hybrid_optimized(
      ops::builtin::Register_DEPTHWISE_CONVOLUTION_NEON_OPT(), input, filter,
      output, padding ? Padding_SAME : Padding_VALID,
      /* dilation_factor = */ 1,
      /* stride_width = */ sw,
      /* stride_height = */ sw);
  hybrid_optimized.SetInput(input_data);
  hybrid_optimized.SetFilter(filter_data);
  hybrid_optimized.SetBias(bias_data);
  hybrid_optimized.Invoke();
  std::vector<float> hybrid_optimized_output = hybrid_optimized.GetOutput();
  EXPECT_THAT(hybrid_generic_output,
              ElementsAreArray(ArrayFloatNear(hybrid_optimized_output)));
}

void RandomTest(int b, int w, int h, int c, int fs) {
  RandomTest(b, w, h, c, fs, false, 1);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest, AccuracyTest32) {
  RandomTest(1, 10, 10, 8, 3);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest, AccuracyTest64) {
  RandomTest(1, 112, 112, 64, 3);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest, AccuracyTest128) {
  RandomTest(1, 56, 56, 128, 3);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest, AccuracyTest256) {
  RandomTest(1, 28, 28, 256, 3);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest, AccuracyTest512) {
  RandomTest(1, 14, 14, 512, 3);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest, AccuracyTest1024) {
  RandomTest(1, 3, 3, 1024, 3);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       AccuracyPaddingTest32) {
  RandomTest(1, 112, 112, 32, 3, true, 1);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       AccuracyPaddingTest64) {
  RandomTest(1, 112, 112, 64, 3, true, 1);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       AccuracyPaddingTest128) {
  RandomTest(1, 56, 56, 128, 3, true, 1);
}
TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       AccuracyPaddingTest256) {
  RandomTest(1, 28, 28, 256, 3, true, 1);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       AccuracyPaddingTest512) {
  RandomTest(1, 14, 14, 512, 3, true, 1);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       AccuracyPaddingTest1024) {
  RandomTest(1, 3, 3, 1024, 3, true, 1);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       Accuracy2x2StrideTest32) {
  RandomTest(1, 112, 112, 32, 3, false, 2);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       Accuracy2x2StrideTest64) {
  RandomTest(1, 112, 112, 64, 3, false, 2);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       Accuracy2x2StrideTest128) {
  RandomTest(1, 56, 56, 128, 3, false, 2);
}
TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       Accuracy2x2StrideTest256) {
  RandomTest(1, 28, 28, 256, 3, false, 2);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       Accuracy2x2StrideTest512) {
  RandomTest(1, 14, 14, 512, 3, false, 2);
}

TEST_F(PerChannelHybridDepthwiseConvolutionOptimizedOpTest,
       Accuracy2x2StrideTest1024) {
  RandomTest(1, 3, 3, 1024, 3, false, 1);
}

TEST_P(PerChannelHybridDepthwiseConvolutionOpTest, SimpleTest) {
  PerChannelHybridDepthwiseConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {1, 2, 3, 2}},
      {TensorType_INT8,
       // [1 * 2 * 2 * 4] as [input_channel, y, x, output_channel]
       {1, 2, 2, 4},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{1, 2, 3, 4},
       /*per_channel_quantization_offsets=*/{0, 0, 0, 0},
       /*channel_index=*/3},
      {TensorType_FLOAT32, {}}, Padding_VALID);
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
      /*filter data*/
      {
          // [1 * 2 * 2 * 4] as [input_channel, y, x, output_channel]
          // depth multiplier = 2
          1, 2, 3, 4,  // y = 0, x = 0
          3, 4, 5, 6,  // y = 0, x = 1
          7, 8, 5, 6,  // y = 1, x = 0
          3, 4, 1, 2,  // y = 1, x = 1
      });
  m.SetBias({3, -2, 4, 6});

  // Invoke and verify output.
  // output has dimension [1 * 1 * 2 * 4] as [batch, y, x, output_channel]
  m.Invoke();
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          {42.9373, 47.9451, 22.0706, 22.0627, 3, -4.00784, -29.1294, -54.1098},
          0.16)));
}

TEST_P(PerChannelHybridDepthwiseConvolutionOpTest, Simple3x3FilterTest) {
  PerChannelHybridDepthwiseConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {1, 3, 3, 8}},
      {TensorType_INT8,
       // [1 * 3 * 3 * 8] as [input_channel, y, x, output_channel]
       {1, 3, 3, 8},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/
       {1, 2, 3, 4, 4, 3, 2, 1},
       /*per_channel_quantization_offsets=*/{0, 0, 0, 0, 0, 0, 0, 0},
       /*channel_index=*/3},
      {TensorType_FLOAT32, {}}, Padding_VALID);
  m.SetInput({// array of 9 x 8 => [1, 3, 3, 8]
              1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
              0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
              1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
              0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0});
  m.SetFilter(
      /*filter data*/
      {// array of 9 x 8 => [1, 3, 3, 8]
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8});
  m.SetBias({0, 0, 0, 0, 0, 0, 0, 0});

  // Invoke and verify output.
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {9, 18, 0, 0, 36, 54, 0, 0}, 0.16)));
}

TEST_P(PerChannelHybridDepthwiseConvolutionOpTest,
       Simple3x3FilterPaddingSameTest) {
  PerChannelHybridDepthwiseConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {1, 3, 3, 8}},
      {TensorType_INT8,
       // [1 * 3 * 3 * 8] as [input_channel, y, x, output_channel]
       {1, 3, 3, 8},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/
       {1, 2, 3, 4, 4, 3, 2, 1},
       /*per_channel_quantization_offsets=*/{0, 0, 0, 0, 0, 0, 0, 0},
       /*channel_index=*/3},
      {TensorType_FLOAT32, {}}, Padding_SAME);
  m.SetInput({// array of 9 x 8 => [1, 3, 3, 8]
              1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
              0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
              1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
              0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0});
  m.SetFilter(
      /*filter data*/
      {// array of 9 x 8 => [1, 3, 3, 8]
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8});
  m.SetBias({0, 0, 0, 0, 0, 0, 0, 0});

  // Invoke and verify output.
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      // array of 9 x 8 => [1, 3, 3, 8]
                      4,  8,  0, 0,  16, 24, 0,  0,  6,  12, 0,  0,  24, 36, 0,
                      0,  4,  8, 0,  0,  16, 24, 0,  0,  6,  12, 0,  0,  24, 36,
                      0,  0,  9, 18, 0,  0,  36, 54, 0,  0,  6,  12, 0,  0,  24,
                      36, 0,  0, 4,  8,  0,  0,  16, 24, 0,  0,  6,  12, 0,  0,
                      24, 36, 0, 0,  4,  8,  0,  0,  16, 24, 0,  0,
                  },
                  0.16)));
}

INSTANTIATE_TEST_SUITE_P(
    PerChannelHybridDepthwiseConvolutionOpTest,
    PerChannelHybridDepthwiseConvolutionOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kKernelMap)));

}  // namespace
}  // namespace tflite
