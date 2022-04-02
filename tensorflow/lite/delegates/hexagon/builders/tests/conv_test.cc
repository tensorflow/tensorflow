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
#include <numeric>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
using testing::ElementsAreArray;

int NumElements(const std::vector<int>& dims) {
  return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
}

class QuantizedConvolutionOpModel : public SingleOpModelWithHexagon {
 public:
  QuantizedConvolutionOpModel(BuiltinOperator type, const TensorData& input,
                              const TensorData& filter,
                              const TensorData& output, Padding padding_type,
                              int dilation_factor = 1, int stride_length = 1,
                              ActivationFunctionType fused_activation_function =
                                  ActivationFunctionType_NONE) {
    input_ = AddInput(input);

    filter_ = AddInput(filter);

    int bias_size = GetShape(filter_)[0];
    if (type == BuiltinOperator_DEPTHWISE_CONV_2D) {
      bias_size = GetShape(filter_)[3];
    }
    if (filter.per_channel_quantization) {
      // per channel quantization.
      std::vector<float> bias_scale(
          filter.per_channel_quantization_scales.size());
      std::vector<int64_t> bias_zero_points(
          filter.per_channel_quantization_scales.size());
      for (size_t i = 0; i < filter.per_channel_quantization_scales.size();
           ++i) {
        bias_scale[i] = input.scale * filter.per_channel_quantization_scales[i];
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

    output_ = AddOutput(output);

    if (type == BuiltinOperator_DEPTHWISE_CONV_2D) {
      int input_depth = GetShape(input_)[3];
      int output_depth = GetShape(filter_)[3];
      int depth_mul = output_depth / input_depth;
      SetBuiltinOp(
          BuiltinOperator_DEPTHWISE_CONV_2D,
          BuiltinOptions_DepthwiseConv2DOptions,
          CreateDepthwiseConv2DOptions(
              builder_, padding_type, stride_length, stride_length, depth_mul,
              fused_activation_function, dilation_factor, dilation_factor)
              .Union());
    } else {
      SetBuiltinOp(BuiltinOperator_CONV_2D, BuiltinOptions_Conv2DOptions,
                   CreateConv2DOptions(builder_, padding_type, stride_length,
                                       stride_length, fused_activation_function,
                                       dilation_factor, dilation_factor)
                       .Union());
    }

    BuildInterpreter({GetShape(input_), GetShape(filter_), GetShape(bias_)});

    // Filter needs to be a constant.
    // We don't use AddConstInput to allow setting filter values later.
    auto* filter_tensor = interpreter_->tensor(filter_);
    filter_tensor->allocation_type = kTfLiteMmapRo;
  }

  void SetInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<uint8_t>(input_, data);
  }

  void SetFilter(std::initializer_list<float> data) {
    QuantizeAndPopulate<uint8_t>(filter_, data);
  }

  void SetBias(std::initializer_list<float> data) {
    QuantizeAndPopulate<int>(bias_, data);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

  void SetInt8Input(std::initializer_list<float> data) {
    QuantizeAndPopulate<int8_t>(input_, data);
  }

  void SetPerChannelQuantizedFilter(std::initializer_list<float> data) {
    PerChannelSymmetricQuantizeAndPopulate(filter_, data);
  }

  void SetPerChannelQuantizedBias(std::initializer_list<float> data) {
    PerChannelQuantizeBias(bias_, data);
  }

 protected:
  int input_;
  int filter_;
  int bias_;
  int output_;
};

// CONVOLUTION TESTS

TEST(QuantizedConvolutionOpModel, SimpleConvTestNoActivation) {
  QuantizedConvolutionOpModel m(
      BuiltinOperator_CONV_2D, {TensorType_UINT8, {2, 2, 4, 1}, -63.5, 64},
      {TensorType_UINT8, {3, 2, 2, 1}, -63.5, 64},
      {TensorType_UINT8, {}, -127, 128}, Padding_VALID, /**dilation_factor**/ 1,
      /**stride**/ 2);
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

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      18, 2, 5,  // first batch, left
                      18, 2, 5,  // first batch, right
                      17, 4, 3,  // second batch, left
                      37, 4, 3,  // second batch, right
                  },
                  1e-5)));
}

TEST(QuantizedConvolutionOpModel, SimpleConvTestReLU6Activation) {
  QuantizedConvolutionOpModel m(
      BuiltinOperator_CONV_2D, {TensorType_UINT8, {2, 2, 4, 1}, -63.5, 64},
      {TensorType_UINT8, {3, 2, 2, 1}, -63.5, 64},
      {TensorType_UINT8, {}, -127, 128}, Padding_VALID, /**dilation_factor**/ 1,
      /**stride**/ 2, ActivationFunctionType_RELU6);
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

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      6, 2, 5,  // first batch, left
                      6, 2, 5,  // first batch, right
                      6, 4, 3,  // second batch, left
                      6, 4, 3,  // second batch, right
                  },
                  1e-5)));
}

// Same as above, but the output min/max matches the RELU bounds.
// Therefore, a Requantize node will not get added after Supernode.
TEST(QuantizedConvolutionOpModel,
     SimpleConvTestReLU6Activation_NoRequantizeRequired) {
  QuantizedConvolutionOpModel m(
      BuiltinOperator_CONV_2D, {TensorType_UINT8, {2, 2, 4, 1}, -63.5, 64},
      {TensorType_UINT8, {3, 2, 2, 1}, -63.5, 64}, {TensorType_UINT8, {}, 0, 6},
      Padding_VALID, /**dilation_factor**/ 1,
      /**stride**/ 2, ActivationFunctionType_RELU6);
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

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      6, 2, 5,  // first batch, left
                      6, 2, 5,  // first batch, right
                      6, 4, 3,  // second batch, left
                      6, 4, 3,  // second batch, right
                  },
                  2e-2)));
}

TEST(QuantizedConvolutionOpModel, SimplePerTensor_Int8) {
  QuantizedConvolutionOpModel m(
      BuiltinOperator_CONV_2D,
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
       /*per_channel_quantization_offsets=*/{0},
       /*channel_index=*/0},
      {TensorType_INT8, {}, -63.5, 64, 0.5, -1}, Padding_VALID);
  m.SetInt8Input({
      // [1 * 2 * 3 * 2] as [batch, y, x, input_channel]
      3, 2,    // batch = 0, y = 0, x = 0
      1, -1,   // batch = 0, y = 0, x = 1
      -2, -3,  // batch = 0, y = 0, x = 2
      4, 3,    // batch = 0, y = 1, x = 0
      2, -2,   // batch = 0, y = 1, x = 1
      -3, -4,  // batch = 0, y = 1, x = 2
  });
  m.SetPerChannelQuantizedFilter(
      // [2 * 2 * 2 * 2] as [output_channel, y, x,input_channel]
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
  m.SetPerChannelQuantizedBias({3, -2});

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({31, 56, -57, -44}, 1e-5)));
}

TEST(QuantizedConvolutionOpModel, SimplePerChannel_Int8) {
  QuantizedConvolutionOpModel m(
      BuiltinOperator_CONV_2D,
      {TensorType_INT8, {1, 2, 3, 2}, -63.5, 64, 0.5, -1},
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
      {TensorType_INT8, {}, -63.5, 64, 0.5, -1}, Padding_VALID);
  m.SetInt8Input({
      // [1 * 2 * 3 * 2] as [batch, y, x, input_channel]
      3, 2,    // batch = 0, y = 0, x = 0
      1, -1,   // batch = 0, y = 0, x = 1
      -2, -3,  // batch = 0, y = 0, x = 2
      4, 3,    // batch = 0, y = 1, x = 0
      2, -2,   // batch = 0, y = 1, x = 1
      -3, -4,  // batch = 0, y = 1, x = 2
  });
  m.SetPerChannelQuantizedFilter(
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
  m.SetPerChannelQuantizedBias({3, -2});

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({31, 64, -57, -46}, 0.6f)));
}

// DEPTHWISE CONVOLUTION TESTS

TEST(QuantizedConvolutionOpModel, SimpleDilatedDepthwiseConvTestPaddingValid) {
  const int depth = 1;
  const int image_width = 9;
  const int image_height = 9;
  const int image_batch_count = 1;
  const int filter_size = 3;
  const int filter_count = 1;
  const int dilation_factor = 3;
  QuantizedConvolutionOpModel m(
      BuiltinOperator_DEPTHWISE_CONV_2D,
      {TensorType_UINT8,
       {image_batch_count, image_height, image_width, depth},
       0,
       255},
      {TensorType_UINT8,
       {depth, filter_size, filter_size, filter_count},
       0,
       255},
      {TensorType_UINT8, {}, 0, 255}, Padding_VALID, dilation_factor);

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
  m.ApplyDelegateAndInvoke();

  // Since the dilation rate is 3 this will reduce the size of the output from
  // 10x10 to 3x3 of all 5s. Specifically:
  // | 5 | 5 | 5 |
  // | 5 | 5 | 5 |
  // | 5 | 5 | 5 |
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray({5, 5, 5, 5, 5, 5, 5, 5, 5}));
}

TEST(QuantizedConvolutionOpModel, DepthwiseConv5x5) {
  QuantizedConvolutionOpModel m(BuiltinOperator_DEPTHWISE_CONV_2D,
                                {TensorType_UINT8, {1, 6, 6, 2}, -63.5, 64},
                                {TensorType_UINT8, {1, 5, 5, 2}, -63.5, 64},
                                {TensorType_UINT8, {}, -127, 128},
                                Padding_VALID);
  // clang-format off
  m.SetInput({0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0});
  // clang-format on
  m.SetFilter({1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2,
               3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4,
               5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5});
  m.SetBias({1, 2});

  // Reference output.
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();

  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output, 1e-5)));
}

// Depthwise Conv with multiplier > 1 but input depth==1 should resolve into a
// Conv op.
TEST(QuantizedConvolutionOpModel, DepthwiseConvWithMultiplier_InputDepth1) {
  QuantizedConvolutionOpModel m(BuiltinOperator_DEPTHWISE_CONV_2D,
                                {TensorType_UINT8, {1, 6, 6, 1}, -63.5, 64},
                                {TensorType_UINT8, {1, 5, 5, 3}, -63.5, 64},
                                {TensorType_UINT8, {}, -127, 128},
                                Padding_VALID);
  // clang-format off
  m.SetInput({0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0});
  m.SetFilter({1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               6, 7, 8, 9, 10,
               6, 7, 8, 9, 10,
               6, 7, 8, 9, 10,
               6, 7, 8, 9, 10,
               6, 7, 8, 9, 10,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5});
  // clang-format on
  m.SetBias({1, 2, 3});

  // Reference output.
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();

  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output, 1e-5)));
}

// Depthwise Conv with multiplier > 1 but input depth==1 should resolve into a
// Conv op.
TEST(QuantizedConvolutionOpModel,
     DepthwiseConvWithMultiplier_InputDepth1_RELU) {
  QuantizedConvolutionOpModel m(BuiltinOperator_DEPTHWISE_CONV_2D,
                                {TensorType_UINT8, {1, 6, 6, 1}, -63.5, 64},
                                {TensorType_UINT8, {1, 5, 5, 3}, -63.5, 64},
                                {TensorType_UINT8, {}, -127, 128},
                                Padding_VALID, /**dilation_factor**/ 1,
                                /**stride**/ 2, ActivationFunctionType_RELU6);
  // clang-format off
  m.SetInput({0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0});
  m.SetFilter({1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               6, 7, 8, 9, 10,
               6, 7, 8, 9, 10,
               6, 7, 8, 9, 10,
               6, 7, 8, 9, 10,
               6, 7, 8, 9, 10,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5,
               1, 2, 3, 4, 5});
  // clang-format on
  m.SetBias({1, 2, 3});

  // Reference output.
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();

  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output, 1e-5)));
}

TEST(QuantizedConvolutionOpModel, DepthwiseConvSimplePerTensor_Int8) {
  QuantizedConvolutionOpModel m(
      BuiltinOperator_DEPTHWISE_CONV_2D,
      {TensorType_INT8, {1, 2, 3, 1}, -63.5, 64, 0.5, -1},
      {TensorType_INT8,
       // [1 * 2 * 2 * 4] as [input_channel, y, x, output_channel]
       {1, 2, 2, 4},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{1},
       /*per_channel_quantization_offsets=*/{0},
       /*channel_index=*/3},
      {TensorType_INT8, {}, -63.5, 64, 0.5, -1}, Padding_VALID);
  m.SetInt8Input({
      // [1 * 2 * 3 * 1] as [batch, y, x, input_channel]
      3,   // batch = 0, y = 0, x = 0
      1,   // batch = 0, y = 0, x = 1
      -2,  // batch = 0, y = 0, x = 2
      4,   // batch = 0, y = 1, x = 0
      2,   // batch = 0, y = 1, x = 1
      -3,  // batch = 0, y = 1, x = 2
  });
  m.SetPerChannelQuantizedFilter({
      // [1 * 2 * 2 * 4] as [input_channel, y, x, output_channel]
      // depth multiplier = 2
      1, 2, 3, 4,  // y = 0, x = 0
      3, 4, 5, 6,  // y = 0, x = 1
      7, 8, 5, 6,  // y = 1, x = 0
      3, 4, 1, 2,  // y = 1, x = 1
  });
  m.SetPerChannelQuantizedBias({3, -2, 4, 6});

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(
      m.GetDequantizedOutput<int8_t>(),
      ElementsAreArray(ArrayFloatNear({43, 48, 40, 52, 3, -4, 4, 4}, 0.6f)));
}

TEST(QuantizedConvolutionOpModel, DepthwiseConvSimplePerTensor_Int8_RELU1) {
  QuantizedConvolutionOpModel m(
      BuiltinOperator_DEPTHWISE_CONV_2D,
      {TensorType_INT8, {1, 2, 3, 1}, -63.5, 64, 0.5, -1},
      {TensorType_INT8,
       // [1 * 2 * 2 * 4] as [input_channel, y, x, output_channel]
       {1, 2, 2, 4},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{0.1, 2, 3, 0.4},
       /*per_channel_quantization_offsets=*/{0, 0, 0, 0},
       /*channel_index=*/3},
      {TensorType_INT8, {}, -63.5, 64, 0.5, -1}, Padding_VALID,
      /**dilation_factor**/ 1,
      /**stride**/ 1, ActivationFunctionType_RELU_N1_TO_1);
  m.SetInt8Input({
      // [1 * 2 * 3 * 1] as [batch, y, x, input_channel]
      3,   // batch = 0, y = 0, x = 0
      1,   // batch = 0, y = 0, x = 1
      -2,  // batch = 0, y = 0, x = 2
      4,   // batch = 0, y = 1, x = 0
      2,   // batch = 0, y = 1, x = 1
      -4,  // batch = 0, y = 1, x = 2
  });
  m.SetPerChannelQuantizedFilter({
      // [1 * 2 * 2 * 4] as [input_channel, y, x, output_channel]
      // depth multiplier = 2
      1, 2, 3, 4,  // y = 0, x = 0
      3, 4, 5, 6,  // y = 0, x = 1
      7, 8, 5, 6,  // y = 1, x = 0
      3, 4, 1, 2,  // y = 1, x = 1
  });
  m.SetPerChannelQuantizedBias({3, -2, 4, 6});

  // Reference output.
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<int8_t>();

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output, 1e-2)));
}

TEST(QuantizedConvolutionOpModel, DepthwiseConvSimplePerAxis_Int8) {
  QuantizedConvolutionOpModel m(
      BuiltinOperator_DEPTHWISE_CONV_2D,
      {TensorType_INT8, {1, 2, 3, 1}, -63.5, 64, 0.5, -1},
      {TensorType_INT8,
       // [1 * 2 * 2 * 4] as [input_channel, y, x, output_channel]
       {1, 2, 2, 4},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{0.1, 2, 3, 0.4},
       /*per_channel_quantization_offsets=*/{0, 0, 0, 0},
       /*channel_index=*/3},
      {TensorType_INT8, {}, -63.5, 64, 0.5, -1}, Padding_VALID);
  m.SetInt8Input({
      // [1 * 2 * 3 * 1] as [batch, y, x, input_channel]
      3,   // batch = 0, y = 0, x = 0
      1,   // batch = 0, y = 0, x = 1
      -2,  // batch = 0, y = 0, x = 2
      4,   // batch = 0, y = 1, x = 0
      2,   // batch = 0, y = 1, x = 1
      -4,  // batch = 0, y = 1, x = 2
  });
  m.SetPerChannelQuantizedFilter({
      // [1 * 2 * 2 * 4] as [input_channel, y, x, output_channel]
      // depth multiplier = 2
      1, 2, 3, 4,  // y = 0, x = 0
      3, 4, 5, 6,  // y = 0, x = 1
      7, 8, 5, 6,  // y = 1, x = 0
      3, 4, 1, 2,  // y = 1, x = 1
  });
  m.SetPerChannelQuantizedBias({3, -2, 4, 6});

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(
      m.GetDequantizedOutput<int8_t>(),
      ElementsAreArray(ArrayFloatNear({43, 48, 42, 52, 0, -8, 6, 2}, 0.6f)));
}

TEST(QuantizedConvolutionOpModel, DepthwiseConvPerChannel_3x3Filter) {
  QuantizedConvolutionOpModel m(
      BuiltinOperator_DEPTHWISE_CONV_2D,
      {TensorType_INT8, {1, 3, 3, 8}, -63.5, 64, 0.5, -1},
      {TensorType_INT8,
       // [1 * 3 * 3 * 8] as [input_channel, y, x, output_channel]
       {1, 3, 3, 8},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/
       {0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1},
       /*per_channel_quantization_offsets=*/{0, 0, 0, 0, 0, 0, 0, 0},
       /*channel_index=*/3},
      {TensorType_INT8, {}, -63.5, 64, 0.5, -1}, Padding_VALID);
  m.SetInt8Input({// array of 9 x 8 => [1, 3, 3, 8]
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
                  0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
                  0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0});
  m.SetPerChannelQuantizedFilter(
      {// array of 9 x 8 => [1, 3, 3, 8]
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8});
  m.SetPerChannelQuantizedBias({0, 0, 0, 0, 0, 0, 0, 0});

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(
      m.GetDequantizedOutput<int8_t>(),
      ElementsAreArray(ArrayFloatNear({9, 18, 0, 0, 47, 54, 0, 0}, 0.6f)));
}

TEST(QuantizedConvolutionOpModel,
     DepthwiseConvPerChannel_3x3FilterPaddingSame) {
  QuantizedConvolutionOpModel m(
      BuiltinOperator_DEPTHWISE_CONV_2D,
      {TensorType_INT8, {1, 3, 3, 8}, -63.5, 64, 0.5, -1},
      {TensorType_INT8,
       // [1 * 3 * 3 * 8] as [input_channel, y, x, output_channel]
       {1, 3, 3, 8},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/
       {0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1},
       /*per_channel_quantization_offsets=*/{0, 0, 0, 0, 0, 0, 0, 0},
       /*channel_index=*/3},
      {TensorType_INT8, {}, -63.5, 64, 0.5, -1}, Padding_SAME);
  m.SetInt8Input({// array of 9 x 8 => [1, 3, 3, 8]
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
                  0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
                  0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0});
  m.SetPerChannelQuantizedFilter(
      {// array of 9 x 8 => [1, 3, 3, 8]
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
       1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8});
  m.SetPerChannelQuantizedBias({0, 0, 0, 0, 0, 0, 0, 0});

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      // array of 9 x 8 => [1, 3, 3, 8]
                      4, 8,  0, 0, 21, 24, 0, 0, 6, 12, 0, 0, 31.5, 36, 0, 0,
                      4, 8,  0, 0, 21, 24, 0, 0, 6, 12, 0, 0, 31.5, 36, 0, 0,
                      9, 18, 0, 0, 47, 54, 0, 0, 6, 12, 0, 0, 31.5, 36, 0, 0,
                      4, 8,  0, 0, 21, 24, 0, 0, 6, 12, 0, 0, 31.5, 36, 0, 0,
                      4, 8,  0, 0, 21, 24, 0, 0,
                  },
                  0.6f)));
}

}  // namespace tflite
