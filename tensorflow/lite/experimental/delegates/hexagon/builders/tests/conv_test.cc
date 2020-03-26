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
#include "tensorflow/lite/experimental/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"
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
    // per tensor quantization.
    auto bias_scale = GetScale(input_) * GetScale(filter_);
    TensorData bias{TensorType_INT32, {bias_size}, 0, 0, bias_scale};
    bias_ = AddInput(bias);

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
    QuantizeAndPopulate<int32_t>(bias_, data);
  }

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }

 protected:
  int input_;
  int filter_;
  int bias_;
  int output_;
};

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
  EXPECT_THAT(m.GetDequantizedOutput(),
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
  m.Invoke();
  auto reference_output = m.GetDequantizedOutput();

  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(reference_output, 1e-5)));
}

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

  EXPECT_THAT(m.GetDequantizedOutput(),
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

  EXPECT_THAT(m.GetDequantizedOutput(), ElementsAreArray(ArrayFloatNear(
                                            {
                                                6, 2, 5,  // first batch, left
                                                6, 2, 5,  // first batch, right
                                                6, 4, 3,  // second batch, left
                                                6, 4, 3,  // second batch, right
                                            },
                                            1e-5)));
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
  m.Invoke();
  auto reference_output = m.GetDequantizedOutput();

  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput(),
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
  m.Invoke();
  auto reference_output = m.GetDequantizedOutput();

  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(reference_output, 1e-5)));
}

}  // namespace tflite
