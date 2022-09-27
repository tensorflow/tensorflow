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
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/internal/test_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {

namespace ops {
namespace builtin {

TfLiteRegistration* Register_DEPTHWISE_CONV_2D_UINT8();
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
        tflite::TensorType bias_type = TensorType_INT32;
        if (input.type == TensorType_INT16) {
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
    // The CPU kernel now ignores `depthwise_multiplier`. However delegates
    // like NNAPI still relies on the attribute.
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

    resolver_ = std::make_unique<SingleOpResolver>(
        BuiltinOperator_DEPTHWISE_CONV_2D, registration);

    BuildInterpreter({GetShape(input_), GetShape(filter_), GetShape(bias_)});
  }

 protected:
  int input_;
  int filter_;
  int bias_;
  int output_;
};

class DepthwiseConvolutionOpModel : public BaseDepthwiseConvolutionOpModel {
 public:
  using BaseDepthwiseConvolutionOpModel::BaseDepthwiseConvolutionOpModel;

  void SetFilter(std::initializer_list<float> f) { PopulateTensor(filter_, f); }

  void SetBias(std::initializer_list<float> f) { PopulateTensor(bias_, f); }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

const auto kKernelMap = new std::map<string, TfLiteRegistration*>({
    {"Reference", ops::builtin::Register_DEPTHWISE_CONVOLUTION_REF()},
    {"GenericOptimized",
     ops::builtin::Register_DEPTHWISE_CONVOLUTION_GENERIC_OPT()},
    {"NeonOptimized", ops::builtin::Register_DEPTHWISE_CONVOLUTION_NEON_OPT()},
});

class DepthwiseConvolutionOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kKernelMap;
  }
};

TEST_P(DepthwiseConvolutionOpTest, ActivationReluTest) {
  DepthwiseConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {1, 3, 2, 2}},
      {TensorType_FLOAT32, {1, 2, 2, 4}}, {TensorType_FLOAT32, {}},
      Padding_VALID,
      /*dilation_factor*/ 1,
      /*stride_width*/ 1,
      /*stride_height*/ 1,
      /*ActivationFunctionType*/ ActivationFunctionType_RELU);

  m.SetInput({
      1, 2, 7, 8,    // column 1
      3, 4, 9, 10,   // column 2
      5, 6, 11, 12,  // column 3
  });
  m.SetFilter({
      1, 2, 3, 4,        //
      -9, 10, -11, 12,   //
      5, 6, 7, 8,        //
      13, -14, 15, -16,  //
  });
  m.SetBias({1, 2, 3, 4});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 71, 0, 99, 0,   //
                                 91, 0, 127, 0,  //
                             }));
}

TEST_P(DepthwiseConvolutionOpTest, ActivationReluN1Test) {
  DepthwiseConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {1, 3, 2, 2}},
      {TensorType_FLOAT32, {1, 2, 2, 4}}, {TensorType_FLOAT32, {}},
      Padding_VALID,
      /*dilation_factor*/ 1,
      /*stride_width*/ 1,
      /*stride_height*/ 1,
      /*ActivationFunctionType*/ ActivationFunctionType_RELU_N1_TO_1);

  m.SetInput({
      1, 2, 7, 8,    // column 1
      3, 4, 9, 10,   // column 2
      5, 6, 11, 12,  // column 3
  });
  m.SetFilter({
      1, 2, 3, 4,        //
      -9, 10, -11, 12,   //
      5, 6, 7, 8,        //
      13, -14, 15, -16,  //
  });
  m.SetBias({1, 2, 3, 4});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 1, -1, 1, -1,  //
                                 1, -1, 1, -1,  //
                             }));
}

TEST_P(DepthwiseConvolutionOpTest, ActivationRelu6Test) {
  DepthwiseConvolutionOpModel m(
      GetRegistration(), {TensorType_FLOAT32, {1, 3, 2, 2}},
      {TensorType_FLOAT32, {1, 2, 2, 4}}, {TensorType_FLOAT32, {}},
      Padding_VALID,
      /*dilation_factor*/ 1,
      /*stride_width*/ 1,
      /*stride_height*/ 1,
      /*ActivationFunctionType*/ ActivationFunctionType_RELU6);

  m.SetInput({
      1, 2, 7, 8,    // column 1
      3, 4, 9, 10,   // column 2
      5, 6, 11, 12,  // column 3
  });
  m.SetFilter({
      1, 2, 3, 4,        //
      -9, 10, -11, 12,   //
      5, 6, 7, 8,        //
      13, -14, 15, -16,  //
  });
  m.SetBias({1, 2, 3, 4});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 6, 0, 6, 0,  //
                                 6, 0, 6, 0,  //
                             }));
}

void StrideTest(TfLiteRegistration* registration, int num_thread) {
  DepthwiseConvolutionOpModel m(
      registration, {TensorType_FLOAT32, {1, 3, 2, 2}},
      {TensorType_FLOAT32, {1, 2, 2, 4}}, {TensorType_FLOAT32, {}},
      Padding_VALID,
      /*dilation_factor*/ 1,
      /*stride_width*/ 2,
      /*stride_height*/ 2,
      /*ActivationFunctionType*/ ActivationFunctionType_NONE);

  m.SetInput({
      1, 2, 7, 8,    // column 1
      3, 4, 9, 10,   // column 2
      5, 6, 11, 12,  // column 3
  });
  m.SetFilter({
      1, 2, 3, 4,        //
      -9, 10, -11, 12,   //
      5, 6, 7, 8,        //
      13, -14, 15, -16,  //
  });
  m.SetBias({1, 2, 3, 4});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 71, -34, 99, -20,  //
                             }));
}

TEST_P(DepthwiseConvolutionOpTest, StrideTest) {
  StrideTest(GetRegistration(), /*num_thread=*/1);
}

TEST_P(DepthwiseConvolutionOpTest, MultithreadStrideTest) {
  StrideTest(GetRegistration(), /*num_thread=*/4);
}

void PaddingTest(TfLiteRegistration* registration, int num_thread) {
  DepthwiseConvolutionOpModel m(
      registration, {TensorType_FLOAT32, {1, 3, 2, 2}},
      {TensorType_FLOAT32, {1, 2, 2, 4}}, {TensorType_FLOAT32, {}},
      Padding_SAME,
      /*dilation_factor*/ 1,
      /*stride_width*/ 2,
      /*stride_height*/ 2,
      /*ActivationFunctionType*/ ActivationFunctionType_NONE);

  m.SetInput({
      1, 2, 7, 8,    // column 1
      3, 4, 9, 10,   // column 2
      5, 6, 11, 12,  // column 3
  });
  m.SetFilter({
      1, 2, 3, 4,        //
      -9, 10, -11, 12,   //
      5, 6, 7, 8,        //
      13, -14, 15, -16,  //
  });
  m.SetBias({1, 2, 3, 4});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 71, -34, 99, -20,     //
                                 -93, 122, -111, 172,  //
                             }));
}

TEST_P(DepthwiseConvolutionOpTest, PaddingTest) {
  PaddingTest(GetRegistration(), /*num_thread=*/1);
}

TEST_P(DepthwiseConvolutionOpTest, MultithreadPaddingTest) {
  PaddingTest(GetRegistration(), /*num_thread=*/4);
}

void SimpleTest(TfLiteRegistration* registration, int num_thread) {
  DepthwiseConvolutionOpModel m(registration,
                                {TensorType_FLOAT32, {1, 3, 2, 2}},
                                {TensorType_FLOAT32, {1, 2, 2, 4}},
                                {TensorType_FLOAT32, {}}, Padding_VALID);

  m.SetInput({
      1, 2, 7, 8,    // column 1
      3, 4, 9, 10,   // column 2
      5, 6, 11, 12,  // column 3
  });
  m.SetFilter({
      1, 2, 3, 4,        //
      -9, 10, -11, 12,   //
      5, 6, 7, 8,        //
      13, -14, 15, -16,  //
  });
  m.SetBias({1, 2, 3, 4});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 71, -34, 99, -20,  //
                                 91, -26, 127, -4,  //
                             }));
}

TEST_P(DepthwiseConvolutionOpTest, SimpleTest) {
  SimpleTest(GetRegistration(), /*num_thread=*/1);
}

TEST_P(DepthwiseConvolutionOpTest, MultithreadSimpleTest) {
  SimpleTest(GetRegistration(), /*num_thread=*/4);
}

void SimpleDilatedTestPaddingValid(TfLiteRegistration* registration,
                                   int num_thread) {
  const int depth = 1;
  const int image_width = 9;
  const int image_height = 9;
  const int image_batch_count = 1;
  const int filter_size = 3;
  const int filter_count = 1;
  const int dilation_factor = 3;
  DepthwiseConvolutionOpModel m(
      registration,
      {TensorType_FLOAT32,
       {image_batch_count, image_height, image_width, depth}},
      {TensorType_FLOAT32, {depth, filter_size, filter_size, filter_count}},
      {TensorType_FLOAT32, {}}, Padding_VALID, dilation_factor);

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

TEST_P(DepthwiseConvolutionOpTest, SimpleDilatedTestPaddingValid) {
  SimpleDilatedTestPaddingValid(GetRegistration(), /*num_thread=*/1);
}

TEST_P(DepthwiseConvolutionOpTest, MultithreadSimpleDilatedTestPaddingValid) {
  SimpleDilatedTestPaddingValid(GetRegistration(), /*num_thread=*/4);
}

void SimpleDilatedTestPaddingSame(TfLiteRegistration* registration,
                                  int num_thread) {
  const int depth = 1;
  const int image_width = 3;
  const int image_height = 3;
  const int image_batch_count = 1;
  const int filter_size = 2;
  const int filter_count = 1;
  const int dilation_factor = 2;
  DepthwiseConvolutionOpModel m(
      registration,
      {TensorType_FLOAT32,
       {image_batch_count, image_height, image_width, depth}},
      {TensorType_FLOAT32, {depth, filter_size, filter_size, filter_count}},
      {TensorType_FLOAT32, {}}, Padding_SAME, dilation_factor);

  // The image matrix is:
  // | 1 | 1 | 1 |
  // | 1 | 1 | 1 |
  // | 1 | 1 | 1 |
  m.SetInput({1, 1, 1, 1, 1, 1, 1, 1, 1});
  // The filter matrix is:
  // | 1 | 2 |
  // | 3 | 4 |
  m.SetFilter({1, 2, 3, 4});
  // No bias for this test.
  m.SetBias({0});
  m.SetNumThreads(num_thread);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // Output:
  // | 4 | 7 | 3 |
  // | 6 |10 | 4 |
  // | 2 | 3 | 1 |
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({4, 7, 3, 6, 10, 4, 2, 3, 1}));
}

TEST_P(DepthwiseConvolutionOpTest, SimpleDilatedTestPaddingSame) {
  SimpleDilatedTestPaddingSame(GetRegistration(), /*num_thread=*/1);
}

TEST_P(DepthwiseConvolutionOpTest, MultithreadSimpleDilatedTestPaddingSame) {
  SimpleDilatedTestPaddingSame(GetRegistration(), /*num_thread=*/4);
}

void BatchPaddingValidTest(TfLiteRegistration* registration, int num_thread) {
  const int input_batch = 2;
  const int input_width = 3;
  const int input_height = 3;
  const int input_depth = 4;
  const int filter_batch = 1;
  const int filter_size = 3;
  const int filter_depth = 4;
  DepthwiseConvolutionOpModel m(
      registration,
      {TensorType_FLOAT32,
       {input_batch, input_height, input_width, input_depth}},
      {TensorType_FLOAT32,
       {filter_batch, filter_size, filter_size, filter_depth}},
      {TensorType_FLOAT32, {}}, Padding_VALID);

  // clang-format off
  m.SetInput({
      // array of 3 x 24 => [2, 3, 3, 4]
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0
  });

  m.SetFilter({
      // array of 9 x 4 => [1, 3, 3, 4]
      1, 2, 3, 4,
      1, 2, 3, 4,
      1, 2, 3, 4,
      1, 2, 3, 4,
      1, 2, 3, 4,
      1, 2, 3, 4,
      1, 2, 3, 4,
      1, 2, 3, 4,
      1, 2, 3, 4
  });
  // clang-format on

  // No bias for this test.
  m.SetBias({0, 0, 0, 0});
  m.SetNumThreads(num_thread);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // clang-format off
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({
        9, 18, 0, 0,
        9, 18, 0, 0
      }));
  // clang-format on
}

TEST_P(DepthwiseConvolutionOpTest, BatchPaddingValidTest) {
  BatchPaddingValidTest(GetRegistration(), /*num_thread=*/1);
}

TEST_P(DepthwiseConvolutionOpTest, MultithreadBatchPaddingValidTest) {
  BatchPaddingValidTest(GetRegistration(), /*num_thread=*/4);
}

void BatchPaddingSameTest(TfLiteRegistration* registration, int num_thread) {
  const int input_batch = 4;
  const int input_width = 2;
  const int input_height = 2;
  const int input_depth = 1;
  const int filter_batch = 1;
  const int filter_size = 3;
  const int filter_depth = 1;
  DepthwiseConvolutionOpModel m(
      registration,
      {TensorType_FLOAT32,
       {input_batch, input_height, input_width, input_depth}},
      {TensorType_FLOAT32,
       {filter_batch, filter_size, filter_size, filter_depth}},
      {TensorType_FLOAT32, {}}, Padding_SAME);

  // clang-format off
  m.SetInput({
      // array of 4 x 4 => [4, 2, 2, 1]
      1, 1, 1, 1,
      0, 0, 0, 0,
      1, 1, 2, 2,
      2, 2, 2, 2
  });

  m.SetFilter({
      // array of 3 x 3 => [1, 3, 3, 1]
      1, 1, 1,
      0, 2, 0,
      1, 1, 1
  });
  // clang-format on

  // No bias for this test.
  m.SetBias({0});
  m.SetNumThreads(num_thread);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // clang-format off
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({
        4, 4, 4, 4,
        0, 0, 0, 0,
        6, 6, 6, 6,
        8, 8, 8, 8
      }));
  // clang-format on
}

TEST_P(DepthwiseConvolutionOpTest, BatchPaddingSameTest) {
  BatchPaddingSameTest(GetRegistration(), /*num_thread=*/1);
}

TEST_P(DepthwiseConvolutionOpTest, MultithreadBatchPaddingSameTest) {
  BatchPaddingSameTest(GetRegistration(), /*num_thread=*/4);
}

class QuantizedDepthwiseConvolutionOpModel
    : public BaseDepthwiseConvolutionOpModel {
 public:
  using BaseDepthwiseConvolutionOpModel::BaseDepthwiseConvolutionOpModel;

  void SetInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<uint8_t>(input_, data);
  }
  void SetInput(const std::vector<float>& data) {
    QuantizeAndPopulate<uint8_t>(input_, data);
  }

  void SetFilter(std::initializer_list<float> data) {
    QuantizeAndPopulate<uint8_t>(filter_, data);
  }
  void SetFilter(const std::vector<float>& data) {
    QuantizeAndPopulate<uint8_t>(filter_, data);
  }

  void SetBias(std::initializer_list<float> data) {
    QuantizeAndPopulate<int32_t>(bias_, data);
  }
  void SetBias(const std::vector<float>& data) {
    QuantizeAndPopulate<int32_t>(bias_, data);
  }

  std::vector<uint8_t> GetOutput() { return ExtractVector<uint8_t>(output_); }
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }
};

const auto kQuantizedKernelMap = new std::map<string, TfLiteRegistration*>({
    {"Reference", ops::builtin::Register_DEPTHWISE_CONVOLUTION_REF()},
    {"GenericOptimized",
     ops::builtin::Register_DEPTHWISE_CONVOLUTION_GENERIC_OPT()},
    {"NeonOptimized", ops::builtin::Register_DEPTHWISE_CONVOLUTION_NEON_OPT()},
    {"Uint8", ops::builtin::Register_DEPTHWISE_CONV_2D_UINT8()},
});

class QuantizedDepthwiseConvolutionOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kQuantizedKernelMap;
  }
};

// Only enable this test for neon.
#ifdef USE_NEON
TEST_F(QuantizedDepthwiseConvolutionOpTest, LargeOutputChannelTest) {
  const TensorData input({TensorType_UINT8, {1, 4, 4, 2400}, -63.5, 64});
  const TensorData filter({TensorType_UINT8, {1, 3, 3, 2400}, -63.5, 64});
  const TensorData output({TensorType_UINT8, {}, -127, 128});
  const Padding padding = Padding_VALID;

  // Populate input, filter & bias data.
  const int input_size = 1 * 4 * 4 * 2400;
  const int filter_size = 1 * 3 * 3 * 2400;
  const int bias_size = 2400;
  std::vector<float> input_data(input_size);
  std::vector<float> filter_data(filter_size);
  std::vector<float> bias_data(bias_size);
  for (int i = 0; i < input_size; ++i) {
    input_data[i] = UniformRandomFloat(-1, -1);
  }
  for (int i = 0; i < filter_size; ++i) {
    filter_data[i] = UniformRandomFloat(-1, -1);
  }
  for (int i = 0; i < bias_size; ++i) {
    bias_data[i] = UniformRandomFloat(-1, -1);
  }

  // Make sure reference impl & optimized impl produce the same result.
  QuantizedDepthwiseConvolutionOpModel reference_impl(
      ops::builtin::Register_DEPTHWISE_CONVOLUTION_REF(), input, filter, output,
      padding);
  reference_impl.SetInput(input_data);
  reference_impl.SetFilter(filter_data);
  reference_impl.SetBias(bias_data);
  reference_impl.Invoke();

  QuantizedDepthwiseConvolutionOpModel optimized_impl(
      ops::builtin::Register_DEPTHWISE_CONVOLUTION_GENERIC_OPT(), input, filter,
      output, padding);
  optimized_impl.SetInput(input_data);
  optimized_impl.SetFilter(filter_data);
  optimized_impl.SetBias(bias_data);
  optimized_impl.Invoke();

  EXPECT_THAT(reference_impl.GetOutput(), optimized_impl.GetOutput());
}
#endif

// In this test we set the input and output scales so that the results match
// exactly the 'non-quantized' version.
TEST_P(QuantizedDepthwiseConvolutionOpTest, SimpleTestQuantized) {
  QuantizedDepthwiseConvolutionOpModel m(
      GetRegistration(), {TensorType_UINT8, {1, 3, 2, 2}, -63.5, 64},
      {TensorType_UINT8, {1, 2, 2, 4}, -63.5, 64},
      {TensorType_UINT8, {}, -127, 128}, Padding_VALID);

  m.SetInput({
      1, 2, 7, 8,    // column 1
      3, 4, 9, 10,   // column 2
      5, 6, 11, 12,  // column 3
  });
  m.SetFilter({
      1, 2, 3, 4,        //
      -9, 10, -11, 12,   //
      5, 6, 7, 8,        //
      13, -14, 15, -16,  //
  });
  m.SetBias({1, 2, 3, 4});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput(), ElementsAreArray(ArrayFloatNear(
                                            {
                                                71, -34, 99, -20,  //
                                                91, -26, 127, -4,  //
                                            },
                                            1e-5)));
  // For good  measure, let's also verify the quantized values:
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 198, 93, 226, 107,   //
                                 218, 101, 254, 123,  //
                             }));
}

TEST_P(DepthwiseConvolutionOpTest,
       SimpleTestQuantizedFilterMultiplierGreaterThan1) {
  QuantizedDepthwiseConvolutionOpModel quant_op(
      GetRegistration(), {TensorType_UINT8, {1, 3, 2, 2}, -63.5, 64},
      {TensorType_UINT8, {1, 2, 2, 4}, -128.5, 128},
      {TensorType_UINT8, {}, -127, 128}, Padding_VALID);
  DepthwiseConvolutionOpModel float_op(GetRegistration(),
                                       {TensorType_FLOAT32, {1, 3, 2, 2}},
                                       {TensorType_FLOAT32, {1, 2, 2, 4}},
                                       {TensorType_FLOAT32, {}}, Padding_VALID);

  std::initializer_list<float> input = {
      1, 2, 7,  8,   // column 1
      3, 4, 9,  10,  // column 2
      5, 6, 11, 12,  // column 3
  };
  std::initializer_list<float> filter = {
      1,  2,   3,   4,    //
      -9, 10,  -11, 12,   //
      5,  6,   7,   8,    //
      13, -14, 15,  -16,  //
  };
  std::initializer_list<float> bias = {1, 2, 3, 4};

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

TEST_P(DepthwiseConvolutionOpTest,
       SimpleTestQuantizedOutputMultiplierGreaterThan1) {
  QuantizedDepthwiseConvolutionOpModel quant_op(
      GetRegistration(), {TensorType_UINT8, {1, 3, 2, 2}, -128.5, 128},
      {TensorType_UINT8, {1, 2, 2, 4}, -128.5, 128},
      {TensorType_UINT8, {}, -127, 128}, Padding_VALID);
  DepthwiseConvolutionOpModel float_op(GetRegistration(),
                                       {TensorType_FLOAT32, {1, 3, 2, 2}},
                                       {TensorType_FLOAT32, {1, 2, 2, 4}},
                                       {TensorType_FLOAT32, {}}, Padding_VALID);

  std::initializer_list<float> input = {
      1, 2, 7,  8,   // column 1
      3, 4, 9,  10,  // column 2
      5, 6, 11, 12,  // column 3
  };
  std::initializer_list<float> filter = {
      1,  2,   3,   4,    //
      -9, 10,  -11, 12,   //
      5,  6,   7,   8,    //
      13, -14, 15,  -16,  //
  };
  std::initializer_list<float> bias = {1, 2, 3, 4};

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

TEST_P(QuantizedDepthwiseConvolutionOpTest, SimpleDilatedTestPaddingValid) {
  const int depth = 1;
  const int image_width = 9;
  const int image_height = 9;
  const int image_batch_count = 1;
  const int filter_size = 3;
  const int filter_count = 1;
  const int dilation_factor = 3;
  QuantizedDepthwiseConvolutionOpModel m(
      GetRegistration(),
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // Since the dilation rate is 3 this will reduce the size of the output from
  // 10x10 to 3x3 of all 5s. Specifically:
  // | 5 | 5 | 5 |
  // | 5 | 5 | 5 |
  // | 5 | 5 | 5 |
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray({5, 5, 5, 5, 5, 5, 5, 5, 5}));
}

TEST_P(QuantizedDepthwiseConvolutionOpTest, SimpleDilatedTestPaddingSame) {
  const int depth = 1;
  const int image_width = 3;
  const int image_height = 3;
  const int image_batch_count = 1;
  const int filter_size = 2;
  const int filter_count = 1;
  const int dilation_factor = 2;
  QuantizedDepthwiseConvolutionOpModel m(
      GetRegistration(),
      {TensorType_UINT8,
       {image_batch_count, image_height, image_width, depth},
       0,
       255},
      {TensorType_UINT8,
       {depth, filter_size, filter_size, filter_count},
       0,
       255},
      {TensorType_UINT8, {}, 0, 255}, Padding_SAME, dilation_factor);

  // The image matrix is:
  // | 1 | 1 | 1 |
  // | 1 | 1 | 1 |
  // | 1 | 1 | 1 |
  m.SetInput({1, 1, 1, 1, 1, 1, 1, 1, 1});
  // The filter matrix is:
  // | 1 | 2 |
  // | 3 | 4 |
  m.SetFilter({1, 2, 3, 4});
  // No bias for this test.
  m.SetBias({0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // Output:
  // | 4 | 7 | 3 |
  // | 6 |10 | 4 |
  // | 2 | 3 | 1 |
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray({4, 7, 3, 6, 10, 4, 2, 3, 1}));
}

TEST_P(QuantizedDepthwiseConvolutionOpTest, MultithreadOnRowUint8GeneralTest) {
  const int depth = 1;
  const int image_width = 4;
  const int image_height = 28;
  const int image_batch_count = 3;
  const int filter_size = 3;
  const int filter_count = 1;

  QuantizedDepthwiseConvolutionOpModel m(
      GetRegistration(),
      {TensorType_UINT8,
       {image_batch_count, image_height, image_width, depth},
       0,
       255},
      {TensorType_UINT8,
       {depth, filter_size, filter_size, filter_count},
       0,
       255},
      {TensorType_UINT8, {}, 0, 255}, Padding_VALID);

  // clang-format off
  m.SetInput({
      0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,
      0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,
      1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,
      1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,
      1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,
      0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,
      0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,

      0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,
      0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,
      2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,
      2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,
      2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,
      0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,
      0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,

      0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,
      0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,
      3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,
      3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,
      3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,
      0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,
      0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0
  });
  // clang-format on

  // The filter matrix is:
  // | 1 | 2 | 3 |
  // | 4 | 5 | 6 |
  // | 7 | 8 | 9 |
  m.SetFilter({1, 2, 3, 4, 5, 6, 7, 8, 9});
  // No bias for this test.
  m.SetBias({0});
  m.SetNumThreads(4);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // clang-format off
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray({
          0, 0,    0, 0,    0, 0,    0, 0,
          0, 0,    0, 0,    24, 24,  39, 39,
          45, 45,  45, 45,  45, 45,  45, 45,
          45, 45,  45, 45,  45, 45,  45, 45,
          45, 45,  45, 45,  21, 21,  6, 6,
          0, 0,    0, 0,    0, 0,    0, 0,
          0, 0,    0, 0,

          0, 0,    0, 0,    0, 0,    0, 0,
          0, 0,    0, 0,    48, 48,  78, 78,
          90, 90,  90, 90,  90, 90,  90, 90,
          90, 90,  90, 90,  90, 90,  90, 90,
          90, 90,  90, 90,  42, 42,  12, 12,
          0, 0,    0, 0,    0, 0,    0, 0,
          0, 0,    0, 0,

          0, 0,      0, 0,      0, 0,      0, 0,
          0, 0,      0, 0,      72, 72,    117, 117,
          135, 135,  135, 135,  135, 135,  135, 135,
          135, 135,  135, 135,  135, 135,  135, 135,
          135, 135,  135, 135,  63, 63,    18, 18,
          0, 0,      0, 0,      0, 0,      0, 0,
          0, 0,      0, 0,
      }));
  // clang-format on
}

TEST_P(QuantizedDepthwiseConvolutionOpTest,
       MultithreadOnBatchUint8GeneralTest) {
  const int depth = 1;
  const int image_width = 8;
  const int image_height = 4;
  const int image_batch_count = 6;
  const int filter_size = 3;
  const int filter_count = 1;

  QuantizedDepthwiseConvolutionOpModel m(
      GetRegistration(),
      {TensorType_UINT8,
       {image_batch_count, image_height, image_width, depth},
       0,
       255},
      {TensorType_UINT8,
       {depth, filter_size, filter_size, filter_count},
       0,
       255},
      {TensorType_UINT8, {}, 0, 255}, Padding_VALID);

  // clang-format off
  m.SetInput({
      0, 0, 0, 0,  0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1,
      1, 1, 1, 1,  1, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,

      0, 0, 0, 0,  0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1,
      1, 1, 1, 1,  1, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,

      0, 0, 0, 0,  0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1,
      1, 1, 1, 1,  1, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,

      0, 0, 0, 0,  0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1,
      1, 1, 1, 1,  1, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,

      0, 0, 0, 0,  0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1,
      1, 1, 1, 1,  1, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,

      0, 0, 0, 0,  0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1,
      1, 1, 1, 1,  1, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0
  });
  // clang-format on

  // The filter matrix is:
  // | 1 | 2 | 3 |
  // | 4 | 5 | 6 |
  // | 7 | 8 | 9 |
  m.SetFilter({1, 2, 3, 4, 5, 6, 7, 8, 9});
  // No bias for this test.
  m.SetBias({0});
  m.SetNumThreads(4);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // clang-format off
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray({
          39, 39, 39, 39, 39, 39,
          21, 21, 21, 21, 21, 21,

          39, 39, 39, 39, 39, 39,
          21, 21, 21, 21, 21, 21,

          39, 39, 39, 39, 39, 39,
          21, 21, 21, 21, 21, 21,

          39, 39, 39, 39, 39, 39,
          21, 21, 21, 21, 21, 21,

          39, 39, 39, 39, 39, 39,
          21, 21, 21, 21, 21, 21,

          39, 39, 39, 39, 39, 39,
          21, 21, 21, 21, 21, 21
      }));
  // clang-format on
}

TEST_P(QuantizedDepthwiseConvolutionOpTest, MultithreadOnRowValidPaddingTest) {
  // This test runs through DepthwiseConv3x3Filter with __aarch64__, and runs
  // through DepthwiseConvGeneral with other configs.
  const int input_batch = 1;
  const int input_width = 3;
  const int input_height = 5;
  const int input_depth = 8;
  const int filter_batch = 1;
  const int filter_size = 3;
  const int filter_depth = 8;

  QuantizedDepthwiseConvolutionOpModel m(
      GetRegistration(),
      {TensorType_UINT8,
       {input_batch, input_height, input_width, input_depth},
       0,
       128},
      {TensorType_UINT8,
       {filter_batch, filter_size, filter_size, filter_depth},
       0,
       128},
      {TensorType_UINT8, {}, 0, 255}, Padding_VALID);

  // clang-format off
  m.SetInput({
    // array of 15 x 8 => [1, 5, 3, 8]
      1, 1, 0, 0,  1, 1, 0, 0,
      1, 1, 0, 0,  1, 1, 0, 0,
      1, 1, 0, 0,  1, 1, 0, 0,
      1, 1, 0, 0,  1, 1, 0, 0,
      1, 1, 0, 0,  1, 1, 0, 0,
      1, 1, 0, 0,  1, 1, 0, 0,
      1, 1, 0, 0,  1, 1, 0, 0,
      1, 1, 0, 0,  1, 1, 0, 0,
      1, 1, 0, 0,  1, 1, 0, 0,
      1, 1, 0, 0,  1, 1, 0, 0,
      1, 1, 0, 0,  1, 1, 0, 0,
      1, 1, 0, 0,  1, 1, 0, 0,
      1, 1, 0, 0,  1, 1, 0, 0,
      1, 1, 0, 0,  1, 1, 0, 0,
      1, 1, 0, 0,  1, 1, 0, 0
  });

  m.SetFilter({
    // array of 9 x 8 => [1, 3, 3, 8]
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8
  });
  // clang-format on

  // No bias for this test.
  m.SetBias({0, 0, 0, 0, 0, 0, 0, 0});
  m.SetNumThreads(4);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // clang-format off
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray({
        9, 18, 0, 0, 46, 55, 0, 0,
        9, 18, 0, 0, 46, 55, 0, 0,
        9, 18, 0, 0, 46, 55, 0, 0
      }));
  // clang-format on
}

TEST_P(QuantizedDepthwiseConvolutionOpTest, MultithreadOnRowSamePaddingTest) {
  // This test runs through DepthwiseConv3x3Filter with __aarch64__, and runs
  // through DepthwiseConvGeneral with other configs.
  const int input_batch = 1;
  const int input_width = 3;
  const int input_height = 3;
  const int input_depth = 8;
  const int filter_batch = 1;
  const int filter_size = 3;
  const int filter_depth = 8;

  QuantizedDepthwiseConvolutionOpModel m(
      GetRegistration(),
      {TensorType_UINT8,
       {input_batch, input_height, input_width, input_depth},
       0,
       128},
      {TensorType_UINT8,
       {filter_batch, filter_size, filter_size, filter_depth},
       0,
       128},
      {TensorType_UINT8, {}, 0, 255}, Padding_SAME);

  // clang-format off
  m.SetInput({
      // array of 9 x 8 => [1, 3, 3, 8]
      1, 1, 0, 0,  1, 1, 0, 0,
      1, 1, 0, 0,  1, 1, 0, 0,
      1, 1, 0, 0,  1, 1, 0, 0,
      1, 1, 0, 0,  1, 1, 0, 0,
      1, 1, 0, 0,  1, 1, 0, 0,
      1, 1, 0, 0,  1, 1, 0, 0,
      1, 1, 0, 0,  1, 1, 0, 0,
      1, 1, 0, 0,  1, 1, 0, 0,
      1, 1, 0, 0,  1, 1, 0, 0
  });

  m.SetFilter({
      // array of 9 x 8 => [1, 3, 3, 8]
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8
  });
  // clang-format on

  // No bias for this test.
  m.SetBias({0, 0, 0, 0, 0, 0, 0, 0});
  m.SetNumThreads(3);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // clang-format off
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray({
        // array of 9 x 8 => [1, 3, 3, 8]
        4, 8, 0, 0, 20, 24, 0, 0,
        6, 12, 0, 0, 30, 37, 0, 0,
        4, 8, 0, 0, 20, 24, 0, 0,
        6, 12, 0, 0, 30, 37, 0, 0,
        9, 18, 0, 0, 46, 55, 0, 0,
        6, 12, 0, 0, 30, 37, 0, 0,
        4, 8, 0, 0, 20, 24, 0, 0,
        6, 12, 0, 0, 30, 37, 0, 0,
        4, 8, 0, 0, 20, 24, 0, 0,
      }));
  // clang-format on
}

TEST_P(QuantizedDepthwiseConvolutionOpTest,
       MultithreadOnBatchValidPaddingTest) {
  // This test runs through DepthwiseConv3x3Filter with __aarch64__, and runs
  // through DepthwiseConvGeneral with other configs.
  const int input_batch = 2;
  const int input_width = 3;
  const int input_height = 3;
  const int input_depth = 8;
  const int filter_batch = 1;
  const int filter_size = 3;
  const int filter_depth = 8;

  QuantizedDepthwiseConvolutionOpModel m(
      GetRegistration(),
      {TensorType_UINT8,
       {input_batch, input_height, input_width, input_depth},
       0,
       128},
      {TensorType_UINT8,
       {filter_batch, filter_size, filter_size, filter_depth},
       0,
       128},
      {TensorType_UINT8, {}, 0, 255}, Padding_VALID);

  // clang-format off
  m.SetInput({
      // array of 2 x 3 x 24 => [2, 3, 3, 8]
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,

      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0
  });

  m.SetFilter({
      // array of 9 x 8 => [1, 3, 3, 8]
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8
  });
  // clang-format on

  // No bias for this test.
  m.SetBias({0, 0, 0, 0, 0, 0, 0, 0});
  m.SetNumThreads(2);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // clang-format off
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray({
        9, 18, 0, 0, 46, 55, 0, 0,
        9, 18, 0, 0, 46, 55, 0, 0
      }));
  // clang-format on
}

TEST_P(QuantizedDepthwiseConvolutionOpTest, MultithreadOnBatchSamePaddingTest) {
  // This test runs through DepthwiseConv3x3Filter with __aarch64__, and runs
  // through DepthwiseConvGeneral with other configs.
  const int input_batch = 2;
  const int input_width = 3;
  const int input_height = 3;
  const int input_depth = 8;
  const int filter_batch = 1;
  const int filter_size = 3;
  const int filter_depth = 8;

  QuantizedDepthwiseConvolutionOpModel m(
      GetRegistration(),
      {TensorType_UINT8,
       {input_batch, input_height, input_width, input_depth},
       0,
       128},
      {TensorType_UINT8,
       {filter_batch, filter_size, filter_size, filter_depth},
       0,
       128},
      {TensorType_UINT8, {}, 0, 255}, Padding_SAME);

  // clang-format off
  m.SetInput({
      // array of 2 x 3 x 24 => [2, 3, 3, 8]
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,

      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0
  });

  m.SetFilter({
      // array of 9 x 8 => [1, 3, 3, 8]
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8
  });
  // clang-format on

  // No bias for this test.
  m.SetBias({0, 0, 0, 0, 0, 0, 0, 0});
  m.SetNumThreads(3);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // clang-format off
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray({
        // array of 9 x 16 => [2, 3, 3, 8]
        4, 8,  0, 0, 20, 24, 0, 0,   6, 12, 0, 0, 30, 37, 0, 0,
        4, 8,  0, 0, 20, 24, 0, 0,   6, 12, 0, 0, 30, 37, 0, 0,
        9, 18, 0, 0, 46, 55, 0, 0,   6, 12, 0, 0, 30, 37, 0, 0,
        4, 8,  0, 0, 20, 24, 0, 0,   6, 12, 0, 0, 30, 37, 0, 0,
        4, 8,  0, 0, 20, 24, 0, 0,   4, 8,  0, 0, 20, 24, 0, 0,
        6, 12, 0, 0, 30, 37, 0, 0,   4, 8,  0, 0, 20, 24, 0, 0,
        6, 12, 0, 0, 30, 37, 0, 0,   9, 18, 0, 0, 46, 55, 0, 0,
        6, 12, 0, 0, 30, 37, 0, 0,   4, 8,  0, 0, 20, 24, 0, 0,
        6, 12, 0, 0, 30, 37, 0, 0,   4, 8,  0, 0, 20, 24, 0, 0,
      }));
  // clang-format on
}

TEST_P(QuantizedDepthwiseConvolutionOpTest,
       MultithreadOnRowSamePaddingStrideTest) {
  // This test runs through DepthwiseConv3x3Filter with __aarch64__, and runs
  // through DepthwiseConvGeneral with other configs.
  const int input_batch = 1;
  const int input_width = 3;
  const int input_height = 3;
  const int input_depth = 8;
  const int filter_batch = 1;
  const int filter_size = 3;
  const int filter_depth = 8;

  QuantizedDepthwiseConvolutionOpModel m(
      GetRegistration(),
      {TensorType_UINT8,
       {input_batch, input_height, input_width, input_depth},
       0,
       128},
      {TensorType_UINT8,
       {filter_batch, filter_size, filter_size, filter_depth},
       0,
       128},
      {TensorType_UINT8, {}, 0, 255}, Padding_SAME,
      /* dilation_factor = */ 1,
      /* stride_width = */ 2,
      /* stride_height = */ 2);

  // clang-format off
  m.SetInput({
      // array of 3 x 24 => [1, 3, 3, 8]
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0
  });

  m.SetFilter({
      // array of 9 x 8 => [1, 3, 3, 8]
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8
  });
  // clang-format on

  // No bias for this test.
  m.SetBias({0, 0, 0, 0, 0, 0, 0, 0});
  m.SetNumThreads(4);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // clang-format off
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray({
        4, 8, 0, 0, 20, 24, 0, 0,
        4, 8, 0, 0, 20, 24, 0, 0,
        4, 8, 0, 0, 20, 24, 0, 0,
        4, 8, 0, 0, 20, 24, 0, 0,
      }));
  // clang-format on
}

TEST_P(QuantizedDepthwiseConvolutionOpTest,
       MultithreadOnRowValidPaddingStrideTest) {
  const int input_batch = 1;
  const int input_width = 5;
  const int input_height = 5;
  const int input_depth = 8;
  const int filter_batch = 1;
  const int filter_size = 3;
  const int filter_depth = 8;

  QuantizedDepthwiseConvolutionOpModel m(
      GetRegistration(),
      {TensorType_UINT8,
       {input_batch, input_height, input_width, input_depth},
       0,
       128},
      {TensorType_UINT8,
       {filter_batch, filter_size, filter_size, filter_depth},
       0,
       128},
      {TensorType_UINT8, {}, 0, 255}, Padding_VALID,
      /* dilation_factor = */ 1,
      /* stride_width = */ 2,
      /* stride_height = */ 2);

  // clang-format off
  m.SetInput({
    // array of 8 x 24 + 8 => [1, 5, 5, 8]
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0
  });

  m.SetFilter({
      // array of 9 x 8 => [1, 3, 3, 8]
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8
  });
  // clang-format on

  // No bias for this test.
  m.SetBias({0, 0, 0, 0, 0, 0, 0, 0});
  m.SetNumThreads(4);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // clang-format off
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray({
        9, 18, 0, 0, 46, 55, 0, 0,
        9, 18, 0, 0, 46, 55, 0, 0,
        9, 18, 0, 0, 46, 55, 0, 0,
        9, 18, 0, 0, 46, 55, 0, 0
      }));
  // clang-format on
}

TEST_P(QuantizedDepthwiseConvolutionOpTest,
       MultithreadOnRowDepthMultiplierTest) {
  const int input_batch = 1;
  const int input_width = 3;
  const int input_height = 3;
  const int input_depth = 8;
  const int filter_batch = 1;
  const int filter_size = 3;
  const int filter_depth = 16;

  QuantizedDepthwiseConvolutionOpModel m(
      GetRegistration(),
      {TensorType_UINT8,
       {input_batch, input_height, input_width, input_depth},
       0,
       128},
      {TensorType_UINT8,
       {filter_batch, filter_size, filter_size, filter_depth},
       0,
       128},
      {TensorType_UINT8, {}, 0, 255}, Padding_VALID);

  // clang-format off
  m.SetInput({
      // array of 3 x 24 => [1, 3, 3, 8]
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
      1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0
  });

  m.SetFilter({
      // array of 9 x 16 => [1, 3, 3, 16]
      1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
  });
  // clang-format on

  // No bias for this test.
  m.SetBias({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  m.SetNumThreads(4);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // clang-format off
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray({
        9, 18, 27, 37, 0, 0, 0, 0,
        9, 18, 27, 37, 0, 0, 0, 0
      }));
  // clang-format on
}

TEST_P(QuantizedDepthwiseConvolutionOpTest, MultithreadDifferentPaddingTest) {
  const int input_batch = 1;
  const int input_width = 4;
  const int input_height = 5;
  const int input_depth = 2;
  const int filter_batch = 1;
  const int filter_size = 3;
  const int filter_depth = 2;

  QuantizedDepthwiseConvolutionOpModel m(
      GetRegistration(),
      {TensorType_UINT8,
       {input_batch, input_height, input_width, input_depth},
       0,
       128},
      {TensorType_UINT8,
       {filter_batch, filter_size, filter_size, filter_depth},
       0,
       128},
      {TensorType_UINT8, {}, 0, 255}, Padding_SAME,
      /* dilation_factor = */ 1,
      /* stride_width = */ 2,
      /* stride_height = */ 2);

  // clang-format off
  m.SetInput({
      // array of 2 x 16 => [1, 4, 4, 2]
      1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
      1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0
  });

  m.SetFilter({
      // array of 8 x 2 + 2 => [1, 3, 3, 2]
      1, 2, 1, 2, 1, 2, 1, 2,
      1, 2, 1, 2, 1, 2, 1, 2,
      1, 2
  });
  // clang-format on

  // No bias for this test.
  m.SetBias({0, 0});
  m.SetNumThreads(4);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  // clang-format off
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray({
        6, 0, 4, 0,
        9, 0, 6, 0,
        6, 0, 4, 0
      }));
  // clang-format on
}

class PerChannelQuantizedDepthwiseConvolutionOpModel
    : public BaseDepthwiseConvolutionOpModel {
 public:
  using BaseDepthwiseConvolutionOpModel::BaseDepthwiseConvolutionOpModel;

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

class PerChannelQuantizedDepthwiseConvolutionOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kKernelMap;
  }
};

TEST_P(PerChannelQuantizedDepthwiseConvolutionOpTest, SimplePerTensorTest) {
  PerChannelQuantizedDepthwiseConvolutionOpModel m(
      GetRegistration(), {TensorType_INT8, {1, 2, 3, 2}, -63.5, 64, 0.5, -1},
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear({43, 48, 18, 22, 3, -4, -28, -36})));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({85, 95, 35, 43, 5, -9, -57, -73}));
}

TEST_P(PerChannelQuantizedDepthwiseConvolutionOpTest, SimplePerAxisTest) {
  PerChannelQuantizedDepthwiseConvolutionOpModel m(
      GetRegistration(), {TensorType_INT8, {1, 2, 3, 2}, -63.5, 64, 0.5, -1},
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
      {TensorType_INT8, {}, -63.5, 64, 0.5, -1}, Padding_VALID);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear({43, 48, 21, 22, 3, -4, -30, -54})));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({85, 95, 41, 43, 5, -9, -61, -109}));
}

// Same as previous test, except the shift will be negative for the outputs.
TEST_P(PerChannelQuantizedDepthwiseConvolutionOpTest,
       SimpleTestNegativeOutputShift) {
  PerChannelQuantizedDepthwiseConvolutionOpModel m(
      GetRegistration(), {TensorType_INT8, {1, 2, 3, 2}, -63.5, 64, 0.5, -1},
      {TensorType_INT8,
       // [1 * 2 * 2 * 4] as [input_channel, y, x, output_channel]
       {1, 2, 2, 4},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{0.1, 0.2, 0.3, 0.4},
       /*per_channel_quantization_offsets=*/{0, 0, 0, 0},
       /*channel_index=*/3},
      {TensorType_INT8, {}, -63.5, 64, 0.5, -1}, Padding_VALID);
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear({43, 48, 18.5, 22, 3, -4, -28.5, -36})));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({85, 95, 36, 43, 5, -9, -58, -73}));
}

// Same as previous test, except the shift will be mixed for the outputs.
TEST_P(PerChannelQuantizedDepthwiseConvolutionOpTest,
       SimpleTestMixedOutputShift) {
  PerChannelQuantizedDepthwiseConvolutionOpModel m(
      GetRegistration(), {TensorType_INT8, {1, 2, 3, 2}, -63.5, 64, 0.5, -1},
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear({43, 48, 21, 22, 3, -4, -30, -36})));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({85, 95, 41, 43, 5, -9, -61, -73}));
}

TEST_P(PerChannelQuantizedDepthwiseConvolutionOpTest, Simple3x3FilterTest) {
  PerChannelQuantizedDepthwiseConvolutionOpModel m(
      GetRegistration(), {TensorType_INT8, {1, 3, 3, 8}, -63.5, 64, 0.5, -1},
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({9, 18, 0, 0, 47, 54, 0, 0})));
}

TEST_P(PerChannelQuantizedDepthwiseConvolutionOpTest,
       Simple3x3FilterPaddingSameTest) {
  PerChannelQuantizedDepthwiseConvolutionOpModel m(
      GetRegistration(), {TensorType_INT8, {1, 3, 3, 8}, -63.5, 64, 0.5, -1},
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
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({
                  // array of 9 x 8 => [1, 3, 3, 8]
                  4, 8,  0, 0, 21, 24, 0, 0, 6, 12, 0, 0, 31.5, 36, 0, 0,
                  4, 8,  0, 0, 21, 24, 0, 0, 6, 12, 0, 0, 31.5, 36, 0, 0,
                  9, 18, 0, 0, 47, 54, 0, 0, 6, 12, 0, 0, 31.5, 36, 0, 0,
                  4, 8,  0, 0, 21, 24, 0, 0, 6, 12, 0, 0, 31.5, 36, 0, 0,
                  4, 8,  0, 0, 21, 24, 0, 0,
              })));
}

INSTANTIATE_TEST_SUITE_P(
    DepthwiseConvolutionOpTest, DepthwiseConvolutionOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kKernelMap)));

INSTANTIATE_TEST_SUITE_P(
    QuantizedDepthwiseConvolutionOpTest, QuantizedDepthwiseConvolutionOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kQuantizedKernelMap)));

INSTANTIATE_TEST_SUITE_P(
    PerChannelQuantizedDepthwiseConvolutionOpTest,
    PerChannelQuantizedDepthwiseConvolutionOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kKernelMap)));

}  // namespace
}  // namespace tflite
