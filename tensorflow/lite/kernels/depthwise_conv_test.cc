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

TfLiteRegistration* Register_DEPTHWISE_CONVOLUTION_REF();
TfLiteRegistration* Register_DEPTHWISE_CONVOLUTION_GENERIC_OPT();
TfLiteRegistration* Register_DEPTHWISE_CONVOLUTION_NEON_OPT();

}  // namespace builtin
}  // namespace ops

namespace {

using ::testing::ElementsAreArray;

class BaseDepthwiseConvolutionOpModel : public SingleOpModel {
 public:
  // TODO(ahentz): Also test different activation types, bias, padding types,
  // stride values.
  BaseDepthwiseConvolutionOpModel(TfLiteRegistration* registration,
                                  const TensorData& input,
                                  const TensorData& filter,
                                  const TensorData& output,
                                  Padding padding_type,
                                  int dilation_factor = 1) {
    input_ = AddInput(input);
    filter_ = AddInput(filter);

    int bias_size = GetShape(filter_)[3];
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

    int input_depth = GetShape(input_)[3];
    int output_depth = GetShape(filter_)[3];
    int depth_mul = output_depth / input_depth;

    SetBuiltinOp(
        BuiltinOperator_DEPTHWISE_CONV_2D,
        BuiltinOptions_DepthwiseConv2DOptions,
        CreateDepthwiseConv2DOptions(builder_, padding_type, 1, 1, depth_mul,
                                     ActivationFunctionType_NONE,
                                     dilation_factor, dilation_factor)
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

TEST_P(DepthwiseConvolutionOpTest, SimpleTest) {
  DepthwiseConvolutionOpModel m(GetRegistration(),
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

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 71, -34, 99, -20,  //
                                 91, -26, 127, -4,  //
                             }));
}

TEST_P(DepthwiseConvolutionOpTest, SimpleDilatedTestPaddingValid) {
  const int depth = 1;
  const int image_width = 9;
  const int image_height = 9;
  const int image_batch_count = 1;
  const int filter_size = 3;
  const int filter_count = 1;
  const int dilation_factor = 3;
  DepthwiseConvolutionOpModel m(
      GetRegistration(),
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
  m.Invoke();

  // Since the dilation rate is 3 this will reduce the size of the output from
  // 10x10 to 3x3 of all 5s. Specifically:
  // | 5 | 5 | 5 |
  // | 5 | 5 | 5 |
  // | 5 | 5 | 5 |
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 5, 5, 5, 5, 5, 5, 5, 5}));
}

TEST_P(DepthwiseConvolutionOpTest, SimpleDilatedTestPaddingSame) {
  const int depth = 1;
  const int image_width = 3;
  const int image_height = 3;
  const int image_batch_count = 1;
  const int filter_size = 2;
  const int filter_count = 1;
  const int dilation_factor = 2;
  DepthwiseConvolutionOpModel m(
      GetRegistration(),
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
  m.Invoke();

  // Output:
  // | 4 | 7 | 3 |
  // | 6 |10 | 4 |
  // | 2 | 3 | 1 |
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({4, 7, 3, 6, 10, 4, 2, 3, 1}));
}

class QuantizedDepthwiseConvolutionOpModel
    : public BaseDepthwiseConvolutionOpModel {
 public:
  using BaseDepthwiseConvolutionOpModel::BaseDepthwiseConvolutionOpModel;

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

class QuantizedDepthwiseConvolutionOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kKernelMap;
  }
};

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

  m.Invoke();

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

TEST_P(QuantizedDepthwiseConvolutionOpTest,
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
  quant_op.Invoke();

  float_op.SetInput(input);
  float_op.SetFilter(filter);
  float_op.SetBias(bias);
  float_op.Invoke();

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
  m.Invoke();

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
  m.Invoke();

  // Output:
  // | 4 | 7 | 3 |
  // | 6 |10 | 4 |
  // | 2 | 3 | 1 |
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray({4, 7, 3, 6, 10, 4, 2, 3, 1}));
}

INSTANTIATE_TEST_CASE_P(
    DepthwiseConvolutionOpTest, DepthwiseConvolutionOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kKernelMap)));

INSTANTIATE_TEST_CASE_P(
    QuantizedDepthwiseConvolutionOpTest, QuantizedDepthwiseConvolutionOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kKernelMap)));

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
