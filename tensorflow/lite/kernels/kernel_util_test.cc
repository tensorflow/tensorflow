/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/kernel_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace {

void ReportError(TfLiteContext* context, const char* format, ...) {}

class KernelUtilTest : public ::testing::Test {
 public:
  KernelUtilTest() {
    context_.ReportError = ReportError;

    memset(&tensor1_, 0, sizeof(TfLiteTensor));
    memset(&tensor2_, 0, sizeof(TfLiteTensor));
    tensor1_.dims = nullptr;
    tensor2_.dims = nullptr;
    tensor1_.allocation_type = kTfLiteMmapRo;
    tensor2_.allocation_type = kTfLiteMmapRo;
  }
  ~KernelUtilTest() override {
    TfLiteTensorFree(&tensor1_);
    TfLiteTensorFree(&tensor2_);
  }

  void SetShape(TfLiteTensor* tensor, std::initializer_list<int> dims) {
    TfLiteTensorFree(tensor);
    tensor->dims = TfLiteIntArrayCreate(dims.size());
    int i = 0;
    for (const auto& d : dims) {
      tensor->dims->data[i] = d;
      ++i;
    }
  }

  std::vector<int> GetShape(TfLiteIntArray* dims) {
    std::vector<int> result;
    for (int i = 0; i < dims->size; ++i) {
      result.push_back(dims->data[i]);
    }
    return result;
  }

 protected:
  TfLiteContext context_;
  TfLiteTensor tensor1_;
  TfLiteTensor tensor2_;
};

TEST_F(KernelUtilTest, SameShapeEmpty) {
  EXPECT_TRUE(HaveSameShapes(&tensor1_, &tensor2_));

  SetShape(&tensor1_, {1, 2, 3});
  EXPECT_FALSE(HaveSameShapes(&tensor1_, &tensor2_));

  SetShape(&tensor2_, {1, 2});
  EXPECT_FALSE(HaveSameShapes(&tensor1_, &tensor2_));

  SetShape(&tensor2_, {1, 2, 3, 4});
  EXPECT_FALSE(HaveSameShapes(&tensor1_, &tensor2_));

  SetShape(&tensor2_, {1, 2, 3});
  EXPECT_TRUE(HaveSameShapes(&tensor1_, &tensor2_));

  SetShape(&tensor2_, {});
  EXPECT_FALSE(HaveSameShapes(&tensor1_, &tensor2_));

  SetShape(&tensor1_, {});
  EXPECT_TRUE(HaveSameShapes(&tensor1_, &tensor2_));
}

TEST_F(KernelUtilTest, BroadcastShapeIncompatibleDim) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {1, 3});
  EXPECT_NE(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_EQ(output, nullptr);
}

TEST_F(KernelUtilTest, BroadcastShapeOnes) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 1});
  SetShape(&tensor2_, {1, 3});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {1, 1});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  TfLiteIntArrayFree(output);
}

TEST_F(KernelUtilTest, BroadcastShapeScalars) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_THAT(GetShape(output), ::testing::ElementsAre(1, 2));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {});
  SetShape(&tensor2_, {2});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_THAT(GetShape(output), ::testing::ElementsAre(2));
  TfLiteIntArrayFree(output);
}

TEST_F(KernelUtilTest, BroadcastShapeDifferentSizes) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {3, 1, 1});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_THAT(GetShape(output), ::testing::ElementsAre(3, 1, 2));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {1, 2, 3, 4});
  SetShape(&tensor2_, {1, 3, 1});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_THAT(GetShape(output), ::testing::ElementsAre(1, 2, 3, 4));
  TfLiteIntArrayFree(output);
}

TEST_F(KernelUtilTest, CheckAndPopulate) {
  // Create input.
  TfLiteTensor input = {};
  input.type = kTfLiteInt8;
  input.allocation_type = kTfLiteArenaRw;
  input.dims = TfLiteIntArrayCreate(1);
  input.dims->data[0] = 2;
  TfLiteQuantizationParams input_quant = {0.5, 5};
  input.params = input_quant;
  input.quantization.type = kTfLiteAffineQuantization;
  auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  input_params->scale = TfLiteFloatArrayCreate(1);
  input_params->scale->data[0] = 0.5;
  input_params->zero_point = TfLiteIntArrayCreate(1);
  input_params->zero_point->data[0] = 5;
  input.quantization.params = reinterpret_cast<void*>(input_params);

  // Create filter.
  TfLiteTensor filter = {};
  filter.type = kTfLiteInt8;
  filter.allocation_type = kTfLiteArenaRw;
  filter.dims = TfLiteIntArrayCreate(4);
  filter.dims->data[0] = 3;
  filter.dims->data[1] = 4;
  filter.dims->data[2] = 5;
  filter.dims->data[3] = 6;
  TfLiteQuantizationParams filter_quant = {0.25, 0};
  filter.params = filter_quant;
  filter.quantization.type = kTfLiteAffineQuantization;
  auto* filter_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  filter_params->scale = TfLiteFloatArrayCreate(3);
  filter_params->scale->data[0] = 0.25;
  filter_params->scale->data[1] = 0.125;
  filter_params->scale->data[2] = 0.25;
  filter_params->zero_point = TfLiteIntArrayCreate(3);
  filter_params->zero_point->data[0] = 0;
  filter_params->zero_point->data[1] = 0;
  filter_params->zero_point->data[2] = 0;
  filter_params->quantized_dimension = 0;
  filter.quantization.params = reinterpret_cast<void*>(filter_params);

  // Create bias.
  TfLiteTensor bias = {};
  bias.type = kTfLiteInt32;
  bias.allocation_type = kTfLiteArenaRw;
  bias.dims = TfLiteIntArrayCreate(4);
  TfLiteQuantizationParams bias_quant = {0.125, 9};
  bias.params = bias_quant;
  bias.quantization.type = kTfLiteAffineQuantization;
  auto* bias_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  bias_params->scale = TfLiteFloatArrayCreate(3);
  bias_params->scale->data[0] = 0.125;
  bias_params->scale->data[1] = 0.0625;
  bias_params->scale->data[2] = 0.125;
  bias_params->zero_point = TfLiteIntArrayCreate(3);
  bias_params->zero_point->data[0] = 11;
  bias_params->zero_point->data[1] = 12;
  bias_params->zero_point->data[2] = 15;
  bias.quantization.params = reinterpret_cast<void*>(bias_params);

  // Create output.
  TfLiteTensor output = {};
  output.type = kTfLiteInt8;
  output.allocation_type = kTfLiteArenaRw;
  output.dims = nullptr;
  TfLiteQuantizationParams output_quant = {0.5, -128};
  output.params = output_quant;
  output.quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 0.5;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = -128;
  output.quantization.params = reinterpret_cast<void*>(output_params);

  // Create call parameters.
  TfLiteContext context;
  int32_t multiplier;
  int shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  std::vector<int32_t> per_channel_multiplier(3);
  std::vector<int> per_channel_shift(3);

  // Call and verify results for per channel case.
  EXPECT_EQ(
      kTfLiteOk,
      PopulateConvolutionQuantizationParams(
          &context, &input, &filter, &bias, &output, kTfLiteActRelu,
          &multiplier, &shift, &output_activation_min, &output_activation_max,
          per_channel_multiplier.data(), per_channel_shift.data()));
  EXPECT_THAT(per_channel_multiplier,
              ::testing::ElementsAre(1073741824, 1073741824, 1073741824));
  EXPECT_THAT(per_channel_shift, ::testing::ElementsAre(-1, -2, -1));

  // Release.
  TfLiteTensorFree(&input);
  TfLiteTensorFree(&filter);
  TfLiteTensorFree(&bias);
  TfLiteTensorFree(&output);
}

TEST_F(KernelUtilTest, CheckAndPopulateShift) {
  // Create input of type kTfLiteUInt8.
  TfLiteTensor input = {};
  input.type = kTfLiteUInt8;
  input.allocation_type = kTfLiteArenaRw;
  input.dims = TfLiteIntArrayCreate(1);
  input.dims->data[0] = 2;
  TfLiteQuantizationParams input_quant = {0.5, 5};
  input.params = input_quant;
  input.quantization.type = kTfLiteAffineQuantization;
  auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  input_params->scale = TfLiteFloatArrayCreate(1);
  input_params->scale->data[0] = 0.5;
  input_params->zero_point = TfLiteIntArrayCreate(1);
  input_params->zero_point->data[0] = 5;
  input.quantization.params = reinterpret_cast<void*>(input_params);

  // Create filter of type kTfLiteUInt8.
  TfLiteTensor filter = {};
  filter.type = kTfLiteUInt8;
  filter.allocation_type = kTfLiteArenaRw;
  filter.dims = TfLiteIntArrayCreate(4);
  filter.dims->data[0] = 3;
  filter.dims->data[1] = 4;
  filter.dims->data[2] = 5;
  filter.dims->data[3] = 6;
  TfLiteQuantizationParams filter_quant = {0.25, 0};
  filter.params = filter_quant;
  filter.quantization.type = kTfLiteAffineQuantization;
  auto* filter_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  // Create scale of size one.
  filter_params->scale = TfLiteFloatArrayCreate(1);
  filter_params->scale->data[0] = 0.25;
  filter_params->zero_point = TfLiteIntArrayCreate(1);
  filter_params->zero_point->data[0] = 0;
  filter_params->quantized_dimension = 0;
  filter.quantization.params = reinterpret_cast<void*>(filter_params);

  // Create bias for kTfLiteUInt8.
  TfLiteTensor bias = {};
  bias.type = kTfLiteUInt8;
  bias.allocation_type = kTfLiteArenaRw;
  bias.dims = TfLiteIntArrayCreate(4);
  TfLiteQuantizationParams bias_quant = {0.125, 9};
  bias.params = bias_quant;
  bias.quantization.type = kTfLiteAffineQuantization;
  auto* bias_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  bias_params->scale = TfLiteFloatArrayCreate(3);
  bias_params->scale->data[0] = 0.125;
  bias_params->scale->data[1] = 0.0625;
  bias_params->scale->data[2] = 0.125;
  bias_params->zero_point = TfLiteIntArrayCreate(3);
  bias_params->zero_point->data[0] = 11;
  bias_params->zero_point->data[1] = 12;
  bias_params->zero_point->data[2] = 15;
  bias.quantization.params = reinterpret_cast<void*>(bias_params);

  // Create output for kTfLiteUInt8.
  TfLiteTensor output = {};
  output.type = kTfLiteUInt8;
  output.allocation_type = kTfLiteArenaRw;
  output.dims = nullptr;
  TfLiteQuantizationParams output_quant = {0.5, 128};
  output.params = output_quant;
  output.quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 0.5;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = 128;
  output.quantization.params = reinterpret_cast<void*>(output_params);

  // Create call parameters.
  TfLiteContext context;
  int32_t multiplier;
  int shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  std::vector<int32_t> per_channel_multiplier(1);
  std::vector<int> per_channel_shift(1);

  // Call and verify results for per channel case.
  EXPECT_EQ(
      kTfLiteOk,
      PopulateConvolutionQuantizationParams(
          &context, &input, &filter, &bias, &output, kTfLiteActRelu,
          &multiplier, &shift, &output_activation_min, &output_activation_max,
          per_channel_multiplier.data(), per_channel_shift.data()));
  // Since the filter scale has a size of one i.e number of channels is one in
  // our TC we expect 1073741824 as output
  EXPECT_THAT(per_channel_multiplier, ::testing::ElementsAre(1073741824));
  EXPECT_THAT(per_channel_shift, ::testing::ElementsAre(-1));
  EXPECT_EQ(shift, 1);
  EXPECT_EQ(multiplier, 1073741824);

  // Release.
  TfLiteTensorFree(&input);
  TfLiteTensorFree(&filter);
  TfLiteTensorFree(&bias);
  TfLiteTensorFree(&output);
}

#ifndef __APPLE__  // Some Apple toolchains don't support std::ldexp
TEST_F(KernelUtilTest, CheckAndPopulateZeroValue) {
  // Create input.
  TfLiteTensor input = {};
  input.type = kTfLiteInt8;
  input.allocation_type = kTfLiteArenaRw;
  input.dims = TfLiteIntArrayCreate(1);
  input.dims->data[0] = 2;
  TfLiteQuantizationParams input_quant = {1, 5};
  input.params = input_quant;
  input.quantization.type = kTfLiteAffineQuantization;
  auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  input_params->scale = TfLiteFloatArrayCreate(1);
  input_params->scale->data[0] = 1;
  input_params->zero_point = TfLiteIntArrayCreate(1);
  input_params->zero_point->data[0] = 5;
  input.quantization.params = reinterpret_cast<void*>(input_params);

  // Create filter.
  TfLiteTensor filter = {};
  filter.type = kTfLiteInt8;
  filter.allocation_type = kTfLiteArenaRw;
  filter.dims = TfLiteIntArrayCreate(4);
  filter.dims->data[0] = 3;
  filter.dims->data[1] = 4;
  filter.dims->data[2] = 5;
  filter.dims->data[3] = 6;
  TfLiteQuantizationParams filter_quant = {4.6566129e-10, 0};
  filter.params = filter_quant;
  filter.quantization.type = kTfLiteAffineQuantization;
  auto* filter_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  filter_params->scale = TfLiteFloatArrayCreate(3);
  filter_params->scale->data[0] = std::ldexp(1.0f, -31);
  filter_params->scale->data[1] = std::ldexp(1.0f, -32);
  filter_params->scale->data[2] = std::ldexp(1.0f, -33);
  filter_params->zero_point = TfLiteIntArrayCreate(3);
  filter_params->zero_point->data[0] = 0;
  filter_params->zero_point->data[1] = 0;
  filter_params->zero_point->data[2] = 0;
  filter_params->quantized_dimension = 0;
  filter.quantization.params = reinterpret_cast<void*>(filter_params);

  // Create bias.
  TfLiteTensor bias = {};
  bias.type = kTfLiteInt32;
  bias.allocation_type = kTfLiteArenaRw;
  bias.dims = TfLiteIntArrayCreate(4);
  TfLiteQuantizationParams bias_quant = {4.6566129e-10, 9};
  bias.params = bias_quant;
  bias.quantization.type = kTfLiteAffineQuantization;
  auto* bias_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  bias_params->scale = TfLiteFloatArrayCreate(3);
  bias_params->scale->data[0] = std::ldexp(1.0f, -31);
  bias_params->scale->data[1] = std::ldexp(1.0f, -32);
  bias_params->scale->data[2] = std::ldexp(1.0f, -33);
  bias_params->zero_point = TfLiteIntArrayCreate(3);
  bias_params->zero_point->data[0] = 11;
  bias_params->zero_point->data[1] = 12;
  bias_params->zero_point->data[2] = 15;
  bias.quantization.params = reinterpret_cast<void*>(bias_params);

  // Create output.
  TfLiteTensor output = {};
  output.type = kTfLiteInt8;
  output.allocation_type = kTfLiteArenaRw;
  output.dims = nullptr;
  TfLiteQuantizationParams output_quant = {1, -128};
  output.params = output_quant;
  output.quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 1;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = -128;
  output.quantization.params = reinterpret_cast<void*>(output_params);

  // Create call parameters.
  TfLiteContext context;
  int32_t multiplier;
  int shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  std::vector<int32_t> per_channel_multiplier(3);
  std::vector<int> per_channel_shift(3);

  // Call and verify results for per channel case.
  EXPECT_EQ(
      kTfLiteOk,
      PopulateConvolutionQuantizationParams(
          &context, &input, &filter, &bias, &output, kTfLiteActRelu,
          &multiplier, &shift, &output_activation_min, &output_activation_max,
          per_channel_multiplier.data(), per_channel_shift.data()));
  EXPECT_THAT(per_channel_multiplier,
              ::testing::ElementsAre(1073741824, 1073741824, 0));
  EXPECT_THAT(per_channel_shift, ::testing::ElementsAre(-30, -31, 0));

  // Release.
  TfLiteTensorFree(&input);
  TfLiteTensorFree(&filter);
  TfLiteTensorFree(&bias);
  TfLiteTensorFree(&output);
}
#endif

TEST_F(KernelUtilTest, CheckAndPopulateUint8) {
  // Create input.
  TfLiteTensor input = {};
  input.type = kTfLiteUInt8;
  input.allocation_type = kTfLiteArenaRw;
  input.dims = TfLiteIntArrayCreate(1);
  input.dims->data[0] = 2;
  TfLiteQuantizationParams input_quant = {1, 5};
  input.params = input_quant;
  input.quantization.type = kTfLiteAffineQuantization;
  auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  input_params->scale = TfLiteFloatArrayCreate(1);
  input_params->scale->data[0] = 1;
  input_params->zero_point = TfLiteIntArrayCreate(1);
  input_params->zero_point->data[0] = 5;
  input.quantization.params = reinterpret_cast<void*>(input_params);

  // Create filter.
  TfLiteTensor filter = {};
  filter.type = kTfLiteUInt8;
  filter.allocation_type = kTfLiteArenaRw;
  filter.dims = TfLiteIntArrayCreate(4);
  filter.dims->data[0] = 3;
  filter.dims->data[1] = 4;
  filter.dims->data[2] = 5;
  filter.dims->data[3] = 6;
  TfLiteQuantizationParams filter_quant = {4.6566129e-10, 0};
  filter.params = filter_quant;
  filter.quantization.type = kTfLiteAffineQuantization;
  auto* filter_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  filter_params->scale = TfLiteFloatArrayCreate(1);
  int32_t two_pow_neg_31 = 0x30000000;  // 2^-31 so shift = -30.
  filter_params->scale->data[0] = *reinterpret_cast<float*>(&two_pow_neg_31);
  filter_params->zero_point = TfLiteIntArrayCreate(1);
  filter_params->zero_point->data[0] = 0;
  filter_params->quantized_dimension = 0;
  filter.quantization.params = reinterpret_cast<void*>(filter_params);

  // Create bias.
  TfLiteTensor bias = {};
  bias.type = kTfLiteInt32;
  bias.allocation_type = kTfLiteArenaRw;
  bias.dims = TfLiteIntArrayCreate(4);
  TfLiteQuantizationParams bias_quant = {4.6566129e-10, 9};
  bias.params = bias_quant;
  bias.quantization.type = kTfLiteAffineQuantization;
  auto* bias_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  bias_params->scale = TfLiteFloatArrayCreate(1);
  bias_params->scale->data[0] = 4.6566129e-10;  // 2^-31
  bias_params->zero_point = TfLiteIntArrayCreate(1);
  bias_params->zero_point->data[0] = 11;
  bias.quantization.params = reinterpret_cast<void*>(bias_params);

  // Create output.
  TfLiteTensor output = {};
  output.type = kTfLiteUInt8;
  output.allocation_type = kTfLiteArenaRw;
  output.dims = nullptr;
  TfLiteQuantizationParams output_quant = {1, -128};
  output.params = output_quant;
  output.quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 1;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = -128;
  output.quantization.params = reinterpret_cast<void*>(output_params);

  // Create call parameters.
  TfLiteContext context;
  int32_t multiplier;
  int shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  std::vector<int32_t> per_channel_multiplier(1);
  std::vector<int> per_channel_shift(1);

  // Call and verify results for per channel case.
  EXPECT_EQ(
      kTfLiteOk,
      PopulateConvolutionQuantizationParams(
          &context, &input, &filter, &bias, &output, kTfLiteActRelu,
          &multiplier, &shift, &output_activation_min, &output_activation_max,
          per_channel_multiplier.data(), per_channel_shift.data()));
  EXPECT_THAT(per_channel_multiplier, ::testing::ElementsAre(1073741824));
  EXPECT_THAT(per_channel_shift, ::testing::ElementsAre(-30));

  // Release.
  TfLiteTensorFree(&input);
  TfLiteTensorFree(&filter);
  TfLiteTensorFree(&bias);
  TfLiteTensorFree(&output);
}

TEST_F(KernelUtilTest, CheckAndPopulateWithoutBias) {
  // Create input.
  TfLiteTensor input = {};
  input.type = kTfLiteUInt8;
  input.allocation_type = kTfLiteArenaRw;
  input.dims = TfLiteIntArrayCreate(1);
  input.dims->data[0] = 2;
  TfLiteQuantizationParams input_quant = {1, 5};
  input.params = input_quant;
  input.quantization.type = kTfLiteAffineQuantization;
  auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  input_params->scale = TfLiteFloatArrayCreate(1);
  input_params->scale->data[0] = 1;
  input_params->zero_point = TfLiteIntArrayCreate(1);
  input_params->zero_point->data[0] = 5;
  input.quantization.params = reinterpret_cast<void*>(input_params);

  // Create filter.
  TfLiteTensor filter = {};
  filter.type = kTfLiteUInt8;
  filter.allocation_type = kTfLiteArenaRw;
  filter.dims = TfLiteIntArrayCreate(4);
  filter.dims->data[0] = 3;
  filter.dims->data[1] = 4;
  filter.dims->data[2] = 5;
  filter.dims->data[3] = 6;
  TfLiteQuantizationParams filter_quant = {4.6566129e-10, 0};
  filter.params = filter_quant;
  filter.quantization.type = kTfLiteAffineQuantization;
  auto* filter_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  filter_params->scale = TfLiteFloatArrayCreate(1);
  int32_t two_pow_neg_31 = 0x30000000;  // 2^-31 so shift = -30.
  filter_params->scale->data[0] = *reinterpret_cast<float*>(&two_pow_neg_31);
  filter_params->zero_point = TfLiteIntArrayCreate(1);
  filter_params->zero_point->data[0] = 0;
  filter_params->quantized_dimension = 0;
  filter.quantization.params = reinterpret_cast<void*>(filter_params);

  // Create output.
  TfLiteTensor output = {};
  output.type = kTfLiteUInt8;
  output.allocation_type = kTfLiteArenaRw;
  output.dims = nullptr;
  TfLiteQuantizationParams output_quant = {1, -128};
  output.params = output_quant;
  output.quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 1;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = -128;
  output.quantization.params = reinterpret_cast<void*>(output_params);

  // Create call parameters.
  TfLiteContext context;
  int32_t multiplier;
  int shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  std::vector<int32_t> per_channel_multiplier(1);
  std::vector<int> per_channel_shift(1);

  // Call and verify results for per channel case.
  EXPECT_EQ(
      kTfLiteOk,
      PopulateConvolutionQuantizationParams(
          &context, &input, &filter, nullptr, &output, kTfLiteActRelu,
          &multiplier, &shift, &output_activation_min, &output_activation_max,
          per_channel_multiplier.data(), per_channel_shift.data()));
  EXPECT_THAT(per_channel_multiplier, ::testing::ElementsAre(1073741824));
  EXPECT_THAT(per_channel_shift, ::testing::ElementsAre(-30));

  // Release.
  TfLiteTensorFree(&input);
  TfLiteTensorFree(&filter);
  TfLiteTensorFree(&output);
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
