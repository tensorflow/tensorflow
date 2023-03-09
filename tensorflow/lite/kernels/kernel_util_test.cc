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

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace {
using ::testing::ElementsAre;

struct TestContext : public TfLiteContext {
  string error;
};

void ReportError(TfLiteContext* context, const char* format, ...) {
  TestContext* c = static_cast<TestContext*>(context);
  const size_t kBufferSize = 1024;
  char temp_buffer[kBufferSize];

  va_list args;
  va_start(args, format);
  vsnprintf(temp_buffer, kBufferSize, format, args);
  va_end(args);

  c->error = temp_buffer;
}

class KernelUtilTest : public ::testing::Test {
 public:
  KernelUtilTest() {
    context_.ReportError = ReportError;

    memset(&tensor1_, 0, sizeof(TfLiteTensor));
    memset(&tensor2_, 0, sizeof(TfLiteTensor));
    memset(&tensor3_, 0, sizeof(TfLiteTensor));
    tensor1_.dims = nullptr;
    tensor2_.dims = nullptr;
    tensor3_.dims = nullptr;
    tensor1_.allocation_type = kTfLiteMmapRo;
    tensor2_.allocation_type = kTfLiteMmapRo;
    tensor3_.allocation_type = kTfLiteMmapRo;
  }
  ~KernelUtilTest() override {
    TfLiteTensorFree(&tensor1_);
    TfLiteTensorFree(&tensor2_);
    TfLiteTensorFree(&tensor3_);
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
  TestContext context_;
  TfLiteTensor tensor1_;
  TfLiteTensor tensor2_;
  TfLiteTensor tensor3_;
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
  EXPECT_EQ(context_.error,
            "Given shapes, [1,2] and [1,3], are not broadcastable.");
}

TEST_F(KernelUtilTest, BroadcastShapeIncompatibleDimWithZero) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 0});
  SetShape(&tensor2_, {1, 3});
  EXPECT_NE(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(context_.error,
            "Given shapes, [1,0] and [1,3], are not broadcastable.");
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
  EXPECT_THAT(GetShape(output), ElementsAre(1, 2));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {});
  SetShape(&tensor2_, {2});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(2));
  TfLiteIntArrayFree(output);
}

TEST_F(KernelUtilTest, BroadcastShapeDifferentSizes) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {3, 1, 1});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(3, 1, 2));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {1, 2, 3, 4});
  SetShape(&tensor2_, {1, 3, 1});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(1, 2, 3, 4));
  TfLiteIntArrayFree(output);
}

TEST_F(KernelUtilTest, BroadcastShapeWithZero) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {3, 0, 1});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(3, 0, 2));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {2, 1, 0});
  SetShape(&tensor2_, {1, 3, 1});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(2, 3, 0));
  TfLiteIntArrayFree(output);
}

TEST_F(KernelUtilTest, BroadcastShapeIncompatibleDimOnThreeTensors) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {1, 3});
  SetShape(&tensor3_, {1, 4});
  EXPECT_NE(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(context_.error,
            "Given shapes, [1,2], [1,3] and [1,4], are not broadcastable.");
}

TEST_F(KernelUtilTest, BroadcastShapeIncompatibleDimWithZeroOnThreeTensors) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 1});
  SetShape(&tensor2_, {1, 3});
  SetShape(&tensor3_, {1, 0});
  EXPECT_NE(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(context_.error,
            "Given shapes, [1,1], [1,3] and [1,0], are not broadcastable.");
}

TEST_F(KernelUtilTest, BroadcastShapeOnesOnThreeTensors) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 1});
  SetShape(&tensor2_, {1, 1});
  SetShape(&tensor3_, {1, 3});
  EXPECT_EQ(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {1, 1});
  SetShape(&tensor3_, {1, 1});
  EXPECT_EQ(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {1, 1});
  SetShape(&tensor2_, {1, 4});
  SetShape(&tensor3_, {1, 1});
  EXPECT_EQ(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  TfLiteIntArrayFree(output);
}

TEST_F(KernelUtilTest, BroadcastShapeScalarsOnThreeTensors) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {});
  SetShape(&tensor3_, {});
  EXPECT_EQ(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(1, 2));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {});
  SetShape(&tensor2_, {2});
  SetShape(&tensor3_, {});
  EXPECT_EQ(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(2));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {});
  SetShape(&tensor2_, {});
  SetShape(&tensor3_, {3, 2, 1});
  EXPECT_EQ(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(3, 2, 1));
  TfLiteIntArrayFree(output);
}

TEST_F(KernelUtilTest, BroadcastShapeDifferentSizesOnThreeTensors) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {3, 1, 1});
  SetShape(&tensor3_, {3, 1});
  EXPECT_EQ(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(3, 3, 2));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {3, 4});
  SetShape(&tensor2_, {1, 3, 1});
  SetShape(&tensor3_, {1, 2, 1, 1});
  EXPECT_EQ(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(1, 2, 3, 4));
  TfLiteIntArrayFree(output);
}

TEST_F(KernelUtilTest, BroadcastShapeWithZeroOnThreeTensors) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {3, 1, 1});
  SetShape(&tensor3_, {0, 1});
  EXPECT_EQ(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(3, 0, 2));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {1, 4});
  SetShape(&tensor2_, {1, 0, 1});
  SetShape(&tensor3_, {1, 2, 1, 1});
  EXPECT_EQ(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, &tensor1_, &tensor2_,
                                       &tensor3_, &output));
  EXPECT_THAT(GetShape(output), ElementsAre(1, 2, 0, 4));
  TfLiteIntArrayFree(output);
}

TEST_F(KernelUtilTest, GetShapeDebugString) {
  TfLiteIntArray* dims0 = TfLiteIntArrayCreate(0);
  EXPECT_EQ("[]", GetShapeDebugString(dims0));
  TfLiteIntArrayFree(dims0);

  TfLiteIntArray* dims1 = TfLiteIntArrayCreate(1);
  dims1->data[0] = 1;
  EXPECT_EQ("[1]", GetShapeDebugString(dims1));
  TfLiteIntArrayFree(dims1);

  TfLiteIntArray* dims2 = TfLiteIntArrayCreate(2);
  dims2->data[0] = 2;
  dims2->data[1] = 3;
  EXPECT_EQ("[2,3]", GetShapeDebugString(dims2));
  TfLiteIntArrayFree(dims2);

  TfLiteIntArray* dims3 = TfLiteIntArrayCreate(3);
  dims3->data[0] = 4;
  dims3->data[1] = 5;
  dims3->data[2] = 6;
  EXPECT_EQ("[4,5,6]", GetShapeDebugString(dims3));
  TfLiteIntArrayFree(dims3);
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
  int32_t multiplier;
  int shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  std::vector<int32_t> per_channel_multiplier(3);
  std::vector<int32_t> per_channel_shift(3);

  // Call and verify results for per channel case.
  EXPECT_EQ(
      kTfLiteOk,
      PopulateConvolutionQuantizationParams(
          &context_, &input, &filter, &bias, &output, kTfLiteActRelu,
          &multiplier, &shift, &output_activation_min, &output_activation_max,
          per_channel_multiplier.data(), per_channel_shift.data()));
  EXPECT_THAT(per_channel_multiplier,
              ElementsAre(1073741824, 1073741824, 1073741824));
  EXPECT_THAT(per_channel_shift, ElementsAre(-1, -2, -1));

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
          &context_, &input, &filter, &bias, &output, kTfLiteActRelu,
          &multiplier, &shift, &output_activation_min, &output_activation_max,
          per_channel_multiplier.data(), per_channel_shift.data(), 3));
  // Since the filter scale has a size of one but the number of channels is
  // three, in our TC we expect three 1073741824 as output
  EXPECT_THAT(per_channel_multiplier,
              ElementsAre(1073741824, 1073741824, 1073741824));
  EXPECT_THAT(per_channel_shift, ElementsAre(-1, -1, -1));
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
          &context_, &input, &filter, &bias, &output, kTfLiteActRelu,
          &multiplier, &shift, &output_activation_min, &output_activation_max,
          per_channel_multiplier.data(), per_channel_shift.data(), 3));
  EXPECT_THAT(per_channel_multiplier, ElementsAre(1073741824, 1073741824, 0));
  EXPECT_THAT(per_channel_shift, ElementsAre(-30, -31, 0));

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
          &context_, &input, &filter, &bias, &output, kTfLiteActRelu,
          &multiplier, &shift, &output_activation_min, &output_activation_max,
          per_channel_multiplier.data(), per_channel_shift.data(), 3));
  EXPECT_THAT(per_channel_multiplier,
              ElementsAre(1073741824, 1073741824, 1073741824));
  EXPECT_THAT(per_channel_shift, ElementsAre(-30, -30, -30));

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
          &context_, &input, &filter, nullptr, &output, kTfLiteActRelu,
          &multiplier, &shift, &output_activation_min, &output_activation_max,
          per_channel_multiplier.data(), per_channel_shift.data(), 3));
  EXPECT_THAT(per_channel_multiplier,
              ElementsAre(1073741824, 1073741824, 1073741824));
  EXPECT_THAT(per_channel_shift, ElementsAre(-30, -30, -30));

  // Release.
  TfLiteTensorFree(&input);
  TfLiteTensorFree(&filter);
  TfLiteTensorFree(&output);
}

TEST_F(KernelUtilTest, ActivationRangeQuantizedOverflow) {
  // Create output.
  TfLiteTensor output = {};
  output.type = kTfLiteUInt8;
  output.allocation_type = kTfLiteArenaRw;
  output.dims = nullptr;
  TfLiteQuantizationParams output_quant = {1e-10, -128};
  output.params = output_quant;
  output.quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 1;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = -128;
  output.quantization.params = reinterpret_cast<void*>(output_params);

  // For bounded activation, a too small scale value may cause overflow.
  // Make sure overflow error is handled gracefully.
  int32_t act_min, act_max;
  ASSERT_EQ(kTfLiteOk,
            CalculateActivationRangeQuantized(&context_, kTfLiteActRelu,
                                              &output, &act_min, &act_max));
  ASSERT_NE(kTfLiteOk,
            CalculateActivationRangeQuantized(&context_, kTfLiteActRelu6,
                                              &output, &act_min, &act_max));
  EXPECT_TRUE(absl::StrContains(
      context_.error, "no_integer_overflow_from_quantization was not true"));
  ASSERT_NE(kTfLiteOk,
            CalculateActivationRangeQuantized(&context_, kTfLiteActReluN1To1,
                                              &output, &act_min, &act_max));
  EXPECT_TRUE(absl::StrContains(
      context_.error, "no_integer_overflow_from_quantization was not true"));

  // Release.
  TfLiteTensorFree(&output);
}

TEST_F(KernelUtilTest, IsMobilePlatform) {
  // Note: This isn't meant to be exhaustive, as that would require replicating
  // the method's implementation, but it is a basic smoke check.
#if defined(__ANDROID__)
  EXPECT_TRUE(IsMobilePlatform());
#elif defined(__linux__)
  EXPECT_FALSE(IsMobilePlatform());
#elif defined(_WIN32)
  EXPECT_FALSE(IsMobilePlatform());
#endif
}

TEST_F(KernelUtilTest, HasUnspecifiedDimension) {
  TfLiteTensor tensor;
  TfLiteIntArray* shape_sig = TfLiteIntArrayCreate(3);
  shape_sig->data[0] = 1;
  shape_sig->data[1] = -1;
  shape_sig->data[2] = 3;
  tensor.dims_signature = shape_sig;

  EXPECT_TRUE(HasUnspecifiedDimension(&tensor));

  shape_sig->data[1] = 2;
  EXPECT_FALSE(HasUnspecifiedDimension(&tensor));

  TfLiteIntArrayFree(shape_sig);
}

}  // namespace
}  // namespace tflite
