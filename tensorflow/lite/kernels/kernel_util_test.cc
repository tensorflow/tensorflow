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
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/util.h"

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

class TestWithTfLiteContext : public ::testing::Test {
 public:
  TestWithTfLiteContext() { context_.ReportError = ReportError; }

  // `allocation_type` and `type` are not relavant for most of these tests,
  // so we provide a simpler wrapper to construct tensors.
  TensorUniquePtr BuildTfLiteTensorForTest(std::initializer_list<int> dims) {
    return BuildTfLiteTensor(kTfLiteInt32, dims, kTfLiteDynamic);
  }

 protected:
  TestContext context_;
};

class HaveSameShapeTest : public TestWithTfLiteContext {};

TEST_F(HaveSameShapeTest, NullPointerIsSameShape) {
  TensorUniquePtr t1 = BuildTfLiteTensor();
  t1->dims = nullptr;

  TensorUniquePtr t2 = BuildTfLiteTensor();
  t2->dims = nullptr;

  EXPECT_TRUE(HaveSameShapes(t1.get(), t2.get()));
}

TEST_F(HaveSameShapeTest, NotSameShapeFalse) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({2, 3});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({3});

  EXPECT_FALSE(HaveSameShapes(t1.get(), t2.get()));
}

TEST_F(HaveSameShapeTest, EmptyShapeEqualTrue) {
  TensorUniquePtr t1 = BuildTfLiteTensor();
  TensorUniquePtr t2 = BuildTfLiteTensor();

  EXPECT_TRUE(HaveSameShapes(t1.get(), t2.get()));
}

class BroadcastShapeTest : public TestWithTfLiteContext {};

TEST_F(BroadcastShapeTest, IncompatibleDimNullptr) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 2});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({1, 3});

  TfLiteIntArray* output = nullptr;

  EXPECT_NE(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, t1.get(), t2.get(), &output));
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(context_.error,
            "Given shapes, [1,2] and [1,3], are not broadcastable.");
}

TEST_F(BroadcastShapeTest, IncompatibleDimWithZeroNullptr) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 0});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({1, 3});

  TfLiteIntArray* output = nullptr;

  EXPECT_NE(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, t1.get(), t2.get(), &output));
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(context_.error,
            "Given shapes, [1,0] and [1,3], are not broadcastable.");
}

TEST_F(BroadcastShapeTest, BroadCastSecondDimension) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 1});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({1, 3});

  TfLiteIntArray* raw_output;
  auto status =
      CalculateShapeForBroadcast(&context_, t1.get(), t2.get(), &raw_output);
  ASSERT_EQ(kTfLiteOk, status);
  IntArrayUniquePtr output(raw_output);

  EXPECT_THAT(output.get(), DimsAre({1, 3}));
}

TEST_F(BroadcastShapeTest, ScalarAnd2dBroadcastsTo2d) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 2});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({});

  TfLiteIntArray* raw_output;
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, t1.get(), t2.get(),
                                                  &raw_output));

  IntArrayUniquePtr output(raw_output);
  EXPECT_THAT(output.get(), DimsAre({1, 2}));
}

TEST_F(BroadcastShapeTest, DifferentRankBroadcastsToHigherRank) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 2});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({3, 1, 2});

  TfLiteIntArray* raw_output;
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, t1.get(), t2.get(),
                                                  &raw_output));
  IntArrayUniquePtr output(raw_output);
  EXPECT_THAT(output.get(), DimsAre({3, 1, 2}));
}

TEST_F(BroadcastShapeTest, ZeroDimDifferentRankBroadcastsToHigherRank) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 2});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({3, 0, 2});

  TfLiteIntArray* raw_output;
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, t1.get(), t2.get(),
                                                  &raw_output));
  IntArrayUniquePtr output(raw_output);
  EXPECT_THAT(output.get(), DimsAre({3, 0, 2}));
}

TEST_F(BroadcastShapeTest, ZeroDimSameRankBroadcastsToHigherRank) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 2});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({3, 0, 1});

  TfLiteIntArray* raw_output;
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, t1.get(), t2.get(),
                                                  &raw_output));

  IntArrayUniquePtr output(raw_output);
  EXPECT_THAT(output.get(), DimsAre({3, 0, 2}));
}

TEST_F(BroadcastShapeTest, IncompatibleDimOnThreeTensorsNullptr) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 2});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({1, 3});
  TensorUniquePtr t3 = BuildTfLiteTensorForTest({1, 4});

  TfLiteIntArray* raw_output = nullptr;
  EXPECT_NE(kTfLiteOk, CalculateShapeForBroadcast(&context_, t1.get(), t2.get(),
                                                  t3.get(), &raw_output));
  EXPECT_EQ(raw_output, nullptr);
  EXPECT_EQ(context_.error,
            "Given shapes, [1,2], [1,3] and [1,4], are not broadcastable.");
}

TEST_F(BroadcastShapeTest, IncompatibleDimWithZeroOnThreeTensorsNullptr) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 1});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({1, 3});
  TensorUniquePtr t3 = BuildTfLiteTensorForTest({1, 0});

  TfLiteIntArray* raw_output = nullptr;
  EXPECT_NE(kTfLiteOk, CalculateShapeForBroadcast(&context_, t1.get(), t2.get(),
                                                  t3.get(), &raw_output));
  EXPECT_EQ(raw_output, nullptr);
  EXPECT_EQ(context_.error,
            "Given shapes, [1,1], [1,3] and [1,0], are not broadcastable.");
}

TEST_F(BroadcastShapeTest, ThreeTensorsBroadcastToLarger2ndDim) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 1});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({1, 1});
  TensorUniquePtr t3 = BuildTfLiteTensorForTest({1, 3});

  TfLiteIntArray* raw_output;
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, t1.get(), t2.get(),
                                                  t3.get(), &raw_output));
  IntArrayUniquePtr output(raw_output);
  EXPECT_THAT(output.get(), DimsAre({1, 3}));
}

TEST_F(BroadcastShapeTest, TwoScalarsBroadcastTo2d) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 2});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({});
  TensorUniquePtr t3 = BuildTfLiteTensorForTest({});

  TfLiteIntArray* raw_output;
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, t1.get(), t2.get(),
                                                  t3.get(), &raw_output));
  IntArrayUniquePtr output(raw_output);
  EXPECT_THAT(output.get(), DimsAre({1, 2}));
}

TEST_F(BroadcastShapeTest, DifferentSizesOnThreeTensorsBroadcastToLargerRank) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 2});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({3, 1, 1});
  TensorUniquePtr t3 = BuildTfLiteTensorForTest({3, 1});

  TfLiteIntArray* raw_output;
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, t1.get(), t2.get(),
                                                  t3.get(), &raw_output));
  IntArrayUniquePtr output(raw_output);
  EXPECT_THAT(output.get(), DimsAre({3, 3, 2}));
}

TEST_F(BroadcastShapeTest,
       DifferentSizesOnThreeTensors4dBroadcastToLargerRank) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({3, 4});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({1, 3, 1});
  TensorUniquePtr t3 = BuildTfLiteTensorForTest({1, 2, 1, 1});

  TfLiteIntArray* raw_output;
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, t1.get(), t2.get(),
                                                  t3.get(), &raw_output));
  IntArrayUniquePtr output(raw_output);
  EXPECT_THAT(output.get(), DimsAre({1, 2, 3, 4}));
}

TEST_F(BroadcastShapeTest, ZeroOnThreeTensorsBroadcastToLargerRank) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 2});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({3, 1, 1});
  TensorUniquePtr t3 = BuildTfLiteTensorForTest({0, 1});

  TfLiteIntArray* raw_output;
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, t1.get(), t2.get(),
                                                  t3.get(), &raw_output));
  IntArrayUniquePtr output(raw_output);
  EXPECT_THAT(output.get(), DimsAre({3, 0, 2}));
}

TEST(GetShapeDebugStringTest, GetShapeDebugString) {
  IntArrayUniquePtr dims0 = BuildTfLiteArray({});
  EXPECT_EQ("[]", GetShapeDebugString(dims0.get()));

  IntArrayUniquePtr dims1 = BuildTfLiteArray({1});
  dims1->data[0] = 1;
  EXPECT_EQ("[1]", GetShapeDebugString(dims1.get()));

  IntArrayUniquePtr dims2 = BuildTfLiteArray({2, 3});
  dims2->data[0] = 2;
  dims2->data[1] = 3;
  EXPECT_EQ("[2,3]", GetShapeDebugString(dims2.get()));

  IntArrayUniquePtr dims3 = BuildTfLiteArray({4, 5, 6});
  dims3->data[0] = 4;
  dims3->data[1] = 5;
  dims3->data[2] = 6;
  EXPECT_EQ("[4,5,6]", GetShapeDebugString(dims3.get()));
}

class QuantizationParamsTest : public TestWithTfLiteContext {};

TEST_F(QuantizationParamsTest, PerChannelConvolution) {
  // Create input.
  TensorUniquePtr input = BuildTfLiteTensor();
  input->type = kTfLiteInt8;
  input->allocation_type = kTfLiteArenaRw;
  input->dims = TfLiteIntArrayCreate(1);
  input->dims->data[0] = 2;
  TfLiteQuantizationParams input_quant = {0.5, 5};
  input->params = input_quant;
  input->quantization.type = kTfLiteAffineQuantization;
  auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  input_params->scale = TfLiteFloatArrayCreate(1);
  input_params->scale->data[0] = 0.5;
  input_params->zero_point = TfLiteIntArrayCreate(1);
  input_params->zero_point->data[0] = 5;
  input->quantization.params = reinterpret_cast<void*>(input_params);

  // Create filter.
  TensorUniquePtr filter = BuildTfLiteTensor();
  filter->type = kTfLiteInt8;
  filter->allocation_type = kTfLiteArenaRw;
  filter->dims = TfLiteIntArrayCreate(4);
  filter->dims->data[0] = 3;
  filter->dims->data[1] = 4;
  filter->dims->data[2] = 5;
  filter->dims->data[3] = 6;
  TfLiteQuantizationParams filter_quant = {0.25, 0};
  filter->params = filter_quant;
  filter->quantization.type = kTfLiteAffineQuantization;
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
  filter->quantization.params = reinterpret_cast<void*>(filter_params);

  // Create bias.
  TensorUniquePtr bias = BuildTfLiteTensor();
  bias->type = kTfLiteInt32;
  bias->allocation_type = kTfLiteArenaRw;
  bias->dims = TfLiteIntArrayCreate(4);
  TfLiteQuantizationParams bias_quant = {0.125, 9};
  bias->params = bias_quant;
  bias->quantization.type = kTfLiteAffineQuantization;
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
  bias->quantization.params = reinterpret_cast<void*>(bias_params);

  // Create output.
  TensorUniquePtr output = BuildTfLiteTensor();
  output->type = kTfLiteInt8;
  output->allocation_type = kTfLiteArenaRw;
  output->dims = nullptr;
  TfLiteQuantizationParams output_quant = {0.5, -128};
  output->params = output_quant;
  output->quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 0.5;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = -128;
  output->quantization.params = reinterpret_cast<void*>(output_params);

  // Create call parameters.
  int32_t multiplier;
  int shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  std::vector<int32_t> per_channel_multiplier(3);
  std::vector<int32_t> per_channel_shift(3);

  // Call and verify results for per channel case.
  auto status = PopulateConvolutionQuantizationParams(
      &context_, input.get(), filter.get(), bias.get(), output.get(),
      kTfLiteActRelu, &multiplier, &shift, &output_activation_min,
      &output_activation_max, per_channel_multiplier.data(),
      per_channel_shift.data());
  EXPECT_EQ(kTfLiteOk, status);
  EXPECT_THAT(per_channel_multiplier,
              ElementsAre(1073741824, 1073741824, 1073741824));
  EXPECT_THAT(per_channel_shift, ElementsAre(-1, -2, -1));
}

TEST_F(QuantizationParamsTest, CheckAndPopulateShift) {
  // Create input of type kTfLiteUInt8.

  TensorUniquePtr input = BuildTfLiteTensor();
  input->type = kTfLiteUInt8;
  input->allocation_type = kTfLiteArenaRw;
  input->dims = TfLiteIntArrayCreate(1);
  input->dims->data[0] = 2;
  TfLiteQuantizationParams input_quant = {0.5, 5};
  input->params = input_quant;
  input->quantization.type = kTfLiteAffineQuantization;
  auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  input_params->scale = TfLiteFloatArrayCreate(1);
  input_params->scale->data[0] = 0.5;
  input_params->zero_point = TfLiteIntArrayCreate(1);
  input_params->zero_point->data[0] = 5;
  input->quantization.params = reinterpret_cast<void*>(input_params);

  // Create filter of type kTfLiteUInt8.
  TensorUniquePtr filter = BuildTfLiteTensor();
  filter->type = kTfLiteUInt8;
  filter->allocation_type = kTfLiteArenaRw;
  filter->dims = TfLiteIntArrayCreate(4);
  filter->dims->data[0] = 3;
  filter->dims->data[1] = 4;
  filter->dims->data[2] = 5;
  filter->dims->data[3] = 6;
  TfLiteQuantizationParams filter_quant = {0.25, 0};
  filter->params = filter_quant;
  filter->quantization.type = kTfLiteAffineQuantization;
  auto* filter_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  // Create scale of size one.
  filter_params->scale = TfLiteFloatArrayCreate(1);
  filter_params->scale->data[0] = 0.25;
  filter_params->zero_point = TfLiteIntArrayCreate(1);
  filter_params->zero_point->data[0] = 0;
  filter_params->quantized_dimension = 0;
  filter->quantization.params = reinterpret_cast<void*>(filter_params);

  // Create bias for kTfLiteUInt8.
  TensorUniquePtr bias = BuildTfLiteTensor();
  bias->type = kTfLiteUInt8;
  bias->allocation_type = kTfLiteArenaRw;
  bias->dims = TfLiteIntArrayCreate(4);
  TfLiteQuantizationParams bias_quant = {0.125, 9};
  bias->params = bias_quant;
  bias->quantization.type = kTfLiteAffineQuantization;
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
  bias->quantization.params = reinterpret_cast<void*>(bias_params);

  // Create output for kTfLiteUInt8.
  TensorUniquePtr output = BuildTfLiteTensor();
  output->type = kTfLiteUInt8;
  output->allocation_type = kTfLiteArenaRw;
  output->dims = nullptr;
  TfLiteQuantizationParams output_quant = {0.5, 128};
  output->params = output_quant;
  output->quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 0.5;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = 128;
  output->quantization.params = reinterpret_cast<void*>(output_params);

  // Create call parameters.
  int32_t multiplier;
  int shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  std::vector<int32_t> per_channel_multiplier(3);
  std::vector<int> per_channel_shift(3);

  // Call and verify results for per channel case.
  EXPECT_EQ(kTfLiteOk,
            PopulateConvolutionQuantizationParams(
                &context_, input.get(), filter.get(), bias.get(), output.get(),
                kTfLiteActRelu, &multiplier, &shift, &output_activation_min,
                &output_activation_max, per_channel_multiplier.data(),
                per_channel_shift.data(), 3));
  // Since the filter scale has a size of one but the number of channels is
  // three, in our TC we expect three 1073741824 as output
  EXPECT_THAT(per_channel_multiplier,
              ElementsAre(1073741824, 1073741824, 1073741824));
  EXPECT_THAT(per_channel_shift, ElementsAre(-1, -1, -1));
  EXPECT_EQ(shift, 1);
  EXPECT_EQ(multiplier, 1073741824);
}

#ifndef __APPLE__  // Some Apple toolchains don't support std::ldexp
TEST_F(QuantizationParamsTest, CheckAndPopulateZeroValue) {
  // Create input.
  auto input = BuildTfLiteTensor();
  input->type = kTfLiteInt8;
  input->allocation_type = kTfLiteArenaRw;
  input->dims = TfLiteIntArrayCreate(1);
  input->dims->data[0] = 2;
  TfLiteQuantizationParams input_quant = {1, 5};
  input->params = input_quant;
  input->quantization.type = kTfLiteAffineQuantization;
  auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  input_params->scale = TfLiteFloatArrayCreate(1);
  input_params->scale->data[0] = 1;
  input_params->zero_point = TfLiteIntArrayCreate(1);
  input_params->zero_point->data[0] = 5;
  input->quantization.params = reinterpret_cast<void*>(input_params);

  // Create filter.
  auto filter = BuildTfLiteTensor();
  filter->type = kTfLiteInt8;
  filter->allocation_type = kTfLiteArenaRw;
  filter->dims = TfLiteIntArrayCreate(4);
  filter->dims->data[0] = 3;
  filter->dims->data[1] = 4;
  filter->dims->data[2] = 5;
  filter->dims->data[3] = 6;
  TfLiteQuantizationParams filter_quant = {4.6566129e-10, 0};
  filter->params = filter_quant;
  filter->quantization.type = kTfLiteAffineQuantization;
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
  filter->quantization.params = reinterpret_cast<void*>(filter_params);

  // Create bias.
  auto bias = BuildTfLiteTensor();
  bias->type = kTfLiteInt32;
  bias->allocation_type = kTfLiteArenaRw;
  bias->dims = TfLiteIntArrayCreate(4);
  TfLiteQuantizationParams bias_quant = {4.6566129e-10, 9};
  bias->params = bias_quant;
  bias->quantization.type = kTfLiteAffineQuantization;
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
  bias->quantization.params = reinterpret_cast<void*>(bias_params);

  // Create output.
  auto output = BuildTfLiteTensor();
  output->type = kTfLiteInt8;
  output->allocation_type = kTfLiteArenaRw;
  output->dims = nullptr;
  TfLiteQuantizationParams output_quant = {1, -128};
  output->params = output_quant;
  output->quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 1;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = -128;
  output->quantization.params = reinterpret_cast<void*>(output_params);

  // Create call parameters.
  int32_t multiplier;
  int shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  std::vector<int32_t> per_channel_multiplier(3);
  std::vector<int> per_channel_shift(3);

  // Call and verify results for per channel case.
  EXPECT_EQ(kTfLiteOk,
            PopulateConvolutionQuantizationParams(
                &context_, input.get(), filter.get(), bias.get(), output.get(),
                kTfLiteActRelu, &multiplier, &shift, &output_activation_min,
                &output_activation_max, per_channel_multiplier.data(),
                per_channel_shift.data(), 3));
  EXPECT_THAT(per_channel_multiplier, ElementsAre(1073741824, 1073741824, 0));
  EXPECT_THAT(per_channel_shift, ElementsAre(-30, -31, 0));
}
#endif

TEST_F(QuantizationParamsTest, CheckAndPopulateUint8) {
  // Create input.
  auto input = BuildTfLiteTensor();
  input->type = kTfLiteUInt8;
  input->allocation_type = kTfLiteArenaRw;
  input->dims = TfLiteIntArrayCreate(1);
  input->dims->data[0] = 2;
  TfLiteQuantizationParams input_quant = {1, 5};
  input->params = input_quant;
  input->quantization.type = kTfLiteAffineQuantization;
  auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  input_params->scale = TfLiteFloatArrayCreate(1);
  input_params->scale->data[0] = 1;
  input_params->zero_point = TfLiteIntArrayCreate(1);
  input_params->zero_point->data[0] = 5;
  input->quantization.params = reinterpret_cast<void*>(input_params);

  // Create filter.
  auto filter = BuildTfLiteTensor();
  filter->type = kTfLiteUInt8;
  filter->allocation_type = kTfLiteArenaRw;
  filter->dims = TfLiteIntArrayCreate(4);
  filter->dims->data[0] = 3;
  filter->dims->data[1] = 4;
  filter->dims->data[2] = 5;
  filter->dims->data[3] = 6;
  TfLiteQuantizationParams filter_quant = {4.6566129e-10, 0};
  filter->params = filter_quant;
  filter->quantization.type = kTfLiteAffineQuantization;
  auto* filter_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  filter_params->scale = TfLiteFloatArrayCreate(1);
  int32_t two_pow_neg_31 = 0x30000000;  // 2^-31 so shift = -30.
  filter_params->scale->data[0] = *reinterpret_cast<float*>(&two_pow_neg_31);
  filter_params->zero_point = TfLiteIntArrayCreate(1);
  filter_params->zero_point->data[0] = 0;
  filter_params->quantized_dimension = 0;
  filter->quantization.params = reinterpret_cast<void*>(filter_params);

  // Create bias.
  auto bias = BuildTfLiteTensor();
  bias->type = kTfLiteInt32;
  bias->allocation_type = kTfLiteArenaRw;
  bias->dims = TfLiteIntArrayCreate(4);
  TfLiteQuantizationParams bias_quant = {4.6566129e-10, 9};
  bias->params = bias_quant;
  bias->quantization.type = kTfLiteAffineQuantization;
  auto* bias_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  bias_params->scale = TfLiteFloatArrayCreate(1);
  bias_params->scale->data[0] = 4.6566129e-10;  // 2^-31
  bias_params->zero_point = TfLiteIntArrayCreate(1);
  bias_params->zero_point->data[0] = 11;
  bias->quantization.params = reinterpret_cast<void*>(bias_params);

  // Create output.
  auto output = BuildTfLiteTensor();
  output->type = kTfLiteUInt8;
  output->allocation_type = kTfLiteArenaRw;
  output->dims = nullptr;
  TfLiteQuantizationParams output_quant = {1, -128};
  output->params = output_quant;
  output->quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 1;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = -128;
  output->quantization.params = reinterpret_cast<void*>(output_params);

  // Create call parameters.
  int32_t multiplier;
  int shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  std::vector<int32_t> per_channel_multiplier(3);
  std::vector<int> per_channel_shift(3);

  // Call and verify results for per channel case.
  EXPECT_EQ(kTfLiteOk,
            PopulateConvolutionQuantizationParams(
                &context_, input.get(), filter.get(), bias.get(), output.get(),
                kTfLiteActRelu, &multiplier, &shift, &output_activation_min,
                &output_activation_max, per_channel_multiplier.data(),
                per_channel_shift.data(), 3));
  EXPECT_THAT(per_channel_multiplier,
              ElementsAre(1073741824, 1073741824, 1073741824));
  EXPECT_THAT(per_channel_shift, ElementsAre(-30, -30, -30));
}

TEST_F(QuantizationParamsTest, CheckAndPopulateWithoutBias) {
  // Create input.
  auto input = BuildTfLiteTensor();
  input->type = kTfLiteUInt8;
  input->allocation_type = kTfLiteArenaRw;
  input->dims = TfLiteIntArrayCreate(1);
  input->dims->data[0] = 2;
  TfLiteQuantizationParams input_quant = {1, 5};
  input->params = input_quant;
  input->quantization.type = kTfLiteAffineQuantization;
  auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  input_params->scale = TfLiteFloatArrayCreate(1);
  input_params->scale->data[0] = 1;
  input_params->zero_point = TfLiteIntArrayCreate(1);
  input_params->zero_point->data[0] = 5;
  input->quantization.params = reinterpret_cast<void*>(input_params);

  // Create filter.
  auto filter = BuildTfLiteTensor();
  filter->type = kTfLiteUInt8;
  filter->allocation_type = kTfLiteArenaRw;
  filter->dims = TfLiteIntArrayCreate(4);
  filter->dims->data[0] = 3;
  filter->dims->data[1] = 4;
  filter->dims->data[2] = 5;
  filter->dims->data[3] = 6;
  TfLiteQuantizationParams filter_quant = {4.6566129e-10, 0};
  filter->params = filter_quant;
  filter->quantization.type = kTfLiteAffineQuantization;
  auto* filter_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  filter_params->scale = TfLiteFloatArrayCreate(1);
  int32_t two_pow_neg_31 = 0x30000000;  // 2^-31 so shift = -30.
  filter_params->scale->data[0] = *reinterpret_cast<float*>(&two_pow_neg_31);
  filter_params->zero_point = TfLiteIntArrayCreate(1);
  filter_params->zero_point->data[0] = 0;
  filter_params->quantized_dimension = 0;
  filter->quantization.params = reinterpret_cast<void*>(filter_params);

  // Create output.
  auto output = BuildTfLiteTensor();
  output->type = kTfLiteUInt8;
  output->allocation_type = kTfLiteArenaRw;
  output->dims = nullptr;
  TfLiteQuantizationParams output_quant = {1, -128};
  output->params = output_quant;
  output->quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 1;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = -128;
  output->quantization.params = reinterpret_cast<void*>(output_params);

  // Create call parameters.
  int32_t multiplier;
  int shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  std::vector<int32_t> per_channel_multiplier(3);
  std::vector<int> per_channel_shift(3);

  // Call and verify results for per channel case.
  EXPECT_EQ(kTfLiteOk,
            PopulateConvolutionQuantizationParams(
                &context_, input.get(), filter.get(), nullptr, output.get(),
                kTfLiteActRelu, &multiplier, &shift, &output_activation_min,
                &output_activation_max, per_channel_multiplier.data(),
                per_channel_shift.data(), 3));
  EXPECT_THAT(per_channel_multiplier,
              ElementsAre(1073741824, 1073741824, 1073741824));
  EXPECT_THAT(per_channel_shift, ElementsAre(-30, -30, -30));
}

TEST_F(QuantizationParamsTest, ActivationRangeQuantizedOverflow) {
  // Create output.
  auto output = BuildTfLiteTensor();
  output->type = kTfLiteUInt8;
  output->allocation_type = kTfLiteArenaRw;
  output->dims = nullptr;
  TfLiteQuantizationParams output_quant = {1e-10, -128};
  output->params = output_quant;
  output->quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 1;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = -128;
  output->quantization.params = reinterpret_cast<void*>(output_params);

  // For bounded activation, a too small scale value may cause overflow.
  // Make sure overflow error is handled gracefully.
  int32_t act_min, act_max;
  ASSERT_EQ(kTfLiteOk,
            CalculateActivationRangeQuantized(
                &context_, kTfLiteActRelu, output.get(), &act_min, &act_max));
  ASSERT_NE(kTfLiteOk,
            CalculateActivationRangeQuantized(
                &context_, kTfLiteActRelu6, output.get(), &act_min, &act_max));
  EXPECT_TRUE(absl::StrContains(
      context_.error, "no_integer_overflow_from_quantization was not true"));
  ASSERT_NE(kTfLiteOk, CalculateActivationRangeQuantized(
                           &context_, kTfLiteActReluN1To1, output.get(),
                           &act_min, &act_max));
  EXPECT_TRUE(absl::StrContains(
      context_.error, "no_integer_overflow_from_quantization was not true"));
}

TEST_F(QuantizationParamsTest, IsMobilePlatform) {
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

TEST(HasUnspecifiedDimensions, ReturnsTrueIfADimIsMinusOne) {
  auto tensor = BuildTfLiteTensor(kTfLiteInt32, {1, 1, 3}, kTfLiteDynamic);
  tensor->dims_signature = ConvertVectorToTfLiteIntArray({1, -1, 3});
  EXPECT_TRUE(HasUnspecifiedDimension(tensor.get()));
}

TEST(HasUnspecifiedDimensions, ReturnsFalseIfAllPostiveDims) {
  auto tensor = BuildTfLiteTensor(kTfLiteInt32, {1, 1, 3}, kTfLiteDynamic);
  tensor->dims_signature = ConvertVectorToTfLiteIntArray({1, 1, 3});
  EXPECT_FALSE(HasUnspecifiedDimension(tensor.get()));
}

// Sets up a TFLite context and default values to initialize/resize test
// tensors.
class SetTensorAllocationTypeTest : public testing::Test {
 public:
  SetTensorAllocationTypeTest() {
    tensor_->type = kTfLiteInt32;
    tensor_->allocation_type = kTfLiteDynamic;
  }

 protected:
  Interpreter interpreter_;
  TfLiteContext& context_ = *interpreter_.primary_subgraph().context();
  IntArrayUniquePtr dims_ = BuildTfLiteArray({2, 3, 4});
  TensorUniquePtr tensor_ = BuildTfLiteTensor();
};

TEST_F(SetTensorAllocationTypeTest,
       SetUnallocatedDynamicTensorToDynamicIsANoop) {
  tensor_->allocation_type = kTfLiteDynamic;
  SetTensorToDynamic(tensor_.get());
  EXPECT_EQ(tensor_->data.data, nullptr);
  EXPECT_EQ(tensor_->allocation_type, kTfLiteDynamic);
}

TEST_F(SetTensorAllocationTypeTest, SetAllocatedDynamicTensorToDynamicIsANoop) {
  tensor_->allocation_type = kTfLiteDynamic;
  ASSERT_EQ(context_.ResizeTensor(&context_, tensor_.get(), dims_.release()),
            kTfLiteOk);
  const void* const original_data = tensor_->data.data;
  SetTensorToDynamic(tensor_.get());
  EXPECT_EQ(tensor_->data.data, original_data);
  EXPECT_EQ(tensor_->allocation_type, kTfLiteDynamic);
}

TEST_F(SetTensorAllocationTypeTest,
       SetAllocatedPersistentRoTensorToDynamicFreesExistingTensorData) {
  tensor_->allocation_type = kTfLitePersistentRo;
  ASSERT_EQ(context_.ResizeTensor(&context_, tensor_.get(), dims_.release()),
            kTfLiteOk);

  // Leak checker will raise an error if data is not freed.
  SetTensorToDynamic(tensor_.get());
  EXPECT_EQ(tensor_->data.data, nullptr);
  EXPECT_EQ(tensor_->allocation_type, kTfLiteDynamic);
}

TEST_F(SetTensorAllocationTypeTest,
       SetUnallocatedPersistentRoTensorToPersistentRoIsANoop) {
  tensor_->allocation_type = kTfLitePersistentRo;
  SetTensorToPersistentRo(tensor_.get());
  EXPECT_EQ(tensor_->data.data, nullptr);
  EXPECT_EQ(tensor_->allocation_type, kTfLitePersistentRo);
}

TEST_F(SetTensorAllocationTypeTest,
       SetAllocatedPersistentRoTensorToPersistentRoIsANoop) {
  tensor_->allocation_type = kTfLitePersistentRo;
  ASSERT_EQ(context_.ResizeTensor(&context_, tensor_.get(), dims_.release()),
            kTfLiteOk);
  const void* const original_data = tensor_->data.data;
  SetTensorToPersistentRo(tensor_.get());
  EXPECT_EQ(tensor_->data.data, original_data);
  EXPECT_EQ(tensor_->allocation_type, kTfLitePersistentRo);
}

TEST_F(SetTensorAllocationTypeTest,
       SetAllocatedDynamicTensorToPersistentRoFreesExistingTensorData) {
  tensor_->allocation_type = kTfLiteDynamic;
  ASSERT_EQ(context_.ResizeTensor(&context_, tensor_.get(), dims_.release()),
            kTfLiteOk);

  // Leak checker will raise an error if data is not freed.
  SetTensorToPersistentRo(tensor_.get());
  EXPECT_EQ(tensor_->data.data, nullptr);
  EXPECT_EQ(tensor_->allocation_type, kTfLitePersistentRo);
}

}  // namespace
}  // namespace tflite
