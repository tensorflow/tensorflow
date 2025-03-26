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
#include "tensorflow/lite/tools/optimize/calibration/logging_op_resolver.h"

#include <string>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/optimize/calibration/calibration_common.h"

namespace tflite {
namespace optimize {
namespace calibration {
namespace {

TfLiteStatus ConvPrepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus ConvEval(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus AddPrepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus AddEval(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus CustomPrepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus CustomEval(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus WrappingInvoke(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TEST(LoggingOpResolverTest, KernelInvokesAreReplaced) {
  MutableOpResolver base_resolver;
  TfLiteRegistration conv_registration = {};
  conv_registration.prepare = ConvPrepare;
  conv_registration.invoke = ConvEval;

  base_resolver.AddBuiltin(BuiltinOperator_CONV_2D, &conv_registration);

  TfLiteRegistration add_registration = {};
  add_registration.prepare = AddPrepare;
  add_registration.invoke = AddEval;

  base_resolver.AddBuiltin(BuiltinOperator_ADD, &add_registration);
  BuiltinOpsSet ops_to_replace = {
      {BuiltinOperator_CONV_2D, /*version*/ 1},
      {BuiltinOperator_ADD, /*version*/ 1},
  };

  LoggingOpResolver resolver(ops_to_replace, CustomOpsSet(), base_resolver,
                             WrappingInvoke, /*error_reporter=*/nullptr);

  auto reg = resolver.FindOp(BuiltinOperator_CONV_2D, 1);

  EXPECT_EQ(reg->builtin_code, BuiltinOperator_CONV_2D);
  EXPECT_TRUE(reg->prepare == ConvPrepare);
  EXPECT_TRUE(reg->invoke == WrappingInvoke);

  reg = resolver.FindOp(BuiltinOperator_ADD, 1);

  EXPECT_EQ(reg->builtin_code, BuiltinOperator_ADD);
  EXPECT_TRUE(reg->prepare == AddPrepare);
  EXPECT_TRUE(reg->invoke == WrappingInvoke);
}

TEST(LoggingOpResolverTest, OriginalKernelInvokesAreRetained) {
  MutableOpResolver base_resolver;
  TfLiteRegistration conv_registration = {};
  conv_registration.prepare = ConvPrepare;
  conv_registration.invoke = ConvEval;

  base_resolver.AddBuiltin(BuiltinOperator_CONV_2D, &conv_registration);

  TfLiteRegistration add_registration = {};
  add_registration.prepare = AddPrepare;
  add_registration.invoke = AddEval;

  base_resolver.AddBuiltin(BuiltinOperator_ADD, &add_registration);
  BuiltinOpsSet ops_to_replace = {
      {BuiltinOperator_CONV_2D, /*version*/ 1},
      {BuiltinOperator_ADD, /*version*/ 1},
  };

  LoggingOpResolver resolver(ops_to_replace, CustomOpsSet(), base_resolver,
                             WrappingInvoke, /*error_reporter=*/nullptr);
  auto kernel_invoke =
      resolver.GetWrappedKernelInvoke(BuiltinOperator_CONV_2D, 1);
  EXPECT_TRUE(kernel_invoke == ConvEval);
  kernel_invoke = resolver.GetWrappedKernelInvoke(BuiltinOperator_ADD, 1);
  EXPECT_TRUE(kernel_invoke == AddEval);
}

TEST(LoggingOpResolverTest, OnlyOpsInReplacementSetAreReplaces) {
  MutableOpResolver base_resolver;
  TfLiteRegistration conv_registration = {};
  conv_registration.prepare = ConvPrepare;
  conv_registration.invoke = ConvEval;

  base_resolver.AddBuiltin(BuiltinOperator_CONV_2D, &conv_registration);

  TfLiteRegistration add_registration = {};
  add_registration.prepare = AddPrepare;
  add_registration.invoke = AddEval;

  base_resolver.AddBuiltin(BuiltinOperator_ADD, &add_registration);
  // Only replace conv2d
  BuiltinOpsSet ops_to_replace = {
      {BuiltinOperator_CONV_2D, /*version*/ 1},
  };

  LoggingOpResolver resolver(ops_to_replace, CustomOpsSet(), base_resolver,
                             WrappingInvoke, /*error_reporter=*/nullptr);
  auto reg = resolver.FindOp(BuiltinOperator_CONV_2D, 1);
  EXPECT_EQ(reg->builtin_code, BuiltinOperator_CONV_2D);
  EXPECT_TRUE(reg->prepare == ConvPrepare);
  EXPECT_TRUE(reg->invoke == WrappingInvoke);

  reg = resolver.FindOp(BuiltinOperator_ADD, 1);
  EXPECT_EQ(nullptr, reg);
}

TEST(LoggingOpResolverTest, CustomOps) {
  MutableOpResolver base_resolver;
  TfLiteRegistration custom_registration = {};
  custom_registration.prepare = CustomPrepare;
  custom_registration.invoke = CustomEval;

  std::string custom_op_name = "custom";
  base_resolver.AddCustom(custom_op_name.c_str(), &custom_registration);

  CustomOpsSet ops_to_replace = {
      {custom_op_name, /*version*/ 1},
  };

  LoggingOpResolver resolver(BuiltinOpsSet(), ops_to_replace, base_resolver,
                             WrappingInvoke, /*error_reporter=*/nullptr);

  auto reg = resolver.FindOp(custom_op_name.c_str(), 1);

  EXPECT_EQ(reg->builtin_code, BuiltinOperator_CUSTOM);
  EXPECT_EQ(reg->custom_name, custom_op_name.c_str());
  EXPECT_TRUE(reg->prepare == CustomPrepare);
  EXPECT_TRUE(reg->invoke == WrappingInvoke);
}

TEST(LoggingOpResolverTest, UnresolvedCustomOps) {
  // No custom op registration.
  MutableOpResolver base_resolver;

  std::string custom_op_name = "unresolved_custom_op";

  CustomOpsSet ops_to_replace = {
      {custom_op_name, /*version*/ 1},
  };

  // Expect no death.
  LoggingOpResolver(BuiltinOpsSet(), ops_to_replace, base_resolver,
                    WrappingInvoke, /*error_reporter=*/nullptr);
}

TEST(LoggingOpResolverTest, UnresolvedBuiltinOps) {
  // No builtin op registration.
  MutableOpResolver base_resolver;

  BuiltinOpsSet ops_to_replace = {
      {BuiltinOperator_CONV_2D, /*version*/ 1},
      {BuiltinOperator_ADD, /*version*/ 1},
  };

  // Expect no death.
  LoggingOpResolver resolver(ops_to_replace, CustomOpsSet(), base_resolver,
                             WrappingInvoke, /*error_reporter=*/nullptr);
}

TEST(LoggingOpResolverTest, FlexOps) {
  // No flex op registration.
  MutableOpResolver base_resolver;

  std::string custom_op_name = "FlexAdd";

  CustomOpsSet ops_to_replace = {
      {custom_op_name, /*version*/ 1},
  };

  LoggingOpResolver resolver(BuiltinOpsSet(), ops_to_replace, base_resolver,
                             WrappingInvoke, /*error_reporter=*/nullptr);

  auto reg = resolver.FindOp(custom_op_name.c_str(), 1);

  EXPECT_TRUE(!reg);
}

}  // namespace
}  // namespace calibration
}  // namespace optimize
}  // namespace tflite
