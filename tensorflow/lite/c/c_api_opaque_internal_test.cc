/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/c/c_api_opaque_internal.h"

#include <memory>

#include <gtest/gtest.h>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

using tflite::FlatBufferModel;
using tflite::Interpreter;
using tflite::InterpreterBuilder;
using tflite::internal::CommonOpaqueConversionUtil;
using tflite::ops::builtin::BuiltinOpResolver;

TEST(ObtainRegistrationFromContext, ProducesValidResult) {
  BuiltinOpResolver op_resolver;
  std::unique_ptr<Interpreter> interpreter;
  std::unique_ptr<FlatBufferModel> model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);
  InterpreterBuilder builder(*model, op_resolver);
  ASSERT_EQ(builder(&interpreter), kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
  TfLiteContext* context = interpreter->primary_subgraph().context();
  const TfLiteRegistration* registration = tflite::ops::builtin::Register_ADD();

  TfLiteOperator* registration_external =
      CommonOpaqueConversionUtil::ObtainOperator(context, registration, 42);

  ASSERT_EQ(registration_external->builtin_code, kTfLiteBuiltinAdd);
  ASSERT_EQ(registration_external->version, registration->version);
  ASSERT_EQ(registration_external->custom_name, registration->custom_name);
  ASSERT_EQ(registration_external->node_index, 42);
}

TEST(ObtainRegistrationFromContext, CachingWorks) {
  BuiltinOpResolver op_resolver;
  std::unique_ptr<Interpreter> interpreter;
  std::unique_ptr<FlatBufferModel> model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);
  InterpreterBuilder builder(*model, op_resolver);
  ASSERT_EQ(builder(&interpreter), kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
  TfLiteContext* context = interpreter->primary_subgraph().context();
  const TfLiteRegistration* registration = tflite::ops::builtin::Register_ADD();

  // Call it twice, and verify that we get the same result back.
  TfLiteOperator* registration_external1 =
      CommonOpaqueConversionUtil::ObtainOperator(context, registration, 0);
  TfLiteOperator* registration_external2 =
      CommonOpaqueConversionUtil::ObtainOperator(context, registration, 1);
  ASSERT_EQ(registration_external1, registration_external2);
}
