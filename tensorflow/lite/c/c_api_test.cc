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

#include "tensorflow/lite/c/c_api.h"

#include <stdarg.h>
#include <stdint.h>

#include <array>
#include <cmath>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/testing/util.h"

namespace {

TEST(CAPI, Version) { EXPECT_STRNE("", TfLiteVersion()); }

TEST(CApiSimple, Smoke) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsSetNumThreads(options, 2);

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);

  // The options/model can be deleted immediately after interpreter creation.
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);

  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterGetInputTensorCount(interpreter), 1);
  ASSERT_EQ(TfLiteInterpreterGetOutputTensorCount(interpreter), 1);

  std::array<int, 1> input_dims = {2};
  ASSERT_EQ(TfLiteInterpreterResizeInputTensor(
                interpreter, 0, input_dims.data(), input_dims.size()),
            kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);

  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
  ASSERT_NE(input_tensor, nullptr);
  EXPECT_EQ(TfLiteTensorType(input_tensor), kTfLiteFloat32);
  EXPECT_EQ(TfLiteTensorNumDims(input_tensor), 1);
  EXPECT_EQ(TfLiteTensorDim(input_tensor, 0), 2);
  EXPECT_EQ(TfLiteTensorByteSize(input_tensor), sizeof(float) * 2);
  EXPECT_NE(TfLiteTensorData(input_tensor), nullptr);
  EXPECT_STREQ(TfLiteTensorName(input_tensor), "input");

  TfLiteQuantizationParams input_params =
      TfLiteTensorQuantizationParams(input_tensor);
  EXPECT_EQ(input_params.scale, 0.f);
  EXPECT_EQ(input_params.zero_point, 0);

  std::array<float, 2> input = {1.f, 3.f};
  ASSERT_EQ(TfLiteTensorCopyFromBuffer(input_tensor, input.data(),
                                       input.size() * sizeof(float)),
            kTfLiteOk);

  ASSERT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  ASSERT_NE(output_tensor, nullptr);
  EXPECT_EQ(TfLiteTensorType(output_tensor), kTfLiteFloat32);
  EXPECT_EQ(TfLiteTensorNumDims(output_tensor), 1);
  EXPECT_EQ(TfLiteTensorDim(output_tensor, 0), 2);
  EXPECT_EQ(TfLiteTensorByteSize(output_tensor), sizeof(float) * 2);
  EXPECT_NE(TfLiteTensorData(output_tensor), nullptr);
  EXPECT_STREQ(TfLiteTensorName(output_tensor), "output");

  TfLiteQuantizationParams output_params =
      TfLiteTensorQuantizationParams(output_tensor);
  EXPECT_EQ(output_params.scale, 0.f);
  EXPECT_EQ(output_params.zero_point, 0);

  std::array<float, 2> output;
  ASSERT_EQ(TfLiteTensorCopyToBuffer(output_tensor, output.data(),
                                     output.size() * sizeof(float)),
            kTfLiteOk);
  EXPECT_EQ(output[0], 3.f);
  EXPECT_EQ(output[1], 9.f);

  TfLiteInterpreterDelete(interpreter);
}

TEST(CApiSimple, QuantizationParams) {
  TfLiteModel* model = TfLiteModelCreateFromFile(
      "tensorflow/lite/testdata/add_quantized.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, nullptr);
  ASSERT_NE(interpreter, nullptr);

  TfLiteModelDelete(model);

  const std::array<int, 1> input_dims = {2};
  ASSERT_EQ(TfLiteInterpreterResizeInputTensor(
                interpreter, 0, input_dims.data(), input_dims.size()),
            kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);

  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
  ASSERT_NE(input_tensor, nullptr);
  EXPECT_EQ(TfLiteTensorType(input_tensor), kTfLiteUInt8);
  EXPECT_EQ(TfLiteTensorNumDims(input_tensor), 1);
  EXPECT_EQ(TfLiteTensorDim(input_tensor, 0), 2);

  TfLiteQuantizationParams input_params =
      TfLiteTensorQuantizationParams(input_tensor);
  EXPECT_EQ(input_params.scale, 0.003922f);
  EXPECT_EQ(input_params.zero_point, 0);

  const std::array<uint8_t, 2> input = {1, 3};
  ASSERT_EQ(TfLiteTensorCopyFromBuffer(input_tensor, input.data(),
                                       input.size() * sizeof(uint8_t)),
            kTfLiteOk);

  ASSERT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  ASSERT_NE(output_tensor, nullptr);

  TfLiteQuantizationParams output_params =
      TfLiteTensorQuantizationParams(output_tensor);
  EXPECT_EQ(output_params.scale, 0.003922f);
  EXPECT_EQ(output_params.zero_point, 0);

  std::array<uint8_t, 2> output;
  ASSERT_EQ(TfLiteTensorCopyToBuffer(output_tensor, output.data(),
                                     output.size() * sizeof(uint8_t)),
            kTfLiteOk);
  EXPECT_EQ(output[0], 3);
  EXPECT_EQ(output[1], 9);

  const float dequantizedOutput0 =
      output_params.scale * (output[0] - output_params.zero_point);
  const float dequantizedOutput1 =
      output_params.scale * (output[1] - output_params.zero_point);
  EXPECT_EQ(dequantizedOutput0, 0.011766f);
  EXPECT_EQ(dequantizedOutput1, 0.035298f);

  TfLiteInterpreterDelete(interpreter);
}

TEST(CApiSimple, Delegate) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");

  // Create and install a delegate instance.
  bool delegate_prepared = false;
  TfLiteDelegate delegate = TfLiteDelegateCreate();
  delegate.data_ = &delegate_prepared;
  delegate.Prepare = [](TfLiteContext* context, TfLiteDelegate* delegate) {
    *static_cast<bool*>(delegate->data_) = true;
    return kTfLiteOk;
  };
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, &delegate);
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

  // The delegate should have been applied.
  EXPECT_TRUE(delegate_prepared);

  // Subsequent execution should behave properly (the delegate is a no-op).
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);
  TfLiteInterpreterDelete(interpreter);
}

TEST(CApiSimple, DelegateFails) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");

  // Create and install a delegate instance.
  TfLiteDelegate delegate = TfLiteDelegateCreate();
  delegate.Prepare = [](TfLiteContext* context, TfLiteDelegate* delegate) {
    return kTfLiteError;
  };
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, &delegate);
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

  // Interpreter creation should fail as delegate preparation failed.
  EXPECT_EQ(nullptr, interpreter);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
}

TEST(CApiSimple, ErrorReporter) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();

  // Install a custom error reporter into the interpreter by way of options.
  tflite::TestErrorReporter reporter;
  TfLiteInterpreterOptionsSetErrorReporter(
      options,
      [](void* user_data, const char* format, va_list args) {
        reinterpret_cast<tflite::TestErrorReporter*>(user_data)->Report(format,
                                                                        args);
      },
      &reporter);
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

  // The options/model can be deleted immediately after interpreter creation.
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);

  // Invoke the interpreter before tensor allocation.
  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteError);

  // The error should propagate to the custom error reporter.
  EXPECT_EQ(reporter.error_messages(),
            "Invoke called on model that is not ready.");
  EXPECT_EQ(reporter.num_calls(), 1);

  TfLiteInterpreterDelete(interpreter);
}

TEST(CApiSimple, ValidModel) {
  std::ifstream model_file("tensorflow/lite/testdata/add.bin");

  model_file.seekg(0, std::ios_base::end);
  std::vector<char> model_buffer(model_file.tellg());

  model_file.seekg(0, std::ios_base::beg);
  model_file.read(model_buffer.data(), model_buffer.size());

  TfLiteModel* model =
      TfLiteModelCreate(model_buffer.data(), model_buffer.size());
  ASSERT_NE(model, nullptr);
  TfLiteModelDelete(model);
}

TEST(CApiSimple, ValidModelFromFile) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);
  TfLiteModelDelete(model);
}

TEST(CApiSimple, InvalidModel) {
  std::vector<char> invalid_model(20, 'c');
  TfLiteModel* model =
      TfLiteModelCreate(invalid_model.data(), invalid_model.size());
  ASSERT_EQ(model, nullptr);
}

TEST(CApiSimple, InvalidModelFromFile) {
  TfLiteModel* model = TfLiteModelCreateFromFile("invalid/path/foo.tflite");
  ASSERT_EQ(model, nullptr);
}

struct SinhParams {
  bool use_cosh_instead = false;
};

void* FlexSinhInit(TfLiteOpaqueContext* context, const char* buffer,
                   size_t length) {
  auto sinh_params = new SinhParams;
  // The buffer that is passed into here is the custom_options
  // field from the flatbuffer (tensorflow/lite/schema/schema.fbs)
  // `Operator` for this node.
  // Typically it should be stored as a FlexBuffer, but for this test
  // we assume that it is just a string.
  if (std::string(buffer, length) == "use_cosh") {
    sinh_params->use_cosh_instead = true;
  }
  return sinh_params;
}

void FlexSinhFree(TfLiteOpaqueContext* context, void* data) {
  delete static_cast<SinhParams*>(data);
}

TfLiteStatus FlexSinhPrepare(TfLiteOpaqueContext* context,
                             TfLiteOpaqueNode* node) {
  return kTfLiteOk;
}

TfLiteStatus FlexSinhEval(TfLiteOpaqueContext* context,
                          TfLiteOpaqueNode* node) {
  auto sinh_params =
      static_cast<SinhParams*>(TfLiteOpaqueNodeGetUserData(node));
  const TfLiteOpaqueTensor* input = TfLiteOpaqueNodeGetInput(context, node, 0);
  size_t input_bytes = TfLiteOpaqueTensorByteSize(input);
  void* data_ptr = TfLiteOpaqueTensorData(input);
  float input_value;
  memcpy(&input_value, data_ptr, input_bytes);

  TfLiteOpaqueTensor* output = TfLiteOpaqueNodeGetOutput(context, node, 0);
  float output_value = sinh_params->use_cosh_instead ? std::cosh(input_value)
                                                     : std::sinh(input_value);
  TfLiteOpaqueTensorCopyFromBuffer(output, &output_value, sizeof(output_value));
  return kTfLiteOk;
}

TEST(CApiSimple, CustomOpSupport) {
  TfLiteModel* model = TfLiteModelCreateFromFile(
      "tensorflow/lite/testdata/custom_sinh.bin");
  ASSERT_NE(model, nullptr);

  TfLiteRegistrationExternal* reg = TfLiteRegistrationExternalCreate("Sinh", 1);
  TfLiteRegistrationExternalSetInit(reg, &FlexSinhInit);
  TfLiteRegistrationExternalSetFree(reg, &FlexSinhFree);
  TfLiteRegistrationExternalSetPrepare(reg, &FlexSinhPrepare);
  TfLiteRegistrationExternalSetInvoke(reg, &FlexSinhEval);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddRegistrationExternal(options, reg);

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

  TfLiteInterpreterOptionsDelete(options);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
  float input_value = 1.0f;
  TfLiteTensorCopyFromBuffer(input_tensor, &input_value, sizeof(float));

  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  float output_value;
  TfLiteTensorCopyToBuffer(output_tensor, &output_value, sizeof(float));
  EXPECT_EQ(output_value, std::sinh(1.0f));

  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
  TfLiteRegistrationExternalDelete(reg);
}

const TfLiteRegistration* find_builtin_op_add(void* user_data,
                                              TfLiteBuiltinOperator op,
                                              int version) {
  static TfLiteRegistration registration{/*init=*/nullptr,
                                         /*free=*/nullptr,
                                         /*prepare=*/nullptr,
                                         /*invoke=*/nullptr,
                                         /*profiling_string=*/nullptr,
                                         /*builtin_code=*/kTfLiteBuiltinAdd,
                                         /*custom_name=*/nullptr,
                                         /*version=*/1};
  if (op == kTfLiteBuiltinAdd && version == 1) {
    return &registration;
  }
  return nullptr;
}

const TfLiteRegistration* find_custom_op_sinh(void* user_data, const char* op,
                                              int version) {
  static TfLiteRegistration registration{/*init=*/nullptr,
                                         /*free=*/nullptr,
                                         /*prepare=*/nullptr,
                                         /*invoke=*/nullptr,
                                         /*profiling_string=*/nullptr,
                                         /*builtin_code=*/kTfLiteBuiltinCustom,
                                         /*custom_name=*/"Sinh",
                                         /*version=*/1};
  if (strcmp(op, "Sinh") == 0 && version == 1) {
    return &registration;
  }
  return nullptr;
}

TEST(CApiSimple, CallbackOpResolver) {
  tflite::internal::CallbackOpResolver resolver;
  struct TfLiteOpResolverCallbacks callbacks {};
  callbacks.find_builtin_op = find_builtin_op_add;
  callbacks.find_custom_op = find_custom_op_sinh;

  resolver.SetCallbacks(callbacks);
  auto reg_add = resolver.FindOp(
      static_cast<::tflite::BuiltinOperator>(kTfLiteBuiltinAdd), 1);
  ASSERT_NE(reg_add, nullptr);
  EXPECT_EQ(reg_add->builtin_code, kTfLiteBuiltinAdd);
  EXPECT_EQ(reg_add->version, 1);
  EXPECT_EQ(reg_add->registration_external, nullptr);

  EXPECT_EQ(
      resolver.FindOp(
          static_cast<::tflite::BuiltinOperator>(kTfLiteBuiltinConv2d), 1),
      nullptr);

  auto reg_sinh = resolver.FindOp("Sinh", 1);
  ASSERT_NE(reg_sinh, nullptr);
  EXPECT_EQ(reg_sinh->builtin_code, kTfLiteBuiltinCustom);
  EXPECT_EQ(reg_sinh->custom_name, "Sinh");
  EXPECT_EQ(reg_sinh->version, 1);
  EXPECT_EQ(reg_sinh->registration_external, nullptr);

  EXPECT_EQ(resolver.FindOp("Cosh", 1), nullptr);
}

const TfLiteRegistration_V1* dummy_find_builtin_op_v1(void* user_data,
                                                      TfLiteBuiltinOperator op,
                                                      int version) {
  static TfLiteRegistration_V1 registration_v1{
      nullptr, nullptr,           nullptr, nullptr,
      nullptr, kTfLiteBuiltinAdd, nullptr, 1};
  if (op == kTfLiteBuiltinAdd) {
    return &registration_v1;
  }
  return nullptr;
}

const TfLiteRegistration_V1* dummy_find_custom_op_v1(void* user_data,
                                                     const char* op,
                                                     int version) {
  static TfLiteRegistration_V1 registration_v1{
      nullptr, nullptr, nullptr, nullptr, nullptr, kTfLiteBuiltinCustom,
      "Sinh",  1};
  if (strcmp(op, "Sinh") == 0) {
    return &registration_v1;
  }
  return nullptr;
}

TEST(CApiSimple, CallbackOpResolver_V1) {
  tflite::internal::CallbackOpResolver resolver;
  struct TfLiteOpResolverCallbacks callbacks {};
  callbacks.find_builtin_op_v1 = dummy_find_builtin_op_v1;
  callbacks.find_custom_op_v1 = dummy_find_custom_op_v1;

  resolver.SetCallbacks(callbacks);
  auto reg_add = resolver.FindOp(
      static_cast<::tflite::BuiltinOperator>(kTfLiteBuiltinAdd), 1);
  ASSERT_NE(reg_add, nullptr);
  EXPECT_EQ(reg_add->builtin_code, kTfLiteBuiltinAdd);
  EXPECT_EQ(reg_add->version, 1);
  EXPECT_EQ(reg_add->registration_external, nullptr);

  EXPECT_EQ(
      resolver.FindOp(
          static_cast<::tflite::BuiltinOperator>(kTfLiteBuiltinConv2d), 1),
      nullptr);

  // Query kTfLiteBuiltinAdd multiple times to check if caching logic works.
  for (int i = 0; i < 10; ++i) {
    auto reg_add = resolver.FindOp(
        static_cast<::tflite::BuiltinOperator>(kTfLiteBuiltinAdd), 1);
    ASSERT_NE(reg_add, nullptr);
    EXPECT_EQ(reg_add->builtin_code, kTfLiteBuiltinAdd);
    EXPECT_EQ(reg_add->version, 1);
    EXPECT_EQ(reg_add->registration_external, nullptr);
  }

  auto reg_sinh = resolver.FindOp("Sinh", 1);
  ASSERT_NE(reg_sinh, nullptr);
  EXPECT_EQ(reg_sinh->builtin_code, kTfLiteBuiltinCustom);
  EXPECT_EQ(reg_sinh->custom_name, "Sinh");
  EXPECT_EQ(reg_sinh->version, 1);
  EXPECT_EQ(reg_sinh->registration_external, nullptr);

  EXPECT_EQ(resolver.FindOp("Cosh", 1), nullptr);

  // Query "Sinh" multiple times to check if caching logic works.
  for (int i = 0; i < 10; ++i) {
    auto reg_sinh = resolver.FindOp("Sinh", 1);
    ASSERT_NE(reg_sinh, nullptr);
    EXPECT_EQ(reg_sinh->builtin_code, kTfLiteBuiltinCustom);
    EXPECT_EQ(reg_sinh->custom_name, "Sinh");
    EXPECT_EQ(reg_sinh->version, 1);
    EXPECT_EQ(reg_sinh->registration_external, nullptr);
  }
}

}  // namespace
