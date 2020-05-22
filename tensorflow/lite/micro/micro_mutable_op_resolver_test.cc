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

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace {
void* MockInit(TfLiteContext* context, const char* buffer, size_t length) {
  // Do nothing.
  return nullptr;
}

void MockFree(TfLiteContext* context, void* buffer) {
  // Do nothing.
}

TfLiteStatus MockPrepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus MockInvoke(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

class MockErrorReporter : public ErrorReporter {
 public:
  MockErrorReporter() : has_been_called_(false) {}
  int Report(const char* format, va_list args) override {
    has_been_called_ = true;
    return 0;
  };

  bool HasBeenCalled() { return has_been_called_; }

 private:
  bool has_been_called_;
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestOperations) {
  using tflite::BuiltinOperator_CONV_2D;
  using tflite::BuiltinOperator_RELU;
  using tflite::MicroOpResolver;
  using tflite::OpResolver;

  static TfLiteRegistration r = {tflite::MockInit, tflite::MockFree,
                                 tflite::MockPrepare, tflite::MockInvoke};

  // We need space for 7 operators because of 2 ops, one with 3 versions, one
  // with 4 versions.
  MicroOpResolver<7> micro_op_resolver;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, micro_op_resolver.AddBuiltin(
                                         BuiltinOperator_CONV_2D, &r, 1, 3));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          micro_op_resolver.AddCustom("mock_custom", &r, 1, 4));
  OpResolver* resolver = &micro_op_resolver;

  const TfLiteRegistration* registration =
      resolver->FindOp(BuiltinOperator_CONV_2D, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, registration->init(nullptr, nullptr, 0));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(nullptr, nullptr));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(nullptr, nullptr));

  TF_LITE_MICRO_EXPECT_EQ(7, micro_op_resolver.GetRegistrationLength());

  registration = resolver->FindOp(BuiltinOperator_CONV_2D, 10);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, registration);

  registration = resolver->FindOp(BuiltinOperator_RELU, 0);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, registration);

  registration = resolver->FindOp("mock_custom", 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, registration->init(nullptr, nullptr, 0));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(nullptr, nullptr));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(nullptr, nullptr));

  registration = resolver->FindOp("mock_custom", 10);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, registration);

  registration = resolver->FindOp("nonexistent_custom", 0);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, registration);
}

TF_LITE_MICRO_TEST(TestOpRegistrationOverflow) {
  using tflite::BuiltinOperator_CONV_2D;
  using tflite::BuiltinOperator_RELU;
  using tflite::MicroOpResolver;
  using tflite::OpResolver;

  static TfLiteRegistration r = {tflite::MockInit, tflite::MockFree,
                                 tflite::MockPrepare, tflite::MockInvoke};

  MicroOpResolver<4> micro_op_resolver;
  // Register 7 ops, but only 4 is expected because the class is created with
  // that limit..
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, micro_op_resolver.AddBuiltin(
                                         BuiltinOperator_CONV_2D, &r, 0, 2));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                          micro_op_resolver.AddCustom("mock_custom", &r, 0, 3));
  OpResolver* resolver = &micro_op_resolver;

  TF_LITE_MICRO_EXPECT_EQ(4, micro_op_resolver.GetRegistrationLength());
}

TF_LITE_MICRO_TEST(TestZeroVersionRegistration) {
  using tflite::MicroOpResolver;
  using tflite::OpResolver;

  static TfLiteRegistration r = {tflite::MockInit, tflite::MockFree,
                                 tflite::MockPrepare, tflite::MockInvoke};

  MicroOpResolver<1> micro_op_resolver;
  micro_op_resolver.AddCustom("mock_custom", &r,
                              tflite::MicroOpResolverAnyVersion());

  TF_LITE_MICRO_EXPECT_EQ(1, micro_op_resolver.GetRegistrationLength());

  OpResolver* resolver = &micro_op_resolver;

  const TfLiteRegistration* registration = resolver->FindOp("mock_custom", 0);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, registration->init(nullptr, nullptr, 0));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(nullptr, nullptr));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(nullptr, nullptr));

  registration = resolver->FindOp("mock_custom", 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, registration->init(nullptr, nullptr, 0));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(nullptr, nullptr));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(nullptr, nullptr));

  registration = resolver->FindOp("mock_custom", 42);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, registration->init(nullptr, nullptr, 0));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(nullptr, nullptr));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(nullptr, nullptr));
}

TF_LITE_MICRO_TEST(TestZeroModelVersion) {
  using tflite::MicroOpResolver;
  using tflite::OpResolver;

  static TfLiteRegistration r = {tflite::MockInit, tflite::MockFree,
                                 tflite::MockPrepare, tflite::MockInvoke};

  MicroOpResolver<2> micro_op_resolver;
  micro_op_resolver.AddCustom("mock_custom", &r, 1, 2);
  TF_LITE_MICRO_EXPECT_EQ(2, micro_op_resolver.GetRegistrationLength());
  OpResolver* resolver = &micro_op_resolver;

  // If the Op version in the model is 0, we should always get the first
  // registration.
  const TfLiteRegistration* registration = resolver->FindOp("mock_custom", 0);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);
  TF_LITE_MICRO_EXPECT_EQ(1, registration->version);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, registration->init(nullptr, nullptr, 0));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(nullptr, nullptr));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(nullptr, nullptr));

  // If a non-zero version is requested, the correct version'd op should be
  // returned. TODO(b/151245712): Realistically, we are better off removing
  // these version checks altogether.
  for (int i = 1; i <= 2; ++i) {
    registration = resolver->FindOp("mock_custom", i);
    TF_LITE_MICRO_EXPECT_NE(nullptr, registration);
    TF_LITE_MICRO_EXPECT_EQ(i, registration->version);
    TF_LITE_MICRO_EXPECT_EQ(nullptr, registration->init(nullptr, nullptr, 0));
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(nullptr, nullptr));
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(nullptr, nullptr));
  }

  registration = resolver->FindOp("mock_custom", 42);
  TF_LITE_MICRO_EXPECT_EQ(nullptr, registration);
}

TF_LITE_MICRO_TEST(TestBuiltinRegistrationErrorReporting) {
  using tflite::BuiltinOperator_CONV_2D;
  using tflite::BuiltinOperator_RELU;
  using tflite::MicroOpResolver;

  static TfLiteRegistration r = {tflite::MockInit, tflite::MockFree,
                                 tflite::MockPrepare, tflite::MockInvoke};

  tflite::MockErrorReporter mock_reporter;
  MicroOpResolver<1> micro_op_resolver(&mock_reporter);
  TF_LITE_MICRO_EXPECT_EQ(false, mock_reporter.HasBeenCalled());
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, micro_op_resolver.AddBuiltin(BuiltinOperator_CONV_2D, &r));
  TF_LITE_MICRO_EXPECT_EQ(false, mock_reporter.HasBeenCalled());
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteError, micro_op_resolver.AddBuiltin(BuiltinOperator_RELU, &r));
  TF_LITE_MICRO_EXPECT_EQ(true, mock_reporter.HasBeenCalled());
}

TF_LITE_MICRO_TEST(TestCustomRegistrationErrorReporting) {
  using tflite::BuiltinOperator_CONV_2D;
  using tflite::BuiltinOperator_RELU;
  using tflite::MicroOpResolver;

  static TfLiteRegistration r = {tflite::MockInit, tflite::MockFree,
                                 tflite::MockPrepare, tflite::MockInvoke};

  tflite::MockErrorReporter mock_reporter;
  MicroOpResolver<1> micro_op_resolver(&mock_reporter);
  TF_LITE_MICRO_EXPECT_EQ(false, mock_reporter.HasBeenCalled());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          micro_op_resolver.AddCustom("mock_custom_0", &r));
  TF_LITE_MICRO_EXPECT_EQ(false, mock_reporter.HasBeenCalled());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                          micro_op_resolver.AddCustom("mock_custom_1", &r));
  TF_LITE_MICRO_EXPECT_EQ(true, mock_reporter.HasBeenCalled());
}

TF_LITE_MICRO_TEST(TestBuiltinVersionRegistrationErrorReporting) {
  using tflite::BuiltinOperator_CONV_2D;
  using tflite::BuiltinOperator_RELU;
  using tflite::MicroOpResolver;

  static TfLiteRegistration r = {tflite::MockInit, tflite::MockFree,
                                 tflite::MockPrepare, tflite::MockInvoke};

  tflite::MockErrorReporter mock_reporter;
  MicroOpResolver<2> micro_op_resolver(&mock_reporter);
  TF_LITE_MICRO_EXPECT_EQ(false, mock_reporter.HasBeenCalled());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, micro_op_resolver.AddBuiltin(
                                         BuiltinOperator_CONV_2D, &r, 1, 2));
  TF_LITE_MICRO_EXPECT_EQ(false, mock_reporter.HasBeenCalled());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError, micro_op_resolver.AddBuiltin(
                                            BuiltinOperator_RELU, &r, 1, 2));
  TF_LITE_MICRO_EXPECT_EQ(true, mock_reporter.HasBeenCalled());
}

TF_LITE_MICRO_TEST(TestCustomVersionRegistrationErrorReporting) {
  using tflite::BuiltinOperator_CONV_2D;
  using tflite::BuiltinOperator_RELU;
  using tflite::MicroOpResolver;

  static TfLiteRegistration r = {tflite::MockInit, tflite::MockFree,
                                 tflite::MockPrepare, tflite::MockInvoke};

  tflite::MockErrorReporter mock_reporter;
  MicroOpResolver<2> micro_op_resolver(&mock_reporter);
  TF_LITE_MICRO_EXPECT_EQ(false, mock_reporter.HasBeenCalled());
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, micro_op_resolver.AddCustom("mock_custom_0", &r, 1, 2));
  TF_LITE_MICRO_EXPECT_EQ(false, mock_reporter.HasBeenCalled());
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteError, micro_op_resolver.AddCustom("mock_custom_1", &r, 1, 2));
  TF_LITE_MICRO_EXPECT_EQ(true, mock_reporter.HasBeenCalled());
}

TF_LITE_MICRO_TESTS_END
