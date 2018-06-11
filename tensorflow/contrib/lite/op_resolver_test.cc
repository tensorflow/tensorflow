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

#include "tensorflow/contrib/lite/op_resolver.h"

#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/testing/util.h"

namespace tflite {
namespace {

// We need some dummy functions to identify the registrations.
TfLiteStatus DummyInvoke(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteRegistration* GetDummyRegistration() {
  static TfLiteRegistration registration = {
      .init = nullptr,
      .free = nullptr,
      .prepare = nullptr,
      .invoke = DummyInvoke,
  };
  return &registration;
}

TEST(MutableOpResolverTest, FinOp) {
  MutableOpResolver resolver;
  resolver.AddBuiltin(BuiltinOperator_ADD, GetDummyRegistration());

  const TfLiteRegistration* found_registration =
      resolver.FindOp(BuiltinOperator_ADD, 1);
  ASSERT_NE(found_registration, nullptr);
  EXPECT_TRUE(found_registration->invoke == DummyInvoke);
  EXPECT_EQ(found_registration->builtin_code, BuiltinOperator_ADD);
  EXPECT_EQ(found_registration->version, 1);
}

TEST(MutableOpResolverTest, FindMissingOp) {
  MutableOpResolver resolver;
  resolver.AddBuiltin(BuiltinOperator_ADD, GetDummyRegistration());

  const TfLiteRegistration* found_registration =
      resolver.FindOp(BuiltinOperator_CONV_2D, 1);
  EXPECT_EQ(found_registration, nullptr);
}

TEST(MutableOpResolverTest, RegisterOpWithMultipleVersions) {
  MutableOpResolver resolver;
  // The kernel supports version 2 and 3
  resolver.AddBuiltin(BuiltinOperator_ADD, GetDummyRegistration(), 2, 3);

  const TfLiteRegistration* found_registration;

  found_registration = resolver.FindOp(BuiltinOperator_ADD, 2);
  ASSERT_NE(found_registration, nullptr);
  EXPECT_TRUE(found_registration->invoke == DummyInvoke);
  EXPECT_EQ(found_registration->version, 2);

  found_registration = resolver.FindOp(BuiltinOperator_ADD, 3);
  ASSERT_NE(found_registration, nullptr);
  EXPECT_TRUE(found_registration->invoke == DummyInvoke);
  EXPECT_EQ(found_registration->version, 3);
}

TEST(MutableOpResolverTest, FindOpWithUnsupportedVersions) {
  MutableOpResolver resolver;
  // The kernel supports version 2 and 3
  resolver.AddBuiltin(BuiltinOperator_ADD, GetDummyRegistration(), 2, 3);

  const TfLiteRegistration* found_registration;

  found_registration = resolver.FindOp(BuiltinOperator_ADD, 1);
  EXPECT_EQ(found_registration, nullptr);

  found_registration = resolver.FindOp(BuiltinOperator_ADD, 4);
  EXPECT_EQ(found_registration, nullptr);
}

TEST(MutableOpResolverTest, FindCustomOp) {
  MutableOpResolver resolver;
  resolver.AddCustom("AWESOME", GetDummyRegistration());

  const TfLiteRegistration* found_registration = resolver.FindOp("AWESOME", 1);
  ASSERT_NE(found_registration, nullptr);
  EXPECT_EQ(found_registration->builtin_code, BuiltinOperator_CUSTOM);
  EXPECT_TRUE(found_registration->invoke == DummyInvoke);
  EXPECT_EQ(found_registration->version, 1);
  // TODO(ycling): The `custom_name` in TfLiteRegistration isn't properly
  // filled yet. Fix this and add tests.
}

TEST(MutableOpResolverTest, FindMissingCustomOp) {
  MutableOpResolver resolver;
  resolver.AddCustom("AWESOME", GetDummyRegistration());

  const TfLiteRegistration* found_registration =
      resolver.FindOp("EXCELLENT", 1);
  EXPECT_EQ(found_registration, nullptr);
}

TEST(MutableOpResolverTest, FindCustomOpWithUnsupportedVersion) {
  MutableOpResolver resolver;
  resolver.AddCustom("AWESOME", GetDummyRegistration());

  const TfLiteRegistration* found_registration = resolver.FindOp("AWESOME", 2);
  EXPECT_EQ(found_registration, nullptr);
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
