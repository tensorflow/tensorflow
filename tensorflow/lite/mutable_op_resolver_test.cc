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

#include "tensorflow/lite/mutable_op_resolver.h"

#include <stddef.h>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"

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

TfLiteStatus Dummy2Invoke(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus Dummy2Prepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

void* Dummy2Init(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

void Dummy2free(TfLiteContext* context, void* buffer) {}

TfLiteRegistration* GetDummy2Registration() {
  static TfLiteRegistration registration = {
      .init = Dummy2Init,
      .free = Dummy2free,
      .prepare = Dummy2Prepare,
      .invoke = Dummy2Invoke,
  };
  return &registration;
}

TEST(MutableOpResolverTest, FindOp) {
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

TEST(MutableOpResolverTest, RegisterOpWithSingleVersion) {
  MutableOpResolver resolver;
  // The kernel supports version 2 only
  resolver.AddBuiltin(BuiltinOperator_ADD, GetDummyRegistration(), 2);

  const TfLiteRegistration* found_registration;

  found_registration = resolver.FindOp(BuiltinOperator_ADD, 1);
  ASSERT_EQ(found_registration, nullptr);

  found_registration = resolver.FindOp(BuiltinOperator_ADD, 2);
  ASSERT_NE(found_registration, nullptr);
  EXPECT_TRUE(found_registration->invoke == DummyInvoke);
  EXPECT_EQ(found_registration->version, 2);

  found_registration = resolver.FindOp(BuiltinOperator_ADD, 3);
  ASSERT_EQ(found_registration, nullptr);
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
}

TEST(MutableOpResolverTest, FindCustomName) {
  MutableOpResolver resolver;
  TfLiteRegistration* reg = GetDummyRegistration();

  reg->custom_name = "UPDATED";
  resolver.AddCustom(reg->custom_name, reg);
  const TfLiteRegistration* found_registration =
      resolver.FindOp(reg->custom_name, 1);

  ASSERT_NE(found_registration, nullptr);
  EXPECT_EQ(found_registration->builtin_code, BuiltinOperator_CUSTOM);
  EXPECT_EQ(found_registration->invoke, GetDummyRegistration()->invoke);
  EXPECT_EQ(found_registration->version, 1);
  EXPECT_EQ(found_registration->custom_name, "UPDATED");
}

TEST(MutableOpResolverTest, FindBuiltinName) {
  MutableOpResolver resolver1;
  TfLiteRegistration* reg = GetDummy2Registration();

  reg->custom_name = "UPDATED";
  resolver1.AddBuiltin(BuiltinOperator_ADD, reg);

  ASSERT_EQ(resolver1.FindOp(BuiltinOperator_ADD, 1)->invoke,
            GetDummy2Registration()->invoke);
  ASSERT_EQ(resolver1.FindOp(BuiltinOperator_ADD, 1)->prepare,
            GetDummy2Registration()->prepare);
  ASSERT_EQ(resolver1.FindOp(BuiltinOperator_ADD, 1)->init,
            GetDummy2Registration()->init);
  ASSERT_EQ(resolver1.FindOp(BuiltinOperator_ADD, 1)->free,
            GetDummy2Registration()->free);
  // custom_name for builtin ops will be nullptr
  EXPECT_EQ(resolver1.FindOp(BuiltinOperator_ADD, 1)->custom_name, nullptr);
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

TEST(MutableOpResolverTest, AddAll) {
  MutableOpResolver resolver1;
  resolver1.AddBuiltin(BuiltinOperator_ADD, GetDummyRegistration());
  resolver1.AddBuiltin(BuiltinOperator_MUL, GetDummy2Registration());

  MutableOpResolver resolver2;
  resolver2.AddBuiltin(BuiltinOperator_SUB, GetDummyRegistration());
  resolver2.AddBuiltin(BuiltinOperator_ADD, GetDummy2Registration());

  resolver1.AddAll(resolver2);

  // resolver2's ADD op should replace resolver1's ADD op, while augmenting
  // non-overlapping ops.
  ASSERT_EQ(resolver1.FindOp(BuiltinOperator_ADD, 1)->invoke,
            GetDummy2Registration()->invoke);
  ASSERT_EQ(resolver1.FindOp(BuiltinOperator_MUL, 1)->invoke,
            GetDummy2Registration()->invoke);
  ASSERT_EQ(resolver1.FindOp(BuiltinOperator_SUB, 1)->invoke,
            GetDummyRegistration()->invoke);
}

class ChainingMutableOpResolver : public MutableOpResolver {
 public:
  using MutableOpResolver::ChainOpResolver;
};

TEST(MutableOpResolverTest, ChainOpResolver) {
  ChainingMutableOpResolver resolver1;
  resolver1.AddBuiltin(BuiltinOperator_ADD, GetDummyRegistration());
  resolver1.AddBuiltin(BuiltinOperator_MUL, GetDummy2Registration());

  MutableOpResolver resolver2;
  resolver2.AddBuiltin(BuiltinOperator_SUB, GetDummyRegistration());
  resolver2.AddBuiltin(BuiltinOperator_ADD, GetDummy2Registration());

  resolver1.ChainOpResolver(&resolver2);

  // resolver2's ADD op should NOT replace resolver1's ADD op;
  // but resolver2's non-overlapping ops should augment resolver1's.
  ASSERT_EQ(resolver1.FindOp(BuiltinOperator_ADD, 1)->invoke,
            GetDummyRegistration()->invoke);
  ASSERT_EQ(resolver1.FindOp(BuiltinOperator_MUL, 1)->invoke,
            GetDummy2Registration()->invoke);
  ASSERT_EQ(resolver1.FindOp(BuiltinOperator_SUB, 1)->invoke,
            GetDummyRegistration()->invoke);
}

TEST(MutableOpResolverTest, CopyConstructChainedOpResolver) {
  MutableOpResolver resolver;
  resolver.AddBuiltin(BuiltinOperator_ADD, GetDummyRegistration());
  resolver.AddBuiltin(BuiltinOperator_SUB, GetDummy2Registration());
  resolver.AddCustom("MyCustom", GetDummy2Registration());

  ChainingMutableOpResolver resolver2;
  resolver2.ChainOpResolver(&resolver);

  MutableOpResolver resolver3(resolver2);

  ASSERT_EQ(resolver3.FindOp(BuiltinOperator_ADD, 1)->invoke,
            GetDummyRegistration()->invoke);
  ASSERT_EQ(resolver3.FindOp(BuiltinOperator_SUB, 1)->invoke,
            GetDummy2Registration()->invoke);
  ASSERT_EQ(resolver3.FindOp(BuiltinOperator_MUL, 1), nullptr);
  ASSERT_EQ(resolver3.FindOp("MyCustom", 1)->invoke,
            GetDummy2Registration()->invoke);
  ASSERT_EQ(resolver3.FindOp("NotMyCustom", 1), nullptr);
}

TEST(MutableOpResolverTest, AssignChainedOpResolver) {
  MutableOpResolver resolver;
  resolver.AddBuiltin(BuiltinOperator_ADD, GetDummyRegistration());
  resolver.AddBuiltin(BuiltinOperator_SUB, GetDummy2Registration());
  resolver.AddCustom("MyCustom", GetDummy2Registration());

  ChainingMutableOpResolver resolver2;
  resolver2.ChainOpResolver(&resolver);

  MutableOpResolver resolver3;
  resolver3 = resolver2;

  ASSERT_EQ(resolver3.FindOp(BuiltinOperator_ADD, 1)->invoke,
            GetDummyRegistration()->invoke);
  ASSERT_EQ(resolver3.FindOp(BuiltinOperator_SUB, 1)->invoke,
            GetDummy2Registration()->invoke);
  ASSERT_EQ(resolver3.FindOp(BuiltinOperator_MUL, 1), nullptr);
  ASSERT_EQ(resolver3.FindOp("MyCustom", 1)->invoke,
            GetDummy2Registration()->invoke);
  ASSERT_EQ(resolver3.FindOp("NotMyCustom", 1), nullptr);
}

TEST(MutableOpResolverTest, AddAllChainedOpResolver) {
  MutableOpResolver resolver;
  resolver.AddBuiltin(BuiltinOperator_ADD, GetDummyRegistration());
  resolver.AddBuiltin(BuiltinOperator_SUB, GetDummy2Registration());
  resolver.AddCustom("MyCustom", GetDummy2Registration());

  ChainingMutableOpResolver resolver2;
  resolver2.ChainOpResolver(&resolver);

  MutableOpResolver resolver3;
  resolver3.AddAll(resolver2);

  ASSERT_EQ(resolver3.FindOp(BuiltinOperator_ADD, 1)->invoke,
            GetDummyRegistration()->invoke);
  ASSERT_EQ(resolver3.FindOp(BuiltinOperator_SUB, 1)->invoke,
            GetDummy2Registration()->invoke);
  ASSERT_EQ(resolver3.FindOp(BuiltinOperator_MUL, 1), nullptr);
  ASSERT_EQ(resolver3.FindOp("MyCustom", 1)->invoke,
            GetDummy2Registration()->invoke);
  ASSERT_EQ(resolver3.FindOp("NotMyCustom", 1), nullptr);
}

TEST(MutableOpResolverTest, ChainOpResolverCustomOpPrecedence) {
  MutableOpResolver resolver1;
  resolver1.AddCustom("MyCustom", GetDummyRegistration());

  MutableOpResolver resolver2;
  resolver2.AddCustom("MyCustom", GetDummy2Registration());

  ChainingMutableOpResolver resolver3;
  resolver3.ChainOpResolver(&resolver1);
  resolver3.ChainOpResolver(&resolver2);

  ASSERT_EQ(resolver3.FindOp("MyCustom", 1)->invoke,
            GetDummyRegistration()->invoke);
}

TEST(MutableOpResolverTest, ChainOpResolverBuiltinOpPrecedence) {
  MutableOpResolver resolver1;
  resolver1.AddBuiltin(BuiltinOperator_ADD, GetDummyRegistration());

  MutableOpResolver resolver2;
  resolver2.AddBuiltin(BuiltinOperator_ADD, GetDummy2Registration());

  ChainingMutableOpResolver resolver3;
  resolver3.ChainOpResolver(&resolver1);
  resolver3.ChainOpResolver(&resolver2);

  ASSERT_EQ(resolver3.FindOp(BuiltinOperator_ADD, 1)->invoke,
            GetDummyRegistration()->invoke);
}

TEST(MutableOpResolverTest, ChainOpResolverAddVersusChainPrecedence) {
  MutableOpResolver resolver1;
  resolver1.AddCustom("MyCustom", GetDummyRegistration());

  ChainingMutableOpResolver resolver2;
  resolver2.ChainOpResolver(&resolver1);

  MutableOpResolver resolver3;
  resolver3.AddCustom("MyCustom", GetDummy2Registration());

  ChainingMutableOpResolver resolver4;
  resolver4.ChainOpResolver(&resolver2);
  resolver4.ChainOpResolver(&resolver3);

  ASSERT_EQ(resolver4.FindOp("MyCustom", 1)->invoke,
            GetDummyRegistration()->invoke);
}

TEST(MutableOpResolverTest, AddAllAddVersusChainPrecedence) {
  MutableOpResolver resolver1;
  resolver1.AddCustom("MyCustom", GetDummyRegistration());

  ChainingMutableOpResolver resolver2;
  resolver2.ChainOpResolver(&resolver1);

  MutableOpResolver resolver3;
  resolver3.AddCustom("MyCustom", GetDummy2Registration());

  MutableOpResolver resolver4;
  resolver4.AddAll(resolver2);
  resolver4.AddAll(resolver3);

  ASSERT_EQ(resolver4.FindOp("MyCustom", 1)->invoke,
            GetDummy2Registration()->invoke);
}

}  // namespace
}  // namespace tflite
