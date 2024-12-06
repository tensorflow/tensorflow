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
#include "tensorflow/lite/core/async/interop/c/attribute_map.h"

#include <cstddef>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/async/interop/c/types.h"

namespace {

TEST(AttributeMapTest, AttributeMapCreateTypeCheckTest) {
  {
    auto* attr = TfLiteAttributeMapCreate(kTfLiteAttrMapTypeBuffer);
    EXPECT_TRUE(TfLiteAttributeMapIsBufferAttributeMap(attr));
    EXPECT_FALSE(TfLiteAttributeMapIsSyncAttributeMap(attr));
    TfLiteAttributeMapDelete(attr);
  }
  {
    auto* attr = TfLiteAttributeMapCreate(kTfLiteAttrMapTypeSync);
    EXPECT_FALSE(TfLiteAttributeMapIsBufferAttributeMap(attr));
    EXPECT_TRUE(TfLiteAttributeMapIsSyncAttributeMap(attr));
    TfLiteAttributeMapDelete(attr);
  }
}

TEST(AttributeMapTest, AttributeMapAccessor) {
  auto* attr = TfLiteAttributeMapCreate(kTfLiteAttrMapTypeBuffer);
  {
    TfLiteAttributeMapSetSizeTBufferAttr(attr, kTfLiteBufferAttrKeyAlignment,
                                         42);
    size_t result = 0;
    EXPECT_TRUE(TfLiteAttributeMapGetSizeTBufferAttr(
        attr, kTfLiteBufferAttrKeyAlignment, &result));
    EXPECT_EQ(42, result);
    EXPECT_FALSE(TfLiteAttributeMapGetSizeTBufferAttr(
        attr, kTfLiteBufferAttrKeyOffset, &result));
  }
  {
    const char str[] = "some string";
    // Overriding key 1.
    TfLiteAttributeMapSetStringBufferAttr(
        attr, kTfLiteBufferAttrKeyResourceTypeName, str);
    const char* result = nullptr;
    EXPECT_TRUE(TfLiteAttributeMapGetStringBufferAttr(
        attr, kTfLiteBufferAttrKeyResourceTypeName, &result));
    EXPECT_EQ(str, result);
    EXPECT_FALSE(TfLiteAttributeMapGetStringBufferAttr(
        attr, kTfLiteBufferAttrKeyAlignment, &result));
    EXPECT_FALSE(TfLiteAttributeMapSetStringSyncAttr(
        attr, kTfLiteSynchronizationAttrKeyObjectTypeName, str));
    EXPECT_FALSE(TfLiteAttributeMapGetStringSyncAttr(
        attr, kTfLiteSynchronizationAttrKeyObjectTypeName, &result));
  }
  TfLiteAttributeMapDelete(attr);
}

TEST(AttributeMapTest, UnCheckedAttributeMapAccessor) {
  auto* attr = TfLiteAttributeMapCreate(kTfLiteAttrMapTypeBuffer);
  {
    TfLiteAttributeMapSetSizeTAttr(attr, 1, 42);
    size_t result = 0;
    EXPECT_TRUE(TfLiteAttributeMapGetSizeTAttr(attr, 1, &result));
    EXPECT_EQ(42, result);
    EXPECT_FALSE(TfLiteAttributeMapGetSizeTAttr(attr, 2, &result));
  }
  {
    TfLiteAttributeMapSetIntAttr(attr, 3, 21);
    int result = 0;
    EXPECT_TRUE(TfLiteAttributeMapGetIntAttr(attr, 3, &result));
    EXPECT_EQ(21, result);
    EXPECT_FALSE(TfLiteAttributeMapGetIntAttr(attr, 4, &result));
  }
  {
    const char str[] = "some string";
    // Overriding key 1.
    TfLiteAttributeMapSetStringAttr(attr, 1, str);
    const char* result = nullptr;
    EXPECT_TRUE(TfLiteAttributeMapGetStringAttr(attr, 1, &result));
    EXPECT_EQ(str, result);
    EXPECT_FALSE(TfLiteAttributeMapGetStringAttr(attr, 2, &result));
  }
  {
    TfLiteAttributeMapSetBoolAttr(
        attr, kTfLiteBufferAttrKeyCurrentHostCoherencyState, true);
    bool result = false;
    EXPECT_TRUE(TfLiteAttributeMapGetBoolAttr(
        attr, kTfLiteBufferAttrKeyCurrentHostCoherencyState, &result));
    EXPECT_TRUE(result);
    EXPECT_FALSE(TfLiteAttributeMapGetBoolAttr(
        attr, kTfLiteBufferAttrKeyPreferredHostCoherencyState, &result));
  }
  TfLiteAttributeMapDelete(attr);
}

TEST(AttributeMapTest, UnCheckedAttributeMapCustomAccessor) {
  auto* attr = TfLiteAttributeMapCreate(kTfLiteAttrMapTypeBuffer);
  {
    TfLiteAttributeMapSetCustomSizeTAttr(attr, "foo", 42);
    size_t result = 0;
    EXPECT_TRUE(TfLiteAttributeMapGetCustomSizeTAttr(attr, "foo", &result));
    EXPECT_EQ(42, result);
    EXPECT_FALSE(TfLiteAttributeMapGetCustomSizeTAttr(attr, "bar", &result));
  }
  {
    TfLiteAttributeMapSetCustomIntAttr(attr, "baz", 21);
    int result = 0;
    EXPECT_TRUE(TfLiteAttributeMapGetCustomIntAttr(attr, "baz", &result));
    EXPECT_EQ(21, result);
    EXPECT_FALSE(TfLiteAttributeMapGetCustomIntAttr(attr, "quux", &result));
  }
  {
    const char str[] = "some string";
    // Overriding key "foo".
    TfLiteAttributeMapSetCustomStringAttr(attr, "foo", str);
    const char* result = nullptr;
    EXPECT_TRUE(TfLiteAttributeMapGetCustomStringAttr(attr, "foo", &result));
    EXPECT_EQ(str, result);
    EXPECT_FALSE(TfLiteAttributeMapGetCustomStringAttr(attr, "bar", &result));
  }
  {
    TfLiteAttributeMapSetCustomBoolAttr(attr, "foo", true);
    bool result = false;
    EXPECT_TRUE(TfLiteAttributeMapGetCustomBoolAttr(attr, "foo", &result));
    EXPECT_TRUE(result);
    EXPECT_FALSE(TfLiteAttributeMapGetCustomBoolAttr(attr, "bar", &result));
  }
  TfLiteAttributeMapDelete(attr);
}

}  // namespace
