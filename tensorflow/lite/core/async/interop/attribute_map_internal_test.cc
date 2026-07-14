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
#include "tensorflow/lite/core/async/interop/attribute_map_internal.h"

#include <cstddef>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/async/interop/c/types.h"

namespace tflite {
namespace interop {
namespace {

TEST(AttributeMapTest, TypeTest) {
  {
    auto attrs = AttributeMap(kTfLiteAttrMapTypeBuffer);
    EXPECT_TRUE(attrs.IsBufferAttributeMap());
    EXPECT_FALSE(attrs.IsSyncAttributeMap());
  }

  {
    auto attrs = AttributeMap(kTfLiteAttrMapTypeSync);
    EXPECT_TRUE(attrs.IsSyncAttributeMap());
    EXPECT_FALSE(attrs.IsBufferAttributeMap());
  }
}

TEST(AttributeMapTest, AccessorTest) {
  auto attrs = AttributeMap(kTfLiteAttrMapTypeBuffer);
  {
    attrs.SetAttr(kTfLiteBufferAttrKeyAlignment, size_t(8));
    size_t result;
    EXPECT_TRUE(attrs.GetAttr(kTfLiteBufferAttrKeyAlignment, &result));
    EXPECT_EQ(8, result);
  }
  {
    attrs.SetCustomAttr("Foo", 12);
    int result;
    EXPECT_FALSE(attrs.GetCustomAttr("Bar", &result));
    EXPECT_TRUE(attrs.GetCustomAttr("Foo", &result));
    EXPECT_EQ(12, result);
  }
}

TEST(AttributeMapTest, ReconcileFailDifferentTypes) {
  auto attrs1 = AttributeMap(kTfLiteAttrMapTypeBuffer);
  auto attrs2 = AttributeMap(kTfLiteAttrMapTypeSync);
  auto attrs3 = AttributeMap(kTfLiteAttrMapTypeBuffer);
  EXPECT_FALSE(
      attrs1.ReconcileAttributes(&attrs2, &attrs3, /*conflict=*/nullptr));
  EXPECT_FALSE(attrs1.CheckAttributeCoverage(&attrs2, &attrs3));
}

TEST(AttributeMapTest, NullptrTest) {
  auto attrs1 = AttributeMap(kTfLiteAttrMapTypeBuffer);
  auto attrs2 = AttributeMap(kTfLiteAttrMapTypeBuffer);
  EXPECT_FALSE(attrs1.ReconcileAttributes(/*other=*/nullptr, &attrs2,
                                          /*conflict=*/nullptr));
  EXPECT_FALSE(attrs1.ReconcileAttributes(&attrs2, /*merged=*/nullptr,
                                          /*conflict=*/nullptr));
  EXPECT_FALSE(attrs1.CheckAttributeCoverage(/*other=*/nullptr,
                                             /*conflict=*/nullptr));
}

TEST(AttributeMapTest, ReconcileDifferentTypes) {
  auto attrs1 = AttributeMap(kTfLiteAttrMapTypeBuffer);
  auto attrs2 = AttributeMap(kTfLiteAttrMapTypeSync);
  auto attrs3 = AttributeMap(kTfLiteAttrMapTypeBuffer);
  EXPECT_FALSE(attrs1.ReconcileAttributes(&attrs2, &attrs3,
                                          /*conflict=*/nullptr));
}

TEST(AttributeMapTest, ReconcileTest) {
  auto attrs1 = AttributeMap(kTfLiteAttrMapTypeBuffer);
  attrs1.SetAttr(kTfLiteBufferAttrKeyAlignment, size_t(8));
  auto attrs2 = AttributeMap(kTfLiteAttrMapTypeBuffer);
  attrs2.SetAttr(kTfLiteBufferAttrKeyAlignment, size_t(4));
  auto attrs3 = AttributeMap(kTfLiteAttrMapTypeSync);
  auto attrs4 = AttributeMap(kTfLiteAttrMapTypeSync);
  EXPECT_TRUE(attrs1.ReconcileAttributes(&attrs2, &attrs3, &attrs4));
  EXPECT_TRUE(attrs3.IsBufferAttributeMap());
  EXPECT_TRUE(attrs4.IsBufferAttributeMap());
  size_t result;
  EXPECT_TRUE(attrs3.GetAttr(kTfLiteBufferAttrKeyAlignment, &result));
  EXPECT_EQ(8, result);
}

TEST(AttributeMapTest, CoverageTest) {
  auto attrs1 = AttributeMap(kTfLiteAttrMapTypeBuffer);
  attrs1.SetAttr(kTfLiteBufferAttrKeyAlignment, size_t(8));
  auto attrs2 = AttributeMap(kTfLiteAttrMapTypeBuffer);
  attrs2.SetAttr(kTfLiteBufferAttrKeyAlignment, size_t(4));
  auto attrs3 = AttributeMap(kTfLiteAttrMapTypeSync);
  EXPECT_TRUE(attrs1.CheckAttributeCoverage(&attrs2, &attrs3));
  EXPECT_TRUE(attrs3.IsBufferAttributeMap());
}

TEST(AttributeMapTest, CoverageFailedTest) {
  auto attrs1 = AttributeMap(kTfLiteAttrMapTypeBuffer);
  attrs1.SetAttr(kTfLiteBufferAttrKeyAlignment, size_t(10));
  auto attrs2 = AttributeMap(kTfLiteAttrMapTypeBuffer);
  attrs2.SetAttr(kTfLiteBufferAttrKeyAlignment, size_t(4));
  auto conflict = AttributeMap(kTfLiteAttrMapTypeSync);
  EXPECT_FALSE(attrs1.CheckAttributeCoverage(&attrs2, &conflict));
  EXPECT_TRUE(conflict.IsBufferAttributeMap());
  size_t result;
  EXPECT_TRUE(conflict.GetAttr(kTfLiteBufferAttrKeyAlignment, &result));
  EXPECT_EQ(4, result);
}

}  // namespace
}  // namespace interop
}  // namespace tflite
