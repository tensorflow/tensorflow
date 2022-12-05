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
#include "tensorflow/lite/core/async/interop/reconcile_fns.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <tuple>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/async/interop/attribute_keys.h"
#include "tensorflow/lite/core/async/interop/attribute_map_internal.h"

namespace tflite::interop {
namespace {

using ContainerT = AttributeMap::ContainerT;

template <typename ValT, typename KeyT>
void SetAttr(ContainerT* c, KeyT k, ValT v) {
  c->insert_or_assign(static_cast<uint32_t>(k), v);
}

template <typename ValT, typename KeyT>
ValT GetAttr(const ContainerT& c, KeyT k) {
  return c.at(static_cast<uint32_t>(k)).Get<ValT>();
}

TEST(ReconcileTest, NullCheck) {
  ContainerT m1, m2;
  // `merged` nullptr
  EXPECT_FALSE(ReconcileGeneralAttributeKeys(kTfLiteBufferAttrMap, &m1, &m2,
                                             /*merged=*/nullptr,
                                             /*conflict=*/nullptr));
  // `lhs` nullptr
  EXPECT_FALSE(ReconcileGeneralAttributeKeys(kTfLiteBufferAttrMap,
                                             /*lhs=*/nullptr, &m1, &m2,
                                             /*conflict=*/nullptr));
  // `rhs` nullptr
  EXPECT_FALSE(ReconcileGeneralAttributeKeys(kTfLiteBufferAttrMap, &m1,
                                             /*rhs=*/nullptr, &m2,
                                             /*conflict=*/nullptr));
  // `lhs` nullptr
  EXPECT_FALSE(CheckGeneralAttributeKeysCoverage(kTfLiteBufferAttrMap,
                                                 /*lhs=*/nullptr, &m1, &m2));
  // `rhs` nullptr
  EXPECT_FALSE(CheckGeneralAttributeKeysCoverage(kTfLiteBufferAttrMap, &m1,
                                                 /*rhs=*/nullptr, &m2));
}

TEST(ReconcileTest, MissingAttributeTest) {
  {
    ContainerT lhs, rhs, merged;
    SetAttr(&lhs, TfLiteBufferAttributeKey::kAlignment, size_t(4));
    EXPECT_TRUE(ReconcileGeneralAttributeKeys(kTfLiteBufferAttrMap, &lhs, &rhs,
                                              &merged, nullptr));
    EXPECT_EQ(4, GetAttr<size_t>(merged, TfLiteBufferAttributeKey::kAlignment));
  }

  {
    ContainerT lhs, rhs, merged;
    SetAttr(&rhs, TfLiteBufferAttributeKey::kAlignment, size_t(4));
    EXPECT_TRUE(ReconcileGeneralAttributeKeys(kTfLiteBufferAttrMap, &lhs, &rhs,
                                              &merged, nullptr));
    EXPECT_EQ(4, GetAttr<size_t>(merged, TfLiteBufferAttributeKey::kAlignment));
  }

  {
    ContainerT lhs, rhs, merged;
    const char value[] = "string";
    SetAttr(&rhs, TfLiteSyncAttributeKey::kSyncObjectTypeName, value);
    EXPECT_TRUE(ReconcileGeneralAttributeKeys(kTfLiteSyncAttrMap, &lhs, &rhs,
                                              &merged, nullptr));
    EXPECT_EQ(value, GetAttr<const char*>(
                         merged, TfLiteSyncAttributeKey::kSyncObjectTypeName));
  }
}

TEST(CheckCoverageTest, MissingAttributeTest) {
  {
    ContainerT lhs, rhs;
    SetAttr(&lhs, TfLiteBufferAttributeKey::kAlignment, size_t(4));
    EXPECT_TRUE(CheckGeneralAttributeKeysCoverage(kTfLiteBufferAttrMap, &lhs,
                                                  &rhs, nullptr));
  }

  {
    ContainerT lhs, rhs, merged;
    SetAttr(&rhs, TfLiteBufferAttributeKey::kAlignment, size_t(4));
    EXPECT_TRUE(ReconcileGeneralAttributeKeys(kTfLiteBufferAttrMap, &lhs, &rhs,
                                              &merged, nullptr));
    EXPECT_FALSE(CheckGeneralAttributeKeysCoverage(kTfLiteBufferAttrMap, &lhs,
                                                   &rhs, nullptr));
  }
}

class ReconcileAlignmentTest
    : public testing::TestWithParam<std::tuple<size_t, size_t, size_t>> {};

TEST_P(ReconcileAlignmentTest, Test) {
  ContainerT lhs, rhs, merged;
  SetAttr(&lhs, TfLiteBufferAttributeKey::kAlignment, std::get<0>(GetParam()));
  SetAttr(&rhs, TfLiteBufferAttributeKey::kAlignment, std::get<1>(GetParam()));
  EXPECT_TRUE(ReconcileGeneralAttributeKeys(kTfLiteBufferAttrMap, &lhs, &rhs,
                                            &merged, nullptr));
  EXPECT_EQ(std::get<2>(GetParam()),
            GetAttr<size_t>(merged, TfLiteBufferAttributeKey::kAlignment));
}

INSTANTIATE_TEST_SUITE_P(ReconcileAlignmentTest, ReconcileAlignmentTest,
                         testing::Values(std::make_tuple(4, 4, 4),
                                         std::make_tuple(1, 4, 4),
                                         std::make_tuple(8, 4, 8),
                                         std::make_tuple(8, 3, 24)));

class CheckAlignmentTest
    : public testing::TestWithParam<std::tuple<size_t, size_t, bool>> {};

TEST_P(CheckAlignmentTest, Test) {
  ContainerT lhs, rhs, conflict;
  SetAttr(&lhs, TfLiteBufferAttributeKey::kAlignment, std::get<0>(GetParam()));
  SetAttr(&rhs, TfLiteBufferAttributeKey::kAlignment, std::get<1>(GetParam()));
  EXPECT_EQ(std::get<2>(GetParam()),
            CheckGeneralAttributeKeysCoverage(kTfLiteBufferAttrMap, &lhs, &rhs,
                                              &conflict));
  EXPECT_EQ(!std::get<2>(GetParam()),
            conflict.count(
                static_cast<uint32_t>(TfLiteBufferAttributeKey::kAlignment)));
}

INSTANTIATE_TEST_SUITE_P(CheckAlignmentTest, CheckAlignmentTest,
                         testing::Values(std::make_tuple(4, 4, true),
                                         std::make_tuple(4, 1, true),
                                         std::make_tuple(1, 4, false)));

class ReconcilePaddingTest
    : public testing::TestWithParam<std::tuple<size_t, size_t, size_t>> {};

TEST_P(ReconcilePaddingTest, Test) {
  ContainerT lhs, rhs, merged;
  SetAttr(&lhs, TfLiteBufferAttributeKey::kPadding, std::get<0>(GetParam()));
  SetAttr(&rhs, TfLiteBufferAttributeKey::kPadding, std::get<1>(GetParam()));
  EXPECT_TRUE(ReconcileGeneralAttributeKeys(kTfLiteBufferAttrMap, &lhs, &rhs,
                                            &merged, nullptr));
  EXPECT_EQ(std::get<2>(GetParam()),
            GetAttr<size_t>(merged, TfLiteBufferAttributeKey::kPadding));
}

INSTANTIATE_TEST_SUITE_P(ReconcilePaddingTest, ReconcilePaddingTest,
                         testing::Values(std::make_tuple(4, 4, 4),
                                         std::make_tuple(1, 4, 4),
                                         std::make_tuple(8, 4, 8),
                                         std::make_tuple(8, 3, 24)));

class CheckPaddingTest
    : public testing::TestWithParam<std::tuple<size_t, size_t, bool>> {};

TEST_P(CheckPaddingTest, Test) {
  ContainerT lhs, rhs, conflict;
  SetAttr(&lhs, TfLiteBufferAttributeKey::kPadding, std::get<0>(GetParam()));
  SetAttr(&rhs, TfLiteBufferAttributeKey::kPadding, std::get<1>(GetParam()));
  EXPECT_EQ(std::get<2>(GetParam()),
            CheckGeneralAttributeKeysCoverage(kTfLiteBufferAttrMap, &lhs, &rhs,
                                              &conflict));
  EXPECT_EQ(!std::get<2>(GetParam()), conflict.count(static_cast<uint32_t>(
                                          TfLiteBufferAttributeKey::kPadding)));
}

INSTANTIATE_TEST_SUITE_P(CheckPaddingTest, CheckPaddingTest,
                         testing::Values(std::make_tuple(4, 4, true),
                                         std::make_tuple(4, 1, true),
                                         std::make_tuple(1, 4, false)));

class ReconcileSizeTest
    : public testing::TestWithParam<std::tuple<size_t, size_t, size_t>> {};

TEST_P(ReconcileSizeTest, Test) {
  ContainerT lhs, rhs, merged;
  SetAttr(&lhs, TfLiteBufferAttributeKey::kSize, std::get<0>(GetParam()));
  SetAttr(&rhs, TfLiteBufferAttributeKey::kSize, std::get<1>(GetParam()));
  EXPECT_TRUE(ReconcileGeneralAttributeKeys(kTfLiteBufferAttrMap, &lhs, &rhs,
                                            &merged, nullptr));
  EXPECT_EQ(std::get<2>(GetParam()),
            GetAttr<size_t>(merged, TfLiteBufferAttributeKey::kSize));
}

INSTANTIATE_TEST_SUITE_P(ReconcileSizeTest, ReconcileSizeTest,
                         testing::Values(std::make_tuple(4, 4, 4),
                                         std::make_tuple(1, 4, 4),
                                         std::make_tuple(8, 4, 8),
                                         std::make_tuple(8, 3, 8)));

class CheckSizeTest
    : public testing::TestWithParam<std::tuple<size_t, size_t, bool>> {};

TEST_P(CheckSizeTest, Test) {
  ContainerT lhs, rhs, conflict;
  SetAttr(&lhs, TfLiteBufferAttributeKey::kSize, std::get<0>(GetParam()));
  SetAttr(&rhs, TfLiteBufferAttributeKey::kSize, std::get<1>(GetParam()));
  EXPECT_EQ(std::get<2>(GetParam()),
            CheckGeneralAttributeKeysCoverage(kTfLiteBufferAttrMap, &lhs, &rhs,
                                              &conflict));
  EXPECT_EQ(
      !std::get<2>(GetParam()),
      conflict.count(static_cast<uint32_t>(TfLiteBufferAttributeKey::kSize)));
}

INSTANTIATE_TEST_SUITE_P(CheckSizeTest, CheckSizeTest,
                         testing::Values(std::make_tuple(4, 4, true),
                                         std::make_tuple(4, 1, true),
                                         std::make_tuple(1, 4, false)));

class ReconcileNameTest
    : public testing::TestWithParam<std::tuple<TfLiteAttrMapType, uint32_t>> {};

TEST_P(ReconcileNameTest, Test) {
  constexpr char name_string1[] = "string1";
  std::string name_string1_1 = "string1";
  constexpr char name_string2[] = "string2";
  {
    ContainerT lhs, rhs, merged;
    SetAttr(&lhs, std::get<1>(GetParam()), name_string1);
    SetAttr(&rhs, std::get<1>(GetParam()), name_string1_1.c_str());
    EXPECT_TRUE(ReconcileGeneralAttributeKeys(std::get<0>(GetParam()), &lhs,
                                              &rhs, &merged, nullptr));
    EXPECT_EQ(0, strcmp(GetAttr<const char*>(merged, std::get<1>(GetParam())),
                        name_string1));
  }
  {
    ContainerT lhs, rhs, merged, conflict;
    SetAttr(&lhs, std::get<1>(GetParam()), name_string1);
    SetAttr(&rhs, std::get<1>(GetParam()), name_string2);
    EXPECT_FALSE(ReconcileGeneralAttributeKeys(std::get<0>(GetParam()), &lhs,
                                               &rhs, &merged, &conflict));
    EXPECT_TRUE(conflict.count(std::get<1>(GetParam())));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ReconcileNameTest, ReconcileNameTest,
    testing::Values(
        std::make_tuple(kTfLiteBufferAttrMap,
                        static_cast<uint32_t>(
                            TfLiteBufferAttributeKey::kBufferResourceTypeName)),
        std::make_tuple(kTfLiteSyncAttrMap,
                        static_cast<uint32_t>(
                            TfLiteSyncAttributeKey::kSyncObjectTypeName))));

class CheckNameTest
    : public testing::TestWithParam<std::tuple<TfLiteAttrMapType, uint32_t>> {};

TEST_P(CheckNameTest, Test) {
  constexpr char name_string1[] = "string1";
  std::string name_string1_1 = "string1";
  constexpr char name_string2[] = "string2";
  {
    ContainerT lhs, rhs;
    SetAttr(&lhs, std::get<1>(GetParam()), name_string1);
    SetAttr(&rhs, std::get<1>(GetParam()), name_string1_1.c_str());
    EXPECT_TRUE(CheckGeneralAttributeKeysCoverage(std::get<0>(GetParam()), &lhs,
                                                  &rhs, nullptr));
  }
  {
    ContainerT lhs, rhs, conflict;
    SetAttr(&lhs, std::get<1>(GetParam()), name_string1);
    SetAttr(&rhs, std::get<1>(GetParam()), name_string2);
    EXPECT_FALSE(CheckGeneralAttributeKeysCoverage(std::get<0>(GetParam()),
                                                   &lhs, &rhs, &conflict));
    EXPECT_TRUE(conflict.count(std::get<1>(GetParam())));
  }
}

INSTANTIATE_TEST_SUITE_P(
    CheckNameTest, CheckNameTest,
    testing::Values(
        std::make_tuple(kTfLiteBufferAttrMap,
                        static_cast<uint32_t>(
                            TfLiteBufferAttributeKey::kBufferResourceTypeName)),
        std::make_tuple(kTfLiteSyncAttrMap,
                        static_cast<uint32_t>(
                            TfLiteSyncAttributeKey::kSyncObjectTypeName))));

}  // namespace
}  // namespace tflite::interop
