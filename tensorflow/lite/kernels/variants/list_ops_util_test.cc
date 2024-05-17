/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/variants/list_ops_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/array.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace variants {
namespace {

TEST(TensorAsShape, ScalarTensor_ReturnsEmptyIntArray) {
  TensorUniquePtr scalar_tensor =
      BuildTfLiteTensor(kTfLiteInt32, BuildTfLiteArray({}), kTfLiteDynamic);

  IntArrayUniquePtr shape_from_tensor = TensorAsShape(*scalar_tensor);
  ASSERT_THAT(shape_from_tensor.get(), DimsAre({}));
}

TEST(TensorAsShape, SingleElementTensor_ReturnsSize1Shape) {
  TensorUniquePtr single_el_tensor =
      BuildTfLiteTensor(kTfLiteInt32, BuildTfLiteArray({1}), kTfLiteDynamic);
  single_el_tensor->data.i32[0] = 10;

  IntArrayUniquePtr shape_from_tensor = TensorAsShape(*single_el_tensor);
  ASSERT_THAT(shape_from_tensor.get(), DimsAre({10}));
}

TEST(TensorAsShape, OneDMultipleElementShape_ReturnsHighRankedShape) {
  TensorUniquePtr one_d_mul_el_tensor =
      BuildTfLiteTensor(kTfLiteInt32, BuildTfLiteArray({3}), kTfLiteDynamic);
  one_d_mul_el_tensor->data.i32[0] = 10;
  one_d_mul_el_tensor->data.i32[1] = 9;
  one_d_mul_el_tensor->data.i32[2] = 8;

  IntArrayUniquePtr shape_from_tensor = TensorAsShape(*one_d_mul_el_tensor);
  ASSERT_THAT(shape_from_tensor.get(), DimsAre({10, 9, 8}));
}

TEST(MergeShapesOrNull, IncompatibleSameRank_ReturnsNull) {
  IntArrayUniquePtr l = BuildTfLiteArray({2, 3});
  IntArrayUniquePtr r = BuildTfLiteArray({3, 3});
  EXPECT_EQ(MergeShapesOrNull(std::move(l), std::move(r)).get(), nullptr);
}

TEST(MergeShapesOrNull, NotSameRank_ReturnsNull) {
  IntArrayUniquePtr l = BuildTfLiteArray({1});
  IntArrayUniquePtr r = BuildTfLiteArray({1, 2});
  EXPECT_EQ(MergeShapesOrNull(std::move(l), std::move(r)).get(), nullptr);
}

TEST(MergeShapesOrNull, MergeShapesOrNullSameRankNENull) {
  IntArrayUniquePtr l = BuildTfLiteArray({1});
  IntArrayUniquePtr r = BuildTfLiteArray({2});
  EXPECT_EQ(MergeShapesOrNull(std::move(l), std::move(r)).get(), nullptr);
}

TEST(MergeShapesOrNull, RankedUnknownLKnownR_ReturnsStatic) {
  IntArrayUniquePtr l = BuildTfLiteArray({-1});
  IntArrayUniquePtr r = BuildTfLiteArray({2});
  EXPECT_THAT(MergeShapesOrNull(std::move(l), std::move(r)).get(),
              DimsAre({2}));
}

TEST(MergeShapesOrNull, UnknownRKnownL_ReturnsStatic) {
  IntArrayUniquePtr l = BuildTfLiteArray({2});
  IntArrayUniquePtr r = BuildTfLiteArray({-1});
  EXPECT_THAT(MergeShapesOrNull(std::move(l), std::move(r)).get(),
              DimsAre({2}));
}

TEST(MergeShapesOrNull, UnknownBoth_ReturnsUnknown) {
  IntArrayUniquePtr l = BuildTfLiteArray({-1});
  IntArrayUniquePtr r = BuildTfLiteArray({-1});
  EXPECT_THAT(MergeShapesOrNull(std::move(l), std::move(r)).get(),
              DimsAre({-1}));
}

TEST(MergeShapesOrNull, RankedUnknownDifferentDims_ConstrainsUnknownDims) {
  IntArrayUniquePtr l = BuildTfLiteArray({-1, 2, 5});
  IntArrayUniquePtr r = BuildTfLiteArray({1, -1, 5});
  EXPECT_THAT(MergeShapesOrNull(std::move(l), std::move(r)).get(),
              DimsAre({1, 2, 5}));
}

TEST(MergeShapesOrNull, BothUnranked_ReturnsUnranked) {
  IntArrayUniquePtr l = BuildTfLiteArray({});
  IntArrayUniquePtr r = BuildTfLiteArray({});
  EXPECT_THAT(MergeShapesOrNull(std::move(l), std::move(r)).get(), DimsAre({}));
}

TEST(MergeShapesOrNull, UrankedAndStatic1D_ReturnsStatic1D) {
  IntArrayUniquePtr l = BuildTfLiteArray({});
  IntArrayUniquePtr r = BuildTfLiteArray({1});
  EXPECT_THAT(MergeShapesOrNull(std::move(l), std::move(r)).get(),
              DimsAre({1}));
}

TEST(MergeShapesOrNull, UnrankedAndStaticND_ReturnsStaticND) {
  IntArrayUniquePtr l = BuildTfLiteArray({});
  IntArrayUniquePtr r = BuildTfLiteArray({2, 3});
  EXPECT_THAT(MergeShapesOrNull(std::move(l), std::move(r)).get(),
              DimsAre({2, 3}));
}

TEST(MergeShapesOrNull, UnrankedAndRankedUnknown_ReturnsRankedUnknown) {
  IntArrayUniquePtr l = BuildTfLiteArray({});
  IntArrayUniquePtr r = BuildTfLiteArray({-1});
  EXPECT_THAT(MergeShapesOrNull(std::move(l), std::move(r)).get(),
              DimsAre({-1}));
}

TEST(MergeShapesOrNull, NullInput_ReturnsOther) {
  EXPECT_THAT(MergeShapesOrNull(BuildTfLiteArray({3}), nullptr).get(),
              DimsAre({3}));
  EXPECT_THAT(MergeShapesOrNull(nullptr, BuildTfLiteArray({2})).get(),
              DimsAre({2}));
  EXPECT_EQ(MergeShapesOrNull(nullptr, nullptr).get(), nullptr);
}

TEST(MergeShapesOrNull, NullInput_ReturnsUnrankedOther) {
  EXPECT_THAT(MergeShapesOrNull(BuildTfLiteArray({}), nullptr).get(),
              DimsAre({}));
  EXPECT_THAT(MergeShapesOrNull(nullptr, BuildTfLiteArray({})).get(),
              DimsAre({}));
}

TEST(ElementsSameShape, NoElements_SucceedsWithNullptr) {
  TensorArray arr = {kTfLiteInt32, BuildTfLiteArray({})};
  arr.Resize(2);
  IntArrayUniquePtr res;
  ASSERT_EQ(GetShapeIfAllEqual(arr, res), kTfLiteOk);
  EXPECT_EQ(res.get(), nullptr);
}

TEST(ElementsSameShape, ZeroSize_SucceedsWithNullptr) {
  TensorArray arr = {kTfLiteInt32, BuildTfLiteArray({})};
  IntArrayUniquePtr res;
  ASSERT_EQ(GetShapeIfAllEqual(arr, res), kTfLiteOk);
  EXPECT_EQ(res.get(), nullptr);
}

TEST(ElementsSameShape, OneSize_SucceedsWithShape) {
  TensorArray arr = {kTfLiteInt32, BuildTfLiteArray({})};
  arr.Resize(1);
  arr.Set(0, BuildTfLiteTensor(kTfLiteInt32, BuildTfLiteArray({2}),
                               kTfLiteDynamic));
  IntArrayUniquePtr res;
  ASSERT_EQ(GetShapeIfAllEqual(arr, res), kTfLiteOk);
  EXPECT_THAT(res.get(), DimsAre({2}));
}

TEST(ElementsSameShape, MultipleElements_AllSet_SucceedsWithShape) {
  TensorArray arr = {kTfLiteInt32, BuildTfLiteArray({})};
  arr.Resize(2);
  arr.Set(0, BuildTfLiteTensor(kTfLiteInt32, BuildTfLiteArray({2}),
                               kTfLiteDynamic));
  arr.Set(1, BuildTfLiteTensor(kTfLiteInt32, BuildTfLiteArray({2}),
                               kTfLiteDynamic));
  IntArrayUniquePtr res;
  EXPECT_EQ(GetShapeIfAllEqual(arr, res), kTfLiteOk);
  EXPECT_THAT(res.get(), DimsAre({2}));
}

TEST(ElementsSameShape, MultipleElements_SomeSet_SucceedsWithShape) {
  TensorArray arr = {kTfLiteInt32, BuildTfLiteArray({})};
  arr.Resize(3);
  arr.Set(0, BuildTfLiteTensor(kTfLiteInt32, BuildTfLiteArray({2}),
                               kTfLiteDynamic));
  arr.Set(2, BuildTfLiteTensor(kTfLiteInt32, BuildTfLiteArray({2}),
                               kTfLiteDynamic));
  IntArrayUniquePtr res;
  EXPECT_EQ(GetShapeIfAllEqual(arr, res), kTfLiteOk);
  EXPECT_THAT(res.get(), DimsAre({2}));
}

TEST(ElementsSameShape, MultipleElements_SomeSetNotSameRank_Fails) {
  TensorArray arr = {kTfLiteInt32, BuildTfLiteArray({})};
  arr.Resize(3);
  arr.Set(0, BuildTfLiteTensor(kTfLiteInt32, BuildTfLiteArray({2}),
                               kTfLiteDynamic));
  arr.Set(2, BuildTfLiteTensor(kTfLiteInt32, BuildTfLiteArray({2, 3}),
                               kTfLiteDynamic));
  IntArrayUniquePtr res;
  EXPECT_EQ(GetShapeIfAllEqual(arr, res), kTfLiteError);
}

TEST(ElementsSameShape, MultipleElements_SomeSetNotSameDim_Fails) {
  TensorArray arr = {kTfLiteInt32, BuildTfLiteArray({})};
  arr.Resize(3);
  arr.Set(0, BuildTfLiteTensor(kTfLiteInt32, BuildTfLiteArray({2, 2}),
                               kTfLiteDynamic));
  arr.Set(2, BuildTfLiteTensor(kTfLiteInt32, BuildTfLiteArray({2, 3}),
                               kTfLiteDynamic));
  IntArrayUniquePtr res;
  EXPECT_EQ(GetShapeIfAllEqual(arr, res), kTfLiteError);
}

}  // namespace
}  // namespace variants
}  // namespace tflite
