/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/util.h"

#include <stddef.h>
#include <stdlib.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using testing::ElementsAreArray;

TEST(ConvertVectorToTfLiteIntArray, TestWithVector) {
  std::vector<int> input = {1, 2};
  TfLiteIntArray* output = ConvertVectorToTfLiteIntArray(input);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->size, 2);
  EXPECT_EQ(output->data[0], 1);
  EXPECT_EQ(output->data[1], 2);
  TfLiteIntArrayFree(output);
}

TEST(ConvertVectorToTfLiteIntArray, TestWithEmptyVector) {
  std::vector<int> input;
  TfLiteIntArray* output = ConvertVectorToTfLiteIntArray(input);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->size, 0);
  TfLiteIntArrayFree(output);
}

TEST(UtilTest, IsFlexOp) {
  EXPECT_TRUE(IsFlexOp("Flex"));
  EXPECT_TRUE(IsFlexOp("FlexOp"));
  EXPECT_FALSE(IsFlexOp("flex"));
  EXPECT_FALSE(IsFlexOp("Fle"));
  EXPECT_FALSE(IsFlexOp("OpFlex"));
  EXPECT_FALSE(IsFlexOp(nullptr));
  EXPECT_FALSE(IsFlexOp(""));
}

TEST(EqualArrayAndTfLiteIntArray, TestWithTFLiteArrayEmpty) {
  int input[] = {1, 2, 3, 4};
  EXPECT_FALSE(EqualArrayAndTfLiteIntArray(nullptr, 4, input));
}

TEST(EqualArrayAndTfLiteIntArray, TestWithTFLiteArrayWrongSize) {
  int input[] = {1, 2, 3, 4};
  TfLiteIntArray* output = ConvertArrayToTfLiteIntArray(4, input);
  EXPECT_FALSE(EqualArrayAndTfLiteIntArray(output, 3, input));
  free(output);
}

TEST(EqualArrayAndTfLiteIntArray, TestMismatch) {
  int input[] = {1, 2, 3, 4};
  TfLiteIntArray* output = ConvertVectorToTfLiteIntArray({1, 2, 2, 4});
  EXPECT_FALSE(EqualArrayAndTfLiteIntArray(output, 4, input));
  free(output);
}

TEST(EqualArrayAndTfLiteIntArray, TestMatch) {
  int input[] = {1, 2, 3, 4};
  TfLiteIntArray* output = ConvertArrayToTfLiteIntArray(4, input);
  EXPECT_TRUE(EqualArrayAndTfLiteIntArray(output, 4, input));
  free(output);
}

TEST(CombineHashes, TestHashOutputsEquals) {
  size_t output1 = CombineHashes({1, 2, 3, 4});
  size_t output2 = CombineHashes({1, 2, 3, 4});
  EXPECT_EQ(output1, output2);
}

TEST(CombineHashes, TestHashOutputsDifferent) {
  size_t output1 = CombineHashes({1, 2, 3, 4});
  size_t output2 = CombineHashes({1, 2, 2, 4});
  EXPECT_NE(output1, output2);
}

TEST(GetOpNameByRegistration, ValidBuiltinCode) {
  TfLiteRegistration registration{};
  registration.builtin_code = tflite::BuiltinOperator_ADD;
  const auto op_name = GetOpNameByRegistration(registration);
  EXPECT_EQ("ADD", op_name);
}

TEST(GetOpNameByRegistration, InvalidBuiltinCode) {
  TfLiteRegistration registration{};
  registration.builtin_code = -1;
  const auto op_name = GetOpNameByRegistration(registration);
  EXPECT_EQ("", op_name);
}

TEST(GetOpNameByRegistration, CustomName) {
  TfLiteRegistration registration{};
  registration.builtin_code = tflite::BuiltinOperator_CUSTOM;
  registration.custom_name = "TestOp";
  auto op_name = GetOpNameByRegistration(registration);
  EXPECT_EQ("CUSTOM TestOp", op_name);

  registration.builtin_code = tflite::BuiltinOperator_DELEGATE;
  registration.custom_name = "TestDelegate";
  op_name = GetOpNameByRegistration(registration);
  EXPECT_EQ("DELEGATE TestDelegate", op_name);
}

TEST(ValidationSubgraph, NameIsDetected) {
  EXPECT_FALSE(IsValidationSubgraph(nullptr));
  EXPECT_FALSE(IsValidationSubgraph(""));
  EXPECT_FALSE(IsValidationSubgraph("a name"));
  EXPECT_FALSE(IsValidationSubgraph("VALIDATIONfoo"));
  EXPECT_TRUE(IsValidationSubgraph("VALIDATION:"));
  EXPECT_TRUE(IsValidationSubgraph("VALIDATION:main"));
}

TEST(MultiplyAndCheckOverflow, Validate) {
  size_t res = 0;
  EXPECT_TRUE(MultiplyAndCheckOverflow(1, 2, &res) == kTfLiteOk);
  EXPECT_FALSE(MultiplyAndCheckOverflow(static_cast<size_t>(123456789023),
                                        1223423425, &res) == kTfLiteOk);
}

TEST(CheckedNumElements, Validate) {
  std::vector<int> dims = {2, 3, 4};
  size_t count = 0;

  EXPECT_EQ(CheckedNumElements(dims, count), kTfLiteOk);
  EXPECT_EQ(count, 24);
}

TEST(CheckedNumElements, RejectsNegativeDimension) {
  std::vector<int> dims = {2, -1, 4};
  size_t count = 0;

  EXPECT_EQ(CheckedNumElements(dims, count), kTfLiteError);
}

TEST(CheckedNumElements, RejectsOverflow) {
  std::vector<int> dims = {std::numeric_limits<int>::max(),
                           std::numeric_limits<int>::max(),
                           std::numeric_limits<int>::max()};
  size_t count = 0;

  EXPECT_EQ(CheckedNumElements(dims, count), kTfLiteError);
}

TEST(CheckedNumElementsToInt, Validate) {
  std::vector<int> dims = {2, 3, 4};
  int count = 0;

  EXPECT_EQ(CheckedNumElements(dims, count), kTfLiteOk);
  EXPECT_EQ(count, 24);
}

TEST(CheckedNumElementsToInt, RejectsOverflow) {
  std::vector<int> dims = {std::numeric_limits<int>::max(),
                           std::numeric_limits<int>::max()};
  int count = 0;

  EXPECT_EQ(CheckedNumElements(dims, count), kTfLiteError);
}

TEST(FourBitTest, BytesRequiredEven) {
  TfLiteContext context;

  int dims[] = {2, 3, 1, 5};
  const int* dims_ptr = &dims[0];
  size_t dims_size = 4;
  size_t required_bytes_four_bit;
  tflite::BytesRequired(kTfLiteInt4, dims_ptr, dims_size,
                        &required_bytes_four_bit, &context);

  ASSERT_EQ(required_bytes_four_bit, 15);
}

TEST(FourBitTest, BytesRequiredOdd) {
  TfLiteContext context;

  int dims[] = {5, 1, 1, 1};
  const int* dims_ptr = &dims[0];
  size_t dims_size = 2;
  size_t required_bytes_four_bit;
  tflite::BytesRequired(kTfLiteInt4, dims_ptr, dims_size,
                        &required_bytes_four_bit, &context);

  ASSERT_EQ(required_bytes_four_bit, 3);
}

void SilentReportError(TfLiteContext*, const char*, ...) {}

TEST(FourBitTest, BytesRequiredSizeMaxBoundaryDoesNotWrapToZero) {
  TfLiteContext context{};
  context.ReportError = SilentReportError;

  // Fermat prime factorization of SIZE_MAX on 64-bit (2^64 - 1) and 32-bit
  // (2^32 - 1) platforms. Every dimension fits in a positive 32-bit signed int.
#if SIZE_MAX == UINT64_MAX
  constexpr int kSizeMaxDims[] = {3, 5, 17, 257, 641, 65537, 6700417};
  // 2^64 - 2 = 2 * 7 * 7 * 73 * 127 * 337 * 92737 * 649657.
  constexpr int kSizeMaxMinusOneDims[] = {14, 7, 73, 127, 337, 92737, 649657};
#else
  constexpr int kSizeMaxDims[] = {3, 5, 17, 257, 65537};
  // 2^32 - 2 = 2 * (2^31 - 1) = 2 * 2147483647.
  constexpr int kSizeMaxMinusOneDims[] = {2, 2147483647};
#endif
  constexpr size_t kNumDims = sizeof(kSizeMaxDims) / sizeof(kSizeMaxDims[0]);

  for (TfLiteType type : {kTfLiteInt4, kTfLiteUInt4}) {
    size_t required_bytes = 0;
    ASSERT_EQ(tflite::BytesRequired(type, kSizeMaxDims, kNumDims,
                                    &required_bytes, &context),
              kTfLiteOk);
    EXPECT_EQ(required_bytes, (std::numeric_limits<size_t>::max() / 2) + 1);
  }

  {
    size_t required_bytes = 0;
    ASSERT_EQ(tflite::BytesRequired(kTfLiteInt2, kSizeMaxDims, kNumDims,
                                    &required_bytes, &context),
              kTfLiteOk);
    EXPECT_EQ(required_bytes, (std::numeric_limits<size_t>::max() / 4) + 1);
  }

  {
    size_t required_bytes = 0;
    constexpr size_t kNumMinusOneDims =
        sizeof(kSizeMaxMinusOneDims) / sizeof(kSizeMaxMinusOneDims[0]);
    ASSERT_EQ(
        tflite::BytesRequired(kTfLiteInt2, kSizeMaxMinusOneDims,
                              kNumMinusOneDims, &required_bytes, &context),
        kTfLiteOk);
    EXPECT_EQ(required_bytes,
              ((std::numeric_limits<size_t>::max() - 1) / 4) + 1);
  }
}

TEST(BytesRequiredTest, RejectsUnsupportedTypes) {
  TfLiteContext context{};
  context.ReportError = SilentReportError;
  constexpr int kDims[] = {2, 4};

  for (TfLiteType type :
       {kTfLiteNoType, kTfLiteString, kTfLiteResource, kTfLiteVariant}) {
    size_t required_bytes = 0;
    EXPECT_EQ(tflite::BytesRequired(type, kDims, 2, &required_bytes, &context),
              kTfLiteError);
  }
}

TEST(TestMakeUniqueTensor, Valid) {
  TensorUniquePtr t = BuildTfLiteTensor(kTfLiteInt32, {2, 3}, kTfLiteDynamic);
  ASSERT_NE(t.get(), nullptr);
  ASSERT_EQ(t->buffer_handle, kTfLiteNullBufferHandle);

  EXPECT_THAT(t.get(), DimsAre({2, 3}));
  EXPECT_EQ(t->bytes, 24);

  EXPECT_EQ(t->type, kTfLiteInt32);
  EXPECT_EQ(t->allocation_type, kTfLiteDynamic);

  // Check memory has been properly allocated.
  int* data = t->data.i32;
  std::fill_n(data, 6, 0);
  ASSERT_NE(data, nullptr);
  ASSERT_THAT(std::vector<int>(data, data + 6),
              ElementsAreArray({0, 0, 0, 0, 0, 0}));
}

TEST(TestMakeUniqueTensor, NullDimsReturnsNull) {
  TensorUniquePtr t = BuildTfLiteTensor(kTfLiteInt32, nullptr, kTfLiteDynamic);
  ASSERT_EQ(t.get(), nullptr);
}

}  // namespace
}  // namespace tflite
