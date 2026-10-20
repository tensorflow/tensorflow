/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/kernel_utils.h"

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/array.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace kernel_utils {
namespace {

using ::testing::HasSubstr;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

// Minimal TfLiteContext implementation that records reported error messages and
// supports resizing tensor dimensions in unit tests.
struct TestContext : public TfLiteContext {
  TestContext() : TfLiteContext{} {
    this->ReportError = ReportErrorImpl;
    this->ResizeTensor = ResizeTensorImpl;
  }

  static void ReportErrorImpl(TfLiteContext* context, const char* format, ...) {
    TestContext* test_context = static_cast<TestContext*>(context);
    char buffer[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    test_context->error = buffer;
  }

  static TfLiteStatus ResizeTensorImpl(TfLiteContext* context,
                                       TfLiteTensor* tensor,
                                       TfLiteIntArray* new_size) {
    if (tensor->dims != nullptr) {
      TfLiteIntArrayFree(tensor->dims);
    }
    tensor->dims = new_size;
    return kTfLiteOk;
  }

  std::string error;
};

// Creates a TfLiteTensor with the given dimensions without allocating a backing
// data buffer, allowing tests to construct arbitrary (including negative or
// overflowing) shapes safely.
TensorUniquePtr MakeShapeOnlyTensor(absl::Span<const int> dims,
                                    TfLiteType type = kTfLiteFloat32) {
  TensorUniquePtr tensor = BuildTfLiteTensor();
  tensor->type = type;
  tensor->dims =
      BuildTfLiteArray(std::vector<int>(dims.begin(), dims.end())).release();
  return tensor;
}

struct ReadAndNormalizeAxisValidCase {
  std::string test_name;
  int32_t axis_value;
  int rank;
  int expected_axis;
};

using ReadAndNormalizeAxisValidTest =
    TestWithParam<ReadAndNormalizeAxisValidCase>;

TEST_P(ReadAndNormalizeAxisValidTest, NormalizesInt32AndInt64Axis) {
  const ReadAndNormalizeAxisValidCase& test_case = GetParam();
  TestContext context;

  int32_t axis_i32 = test_case.axis_value;
  TensorUniquePtr axis_tensor_i32 = MakeShapeOnlyTensor({1}, kTfLiteInt32);
  axis_tensor_i32->data.i32 = &axis_i32;
  int normalized_i32 = -1;
  EXPECT_EQ(ReadAndNormalizeAxis(&context, *axis_tensor_i32, test_case.rank,
                                 normalized_i32),
            kTfLiteOk);
  EXPECT_EQ(normalized_i32, test_case.expected_axis);

  int64_t axis_i64 = test_case.axis_value;
  TensorUniquePtr axis_tensor_i64 = MakeShapeOnlyTensor({1}, kTfLiteInt64);
  axis_tensor_i64->data.i64 = &axis_i64;
  int normalized_i64 = -1;
  EXPECT_EQ(ReadAndNormalizeAxis(&context, *axis_tensor_i64, test_case.rank,
                                 normalized_i64),
            kTfLiteOk);
  EXPECT_EQ(normalized_i64, test_case.expected_axis);
}

INSTANTIATE_TEST_SUITE_P(
    ReadAndNormalizeAxisValidTests, ReadAndNormalizeAxisValidTest,
    ValuesIn<ReadAndNormalizeAxisValidCase>({
        {"FirstPositiveAxis", 0, 3, 0},
        {"MiddlePositiveAxis", 1, 3, 1},
        {"LastPositiveAxis", 2, 3, 2},
        {"NegativeOneAxis", -1, 3, 2},
        {"NegativeRankAxis", -3, 3, 0},
    }),
    [](const TestParamInfo<ReadAndNormalizeAxisValidCase>& info) {
      return info.param.test_name;
    });

struct ReadAndNormalizeAxisOutOfBoundsCase {
  std::string test_name;
  int64_t axis_value;
  int rank;
};

using ReadAndNormalizeAxisOutOfBoundsTest =
    TestWithParam<ReadAndNormalizeAxisOutOfBoundsCase>;

TEST_P(ReadAndNormalizeAxisOutOfBoundsTest, RejectsOutOfBoundsAxis) {
  const ReadAndNormalizeAxisOutOfBoundsCase& test_case = GetParam();
  TestContext context;
  int64_t axis_i64 = test_case.axis_value;
  TensorUniquePtr axis_tensor = MakeShapeOnlyTensor({1}, kTfLiteInt64);
  axis_tensor->data.i64 = &axis_i64;

  int normalized_axis = -1;
  EXPECT_EQ(ReadAndNormalizeAxis(&context, *axis_tensor, test_case.rank,
                                 normalized_axis),
            kTfLiteError);
  EXPECT_THAT(context.error, HasSubstr("Invalid axis value."));
}

INSTANTIATE_TEST_SUITE_P(
    ReadAndNormalizeAxisOutOfBoundsTests, ReadAndNormalizeAxisOutOfBoundsTest,
    ValuesIn<ReadAndNormalizeAxisOutOfBoundsCase>({
        {"EqualToRank", 3, 3},
        {"GreaterThanRank", 5, 3},
        {"LessThanNegativeRank", -4, 3},
        {"ZeroRank", 0, 0},
        // Lower 32 bits equal 1, ensuring all 64 bits are inspected.
        {"Int64TruncatedToValidInt32", (int64_t{1} << 32) + 1, 4},
    }),
    [](const TestParamInfo<ReadAndNormalizeAxisOutOfBoundsCase>& info) {
      return info.param.test_name;
    });

TEST(ReadAndNormalizeAxisTest, RejectsNegativeRankNullDataAndUnsupportedType) {
  TestContext context;
  int32_t axis_value = 0;
  TensorUniquePtr axis = MakeShapeOnlyTensor({1}, kTfLiteInt32);
  axis->data.i32 = &axis_value;

  int normalized_axis = -1;
  EXPECT_EQ(ReadAndNormalizeAxis(&context, *axis, /*rank=*/-1, normalized_axis),
            kTfLiteError);

  axis->data.raw = nullptr;
  EXPECT_EQ(ReadAndNormalizeAxis(&context, *axis, /*rank=*/3, normalized_axis),
            kTfLiteError);
  EXPECT_THAT(context.error, HasSubstr("Axis data is null."));

  axis->type = kTfLiteFloat32;
  float float_axis = 0.0f;
  axis->data.f = &float_axis;
  EXPECT_EQ(ReadAndNormalizeAxis(&context, *axis, /*rank=*/3, normalized_axis),
            kTfLiteError);
  EXPECT_THAT(context.error, HasSubstr("Unsupported axis type"));
}

struct CheckedDimensionProductValidCase {
  std::string test_name;
  std::vector<int> dims;
  int begin;
  int end;
  int expected_product;
};

using CheckedDimensionProductValidTest =
    TestWithParam<CheckedDimensionProductValidCase>;

TEST_P(CheckedDimensionProductValidTest, ComputesProductForIntAndSizeT) {
  const CheckedDimensionProductValidCase& test_case = GetParam();
  TestContext context;
  TensorUniquePtr tensor = MakeShapeOnlyTensor(test_case.dims);

  int int_product = -1;
  EXPECT_EQ(CheckedDimensionProduct(&context, *tensor, test_case.begin,
                                    test_case.end, int_product),
            kTfLiteOk);
  EXPECT_EQ(int_product, test_case.expected_product);

  size_t size_product = 999;
  EXPECT_EQ(CheckedDimensionProduct(&context, *tensor, test_case.begin,
                                    test_case.end, size_product),
            kTfLiteOk);
  EXPECT_EQ(size_product, static_cast<size_t>(test_case.expected_product));
}

INSTANTIATE_TEST_SUITE_P(
    CheckedDimensionProductValidTests, CheckedDimensionProductValidTest,
    ValuesIn<CheckedDimensionProductValidCase>({
        {"FullRange", {2, 3, 4, 5}, 0, 4, 120},
        {"SubRange", {2, 3, 4, 5}, 1, 3, 12},
        {"SingleDimension", {2, 3, 4, 5}, 2, 3, 4},
        {"EmptySubRange", {2, 3, 4, 5}, 2, 2, 1},
        {"ScalarEmptyRange", {}, 0, 0, 1},
        {"ZeroDimensionInRange", {2, 0, 5}, 0, 3, 0},
        {"ZeroDimensionOutsideRange", {2, 0, 5}, 2, 3, 5},
    }),
    [](const TestParamInfo<CheckedDimensionProductValidCase>& info) {
      return info.param.test_name;
    });

struct CheckedDimensionProductInvalidCase {
  std::string test_name;
  std::vector<int> dims;
  int begin;
  int end;
  std::string expected_error_substr = "";
};

using CheckedDimensionProductInvalidTest =
    TestWithParam<CheckedDimensionProductInvalidCase>;

TEST_P(CheckedDimensionProductInvalidTest, RejectsInvalidRangeOrOverflow) {
  const CheckedDimensionProductInvalidCase& test_case = GetParam();
  TestContext context;
  TensorUniquePtr tensor = MakeShapeOnlyTensor(test_case.dims);

  int product = 0;
  EXPECT_EQ(CheckedDimensionProduct(&context, *tensor, test_case.begin,
                                    test_case.end, product),
            kTfLiteError);
  if (!test_case.expected_error_substr.empty()) {
    EXPECT_THAT(context.error, HasSubstr(test_case.expected_error_substr));
  }
}

INSTANTIATE_TEST_SUITE_P(
    CheckedDimensionProductInvalidTests, CheckedDimensionProductInvalidTest,
    ValuesIn<CheckedDimensionProductInvalidCase>({
        {"NegativeBegin", {2, 3, 4}, -1, 2},
        {"EndBeforeBegin", {2, 3, 4}, 2, 1},
        {"EndExceedsRank", {2, 3, 4}, 0, 4},
        {"NegativeDimensionInRange", {2, -3, 4}, 0, 2},
        {"IntOverflow", {65536, 65536}, 0, 2, "Dimension product overflowed."},
        // Intermediate overflow followed by a zero dimension must still fail.
        {"IntermediateIntOverflowBeforeZero",
         {65536, 65536, 0},
         0,
         3,
         "Dimension product overflowed."},
    }),
    [](const TestParamInfo<CheckedDimensionProductInvalidCase>& info) {
      return info.param.test_name;
    });

TEST(CheckedDimensionProductTest, RejectsNullDimsAndHandlesSizeTOverflow) {
  TestContext context;
  TensorUniquePtr null_dims_tensor = BuildTfLiteTensor();
  int int_product = 0;
  EXPECT_EQ(
      CheckedDimensionProduct(&context, *null_dims_tensor, 0, 0, int_product),
      kTfLiteError);

  if constexpr (sizeof(size_t) >= 8) {
    TensorUniquePtr large_tensor = MakeShapeOnlyTensor({65536, 65536});
    size_t size_product = 0;
    EXPECT_EQ(
        CheckedDimensionProduct(&context, *large_tensor, 0, 2, size_product),
        kTfLiteOk);
    EXPECT_EQ(size_product, size_t{65536} * size_t{65536});
  }

  const int max_int = std::numeric_limits<int>::max();
  TensorUniquePtr size_t_overflow =
      MakeShapeOnlyTensor({max_int, max_int, max_int, 0});
  size_t size_product = 0;
  EXPECT_EQ(
      CheckedDimensionProduct(&context, *size_t_overflow, 0, 4, size_product),
      kTfLiteError);
  EXPECT_THAT(context.error, HasSubstr("Dimension product overflowed."));
}

struct ResizeLookupOutputTensorValidCase {
  std::string test_name;
  std::vector<int> lookup_dims;
  std::vector<int> value_dims;
  std::vector<int> expected_output_dims;
};

using ResizeLookupOutputTensorValidTest =
    TestWithParam<ResizeLookupOutputTensorValidCase>;

TEST_P(ResizeLookupOutputTensorValidTest,
       ResizesOutputToLookupAndTrailingValueDims) {
  const ResizeLookupOutputTensorValidCase& test_case = GetParam();
  TestContext context;
  TensorUniquePtr lookup =
      MakeShapeOnlyTensor(test_case.lookup_dims, kTfLiteInt32);
  TensorUniquePtr value =
      MakeShapeOnlyTensor(test_case.value_dims, kTfLiteFloat32);
  TensorUniquePtr output = BuildTfLiteTensor();

  EXPECT_EQ(ResizeLookupOutputTensor(&context, *lookup, *value, output.get()),
            kTfLiteOk);
  EXPECT_THAT(output.get(), DimsAre(test_case.expected_output_dims));
}

INSTANTIATE_TEST_SUITE_P(
    ResizeLookupOutputTensorValidTests, ResizeLookupOutputTensorValidTest,
    ValuesIn<ResizeLookupOutputTensorValidCase>({
        {"OneDimValue", {4}, {10}, {4}},
        {"TwoDimValue", {4}, {10, 3}, {4, 3}},
        {"ThreeDimValue", {4}, {10, 3, 5}, {4, 3, 5}},
        {"ZeroSizedDimensions", {0}, {10, 0, 5}, {0, 0, 5}},
    }),
    [](const TestParamInfo<ResizeLookupOutputTensorValidCase>& info) {
      return info.param.test_name;
    });

struct ResizeLookupOutputTensorInvalidRankCase {
  std::string test_name;
  std::vector<int> lookup_dims;
  std::vector<int> value_dims;
};

using ResizeLookupOutputTensorInvalidRankTest =
    TestWithParam<ResizeLookupOutputTensorInvalidRankCase>;

TEST_P(ResizeLookupOutputTensorInvalidRankTest, RejectsInvalidRanks) {
  const ResizeLookupOutputTensorInvalidRankCase& test_case = GetParam();
  TestContext context;
  TensorUniquePtr lookup =
      MakeShapeOnlyTensor(test_case.lookup_dims, kTfLiteInt32);
  TensorUniquePtr value =
      MakeShapeOnlyTensor(test_case.value_dims, kTfLiteFloat32);
  TensorUniquePtr output = BuildTfLiteTensor();

  EXPECT_EQ(ResizeLookupOutputTensor(&context, *lookup, *value, output.get()),
            kTfLiteError);
}

INSTANTIATE_TEST_SUITE_P(
    ResizeLookupOutputTensorInvalidRankTests,
    ResizeLookupOutputTensorInvalidRankTest,
    ValuesIn<ResizeLookupOutputTensorInvalidRankCase>({
        {"ScalarLookup", {}, {10, 3}},
        {"TwoDimLookup", {4, 2}, {10, 3}},
        {"ScalarValue", {4}, {}},
    }),
    [](const TestParamInfo<ResizeLookupOutputTensorInvalidRankCase>& info) {
      return info.param.test_name;
    });

TEST(ResizeLookupOutputTensorTest, RejectsNullPointers) {
  TestContext context;
  TensorUniquePtr lookup = MakeShapeOnlyTensor({4}, kTfLiteInt32);
  TensorUniquePtr value = MakeShapeOnlyTensor({10, 3}, kTfLiteFloat32);
  TensorUniquePtr null_dims = BuildTfLiteTensor();
  TensorUniquePtr output = BuildTfLiteTensor();

  EXPECT_EQ(
      ResizeLookupOutputTensor(&context, *lookup, *value, /*output=*/nullptr),
      kTfLiteError);
  EXPECT_EQ(
      ResizeLookupOutputTensor(&context, *null_dims, *value, output.get()),
      kTfLiteError);
  EXPECT_EQ(
      ResizeLookupOutputTensor(&context, *lookup, *null_dims, output.get()),
      kTfLiteError);
}

TEST(TensorElementCountAndDataValidationTest,
     ValidatesElementCountLimitsAndDataBuffer) {
  TestContext context;
  int32_t dummy_storage = 0;

  TensorUniquePtr valid_tensor = MakeShapeOnlyTensor({2, 3}, kTfLiteInt32);
  valid_tensor->data.i32 = &dummy_storage;

  size_t size_count = 0;
  EXPECT_EQ(GetTensorElementCountAndValidateData(
                &context, *valid_tensor, "Input", ElementCountLimit::kSizeT,
                size_count),
            kTfLiteOk);
  EXPECT_EQ(size_count, 6u);

  int int_count = 0;
  EXPECT_EQ(
      CheckedTensorElementsAndData(&context, *valid_tensor, "Input", int_count),
      kTfLiteOk);
  EXPECT_EQ(int_count, 6);
  EXPECT_EQ(ValidateTensorElementsAndData(&context, *valid_tensor, "Input"),
            kTfLiteOk);

  // Empty tensors are allowed to have a null data buffer.
  TensorUniquePtr empty_tensor = MakeShapeOnlyTensor({2, 0, 3}, kTfLiteInt32);
  EXPECT_EQ(
      CheckedTensorElementsAndData(&context, *empty_tensor, "Empty", int_count),
      kTfLiteOk);
  EXPECT_EQ(int_count, 0);

  // Non-empty tensors with null data must be rejected.
  valid_tensor->data.raw = nullptr;
  EXPECT_EQ(ValidateTensorElementsAndData(&context, *valid_tensor, "Missing"),
            kTfLiteError);
  EXPECT_THAT(context.error, HasSubstr("Missing data is null."));

  // Invalid shapes (negative dimension) must be rejected.
  TensorUniquePtr negative_dim_tensor =
      MakeShapeOnlyTensor({2, -1}, kTfLiteInt32);
  negative_dim_tensor->data.i32 = &dummy_storage;
  EXPECT_EQ(
      ValidateTensorElementsAndData(&context, *negative_dim_tensor, "BadShape"),
      kTfLiteError);
  EXPECT_THAT(
      context.error,
      HasSubstr("BadShape tensor shape is invalid or size overflowed."));

  // Element counts exceeding INT_MAX pass with kSizeT (on 64-bit platforms)
  // and fail with kInt / CheckedTensorElementsAndData.
  if constexpr (sizeof(size_t) >= 8) {
    TensorUniquePtr large_tensor =
        MakeShapeOnlyTensor({65536, 65536}, kTfLiteInt32);
    large_tensor->data.i32 = &dummy_storage;
    EXPECT_EQ(ValidateTensorElementsAndData(&context, *large_tensor, "Large",
                                            ElementCountLimit::kSizeT),
              kTfLiteOk);
    EXPECT_EQ(ValidateTensorElementsAndData(&context, *large_tensor, "Large",
                                            ElementCountLimit::kInt),
              kTfLiteError);
    EXPECT_EQ(CheckedTensorElementsAndData(&context, *large_tensor, "Large",
                                           int_count),
              kTfLiteError);
  }
}

}  // namespace
}  // namespace kernel_utils
}  // namespace tflite
