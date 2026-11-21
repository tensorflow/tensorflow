/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class CropAndResizeOpTest : public OpsTestBase {
 protected:
  template <typename T>
  void MakeOp(float extrapolation_value, const std::string& method) {
    TF_EXPECT_OK(NodeDefBuilder("crop_and_resize_op", "CropAndResize")
                     .Input(FakeInput(DataTypeToEnum<T>::value))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Attr("extrapolation_value", extrapolation_value)
                     .Attr("method", method)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

namespace {
// All of these box coordinates are finite, so they pass the kernel's
// boxes-finiteness validation, but (kLargeNegativeBox - kLargePositiveBox)
// multiplied by (image_width - 1) overflows float32 to -inf. The sampling
// coordinate then becomes NaN, because 0 * -inf is NaN. With the box signs
// mirrored (or y1 at -FLT_MAX and y2 at 0), the scale overflows to +inf and
// the coordinate is NaN through -inf + 0 * inf as well.
constexpr float kLargeNegativeBox = -1.81283853e38f;  // 0xffff0862
constexpr float kLargePositiveBox = 4.2949673e8f;     // 0x4dcccccd
constexpr float kSmallDenormalBox = 0x1.65p-141f;     // 0x00000165
constexpr float kLowestFloatBox = -3.40282347e+38f;   // 0xff7fffff (-FLT_MAX)
}  // namespace

#define REGISTER_TEST(T)                                               \
  TEST_F(CropAndResizeOpTest, TestCropAndResize##T) {                  \
    MakeOp<T>(0, "bilinear");                                          \
    AddInputFromArray<T>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});     \
    AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});       \
    AddInputFromArray<int32>(TensorShape({1}), {0});                   \
    AddInputFromArray<int32>(TensorShape({2}), {1, 1});                \
    TF_ASSERT_OK(RunOpKernel());                                       \
                                                                       \
    Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1})); \
    test::FillValues<float>(&expected, {2.5});                         \
    test::ExpectTensorEqual<float>(expected, *GetOutput(0));           \
  }                                                                    \
                                                                       \
  TEST_F(CropAndResizeOpTest, TestCropAndResize##T##nearest) {         \
    MakeOp<T>(0, "nearest");                                           \
    AddInputFromArray<T>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});     \
    AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});       \
    AddInputFromArray<int32>(TensorShape({1}), {0});                   \
    AddInputFromArray<int32>(TensorShape({2}), {1, 1});                \
    TF_ASSERT_OK(RunOpKernel());                                       \
                                                                       \
    Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1})); \
    test::FillValues<float>(&expected, {4.0});                         \
    test::ExpectTensorEqual<float>(expected, *GetOutput(0));           \
  }

REGISTER_TEST(float)
REGISTER_TEST(double)
REGISTER_TEST(uint8_t)
REGISTER_TEST(uint16_t)
REGISTER_TEST(int8_t)
REGISTER_TEST(int16_t)
REGISTER_TEST(int32_t)
REGISTER_TEST(int64_t)

#undef REGISTER_TEST

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To1x1Uint8) {
  MakeOp<uint8_t>(0, "bilinear");
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<uint8_t>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<int32_t>(TensorShape({1}), {0});
  AddInputFromArray<int32_t>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillValues<float>(&expected, {2.5});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To1x1Uint8NearestNeibor) {
  MakeOp<uint8_t>(0, "nearest");
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<uint8_t>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<int32_t>(TensorShape({1}), {0});
  AddInputFromArray<int32_t>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillValues<float>(&expected, {4.0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To1x1Flipped) {
  MakeOp<float>(0, "bilinear");
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {1, 1, 0, 0});
  AddInputFromArray<int32_t>(TensorShape({1}), {0});
  AddInputFromArray<int32_t>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillValues<float>(&expected, {2.5});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To1x1FlippedNearestNeighbor) {
  MakeOp<float>(0, "nearest");
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {1, 1, 0, 0});
  AddInputFromArray<int32_t>(TensorShape({1}), {0});
  AddInputFromArray<int32_t>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillValues<float>(&expected, {4.0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To3x3) {
  MakeOp<float>(0, "bilinear");
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<int32_t>(TensorShape({1}), {0});
  AddInputFromArray<int32_t>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));
  // clang-format off
  test::FillValues<float>(&expected,
    {1,  1.5,  2,
     2,  2.5,  3,
     3,  3.5,  4});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To3x3NearestNeighbor) {
  MakeOp<float>(0, "nearest");
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<int32_t>(TensorShape({1}), {0});
  AddInputFromArray<int32_t>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));
  // clang-format off
  test::FillValues<float>(&expected,
    {1,  2,  2,
     3,  4,  4,
     3,  4,  4});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To3x3Flipped) {
  MakeOp<float>(0, "bilinear");
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {1, 1, 0, 0});
  AddInputFromArray<int32_t>(TensorShape({1}), {0});
  AddInputFromArray<int32_t>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));
  // clang-format off
  test::FillValues<float>(&expected,
    {4,  3.5,  3,
     3,  2.5,  2,
     2,  1.5,  1});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To3x3FlippedNearestNeighbor) {
  MakeOp<float>(0, "nearest");
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {1, 1, 0, 0});
  AddInputFromArray<int32_t>(TensorShape({1}), {0});
  AddInputFromArray<int32_t>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));
  // clang-format off
  test::FillValues<float>(&expected,
    {4,  4,  3,
     4,  4,  3,
     2,  2,  1});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize3x3To2x2) {
  MakeOp<float>(0, "bilinear");
  // Input:
  //  1, 2, 3
  //  4, 5, 6
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<float>(TensorShape({2, 4}), {0, 0, 1, 1, 0, 0, 0.5, 0.5});
  AddInputFromArray<int32_t>(TensorShape({2}), {0, 0});
  AddInputFromArray<int32_t>(TensorShape({2}), {2, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1,  3,
     7,  9,
     1,  2,
     4,  5});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize3x3To2x2NearestNeighbor) {
  MakeOp<float>(0, "nearest");
  // Input:
  //  1, 2, 3
  //  4, 5, 6
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<float>(TensorShape({2, 4}), {0, 0, 1, 1, 0, 0, 0.5, 0.5});
  AddInputFromArray<int32_t>(TensorShape({2}), {0, 0});
  AddInputFromArray<int32_t>(TensorShape({2}), {2, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1,  3,
     7,  9,
     1,  2,
     4,  5});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize3x3To2x2Flipped) {
  MakeOp<float>(0, "bilinear");
  // Input:
  //  1, 2, 3
  //  4, 5, 6
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<float>(TensorShape({2, 4}), {1, 1, 0, 0, 0.5, 0.5, 0, 0});
  AddInputFromArray<int32_t>(TensorShape({2}), {0, 0});
  AddInputFromArray<int32_t>(TensorShape({2}), {2, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {9,  7,
     3,  1,
     5,  4,
     2,  1});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize3x3To2x2FlippedNearestNeighbor) {
  MakeOp<float>(0, "nearest");
  // Input:
  //  1, 2, 3
  //  4, 5, 6
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<float>(TensorShape({2, 4}), {1, 1, 0, 0, 0.5, 0.5, 0, 0});
  AddInputFromArray<int32_t>(TensorShape({2}), {0, 0});
  AddInputFromArray<int32_t>(TensorShape({2}), {2, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {9,  7,
     3,  1,
     5,  4,
     2,  1});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To3x3Extrapolated) {
  const float v = -1;
  MakeOp<float>(v, "bilinear");
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {-1, -1, 1, 1});
  AddInputFromArray<int32_t>(TensorShape({1}), {0});
  AddInputFromArray<int32_t>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));
  // clang-format off
  test::FillValues<float>(&expected,
    {v,  v,  v,
     v,  1,  2,
     v,  3,  4});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResizeFiniteBoxesOverflowingWidthScale) {
  const float v = -1;
  MakeOp<float>(v, "bilinear");
  // Input:
  //  1,  2,  3,  4
  //  5,  6,  7,  8
  //  9, 10, 11, 12
  // 13, 14, 15, 16
  AddInputFromArray<float>(
      TensorShape({1, 4, 4, 1}),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  // width_scale = (kLargeNegativeBox - kLargePositiveBox) * 3 / 2 is -inf, so
  // in_x is NaN for x == 0 (0 * -inf) and -inf for x > 0. Every sampled column
  // must take the extrapolation value instead of indexing with floorf(NaN).
  AddInputFromArray<float>(
      TensorShape({1, 4}),
      {0.1f, kLargePositiveBox, kSmallDenormalBox, kLargeNegativeBox});
  AddInputFromArray<int32_t>(TensorShape({1}), {0});
  AddInputFromArray<int32_t>(TensorShape({2}), {2, 3});
  TF_ASSERT_OK(RunOpKernel());

  const Tensor& output = *GetOutput(0);
  ASSERT_EQ(output.NumElements(), 6);
  for (int i = 0; i < output.NumElements(); ++i) {
    EXPECT_EQ(output.flat<float>()(i), v);
  }
}

TEST_F(CropAndResizeOpTest,
       TestCropAndResizeFiniteBoxesOverflowingHeightScale) {
  const float v = -1;
  MakeOp<float>(v, "bilinear");
  AddInputFromArray<float>(
      TensorShape({1, 4, 4, 1}),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  // Same construction on the y axis: height_scale is -inf, so in_y is NaN for
  // y == 0 and -inf for y > 0, and every row must be extrapolated.
  AddInputFromArray<float>(
      TensorShape({1, 4}),
      {kLargePositiveBox, 0.1f, kLargeNegativeBox, kSmallDenormalBox});
  AddInputFromArray<int32_t>(TensorShape({1}), {0});
  AddInputFromArray<int32_t>(TensorShape({2}), {3, 2});
  TF_ASSERT_OK(RunOpKernel());

  const Tensor& output = *GetOutput(0);
  ASSERT_EQ(output.NumElements(), 6);
  for (int i = 0; i < output.NumElements(); ++i) {
    EXPECT_EQ(output.flat<float>()(i), v);
  }
}

TEST_F(CropAndResizeOpTest,
       TestCropAndResizeFiniteBoxesOverflowingHeightScalePosInf) {
  const float v = -1;
  MakeOp<float>(v, "bilinear");
  AddInputFromArray<float>(
      TensorShape({1, 4, 4, 1}),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  // Here y1 is -FLT_MAX and y2 is 0, so height_scale overflows to +inf while
  // width_scale stays 0: in_y is NaN for y == 0 (-inf + 0 * inf) and for
  // y > 0 (-inf + inf), and in_x stays 0. Every row must be extrapolated
  // instead of indexing with floorf(NaN).
  AddInputFromArray<float>(TensorShape({1, 4}),
                           {kLowestFloatBox, 0.0f, 0.0f, 0.0f});
  AddInputFromArray<int32_t>(TensorShape({1}), {0});
  AddInputFromArray<int32_t>(TensorShape({2}), {16, 1});
  TF_ASSERT_OK(RunOpKernel());

  const Tensor& output = *GetOutput(0);
  ASSERT_EQ(output.NumElements(), 16);
  for (int i = 0; i < output.NumElements(); ++i) {
    EXPECT_EQ(output.flat<float>()(i), v);
  }
}

TEST_F(CropAndResizeOpTest,
       TestCropAndResizeFiniteBoxesOverflowingScaleNearest) {
  const float v = -1;
  MakeOp<float>(v, "nearest");
  AddInputFromArray<float>(
      TensorShape({1, 4, 4, 1}),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  // The nearest-neighbor method samples the same NaN coordinate through
  // roundf() and must also take the extrapolation value.
  AddInputFromArray<float>(
      TensorShape({1, 4}),
      {0.1f, kLargePositiveBox, kSmallDenormalBox, kLargeNegativeBox});
  AddInputFromArray<int32_t>(TensorShape({1}), {0});
  AddInputFromArray<int32_t>(TensorShape({2}), {2, 3});
  TF_ASSERT_OK(RunOpKernel());

  const Tensor& output = *GetOutput(0);
  ASSERT_EQ(output.NumElements(), 6);
  for (int i = 0; i < output.NumElements(); ++i) {
    EXPECT_EQ(output.flat<float>()(i), v);
  }
}

TEST_F(CropAndResizeOpTest,
       TestCropAndResizeFiniteBoxesOverflowingScalePosInf) {
  const float v = -1;
  MakeOp<float>(v, "bilinear");
  AddInputFromArray<float>(
      TensorShape({1, 4, 4, 1}),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  // Mirrored coordinates: width_scale is +inf, and in_x is NaN both for
  // x == 0 (0 * inf) and for x > 0 (-inf + inf).
  AddInputFromArray<float>(
      TensorShape({1, 4}),
      {0.1f, kLargeNegativeBox, kSmallDenormalBox, kLargePositiveBox});
  AddInputFromArray<int32_t>(TensorShape({1}), {0});
  AddInputFromArray<int32_t>(TensorShape({2}), {2, 3});
  TF_ASSERT_OK(RunOpKernel());

  const Tensor& output = *GetOutput(0);
  ASSERT_EQ(output.NumElements(), 6);
  for (int i = 0; i < output.NumElements(); ++i) {
    EXPECT_EQ(output.flat<float>()(i), v);
  }
}

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To3x3NoCrop) {
  MakeOp<float>(0, "bilinear");
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({0, 4}), {});
  AddInputFromArray<int32_t>(TensorShape({0}), {});
  AddInputFromArray<int32_t>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({0, 3, 3, 1}));
  // clang-format off
  test::FillValues<float>(&expected, {});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestInvalidInputShape) {
  MakeOp<float>(0, "bilinear");
  AddInputFromArray<float>(TensorShape({2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<int32_t>(TensorShape({1}), {0});
  AddInputFromArray<int32_t>(TensorShape({2}), {4, 4});
  absl::Status s = RunOpKernel();
  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.ToString(), "input image must be 4-D")) << s;
}

TEST_F(CropAndResizeOpTest, TestInvalidBoxIndexShape) {
  MakeOp<float>(0, "bilinear");
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<int32_t>(TensorShape({2}), {0, 0});
  AddInputFromArray<int32_t>(TensorShape({2}), {4, 4});
  absl::Status s = RunOpKernel();
  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(
      absl::StrContains(s.ToString(), "box_index has incompatible shape"))
      << s;
}

TEST_F(CropAndResizeOpTest, TestInvalidBoxIndex) {
  MakeOp<float>(0, "bilinear");
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<int32_t>(TensorShape({1}), {1});
  AddInputFromArray<int32_t>(TensorShape({2}), {3, 3});
  absl::Status s = RunOpKernel();
  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.ToString(),
                                "box_index has values outside [0, batch_size)"))
      << s;
}

TEST_F(CropAndResizeOpTest, TestWithSharding) {
  MakeOp<float>(0, "bilinear");
  // Generate a relatively large input (999x999) so that sharding happens.
  const int kLength = 999;  // Length of the input. Must use an odd number.
  const int kHalf = (kLength + 1) / 2;  // Half size for the cropped result.

  // Input:
  //  0, 1, 2, ..., 998
  //  0, 1, 2, ..., 998
  //  ... (altogether 999 lines)
  //  0, 1, 2, ..., 998
  AddInput<float>(TensorShape({1, kLength, kLength, 1}),
                  [=](int i) -> float { return i % kLength; });
  AddInputFromArray<float>(TensorShape({2, 4}),
                           {0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1});
  AddInputFromArray<int32_t>(TensorShape({2}), {0, 0});
  AddInputFromArray<int32_t>(TensorShape({2}), {kHalf, kHalf});

  TF_ASSERT_OK(RunOpKernel());

  // Generate result tensor.
  // Result 1:
  //  0, 1, 2, ..., 499
  //  ... (altogether 500 lines)
  //  0, 1, 2, ..., 499
  Tensor result1(allocator(), DT_FLOAT, TensorShape({1, kHalf, kHalf, 1}));
  test::FillFn<float>(&result1, [=](int i) -> float { return i % kHalf; });

  // Result 2:
  //  499, 500, 501, ..., 998
  //  ... (altogether 500 lines)
  //  499, 500, 501, ..., 998
  Tensor result2(allocator(), DT_FLOAT, TensorShape({1, kHalf, kHalf, 1}));
  test::FillFn<float>(&result2,
                      [=](int i) -> float { return i % kHalf + kHalf - 1; });

  // Expected result is the concat of the two tensors.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, kHalf, kHalf, 1}));
  TF_ASSERT_OK(tensor::Concat({result1, result2}, &expected));

  // Compare result.
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

}  // namespace tensorflow
