/* Copyright 2015 Google Inc. All Rights Reserved.

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
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class ResizeBilinearOpTest : public OpsTestBase {
 protected:
  ResizeBilinearOpTest() {
    TF_EXPECT_OK(NodeDefBuilder("resize_bilinear_op", "ResizeBilinear")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Attr("align_corners", false)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

class ResizeBilinearOpAlignCornersTest : public OpsTestBase {
 protected:
  ResizeBilinearOpAlignCornersTest() {
    TF_EXPECT_OK(NodeDefBuilder("resize_bilinear_op", "ResizeBilinear")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Attr("align_corners", true)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(ResizeBilinearOpTest, TestBilinear2x2To1x1) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  // When scaling down, we have to arbitrarily pick a pixel from the
  // original input. In this case, we choose the top/left most pixel.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillValues<float>(&expected, {1.0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpAlignCornersTest, TestBilinearAlignCorners2x2To1x1) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  // When scaling down, we have to arbitrarily pick a pixel from the
  // original input. In this case, we choose the top/left most pixel.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillValues<float>(&expected, {1.0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpTest, TestBilinear2x2To3x3) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1,     5.0/3,   2,
     7.0/3, 3,       10.0/3,
     3,     11.0/3,  4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpAlignCornersTest, TestBilinearAlignCorners2x2To3x3) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // The corners exactly align with the original corners, and we bilinear
  // interpolate the values in between.

  // clang-format off
  test::FillValues<float>(&expected,
    {1,  1.5,  2,
     2,  2.5,  3,
     3,  3.5,  4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpTest, TestBilinear3x3To2x2) {
  // Input:
  //  1, 2, 3
  //  4, 5, 6
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<int32>(TensorShape({2}), {2, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1,   2.5,
     5.5,   7});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpAlignCornersTest, TestBilinearAlignCorners3x3To2x2) {
  // Input:
  //  1, 2, 3
  //  4, 5, 6
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<int32>(TensorShape({2}), {2, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1,  3,
     7,  9});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpTest, TestBilinear3x3To4x4) {
  // Input:
  //  1, 2, 3,
  //  4, 5, 6,
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<int32>(TensorShape({2}), {4, 4});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 4, 4, 1}));
  // clang-format off
  test::FillValues<float>(&expected,
    {1, 1.75, 2.5, 3,
     3.25, 4, 4.75, 5.25,
     5.5, 6.25, 7, 7.5,
     7,  7.75, 8.5, 9});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpTest, TestBilinear4x4To3x3) {
  // Input:
  //  1,  2,  3,  4
  //  5,  6,  7,  8
  //  9, 10, 11, 12
  // 13, 14, 15, 16
  AddInputFromArray<float>(
      TensorShape({1, 4, 4, 1}),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1,       7.0/3, 11.0/3,
     19.0/3, 23.0/3, 27.0/3,
     35.0/3, 39.0/3, 43.0/3});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpAlignCornersTest, TestBilinearAlignCorners4x4To3x3) {
  // Input:
  //  1,  2,  3,  4
  //  5,  6,  7,  8
  //  9, 10, 11, 12
  // 13, 14, 15, 16
  AddInputFromArray<float>(
      TensorShape({1, 4, 4, 1}),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    { 1,  2.5,  4,
      7,  8.5, 10,
     13, 14.5, 16});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpTest, TestBilinear2x2To3x3Batch2) {
  // Input:
  //  1, 2
  //  3, 4
  //
  // repeated twice
  AddInputFromArray<float>(TensorShape({2, 2, 2, 1}), {1, 2, 3, 4, 1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3, 3, 1}));
  // clang-format off
  test::FillValues<float>(&expected,
    {1, 5.0/3, 2, 7.0/3, 3, 10.0/3, 3, 11.0/3, 4,
     1, 5.0/3, 2, 7.0/3, 3, 10.0/3, 3, 11.0/3, 4
    });
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpTest, TestBilinear2x2x2To3x3x2) {
  AddInputFromArray<float>(TensorShape({1, 2, 2, 2}),
                           {1, -1, 2, -2, 3, -3, 4, -4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 2}));
  // clang-format off
  test::FillValues<float>(&expected,
    {
      1,      -1,
      5.0/3,  -5.0/3,
      2,      -2,
      7.0/3,  -7.0/3,
      3,      -3,
      10.0/3, -10.0/3,
      3,      -3,
      11.0/3, -11.0/3,
      4,      -4
    });
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpTest, TestBilinear2x2To4x4) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {4, 4});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 4, 4, 1}));
  // clang-format off
  test::FillValues<float>(&expected,
    {1,  1.5, 2, 2,
     2,  2.5, 3, 3,
     3,  3.5, 4, 4,
     3,  3.5, 4, 4});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeBilinearOpTest, TestInvalidInputShape) {
  AddInputFromArray<float>(TensorShape({2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {4, 4});
  ASSERT_FALSE(RunOpKernel().ok());
}

TEST_F(ResizeBilinearOpTest, TestInvalidSizeDim) {
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2, 1}), {4, 4});
  ASSERT_FALSE(RunOpKernel().ok());
}
TEST_F(ResizeBilinearOpTest, TestInvalidSizeElements) {
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({3}), {4, 4, 1});
  ASSERT_FALSE(RunOpKernel().ok());
}

}  // namespace tensorflow
