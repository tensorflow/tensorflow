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

// TODO(shlens, sherrym): Consider adding additional tests in image_ops.py in
// order to compare the reference implementation for image resizing in Python
// Image Library.
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
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
enum class TestDevice { kCPU, kGPU };

class ResizeNearestNeighborOpTestBase
    : public OpsTestBase,
      public ::testing::WithParamInterface<TestDevice> {
 protected:
  explicit ResizeNearestNeighborOpTestBase(bool half_pixel_centers)
      : align_corners_(false), half_pixel_centers_(half_pixel_centers) {}
  void SetUp() override {
    if (GetParam() == TestDevice::kGPU) {
      std::unique_ptr<Device> device_gpu(
          DeviceFactory::NewDevice(/*type=*/"GPU", /*options=*/{},
                                   /*name_prefix=*/"/job:a/replica:0/task:0"));
      SetDevice(DEVICE_GPU, std::move(device_gpu));
    }

    TF_EXPECT_OK(NodeDefBuilder("resize_nn_op", "ResizeNearestNeighbor")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Attr("align_corners", align_corners_)
                     .Attr("half_pixel_centers", half_pixel_centers_)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
  bool align_corners_;
  bool half_pixel_centers_;
};

class ResizeNearestNeighborOpTest : public ResizeNearestNeighborOpTestBase {
 protected:
  ResizeNearestNeighborOpTest() : ResizeNearestNeighborOpTestBase(false) {}
};

class ResizeNearestNeighborHalfPixelCentersOpTest
    : public ResizeNearestNeighborOpTestBase {
 protected:
  ResizeNearestNeighborHalfPixelCentersOpTest()
      : ResizeNearestNeighborOpTestBase(true) {}
};

// TODO(jflynn): Add some actual tests for the half pixel centers case.

class ResizeNearestNeighborOpAlignCornersTest
    : public OpsTestBase,
      public ::testing::WithParamInterface<TestDevice> {
 protected:
  ResizeNearestNeighborOpAlignCornersTest() : align_corners_(true) {}
  void SetUp() override {
    if (GetParam() == TestDevice::kGPU) {
      std::unique_ptr<Device> device_gpu(
          DeviceFactory::NewDevice(/*type=*/"GPU", /*options=*/{},
                                   /*name_prefix=*/"/job:a/replica:0/task:0"));
      SetDevice(DEVICE_GPU, std::move(device_gpu));
    }

    TF_EXPECT_OK(NodeDefBuilder("resize_nn_op", "ResizeNearestNeighbor")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Attr("align_corners", align_corners_)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
  bool align_corners_;
};

TEST_P(ResizeNearestNeighborOpTest, TestNearest2x2To1x1) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));

  // clang-format off
  test::FillValues<float>(&expected, {1});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborOpAlignCornersTest,
       TestNearest2x2AlignCornersTo1x1) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));

  // clang-format off
  test::FillValues<float>(&expected, {1});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborOpTest, TestNearest2x2To3x3) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 1, 2,
     1, 1, 2,
     3, 3, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborOpAlignCornersTest,
       TestNearestAlignCorners2x2To3x3) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 2, 2,
     3, 4, 4,
     3, 4, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborOpTest, TestNearest3x3To2x2) {
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
    {1, 2,
     4, 5});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborOpAlignCornersTest,
       TestNearestAlignCorners3x3To2x2) {
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
    {1, 3,
     7, 9});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborOpTest, TestNearest2x2To2x5) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {2, 5});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 5, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 1, 1, 2, 2,
     3, 3, 3, 4, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborOpTest, TestNearestNeighbor4x4To3x3) {
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
    {1,  2,  3,
     5,  6,  7,
     9, 10, 11});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborOpAlignCornersTest,
       TestNearestNeighborAlignCorners4x4To3x3) {
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
    { 1,  3,  4,
      9, 11, 12,
     13, 15, 16});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborOpTest, TestNearest2x2To5x2) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {5, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 5, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 2,
     1, 2,
     1, 2,
     3, 4,
     3, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborOpTest, TestNearest2x2To4x4) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {4, 4});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 4, 4, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 1, 2, 2,
     1, 1, 2, 2,
     3, 3, 4, 4,
     3, 3, 4, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborOpTest, TestNearest2x2x2x2To2x3x3x2) {
  // Input:
  //  [ [ 1, 1 ], [ 2, 2],
  //    [ 3, 3 ], [ 4, 4] ],
  //  [ [ 5, 5 ], [ 6, 6],
  //    [ 7, 7 ], [ 8, 8] ]
  AddInputFromArray<float>(TensorShape({2, 2, 2, 2}),
                           {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3, 3, 2}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 1, 1,
     1, 2, 2,
     1, 1, 1,
     1, 2, 2,
     3, 3, 3,
     3, 4, 4,
     5, 5, 5,
     5, 6, 6,
     5, 5, 5,
     5, 6, 6,
     7, 7, 7,
     7, 8, 8});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborHalfPixelCentersOpTest, TestNearest5x2To2x2) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 5, 1}),
                           {1, 2, 3, 4, 5, 1, 2, 3, 4, 5});
  AddInputFromArray<int32>(TensorShape({2}), {2, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected, {2, 4, 2, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborHalfPixelCentersOpTest, TestNearest2x2To1x1) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));

  // clang-format off
  test::FillValues<float>(&expected, {4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborHalfPixelCentersOpTest, TestNearest2x2To3x3) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 2, 2,
     3, 4, 4,
     3, 4, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborHalfPixelCentersOpTest, TestNearest3x3To2x2) {
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
    {1, 3,
     7, 9});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborHalfPixelCentersOpTest, TestNearest2x2To2x5) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {2, 5});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 5, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 1, 2, 2, 2,
     3, 3, 4, 4, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborHalfPixelCentersOpTest,
       TestNearestNeighbor4x4To3x3) {
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
    {1,  3,  4,
     9, 11, 12,
     13, 15, 16});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborHalfPixelCentersOpTest, TestNearest2x2To5x2) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {5, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 5, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 2,
     1, 2,
     3, 4,
     3, 4,
     3, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborHalfPixelCentersOpTest, TestNearest2x2To4x4) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {4, 4});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 4, 4, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 1, 2, 2,
     1, 1, 2, 2,
     3, 3, 4, 4,
     3, 3, 4, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_P(ResizeNearestNeighborHalfPixelCentersOpTest,
       TestNearest2x2x2x2To2x3x3x2) {
  // Input:
  //  [ [ 1, 1 ], [ 2, 2],
  //    [ 3, 3 ], [ 4, 4] ],
  //  [ [ 5, 5 ], [ 6, 6],
  //    [ 7, 7 ], [ 8, 8] ]
  AddInputFromArray<float>(TensorShape({2, 2, 2, 2}),
                           {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3, 3, 2}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 1, 2, 2, 2, 2,
     3, 3, 4, 4, 4, 4,
     3, 3, 4, 4, 4, 4,
     5, 5, 6, 6, 6, 6,
     7, 7, 8, 8, 8, 8,
     7, 7, 8, 8, 8, 8});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

INSTANTIATE_TEST_SUITE_P(ResizeNearestNeighborOpTestCpu,
                         ResizeNearestNeighborOpTest,
                         ::testing::Values(TestDevice::kCPU));
INSTANTIATE_TEST_SUITE_P(ResizeNearestNeighborHalfPixelCentersOpTestCpu,
                         ResizeNearestNeighborHalfPixelCentersOpTest,
                         ::testing::Values(TestDevice::kCPU));
INSTANTIATE_TEST_SUITE_P(ResizeNearestNeighborOpAlignCornersTestCpu,
                         ResizeNearestNeighborOpAlignCornersTest,
                         ::testing::Values(TestDevice::kCPU));
#if GOOGLE_CUDA
// Instantiate tests for kGPU.
INSTANTIATE_TEST_SUITE_P(ResizeNearestNeighborOpTestGpu,
                         ResizeNearestNeighborOpTest,
                         ::testing::Values(TestDevice::kGPU));
INSTANTIATE_TEST_SUITE_P(ResizeNearestNeighborHalfPixelCentersOpTestGpu,
                         ResizeNearestNeighborHalfPixelCentersOpTest,
                         ::testing::Values(TestDevice::kGPU));
INSTANTIATE_TEST_SUITE_P(ResizeNearestNeighborOpAlignCornersTestGpu,
                         ResizeNearestNeighborOpAlignCornersTest,
                         ::testing::Values(TestDevice::kGPU));
#endif  // GOOGLE_CUDA
}  // namespace tensorflow
