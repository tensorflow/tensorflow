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

class CropAndResizeOpTest : public OpsTestBase {
 protected:
  void MakeOp(float extrapolation_value) {
    TF_EXPECT_OK(NodeDefBuilder("crop_and_resize_op", "CropAndResize")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Attr("extrapolation_value", extrapolation_value)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To1x1) {
  MakeOp(0);
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<int32>(TensorShape({1}), {0});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillValues<float>(&expected, {2.5});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To1x1Flipped) {
  MakeOp(0);
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {1, 1, 0, 0});
  AddInputFromArray<int32>(TensorShape({1}), {0});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillValues<float>(&expected, {2.5});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To3x3) {
  MakeOp(0);
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<int32>(TensorShape({1}), {0});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
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

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To3x3Flipped) {
  MakeOp(0);
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {1, 1, 0, 0});
  AddInputFromArray<int32>(TensorShape({1}), {0});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
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

TEST_F(CropAndResizeOpTest, TestCropAndResize3x3To2x2) {
  MakeOp(0);
  // Input:
  //  1, 2, 3
  //  4, 5, 6
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<float>(TensorShape({2, 4}), {0, 0, 1, 1, 0, 0, 0.5, 0.5});
  AddInputFromArray<int32>(TensorShape({2}), {0, 0});
  AddInputFromArray<int32>(TensorShape({2}), {2, 2});
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
  MakeOp(0);
  // Input:
  //  1, 2, 3
  //  4, 5, 6
  //  7, 8, 9
  AddInputFromArray<float>(TensorShape({1, 3, 3, 1}),
                           {1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<float>(TensorShape({2, 4}), {1, 1, 0, 0, 0.5, 0.5, 0, 0});
  AddInputFromArray<int32>(TensorShape({2}), {0, 0});
  AddInputFromArray<int32>(TensorShape({2}), {2, 2});
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
  MakeOp(v);
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {-1, -1, 1, 1});
  AddInputFromArray<int32>(TensorShape({1}), {0});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
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

TEST_F(CropAndResizeOpTest, TestCropAndResize2x2To3x3NoCrop) {
  MakeOp(0);
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({0, 4}), {});
  AddInputFromArray<int32>(TensorShape({0}), {});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({0, 3, 3, 1}));
  // clang-format off
  test::FillValues<float>(&expected, {});
  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(CropAndResizeOpTest, TestInvalidInputShape) {
  MakeOp(0);
  AddInputFromArray<float>(TensorShape({2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<int32>(TensorShape({1}), {0});
  AddInputFromArray<int32>(TensorShape({2}), {4, 4});
  Status s = RunOpKernel();
  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(StringPiece(s.ToString()).contains("input image must be 4-D"))
      << s;
}

TEST_F(CropAndResizeOpTest, TestInvalidBoxIndexShape) {
  MakeOp(0);
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<int32>(TensorShape({2}), {0, 0});
  AddInputFromArray<int32>(TensorShape({2}), {4, 4});
  Status s = RunOpKernel();
  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(
      StringPiece(s.ToString()).contains("box_ind has incompatible shape"))
      << s;
}

TEST_F(CropAndResizeOpTest, TestInvalidBoxIndex) {
  MakeOp(0);
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<int32>(TensorShape({1}), {1});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  Status s = RunOpKernel();
  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(StringPiece(s.ToString())
                  .contains("box_ind has values outside [0, batch)"))
      << s;
}

}  // namespace tensorflow
