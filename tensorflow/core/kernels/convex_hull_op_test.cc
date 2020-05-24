/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include <functional>
#include <memory>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class ConvexHullOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType data_type, bool clockwise) {
    TF_ASSERT_OK(NodeDefBuilder("convex_hull_op", "ConvexHull")
                     .Input(FakeInput(data_type))
                     .Attr("clockwise", clockwise)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

// If only one point, return the input point
TEST_F(ConvexHullOpTest, OnePoint) {
  MakeOp(DT_FLOAT, false);

  // Feed and run
  AddInputFromArray<float>(TensorShape({2, 1, 2}), {0, 1, 2, 3});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 1, 2}));
  test::FillValues<float>(&expected, {0, 1, 2, 3});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

// If two points, return the input points(with differnt order)
TEST_F(ConvexHullOpTest, TwoPoints) {
  MakeOp(DT_INT32, false);

  // Feed and run
  AddInputFromArray<int>(TensorShape({2, 2, 2}), {0, 1, 2, 3, 4, 5, 6, 7});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 2}));
  test::FillValues<float>(&expected, {0, 1, 2, 3, 4, 5, 6, 7});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

// If three points not on the same line, return the input points(with differnt
// order)
TEST_F(ConvexHullOpTest, ThreePoints) {
  MakeOp(DT_FLOAT, true);

  // Feed and run
  AddInputFromArray<float>(TensorShape({2, 3, 2}),
                           {5, 3, 2, 7, 4, 9, 6, -1, 0, 10, -8, 11});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3, 2}));
  test::FillValues<float>(&expected, {2, 7, 4, 9, 5, 3, -8, 11, 0, 10, 6, -1});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

// If three points on the same line, return the start and end points (with
// repitition)
TEST_F(ConvexHullOpTest, ThreePoints_SameLine) {
  MakeOp(DT_INT32, false);

  // Feed and run
  AddInputFromArray<int>(TensorShape({2, 3, 2}),
                         {5, 23, 9, 27, 1, 19, -7, 2, 0, 9, 3, 12});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3, 2}));
  test::FillValues<float>(&expected,
                          {9, 27, 1, 19, 1, 19, 3, 12, -7, 2, -7, 2});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

// If more than three points,
// return the convex hull with repititive points padded in the end
TEST_F(ConvexHullOpTest, MorePoints) {
  MakeOp(DT_FLOAT, true);

  // Feed and run
  AddInputFromArray<float>(TensorShape({2, 5, 2}),
                           {-1, -13, -6, 33, 74, 58, -99, 48,  11, -5,
                            32, -54, 4,  12, 24, -7, 23,  -21, 16, 47});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 5, 2}));
  test::FillValues<float>(&expected,
                          {-99, 48, 74, 58, 11, -5,  -1, -13, -1, -13,
                           4,   12, 16, 47, 32, -54, 32, -54, 32, -54});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ConvexHullOpTest, Error_InputShapeMustBe3D) {
  MakeOp(DT_FLOAT, false);

  // Feed and run
  AddInputFromArray<float>(TensorShape({}), {5});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      absl::StrContains(s.ToString(), "input shape must be 3-dimensional"))
      << s;
}

TEST_F(ConvexHullOpTest, Error_InputDimMustBe2) {
  MakeOp(DT_FLOAT, false);

  // Feed and run
  AddInputFromArray<float>(TensorShape({1, 1, 3}), {5, 6, 7});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(s.ToString(), "point dimension must be 2"))
      << s;
}

}  // namespace
}  // namespace tensorflow
