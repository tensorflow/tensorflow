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
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class NonMaxSuppressionOpTest : public OpsTestBase {
 protected:
  void MakeOp(float iou_threshold) {
    TF_EXPECT_OK(NodeDefBuilder("non_max_suppression_op", "NonMaxSuppression")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Attr("iou_threshold", iou_threshold)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(NonMaxSuppressionOpTest, TestSelectFromThreeClusters) {
  MakeOp(.5);
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionOpTest, TestSelectFromThreeClustersFlippedCoordinates) {
  MakeOp(.5);
  AddInputFromArray<float>(TensorShape({6, 4}),
                           {1, 1,  0, 0,  0, 0.1f,  1, 1.1f,  0, .9f, 1, -0.1f,
                            0, 10, 1, 11, 1, 10.1f, 0, 11.1f, 1, 101, 0, 100});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionOpTest, TestSelectAtMostTwoBoxesFromThreeClusters) {
  MakeOp(.5);
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int>(&expected, {3, 0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionOpTest, TestSelectWithNegativeScores) {
  MakeOp(.5);
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(
      TensorShape({6}), {.9f - 10.0f, .75f - 10.0f, .6f - 10.0f, .95f - 10.0f,
                         .5f - 10.0f, .3f - 10.0f});
  AddInputFromArray<int>(TensorShape({}), {6});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionOpTest, TestSelectAtMostThirtyBoxesFromThreeClusters) {
  MakeOp(.5);
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {30});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionOpTest, TestSelectSingleBox) {
  MakeOp(.5);
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<float>(TensorShape({1}), {.9f});
  AddInputFromArray<int>(TensorShape({}), {3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int>(&expected, {0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionOpTest, TestSelectFromTenIdenticalBoxes) {
  MakeOp(.5);

  int num_boxes = 10;
  std::vector<float> corners(num_boxes * 4);
  std::vector<float> scores(num_boxes);
  for (int i = 0; i < num_boxes; ++i) {
    corners[i * 4 + 0] = 0;
    corners[i * 4 + 1] = 0;
    corners[i * 4 + 2] = 1;
    corners[i * 4 + 3] = 1;
    scores[i] = .9;
  }
  AddInputFromArray<float>(TensorShape({num_boxes, 4}), corners);
  AddInputFromArray<float>(TensorShape({num_boxes}), scores);
  AddInputFromArray<int>(TensorShape({}), {3});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int>(&expected, {0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionOpTest, TestInconsistentBoxAndScoreShapes) {
  MakeOp(.5);
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({5}), {.9f, .75f, .6f, .95f, .5f});
  AddInputFromArray<int>(TensorShape({}), {30});
  Status s = RunOpKernel();

  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(
      str_util::StrContains(s.ToString(), "scores has incompatible shape"))
      << s;
}

TEST_F(NonMaxSuppressionOpTest, TestInvalidIOUThreshold) {
  MakeOp(1.2);
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<float>(TensorShape({1}), {.9f});
  AddInputFromArray<int>(TensorShape({}), {3});
  Status s = RunOpKernel();

  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(
      str_util::StrContains(s.ToString(), "iou_threshold must be in [0, 1]"))
      << s;
}

TEST_F(NonMaxSuppressionOpTest, TestEmptyInput) {
  MakeOp(.5);
  AddInputFromArray<float>(TensorShape({0, 4}), {});
  AddInputFromArray<float>(TensorShape({0}), {});
  AddInputFromArray<int>(TensorShape({}), {30});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({0}));
  test::FillValues<int>(&expected, {});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

//
// NonMaxSuppressionV2Op Tests
//

class NonMaxSuppressionV2OpTest : public OpsTestBase {
 protected:
  void MakeOp() {
    TF_EXPECT_OK(NodeDefBuilder("non_max_suppression_op", "NonMaxSuppressionV2")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(NonMaxSuppressionV2OpTest, TestSelectFromThreeClusters) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV2OpTest,
       TestSelectFromThreeClustersFlippedCoordinates) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({6, 4}),
                           {1, 1,  0, 0,  0, 0.1f,  1, 1.1f,  0, .9f, 1, -0.1f,
                            0, 10, 1, 11, 1, 10.1f, 0, 11.1f, 1, 101, 0, 100});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV2OpTest, TestSelectAtMostTwoBoxesFromThreeClusters) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {2});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int>(&expected, {3, 0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV2OpTest,
       TestSelectAtMostThirtyBoxesFromThreeClusters) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV2OpTest, TestSelectSingleBox) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<float>(TensorShape({1}), {.9f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int>(&expected, {0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV2OpTest, TestSelectFromTenIdenticalBoxes) {
  MakeOp();

  int num_boxes = 10;
  std::vector<float> corners(num_boxes * 4);
  std::vector<float> scores(num_boxes);
  for (int i = 0; i < num_boxes; ++i) {
    corners[i * 4 + 0] = 0;
    corners[i * 4 + 1] = 0;
    corners[i * 4 + 2] = 1;
    corners[i * 4 + 3] = 1;
    scores[i] = .9;
  }
  AddInputFromArray<float>(TensorShape({num_boxes, 4}), corners);
  AddInputFromArray<float>(TensorShape({num_boxes}), scores);
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int>(&expected, {0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV2OpTest, TestInconsistentBoxAndScoreShapes) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({5}), {.9f, .75f, .6f, .95f, .5f});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  Status s = RunOpKernel();

  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(
      str_util::StrContains(s.ToString(), "scores has incompatible shape"))
      << s;
}

TEST_F(NonMaxSuppressionV2OpTest, TestInvalidIOUThreshold) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<float>(TensorShape({1}), {.9f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {1.2f});
  Status s = RunOpKernel();

  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(
      str_util::StrContains(s.ToString(), "iou_threshold must be in [0, 1]"))
      << s;
}

TEST_F(NonMaxSuppressionV2OpTest, TestEmptyInput) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({0, 4}), {});
  AddInputFromArray<float>(TensorShape({0}), {});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({0}));
  test::FillValues<int>(&expected, {});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

//
// NonMaxSuppressionV3Op Tests
//

class NonMaxSuppressionV3OpTest : public OpsTestBase {
 protected:
  void MakeOp() {
    TF_EXPECT_OK(NodeDefBuilder("non_max_suppression_op", "NonMaxSuppressionV3")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(NonMaxSuppressionV3OpTest, TestSelectFromThreeClusters) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3OpTest,
       TestSelectFromThreeClustersWithScoreThreshold) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {0.5f});
  AddInputFromArray<float>(TensorShape({}), {0.4f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int>(&expected, {3, 0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3OpTest,
       TestSelectFromThreeClustersWithScoreThresholdZeroScores) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.1, 0, 0, .3, .2, -5.0});
  // If we ask for more boxes than we actually expect to get back;
  // should still only get 2 boxes back.
  AddInputFromArray<int>(TensorShape({}), {6});
  AddInputFromArray<float>(TensorShape({}), {0.5f});
  AddInputFromArray<float>(TensorShape({}), {-3.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int>(&expected, {3, 0});

  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3OpTest,
       TestSelectFromThreeClustersFlippedCoordinates) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({6, 4}),
                           {1, 1,  0, 0,  0, 0.1f,  1, 1.1f,  0, .9f, 1, -0.1f,
                            0, 10, 1, 11, 1, 10.1f, 0, 11.1f, 1, 101, 0, 100});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3OpTest, TestSelectAtMostTwoBoxesFromThreeClusters) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {2});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int>(&expected, {3, 0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3OpTest,
       TestSelectAtMostThirtyBoxesFromThreeClusters) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3OpTest, TestSelectSingleBox) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<float>(TensorShape({1}), {.9f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int>(&expected, {0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3OpTest, TestSelectFromTenIdenticalBoxes) {
  MakeOp();

  int num_boxes = 10;
  std::vector<float> corners(num_boxes * 4);
  std::vector<float> scores(num_boxes);
  for (int i = 0; i < num_boxes; ++i) {
    corners[i * 4 + 0] = 0;
    corners[i * 4 + 1] = 0;
    corners[i * 4 + 2] = 1;
    corners[i * 4 + 3] = 1;
    scores[i] = .9;
  }
  AddInputFromArray<float>(TensorShape({num_boxes, 4}), corners);
  AddInputFromArray<float>(TensorShape({num_boxes}), scores);
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int>(&expected, {0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3OpTest, TestInconsistentBoxAndScoreShapes) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({5}), {.9f, .75f, .6f, .95f, .5f});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  Status s = RunOpKernel();

  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(
      str_util::StrContains(s.ToString(), "scores has incompatible shape"))
      << s;
}

TEST_F(NonMaxSuppressionV3OpTest, TestInvalidIOUThreshold) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<float>(TensorShape({1}), {.9f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {1.2f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  Status s = RunOpKernel();

  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(
      str_util::StrContains(s.ToString(), "iou_threshold must be in [0, 1]"))
      << s;
}

TEST_F(NonMaxSuppressionV3OpTest, TestEmptyInput) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({0, 4}), {});
  AddInputFromArray<float>(TensorShape({0}), {});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({0}));
  test::FillValues<int>(&expected, {});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

//
// NonMaxSuppressionV4Op Tests
//

class NonMaxSuppressionV4OpTest : public OpsTestBase {
 protected:
  void MakeOp() {
    TF_EXPECT_OK(NodeDefBuilder("non_max_suppression_op", "NonMaxSuppressionV4")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("pad_to_max_output_size", true)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(NonMaxSuppressionV4OpTest, TestSelectFromThreeClustersPadFive) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {5});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  const auto expected_indices = test::AsTensor<int>({3, 0, 5, 0, 0});
  test::ExpectTensorEqual<int>(expected_indices, *GetOutput(0));
  Tensor expected_num_valid = test::AsScalar<int>(3);
  test::ExpectTensorEqual<int>(expected_num_valid, *GetOutput(1));
}

TEST_F(NonMaxSuppressionV4OpTest, TestSelectFromThreeClustersPadFiveScoreThr) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {6});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.4f});
  TF_ASSERT_OK(RunOpKernel());

  const auto expected_indices = test::AsTensor<int>({3, 0, 0, 0, 0, 0});
  test::ExpectTensorEqual<int>(expected_indices, *GetOutput(0));
  Tensor expected_num_valid = test::AsScalar<int>(2);
  test::ExpectTensorEqual<int>(expected_num_valid, *GetOutput(1));
}

//
// NonMaxSuppressionWithOverlapsOp Tests
//

class NonMaxSuppressionWithOverlapsOpTest : public OpsTestBase {
 protected:
  void MakeOp() {
    TF_EXPECT_OK(NodeDefBuilder("non_max_suppression_op",
                                "NonMaxSuppressionWithOverlaps")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }

  void AddIoUInput(const std::vector<float>& boxes) {
    ASSERT_EQ((boxes.size() % 4), 0);
    size_t num_boxes = boxes.size() / 4;
    std::vector<float> iou_overlaps(num_boxes * num_boxes);

    // compute the pairwise IoU overlaps
    auto corner_access = [&boxes](size_t box_idx, size_t corner_idx) {
      return boxes[box_idx * 4 + corner_idx];
    };
    for (size_t i = 0; i < num_boxes; ++i) {
      for (size_t j = 0; j < num_boxes; ++j) {
        const float ymin_i =
            std::min<float>(corner_access(i, 0), corner_access(i, 2));
        const float xmin_i =
            std::min<float>(corner_access(i, 1), corner_access(i, 3));
        const float ymax_i =
            std::max<float>(corner_access(i, 0), corner_access(i, 2));
        const float xmax_i =
            std::max<float>(corner_access(i, 1), corner_access(i, 3));
        const float ymin_j =
            std::min<float>(corner_access(j, 0), corner_access(j, 2));
        const float xmin_j =
            std::min<float>(corner_access(j, 1), corner_access(j, 3));
        const float ymax_j =
            std::max<float>(corner_access(j, 0), corner_access(j, 2));
        const float xmax_j =
            std::max<float>(corner_access(j, 1), corner_access(j, 3));
        const float area_i = (ymax_i - ymin_i) * (xmax_i - xmin_i);
        const float area_j = (ymax_j - ymin_j) * (xmax_j - xmin_j);

        float iou;
        if (area_i <= 0 || area_j <= 0) {
          iou = 0.0;
        } else {
          const float intersection_ymin = std::max<float>(ymin_i, ymin_j);
          const float intersection_xmin = std::max<float>(xmin_i, xmin_j);
          const float intersection_ymax = std::min<float>(ymax_i, ymax_j);
          const float intersection_xmax = std::min<float>(xmax_i, xmax_j);
          const float intersection_area =
              std::max<float>(intersection_ymax - intersection_ymin, 0.0) *
              std::max<float>(intersection_xmax - intersection_xmin, 0.0);
          iou = intersection_area / (area_i + area_j - intersection_area);
        }
        iou_overlaps[i * num_boxes + j] = iou;
      }
    }

    AddInputFromArray<float>(TensorShape({static_cast<signed>(num_boxes),
                                          static_cast<signed>(num_boxes)}),
                             iou_overlaps);
  }
};

TEST_F(NonMaxSuppressionWithOverlapsOpTest, TestSelectFromThreeClusters) {
  MakeOp();
  AddIoUInput({0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
               0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionWithOverlapsOpTest,
       TestSelectFromThreeClustersFlippedCoordinates) {
  MakeOp();
  AddIoUInput({1, 1,  0, 0,  0, 0.1f,  1, 1.1f,  0, .9f, 1, -0.1f,
               0, 10, 1, 11, 1, 10.1f, 0, 11.1f, 1, 101, 0, 100});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionWithOverlapsOpTest,
       TestSelectAtMostTwoBoxesFromThreeClusters) {
  MakeOp();
  AddIoUInput({0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
               0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {2});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int>(&expected, {3, 0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionWithOverlapsOpTest,
       TestSelectAtMostThirtyBoxesFromThreeClusters) {
  MakeOp();
  AddIoUInput({0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
               0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionWithOverlapsOpTest, TestSelectSingleBox) {
  MakeOp();
  AddIoUInput({0, 0, 1, 1});
  AddInputFromArray<float>(TensorShape({1}), {.9f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int>(&expected, {0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionWithOverlapsOpTest, TestSelectFromTenIdenticalBoxes) {
  MakeOp();

  int num_boxes = 10;
  std::vector<float> corners(num_boxes * 4);
  std::vector<float> scores(num_boxes);
  for (int i = 0; i < num_boxes; ++i) {
    corners[i * 4 + 0] = 0;
    corners[i * 4 + 1] = 0;
    corners[i * 4 + 2] = 1;
    corners[i * 4 + 3] = 1;
    scores[i] = .9;
  }
  AddIoUInput(corners);
  AddInputFromArray<float>(TensorShape({num_boxes}), scores);
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int>(&expected, {0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionWithOverlapsOpTest, TestInconsistentBoxAndScoreShapes) {
  MakeOp();
  AddIoUInput({0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
               0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({5}), {.9f, .75f, .6f, .95f, .5f});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  Status s = RunOpKernel();

  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(
      str_util::StrContains(s.ToString(), "scores has incompatible shape"))
      << s;
}

TEST_F(NonMaxSuppressionWithOverlapsOpTest, TestInvalidOverlapsShape) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({2, 3}), {0, 0, 0, 0, 0, 0});
  AddInputFromArray<float>(TensorShape({2}), {0.5f, 0.5f});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<float>(TensorShape({}), {0.f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  Status s = RunOpKernel();

  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(str_util::StrContains(s.ToString(), "overlaps must be square"))
      << s;
}

TEST_F(NonMaxSuppressionWithOverlapsOpTest, TestThresholdGreaterOne) {
  MakeOp();
  AddIoUInput({0, 0, 1, 1});
  AddInputFromArray<float>(TensorShape({1}), {.9f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {1.2f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());
}

TEST_F(NonMaxSuppressionWithOverlapsOpTest, TestThresholdSmallerZero) {
  MakeOp();
  AddIoUInput({0, 0, 1, 1});
  AddInputFromArray<float>(TensorShape({1}), {.9f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {-0.2f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());
}

TEST_F(NonMaxSuppressionWithOverlapsOpTest, TestEmptyInput) {
  MakeOp();
  AddIoUInput({});
  AddInputFromArray<float>(TensorShape({0}), {});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({0}));
  test::FillValues<int>(&expected, {});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

class CombinedNonMaxSuppressionOpTest : public OpsTestBase {
 protected:
  void MakeOp(bool pad_per_class = false) {
    TF_EXPECT_OK(NodeDefBuilder("combined_non_max_suppression_op",
                                "CombinedNonMaxSuppression")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("pad_per_class", pad_per_class)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(CombinedNonMaxSuppressionOpTest, TestEmptyInput) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({0, 0, 0, 4}), {});
  AddInputFromArray<float>(TensorShape({0, 0, 0}), {});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<int>(TensorShape({}), {10});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  // boxes
  Tensor expected_boxes(allocator(), DT_FLOAT, TensorShape({0, 10, 4}));
  test::FillValues<float>(&expected_boxes, {});
  test::ExpectTensorEqual<float>(expected_boxes, *GetOutput(0));
  
  // scores
  Tensor expected_scores(allocator(), DT_FLOAT, TensorShape({0, 10}));
  test::FillValues<float>(&expected_scores, {});
  test::ExpectTensorEqual<float>(expected_scores, *GetOutput(1));
  
  // classes
  Tensor expected_classes(allocator(), DT_FLOAT, TensorShape({0, 10}));
  test::FillValues<float>(&expected_classes, {});
  test::ExpectTensorEqual<float>(expected_classes, *GetOutput(2));
  
  // valid
  Tensor expected_valid_d(allocator(), DT_INT32, TensorShape({0}));
  test::FillValues<int>(&expected_valid_d, {});
  test::ExpectTensorEqual<int>(expected_valid_d, *GetOutput(3));
  
  // indices
  Tensor expected_indices(allocator(), DT_INT32, TensorShape({0, 10}));
  test::FillValues<int>(&expected_indices, {});
  test::ExpectTensorEqual<int>(expected_indices, *GetOutput(4));
}

TEST_F(CombinedNonMaxSuppressionOpTest, TestSelectFromThreeClusters) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({1, 6, 1, 4}),
      {0, 0,    0.1, 0.1, 0, 0.01f, 0.1, 0.11f, 0, -0.01, 0.1, 0.09f,
       0, 0.11, 0.1, 0.2, 0, 0.12f, 0.1, 0.21f, 0,  0.3, 1,   0.4});
  AddInputFromArray<float>(TensorShape({1, 6, 1}), 
       {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());
  
  // boxes
  Tensor expected_boxes(allocator(), DT_FLOAT, TensorShape({1, 3, 4}));
  test::FillValues<float>(&expected_boxes, {0, 0.11, 0.1, 0.2, 0, 0, 0.1, 0.1, 
          0, 0.3, 1, 0.4});
  test::ExpectTensorEqual<float>(expected_boxes, *GetOutput(0));
  // scores
  Tensor expected_scores(allocator(), DT_FLOAT, TensorShape({1, 3}));
  test::FillValues<float>(&expected_scores, {0.95, 0.9, 0.3});
  test::ExpectTensorEqual<float>(expected_scores, *GetOutput(1));
  // classes
  Tensor expected_classes(allocator(), DT_FLOAT, TensorShape({1, 3}));
  test::FillValues<float>(&expected_classes, {0, 0, 0});
  test::ExpectTensorEqual<float>(expected_classes, *GetOutput(2));
  // valid
  Tensor expected_valid_d(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int>(&expected_valid_d, {3});
  test::ExpectTensorEqual<int>(expected_valid_d, *GetOutput(3));
  // indices
  Tensor expected_indices(allocator(), DT_INT32, TensorShape({1, 3}));
  test::FillValues<int>(&expected_indices, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected_indices, *GetOutput(4));
}

TEST_F(CombinedNonMaxSuppressionOpTest,
        TestSelectFromThreeClustersWithScoreThreshold) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({1, 6, 1, 4}),
      {0, 0,    0.1, 0.1, 0, 0.01f, 0.1, 0.11f, 0, -0.01, 0.1, 0.09f,
       0, 0.11, 0.1, 0.2, 0, 0.12f, 0.1, 0.21f, 0,  0.3, 1,   0.4});
  AddInputFromArray<float>(TensorShape({1, 6, 1}), 
       {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.4f});
  TF_ASSERT_OK(RunOpKernel());
  
  // boxes
  Tensor expected_boxes(allocator(), DT_FLOAT, TensorShape({1, 3, 4}));
  test::FillValues<float>(&expected_boxes, {0, 0.11, 0.1, 0.2, 0, 0, 0.1, 0.1,
      0, 0, 0, 0});
  test::ExpectTensorEqual<float>(expected_boxes, *GetOutput(0));
  // scores
  Tensor expected_scores(allocator(), DT_FLOAT, TensorShape({1, 3}));
  test::FillValues<float>(&expected_scores, {0.95, 0.9, 0});
  test::ExpectTensorEqual<float>(expected_scores, *GetOutput(1));
  // classes
  Tensor expected_classes(allocator(), DT_FLOAT, TensorShape({1, 3}));
  test::FillValues<float>(&expected_classes, {0, 0, 0});
  test::ExpectTensorEqual<float>(expected_classes, *GetOutput(2));
  // valid
  Tensor expected_valid_d(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int>(&expected_valid_d, {2});
  test::ExpectTensorEqual<int>(expected_valid_d, *GetOutput(3));
  // indices
  Tensor expected_indices(allocator(), DT_INT32, TensorShape({1, 3}));
  test::FillValues<int>(&expected_indices, {3, 0, 0});
  test::ExpectTensorEqual<int>(expected_indices, *GetOutput(4));
}

TEST_F(CombinedNonMaxSuppressionOpTest,
        TestSelectFromThreeClustersWithScoreThresholdZeroScores) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({1, 6, 1, 4}),
      {0, 0,    0.1, 0.1, 0, 0.01f, 0.1, 0.11f, 0, -0.01, 0.1, 0.09f,
       0, 0.11, 0.1, 0.2, 0, 0.12f, 0.1, 0.21f, 0,  0.3, 1,   0.4});
  AddInputFromArray<float>(TensorShape({1, 6, 1}), 
       {.1f, 0, 0, .3f, .2f, -5.0f});
  // If we ask for more boxes than we actually expect to get back;
  // should still only get 2 boxes back.
  AddInputFromArray<int>(TensorShape({}), {4});
  AddInputFromArray<int>(TensorShape({}), {5});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {-3.0f});
  TF_ASSERT_OK(RunOpKernel());
  
  // boxes
  Tensor expected_boxes(allocator(), DT_FLOAT, TensorShape({1, 5, 4}));
  test::FillValues<float>(&expected_boxes, {0, 0.11, 0.1, 0.2, 0, 0, 0.1, 0.1, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,});
  test::ExpectTensorEqual<float>(expected_boxes, *GetOutput(0));
  // scores
  Tensor expected_scores(allocator(), DT_FLOAT, TensorShape({1, 5}));
  test::FillValues<float>(&expected_scores, {0.3, 0.1, 0, 0, 0});
  test::ExpectTensorEqual<float>(expected_scores, *GetOutput(1));
  // classes
  Tensor expected_classes(allocator(), DT_FLOAT, TensorShape({1, 5}));
  test::FillValues<float>(&expected_classes, {0, 0, 0, 0, 0});
  test::ExpectTensorEqual<float>(expected_classes, *GetOutput(2));
  // valid
  Tensor expected_valid_d(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int>(&expected_valid_d, {2});
  test::ExpectTensorEqual<int>(expected_valid_d, *GetOutput(3));
  // indices
  Tensor expected_indices(allocator(), DT_INT32, TensorShape({1, 5}));
  test::FillValues<int>(&expected_indices, {3, 0, 0, 0, 0});
  test::ExpectTensorEqual<int>(expected_indices, *GetOutput(4));
}

TEST_F(CombinedNonMaxSuppressionOpTest, TestSelectSingleBox) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({1, 1, 1, 4}), {0, 0, 1, 1});  
  AddInputFromArray<float>(TensorShape({1, 1, 1}), {.9f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<int>(TensorShape({}), {1});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());
  
  // boxes
  Tensor expected_boxes(allocator(), DT_FLOAT, TensorShape({1, 1, 4}));
  test::FillValues<float>(&expected_boxes, {0, 0, 1, 1});
  test::ExpectTensorEqual<float>(expected_boxes, *GetOutput(0));
  // scores
  Tensor expected_scores(allocator(), DT_FLOAT, TensorShape({1, 1}));
  test::FillValues<float>(&expected_scores, {0.9});
  test::ExpectTensorEqual<float>(expected_scores, *GetOutput(1));
  // classes
  Tensor expected_classes(allocator(), DT_FLOAT, TensorShape({1, 1}));
  test::FillValues<float>(&expected_classes, {0});
  test::ExpectTensorEqual<float>(expected_classes, *GetOutput(2));
  // valid
  Tensor expected_valid_d(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int>(&expected_valid_d, {1});
  test::ExpectTensorEqual<int>(expected_valid_d, *GetOutput(3));
  // indices
  Tensor expected_indices(allocator(), DT_INT32, TensorShape({1, 1}));
  test::FillValues<int>(&expected_indices, {0});
  test::ExpectTensorEqual<int>(expected_indices, *GetOutput(4));
}

TEST_F(CombinedNonMaxSuppressionOpTest,
        TestSelectFromTwoBatchesWithScoreThreshold) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({2, 6, 1, 4}),
      {0, 0,    0.1, 0.1, 0, 0.01f, 0.1, 0.11f, 0, -0.01, 0.1, 0.09f,
       0, 0.11, 0.1, 0.2, 0, 0.12f, 0.1, 0.21f, 0,  0.3,  1,   0.4,
       0, 0,    0.2, 0.2, 0, 0.02f, 0.2, 0.22f, 0, -0.02, 0.2, 0.19f,
       0, 0.21, 0.2, 0.3, 0, 0.22f, 0.2, 0.31f, 0,  0.4,  1,   0.5
       });
  AddInputFromArray<float>(TensorShape({2, 6, 1}), 
       {.9f, .75f, .6f, .95f, .5f, .3f, .9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.4f});
  TF_ASSERT_OK(RunOpKernel());
  
  // boxes
  Tensor expected_boxes(allocator(), DT_FLOAT, TensorShape({2, 3, 4}));
  test::FillValues<float>(&expected_boxes, {0, 0.11, 0.1, 0.2, 0, 0, 0.1, 0.1, 
          0, 0, 0, 0, 0, 0.21, 0.2, 0.3, 0, 0, 0.2, 0.2, 0, 0, 0, 0});
  test::ExpectTensorEqual<float>(expected_boxes, *GetOutput(0));
  // scores
  Tensor expected_scores(allocator(), DT_FLOAT, TensorShape({2, 3}));
  test::FillValues<float>(&expected_scores, {0.95, 0.9, 0, 0.95, 0.9, 0});
  test::ExpectTensorEqual<float>(expected_scores, *GetOutput(1));
  // classes
  Tensor expected_classes(allocator(), DT_FLOAT, TensorShape({2, 3}));
  test::FillValues<float>(&expected_classes, {0, 0, 0, 0, 0, 0});
  test::ExpectTensorEqual<float>(expected_classes, *GetOutput(2));
  // valid
  Tensor expected_valid_d(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int>(&expected_valid_d, {2, 2});
  test::ExpectTensorEqual<int>(expected_valid_d, *GetOutput(3));
  // indices
  Tensor expected_indices(allocator(), DT_INT32, TensorShape({2, 3}));
  test::FillValues<int>(&expected_indices, {3, 0, 0, 3, 0, 0});
  test::ExpectTensorEqual<int>(expected_indices, *GetOutput(4));
}

TEST_F(CombinedNonMaxSuppressionOpTest,
        TestSelectFromTwoBatchesTwoClasses) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({2, 6, 1, 4}),
      {0, 0,    0.1, 0.1, 0, 0.01f, 0.1, 0.11f, 0, -0.01, 0.1, 0.09f,
       0, 0.11, 0.1, 0.2, 0, 0.12f, 0.1, 0.21f, 0,  0.3,  1,   0.4,
       0, 0,    0.2, 0.2, 0, 0.02f, 0.2, 0.22f, 0, -0.02, 0.2, 0.19f,
       0, 0.21, 0.2, 0.3, 0, 0.22f, 0.2, 0.31f, 0,  0.4,  1,   0.5
       });
  AddInputFromArray<float>(TensorShape({2, 6, 2}), 
       {0.1f, 0.9f, 0.75f, 0.8f, 0.6f, 0.3f, 0.95f, 0.1f, 0.5f, 0.5f, 0.3f, 0.1f, 
       0.1f, 0.9f, 0.75f, 0.8f, 0.6f, 0.3f, 0.95f, 0.1f, 0.5f, 0.5f, 0.3f, 0.1f}); 
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());
  
  // boxes
  Tensor expected_boxes(allocator(), DT_FLOAT, TensorShape({2, 3, 4}));
  test::FillValues<float>(&expected_boxes, {0, 0.11, 0.1, 0.2, 0, 0, 0.1, 0.1, 
          0, 0.01f, 0.1, 0.11f, 0, 0.21, 0.2, 0.3, 0, 0, 0.2, 0.2, 
          0, 0.02f, 0.2, 0.22f});
  test::ExpectTensorEqual<float>(expected_boxes, *GetOutput(0));
  // scores
  Tensor expected_scores(allocator(), DT_FLOAT, TensorShape({2, 3}));
  test::FillValues<float>(&expected_scores, {0.95, 0.9, 0.75, 0.95, 0.9, 0.75});
  test::ExpectTensorEqual<float>(expected_scores, *GetOutput(1));
  // classes
  Tensor expected_classes(allocator(), DT_FLOAT, TensorShape({2, 3}));
  test::FillValues<float>(&expected_classes, {0, 1, 0, 0, 1, 0});
  test::ExpectTensorEqual<float>(expected_classes, *GetOutput(2));
  // valid
  Tensor expected_valid_d(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int>(&expected_valid_d, {3, 3});
  test::ExpectTensorEqual<int>(expected_valid_d, *GetOutput(3));
  // indices
  Tensor expected_indices(allocator(), DT_INT32, TensorShape({2, 3}));
  test::FillValues<int>(&expected_indices, {3, 0, 1, 3, 0, 1});
  test::ExpectTensorEqual<int>(expected_indices, *GetOutput(4));
}

TEST_F(CombinedNonMaxSuppressionOpTest,
        TestSelectFromTwoBatchesTwoClassesWithScoreThreshold) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({2, 6, 1, 4}),
      {0, 0,    0.1, 0.1, 0, 0.01f, 0.1, 0.11f, 0, -0.01, 0.1, 0.09f,
       0, 0.11, 0.1, 0.2, 0, 0.12f, 0.1, 0.21f, 0,  0.3,  1,   0.4,
       0, 0,    0.2, 0.2, 0, 0.02f, 0.2, 0.22f, 0, -0.02, 0.2, 0.19f,
       0, 0.21, 0.2, 0.3, 0, 0.22f, 0.2, 0.31f, 0,  0.4,  1,   0.5
       });
  AddInputFromArray<float>(TensorShape({2, 6, 2}), 
      {0.1f, 0.9f, 0.75f, 0.8f, 0.6f, 0.3f, 0.95f, 0.1f, 0.5f, 0.5f, 0.3f, 0.1f, 
       0.1f, 0.9f, 0.75f, 0.8f, 0.6f, 0.3f, 0.95f, 0.1f, 0.5f, 0.5f, 0.3f, 0.1f
       }); 
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.8f});
  TF_ASSERT_OK(RunOpKernel());
  
  // boxes
  Tensor expected_boxes(allocator(), DT_FLOAT, TensorShape({2, 3, 4}));
  test::FillValues<float>(&expected_boxes, {0, 0.11, 0.1, 0.2, 0, 0, 0.1, 0.1, 
          0, 0, 0, 0, 0, 0.21, 0.2, 0.3, 0, 0, 0.2, 0.2, 0, 0, 0, 0});
  test::ExpectTensorEqual<float>(expected_boxes, *GetOutput(0));
  // scores
  Tensor expected_scores(allocator(), DT_FLOAT, TensorShape({2, 3}));
  test::FillValues<float>(&expected_scores, {0.95, 0.9, 0, 0.95, 0.9, 0});
  test::ExpectTensorEqual<float>(expected_scores, *GetOutput(1));
  // classes
  Tensor expected_classes(allocator(), DT_FLOAT, TensorShape({2, 3}));
  test::FillValues<float>(&expected_classes, {0, 1, 0, 0, 1, 0});
  test::ExpectTensorEqual<float>(expected_classes, *GetOutput(2));
  // valid
  Tensor expected_valid_d(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int>(&expected_valid_d, {2, 2});
  test::ExpectTensorEqual<int>(expected_valid_d, *GetOutput(3));
  // indices
  Tensor expected_indices(allocator(), DT_INT32, TensorShape({2, 3}));
  test::FillValues<int>(&expected_indices, {3, 0, 0, 3, 0, 0});
  test::ExpectTensorEqual<int>(expected_indices, *GetOutput(4));
}

TEST_F(CombinedNonMaxSuppressionOpTest,
        TestSelectFromTwoBatchesTwoClassesWithScoreThresholdPaddedTotalSize) {
  MakeOp(true);
  AddInputFromArray<float>(
      TensorShape({2, 6, 1, 4}),
      {0, 0,    0.1, 0.1, 0, 0.01f, 0.1, 0.11f, 0, -0.01, 0.1, 0.09f,
       0, 0.11, 0.1, 0.2, 0, 0.12f, 0.1, 0.21f, 0,  0.3,  1,   0.4,
       0, 0,    0.2, 0.2, 0, 0.02f, 0.2, 0.22f, 0, -0.02, 0.2, 0.19f,
       0, 0.21, 0.2, 0.3, 0, 0.22f, 0.2, 0.31f, 0,  0.4,  1,   0.5
       });
  AddInputFromArray<float>(TensorShape({2, 6, 2}), 
      {0.1f, 0.9f, 0.75f, 0.8f, 0.6f, 0.3f, 0.95f, 0.1f, 0.5f, 0.5f, 0.3f, 0.1f, 
       0.1f, 0.9f, 0.75f, 0.8f, 0.6f, 0.3f, 0.95f, 0.1f, 0.5f, 0.5f, 0.3f, 0.1f
       }); 
  AddInputFromArray<int>(TensorShape({}), {10});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.8f});
  TF_ASSERT_OK(RunOpKernel());
  
  // boxes
  Tensor expected_boxes(allocator(), DT_FLOAT, TensorShape({2, 3, 4}));
  test::FillValues<float>(&expected_boxes, {0, 0.11, 0.1, 0.2, 0, 0, 0.1, 0.1, 
          0, 0, 0, 0, 0, 0.21, 0.2, 0.3, 0, 0, 0.2, 0.2, 0, 0, 0, 0});
  test::ExpectTensorEqual<float>(expected_boxes, *GetOutput(0));
  // scores
  Tensor expected_scores(allocator(), DT_FLOAT, TensorShape({2, 3}));
  test::FillValues<float>(&expected_scores, {0.95, 0.9, 0, 0.95, 0.9, 0});
  test::ExpectTensorEqual<float>(expected_scores, *GetOutput(1));
  // classes
  Tensor expected_classes(allocator(), DT_FLOAT, TensorShape({2, 3}));
  test::FillValues<float>(&expected_classes, {0, 1, 0, 0, 1, 0});
  test::ExpectTensorEqual<float>(expected_classes, *GetOutput(2));
  // valid
  Tensor expected_valid_d(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int>(&expected_valid_d, {2, 2});
  test::ExpectTensorEqual<int>(expected_valid_d, *GetOutput(3));
  // indices
  Tensor expected_indices(allocator(), DT_INT32, TensorShape({2, 3}));
  test::FillValues<int>(&expected_indices, {3, 0, 0, 3, 0, 0});
  test::ExpectTensorEqual<int>(expected_indices, *GetOutput(4));
}

TEST_F(CombinedNonMaxSuppressionOpTest,
        TestSelectFromTwoBatchesTwoClassesWithScoreThresholdPaddedPerClass) {
  MakeOp(true);
  AddInputFromArray<float>(
      TensorShape({2, 6, 1, 4}),
      {0, 0,    0.1, 0.1, 0, 0.01f, 0.1, 0.11f, 0, -0.01, 0.1, 0.09f,
       0, 0.11, 0.1, 0.2, 0, 0.12f, 0.1, 0.21f, 0,  0.3,  1,   0.4,
       0, 0,    0.2, 0.2, 0, 0.02f, 0.2, 0.22f, 0, -0.02, 0.2, 0.19f,
       0, 0.21, 0.2, 0.3, 0, 0.22f, 0.2, 0.31f, 0,  0.4,  1,   0.5
       });
  AddInputFromArray<float>(TensorShape({2, 6, 2}),
      {0.1f, 0.9f, 0.75f, 0.8f, 0.6f, 0.3f, 0.95f, 0.1f, 0.5f, 0.5f, 0.3f, 0.1f,
       0.1f, 0.9f, 0.75f, 0.8f, 0.6f, 0.3f, 0.95f, 0.1f, 0.5f, 0.5f, 0.3f, 0.1f
       });
  AddInputFromArray<int>(TensorShape({}), {2});
  AddInputFromArray<int>(TensorShape({}), {50});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.8f});
  TF_ASSERT_OK(RunOpKernel());

  // boxes
  Tensor expected_boxes(allocator(), DT_FLOAT, TensorShape({2, 4, 4}));
  test::FillValues<float>(&expected_boxes, {0, 0.11, 0.1, 0.2, 0, 0, 0.1, 0.1,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0.21, 0.2, 0.3, 0, 0, 0.2, 0.2, 0, 0, 0, 0,
          0, 0, 0, 0});
  test::ExpectTensorEqual<float>(expected_boxes, *GetOutput(0));
  // scores
  Tensor expected_scores(allocator(), DT_FLOAT, TensorShape({2, 4}));
  test::FillValues<float>(&expected_scores, {0.95, 0.9, 0, 0, 0.95, 0.9, 0, 0});
  test::ExpectTensorEqual<float>(expected_scores, *GetOutput(1));
  // classes
  Tensor expected_classes(allocator(), DT_FLOAT, TensorShape({2, 4}));
  test::FillValues<float>(&expected_classes, {0, 1, 0, 0, 0, 1, 0, 0});
  test::ExpectTensorEqual<float>(expected_classes, *GetOutput(2));
  // valid
  Tensor expected_valid_d(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int>(&expected_valid_d, {2, 2});
  test::ExpectTensorEqual<int>(expected_valid_d, *GetOutput(3));
  // indices
  Tensor expected_indices(allocator(), DT_INT32, TensorShape({2, 4}));
  test::FillValues<int>(&expected_indices, {3, 0, 0, 0, 3, 0, 0, 0});
  test::ExpectTensorEqual<int>(expected_indices, *GetOutput(4));
}

TEST_F(CombinedNonMaxSuppressionOpTest,
        TestSelectFromTwoBatchesTwoClassesTotalSize) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({2, 6, 1, 4}),
      {0, 0,    0.1, 0.1, 0, 0.01f, 0.1, 0.11f, 0, -0.01, 0.1, 0.09f,
       0, 0.11, 0.1, 0.2, 0, 0.12f, 0.1, 0.21f, 0,  0.3,  1,   0.4,
       0, 0,    0.2, 0.2, 0, 0.02f, 0.2, 0.22f, 0, -0.02, 0.2, 0.19f,
       0, 0.21, 0.2, 0.3, 0, 0.22f, 0.2, 0.31f, 0,  0.4,  1,   0.5
       });
  AddInputFromArray<float>(TensorShape({2, 6, 2}), 
      {0.1f, 0.9f, 0.75f, 0.8f, 0.6f, 0.3f, 0.95f, 0.1f, 0.5f, 0.5f, 0.3f, 0.1f, 
       0.1f, 0.9f, 0.75f, 0.8f, 0.6f, 0.3f, 0.95f, 0.1f, 0.5f, 0.5f, 0.3f, 0.1f
       }); 
  AddInputFromArray<int>(TensorShape({}), {3});
  // Total size per batch is more than size per class
  AddInputFromArray<int>(TensorShape({}), {5});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.1f});
  TF_ASSERT_OK(RunOpKernel());
  
  // boxes
  Tensor expected_boxes(allocator(), DT_FLOAT, TensorShape({2, 5, 4}));
  test::FillValues<float>(&expected_boxes, {0, 0.11, 0.1, 0.2, 0, 0, 0.1, 0.1, 
          0, 0.01f, 0.1, 0.11f, 0, 0.12f, 0.1, 0.21f, 0,  0.3, 1, 0.4,
          0, 0.21, 0.2, 0.3, 0, 0, 0.2, 0.2, 0, 0.02f, 0.2, 0.22f,
          0, 0.22f, 0.2, 0.31f, 0, 0.4, 1, 0.5});
  test::ExpectTensorEqual<float>(expected_boxes, *GetOutput(0));
  // scores
  Tensor expected_scores(allocator(), DT_FLOAT, TensorShape({2, 5}));
  test::FillValues<float>(&expected_scores, {0.95, 0.9, 0.75, 0.5, 0.3, 
          0.95, 0.9, 0.75, 0.5, 0.3});
  test::ExpectTensorEqual<float>(expected_scores, *GetOutput(1));
  // classes
  Tensor expected_classes(allocator(), DT_FLOAT, TensorShape({2, 5}));
  test::FillValues<float>(&expected_classes, {0, 1, 0, 1, 0, 0, 1, 0, 1, 0});
  test::ExpectTensorEqual<float>(expected_classes, *GetOutput(2));
  // valid
  Tensor expected_valid_d(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int>(&expected_valid_d, {5, 5});
  test::ExpectTensorEqual<int>(expected_valid_d, *GetOutput(3));
  // indices
  Tensor expected_indices(allocator(), DT_INT32, TensorShape({2, 5}));
  test::FillValues<int>(&expected_indices, {3, 0, 1, 4, 5, 3, 0, 1, 4, 5});
  test::ExpectTensorEqual<int>(expected_indices, *GetOutput(4));
}

TEST_F(CombinedNonMaxSuppressionOpTest,
        TestSelectFromTwoBatchesTwoClassesForBoxesAndScores) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({2, 6, 2, 4}),
      // batch 0, box1 of class 1 should get selected
      {0, 0,     0.1, 0.1,   0, 0,     0.1, 0.1, 
       0, 0.01f, 0.1, 0.11f, 0, 0.6f,  0.1, 0.7f,
       0, -0.01, 0.1, 0.09f, 0, -0.01, 0.1, 0.09f, 
       0, 0.11,  0.1, 0.2,   0, 0.11,  0.1, 0.2, 
       0, 0.12f, 0.1, 0.21f, 0, 0.12f, 0.1, 0.21f, 
       0, 0.3,   1,   0.4,   0, 0.3,   1,   0.4, 
      // batch 1, box1 of class 0 should get selected
       0, 0,     0.2, 0.2,   0, 0,     0.2, 0.2, 
       0, 0.02f, 0.2, 0.22f, 0, 0.02f, 0.2, 0.22f,
       0, -0.02, 0.2, 0.19f, 0, -0.02, 0.2, 0.19f,
       0, 0.21,  0.2, 0.3,   0, 0.21,  0.2, 0.3, 
       0, 0.22f, 0.2, 0.31f, 0, 0.22f, 0.2, 0.31f,
       0,  0.4,  1,   0.5,   0, 0.4,   1,   0.5});

  AddInputFromArray<float>(TensorShape({2, 6, 2}), 
      {0.1f, 0.9f, 0.75f, 0.8f, 0.6f, 0.3f, 0.95f, 0.1f, 0.5f, 0.5f, 0.3f, 0.1f, 
       0.1f, 0.9f, 0.75f, 0.8f, 0.6f, 0.3f, 0.95f, 0.1f, 0.5f, 0.5f, 0.3f, 0.1f
       }); 
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());
  
  // boxes
  Tensor expected_boxes(allocator(), DT_FLOAT, TensorShape({2, 3, 4}));
  test::FillValues<float>(&expected_boxes, {0, 0.11, 0.1, 0.2, 0, 0, 0.1, 0.1, 
           0, 0.6f, 0.1, 0.7f, 0, 0.21, 0.2, 0.3, 0, 0, 0.2, 0.2, 0, 0.02f, 0.2, 0.22f});
  test::ExpectTensorEqual<float>(expected_boxes, *GetOutput(0));
  // scores
  Tensor expected_scores(allocator(), DT_FLOAT, TensorShape({2, 3}));
  test::FillValues<float>(&expected_scores, {0.95, 0.9, 0.8, 0.95, 0.9, 0.75});
  test::ExpectTensorEqual<float>(expected_scores, *GetOutput(1));
  // classes
  Tensor expected_classes(allocator(), DT_FLOAT, TensorShape({2, 3}));
  test::FillValues<float>(&expected_classes, {0, 1, 1, 0, 1, 0});
  test::ExpectTensorEqual<float>(expected_classes, *GetOutput(2));
  // valid
  Tensor expected_valid_d(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int>(&expected_valid_d, {3, 3});
  test::ExpectTensorEqual<int>(expected_valid_d, *GetOutput(3));
  // indices
  Tensor expected_indices(allocator(), DT_INT32, TensorShape({2, 3}));
  test::FillValues<int>(&expected_indices, {3, 0, 1, 3, 0, 1});
  test::ExpectTensorEqual<int>(expected_indices, *GetOutput(4));
}

}  // namespace tensorflow
