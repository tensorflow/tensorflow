/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
class CombinedNonMaxSuppressionOpTest : public OpsTestBase {
 protected:
  void MakeOp(bool pad_per_class = false, bool clip_boxes = true) {
    SetDevice(DEVICE_GPU,
              std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                  "GPU", {}, "/job:a/replica:0/task:0")));

    TF_EXPECT_OK(NodeDefBuilder("combined_non_max_suppression_op",
                                "CombinedNonMaxSuppression")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("pad_per_class", pad_per_class)
                     .Attr("clip_boxes", clip_boxes)
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
}

TEST_F(CombinedNonMaxSuppressionOpTest, TestSelectFromThreeClusters) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({1, 6, 1, 4}),
      {0, 0,    0.1, 0.1, 0, 0.01f, 0.1, 0.11f, 0, -0.01, 0.1, 0.09f,
       0, 0.11, 0.1, 0.2, 0, 0.12f, 0.1, 0.21f, 0, 0.3,   1,   0.4});
  AddInputFromArray<float>(TensorShape({1, 6, 1}),
                           {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  // boxes
  Tensor expected_boxes(allocator(), DT_FLOAT, TensorShape({1, 3, 4}));
  test::FillValues<float>(&expected_boxes,
                          {0, 0.11, 0.1, 0.2, 0, 0, 0.1, 0.1, 0, 0.3, 1, 0.4});
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
}

TEST_F(CombinedNonMaxSuppressionOpTest,
       TestSelectFromThreeClustersNoBoxClipping) {
  MakeOp(false, false);
  AddInputFromArray<float>(TensorShape({1, 6, 1, 4}),
                           {0, 0,  10, 10, 0, 1,  10, 11, 0, 1,  10,  9,
                            0, 11, 10, 20, 0, 12, 10, 21, 0, 30, 100, 40});
  AddInputFromArray<float>(TensorShape({1, 6, 1}),
                           {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  // boxes
  Tensor expected_boxes(allocator(), DT_FLOAT, TensorShape({1, 3, 4}));
  test::FillValues<float>(&expected_boxes,
                          {0, 11, 10, 20, 0, 0, 10, 10, 0, 30, 100, 40});
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
}

TEST_F(CombinedNonMaxSuppressionOpTest,
       TestSelectFromThreeClustersWithScoreThreshold) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({1, 6, 1, 4}),
      {0, 0,    0.1, 0.1, 0, 0.01f, 0.1, 0.11f, 0, -0.01, 0.1, 0.09f,
       0, 0.11, 0.1, 0.2, 0, 0.12f, 0.1, 0.21f, 0, 0.3,   1,   0.4});
  AddInputFromArray<float>(TensorShape({1, 6, 1}),
                           {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.4f});
  TF_ASSERT_OK(RunOpKernel());

  // boxes
  Tensor expected_boxes(allocator(), DT_FLOAT, TensorShape({1, 3, 4}));
  test::FillValues<float>(&expected_boxes,
                          {0, 0.11, 0.1, 0.2, 0, 0, 0.1, 0.1, 0, 0, 0, 0});
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

}

TEST_F(CombinedNonMaxSuppressionOpTest,
       TestSelectFromThreeClustersWithScoreThresholdZeroScores) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({1, 6, 1, 4}),
      {0, 0,    0.1, 0.1, 0, 0.01f, 0.1, 0.11f, 0, -0.01, 0.1, 0.09f,
       0, 0.11, 0.1, 0.2, 0, 0.12f, 0.1, 0.21f, 0, 0.3,   1,   0.4});
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
  test::FillValues<float>(
      &expected_boxes,
      {
          0, 0.11, 0.1, 0.2, 0, 0, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      });
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
}

TEST_F(CombinedNonMaxSuppressionOpTest,
       TestSelectFromTwoBatchesWithScoreThreshold) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({2, 6, 1, 4}),
      {0, 0,    0.1, 0.1, 0, 0.01f, 0.1, 0.11f, 0, -0.01, 0.1, 0.09f,
       0, 0.11, 0.1, 0.2, 0, 0.12f, 0.1, 0.21f, 0, 0.3,   1,   0.4,
       0, 0,    0.2, 0.2, 0, 0.02f, 0.2, 0.22f, 0, -0.02, 0.2, 0.19f,
       0, 0.21, 0.2, 0.3, 0, 0.22f, 0.2, 0.31f, 0, 0.4,   1,   0.5});
  AddInputFromArray<float>(
      TensorShape({2, 6, 1}),
      {.9f, .75f, .6f, .95f, .5f, .3f, .9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.4f});
  TF_ASSERT_OK(RunOpKernel());

  // boxes
  Tensor expected_boxes(allocator(), DT_FLOAT, TensorShape({2, 3, 4}));
  test::FillValues<float>(&expected_boxes,
                          {0, 0.11, 0.1, 0.2, 0, 0, 0.1, 0.1, 0, 0, 0, 0,
                           0, 0.21, 0.2, 0.3, 0, 0, 0.2, 0.2, 0, 0, 0, 0});
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
}

TEST_F(CombinedNonMaxSuppressionOpTest, TestSelectFromTwoBatchesTwoClasses) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({2, 6, 1, 4}),
      {0, 0,    0.1, 0.1,   0, 0.01f, 0.1, 0.11f,   0, -0.01, 0.1, 0.09f,
       0, 0.11, 0.1, 0.2,   0, 0.12f, 0.1, 0.21f,   0, 0.3,   1,   0.4,
       0, 0,    0.2, 0.2,   0, 0.02f, 0.2, 0.22f,   0, -0.02, 0.2, 0.19f,
       0, 0.21, 0.2, 0.3,   0, 0.22f, 0.2, 0.31f,   0, 0.4,   1,   0.5});
  AddInputFromArray<float>(TensorShape({2, 6, 2}),
                           {0.1f, 0.9f, 0.75f, 0.8f, 0.6f, 0.3f, 0.95f, 0.1f,
                            0.5f, 0.5f, 0.3f,  0.1f, 0.1f, 0.9f, 0.75f, 0.8f,
                            0.6f, 0.3f, 0.95f, 0.1f, 0.5f, 0.5f, 0.3f,  0.1f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  // boxes
  Tensor expected_boxes(allocator(), DT_FLOAT, TensorShape({2, 3, 4}));
  test::FillValues<float>(
      &expected_boxes,
      {0, 0.11, 0.1, 0.2, 0, 0, 0.1, 0.1, 0, 0.01f, 0.1, 0.11f,
       0, 0.21, 0.2, 0.3, 0, 0, 0.2, 0.2, 0, 0.02f, 0.2, 0.22f});
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
}

TEST_F(CombinedNonMaxSuppressionOpTest,
       TestSelectFromTwoBatchesTwoClassesWithScoreThreshold) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({2, 6, 1, 4}),
      {0, 0,    0.1, 0.1, 0, 0.01f, 0.1, 0.11f, 0, -0.01, 0.1, 0.09f,
       0, 0.11, 0.1, 0.2, 0, 0.12f, 0.1, 0.21f, 0, 0.3,   1,   0.4,
       0, 0,    0.2, 0.2, 0, 0.02f, 0.2, 0.22f, 0, -0.02, 0.2, 0.19f,
       0, 0.21, 0.2, 0.3, 0, 0.22f, 0.2, 0.31f, 0, 0.4,   1,   0.5});
  AddInputFromArray<float>(TensorShape({2, 6, 2}),
                           {0.1f, 0.9f, 0.75f, 0.8f, 0.6f, 0.3f, 0.95f, 0.1f,
                            0.5f, 0.5f, 0.3f,  0.1f, 0.1f, 0.9f, 0.75f, 0.8f,
                            0.6f, 0.3f, 0.95f, 0.1f, 0.5f, 0.5f, 0.3f,  0.1f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.8f});
  TF_ASSERT_OK(RunOpKernel());

  // boxes
  Tensor expected_boxes(allocator(), DT_FLOAT, TensorShape({2, 3, 4}));
  test::FillValues<float>(&expected_boxes,
                          {0, 0.11, 0.1, 0.2, 0, 0, 0.1, 0.1, 0, 0, 0, 0,
                           0, 0.21, 0.2, 0.3, 0, 0, 0.2, 0.2, 0, 0, 0, 0});
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
}

TEST_F(CombinedNonMaxSuppressionOpTest,
       TestSelectFromTwoBatchesTwoClassesWithScoreThresholdPaddedTotalSize) {
  MakeOp(true);
  AddInputFromArray<float>(
      TensorShape({2, 6, 1, 4}),
      {0, 0,    0.1, 0.1, 0, 0.01f, 0.1, 0.11f, 0, -0.01, 0.1, 0.09f,
       0, 0.11, 0.1, 0.2, 0, 0.12f, 0.1, 0.21f, 0, 0.3,   1,   0.4,
       0, 0,    0.2, 0.2, 0, 0.02f, 0.2, 0.22f, 0, -0.02, 0.2, 0.19f,
       0, 0.21, 0.2, 0.3, 0, 0.22f, 0.2, 0.31f, 0, 0.4,   1,   0.5});
  AddInputFromArray<float>(TensorShape({2, 6, 2}),
                           {0.1f, 0.9f, 0.75f, 0.8f, 0.6f, 0.3f, 0.95f, 0.1f,
                            0.5f, 0.5f, 0.3f,  0.1f, 0.1f, 0.9f, 0.75f, 0.8f,
                            0.6f, 0.3f, 0.95f, 0.1f, 0.5f, 0.5f, 0.3f,  0.1f});
  AddInputFromArray<int>(TensorShape({}), {10});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.8f});
  TF_ASSERT_OK(RunOpKernel());

  // boxes
  Tensor expected_boxes(allocator(), DT_FLOAT, TensorShape({2, 3, 4}));
  test::FillValues<float>(&expected_boxes,
                          {0, 0.11, 0.1, 0.2, 0, 0, 0.1, 0.1, 0, 0, 0, 0,
                           0, 0.21, 0.2, 0.3, 0, 0, 0.2, 0.2, 0, 0, 0, 0});
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
}

TEST_F(CombinedNonMaxSuppressionOpTest,
       TestSelectFromTwoBatchesTwoClassesWithScoreThresholdPaddedPerClass) {
  MakeOp(true);
  AddInputFromArray<float>(
      TensorShape({2, 6, 1, 4}),
      {0, 0,    0.1, 0.1, 0, 0.01f, 0.1, 0.11f, 0, -0.01, 0.1, 0.09f,
       0, 0.11, 0.1, 0.2, 0, 0.12f, 0.1, 0.21f, 0, 0.3,   1,   0.4,
       0, 0,    0.2, 0.2, 0, 0.02f, 0.2, 0.22f, 0, -0.02, 0.2, 0.19f,
       0, 0.21, 0.2, 0.3, 0, 0.22f, 0.2, 0.31f, 0, 0.4,   1,   0.5});
  AddInputFromArray<float>(TensorShape({2, 6, 2}),
                           {0.1f, 0.9f, 0.75f, 0.8f, 0.6f, 0.3f, 0.95f, 0.1f,
                            0.5f, 0.5f, 0.3f,  0.1f, 0.1f, 0.9f, 0.75f, 0.8f,
                            0.6f, 0.3f, 0.95f, 0.1f, 0.5f, 0.5f, 0.3f,  0.1f});
  AddInputFromArray<int>(TensorShape({}), {2});
  AddInputFromArray<int>(TensorShape({}), {50});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.8f});
  TF_ASSERT_OK(RunOpKernel());

  // boxes
  Tensor expected_boxes(allocator(), DT_FLOAT, TensorShape({2, 4, 4}));
  test::FillValues<float>(
      &expected_boxes,
      {0, 0.11, 0.1, 0.2, 0, 0, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0.21, 0.2, 0.3, 0, 0, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0, 0});
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
}

TEST_F(CombinedNonMaxSuppressionOpTest,
       TestSelectFromTwoBatchesTwoClassesTotalSize) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({2, 6, 1, 4}),
      {0, 0,    0.1, 0.1, 0, 0.01f, 0.1, 0.11f, 0, -0.01, 0.1, 0.09f,
       0, 0.11, 0.1, 0.2, 0, 0.12f, 0.1, 0.21f, 0, 0.3,   1,   0.4,
       0, 0,    0.2, 0.2, 0, 0.02f, 0.2, 0.22f, 0, -0.02, 0.2, 0.19f,
       0, 0.21, 0.2, 0.3, 0, 0.22f, 0.2, 0.31f, 0, 0.4,   1,   0.5});
  AddInputFromArray<float>(TensorShape({2, 6, 2}),
                           {0.1f, 0.9f, 0.75f, 0.8f, 0.6f, 0.3f, 0.95f, 0.1f,
                            0.5f, 0.5f, 0.3f,  0.1f, 0.1f, 0.9f, 0.75f, 0.8f,
                            0.6f, 0.3f, 0.95f, 0.1f, 0.5f, 0.5f, 0.3f,  0.1f});
  AddInputFromArray<int>(TensorShape({}), {3});
  // Total size per batch is more than size per class
  AddInputFromArray<int>(TensorShape({}), {5});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.1f});
  TF_ASSERT_OK(RunOpKernel());

  // boxes
  Tensor expected_boxes(allocator(), DT_FLOAT, TensorShape({2, 5, 4}));
  test::FillValues<float>(
      &expected_boxes, {0,   0.11,  0.1, 0.2,   0,   0,     0.1, 0.1, 0, 0.01f,
                        0.1, 0.11f, 0,   0.12f, 0.1, 0.21f, 0,   0.3, 1, 0.4,
                        0,   0.21,  0.2, 0.3,   0,   0,     0.2, 0.2, 0, 0.02f,
                        0.2, 0.22f, 0,   0.22f, 0.2, 0.31f, 0,   0.4, 1, 0.5});
  test::ExpectTensorEqual<float>(expected_boxes, *GetOutput(0));
  // scores
  Tensor expected_scores(allocator(), DT_FLOAT, TensorShape({2, 5}));
  test::FillValues<float>(
      &expected_scores, {0.95, 0.9, 0.75, 0.5, 0.3, 0.95, 0.9, 0.75, 0.5, 0.3});
  test::ExpectTensorEqual<float>(expected_scores, *GetOutput(1));
  // classes
  Tensor expected_classes(allocator(), DT_FLOAT, TensorShape({2, 5}));
  test::FillValues<float>(&expected_classes, {0, 1, 0, 1, 0, 0, 1, 0, 1, 0});
  test::ExpectTensorEqual<float>(expected_classes, *GetOutput(2));
  // valid
  Tensor expected_valid_d(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int>(&expected_valid_d, {5, 5});
  test::ExpectTensorEqual<int>(expected_valid_d, *GetOutput(3));
}

TEST_F(CombinedNonMaxSuppressionOpTest,
       TestSelectFromTwoBatchesTwoClassesForBoxesAndScores) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({2, 6, 2, 4}),
      // batch 0, box1 of class 1 should get selected
      {0, 0, 0.1, 0.1, 0, 0, 0.1, 0.1, 0, 0.01f, 0.1, 0.11f, 0, 0.6f, 0.1, 0.7f,
       0, -0.01, 0.1, 0.09f, 0, -0.01, 0.1, 0.09f, 0, 0.11, 0.1, 0.2, 0, 0.11,
       0.1, 0.2, 0, 0.12f, 0.1, 0.21f, 0, 0.12f, 0.1, 0.21f, 0, 0.3, 1, 0.4, 0,
       0.3, 1, 0.4,
       // batch 1, box1 of class 0 should get selected
       0, 0, 0.2, 0.2, 0, 0, 0.2, 0.2, 0, 0.02f, 0.2, 0.22f, 0, 0.02f, 0.2,
       0.22f, 0, -0.02, 0.2, 0.19f, 0, -0.02, 0.2, 0.19f, 0, 0.21, 0.2, 0.3, 0,
       0.21, 0.2, 0.3, 0, 0.22f, 0.2, 0.31f, 0, 0.22f, 0.2, 0.31f, 0, 0.4, 1,
       0.5, 0, 0.4, 1, 0.5});

  AddInputFromArray<float>(TensorShape({2, 6, 2}),
                           {0.1f, 0.9f, 0.75f, 0.8f, 0.6f, 0.3f, 0.95f, 0.1f,
                            0.5f, 0.5f, 0.3f,  0.1f, 0.1f, 0.9f, 0.75f, 0.8f,
                            0.6f, 0.3f, 0.95f, 0.1f, 0.5f, 0.5f, 0.3f,  0.1f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  // boxes
  Tensor expected_boxes(allocator(), DT_FLOAT, TensorShape({2, 3, 4}));
  test::FillValues<float>(
      &expected_boxes,
      {0, 0.11, 0.1, 0.2, 0, 0, 0.1, 0.1, 0, 0.6f,  0.1, 0.7f,
       0, 0.21, 0.2, 0.3, 0, 0, 0.2, 0.2, 0, 0.02f, 0.2, 0.22f});
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
}
}