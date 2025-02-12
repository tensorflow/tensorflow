/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradient_checker.h"
#include "tensorflow/cc/framework/testutil.h"
#include "tensorflow/cc/gradients/grad_testutil.h"
#include "tensorflow/cc/ops/array_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace {

using namespace ops;  // NOLINT(build/namespaces)
using ops::internal::MirrorPadGrad;

class ArrayGradTest : public ::testing::Test {
 protected:
  ArrayGradTest() : scope_(Scope::NewRootScope()) {}

  void RunTest(const Output& x, const TensorShape& x_shape, const Output& y,
               const TensorShape& y_shape) {
    TF_ASSERT_OK(scope_.status());
    float max_error;
    TF_ASSERT_OK((ComputeGradientError<float, float, float>(
        scope_, {x}, {x_shape}, {y}, {y_shape}, &max_error)));
    EXPECT_LT(max_error, 1e-3);
  }

  void RunTest(const OutputList& xs, const std::vector<TensorShape>& x_shapes,
               const OutputList& ys, const std::vector<TensorShape>& y_shapes) {
    TF_ASSERT_OK(scope_.status());
    float max_error;
    TF_ASSERT_OK((ComputeGradientError<float, float, float>(
        scope_, xs, x_shapes, ys, y_shapes, &max_error)));
    EXPECT_LT(max_error, 1e-3);
  }

  Scope scope_;
};

TEST_F(ArrayGradTest, StackGrad_Axis0) {
  TensorShape x_shape({1, 2, 3});
  std::vector<Output> xs;
  xs.push_back(Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape)));
  xs.push_back(Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape)));
  auto y = Stack(scope_, xs, Stack::Axis(0));
  TensorShape y_shape({2, 1, 2, 3});
  RunTest(xs, {x_shape, x_shape}, {y}, {y_shape});
}

TEST_F(ArrayGradTest, StackGrad_Axis1) {
  TensorShape x_shape({1, 2, 3});
  std::vector<Output> xs;
  xs.push_back(Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape)));
  xs.push_back(Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape)));
  auto y = Stack(scope_, xs, Stack::Axis(1));
  TensorShape y_shape({1, 2, 2, 3});
  RunTest(xs, {x_shape, x_shape}, {y}, {y_shape});
}

TEST_F(ArrayGradTest, UnstackGrad_Axis0) {
  TensorShape x_shape({4, 2, 3});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  // Unstacking the first dimension results in 4 outputs.
  std::vector<TensorShape> y_shapes(4, TensorShape({2, 3}));
  auto y = Unstack(scope_, x, 4, Unstack::Axis(0));
  RunTest({x}, {x_shape}, y.output, y_shapes);
}

TEST_F(ArrayGradTest, UnstackGrad_Axis1) {
  TensorShape x_shape({4, 2, 3});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  // Unstacking the second dimension results in 2 outputs.
  std::vector<TensorShape> y_shapes(2, TensorShape({4, 3}));
  auto y = Unstack(scope_, x, 2, Unstack::Axis(1));
  RunTest({x}, {x_shape}, y.output, y_shapes);
}

TEST_F(ArrayGradTest, IdentityGrad) {
  TensorShape shape({5, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto y = Identity(scope_, x);
  RunTest(x, shape, y, shape);
}

TEST_F(ArrayGradTest, SplitGrad) {
  TensorShape x_shape({5, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  // Split along the second dimension.
  auto split_dim = Const(scope_, 1, {});
  auto y = Split(scope_, split_dim, x, /* num_split */ 2);
  TensorShape y_shape = TensorShape({5, 1});
  RunTest({x}, {x_shape}, y.output, {y_shape, y_shape});
}

TEST_F(ArrayGradTest, SplitVGrad) {
  TensorShape x_shape({2, 6});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto y = SplitV(scope_, x, {1, 2, 3}, /*axis=*/1, /*num_split=*/3);
  RunTest({x}, {x_shape}, y.output,
          {TensorShape({2, 1}), TensorShape({2, 2}), TensorShape({2, 3})});
}

TEST_F(ArrayGradTest, FillGrad) {
  TensorShape x_shape({});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  TensorShape y_shape({2, 5, 3});
  auto y = Fill(scope_, {2, 5, 3}, x);
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(ArrayGradTest, DiagGrad) {
  TensorShape x_shape({5, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto y = Diag(scope_, x);
  TensorShape y_shape({5, 2, 5, 2});
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(ArrayGradTest, DiagPartGrad) {
  TensorShape x_shape({5, 2, 5, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto y = DiagPart(scope_, x);
  TensorShape y_shape({5, 2});
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(ArrayGradTest, MatrixDiagGrad) {
  TensorShape x_shape({5, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto y = MatrixDiag(scope_, x);
  TensorShape y_shape({5, 2, 2});
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(ArrayGradTest, MatrixBandPartGrad) {
  TensorShape shape({5, 5});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  const int64_t num_lower = 1;
  const int64_t num_upper = 2;
  auto y = MatrixBandPart(scope_, x, num_lower, num_upper);
  RunTest(x, shape, y, shape);
}

TEST_F(ArrayGradTest, GatherNdGrad_SimpleIndexing) {
  TensorShape x_shape({2, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto indices = Const(scope_, {{0, 0}, {1, 1}});
  TensorShape y_shape({2});
  auto y = GatherNd(scope_, x, indices);
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(ArrayGradTest, GatherNdGrad_SliceIndexing) {
  TensorShape shape({2, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto indices = Const(scope_, {{1}, {0}});
  auto y = GatherNd(scope_, x, indices);
  RunTest(x, shape, y, shape);
}

TEST_F(ArrayGradTest, GatherNdGrad_SliceIndexing_Int64) {
  TensorShape shape({2, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto indices = Cast(scope_, Const(scope_, {{1}, {0}}), DT_INT64);
  auto y = GatherNd(scope_, x, indices);
  RunTest(x, shape, y, shape);
}

TEST_F(ArrayGradTest, CheckNumericsGrad) {
  TensorShape shape({5, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto y = CheckNumerics(scope_, x, "CheckNumerics failed");
  RunTest(x, shape, y, shape);
}

TEST_F(ArrayGradTest, ReshapeGrad) {
  TensorShape x_shape({5, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  TensorShape y_shape({2, 5});
  auto y = Reshape(scope_, x, {2, 5});
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(ArrayGradTest, ExpandDimsGrad) {
  TensorShape x_shape({5, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  TensorShape y_shape({1, 5, 2});
  auto y = ExpandDims(scope_, x, 0);
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(ArrayGradTest, SqueezeGrad) {
  TensorShape x_shape({1, 5, 1, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  TensorShape y_shape({5, 2});
  auto y = Squeeze(scope_, x);
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(ArrayGradTest, TransposeGrad) {
  TensorShape x_shape({5, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  TensorShape y_shape({2, 5});
  auto y = Transpose(scope_, x, {1, 0});
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(ArrayGradTest, ReverseSequenceGrad) {
  TensorShape shape({5, 2, 5});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto seq_lengths = Const(scope_, {1, 2, 3, 4, 5});
  // batch_dim defaults to 0.
  auto y = ReverseSequence(scope_, x, seq_lengths, /* seq_dim */ 2);
  RunTest(x, shape, y, shape);
}

TEST_F(ArrayGradTest, ReverseGrad) {
  TensorShape shape({5, 2, 5});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto y = Reverse(scope_, x, {0, 2});
  RunTest(x, shape, y, shape);
}

TEST_F(ArrayGradTest, ScatterNdGrad_SimpleIndexing) {
  TensorShape updates_shape({4});
  auto updates =
      Placeholder(scope_, DT_FLOAT, Placeholder::Shape(updates_shape));
  auto indices = Const(scope_, {{4}, {3}, {1}, {7}});
  TensorShape y_shape({8});
  auto y = ScatterNd(scope_, indices, updates, {8});
  RunTest(updates, updates_shape, y, y_shape);
}

TEST_F(ArrayGradTest, ScatterNdGrad_SliceIndexing) {
  TensorShape updates_shape({2, 4, 4});
  auto updates =
      Placeholder(scope_, DT_FLOAT, Placeholder::Shape(updates_shape));
  auto indices = Const(scope_, {{0}, {2}});
  TensorShape y_shape({4, 4, 4});
  auto y = ScatterNd(scope_, indices, updates, {4, 4, 4});
  RunTest(updates, updates_shape, y, y_shape);
}

TEST_F(ArrayGradTest, ScatterNdNonAliasingAddGrad_SimpleIndexing) {
  TensorShape updates_shape({4});
  TensorShape input_shape({8});
  auto input = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(input_shape));
  auto updates =
      Placeholder(scope_, DT_FLOAT, Placeholder::Shape(updates_shape));
  auto indices = Const(scope_, {{4}, {3}, {1}, {7}});
  auto y = ScatterNdNonAliasingAdd(scope_, input, indices, updates);
  RunTest({input, updates}, {input_shape, updates_shape}, {y}, {input_shape});
}

TEST_F(ArrayGradTest, ScatterNdNonAliasingAddGrad_SliceIndexing) {
  TensorShape updates_shape({2, 4, 4});
  TensorShape input_shape({4, 4, 4});
  auto input = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(input_shape));
  auto updates =
      Placeholder(scope_, DT_FLOAT, Placeholder::Shape(updates_shape));
  auto indices = Const(scope_, {{0}, {2}});
  auto y = ScatterNdNonAliasingAdd(scope_, input, indices, updates);
  RunTest({input, updates}, {input_shape, updates_shape}, {y}, {input_shape});
}

TEST_F(ArrayGradTest, PadGrad) {
  TensorShape x_shape({2, 3});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto paddings = Const(scope_, {{1, 1}, {2, 2}});
  TensorShape y_shape({4, 7});
  auto y = Pad(scope_, x, paddings);
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(ArrayGradTest, SpaceToBatchGrad) {
  TensorShape x_shape({1, 2, 2, 1});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto paddings = Const(scope_, {{1, 1}, {1, 1}});
  TensorShape y_shape({4, 2, 2, 1});
  auto y = SpaceToBatch(scope_, x, paddings, /* block_size */ 2);
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(ArrayGradTest, SpaceToBatchNdGrad) {
  TensorShape x_shape({2, 2, 4, 1});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto block_shape = Const(scope_, {2, 2});
  auto paddings = Const(scope_, {{0, 0}, {2, 0}});
  TensorShape y_shape({8, 1, 3, 1});
  auto y = SpaceToBatchND(scope_, x, block_shape, paddings);
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(ArrayGradTest, BatchToSpaceGrad) {
  TensorShape x_shape({4, 2, 2, 1});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto paddings = Const(scope_, {{1, 1}, {1, 1}});
  TensorShape y_shape({1, 2, 2, 1});
  auto y = BatchToSpace(scope_, x, paddings, /* block_size */ 2);
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(ArrayGradTest, BatchToSpaceNdGrad) {
  TensorShape x_shape({8, 1, 3, 1});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto block_shape = Const(scope_, {2, 2});
  auto paddings = Const(scope_, {{0, 0}, {2, 0}});
  TensorShape y_shape({2, 2, 4, 1});
  auto y = BatchToSpaceND(scope_, x, block_shape, paddings);
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(ArrayGradTest, SpaceToDepthGrad) {
  TensorShape x_shape({1, 2, 2, 1});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  TensorShape y_shape({1, 1, 1, 4});
  auto y = SpaceToDepth(scope_, x, /* block_size */ 2);
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(ArrayGradTest, DepthToSpaceGrad) {
  TensorShape x_shape({1, 1, 1, 4});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  TensorShape y_shape({1, 2, 2, 1});
  auto y = DepthToSpace(scope_, x, /* block_size */ 2);
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(ArrayGradTest, MirrorPadGrad_Reflect) {
  TensorShape x_shape({2, 3});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto paddings = Const(scope_, {{1, 1}, {2, 2}});
  TensorShape y_shape({4, 7});
  auto y = MirrorPad(scope_, x, paddings, "REFLECT");
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(ArrayGradTest, MirrorPadGrad_Symmetric) {
  TensorShape x_shape({2, 3});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto paddings = Const(scope_, {{1, 1}, {2, 2}});
  TensorShape y_shape({4, 7});
  auto y = MirrorPad(scope_, x, paddings, "SYMMETRIC");
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(ArrayGradTest, MirrorPadGradGrad_Reflect) {
  TensorShape x_shape({4, 7});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto paddings = Const(scope_, {{1, 1}, {2, 2}});
  TensorShape y_shape({2, 3});
  auto y = MirrorPadGrad(scope_, x, paddings, "REFLECT");
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(ArrayGradTest, MirrorPadGradGrad_Symmetric) {
  TensorShape x_shape({4, 7});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto paddings = Const(scope_, {{1, 1}, {2, 2}});
  TensorShape y_shape({2, 3});
  auto y = MirrorPadGrad(scope_, x, paddings, "SYMMETRIC");
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(ArrayGradTest, StridedSliceGrad) {
  TensorShape x_shape({6, 4, 4});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));

  // y = x[2:6:2, 1:3, 1:3]
  auto y = StridedSlice(scope_, x, {2, 1, 1}, {6, 3, 3}, {2, 1, 1});
  // y.shape = [2, 2, 2];
  RunTest(x, x_shape, y, {2, 2, 2});

  // y = x[2:6:2, 1:3, 1:3]
  // begin_mask = 1<<1 (ignore begin_index = 1)
  // end_mask = 1<<2 (ignore end_index = 2)
  y = StridedSlice(scope_, x, {2, 1, 1}, {6, 3, 3}, {2, 1, 1},
                   StridedSlice::BeginMask(1 << 1).EndMask(1 << 2));
  // y.shape = [2, 3, 3];
  RunTest(x, x_shape, y, {2, 3, 3});

  // y = [tf.newaxis, 2:6:2, 1:3, 1:3]
  y = StridedSlice(scope_, x, {0, 2, 1, 1}, {0, 6, 3, 3}, {1, 2, 1, 1},
                   StridedSlice::NewAxisMask(1 << 0));
  // y.shape = [1, 2, 2, 2];
  RunTest(x, x_shape, y, {1, 2, 2, 2});
}

TEST_F(ArrayGradTest, SliceGrad) {
  TensorShape x_shape({3, 5, 3});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto y = Slice(scope_, x, {1, 2, 1}, {1, 3, 2});
  RunTest(x, x_shape, y, {1, 3, 2});
}

TEST_F(ArrayGradTest, ConcatV2Grad) {
  TensorShape shape({3, 2, 5});
  std::vector<Output> xs;
  xs.push_back(Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape)));
  xs.push_back(Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape)));
  xs.push_back(Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape)));
  auto axis = Const(scope_, 0);
  auto y = Concat(scope_, xs, axis);
  TensorShape result_shape({9, 2, 5});
  RunTest(xs, {shape, shape, shape}, {y}, {result_shape});
}

TEST_F(ArrayGradTest, BroadcastToGrad) {
  TensorShape x_shape({2, 5});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  TensorShape y_shape({3, 2, 5});
  auto y = BroadcastTo(scope_, x, Const(scope_, {3, 2, 5}));
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(ArrayGradTest, TileGrad) {
  TensorShape x_shape({2, 5});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto y = Tile(scope_, x, Const(scope_, {3, 2}));
  TensorShape y_shape({6, 10});
  RunTest(x, x_shape, y, y_shape);
}

TEST_F(ArrayGradTest, GatherV2Grad_Simple) {
  TensorShape shape({100});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto y = GatherV2(scope_, x, {2, 0, 2, 5}, /*axis=*/0);
  TensorShape y_shape({4});
  RunTest(x, shape, y, y_shape);
}

TEST_F(ArrayGradTest, GatherV2Grad_MoreParamDims) {
  TensorShape shape({100, 2, 3, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto y = GatherV2(scope_, x, {2, 0, 2, 5}, /*axis=*/0);
  TensorShape y_shape({4, 2, 3, 2});
  RunTest(x, shape, y, y_shape);
}

TEST_F(ArrayGradTest, GatherV2Grad_MoreIndexDims) {
  TensorShape shape({100});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto y = GatherV2(scope_, x, {{2, 0}, {2, 5}}, /*axis=*/0);
  TensorShape y_shape({2, 2});
  RunTest(x, shape, y, y_shape);
}

TEST_F(ArrayGradTest, GatherV2Grad_DifferentAxis) {
  TensorShape shape({2, 10, 10, 2, 7});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto y = GatherV2(scope_, x, {2, 0, 2, 5, 5}, /*axis=*/1);
  TensorShape y_shape({2, 5, 10, 2, 7});
  RunTest(x, shape, y, y_shape);
}

TEST_F(ArrayGradTest, GatherV2Grad_DifferentAxis2) {
  TensorShape shape({2, 3, 100, 2, 7});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto y = GatherV2(scope_, x, {2, 0, 2, 5, 5}, /*axis=*/2);
  TensorShape y_shape({2, 3, 5, 2, 7});
  RunTest(x, shape, y, y_shape);
}

TEST_F(ArrayGradTest, GatherV2Grad_LastAxis) {
  TensorShape shape({2, 3, 10});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto y = GatherV2(scope_, x, {2, 0, 2, 5, 5}, /*axis=*/2);
  TensorShape y_shape({2, 3, 5});
  RunTest(x, shape, y, y_shape);
}

TEST_F(ArrayGradTest, GatherV2Grad_LastAxis2) {
  TensorShape shape({2, 3, 7, 10});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto y = GatherV2(scope_, x, {9, 8, 7, 6}, /*axis=*/3);
  TensorShape y_shape({2, 3, 7, 4});
  RunTest(x, shape, y, y_shape);
}

TEST_F(ArrayGradTest, GatherV2Grad_BatchDim) {
  TensorShape shape({2, 100, 3});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  GatherV2::Attrs attrs;
  attrs.batch_dims_ = 1;
  auto y =
      GatherV2(scope_, x, {{2, 0, 2, 5}, {1, 1, 7, 10}}, /*axis=*/1, attrs);
  TensorShape y_shape({2, 4, 3});
  RunTest(x, shape, y, y_shape);
}

TEST_F(ArrayGradTest, GatherV2Grad_BatchDim2) {
  TensorShape shape({2, 19});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  GatherV2::Attrs attrs;
  attrs.batch_dims_ = 1;
  auto y = GatherV2(scope_, x, {{0}, {0}}, /*axis=*/1, attrs);
  TensorShape y_shape({2, 1});
  RunTest(x, shape, y, y_shape);
}

TEST_F(ArrayGradTest, GatherV2Grad_BatchDimWithAxis) {
  TensorShape shape({2, 1, 3});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  GatherV2::Attrs attrs;
  attrs.batch_dims_ = 1;
  auto y = GatherV2(scope_, x, {{0}, {0}}, /*axis=*/2, attrs);
  TensorShape y_shape({2, 1, 1});
  RunTest(x, shape, y, y_shape);
}

TEST_F(ArrayGradTest, GatherV2Grad_TwoBatchDims) {
  TensorShape shape({2, 2, 100});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  GatherV2::Attrs attrs;
  attrs.batch_dims_ = 2;
  auto y = GatherV2(scope_, x, {{{2, 0}, {2, 5}}, {{1, 1}, {7, 10}}},
                    /*axis=*/2, attrs);
  TensorShape y_shape({2, 2, 2});
  RunTest(x, shape, y, y_shape);
}

TEST_F(ArrayGradTest, GatherV2Grad_TwoBatchDimsWithAxis) {
  TensorShape shape({2, 2, 3, 100});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  GatherV2::Attrs attrs;
  attrs.batch_dims_ = 2;
  auto y = GatherV2(scope_, x, {{{2, 0}, {2, 5}}, {{1, 1}, {7, 10}}},
                    /*axis=*/3, attrs);
  TensorShape y_shape({2, 2, 2, 3});
  RunTest(x, shape, y, y_shape);
}

}  // namespace
}  // namespace tensorflow
