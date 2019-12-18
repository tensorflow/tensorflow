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

#include "tensorflow/core/util/matmul_bcast.h"

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

string MatMulBCastToStr(const MatMulBCast& b) {
  if (!b.IsValid()) {
    return "invalid";
  }
  string ret;
  strings::StrAppend(
      &ret, "[", absl::StrJoin(b.output_batch_shape().dim_sizes(), ","), "]");
  strings::StrAppend(&ret, "[", absl::StrJoin(b.x_batch_indices(), ","), "]");
  strings::StrAppend(&ret, "[", absl::StrJoin(b.y_batch_indices(), ","), "]");
  return ret;
}

TEST(MatMulBCastTest, SimpleBroadcast) {
  MatMulBCast bcast({1, 5, 3}, {4, 3, 7});

  EXPECT_TRUE(bcast.IsValid());
  EXPECT_TRUE(bcast.IsBroadcastingRequired());

  EXPECT_EQ(1, bcast.x_batch_size());
  EXPECT_EQ(4, bcast.y_batch_size());
  EXPECT_EQ(4, bcast.output_batch_size());

  EXPECT_EQ("[4][0,0,0,0][0,1,2,3]", MatMulBCastToStr(bcast));
}

TEST(MatMulBCastTest, EmptyBatchBroadcast) {
  MatMulBCast bcast({5, 3}, {3, 7});

  EXPECT_TRUE(bcast.IsValid());
  EXPECT_FALSE(bcast.IsBroadcastingRequired());

  EXPECT_EQ(1, bcast.x_batch_size());
  EXPECT_EQ(1, bcast.y_batch_size());
  EXPECT_EQ(1, bcast.output_batch_size());

  EXPECT_EQ("[][][]", MatMulBCastToStr(bcast));
}

TEST(MatMulBCastTest, BroadcastingNotRequired) {
  MatMulBCast bcast({2, 4, 6, 5, 3}, {2, 4, 6, 3, 7});

  EXPECT_TRUE(bcast.IsValid());
  EXPECT_FALSE(bcast.IsBroadcastingRequired());

  EXPECT_EQ(48, bcast.x_batch_size());
  EXPECT_EQ(48, bcast.y_batch_size());
  EXPECT_EQ(48, bcast.output_batch_size());

  EXPECT_EQ("[2,4,6][][]", MatMulBCastToStr(bcast));
}

TEST(MatMulBCastTest, EmptyWithNonEmptyBatchBroadcast) {
  MatMulBCast bcast1({5, 3}, {6, 3, 7});

  EXPECT_TRUE(bcast1.IsValid());
  EXPECT_TRUE(bcast1.IsBroadcastingRequired());

  EXPECT_EQ(1, bcast1.x_batch_size());
  EXPECT_EQ(6, bcast1.y_batch_size());
  EXPECT_EQ(6, bcast1.output_batch_size());
  EXPECT_EQ("[6][0,0,0,0,0,0][0,1,2,3,4,5]", MatMulBCastToStr(bcast1));

  MatMulBCast bcast2({2, 5, 3}, {3, 7});
  EXPECT_TRUE(bcast2.IsValid());
  EXPECT_TRUE(bcast2.IsBroadcastingRequired());

  EXPECT_EQ(2, bcast2.x_batch_size());
  EXPECT_EQ(1, bcast2.y_batch_size());
  EXPECT_EQ(2, bcast2.output_batch_size());
  EXPECT_EQ("[2][0,1][0,0]", MatMulBCastToStr(bcast2));
}

TEST(MatMulBCastTest, InvalidDimensions) {
  // Too few dimensions.
  MatMulBCast bcast1({3, 3}, {3});
  EXPECT_FALSE(bcast1.IsValid());

  MatMulBCast bcast2({3}, {3, 3});
  EXPECT_FALSE(bcast2.IsValid());

  // Batch dimensions not broadcastable.
  MatMulBCast bcast3({4, 5, 3}, {2, 3, 7});
  EXPECT_FALSE(bcast3.IsValid());

  MatMulBCast bcast4({2, 1, 5, 3}, {1, 3, 1, 3, 7});
  EXPECT_FALSE(bcast4.IsValid());
}

TEST(MatMulBCastTest, BroadcastBothOperands) {
  MatMulBCast bcast({3, 1, 5, 3}, {1, 4, 3, 7});
  EXPECT_TRUE(bcast.IsValid());

  EXPECT_EQ(3, bcast.x_batch_size());
  EXPECT_EQ(4, bcast.y_batch_size());
  EXPECT_EQ(12, bcast.output_batch_size());

  EXPECT_EQ("[3,4][0,0,0,0,1,1,1,1,2,2,2,2][0,1,2,3,0,1,2,3,0,1,2,3]",
            MatMulBCastToStr(bcast));
}

TEST(MatMulBCastTest, DifferentRanks) {
  MatMulBCast bcast({3, 1, 5, 3}, {2, 1, 2, 3, 7});
  EXPECT_TRUE(bcast.IsValid());

  EXPECT_EQ(3, bcast.x_batch_size());
  EXPECT_EQ(4, bcast.y_batch_size());
  EXPECT_EQ(12, bcast.output_batch_size());

  EXPECT_EQ("[2,3,2][0,0,1,1,2,2,0,0,1,1,2,2][0,1,0,1,0,1,2,3,2,3,2,3]",
            MatMulBCastToStr(bcast));
}

}  // namespace
}  // namespace tensorflow
