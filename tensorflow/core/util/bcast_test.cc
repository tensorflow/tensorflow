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

#include "tensorflow/core/util/bcast.h"

#include <cstdint>

#include "absl/strings/str_join.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

string BCast(const tensorflow::BCast::Vec& x, const tensorflow::BCast::Vec& y,
             const bool fewer_dims_optimization = true) {
  tensorflow::BCast b(x, y, fewer_dims_optimization);
  if (!b.IsValid()) {
    return "invalid";
  }
  string ret;
  strings::StrAppend(&ret, "[", absl::StrJoin(b.x_reshape(), ","), "]");
  strings::StrAppend(&ret, "[", absl::StrJoin(b.x_bcast(), ","), "]");
  strings::StrAppend(&ret, "[", absl::StrJoin(b.y_reshape(), ","), "]");
  strings::StrAppend(&ret, "[", absl::StrJoin(b.y_bcast(), ","), "]");
  strings::StrAppend(&ret, "[", absl::StrJoin(b.result_shape(), ","), "]");
  strings::StrAppend(&ret, "[", absl::StrJoin(b.output_shape(), ","), "]");
  strings::StrAppend(&ret, "[", absl::StrJoin(b.grad_x_reduce_idx(), ","), "]");
  strings::StrAppend(&ret, "[", absl::StrJoin(b.grad_y_reduce_idx(), ","), "]");
  return ret;
}

string BCastBatchIndices(const tensorflow::BCast::Vec& x,
                         const tensorflow::BCast::Vec& y,
                         const bool fewer_dims_optimization = true) {
  tensorflow::BCast b(x, y, fewer_dims_optimization,
                      /*return_flattened_batch_indices=*/true);
  string ret;
  strings::StrAppend(&ret, "[", absl::StrJoin(b.x_batch_indices(), ","), "]");
  strings::StrAppend(&ret, "[", absl::StrJoin(b.y_batch_indices(), ","), "]");
  return ret;
}

string BCastList3(const tensorflow::BCast::Vec& x,
                  const tensorflow::BCast::Vec& y,
                  const tensorflow::BCast::Vec& z,
                  const bool fewer_dims_optimization = true) {
  tensorflow::BCastList<3> b({x, y, z}, fewer_dims_optimization);
  if (!b.IsValid()) {
    return "invalid";
  }
  string ret;
  strings::StrAppend(&ret, "[", absl::StrJoin(b.reshape(0), ","), "]");
  strings::StrAppend(&ret, "[", absl::StrJoin(b.bcast(0), ","), "]");
  strings::StrAppend(&ret, "[", absl::StrJoin(b.reshape(1), ","), "]");
  strings::StrAppend(&ret, "[", absl::StrJoin(b.bcast(1), ","), "]");
  strings::StrAppend(&ret, "[", absl::StrJoin(b.reshape(2), ","), "]");
  strings::StrAppend(&ret, "[", absl::StrJoin(b.bcast(2), ","), "]");
  strings::StrAppend(&ret, "[", absl::StrJoin(b.result_shape(), ","), "]");
  strings::StrAppend(&ret, "[", absl::StrJoin(b.output_shape(), ","), "]");
  strings::StrAppend(&ret, "[", absl::StrJoin(b.grad_reduce_idx(0), ","), "]");
  strings::StrAppend(&ret, "[", absl::StrJoin(b.grad_reduce_idx(1), ","), "]");
  strings::StrAppend(&ret, "[", absl::StrJoin(b.grad_reduce_idx(2), ","), "]");
  return ret;
}

TEST(BCastTest, Invalid) {
  for (const bool use_optimization : {true, false}) {
    EXPECT_EQ("invalid", BCast({5, 3, 2}, {3}, use_optimization));
    EXPECT_EQ("invalid", BCast({5, 3, 2}, {2, 2}, use_optimization));
    EXPECT_EQ("invalid", BCast({5, 3, 2}, {10, 1, 1}, use_optimization));
    EXPECT_EQ("invalid",
              BCast({1, 2, 1, 2, 1, 2}, {2, 4, 2, 1, 2, 1}, use_optimization));
  }
}

TEST(BCastListTest, Invalid) {
  for (const bool use_optimization : {true, false}) {
    EXPECT_EQ("invalid", BCastList3({5, 3, 2}, {3}, {1}, use_optimization));
    EXPECT_EQ("invalid", BCastList3({5, 3, 2}, {2, 2}, {1}, use_optimization));
    EXPECT_EQ("invalid",
              BCastList3({5, 3, 2}, {10, 1, 1}, {1}, use_optimization));
    EXPECT_EQ("invalid", BCastList3({1, 2, 1, 2, 1, 2}, {2, 4, 2, 1, 2, 1}, {1},
                                    use_optimization));
    EXPECT_EQ("invalid", BCastList3({5, 3, 2}, {1}, {3}, use_optimization));
    EXPECT_EQ("invalid", BCastList3({5, 3, 2}, {1}, {2, 2}, use_optimization));
    EXPECT_EQ("invalid",
              BCastList3({5, 3, 2}, {1}, {10, 1, 1}, use_optimization));

    EXPECT_EQ("invalid", BCastList3({1}, {5, 3, 2}, {3}, use_optimization));
    EXPECT_EQ("invalid", BCastList3({1}, {5, 3, 2}, {2, 2}, use_optimization));
    EXPECT_EQ("invalid",
              BCastList3({1}, {5, 3, 2}, {10, 1, 1}, use_optimization));
  }
}

TEST(BCastTest, Basic_SameShape) {
  // Effectively no broadcast needed.
  EXPECT_EQ(BCast({11, 7, 5, 3, 2}, {11, 7, 5, 3, 2}),
            "[2310][1][2310][1]"
            "[2310]"
            "[11,7,5,3,2]"
            "[][]");

  EXPECT_EQ(BCast({11, 7, 5, 3, 2}, {11, 7, 5, 3, 2}, false),
            "[11,7,5,3,2][1,1,1,1,1][11,7,5,3,2][1,1,1,1,1]"
            "[11,7,5,3,2]"
            "[11,7,5,3,2]"
            "[][]");
}

TEST(BCastListTest, Basic_SameShape) {
  // Effectively no broadcast needed.
  EXPECT_EQ(BCastList3({11, 7, 5, 3, 2}, {11, 7, 5, 3, 2}, {11, 7, 5, 3, 2}),
            "[2310][1][2310][1][2310][1]"
            "[2310]"
            "[11,7,5,3,2]"
            "[][][]");

  EXPECT_EQ(
      BCastList3({11, 7, 5, 3, 2}, {11, 7, 5, 3, 2}, {11, 7, 5, 3, 2}, false),
      "[11,7,5,3,2][1,1,1,1,1][11,7,5,3,2][1,1,1,1,1][11,7,5,3,2][1,1,1,1,1]"
      "[11,7,5,3,2]"
      "[11,7,5,3,2]"
      "[][][]");
}

TEST(BCastTest, Basic_SameShapeWithZeroDim) {
  // Effectively no broadcast needed.
  EXPECT_EQ(BCast({11, 7, 0, 3, 2}, {11, 7, 0, 3, 2}),
            "[0][1][0][1]"
            "[0]"
            "[11,7,0,3,2]"
            "[][]");

  EXPECT_EQ(BCast({11, 7, 0, 3, 2}, {11, 7, 0, 3, 2}, false),
            "[11,7,0,3,2][1,1,1,1,1][11,7,0,3,2][1,1,1,1,1]"
            "[11,7,0,3,2]"
            "[11,7,0,3,2]"
            "[][]");
}

TEST(BCastListTest, Basic_SameShapeWithZeroDim) {
  // Effectively no broadcast needed.
  EXPECT_EQ(BCastList3({11, 7, 0, 3, 2}, {11, 7, 0, 3, 2}, {11, 7, 0, 3, 2}),
            "[0][1][0][1][0][1]"
            "[0]"
            "[11,7,0,3,2]"
            "[][][]");

  EXPECT_EQ(
      BCastList3({11, 7, 0, 3, 2}, {11, 7, 0, 3, 2}, {11, 7, 0, 3, 2}, false),
      "[11,7,0,3,2][1,1,1,1,1][11,7,0,3,2][1,1,1,1,1][11,7,0,3,2][1,1,1,1,1]"
      "[11,7,0,3,2]"
      "[11,7,0,3,2]"
      "[][][]");
}

TEST(BCastTest, Basic_Scalar_Scalar) {
  // Effectively it's a scalar and a scalar.
  // [1, 1] [1]
  //
  EXPECT_EQ(BCast({1, 1}, {}),
            "[1][1][1][1]"
            "[1]"
            "[1,1]"
            "[0,1][0,1]");

  EXPECT_EQ(BCast({1, 1}, {1}),
            "[1][1][1][1]"
            "[1]"
            "[1,1]"
            "[0,1][0,1]");

  EXPECT_EQ(BCast({1, 1}, {1}, false),
            "[1,1][1,1][1,1][1,1]"
            "[1,1]"
            "[1,1]"
            "[0,1][0,1]");

  // [1] [1, 1]
  EXPECT_EQ(BCast({1}, {1, 1}),
            "[1][1][1][1]"
            "[1]"
            "[1,1]"
            "[0,1][0,1]");

  EXPECT_EQ(BCast({1}, {1, 1}, false),
            "[1,1][1,1][1,1][1,1]"
            "[1,1]"
            "[1,1]"
            "[0,1][0,1]");
}

TEST(BCastTest, Basic_TrueScalar_Scalar) {
  // [] []
  EXPECT_EQ(BCast({}, {}),
            "[1][1][1][1]"
            "[1]"
            "[]"
            "[][]");

  // [] [1]
  EXPECT_EQ(BCast({}, {1}),
            "[1][1][1][1]"
            "[1]"
            "[1]"
            "[0][0]");

  EXPECT_EQ(BCast({}, {1}, false),
            "[1][1][1][1]"
            "[1]"
            "[1]"
            "[0][0]");

  // [] [1, 1]
  EXPECT_EQ(BCast({}, {1, 1}),
            "[1][1][1][1]"
            "[1]"
            "[1,1]"
            "[0,1][0,1]");

  EXPECT_EQ(BCast({}, {1, 1}, false),
            "[1,1][1,1][1,1][1,1]"
            "[1,1]"
            "[1,1]"
            "[0,1][0,1]");

  // [1] []
  EXPECT_EQ(BCast({1}, {}),
            "[1][1][1][1]"
            "[1]"
            "[1]"
            "[0][0]");

  EXPECT_EQ(BCast({1}, {}, false),
            "[1][1][1][1]"
            "[1]"
            "[1]"
            "[0][0]");

  // [1, 1] []
  EXPECT_EQ(BCast({1, 1}, {}),
            "[1][1][1][1]"
            "[1]"
            "[1,1]"
            "[0,1][0,1]");

  EXPECT_EQ(BCast({1, 1}, {}, false),
            "[1,1][1,1][1,1][1,1]"
            "[1,1]"
            "[1,1]"
            "[0,1][0,1]");
}

TEST(BCastListTest, Basic_Scalar_Scalar_Scalar) {
  // Effectively it's a scalar and a scalar.
  // [1, 1] [1] [1]
  EXPECT_EQ(BCastList3({1, 1}, {1}, {1}),
            "[1][1][1][1][1][1]"
            "[1]"
            "[1,1]"
            "[0,1][0,1][0,1]");

  EXPECT_EQ(BCastList3({1, 1}, {1}, {1}, false),
            "[1,1][1,1][1,1][1,1][1,1][1,1]"
            "[1,1]"
            "[1,1]"
            "[0,1][0,1][0,1]");

  // [1] [1, 1] [1]
  EXPECT_EQ(BCastList3({1}, {1, 1}, {1}),
            "[1][1][1][1][1][1]"
            "[1]"
            "[1,1]"
            "[0,1][0,1][0,1]");

  EXPECT_EQ(BCastList3({1}, {1, 1}, {1}, false),
            "[1,1][1,1][1,1][1,1][1,1][1,1]"
            "[1,1]"
            "[1,1]"
            "[0,1][0,1][0,1]");

  // [1] [1] [1, 1]
  EXPECT_EQ(BCastList3({1}, {1}, {1, 1}),
            "[1][1][1][1][1][1]"
            "[1]"
            "[1,1]"
            "[0,1][0,1][0,1]");

  EXPECT_EQ(BCastList3({1}, {1}, {1, 1}, false),
            "[1,1][1,1][1,1][1,1][1,1][1,1]"
            "[1,1]"
            "[1,1]"
            "[0,1][0,1][0,1]");
}

TEST(BCastListTest, Basic_TrueScalar_Scalar_Scalar) {
  // Effectively it's a scalar and a scalar.
  // [1, 1] [1] []
  EXPECT_EQ(BCastList3({1, 1}, {1}, {}),
            "[1][1][1][1][1][1]"
            "[1]"
            "[1,1]"
            "[0,1][0,1][0,1]");

  EXPECT_EQ(BCastList3({1, 1}, {1}, {}, false),
            "[1,1][1,1][1,1][1,1][1,1][1,1]"
            "[1,1]"
            "[1,1]"
            "[0,1][0,1][0,1]");

  // [] [1, 1] [1]
  EXPECT_EQ(BCastList3({}, {1, 1}, {1}),
            "[1][1][1][1][1][1]"
            "[1]"
            "[1,1]"
            "[0,1][0,1][0,1]");

  EXPECT_EQ(BCastList3({}, {1, 1}, {1}, false),
            "[1,1][1,1][1,1][1,1][1,1][1,1]"
            "[1,1]"
            "[1,1]"
            "[0,1][0,1][0,1]");

  // [1] [] [1, 1]
  EXPECT_EQ(BCastList3({1}, {}, {1, 1}),
            "[1][1][1][1][1][1]"
            "[1]"
            "[1,1]"
            "[0,1][0,1][0,1]");

  EXPECT_EQ(BCastList3({1}, {}, {1, 1}, false),
            "[1,1][1,1][1,1][1,1][1,1][1,1]"
            "[1,1]"
            "[1,1]"
            "[0,1][0,1][0,1]");
}

TEST(BCastTest, Basic_Tensor_Scalar) {
  // Effectively it's a tensor and a scalar.
  // [11, 7, 5, 3, 2] [1]
  EXPECT_EQ(BCast({11, 7, 5, 3, 2}, {1}),
            "[2310][1][1][2310]"
            "[2310]"
            "[11,7,5,3,2]"
            "[][0,1,2,3,4]");

  EXPECT_EQ(BCast({11, 7, 5, 3, 2}, {1}, false),
            "[11,7,5,3,2][1,1,1,1,1][1,1,1,1,1][11,7,5,3,2]"
            "[11,7,5,3,2]"
            "[11,7,5,3,2]"
            "[][0,1,2,3,4]");

  // [1] [11, 7, 5, 3, 2]
  EXPECT_EQ(BCast({1}, {11, 7, 5, 3, 2}),
            "[1][2310][2310][1]"
            "[2310]"
            "[11,7,5,3,2]"
            "[0,1,2,3,4][]");

  EXPECT_EQ(BCast({1}, {11, 7, 5, 3, 2}, false),
            "[1,1,1,1,1][11,7,5,3,2][11,7,5,3,2][1,1,1,1,1]"
            "[11,7,5,3,2]"
            "[11,7,5,3,2]"
            "[0,1,2,3,4][]");

  // int32 edge-case:
  EXPECT_EQ(BCast({1, 2147483648}, {1}),
            "[2147483648][1][1][2147483648]"
            "[2147483648]"
            "[1,2147483648]"
            "[0][0,1]");
}

TEST(BCastTest, Basic_Tensor_With_DimSize_1_Scalar) {
  // Effectively it's a tensor and a scalar.
  // [11, 7, 5, 3, 2, 1] [1]
  EXPECT_EQ(BCast({11, 7, 5, 3, 2, 1}, {1}),
            "[2310][1][1][2310]"
            "[2310]"
            "[11,7,5,3,2,1]"
            "[5][0,1,2,3,4,5]");

  EXPECT_EQ(BCast({11, 7, 5, 3, 2, 1}, {1}, false),
            "[11,7,5,3,2,1][1,1,1,1,1,1][1,1,1,1,1,1][11,7,5,3,2,1]"
            "[11,7,5,3,2,1]"
            "[11,7,5,3,2,1]"
            "[5][0,1,2,3,4,5]");

  // [1] [11, 7, 5, 3, 2, 1]
  EXPECT_EQ(BCast({1}, {11, 7, 5, 3, 2, 1}),
            "[1][2310][2310][1]"
            "[2310]"
            "[11,7,5,3,2,1]"
            "[0,1,2,3,4,5][5]");

  EXPECT_EQ(BCast({1}, {11, 7, 5, 3, 2, 1}, false),
            "[1,1,1,1,1,1][11,7,5,3,2,1][11,7,5,3,2,1][1,1,1,1,1,1]"
            "[11,7,5,3,2,1]"
            "[11,7,5,3,2,1]"
            "[0,1,2,3,4,5][5]");

  // Effectively it's a tensor and a scalar.
  // [11, 7, 5, 1, 1, 3, 2, 1] [1]
  EXPECT_EQ(BCast({11, 7, 5, 1, 1, 3, 2, 1, 1}, {1}),
            "[2310][1][1][2310]"
            "[2310]"
            "[11,7,5,1,1,3,2,1,1]"
            "[3,4,7,8][0,1,2,3,4,5,6,7,8]");

  EXPECT_EQ(BCast({11, 7, 5, 1, 1, 3, 2, 1, 1}, {1}, false),
            "[11,7,5,1,1,3,2,1,1][1,1,1,1,1,1,1,1,1]"  // x_reshape(), x_bcast()
            "[1,1,1,1,1,1,1,1,1][11,7,5,1,1,3,2,1,1]"  // y_reshape(), y_bcast()
            "[11,7,5,1,1,3,2,1,1]"
            "[11,7,5,1,1,3,2,1,1]"
            "[3,4,7,8][0,1,2,3,4,5,6,7,8]");

  // [1] [11, 7, 5, 1, 1, 3, 2, 1]
  EXPECT_EQ(BCast({1}, {11, 7, 5, 1, 1, 3, 2, 1, 1}),
            "[1][2310][2310][1]"
            "[2310]"
            "[11,7,5,1,1,3,2,1,1]"
            "[0,1,2,3,4,5,6,7,8][3,4,7,8]");

  EXPECT_EQ(BCast({1}, {11, 7, 5, 1, 1, 3, 2, 1, 1}, false),
            "[1,1,1,1,1,1,1,1,1][11,7,5,1,1,3,2,1,1]"  // x_reshape(), x_bcast()
            "[11,7,5,1,1,3,2,1,1][1,1,1,1,1,1,1,1,1]"  // y_reshape(), y_bcast()
            "[11,7,5,1,1,3,2,1,1]"
            "[11,7,5,1,1,3,2,1,1]"
            "[0,1,2,3,4,5,6,7,8][3,4,7,8]");
}

TEST(BCastTest, Basic_Tensor_Vector) {
  // [11, 7, 5, 3, 2] [2]
  EXPECT_EQ(BCast({11, 7, 5, 3, 2}, {2}),
            "[1155,2][1,1][1,2][1155,1]"
            "[1155,2]"
            "[11,7,5,3,2]"
            "[][0,1,2,3]");

  EXPECT_EQ(BCast({11, 7, 5, 3, 2}, {2}, false),
            "[11,7,5,3,2][1,1,1,1,1][1,1,1,1,2][11,7,5,3,1]"
            "[11,7,5,3,2]"
            "[11,7,5,3,2]"
            "[][0,1,2,3]");

  // [2] [11, 7, 5, 3, 2]
  EXPECT_EQ(BCast({2}, {11, 7, 5, 3, 2}),
            "[1,2][1155,1][1155,2][1,1]"
            "[1155,2]"
            "[11,7,5,3,2]"
            "[0,1,2,3][]");

  EXPECT_EQ(BCast({2}, {11, 7, 5, 3, 2}, false),
            "[1,1,1,1,2][11,7,5,3,1][11,7,5,3,2][1,1,1,1,1]"
            "[11,7,5,3,2]"
            "[11,7,5,3,2]"
            "[0,1,2,3][]");
}

TEST(BCastTest, Basic_Tensor_Matrix) {
  // [11, 7, 5, 3, 2] [3, 2]
  EXPECT_EQ(BCast({11, 7, 5, 3, 2}, {3, 2}),
            "[385,6][1,1][1,6][385,1]"
            "[385,6]"
            "[11,7,5,3,2]"
            "[][0,1,2]");

  EXPECT_EQ(BCast({11, 7, 5, 3, 2}, {3, 2}, false),
            "[11,7,5,3,2][1,1,1,1,1][1,1,1,3,2][11,7,5,1,1]"
            "[11,7,5,3,2]"
            "[11,7,5,3,2]"
            "[][0,1,2]");

  // [3, 2] [11, 7, 5, 3, 2]
  EXPECT_EQ(BCast({3, 2}, {11, 7, 5, 3, 2}),
            "[1,6][385,1][385,6][1,1]"
            "[385,6]"
            "[11,7,5,3,2]"
            "[0,1,2][]");

  EXPECT_EQ(BCast({3, 2}, {11, 7, 5, 3, 2}, false),
            "[1,1,1,3,2][11,7,5,1,1][11,7,5,3,2][1,1,1,1,1]"
            "[11,7,5,3,2]"
            "[11,7,5,3,2]"
            "[0,1,2][]");
}

TEST(BCastTest, Basic_Tensor_Matrix_Column) {
  // [11, 7, 5, 3, 2] [3, 1]
  EXPECT_EQ(BCast({11, 7, 5, 3, 2}, {3, 1}),
            "[385,3,2][1,1,1][1,3,1][385,1,2]"
            "[385,3,2]"
            "[11,7,5,3,2]"
            "[][0,1,2,4]");

  EXPECT_EQ(BCast({11, 7, 5, 3, 2}, {3, 1}, false),
            "[11,7,5,3,2][1,1,1,1,1][1,1,1,3,1][11,7,5,1,2]"
            "[11,7,5,3,2]"
            "[11,7,5,3,2]"
            "[][0,1,2,4]");

  // [3, 1] [11, 7, 5, 3, 2]
  EXPECT_EQ(BCast({3, 1}, {11, 7, 5, 3, 2}),
            "[1,3,1][385,1,2][385,3,2][1,1,1]"
            "[385,3,2]"
            "[11,7,5,3,2]"
            "[0,1,2,4][]");

  EXPECT_EQ(BCast({3, 1}, {11, 7, 5, 3, 2}, false),
            "[1,1,1,3,1][11,7,5,1,2][11,7,5,3,2][1,1,1,1,1]"
            "[11,7,5,3,2]"
            "[11,7,5,3,2]"
            "[0,1,2,4][]");
}

TEST(BCastTest, Basic_Tensor_Matrix_As_Tensor) {
  // [11, 7, 5, 3, 2] [7, 5, 1, 1]
  EXPECT_EQ(BCast({11, 7, 5, 3, 2}, {7, 5, 1, 1}),
            "[11,35,6][1,1,1][1,35,1][11,1,6]"
            "[11,35,6]"
            "[11,7,5,3,2]"
            "[][0,3,4]");

  EXPECT_EQ(BCast({11, 7, 5, 3, 2}, {7, 5, 1, 1}, false),
            "[11,7,5,3,2][1,1,1,1,1][1,7,5,1,1][11,1,1,3,2]"
            "[11,7,5,3,2]"
            "[11,7,5,3,2]"
            "[][0,3,4]");

  // [7, 5, 1, 1] [11, 7, 5, 3, 2]
  EXPECT_EQ(BCast({7, 5, 1, 1}, {11, 7, 5, 3, 2}),
            "[1,35,1][11,1,6][11,35,6][1,1,1]"
            "[11,35,6]"
            "[11,7,5,3,2]"
            "[0,3,4][]");

  EXPECT_EQ(BCast({7, 5, 1, 1}, {11, 7, 5, 3, 2}, false),
            "[1,7,5,1,1][11,1,1,3,2][11,7,5,3,2][1,1,1,1,1]"
            "[11,7,5,3,2][11,7,5,3,2]"
            "[0,3,4][]");
}

TEST(BCastTest, Basic_SymbolicShape) {
  constexpr int64_t kSymDim1 = -10'000'000'000;
  constexpr int64_t kSymDim2 = -10'000'000'001;

  const tensorflow::BCast bcast({10, kSymDim1, kSymDim2}, {10, 1, 1}, false);
  EXPECT_TRUE(bcast.IsValid());
  EXPECT_EQ(bcast.output_batch_size(), -1);
}

TEST(BCastTest, Complex_BCast_To_Each_Other) {
  // Rare cases. x and y broadcast to each other.  x and y are of
  // different ranks.
  // Can be verified in numpy as:
  //   import numpy as np
  //   x = np.arange(0,110).reshape([11,1,5,1,2])
  //   y = np.arange(0,21).reshape([7,1,3,1])
  //   np.shape(x + y)
  //   Out[.]: (11, 7, 5, 3, 2)
  string truth =
      "[11,1,5,1,2][1,7,1,3,1][1,7,1,3,1][11,1,5,1,2]"
      "[11,7,5,3,2]"
      "[11,7,5,3,2]"
      "[1,3][0,2,4]";

  EXPECT_EQ(BCast({11, 1, 5, 1, 2}, {7, 1, 3, 1}), truth);
  EXPECT_EQ(BCast({11, 1, 5, 1, 2}, {7, 1, 3, 1}, false), truth);
}

TEST(BCastListTest, Complex_BCast_To_Each_Other) {
  // Rare cases. x, y and z broadcast to each other. x,y and z are of
  // different ranks.
  // Can be verified in numpy as:
  //   import numpy as np
  //   x = np.arange(0,22).reshape([11,1,1,1,2])
  //   y = np.arange(0,21).reshape([7,1,3,1])
  //   z = np.arange(0,5).reshape([5,1,1])
  //   np.shape(x + y + z)
  //   Out[.]: (11, 7, 5, 3, 2)
  //
  string truth =
      "[11,1,1,1,2][1,7,5,3,1]"
      "[1,7,1,3,1][11,1,5,1,2]"
      "[1,1,5,1,1][11,7,1,3,2]"
      "[11,7,5,3,2]"
      "[11,7,5,3,2]"
      "[1,2,3][0,2,4][0,1,3,4]";

  EXPECT_EQ(BCastList3({11, 1, 1, 1, 2}, {7, 1, 3, 1}, {5, 1, 1}), truth);
  EXPECT_EQ(BCastList3({11, 1, 1, 1, 2}, {7, 1, 3, 1}, {5, 1, 1}, false),
            truth);
}

TEST(BCastTest, TestZeroDimensionShape) {
  // (2,0,5) and (5) in both orders
  EXPECT_EQ(BCast({2, 0, 5}, {5}),
            "[0,5][1,1][1,5][0,1]"
            "[0,5]"
            "[2,0,5]"
            "[][0,1]");
  EXPECT_EQ(BCast({5}, {2, 0, 5}),
            "[1,5][0,1][0,5][1,1]"
            "[0,5]"
            "[2,0,5]"
            "[0,1][]");

  EXPECT_EQ(BCast({2, 0, 5}, {5}, false),
            "[2,0,5][1,1,1][1,1,5][2,0,1]"
            "[2,0,5]"
            "[2,0,5]"
            "[][0,1]");
  EXPECT_EQ(BCast({5}, {2, 0, 5}, false),
            "[1,1,5][2,0,1][2,0,5][1,1,1]"
            "[2,0,5]"
            "[2,0,5]"
            "[0,1][]");

  // (2,0,3,0,5) and (5) in both orders
  EXPECT_EQ(BCast({2, 0, 3, 0, 5}, {5}),
            "[0,5][1,1][1,5][0,1]"
            "[0,5]"
            "[2,0,3,0,5]"
            "[][0,1,2,3]");
  EXPECT_EQ(BCast({5}, {2, 0, 3, 0, 5}),
            "[1,5][0,1][0,5][1,1]"
            "[0,5]"
            "[2,0,3,0,5]"
            "[0,1,2,3][]");

  EXPECT_EQ(BCast({2, 0, 3, 0, 5}, {5}, false),
            "[2,0,3,0,5][1,1,1,1,1][1,1,1,1,5][2,0,3,0,1]"
            "[2,0,3,0,5]"
            "[2,0,3,0,5]"
            "[][0,1,2,3]");
  EXPECT_EQ(BCast({5}, {2, 0, 3, 0, 5}, false),
            "[1,1,1,1,5][2,0,3,0,1][2,0,3,0,5][1,1,1,1,1]"
            "[2,0,3,0,5]"
            "[2,0,3,0,5]"
            "[0,1,2,3][]");

  // (2,0,3,0,5) and (3,1,5) in both orders
  EXPECT_EQ(BCast({2, 0, 3, 0, 5}, {3, 1, 5}),
            "[0,3,0,5][1,1,1,1][1,3,1,5][0,1,0,1]"
            "[0,3,0,5]"
            "[2,0,3,0,5]"
            "[][0,1,3]");
  EXPECT_EQ(BCast({3, 1, 5}, {2, 0, 3, 0, 5}),
            "[1,3,1,5][0,1,0,1][0,3,0,5][1,1,1,1]"
            "[0,3,0,5]"
            "[2,0,3,0,5]"
            "[0,1,3][]");

  EXPECT_EQ(BCast({2, 0, 3, 0, 5}, {3, 1, 5}, false),
            "[2,0,3,0,5][1,1,1,1,1][1,1,3,1,5][2,0,1,0,1]"
            "[2,0,3,0,5]"
            "[2,0,3,0,5]"
            "[][0,1,3]");
  EXPECT_EQ(BCast({3, 1, 5}, {2, 0, 3, 0, 5}, false),
            "[1,1,3,1,5][2,0,1,0,1][2,0,3,0,5][1,1,1,1,1]"
            "[2,0,3,0,5]"
            "[2,0,3,0,5]"
            "[0,1,3][]");
}

TEST(BCastTest, BatchIndices) {
  EXPECT_EQ("[0,0,0,0][0,1,2,3]", BCastBatchIndices({1}, {4}));
  // Invalid broadcast.
  EXPECT_EQ("[][]", BCastBatchIndices({5}, {7}));
  // Same shape, no batch indices.
  EXPECT_EQ("[][]", BCastBatchIndices({2, 4, 6}, {2, 4, 6}));
  // More complicated broadcasts.
  EXPECT_EQ("[0,0,0,0,1,1,1,1,2,2,2,2][0,1,2,3,0,1,2,3,0,1,2,3]",
            BCastBatchIndices({3, 1}, {1, 4}));
  EXPECT_EQ("[0,0,1,1,2,2,0,0,1,1,2,2][0,1,0,1,0,1,2,3,2,3,2,3]",
            BCastBatchIndices({3, 1}, {2, 1, 2}));
}

void BM_BCastSetup(::testing::benchmark::State& state) {
  const int same_shape = state.range(0);

  if (same_shape) {
    state.SetLabel("same_shapes");
    for (auto s : state) {
      class BCast b({1000, 100}, {1000, 100});
    }
  } else {
    state.SetLabel("different_shapes");
    for (auto s : state) {
      class BCast b({3, 1, 5}, {2, 0, 3, 0, 5});
    }
  }
}
BENCHMARK(BM_BCastSetup)->Arg(0)->Arg(1);

}  // namespace
}  // namespace tensorflow
