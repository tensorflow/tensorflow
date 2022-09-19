/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"

#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/tsl/platform/status_matchers.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;
using ::tsl::testing::IsOkAndHolds;

TEST(GetNonContractingDimsTest, Valid) {
  Shape shape = ParseShape("f32[1,2,3,4,5,6]").ValueOrDie();
  EXPECT_THAT(GetNonContractingDims(shape, /*batch_dims=*/{4},
                                    /*contracting_dims=*/{1, 5}),
              IsOkAndHolds(ElementsAre(0, 2, 3)));
}

struct GetBatchRowColumnShapeTestParams {
  absl::string_view shape;
  std::vector<int64_t> batch_dims;
  std::vector<int64_t> row_dims;
  std::vector<int64_t> col_dims;
  absl::string_view expected_shape;
};

using GetBatchRowColumnShapeTest =
    ::testing::TestWithParam<GetBatchRowColumnShapeTestParams>;

TEST_P(GetBatchRowColumnShapeTest, ValidShape) {
  const GetBatchRowColumnShapeTestParams& params = GetParam();

  Shape shape = ParseShape(params.shape).ValueOrDie();
  EXPECT_THAT(GetBatchRowColumnShape(shape, params.batch_dims, params.row_dims,
                                     params.col_dims),
              IsOkAndHolds(ParseShape(params.expected_shape).ValueOrDie()));
}

INSTANTIATE_TEST_SUITE_P(
    GetBatchRowColumnShapeTests, GetBatchRowColumnShapeTest,
    ::testing::ValuesIn<GetBatchRowColumnShapeTestParams>({
        {"f32[3,4]{1,0}", /*batch_dims=*/{}, /*row_dims=*/{0}, /*col_dims=*/{1},
         "f32[1,3,4]{2,1,0}"},
        {"f32[3,4]{0,1}", {}, {0}, {1}, "f32[1,3,4]{1,2,0}"},
        {"f32[3,4]{1,0}", {}, {1}, {0}, "f32[1,4,3]{1,2,0}"},
        {"f32[3,4,5]{2,1,0}", {0}, {1}, {2}, "f32[3,4,5]{2,1,0}"},
        {"f32[3,4,5]{2,1,0}", {2}, {1}, {0}, "f32[5,4,3]{0,1,2}"},
        {"f32[3,4,5,6,7,8]{5,2,4,1,3,0}",
         {0, 3},
         {1, 4},
         {2, 5},
         "f32[18,28,40]{2,1,0}"},
    }));

TEST(GetBatchRowColumnShapeTest, BatchRowsColsInterleaved) {
  Shape shape = ParseShape("f32[3,4,5,6,7,8]{5,4,3,2,1,0}").ValueOrDie();
  auto result =
      GetBatchRowColumnShape(shape, /*batch_dims=*/{0, 3},
                             /*row_dims=*/{1, 4}, /*col_dims=*/{2, 5});
  EXPECT_FALSE(result.ok());
}

TEST(GetBatchRowColumnShapeTest, WrongPhysicalOrder) {
  Shape shape = ParseShape("f32[3,4,5,6]{3,2,0,1}").ValueOrDie();
  auto result = GetBatchRowColumnShape(shape, /*batch_dims=*/{0, 1},
                                       /*row_dims=*/{2}, /*col_dims=*/{3});
  EXPECT_FALSE(result.ok());
}

using Order = MatrixLayout::Order;

struct GetMatrixLayoutTestParams {
  absl::string_view shape;
  int64_t batch_size;
  int64_t num_rows;
  int64_t num_cols;
  Order order;
  int64_t leading_dim_stride;
  int64_t batch_stride;
};

using GetMatrixLayoutTest = ::testing::TestWithParam<GetMatrixLayoutTestParams>;

TEST_P(GetMatrixLayoutTest, ValidShape) {
  const GetMatrixLayoutTestParams& params = GetParam();

  Shape shape = ParseShape(params.shape).ValueOrDie();
  MatrixLayout result = MatrixLayout::For(shape).ValueOrDie();
  EXPECT_EQ(result.batch_size, params.batch_size);
  EXPECT_EQ(result.num_rows, params.num_rows);
  EXPECT_EQ(result.num_cols, params.num_cols);
  EXPECT_EQ(result.order, params.order);
  EXPECT_EQ(result.leading_dim_stride, params.leading_dim_stride);
  EXPECT_EQ(result.batch_stride, params.batch_stride);
}

INSTANTIATE_TEST_SUITE_P(
    GetMatrixLayoutTests, GetMatrixLayoutTest,
    ::testing::ValuesIn<GetMatrixLayoutTestParams>({
        {"f32[3,4,5]{2,1,0}", /*batch_size=*/3, /*num_rows=*/4, /*num_cols=*/5,
         /*order=*/Order::kRowMajor, /*leading_dim_stride=*/5,
         /*batch_stride=*/20},
        {"f32[3,4,5]{1,2,0}", 3, 4, 5, Order::kColumnMajor, 4, 20},
        {"f32[3,4,5]{2,0,1}", 3, 4, 5, Order::kRowMajor, 15, 5},
        {"f32[3,4,5]{1,0,2}", 3, 4, 5, Order::kColumnMajor, 12, 4},
    }));

TEST(GetMatrixLayoutTest, BatchInMostMinorPhysicalDimension) {
  Shape shape = ParseShape("f32[3,4,5]{0,2,1}").ValueOrDie();
  EXPECT_FALSE(MatrixLayout::For(shape).ok());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
