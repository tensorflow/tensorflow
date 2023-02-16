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

#include "tensorflow/core/kernels/ragged_tensor_to_variant_op_test.h"

#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ragged_tensor_variant.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/tsl/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace {

TEST_F(RaggedTensorToVariantKernelTest, NoValuesInput) {
  // ragged_tensor=[[[], []], [[]], []]
  const std::vector<int64_t> batched_splits_1 = {0, 2, 3, 3};
  const std::vector<int64_t> batched_splits_2 = {0, 0, 0, 0};

  BuildEncodeRaggedTensorGraph<int, int64_t>(
      {batched_splits_1, batched_splits_2}, TensorShape({0}), {}, true);
  TF_ASSERT_OK(RunOpKernel());

  const auto& encoded_list = GetOutput(0)->vec<Variant>();
  EXPECT_EQ(encoded_list.size(), 3);

  ExpectRaggedTensorVariantEqual<int, int64_t>(
      CreateVariantFromRagged<int, int64_t>({{0, 0, 0}}, {}),
      *encoded_list(0).get<RaggedTensorVariant>());
  ExpectRaggedTensorVariantEqual<int, int64_t>(
      CreateVariantFromRagged<int, int64_t>({{0, 0}}, {}),
      *encoded_list(1).get<RaggedTensorVariant>());
  ExpectRaggedTensorVariantEqual<int, int64_t>(
      CreateVariantFromRagged<int, int64_t>({{0}}, {}),
      *encoded_list(2).get<RaggedTensorVariant>());
}

TEST_F(RaggedTensorToVariantKernelTest, 1DValuesRaggedRankOneInput) {
  // ragged_tensor=
  // [ [1, 2, 3],
  //   [       ],
  //   [4, 5   ],
  //   [6      ]]
  const std::vector<int64_t> batched_splits = {0, 3, 3, 5, 6};
  const std::vector<int> batched_values = {1, 2, 3, 4, 5, 6};

  BuildEncodeRaggedTensorGraph<int, int64_t>({batched_splits}, TensorShape({6}),
                                             batched_values, true);
  TF_ASSERT_OK(RunOpKernel());

  const auto& encoded_list = GetOutput(0)->vec<Variant>();
  EXPECT_EQ(encoded_list.size(), 4);

  ExpectRaggedTensorVariantEqual<int, int64_t>(
      CreateVariantFromRagged<int, int64_t>({}, {1, 2, 3}),
      *encoded_list(0).get<RaggedTensorVariant>());
  ExpectRaggedTensorVariantEqual<int, int64_t>(
      CreateVariantFromRagged<int, int64_t>({}, {}),
      *encoded_list(1).get<RaggedTensorVariant>());
  ExpectRaggedTensorVariantEqual<int, int64_t>(
      CreateVariantFromRagged<int, int64_t>({}, {4, 5}),
      *encoded_list(2).get<RaggedTensorVariant>());
  ExpectRaggedTensorVariantEqual<int, int64_t>(
      CreateVariantFromRagged<int, int64_t>({}, {6}),
      *encoded_list(3).get<RaggedTensorVariant>());
}

TEST_F(RaggedTensorToVariantKernelTest, 2DBatchedValuesRankOneInput) {
  // ragged_tensor=
  // [[1, 2],
  //  [4, 5],
  //  [6, 7]]
  const std::vector<int64_t> batched_splits = {0, 1, 2, 3};
  const std::vector<int> batched_values = {1, 2, 4, 5, 6, 7};

  BuildEncodeRaggedTensorGraph<int, int64_t>(
      {batched_splits}, TensorShape({3, 2}), batched_values, true);
  TF_ASSERT_OK(RunOpKernel());

  const auto& encoded_list = GetOutput(0)->vec<Variant>();
  EXPECT_EQ(encoded_list.size(), 3);

  ExpectRaggedTensorVariantEqual<int, int64_t>(
      CreateVariantFromRagged<int, int64_t>({}, {1, 2}, {1, 2}),
      *encoded_list(0).get<RaggedTensorVariant>());
  ExpectRaggedTensorVariantEqual<int, int64_t>(
      CreateVariantFromRagged<int, int64_t>({}, {1, 2}, {4, 5}),
      *encoded_list(1).get<RaggedTensorVariant>());
  ExpectRaggedTensorVariantEqual<int, int64_t>(
      CreateVariantFromRagged<int, int64_t>({}, {1, 2}, {6, 7}),
      *encoded_list(2).get<RaggedTensorVariant>());
}

TEST_F(RaggedTensorToVariantKernelTest, 2DBatchedValuesRankTwoInput) {
  // ragged_tensor=
  // [ [[[1, 2], [4, 5]]],
  //   [[[6 7]]]          ]
  const std::vector<int64_t> batched_splits_1 = {0, 1, 2};
  const std::vector<int64_t> batched_splits_2 = {0, 2, 3};
  const std::vector<int> batched_values = {1, 2, 4, 5, 6, 7};

  BuildEncodeRaggedTensorGraph<int, int64_t>(
      {batched_splits_1, batched_splits_2}, TensorShape({3, 2}), batched_values,
      true);
  TF_ASSERT_OK(RunOpKernel());

  const auto& encoded_list = GetOutput(0)->vec<Variant>();
  EXPECT_EQ(encoded_list.size(), 2);

  ExpectRaggedTensorVariantEqual<int, int64_t>(
      CreateVariantFromRagged<int, int64_t>({{0, 2}}, {2, 2}, {1, 2, 4, 5}),
      *encoded_list(0).get<RaggedTensorVariant>());
  ExpectRaggedTensorVariantEqual<int, int64_t>(
      CreateVariantFromRagged<int, int64_t>({{0, 1}}, {1, 2}, {6, 7}),
      *encoded_list(1).get<RaggedTensorVariant>());
}

TEST_F(RaggedTensorToVariantKernelTest, EmptyRowInBatchedInput) {
  // ragged_tensor =
  // [[ [x],         [x x],       [] ],
  //  [                              ],
  //  [ [x x x x x], [x x x]         ],
  //  [ [],          [x x x x]       ]]
  const std::vector<int64_t> batched_splits_1 = {0, 3, 3, 5, 7};
  const std::vector<int64_t> batched_splits_2 = {0, 1, 3, 3, 8, 11, 11, 15};
  const std::vector<int> batched_values = {1, 2,  3,  4,  5,  6,  7, 8,
                                           9, 10, 11, 12, 13, 14, 15};

  BuildEncodeRaggedTensorGraph<int, int64_t>(
      {batched_splits_1, batched_splits_2}, TensorShape({15}), batched_values,
      true);
  TF_ASSERT_OK(RunOpKernel());

  const auto& encoded_list = GetOutput(0)->vec<Variant>();
  EXPECT_EQ(encoded_list.size(), 4);

  ExpectRaggedTensorVariantEqual<int, int64_t>(
      CreateVariantFromRagged<int, int64_t>({{0, 1, 3, 3}}, {1, 2, 3}),
      *encoded_list(0).get<RaggedTensorVariant>());
  ExpectRaggedTensorVariantEqual<int, int64_t>(
      CreateVariantFromRagged<int, int64_t>({{0}}, {}),
      *encoded_list(1).get<RaggedTensorVariant>());
  ExpectRaggedTensorVariantEqual<int, int64_t>(
      CreateVariantFromRagged<int, int64_t>({{0, 5, 8}},
                                            {4, 5, 6, 7, 8, 9, 10, 11}),
      *encoded_list(2).get<RaggedTensorVariant>());
  ExpectRaggedTensorVariantEqual<int, int64_t>(
      CreateVariantFromRagged<int, int64_t>({{0, 0, 4}}, {12, 13, 14, 15}),
      *encoded_list(3).get<RaggedTensorVariant>());
}

TEST_F(RaggedTensorToVariantKernelTest, NonEmptyBatchedInput) {
  // ragged_tensor =
  // [[     [ [x, x]        ],
  //        [ [x],      [x] ],
  //        [ [x]           ],
  //        [ [x]           ],
  //        [ [x]           ]],
  //  [     [ [x]           ],
  //        [ [x]           ],
  //        [ [x, x, x]     ],
  //        [ [x]           ],
  //        [ [x]           ] ]]
  const std::vector<int64_t> batched_splits_1 = {0, 5, 10};
  const std::vector<int64_t> batched_splits_2 = {0, 1, 3, 4,  5, 6,
                                                 7, 8, 9, 10, 11};
  const std::vector<int64_t> batched_splits_3 = {0, 2, 3, 4,  5,  6,
                                                 7, 8, 9, 12, 13, 14};
  const std::vector<int> batched_values = {0, 1, 1, 2, 2, 3, 4,
                                           5, 6, 7, 8, 9, 8, 9};

  BuildEncodeRaggedTensorGraph<int, int64_t>(
      {batched_splits_1, batched_splits_2, batched_splits_3}, TensorShape({14}),
      batched_values, true);
  TF_ASSERT_OK(RunOpKernel());

  const auto& encoded_list = GetOutput(0)->vec<Variant>();
  EXPECT_EQ(encoded_list.size(), 2);

  ExpectRaggedTensorVariantEqual<int, int64_t>(
      CreateVariantFromRagged<int, int64_t>(
          {{0, 1, 3, 4, 5, 6}, {0, 2, 3, 4, 5, 6, 7}}, {0, 1, 1, 2, 2, 3, 4}),
      *encoded_list(0).get<RaggedTensorVariant>());
  ExpectRaggedTensorVariantEqual<int, int64_t>(
      CreateVariantFromRagged<int, int64_t>(
          {{0, 1, 2, 3, 4, 5}, {0, 1, 2, 5, 6, 7}}, {5, 6, 7, 8, 9, 8, 9}),
      *encoded_list(1).get<RaggedTensorVariant>());
}

TEST_F(RaggedTensorToVariantKernelTest, NonEmptyBatchedInputInt32Splits) {
  // ragged_tensor =
  // [[     [ [x, x]        ],
  //        [ [x],      [x] ],
  //        [ [x]           ],
  //        [ [x]           ],
  //        [ [x]           ]],
  //  [     [ [x]           ],
  //        [ [x]           ],
  //        [ [x, x, x]     ],
  //        [ [x]           ],
  //        [ [x]           ] ]]
  const std::vector<int> batched_splits_1 = {0, 5, 10};
  const std::vector<int> batched_splits_2 = {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  const std::vector<int> batched_splits_3 = {0, 2, 3, 4,  5,  6,
                                             7, 8, 9, 12, 13, 14};
  const std::vector<int> batched_values = {0, 1, 1, 2, 2, 3, 4,
                                           5, 6, 7, 8, 9, 8, 9};

  BuildEncodeRaggedTensorGraph<int, int32>(
      {batched_splits_1, batched_splits_2, batched_splits_3}, TensorShape({14}),
      batched_values, true);
  TF_ASSERT_OK(RunOpKernel());

  const auto& encoded_list = GetOutput(0)->vec<Variant>();
  EXPECT_EQ(encoded_list.size(), 2);

  ExpectRaggedTensorVariantEqual<int, int32>(
      CreateVariantFromRagged<int, int32>(
          {{0, 1, 3, 4, 5, 6}, {0, 2, 3, 4, 5, 6, 7}}, {0, 1, 1, 2, 2, 3, 4}),
      *encoded_list(0).get<RaggedTensorVariant>());
  ExpectRaggedTensorVariantEqual<int, int32>(
      CreateVariantFromRagged<int, int32>(
          {{0, 1, 2, 3, 4, 5}, {0, 1, 2, 5, 6, 7}}, {5, 6, 7, 8, 9, 8, 9}),
      *encoded_list(1).get<RaggedTensorVariant>());
}

TEST_F(RaggedTensorToVariantKernelTest, NonBatchInput) {
  // ragged_tensor =
  // [[ [x],         [x x],       [] ],
  //  [                              ],
  //  [ [x x x x x], [x x x]         ],
  //  [ [],          [x x x x]       ]]
  const std::vector<int64_t> batched_splits_1 = {0, 3, 3, 5, 7};
  const std::vector<int64_t> batched_splits_2 = {0, 1, 3, 3, 8, 11, 11, 15};
  const std::vector<int> batched_values = {1, 2,  3,  4,  5,  6,  7, 8,
                                           9, 10, 11, 12, 13, 14, 15};

  BuildEncodeRaggedTensorGraph<int, int64_t>(
      {batched_splits_1, batched_splits_2}, TensorShape({15}), batched_values,
      false);
  TF_ASSERT_OK(RunOpKernel());

  const auto& encoded_scalar = GetOutput(0)->scalar<Variant>()();

  ExpectRaggedTensorVariantEqual<int, int64_t>(
      CreateVariantFromRagged<int, int64_t>(
          {batched_splits_1, batched_splits_2}, batched_values),
      *encoded_scalar.get<RaggedTensorVariant>());
}

TEST_F(RaggedTensorToVariantKernelTest, ShapeFnTestBatched) {
  ShapeInferenceTestOp op("RaggedTensorToVariant");
  (*op.node_def.mutable_attr())["Tvalues"].set_type(DT_INT32);
  (*op.node_def.mutable_attr())["batched_input"].set_b(true);

  // Tests with len(ragged_splits)==0.
  (*op.node_def.mutable_attr())["RAGGED_RANK"].set_i(0);
  INFER_OK(op, "?", "[?]");

  // Tests with len(ragged_splits)==1.
  (*op.node_def.mutable_attr())["RAGGED_RANK"].set_i(1);
  INFER_OK(op, "?;?", "[?]");
  INFER_OK(op, "?;[?]", "[?]");
  INFER_OK(op, "?;[?,?]", "[?]");
  INFER_OK(op, "[?];[5]", "[?]");
  INFER_OK(op, "[?];[5,2]", "[?]");
  INFER_OK(op, "[5];[5,2]", "[4]");
  INFER_ERROR("Shape must be rank 1 but is rank 0", op, "[];?");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[5,5];?");
  INFER_ERROR("Shape must be at least rank 1 but is rank 0", op, "?;[]");

  // Tests with len(ragged_splits)==2
  (*op.node_def.mutable_attr())["RAGGED_RANK"].set_i(2);
  INFER_OK(op, "?;?;?", "[?]");
  INFER_OK(op, "?;?;[?]", "[?]");
  INFER_OK(op, "?;?;[?,?]", "[?]");
  INFER_OK(op, "[?];[?];[5]", "[?]");
  INFER_OK(op, "[?];[?];[5,2]", "[?]");
  INFER_OK(op, "[6];[?];[5,2]", "[5]");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "?;[5,5];?");

  // Tests with len(ragged_splits)==3
  (*op.node_def.mutable_attr())["RAGGED_RANK"].set_i(3);
  INFER_OK(op, "?;?;?;?", "[?]");
  INFER_OK(op, "?;?;?;[?]", "[?]");
  INFER_OK(op, "?;?;?;[5]", "[?]");
  INFER_OK(op, "[4];?;?;[5]", "[3]");
}

TEST_F(RaggedTensorToVariantKernelTest, ShapeFnTestNotBatched) {
  ShapeInferenceTestOp op("RaggedTensorToVariant");
  (*op.node_def.mutable_attr())["Tvalues"].set_type(DT_INT32);
  (*op.node_def.mutable_attr())["batched_input"].set_b(false);

  // Tests with len(ragged_splits)==0.
  (*op.node_def.mutable_attr())["RAGGED_RANK"].set_i(0);
  INFER_OK(op, "?", "[]");

  // Tests with len(ragged_splits)==1.
  (*op.node_def.mutable_attr())["RAGGED_RANK"].set_i(1);
  INFER_OK(op, "?;?", "[]");
  INFER_OK(op, "?;[?]", "[]");
  INFER_OK(op, "?;[?,?]", "[]");
  INFER_OK(op, "[?];[5]", "[]");
  INFER_OK(op, "[?];[5,2]", "[]");
  INFER_ERROR("Shape must be rank 1 but is rank 0", op, "[];?");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[5,5];?");
  INFER_ERROR("Shape must be at least rank 1 but is rank 0", op, "?;[]");

  // Tests with len(ragged_splits)==2
  (*op.node_def.mutable_attr())["RAGGED_RANK"].set_i(2);
  INFER_OK(op, "?;?;?", "[]");
  INFER_OK(op, "?;?;[?]", "[]");
  INFER_OK(op, "?;?;[?,?]", "[]");
  INFER_OK(op, "[?];[?];[5]", "[]");
  INFER_OK(op, "[?];[?];[5,2]", "[]");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "?;[5,5];?");

  // Tests with len(ragged_splits)==3
  (*op.node_def.mutable_attr())["RAGGED_RANK"].set_i(3);
  INFER_OK(op, "?;?;?;?", "[]");
  INFER_OK(op, "?;?;?;[?]", "[]");
  INFER_OK(op, "?;?;?;[5]", "[]");
}

TEST_F(RaggedTensorToVariantKernelTest, NonRaggedInput) {
  const std::vector<int> values = {1, 2, 3, 4, 5, 6};

  BuildEncodeRaggedTensorGraph<int, int64_t>({}, TensorShape({6}), values,
                                             false);
  TF_ASSERT_OK(RunOpKernel());

  const auto& encoded_scalar = GetOutput(0)->scalar<Variant>()();
  ExpectRaggedTensorVariantEqual<int, int64_t>(
      CreateVariantFromRagged<int, int64_t>({}, values),
      *encoded_scalar.get<RaggedTensorVariant>());
}

TEST_F(RaggedTensorToVariantKernelTest, NonRaggedBatchedInput) {
  // input contains [[[1, 2], [3, 4], [5, 6]],  [[7, 8], [9, 10], [11, 12]]]
  TensorShape shape({2, 3, 2});
  const std::vector<int> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  BuildEncodeRaggedTensorGraph<int, int64_t>({}, shape, values, true);
  TF_ASSERT_OK(RunOpKernel());

  const auto& encoded_list = GetOutput(0)->vec<Variant>();
  EXPECT_EQ(encoded_list.size(), 2);

  // encoded_list[0] conatins [[1, 2], [3, 4], [5, 6]].
  ExpectRaggedTensorVariantEqual<int, int64_t>(
      CreateVariantFromRagged<int, int64_t>({}, {3, 2}, {1, 2, 3, 4, 5, 6}),
      *encoded_list(0).get<RaggedTensorVariant>());
  // encoded_list[1] contains [[7, 8], [9, 10], [11, 12]].
  ExpectRaggedTensorVariantEqual<int, int64_t>(
      CreateVariantFromRagged<int, int64_t>({}, {3, 2}, {7, 8, 9, 10, 11, 12}),
      *encoded_list(1).get<RaggedTensorVariant>());
}

TEST_F(RaggedTensorToVariantKernelTest,
       NestedRowSplitsFirstElementNotZeroError) {
  const std::vector<int64_t> splits = {1, 2};
  BuildEncodeRaggedTensorGraph<int, int64_t>({splits}, TensorShape({0}), {},
                                             true);
  EXPECT_THAT(RunOpKernel(),
              testing::StatusIs(error::INVALID_ARGUMENT,
                                "Requires the first element of "
                                "nested_row_splits[0] to be 0 but is 1"));
}

TEST_F(RaggedTensorToVariantKernelTest, NestedRowSplitsIncreasingError) {
  const std::vector<int64_t> splits = {0, 2, -1};
  BuildEncodeRaggedTensorGraph<int, int64_t>({splits}, TensorShape({0}), {},
                                             true);
  EXPECT_THAT(RunOpKernel(),
              testing::StatusIs(error::INVALID_ARGUMENT,
                                "Requires splits to be monotonically "
                                "increasing, but nested_row_splits[0][2]=-1 is "
                                "smaller than nested_row_splits[0][1]=2"));
}

TEST_F(RaggedTensorToVariantKernelTest, NestedRowSplitsSizeMismatchError) {
  const std::vector<int64_t> splits = {0, 2, 3};
  BuildEncodeRaggedTensorGraph<int, int64_t>({splits}, TensorShape({5}),
                                             {0, 1, 2, 3, 4}, true);
  EXPECT_THAT(
      RunOpKernel(),
      testing::StatusIs(error::INVALID_ARGUMENT,
                        "Requires nested_row_splits[0][-1]=3 to be equal with "
                        "the number of values in this dimension, which is 5."));
}

TEST_F(RaggedTensorToVariantKernelTest,
       NestedRowSplitsInnerDimensionSizeMismatchError) {
  const std::vector<int64_t> splits1 = {0, 2, 3};
  const std::vector<int64_t> splits2 = {0, 3, 3, 4};
  BuildEncodeRaggedTensorGraph<int, int64_t>(
      {splits1, splits2}, TensorShape({5}), {0, 1, 2, 3, 4}, true);
  EXPECT_THAT(
      RunOpKernel(),
      testing::StatusIs(error::INVALID_ARGUMENT,
                        "Requires nested_row_splits[1][-1]=4 to be equal with "
                        "the number of values in this dimension, which is 5."));
}

TEST_F(RaggedTensorToVariantKernelTest,
       NestedRowSplitsSizeOfSplitsMismatchError) {
  const std::vector<int64_t> splits1 = {0, 2};
  const std::vector<int64_t> splits2 = {0, 3, 3, 5};
  BuildEncodeRaggedTensorGraph<int, int64_t>(
      {splits1, splits2}, TensorShape({5}), {0, 1, 2, 3, 4}, true);
  EXPECT_THAT(
      RunOpKernel(),
      testing::StatusIs(error::INVALID_ARGUMENT,
                        "Requires nested_row_splits[0][-1]=2 to be equal with "
                        "the number of values in this dimension, which is 3."));
}

TEST_F(RaggedTensorToVariantKernelTest, NestedRowSplitsEmptySplitsError) {
  const std::vector<int64_t> splits = {};
  BuildEncodeRaggedTensorGraph<int, int64_t>({splits}, TensorShape({5}),
                                             {0, 1, 2, 3, 4}, true);
  EXPECT_THAT(RunOpKernel(),
              testing::StatusIs(error::INVALID_ARGUMENT,
                                "Requires nested_row_splits[0] has at least "
                                "one splits, but is empty."));
}

TEST_F(RaggedTensorToVariantKernelTest, NestedRowSplitsScalarValueError) {
  const std::vector<int64_t> splits = {0, 2};
  BuildEncodeRaggedTensorGraph<int, int64_t>({splits}, TensorShape({}), 1,
                                             true);
  EXPECT_THAT(RunOpKernel(),
              testing::StatusIs(error::INVALID_ARGUMENT,
                                "Requires flat_values to have rank>=1 when "
                                "nested_row_splits is not empty, but is 0."));
}

}  // namespace
}  // namespace tensorflow
