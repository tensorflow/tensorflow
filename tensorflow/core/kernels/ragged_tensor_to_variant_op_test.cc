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

#include <vector>

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
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class RaggedTensorToVariantKernelTest : public ::tensorflow::OpsTestBase {
 protected:
  // Builds the tensorflow test graph for the RaggedTensorToVariant op, and
  // populates the `splits` input with the given values.
  template <typename VALUE_TYPE, typename SPLIT_TYPE>
  void BuildEncodeRaggedTensorGraph(
      const std::vector<std::vector<SPLIT_TYPE>>& ragged_splits,
      const TensorShape& ragged_values_shape,
      const std::vector<VALUE_TYPE>& ragged_values, const bool batched) {
    const auto values_dtype = DataTypeToEnum<VALUE_TYPE>::v();
    const auto splits_dtype = DataTypeToEnum<SPLIT_TYPE>::v();
    int64 num_splits = ragged_splits.size();
    TF_ASSERT_OK(
        NodeDefBuilder("tested_op", "RaggedTensorToVariant")
            .Input(FakeInput(num_splits, splits_dtype))  // ragged_splits
            .Input(FakeInput(values_dtype))              // ragged_values
            .Attr("RAGGED_RANK", num_splits)
            .Attr("Tvalues", values_dtype)
            .Attr("Tsplits", splits_dtype)
            .Attr("batched_input", batched)
            .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
    for (const auto& splits : ragged_splits) {
      int64 splits_size = splits.size();
      AddInputFromArray<SPLIT_TYPE>(TensorShape({splits_size}), splits);
    }
    AddInputFromArray<VALUE_TYPE>(ragged_values_shape, ragged_values);
  }
};

TEST_F(RaggedTensorToVariantKernelTest, NoValuesInput) {
  // ragged_tensor=[[[], []], [[]], []]
  const std::vector<int64> batched_splits_1 = {0, 2, 3, 3};
  const std::vector<int64> batched_splits_2 = {0, 0, 0, 0};

  const std::vector<int64> component_splits_1_1 = {0, 0, 0};
  const std::vector<int64> component_splits_2_1 = {0, 0};
  const std::vector<int64> component_splits_3_1 = {0};

  Tensor expected_splits_1_1(DT_INT64, TensorShape({3}));
  Tensor expected_splits_2_1(DT_INT64, TensorShape({2}));
  Tensor expected_splits_3_1(DT_INT64, TensorShape({1}));

  test::FillValues<int64>(&expected_splits_1_1, component_splits_1_1);
  test::FillValues<int64>(&expected_splits_2_1, component_splits_2_1);
  test::FillValues<int64>(&expected_splits_3_1, component_splits_3_1);

  BuildEncodeRaggedTensorGraph<int, int64>({batched_splits_1, batched_splits_2},
                                           TensorShape({0}), {}, true);
  TF_ASSERT_OK(RunOpKernel());

  const auto& encoded_list = GetOutput(0)->vec<Variant>();
  EXPECT_EQ(encoded_list.size(), 3);

  const Variant& encoded_splits_1_1 =
      encoded_list(0).get<Tensor>()->vec<Variant>()(0);
  const Variant& encoded_values_1 =
      encoded_list(0).get<Tensor>()->vec<Variant>()(1);
  const Variant& encoded_splits_2_1 =
      encoded_list(1).get<Tensor>()->vec<Variant>()(0);
  const Variant& encoded_values_2 =
      encoded_list(1).get<Tensor>()->vec<Variant>()(1);
  const Variant& encoded_splits_3_1 =
      encoded_list(2).get<Tensor>()->vec<Variant>()(0);
  const Variant& encoded_values_3 =
      encoded_list(2).get<Tensor>()->vec<Variant>()(1);

  test::ExpectTensorEqual<int64>(*encoded_splits_1_1.get<Tensor>(),
                                 expected_splits_1_1);
  test::ExpectTensorEqual<int64>(*encoded_splits_2_1.get<Tensor>(),
                                 expected_splits_2_1);
  test::ExpectTensorEqual<int64>(*encoded_splits_3_1.get<Tensor>(),
                                 expected_splits_3_1);
  test::ExpectTensorEqual<int>(*encoded_values_1.get<Tensor>(),
                               Tensor(DT_INT32, TensorShape({0})));
  test::ExpectTensorEqual<int>(*encoded_values_2.get<Tensor>(),
                               Tensor(DT_INT32, TensorShape({0})));
  test::ExpectTensorEqual<int>(*encoded_values_3.get<Tensor>(),
                               Tensor(DT_INT32, TensorShape({0})));
}

TEST_F(RaggedTensorToVariantKernelTest, 1DValuesRaggedRankOneInput) {
  // ragged_tensor=
  // [ [x, x, x],
  //   [       ],
  //   [x, x   ],
  //   [x      ]]
  const std::vector<int64> batched_splits = {0, 3, 3, 5, 6};
  const std::vector<int> batched_values = {1, 2, 3, 4, 5, 6};

  const std::vector<int> component_values_1 = {1, 2, 3};
  const std::vector<int> component_values_3 = {4, 5};
  const std::vector<int> component_values_4 = {6};

  Tensor expected_values_1(DT_INT32, TensorShape({3}));
  Tensor expected_values_2(DT_INT32, TensorShape({0}));
  Tensor expected_values_3(DT_INT32, TensorShape({2}));
  Tensor expected_values_4(DT_INT32, TensorShape({1}));

  test::FillValues<int>(&expected_values_1, component_values_1);
  test::FillValues<int>(&expected_values_3, component_values_3);
  test::FillValues<int>(&expected_values_4, component_values_4);

  BuildEncodeRaggedTensorGraph<int, int64>({batched_splits}, TensorShape({6}),
                                           batched_values, true);
  TF_ASSERT_OK(RunOpKernel());

  const auto& encoded_list = GetOutput(0)->vec<Variant>();
  EXPECT_EQ(encoded_list.size(), 4);

  const Variant& encoded_values_1 =
      encoded_list(0).get<Tensor>()->vec<Variant>()(0);
  const Variant& encoded_values_2 =
      encoded_list(1).get<Tensor>()->vec<Variant>()(0);
  const Variant& encoded_values_3 =
      encoded_list(2).get<Tensor>()->vec<Variant>()(0);
  const Variant& encoded_values_4 =
      encoded_list(3).get<Tensor>()->vec<Variant>()(0);

  test::ExpectTensorEqual<int>(*encoded_values_1.get<Tensor>(),
                               expected_values_1);
  test::ExpectTensorEqual<int>(*encoded_values_2.get<Tensor>(),
                               expected_values_2);
  test::ExpectTensorEqual<int>(*encoded_values_3.get<Tensor>(),
                               expected_values_3);
  test::ExpectTensorEqual<int>(*encoded_values_4.get<Tensor>(),
                               expected_values_4);
}

TEST_F(RaggedTensorToVariantKernelTest, 2DBatchedValuesRankOneInput) {
  // ragged_tensor=
  // [[x, x],
  //  [x, x],
  //  [x, x]]
  const std::vector<int64> batched_splits = {0, 1, 2, 3};
  const std::vector<int> batched_values = {1, 2, 4, 5, 6, 7};

  const std::vector<int> component_values_1 = {1, 2};
  const std::vector<int> component_values_2 = {4, 5};
  const std::vector<int> component_values_3 = {6, 7};

  Tensor expected_values_1(DT_INT32, TensorShape({1, 2}));
  Tensor expected_values_2(DT_INT32, TensorShape({1, 2}));
  Tensor expected_values_3(DT_INT32, TensorShape({1, 2}));

  test::FillValues<int>(&expected_values_1, component_values_1);
  test::FillValues<int>(&expected_values_2, component_values_2);
  test::FillValues<int>(&expected_values_3, component_values_3);

  BuildEncodeRaggedTensorGraph<int, int64>(
      {batched_splits}, TensorShape({3, 2}), batched_values, true);
  TF_ASSERT_OK(RunOpKernel());

  const auto& encoded_list = GetOutput(0)->vec<Variant>();
  EXPECT_EQ(encoded_list.size(), 3);

  const Variant& encoded_values_1 =
      encoded_list(0).get<Tensor>()->vec<Variant>()(0);
  const Variant& encoded_values_2 =
      encoded_list(1).get<Tensor>()->vec<Variant>()(0);
  const Variant& encoded_values_3 =
      encoded_list(2).get<Tensor>()->vec<Variant>()(0);

  test::ExpectTensorEqual<int>(*encoded_values_1.get<Tensor>(),
                               expected_values_1);
  test::ExpectTensorEqual<int>(*encoded_values_2.get<Tensor>(),
                               expected_values_2);
  test::ExpectTensorEqual<int>(*encoded_values_3.get<Tensor>(),
                               expected_values_3);
}

TEST_F(RaggedTensorToVariantKernelTest, 2DBatchedValuesRankTwoInput) {
  // ragged_tensor=[
  // [ [[x, x], [x, x]],
  //   [[x, x]        ] ]
  const std::vector<int64> batched_splits_1 = {0, 1, 2};
  const std::vector<int64> batched_splits_2 = {0, 2, 3};
  const std::vector<int> batched_values = {1, 2, 4, 5, 6, 7};

  const std::vector<int64> component_splits_1_1 = {0, 2};
  const std::vector<int64> component_splits_2_1 = {0, 1};
  const std::vector<int> component_values_1 = {1, 2, 4, 5};
  const std::vector<int> component_values_2 = {6, 7};

  Tensor expected_splits_1_1(DT_INT64, TensorShape({2}));
  Tensor expected_splits_2_1(DT_INT64, TensorShape({2}));
  Tensor expected_values_1(DT_INT32, TensorShape({2, 2}));
  Tensor expected_values_2(DT_INT32, TensorShape({1, 2}));

  test::FillValues<int64>(&expected_splits_1_1, component_splits_1_1);
  test::FillValues<int64>(&expected_splits_2_1, component_splits_2_1);
  test::FillValues<int>(&expected_values_1, component_values_1);
  test::FillValues<int>(&expected_values_2, component_values_2);

  BuildEncodeRaggedTensorGraph<int, int64>({batched_splits_1, batched_splits_2},
                                           TensorShape({3, 2}), batched_values,
                                           true);
  TF_ASSERT_OK(RunOpKernel());

  const auto& encoded_list = GetOutput(0)->vec<Variant>();
  EXPECT_EQ(encoded_list.size(), 2);

  const Variant& encoded_splits_1_1 =
      encoded_list(0).get<Tensor>()->vec<Variant>()(0);
  const Variant& encoded_values_1 =
      encoded_list(0).get<Tensor>()->vec<Variant>()(1);
  const Variant& encoded_splits_2_1 =
      encoded_list(1).get<Tensor>()->vec<Variant>()(0);
  const Variant& encoded_values_2 =
      encoded_list(1).get<Tensor>()->vec<Variant>()(1);

  test::ExpectTensorEqual<int64>(*encoded_splits_1_1.get<Tensor>(),
                                 expected_splits_1_1);
  test::ExpectTensorEqual<int>(*encoded_values_1.get<Tensor>(),
                               expected_values_1);
  test::ExpectTensorEqual<int64>(*encoded_splits_2_1.get<Tensor>(),
                                 expected_splits_2_1);
  test::ExpectTensorEqual<int>(*encoded_values_2.get<Tensor>(),
                               expected_values_2);
}

TEST_F(RaggedTensorToVariantKernelTest, EmptyRowInBatchedInput) {
  // ragged_tensor =
  // [[ [x],         [x x],       [] ],
  //  [                              ],
  //  [ [x x x x x], [x x x]         ],
  //  [ [],          [x x x x]       ]]
  const std::vector<int64> batched_splits_1 = {0, 3, 3, 5, 7};
  const std::vector<int64> batched_splits_2 = {0, 1, 3, 3, 8, 11, 11, 15};
  const std::vector<int> batched_values = {1, 2,  3,  4,  5,  6,  7, 8,
                                           9, 10, 11, 12, 13, 14, 15};
  const std::vector<int64> component_splits_1_1 = {0, 1, 3, 3};
  const std::vector<int64> component_splits_2_1 = {0};
  const std::vector<int64> component_splits_3_1 = {0, 5, 8};
  const std::vector<int64> component_splits_4_1 = {0, 0, 4};
  const std::vector<int> component_values_1 = {1, 2, 3};
  const std::vector<int> component_values_3 = {4, 5, 6, 7, 8, 9, 10, 11};
  const std::vector<int> component_values_4 = {12, 13, 14, 15};

  Tensor expected_splits_1_1(DT_INT64, TensorShape({4}));
  Tensor expected_splits_2_1(DT_INT64, TensorShape({1}));
  Tensor expected_splits_3_1(DT_INT64, TensorShape({3}));
  Tensor expected_splits_4_1(DT_INT64, TensorShape({3}));
  Tensor expected_values_1(DT_INT32, TensorShape({3}));
  Tensor expected_values_2(DT_INT32, TensorShape({0}));
  Tensor expected_values_3(DT_INT32, TensorShape({8}));
  Tensor expected_values_4(DT_INT32, TensorShape({4}));

  test::FillValues<int64>(&expected_splits_1_1, component_splits_1_1);
  test::FillValues<int64>(&expected_splits_2_1, component_splits_2_1);
  test::FillValues<int64>(&expected_splits_3_1, component_splits_3_1);
  test::FillValues<int64>(&expected_splits_4_1, component_splits_4_1);
  test::FillValues<int>(&expected_values_1, component_values_1);
  test::FillValues<int>(&expected_values_3, component_values_3);
  test::FillValues<int>(&expected_values_4, component_values_4);

  BuildEncodeRaggedTensorGraph<int, int64>({batched_splits_1, batched_splits_2},
                                           TensorShape({15}), batched_values,
                                           true);
  TF_ASSERT_OK(RunOpKernel());

  const auto& encoded_list = GetOutput(0)->vec<Variant>();
  EXPECT_EQ(encoded_list.size(), 4);

  const Variant& encoded_splits_1_1 =
      encoded_list(0).get<Tensor>()->vec<Variant>()(0);
  const Variant& encoded_values_1 =
      encoded_list(0).get<Tensor>()->vec<Variant>()(1);
  const Variant& encoded_splits_2_1 =
      encoded_list(1).get<Tensor>()->vec<Variant>()(0);
  const Variant& encoded_values_2 =
      encoded_list(1).get<Tensor>()->vec<Variant>()(1);
  const Variant& encoded_splits_3_1 =
      encoded_list(2).get<Tensor>()->vec<Variant>()(0);
  const Variant& encoded_values_3 =
      encoded_list(2).get<Tensor>()->vec<Variant>()(1);
  const Variant& encoded_splits_4_1 =
      encoded_list(3).get<Tensor>()->vec<Variant>()(0);
  const Variant& encoded_values_4 =
      encoded_list(3).get<Tensor>()->vec<Variant>()(1);

  test::ExpectTensorEqual<int64>(*encoded_splits_1_1.get<Tensor>(),
                                 expected_splits_1_1);
  test::ExpectTensorEqual<int>(*encoded_values_1.get<Tensor>(),
                               expected_values_1);
  test::ExpectTensorEqual<int64>(*encoded_splits_2_1.get<Tensor>(),
                                 expected_splits_2_1);
  test::ExpectTensorEqual<int>(*encoded_values_2.get<Tensor>(),
                               expected_values_2);
  test::ExpectTensorEqual<int64>(*encoded_splits_3_1.get<Tensor>(),
                                 expected_splits_3_1);
  test::ExpectTensorEqual<int>(*encoded_values_3.get<Tensor>(),
                               expected_values_3);
  test::ExpectTensorEqual<int64>(*encoded_splits_4_1.get<Tensor>(),
                                 expected_splits_4_1);
  test::ExpectTensorEqual<int>(*encoded_values_4.get<Tensor>(),
                               expected_values_4);
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
  const std::vector<int64> batched_splits_1 = {0, 5, 10};
  const std::vector<int64> batched_splits_2 = {0, 1, 3, 4,  5, 6,
                                               7, 8, 9, 10, 11};
  const std::vector<int64> batched_splits_3 = {0, 2, 3, 4,  5,  6,
                                               7, 8, 9, 12, 13, 14};
  const std::vector<int> batched_values = {0, 1, 1, 2, 2, 3, 4,
                                           5, 6, 7, 8, 9, 8, 9};
  const std::vector<int64> component_split_1_1 = {0, 1, 3, 4, 5, 6};
  const std::vector<int64> component_split_1_2 = {0, 2, 3, 4, 5, 6, 7};
  const std::vector<int64> component_split_2_1 = {0, 1, 2, 3, 4, 5};
  const std::vector<int64> component_split_2_2 = {0, 1, 2, 5, 6, 7};
  const std::vector<int> component_values_1 = {0, 1, 1, 2, 2, 3, 4};
  const std::vector<int> component_values_2 = {5, 6, 7, 8, 9, 8, 9};

  Tensor expected_splits_1_1(DT_INT64, TensorShape({6}));
  Tensor expected_splits_1_2(DT_INT64, TensorShape({7}));
  Tensor expected_splits_2_1(DT_INT64, TensorShape({6}));
  Tensor expected_splits_2_2(DT_INT64, TensorShape({6}));
  Tensor expected_values_1(DT_INT32, TensorShape({7}));
  Tensor expected_values_2(DT_INT32, TensorShape({7}));

  test::FillValues<int64>(&expected_splits_1_1, component_split_1_1);
  test::FillValues<int64>(&expected_splits_1_2, component_split_1_2);
  test::FillValues<int64>(&expected_splits_2_1, component_split_2_1);
  test::FillValues<int64>(&expected_splits_2_2, component_split_2_2);
  test::FillValues<int>(&expected_values_1, component_values_1);
  test::FillValues<int>(&expected_values_2, component_values_2);

  BuildEncodeRaggedTensorGraph<int, int64>(
      {batched_splits_1, batched_splits_2, batched_splits_3}, TensorShape({14}),
      batched_values, true);
  TF_ASSERT_OK(RunOpKernel());

  const auto& encoded_list = GetOutput(0)->vec<Variant>();
  EXPECT_EQ(encoded_list.size(), 2);

  const Variant& encoded_splits_1_1 =
      encoded_list(0).get<Tensor>()->vec<Variant>()(0);
  const Variant& encoded_splits_1_2 =
      encoded_list(0).get<Tensor>()->vec<Variant>()(1);
  const Variant& encoded_values_1 =
      encoded_list(0).get<Tensor>()->vec<Variant>()(2);
  const Variant& encoded_splits_2_1 =
      encoded_list(1).get<Tensor>()->vec<Variant>()(0);
  const Variant& encoded_splits_2_2 =
      encoded_list(1).get<Tensor>()->vec<Variant>()(1);
  const Variant& encoded_values_2 =
      encoded_list(1).get<Tensor>()->vec<Variant>()(2);

  test::ExpectTensorEqual<int64>(*encoded_splits_1_1.get<Tensor>(),
                                 expected_splits_1_1);
  test::ExpectTensorEqual<int64>(*encoded_splits_1_2.get<Tensor>(),
                                 expected_splits_1_2);
  test::ExpectTensorEqual<int64>(*encoded_splits_2_1.get<Tensor>(),
                                 expected_splits_2_1);
  test::ExpectTensorEqual<int64>(*encoded_splits_2_2.get<Tensor>(),
                                 expected_splits_2_2);
  test::ExpectTensorEqual<int>(*encoded_values_1.get<Tensor>(),
                               expected_values_1);
  test::ExpectTensorEqual<int>(*encoded_values_2.get<Tensor>(),
                               expected_values_2);
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
  const std::vector<int> component_split_1_1 = {0, 1, 3, 4, 5, 6};
  const std::vector<int> component_split_1_2 = {0, 2, 3, 4, 5, 6, 7};
  const std::vector<int> component_split_2_1 = {0, 1, 2, 3, 4, 5};
  const std::vector<int> component_split_2_2 = {0, 1, 2, 5, 6, 7};
  const std::vector<int> component_values_1 = {0, 1, 1, 2, 2, 3, 4};
  const std::vector<int> component_values_2 = {5, 6, 7, 8, 9, 8, 9};

  Tensor expected_splits_1_1(DT_INT32, TensorShape({6}));
  Tensor expected_splits_1_2(DT_INT32, TensorShape({7}));
  Tensor expected_splits_2_1(DT_INT32, TensorShape({6}));
  Tensor expected_splits_2_2(DT_INT32, TensorShape({6}));
  Tensor expected_values_1(DT_INT32, TensorShape({7}));
  Tensor expected_values_2(DT_INT32, TensorShape({7}));

  test::FillValues<int>(&expected_splits_1_1, component_split_1_1);
  test::FillValues<int>(&expected_splits_1_2, component_split_1_2);
  test::FillValues<int>(&expected_splits_2_1, component_split_2_1);
  test::FillValues<int>(&expected_splits_2_2, component_split_2_2);
  test::FillValues<int>(&expected_values_1, component_values_1);
  test::FillValues<int>(&expected_values_2, component_values_2);

  BuildEncodeRaggedTensorGraph<int, int>(
      {batched_splits_1, batched_splits_2, batched_splits_3}, TensorShape({14}),
      batched_values, true);
  TF_ASSERT_OK(RunOpKernel());

  const auto& encoded_list = GetOutput(0)->vec<Variant>();
  EXPECT_EQ(encoded_list.size(), 2);

  const Variant& encoded_splits_1_1 =
      encoded_list(0).get<Tensor>()->vec<Variant>()(0);
  const Variant& encoded_splits_1_2 =
      encoded_list(0).get<Tensor>()->vec<Variant>()(1);
  const Variant& encoded_values_1 =
      encoded_list(0).get<Tensor>()->vec<Variant>()(2);
  const Variant& encoded_splits_2_1 =
      encoded_list(1).get<Tensor>()->vec<Variant>()(0);
  const Variant& encoded_splits_2_2 =
      encoded_list(1).get<Tensor>()->vec<Variant>()(1);
  const Variant& encoded_values_2 =
      encoded_list(1).get<Tensor>()->vec<Variant>()(2);

  test::ExpectTensorEqual<int>(*encoded_splits_1_1.get<Tensor>(),
                               expected_splits_1_1);
  test::ExpectTensorEqual<int>(*encoded_splits_1_2.get<Tensor>(),
                               expected_splits_1_2);
  test::ExpectTensorEqual<int>(*encoded_splits_2_1.get<Tensor>(),
                               expected_splits_2_1);
  test::ExpectTensorEqual<int>(*encoded_splits_2_2.get<Tensor>(),
                               expected_splits_2_2);
  test::ExpectTensorEqual<int>(*encoded_values_1.get<Tensor>(),
                               expected_values_1);
  test::ExpectTensorEqual<int>(*encoded_values_2.get<Tensor>(),
                               expected_values_2);
}

TEST_F(RaggedTensorToVariantKernelTest, NonBatchInput) {
  // ragged_tensor =
  // [[ [x],         [x x],       [] ],
  //  [                              ],
  //  [ [x x x x x], [x x x]         ],
  //  [ [],          [x x x x]       ]]
  const std::vector<int64> batched_splits_1 = {0, 3, 3, 5, 7};
  const std::vector<int64> batched_splits_2 = {0, 1, 3, 3, 8, 11, 11, 15};
  const std::vector<int> batched_values = {1, 2,  3,  4,  5,  6,  7, 8,
                                           9, 10, 11, 12, 13, 14, 15};

  Tensor batched_ragged_splits_1(DT_INT64, TensorShape({5}));
  Tensor batched_ragged_splits_2(DT_INT64, TensorShape({8}));
  Tensor batched_ragged_values(DT_INT32, TensorShape({15}));

  test::FillValues<int64>(&batched_ragged_splits_1, batched_splits_1);
  test::FillValues<int64>(&batched_ragged_splits_2, batched_splits_2);
  test::FillValues<int>(&batched_ragged_values, batched_values);

  BuildEncodeRaggedTensorGraph<int, int64>({batched_splits_1, batched_splits_2},
                                           TensorShape({15}), batched_values,
                                           false);
  TF_ASSERT_OK(RunOpKernel());

  const auto& encoded_scalar = GetOutput(0)->scalar<Variant>()();
  const Variant& encoded_splits_1 =
      encoded_scalar.get<Tensor>()->vec<Variant>()(0);
  const Variant& encoded_splits_2 =
      encoded_scalar.get<Tensor>()->vec<Variant>()(1);
  const Variant& encoded_values =
      encoded_scalar.get<Tensor>()->vec<Variant>()(2);

  test::ExpectTensorEqual<int64>(*encoded_splits_1.get<Tensor>(),
                                 batched_ragged_splits_1);
  test::ExpectTensorEqual<int64>(*encoded_splits_2.get<Tensor>(),
                                 batched_ragged_splits_2);
  test::ExpectTensorEqual<int>(*encoded_values.get<Tensor>(),
                               batched_ragged_values);
}

TEST_F(RaggedTensorToVariantKernelTest, ShapeFnTestBatched) {
  ShapeInferenceTestOp op("RaggedTensorToVariant");
  (*op.node_def.mutable_attr())["batched_input"].set_b(true);

  // Tests with len(ragged_splits)==0.
  (*op.node_def.mutable_attr())["RAGGED_RANK"].set_i(0);
  INFER_ERROR(
      "ragged_rank=0 is not currently supported "
      "when batched_input=true.",
      op, "?");

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
  Tensor expected_values(DT_INT32, TensorShape({6}));
  test::FillValues<int>(&expected_values, values);

  BuildEncodeRaggedTensorGraph<int, int64>({}, TensorShape({6}), values, false);
  TF_ASSERT_OK(RunOpKernel());

  const auto& encoded_scalar = GetOutput(0)->scalar<Variant>()();
  const Variant& encoded_values =
      encoded_scalar.get<Tensor>()->vec<Variant>()(0);

  test::ExpectTensorEqual<int>(*encoded_values.get<Tensor>(), expected_values);
}

}  // namespace
}  // namespace tensorflow
