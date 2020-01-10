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
#include <utility>
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

class RaggedTensorFromVariantKernelTest : public ::tensorflow::OpsTestBase {
 protected:
  // Builds the tensorflow test graph for the RaggedTensorFromVariant op, and
  // populates the variant input with the given values.
  template <typename VALUE_TYPE, typename SPLIT_TYPE>
  void BuildDecodeRaggedTensorGraph(
      const int input_ragged_rank, const int output_ragged_rank,
      const TensorShape& variant_shape,
      const std::vector<Variant>& variant_values) {
    const auto value_dtype = DataTypeToEnum<VALUE_TYPE>::v();
    const auto split_dtype = DataTypeToEnum<SPLIT_TYPE>::v();
    TF_ASSERT_OK(NodeDefBuilder("tested_op", "RaggedTensorFromVariant")
                     .Input(FakeInput(DT_VARIANT))
                     .Attr("input_ragged_rank", input_ragged_rank)
                     .Attr("output_ragged_rank", output_ragged_rank)
                     .Attr("Tvalues", value_dtype)
                     .Attr("Tsplits", split_dtype)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
    AddInputFromArray<Variant>(variant_shape, variant_values);
  }

  template <typename VALUE_TYPE, typename SPLIT_TYPE>
  Tensor CreateVariantFromRagged(
      const std::vector<std::vector<SPLIT_TYPE>>& ragged_splits,
      const TensorShape& ragged_values_shape,
      const std::vector<VALUE_TYPE>& ragged_values) {
    // Step 1: Create Tensors out of ragged splits and values.
    std::vector<Variant> ragged_components;
    for (auto ragged_split : ragged_splits) {
      int splits_size = ragged_split.size();
      Tensor splits(DataTypeToEnum<SPLIT_TYPE>::v(),
                    TensorShape({splits_size}));
      test::FillValues<SPLIT_TYPE>(&splits, ragged_split);
      ragged_components.push_back(splits);
    }
    Tensor values(DataTypeToEnum<VALUE_TYPE>::v(), ragged_values_shape);
    test::FillValues<VALUE_TYPE>(&values, ragged_values);
    ragged_components.push_back(values);

    // Step 2: Encode into a 1-D Variant Tensor.
    int num_splits = ragged_splits.size();
    Tensor encoded_list(DT_VARIANT, TensorShape({num_splits + 1}));
    test::FillValues<Variant>(&encoded_list, ragged_components);
    return encoded_list;
  }
};

TEST_F(RaggedTensorFromVariantKernelTest, ScalarInput) {
  const std::vector<int64> split_1 = {0, 1, 2, 3, 4, 5};
  const std::vector<int64> split_2 = {0, 1, 2, 5, 6, 7};
  const std::vector<int> values = {0, 1, 1, 2, 2, 3, 4};

  Tensor encoded_variant = CreateVariantFromRagged<int, int64>(
      {split_1, split_2}, TensorShape({7}), values);
  Tensor expected_splits_1(DT_INT64, TensorShape({6}));
  Tensor expected_splits_2(DT_INT64, TensorShape({6}));
  Tensor expected_values(DT_INT32, TensorShape({7}));

  test::FillValues<int64>(&expected_splits_1, split_1);
  test::FillValues<int64>(&expected_splits_2, split_2);
  test::FillValues<int>(&expected_values, values);

  int input_ragged_rank = 2;
  int output_ragged_rank = 2;
  BuildDecodeRaggedTensorGraph<int, int64>(input_ragged_rank,
                                           output_ragged_rank, TensorShape({}),
                                           {encoded_variant});
  TF_ASSERT_OK(RunOpKernel());

  test::ExpectTensorEqual<int64>(*GetOutput(0), expected_splits_1);
  test::ExpectTensorEqual<int64>(*GetOutput(1), expected_splits_2);
  test::ExpectTensorEqual<int>(*GetOutput(2), expected_values);
}

TEST_F(RaggedTensorFromVariantKernelTest, OneInputElement) {
  const std::vector<int64> split_1 = {0, 1, 2, 3, 4, 5};
  const std::vector<int64> split_2 = {0, 1, 2, 5, 6, 7};
  const std::vector<int> values = {0, 1, 1, 2, 2, 3, 4};
  const std::vector<int64> batched_splits_1 = {0, 5};

  Tensor encoded_variant = CreateVariantFromRagged<int, int64>(
      {split_1, split_2}, TensorShape({7}), values);
  Tensor expected_splits_1(DT_INT64, TensorShape({2}));
  Tensor expected_splits_2(DT_INT64, TensorShape({6}));
  Tensor expected_splits_3(DT_INT64, TensorShape({6}));
  Tensor expected_values(DT_INT32, TensorShape({7}));

  test::FillValues<int64>(&expected_splits_1, batched_splits_1);
  test::FillValues<int64>(&expected_splits_2, split_1);
  test::FillValues<int64>(&expected_splits_3, split_2);
  test::FillValues<int>(&expected_values, values);

  int input_ragged_rank = 2;
  int output_ragged_rank = 3;
  BuildDecodeRaggedTensorGraph<int, int64>(input_ragged_rank,
                                           output_ragged_rank, TensorShape({1}),
                                           {encoded_variant});
  TF_ASSERT_OK(RunOpKernel());

  test::ExpectTensorEqual<int64>(*GetOutput(0), expected_splits_1);
  test::ExpectTensorEqual<int64>(*GetOutput(1), expected_splits_2);
  test::ExpectTensorEqual<int64>(*GetOutput(2), expected_splits_3);
  test::ExpectTensorEqual<int>(*GetOutput(3), expected_values);
}

TEST_F(RaggedTensorFromVariantKernelTest, TensorIn2DOut) {
  // component_1 = [x, x, x]
  // component_2 = []
  // component_3 = [x, x]
  // component_4 = []
  // batched_ragged =
  // [[component_1, component_2], [component_3, component_4]]
  // [
  //   [ [x, x, x], []  ],
  //   [ [x, x],    [x] ]
  // ]
  const std::vector<int> values_1 = {1, 2, 3};
  const std::vector<int> values_2 = {};
  const std::vector<int> values_3 = {4, 5};
  const std::vector<int> values_4 = {6};
  const std::vector<int64> batched_splits_1 = {0, 2, 4};
  const std::vector<int64> batched_splits_2 = {0, 3, 3, 5, 6};
  const std::vector<int> batched_values = {1, 2, 3, 4, 5, 6};

  Tensor component_variant_1 =
      CreateVariantFromRagged<int, int64>({}, TensorShape({3}), values_1);
  Tensor component_variant_2 =
      CreateVariantFromRagged<int, int64>({}, TensorShape({0}), values_2);
  Tensor component_variant_3 =
      CreateVariantFromRagged<int, int64>({}, TensorShape({2}), values_3);
  Tensor component_variant_4 =
      CreateVariantFromRagged<int, int64>({}, TensorShape({1}), values_4);

  Tensor expected_splits_1(DT_INT64, TensorShape({3}));
  Tensor expected_splits_2(DT_INT64, TensorShape({5}));
  Tensor expected_values(DT_INT32, TensorShape({6}));

  test::FillValues<int64>(&expected_splits_1, batched_splits_1);
  test::FillValues<int64>(&expected_splits_2, batched_splits_2);
  test::FillValues<int>(&expected_values, batched_values);

  int input_ragged_rank = 0;
  int output_ragged_rank = 2;
  BuildDecodeRaggedTensorGraph<int, int64>(
      input_ragged_rank, output_ragged_rank, TensorShape({2, 2}),
      {component_variant_1, component_variant_2, component_variant_3,
       component_variant_4});
  TF_ASSERT_OK(RunOpKernel());

  test::ExpectTensorEqual<int64>(*GetOutput(0), expected_splits_1);
  test::ExpectTensorEqual<int64>(*GetOutput(1), expected_splits_2);
  test::ExpectTensorEqual<int>(*GetOutput(2), expected_values);
}

TEST_F(RaggedTensorFromVariantKernelTest, NonEmpty1DIn3DOut) {
  // ragged_component_1 = [[x]]
  // ragged_component_2 = [[x], [x]]
  // ragged_component_3 = [[x, x]]
  // ragged_component_4 = [[x, x], [x]]
  // ragged_component_5 = [[x], [x, x]]
  // batched_ragged = [[rc1, rc2, rc3, rc4, rc5], [rc4, rc5, rc1, rc3, rc2]]
  const std::vector<int64> component_split_1_1 = {0, 1};
  const std::vector<int64> component_split_2_1 = {0, 1, 2};
  const std::vector<int64> component_split_3_1 = {0, 2};
  const std::vector<int64> component_split_4_1 = {0, 2, 3};
  const std::vector<int64> component_split_5_1 = {0, 1, 3};
  const std::vector<int> component_values_1 = {0};
  const std::vector<int> component_values_2 = {0, 1};
  const std::vector<int> component_values_3 = {0, 1};
  const std::vector<int> component_values_4 = {0, 1, 2};
  const std::vector<int> component_values_5 = {0, 1, 2};

  const std::vector<int64> batched_splits_1 = {0, 5, 10};
  const std::vector<int64> batched_splits_2 = {0,  1,  3,  4,  6, 8,
                                               10, 12, 13, 14, 16};
  const std::vector<int64> batched_splits_3 = {
      0, 1, 2, 3, 5, 7, 8, 9, 11, 13, 14, 15, 17, 18, 20, 21, 22};
  const std::vector<int> batched_values = {0, 0, 1, 0, 1, 0, 1, 2, 0, 1, 2,
                                           0, 1, 2, 0, 1, 2, 0, 0, 1, 0, 1};

  Tensor expected_splits_1(DT_INT64, TensorShape({3}));
  Tensor expected_splits_2(DT_INT64, TensorShape({11}));
  Tensor expected_splits_3(DT_INT64, TensorShape({17}));
  Tensor expected_values(DT_INT32, TensorShape({22}));

  test::FillValues<int64>(&expected_splits_1, batched_splits_1);
  test::FillValues<int64>(&expected_splits_2, batched_splits_2);
  test::FillValues<int64>(&expected_splits_3, batched_splits_3);
  test::FillValues<int>(&expected_values, batched_values);

  Tensor variant_component_1 = CreateVariantFromRagged<int, int64>(
      {component_split_1_1}, TensorShape({1}), component_values_1);
  Tensor variant_component_2 = CreateVariantFromRagged<int, int64>(
      {component_split_2_1}, TensorShape({2}), component_values_2);
  Tensor variant_component_3 = CreateVariantFromRagged<int, int64>(
      {component_split_3_1}, TensorShape({2}), component_values_3);
  Tensor variant_component_4 = CreateVariantFromRagged<int, int64>(
      {component_split_4_1}, TensorShape({3}), component_values_4);
  Tensor variant_component_5 = CreateVariantFromRagged<int, int64>(
      {component_split_5_1}, TensorShape({3}), component_values_5);
  int input_ragged_rank = 1;
  int output_ragged_rank = 3;
  BuildDecodeRaggedTensorGraph<int, int64>(
      input_ragged_rank, output_ragged_rank, TensorShape({2, 5}),
      {variant_component_1, variant_component_2, variant_component_3,
       variant_component_4, variant_component_5, variant_component_4,
       variant_component_5, variant_component_1, variant_component_3,
       variant_component_2});
  TF_ASSERT_OK(RunOpKernel());

  test::ExpectTensorEqual<int64>(*GetOutput(0), expected_splits_1);
  test::ExpectTensorEqual<int64>(*GetOutput(1), expected_splits_2);
  test::ExpectTensorEqual<int64>(*GetOutput(2), expected_splits_3);
  test::ExpectTensorEqual<int>(*GetOutput(3), expected_values);
}

TEST_F(RaggedTensorFromVariantKernelTest,
       NonEmpty2DIn4DOutInferredInputRaggedRank) {
  // ragged_component_1 =
  // [
  //   [ [x]            ],
  //   [ [x],    [x]    ],
  //   [ [x, x]         ],
  //   [ [x, x], [x]    ],
  //   [ [x],    [x, x] ]
  // ]
  // ragged_component_2 =
  // [
  //   [ [x, x], [x]    ],
  //   [ [x],    [x, x] ],
  //   [ [x]            ],
  //   [ [x, x]         ],
  //   [ [x],    [x]    ]
  // ]
  // batched_ragged = [[rc1, rc2], [rc2, rc1]]
  const std::vector<int64> component_split_1_1 = {0, 1, 3, 4, 6, 8};
  const std::vector<int64> component_split_1_2 = {0, 1, 2, 3, 5, 7, 8, 9, 11};
  const std::vector<int64> component_split_2_1 = {0, 2, 4, 5, 6, 8};
  const std::vector<int64> component_split_2_2 = {0, 2, 3, 4, 6, 7, 9, 10, 11};
  const std::vector<int> component_values_1 = {0, 0, 1, 0, 1, 0, 1, 2, 0, 1, 2};
  const std::vector<int> component_values_2 = {0, 1, 2, 0, 1, 2, 0, 0, 1, 0, 1};
  const std::vector<int64> batched_splits_1 = {0, 2, 4};
  const std::vector<int64> batched_splits_2 = {0, 5, 10, 15, 20};
  const std::vector<int64> batched_splits_3 = {0,  1,  3,  4,  6,  8,  10,
                                               12, 13, 14, 16, 18, 20, 21,
                                               22, 24, 25, 27, 28, 30, 32};
  const std::vector<int64> batched_splits_4 = {
      0,  1,  2,  3,  5,  7,  8,  9,  11, 13, 14, 15, 17, 18, 20, 21, 22,
      24, 25, 26, 28, 29, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44};
  const std::vector<int> batched_values = {
      0, 0, 1, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 1, 0, 1,
      0, 1, 2, 0, 1, 2, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 0, 1, 2};

  Tensor expected_splits_1(DT_INT64, TensorShape({3}));
  Tensor expected_splits_2(DT_INT64, TensorShape({5}));
  Tensor expected_splits_3(DT_INT64, TensorShape({21}));
  Tensor expected_splits_4(DT_INT64, TensorShape({33}));
  Tensor expected_values(DT_INT32, TensorShape({44}));
  test::FillValues<int64>(&expected_splits_1, batched_splits_1);
  test::FillValues<int64>(&expected_splits_2, batched_splits_2);
  test::FillValues<int64>(&expected_splits_3, batched_splits_3);
  test::FillValues<int64>(&expected_splits_4, batched_splits_4);
  test::FillValues<int>(&expected_values, batched_values);

  Tensor variant_component_1 = CreateVariantFromRagged<int, int64>(
      {component_split_1_1, component_split_1_2}, TensorShape({11}),
      component_values_1);
  Tensor variant_component_2 = CreateVariantFromRagged<int, int64>(
      {component_split_2_1, component_split_2_2}, TensorShape({11}),
      component_values_2);
  int input_ragged_rank = -1;
  int output_ragged_rank = 4;
  BuildDecodeRaggedTensorGraph<int, int64>(
      input_ragged_rank, output_ragged_rank, TensorShape({2, 2}),
      {variant_component_1, variant_component_2, variant_component_2,
       variant_component_1});
  TF_ASSERT_OK(RunOpKernel());

  test::ExpectTensorEqual<int64>(*GetOutput(0), expected_splits_1);
  test::ExpectTensorEqual<int64>(*GetOutput(1), expected_splits_2);
  test::ExpectTensorEqual<int64>(*GetOutput(2), expected_splits_3);
  test::ExpectTensorEqual<int64>(*GetOutput(3), expected_splits_4);
  test::ExpectTensorEqual<int>(*GetOutput(4), expected_values);
}

TEST_F(RaggedTensorFromVariantKernelTest, EmptyRow1DIn2DOut) {
  // ragged_component_1 = [[x, x, x], []]
  // ragged_component_2 = []
  // batched_ragged = [rc1, rc2] = [[[x, x, x], []], []]
  const std::vector<int64> component_split_1_1 = {0, 3, 3};
  const std::vector<int> component_values_1 = {1, 2, 3};
  const std::vector<int64> component_split_2_1 = {0};
  const std::vector<int64> batched_splits_1 = {0, 2, 2};
  const std::vector<int64> batched_splits_2 = {0, 3, 3};
  const std::vector<int> batched_values = {1, 2, 3};

  Tensor expected_splits_1(DT_INT64, TensorShape({3}));
  Tensor expected_splits_2(DT_INT64, TensorShape({3}));
  Tensor expected_values(DT_INT32, TensorShape({3}));
  test::FillValues<int64>(&expected_splits_1, batched_splits_1);
  test::FillValues<int64>(&expected_splits_2, batched_splits_2);
  test::FillValues<int>(&expected_values, batched_values);

  Tensor variant_component_1 = CreateVariantFromRagged<int, int64>(
      {component_split_1_1}, TensorShape({3}), component_values_1);
  Tensor variant_component_2 = CreateVariantFromRagged<int, int64>(
      {component_split_2_1}, TensorShape({0}), {});  // Empty row.
  int input_ragged_rank = 1;
  int output_ragged_rank = 2;
  BuildDecodeRaggedTensorGraph<int, int64>(
      input_ragged_rank, output_ragged_rank, TensorShape({2}),
      {variant_component_1, variant_component_2});
  TF_ASSERT_OK(RunOpKernel());

  test::ExpectTensorEqual<int64>(*GetOutput(0), expected_splits_1);
  test::ExpectTensorEqual<int64>(*GetOutput(1), expected_splits_2);
  test::ExpectTensorEqual<int>(*GetOutput(2), expected_values);
}

TEST_F(RaggedTensorFromVariantKernelTest, NDValues1DIn2DOut) {
  // ragged_component_1 = [[x]]
  // ragged_component_1 = [[x], [x]]
  // batched_ragged = [rc1, rc2] = [[[x]], [[x], [x]]]
  const std::vector<int64> component_split_1_1 = {0, 1};
  const std::vector<int> component_values_1 = {1, 2};
  const std::vector<int64> component_split_2_1 = {0, 1, 2};
  const std::vector<int> component_values_2 = {1, 2, 3, 4};
  const std::vector<int64> batched_splits_1 = {0, 1, 3};
  const std::vector<int64> batched_splits_2 = {0, 1, 2, 3};
  const std::vector<int> batched_values = {1, 2, 1, 2, 3, 4};

  Tensor expected_splits_1(DT_INT64, TensorShape({3}));
  Tensor expected_splits_2(DT_INT64, TensorShape({4}));
  Tensor expected_values(DT_INT32, TensorShape({3, 2}));
  test::FillValues<int64>(&expected_splits_1, batched_splits_1);
  test::FillValues<int64>(&expected_splits_2, batched_splits_2);
  test::FillValues<int>(&expected_values, batched_values);

  Tensor variant_component_1 = CreateVariantFromRagged<int, int64>(
      {component_split_1_1}, TensorShape({1, 2}), component_values_1);
  Tensor variant_component_2 = CreateVariantFromRagged<int, int64>(
      {component_split_2_1}, TensorShape({2, 2}), component_values_2);
  int input_ragged_rank = 1;
  int output_ragged_rank = 2;
  BuildDecodeRaggedTensorGraph<int, int64>(
      input_ragged_rank, output_ragged_rank, TensorShape({2}),
      {variant_component_1, variant_component_2});

  TF_ASSERT_OK(RunOpKernel());
  test::ExpectTensorEqual<int64>(*GetOutput(0), expected_splits_1);
  test::ExpectTensorEqual<int64>(*GetOutput(1), expected_splits_2);
  test::ExpectTensorEqual<int>(*GetOutput(2), expected_values);
}

TEST_F(RaggedTensorFromVariantKernelTest, NonEmpty1DIn3DOutInt32Splits) {
  // ragged_component_1 = [[x]]
  // ragged_component_2 = [[x], [x]]
  // ragged_component_3 = [[x, x]]
  // ragged_component_4 = [[x, x], [x]]
  // ragged_component_5 = [[x], [x, x]]
  // batched_ragged = [[rc1, rc2, rc3, rc4, rc5], [rc4, rc5, rc1, rc3, rc2]]
  const std::vector<int> component_split_1_1 = {0, 1};
  const std::vector<int> component_split_2_1 = {0, 1, 2};
  const std::vector<int> component_split_3_1 = {0, 2};
  const std::vector<int> component_split_4_1 = {0, 2, 3};
  const std::vector<int> component_split_5_1 = {0, 1, 3};
  const std::vector<int> component_values_1 = {0};
  const std::vector<int> component_values_2 = {0, 1};
  const std::vector<int> component_values_3 = {0, 1};
  const std::vector<int> component_values_4 = {0, 1, 2};
  const std::vector<int> component_values_5 = {0, 1, 2};

  const std::vector<int> batched_splits_1 = {0, 5, 10};
  const std::vector<int> batched_splits_2 = {0,  1,  3,  4,  6, 8,
                                             10, 12, 13, 14, 16};
  const std::vector<int> batched_splits_3 = {0,  1,  2,  3,  5,  7,  8,  9, 11,
                                             13, 14, 15, 17, 18, 20, 21, 22};
  const std::vector<int> batched_values = {0, 0, 1, 0, 1, 0, 1, 2, 0, 1, 2,
                                           0, 1, 2, 0, 1, 2, 0, 0, 1, 0, 1};

  Tensor expected_splits_1(DT_INT32, TensorShape({3}));
  Tensor expected_splits_2(DT_INT32, TensorShape({11}));
  Tensor expected_splits_3(DT_INT32, TensorShape({17}));
  Tensor expected_values(DT_INT32, TensorShape({22}));

  test::FillValues<int>(&expected_splits_1, batched_splits_1);
  test::FillValues<int>(&expected_splits_2, batched_splits_2);
  test::FillValues<int>(&expected_splits_3, batched_splits_3);
  test::FillValues<int>(&expected_values, batched_values);

  Tensor variant_component_1 = CreateVariantFromRagged<int, int>(
      {component_split_1_1}, TensorShape({1}), component_values_1);
  Tensor variant_component_2 = CreateVariantFromRagged<int, int>(
      {component_split_2_1}, TensorShape({2}), component_values_2);
  Tensor variant_component_3 = CreateVariantFromRagged<int, int>(
      {component_split_3_1}, TensorShape({2}), component_values_3);
  Tensor variant_component_4 = CreateVariantFromRagged<int, int>(
      {component_split_4_1}, TensorShape({3}), component_values_4);
  Tensor variant_component_5 = CreateVariantFromRagged<int, int>(
      {component_split_5_1}, TensorShape({3}), component_values_5);
  int input_ragged_rank = 1;
  int output_ragged_rank = 3;
  BuildDecodeRaggedTensorGraph<int, int>(
      input_ragged_rank, output_ragged_rank, TensorShape({2, 5}),
      {variant_component_1, variant_component_2, variant_component_3,
       variant_component_4, variant_component_5, variant_component_4,
       variant_component_5, variant_component_1, variant_component_3,
       variant_component_2});
  TF_ASSERT_OK(RunOpKernel());

  test::ExpectTensorEqual<int>(*GetOutput(0), expected_splits_1);
  test::ExpectTensorEqual<int>(*GetOutput(1), expected_splits_2);
  test::ExpectTensorEqual<int>(*GetOutput(2), expected_splits_3);
  test::ExpectTensorEqual<int>(*GetOutput(3), expected_values);
}

// Tests for invalid inputs.
TEST_F(RaggedTensorFromVariantKernelTest, InvalidInferredInputRaggedRank) {
  Tensor component_variant_1 =
      CreateVariantFromRagged<int, int64>({}, TensorShape({3}), {1, 2, 3});
  Tensor component_variant_2 =
      CreateVariantFromRagged<int, int64>({}, TensorShape({0}), {});
  Tensor component_variant_3 =
      CreateVariantFromRagged<int, int64>({}, TensorShape({2}), {1, 2});
  Tensor component_variant_4 =
      CreateVariantFromRagged<int, int64>({}, TensorShape({1}), {1});

  int input_ragged_rank = -1;
  int output_ragged_rank = 2;
  BuildDecodeRaggedTensorGraph<int, int64>(
      input_ragged_rank, output_ragged_rank, TensorShape({1, 1, 1, 4}),
      {component_variant_1, component_variant_2, component_variant_3,
       component_variant_4});
  EXPECT_TRUE(
      absl::StartsWith(RunOpKernel().error_message(),
                       "Inferred input_ragged_rank (output_ragged_rank - "
                       "encoded_variant.dims()) must be >= 0"));
}

TEST_F(RaggedTensorFromVariantKernelTest, InputDimsAndRaggedRankAttrsMismatch) {
  const std::vector<int64> component_split_1_1 = {0, 1};
  const std::vector<int64> component_split_2_1 = {0, 1, 2};
  const std::vector<int> component_values_1 = {0};
  const std::vector<int> component_values_2 = {0, 1};

  Tensor variant_component_1 = CreateVariantFromRagged<int, int64>(
      {component_split_1_1}, TensorShape({1}), component_values_1);
  Tensor variant_component_2 = CreateVariantFromRagged<int, int64>(
      {component_split_2_1}, TensorShape({2}), component_values_2);

  int input_ragged_rank = 1;
  int output_ragged_rank = 4;
  BuildDecodeRaggedTensorGraph<int, int64>(
      input_ragged_rank, output_ragged_rank, TensorShape({2}),
      {variant_component_1, variant_component_2});
  EXPECT_TRUE(absl::StartsWith(RunOpKernel().error_message(),
                               "output_ragged_rank must be equal to "
                               "input_ragged_rank + encoded_ragged.dims()"));
}

TEST_F(RaggedTensorFromVariantKernelTest, InputDoesNotHoldTensors) {
  int input_ragged_rank = 1;
  int output_ragged_rank = 2;
  BuildDecodeRaggedTensorGraph<int, int64>(
      input_ragged_rank, output_ragged_rank, TensorShape({2}), {1, 2});
  EXPECT_TRUE(absl::StartsWith(
      RunOpKernel().error_message(),
      "Input Variant element at index 0 doesn't hold a Tensor"));
}

TEST_F(RaggedTensorFromVariantKernelTest, InputVariantTensorRankNotOne) {
  Tensor variant_list(DT_VARIANT, TensorShape({2, 1}));
  test::FillValues<Variant>(&variant_list, {1, 2});
  int input_ragged_rank = 1;
  int output_ragged_rank = 2;
  BuildDecodeRaggedTensorGraph<int, int64>(
      input_ragged_rank, output_ragged_rank, TensorShape({1}), {variant_list});
  EXPECT_TRUE(absl::StartsWith(
      RunOpKernel().error_message(),
      "Encoded input Variant must have rank 1, but found rank: 2"));
}

TEST_F(RaggedTensorFromVariantKernelTest,
       InputScalarElementDoesNotMatchInputRaggedRank) {
  const std::vector<int64> component_split_1_1 = {0, 1};
  const std::vector<int> component_values_1 = {1, 2};
  Tensor variant_component_1 = CreateVariantFromRagged<int, int64>(
      {component_split_1_1}, TensorShape({1, 2}), component_values_1);

  int input_ragged_rank = 2;
  int output_ragged_rank = 3;
  BuildDecodeRaggedTensorGraph<int, int64>(input_ragged_rank,
                                           output_ragged_rank, TensorShape({1}),
                                           {variant_component_1});
  EXPECT_TRUE(absl::StartsWith(
      RunOpKernel().error_message(),
      "Encoded input Variant must hold either input_ragged_rank + 1 "
      "Tensors or an empty Tensor"));
}

TEST_F(RaggedTensorFromVariantKernelTest, RaggedSplitNotATensor) {
  Tensor variant_list(DT_VARIANT, TensorShape({2}));
  test::FillValues<Variant>(&variant_list, {1, 2});

  int input_ragged_rank = 1;
  int output_ragged_rank = 2;
  BuildDecodeRaggedTensorGraph<int, int>(input_ragged_rank, output_ragged_rank,
                                         TensorShape({1}), {variant_list});
  EXPECT_TRUE(
      absl::StartsWith(RunOpKernel().error_message(),
                       "Encoded scalar element at index 0 doesn't have a "
                       "splits Tensor at split_index 0"));
}

TEST_F(RaggedTensorFromVariantKernelTest, RaggedSplitTypeMismatch) {
  const std::vector<int64> component_split_1_1 = {0, 1};
  const std::vector<int> component_values_1 = {0};

  Tensor variant_component_1 = CreateVariantFromRagged<int, int64>(
      {component_split_1_1}, TensorShape({1}), component_values_1);

  int input_ragged_rank = 1;
  int output_ragged_rank = 2;
  BuildDecodeRaggedTensorGraph<int, int>(input_ragged_rank, output_ragged_rank,
                                         TensorShape({1}),
                                         {variant_component_1});
  EXPECT_TRUE(absl::StartsWith(RunOpKernel().error_message(),
                               "Expected splits Tensor dtype: 3, found: 9"));
}

TEST_F(RaggedTensorFromVariantKernelTest, RaggedSplitRankNotOne) {
  Tensor splits(DT_INT64, TensorShape({2, 1}));
  test::FillValues<int64>(&splits, {1, 2});
  Tensor values(DT_INT32, {2});
  test::FillValues<int>(&values, {1, 2});
  Tensor encoded_list(DT_VARIANT, TensorShape({2}));
  test::FillValues<Variant>(&encoded_list, {splits, values});

  int input_ragged_rank = 1;
  int output_ragged_rank = 2;
  BuildDecodeRaggedTensorGraph<int, int64>(
      input_ragged_rank, output_ragged_rank, TensorShape({1}), {encoded_list});
  EXPECT_TRUE(absl::StartsWith(RunOpKernel().error_message(),
                               "Ragged splits must have rank 1"));
}

TEST_F(RaggedTensorFromVariantKernelTest, RaggedValuesNotATensor) {
  Tensor splits(DT_INT64, TensorShape({3}));
  test::FillValues<int64>(&splits, {0, 2, 3});
  Tensor variant_list(DT_VARIANT, TensorShape({2}));
  test::FillValues<Variant>(&variant_list, {splits, 2});

  int input_ragged_rank = 1;
  int output_ragged_rank = 2;
  BuildDecodeRaggedTensorGraph<int, int64>(
      input_ragged_rank, output_ragged_rank, TensorShape({1}), {variant_list});
  EXPECT_TRUE(
      absl::StartsWith(RunOpKernel().error_message(),
                       "Encoded scalar element at index 0 doesn't have a "
                       "values Tensor"));
}

TEST_F(RaggedTensorFromVariantKernelTest, RaggedValuesTypeMismatch) {
  const std::vector<int64> component_split_1_1 = {0, 1};
  const std::vector<int> component_values_1 = {0};
  Tensor variant_component_1 = CreateVariantFromRagged<int, int64>(
      {component_split_1_1}, TensorShape({1}), component_values_1);
  int input_ragged_rank = 1;
  int output_ragged_rank = 2;
  BuildDecodeRaggedTensorGraph<tstring, int64>(
      input_ragged_rank, output_ragged_rank, TensorShape({1}),
      {variant_component_1});
  EXPECT_TRUE(
      absl::StartsWith(RunOpKernel().error_message(),
                       "Expected values Tensor dtype: string, found: int32"));
}

TEST_F(RaggedTensorFromVariantKernelTest, RaggedValuesRankNotGreaterThanOne) {
  Tensor variant_component_1 =
      CreateVariantFromRagged<int, int64>({{0, 1}}, TensorShape({}), {1});
  int input_ragged_rank = 1;
  int output_ragged_rank = 2;
  BuildDecodeRaggedTensorGraph<int, int64>(input_ragged_rank,
                                           output_ragged_rank, TensorShape({1}),
                                           {variant_component_1});
  EXPECT_TRUE(absl::StartsWith(RunOpKernel().error_message(),
                               "Ragged values must have rank >= 1"));
}

TEST_F(RaggedTensorFromVariantKernelTest, RaggedValuesRankMismatch) {
  const std::vector<int64> component_split_1_1 = {0, 1};
  const std::vector<int64> component_split_2_1 = {0, 1, 2};
  const std::vector<int> component_values_1 = {0};
  const std::vector<int> component_values_2 = {0, 1, 2, 3};

  Tensor variant_component_1 = CreateVariantFromRagged<int, int64>(
      {component_split_1_1}, TensorShape({1}), component_values_1);
  Tensor variant_component_2 = CreateVariantFromRagged<int, int64>(
      {component_split_2_1}, TensorShape({2, 2}), component_values_2);
  int input_ragged_rank = 1;
  int output_ragged_rank = 2;
  BuildDecodeRaggedTensorGraph<int, int64>(
      input_ragged_rank, output_ragged_rank, TensorShape({2}),
      {variant_component_1, variant_component_2});
  EXPECT_TRUE(absl::StartsWith(RunOpKernel().error_message(),
                               "Rank of values must match for all components"));
}

TEST_F(RaggedTensorFromVariantKernelTest, ShapeFnTest) {
  ShapeInferenceTestOp op("RaggedTensorFromVariant");

  // Tests with input_ragged_rank == 0.
  (*op.node_def.mutable_attr())["input_ragged_rank"].set_i(0);
  (*op.node_def.mutable_attr())["output_ragged_rank"].set_i(1);
  INFER_OK(op, "?", "[?];?");
  INFER_OK(op, "[?]", "[?];?");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[?,?]");

  // Tests with input_ragged_rank == 1.
  (*op.node_def.mutable_attr())["input_ragged_rank"].set_i(1);

  (*op.node_def.mutable_attr())["output_ragged_rank"].set_i(1);
  INFER_OK(op, "?", "[?];?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "[?]");
  INFER_ERROR("Shape must be rank 0 but is rank 2", op, "[?,?]");

  (*op.node_def.mutable_attr())["output_ragged_rank"].set_i(2);
  INFER_OK(op, "?", "[?];[?];?");
  INFER_OK(op, "[?]", "[?];[?];?");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[?,?]");

  (*op.node_def.mutable_attr())["output_ragged_rank"].set_i(3);
  INFER_OK(op, "?", "[?];[?];[?];?");
  INFER_ERROR("Shape must be rank 2 but is rank 1", op, "[?]");
  INFER_OK(op, "[?,?]", "[?];[?];[?];?");
  INFER_ERROR("Shape must be rank 2 but is rank 3", op, "[?,?,?]");

  // Tests with input_ragged_rank == 3.
  (*op.node_def.mutable_attr())["input_ragged_rank"].set_i(3);

  (*op.node_def.mutable_attr())["output_ragged_rank"].set_i(3);
  INFER_OK(op, "?", "[?];[?];[?];?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "[?]");

  (*op.node_def.mutable_attr())["output_ragged_rank"].set_i(4);
  INFER_OK(op, "?", "[?];[?];[?];[?];?");
  INFER_OK(op, "[?]", "[?];[?];[?];[?];?");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[?,?]");

  (*op.node_def.mutable_attr())["output_ragged_rank"].set_i(5);
  INFER_OK(op, "?", "[?];[?];[?];[?];[?];?");
  INFER_ERROR("Shape must be rank 2 but is rank 1", op, "[?]");
  INFER_OK(op, "[?,?]", "[?];[?];[?];[?];[?];?");

  (*op.node_def.mutable_attr())["output_ragged_rank"].set_i(6);
  INFER_OK(op, "?", "[?];[?];[?];[?];[?];[?];?");
  INFER_ERROR("Shape must be rank 3 but is rank 1", op, "[?]");
  INFER_ERROR("Shape must be rank 3 but is rank 2", op, "[?,?]");
  INFER_OK(op, "[?,?,?]", "[?];[?];[?];[?];[?];[?];?");
}

TEST_F(RaggedTensorFromVariantKernelTest, 2DValuesTensorIn1DOut) {
  // [
  //   [
  //     [[x, x], [x, x]],
  //     [[x, x], [x, x]]
  //   ],
  //   [[[x, x], [x, x]]],
  //   [],
  //   [
  //     [[x, x], [x, x]],
  //     [[x, x], [x, x]]
  //   ]
  // ]
  const std::vector<int64> batched_splits_1 = {0, 2, 3, 3, 5};
  const std::vector<int> batched_values = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3,
                                           3, 3, 4, 4, 4, 4, 5, 5, 5, 5};

  Tensor variant_component_1 = CreateVariantFromRagged<int, int64>(
      {}, TensorShape({2, 2, 2}), {1, 1, 1, 1, 2, 2, 2, 2});
  Tensor variant_component_2 = CreateVariantFromRagged<int, int64>(
      {}, TensorShape({1, 2, 2}), {3, 3, 3, 3});
  Tensor variant_component_3 =
      CreateVariantFromRagged<int, int64>({}, TensorShape({0, 2, 2}), {});
  Tensor variant_component_4 = CreateVariantFromRagged<int, int64>(
      {}, TensorShape({2, 2, 2}), {4, 4, 4, 4, 5, 5, 5, 5});

  Tensor expected_splits_1(DT_INT64, TensorShape({5}));
  Tensor expected_values(DT_INT32, TensorShape({5, 2, 2}));
  test::FillValues<int64>(&expected_splits_1, batched_splits_1);
  test::FillValues<int>(&expected_values, batched_values);

  int input_ragged_rank = 0;
  int output_ragged_rank = 1;
  BuildDecodeRaggedTensorGraph<int, int64>(
      input_ragged_rank, output_ragged_rank, TensorShape({4}),
      {variant_component_1, variant_component_2, variant_component_3,
       variant_component_4});
  TF_ASSERT_OK(RunOpKernel());

  test::ExpectTensorEqual<int64>(*GetOutput(0), expected_splits_1);
  test::ExpectTensorEqual<int>(*GetOutput(1), expected_values);
}
}  // namespace
}  // namespace tensorflow
