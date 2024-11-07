// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/cc/litert_model.h"

#include <cstdint>
#include <optional>

#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/core/graph_tools.h"
#include "tensorflow/lite/experimental/litert/test/common.h"

// Tests for CC Wrapper classes around public C api.

namespace litert {
using ::litert::internal::GetSubgraph;
using ::litert::internal::GetSubgraphOps;

namespace {

static constexpr const int32_t kTensorDimensions[] = {1, 2, 3};

static constexpr const auto kRank =
    sizeof(kTensorDimensions) / sizeof(kTensorDimensions[0]);

static constexpr const uint32_t kTensorStrides[] = {6, 3, 1};

static constexpr const LiteRtLayout kLayout = {
    /*.rank=*/kRank,
    /*.dimensions=*/kTensorDimensions,
    /*.strides=*/nullptr,
};

static constexpr const LiteRtRankedTensorType kTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    /*.layout=*/kLayout,
};

//===----------------------------------------------------------------------===//
//                                CC Model                                    //
//===----------------------------------------------------------------------===//

TEST(CcModelTest, SimpleModel) {
  auto model = testing::LoadTestFileModel("one_mul.tflite");

  LiteRtParamIndex num_subgraphs;
  ASSERT_EQ(LiteRtGetNumModelSubgraphs(model.Get(), &num_subgraphs),
            kLiteRtStatusOk);
  EXPECT_EQ(model.NumSubgraphs(), num_subgraphs);
  EXPECT_EQ(model.NumSubgraphs(), 1);

  LiteRtParamIndex main_subgraph_index;
  ASSERT_EQ(LiteRtGetMainModelSubgraphIndex(model.Get(), &main_subgraph_index),
            kLiteRtStatusOk);
  EXPECT_EQ(main_subgraph_index, 0);

  LiteRtSubgraph litert_subgraph_0;
  ASSERT_EQ(LiteRtGetModelSubgraph(model.Get(), /*subgraph_index=*/0,
                                   &litert_subgraph_0),
            kLiteRtStatusOk);

  auto subgraph_0 = model.Subgraph(0);
  ASSERT_TRUE(subgraph_0.ok());
  EXPECT_EQ(subgraph_0->Get(), litert_subgraph_0);

  auto main_subgraph = model.MainSubgraph();
  EXPECT_EQ(main_subgraph->Get(), subgraph_0->Get());
}

//===----------------------------------------------------------------------===//
//                                CC Layout                                   //
//===----------------------------------------------------------------------===//

TEST(CcLayoutTest, NoStrides) {
  constexpr const LiteRtLayout kLayout = {
      /*.rank=*/kRank,
      /*.dimensions=*/kTensorDimensions,
      /*.strides=*/nullptr,
  };

  Layout layout(kLayout);

  ASSERT_EQ(layout.Rank(), kLayout.rank);
  for (auto i = 0; i < layout.Rank(); ++i) {
    ASSERT_EQ(layout.Dimensions()[i], kLayout.dimensions[i]);
  }
  ASSERT_FALSE(layout.HasStrides());
}

TEST(CcLayoutTest, WithStrides) {
  constexpr const LiteRtLayout kLayout = {
      /*.rank=*/kRank,
      /*.dimensions=*/kTensorDimensions,
      /*.strides=*/kTensorStrides,
  };

  Layout layout(kLayout);

  ASSERT_EQ(layout.Rank(), kLayout.rank);
  for (auto i = 0; i < layout.Rank(); ++i) {
    ASSERT_EQ(layout.Dimensions()[i], kLayout.dimensions[i]);
  }
  ASSERT_TRUE(layout.HasStrides());
  for (auto i = 0; i < layout.Rank(); ++i) {
    ASSERT_EQ(layout.Strides()[i], kLayout.strides[i]);
  }
}

TEST(CcLayoutTest, Equal) {
  Layout layout1({
      /*.rank=*/kRank,
      /*.dimensions=*/kTensorDimensions,
      /*.strides=*/kTensorStrides,
  });
  Layout layout2({
      /*.rank=*/kRank,
      /*.dimensions=*/kTensorDimensions,
      /*.strides=*/kTensorStrides,
  });
  ASSERT_TRUE(layout1 == layout2);
}

TEST(CcLayoutTest, NotEqual) {
  Layout layout1({
      /*.rank=*/kRank,
      /*.dimensions=*/kTensorDimensions,
      /*.strides=*/nullptr,
  });
  Layout layout2({
      /*.rank=*/kRank,
      /*.dimensions=*/kTensorDimensions,
      /*.strides=*/kTensorStrides,
  });
  ASSERT_FALSE(layout1 == layout2);
}

//===----------------------------------------------------------------------===//
//                                CC Op                                       //
//===----------------------------------------------------------------------===//

TEST(CcOpTest, SimpleSupportedOp) {
  auto litert_model = testing::LoadTestFileModel("one_mul.tflite");
  ASSERT_RESULT_OK_ASSIGN(auto litert_subgraph,
                          GetSubgraph(litert_model.Get()));
  ASSERT_RESULT_OK_ASSIGN(auto litert_ops, GetSubgraphOps(litert_subgraph));

  Op op(litert_ops[0]);
  EXPECT_EQ(op.Code(), kLiteRtOpCodeTflMul);
  EXPECT_EQ(op.Get(), litert_ops[0]);
}

//===----------------------------------------------------------------------===//
//                           CC RankedTensorType                              //
//===----------------------------------------------------------------------===//

TEST(RankedTensorType, Accessors) {
  Layout layout(kLayout);
  RankedTensorType tensor_type(kTensorType);
  ASSERT_EQ(tensor_type.ElementType(),
            static_cast<ElementType>(kTensorType.element_type));
  ASSERT_TRUE(tensor_type.Layout() == layout);
}

//===----------------------------------------------------------------------===//
//                                CC Tensor                                   //
//===----------------------------------------------------------------------===//

TEST(Tensor, SimpleModel) {
  auto litert_model = testing::LoadTestFileModel("one_mul.tflite");

  ASSERT_RESULT_OK_ASSIGN(auto litert_subgraph,
                          GetSubgraph(litert_model.Get()));

  Subgraph subgraph(litert_subgraph);

  auto inputs = subgraph.Inputs();
  ASSERT_EQ(inputs.size(), 2);

  Tensor input_tensor(inputs[0]);
  ASSERT_EQ(input_tensor.TypeId(), kLiteRtRankedTensorType);

  auto input_ranked_tensor_type = input_tensor.RankedTensorType();
  ASSERT_EQ(input_ranked_tensor_type.ElementType(), ElementType::Float32);

  EXPECT_FALSE(input_tensor.HasWeights());

  auto input_weights = input_tensor.Weights();
  ASSERT_EQ(input_weights.Bytes().size(), 0);

  ASSERT_EQ(input_tensor.DefiningOp(), std::nullopt);

  absl::Span<LiteRtOp> input_uses;
  absl::Span<LiteRtParamIndex> input_user_arg_indices;
  input_tensor.Uses(input_uses, input_user_arg_indices);
  ASSERT_EQ(input_uses.size(), 1);
  ASSERT_EQ(input_user_arg_indices.size(), 1);

  auto outputs = subgraph.Outputs();
  ASSERT_EQ(outputs.size(), 1);

  Tensor output_tensor(outputs[0]);
  ASSERT_EQ(output_tensor.TypeId(), kLiteRtRankedTensorType);

  auto output_defining_op = output_tensor.DefiningOp();
  EXPECT_TRUE(output_defining_op.has_value());

  absl::Span<LiteRtOp> output_uses;
  absl::Span<LiteRtParamIndex> output_user_arg_indices;
  output_tensor.Uses(output_uses, output_user_arg_indices);
  ASSERT_EQ(output_uses.size(), 0);
  ASSERT_EQ(output_user_arg_indices.size(), 0);
}

//===----------------------------------------------------------------------===//
//                               CC Subgraph                                  //
//===----------------------------------------------------------------------===//

TEST(CcSubgraphTest, SimpleModel) {
  auto litert_model = litert::testing::LoadTestFileModel("one_mul.tflite");

  ASSERT_RESULT_OK_ASSIGN(auto litert_subgraph,
                          GetSubgraph(litert_model.Get()));

  litert::Subgraph subgraph(litert_subgraph);
  ASSERT_EQ(subgraph.Inputs().size(), 2);
  ASSERT_EQ(subgraph.Outputs().size(), 1);
  ASSERT_EQ(subgraph.Ops().size(), 1);
}

}  // namespace
}  // namespace litert
