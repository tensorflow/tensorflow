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

#include <cstddef>
#include <cstdint>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_element_type.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/test/common.h"

// Tests for CC Wrapper classes around public C api.

namespace litert {

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
  ASSERT_TRUE(subgraph_0);
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

TEST(CcLayoutTest, NumElements) {
  Layout layout({2, 2, 3});
  auto num_elements = layout.NumElements();
  ASSERT_TRUE(num_elements.has_value());
  EXPECT_EQ(num_elements.value(), 12);
}

//===----------------------------------------------------------------------===//
//                                CC Op                                       //
//===----------------------------------------------------------------------===//

TEST(CcOpTest, SimpleSupportedOp) {
  auto litert_model = testing::LoadTestFileModel("one_mul.tflite");
  auto subgraph = litert_model.MainSubgraph();
  const auto ops = subgraph->Ops();
  const auto& op = ops.front();

  EXPECT_EQ(op.Code(), kLiteRtOpCodeTflMul);
  EXPECT_EQ(op.Inputs().size(), 2);
  EXPECT_EQ(op.Outputs().size(), 1);
}

//===----------------------------------------------------------------------===//
//                           CC RankedTensorType                              //
//===----------------------------------------------------------------------===//

TEST(CcRankedTensorTypeTest, Accessors) {
  Layout layout(kLayout);
  RankedTensorType tensor_type(kTensorType);
  ASSERT_EQ(tensor_type.ElementType(),
            static_cast<ElementType>(kTensorType.element_type));
  ASSERT_TRUE(tensor_type.Layout() == layout);
}

//===----------------------------------------------------------------------===//
//                                CC Tensor                                   //
//===----------------------------------------------------------------------===//

TEST(CcTensorTest, SimpleModel) {
  auto litert_model = testing::LoadTestFileModel("one_mul.tflite");
  auto subgraph = litert_model.MainSubgraph();

  auto inputs = subgraph->Inputs();
  ASSERT_EQ(inputs.size(), 2);

  {
    const Tensor& input_tensor = inputs.front();
    ASSERT_EQ(input_tensor.TypeId(), kLiteRtRankedTensorType);

    auto input_ranked_tensor_type = input_tensor.RankedTensorType();
    ASSERT_EQ(input_ranked_tensor_type.ElementType(), ElementType::Float32);

    EXPECT_FALSE(input_tensor.HasWeights());

    auto input_weights = input_tensor.Weights();
    ASSERT_EQ(input_weights.Bytes().size(), 0);

    ASSERT_EQ(input_tensor.DefiningOp(), std::nullopt);

    const auto uses = input_tensor.Uses();
    ASSERT_EQ(uses.size(), 1);
  }

  auto outputs = subgraph->Outputs();
  ASSERT_EQ(outputs.size(), 1);

  {
    const Tensor& output_tensor = outputs.front();
    ASSERT_EQ(output_tensor.TypeId(), kLiteRtRankedTensorType);

    auto output_defining_op = output_tensor.DefiningOp();
    EXPECT_TRUE(output_defining_op.has_value());

    ASSERT_TRUE(output_tensor.Uses().empty());
  }
}

TEST(CcTensorTest, WeightsData) {
  auto litert_model = testing::LoadTestFileModel("add_cst.tflite");
  auto subgraph = litert_model.MainSubgraph();

  auto data = subgraph->Ops().front().Inputs().back().WeightsData<float>();
  ASSERT_TRUE(data.HasValue());
  EXPECT_THAT(data.Value(), ::testing::ElementsAreArray({1.0, 2.0, 3.0, 4.0}));
}

TEST(CcTensorTest, Name) {
  static constexpr absl::string_view kName = "foo";
  LiteRtTensorT tensor;
  tensor.name = kName;

  Tensor cc_tensor(&tensor);
  EXPECT_EQ(cc_tensor.Name(), kName);
}

TEST(CcTensorTest, QuantizationNone) {
  LiteRtTensorT litert_tensor;
  litert_tensor.q_type_id = kLiteRtQuantizationNone;

  Tensor tensor(&litert_tensor);
  EXPECT_EQ(tensor.QTypeId(), kLiteRtQuantizationNone);
  EXPECT_FALSE(tensor.HasQuantization());
}

TEST(CcTensorTest, QuantizationPerTensor) {
  static constexpr auto kScale = 1.0;
  static constexpr auto kZeroPoint = 1;

  LiteRtTensorT litert_tensor;
  litert_tensor.q_type_id = kLiteRtQuantizationPerTensor;
  litert_tensor.q_type_detail.per_tensor = {kScale, kZeroPoint};

  Tensor tensor(&litert_tensor);
  ASSERT_EQ(tensor.QTypeId(), kLiteRtQuantizationPerTensor);
  ASSERT_TRUE(tensor.HasQuantization());

  const auto per_tensor_quantization = tensor.PerTensorQuantization();
  EXPECT_EQ(per_tensor_quantization.scale, kScale);
  EXPECT_EQ(per_tensor_quantization.zero_point, kZeroPoint);
}

//===----------------------------------------------------------------------===//
//                               CC Subgraph                                  //
//===----------------------------------------------------------------------===//

TEST(CcSubgraphTest, SimpleModel) {
  auto model = testing::LoadTestFileModel("one_mul.tflite");
  auto subgraph = model.MainSubgraph();

  ASSERT_EQ(subgraph->Inputs().size(), 2);
  ASSERT_EQ(subgraph->Outputs().size(), 1);
  ASSERT_EQ(subgraph->Ops().size(), 1);
}

//===----------------------------------------------------------------------===//
//                               CC ElementType                               //
//===----------------------------------------------------------------------===//

TEST(CcElementTypeTest, GetByteWidth) {
  const size_t width = GetByteWidth<ElementType::Bool>();
  EXPECT_EQ(width, 1);
}

TEST(CcElementTypeTest, GetElementType) {
  ElementType ty = GetElementType<float>();
  EXPECT_EQ(ty, ElementType::Float32);
}

}  // namespace
}  // namespace litert
