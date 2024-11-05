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

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/graph_tools.h"
#include "tensorflow/lite/experimental/litert/test/common.h"

namespace {

TEST(Model, SimpleModel) {
  auto model = litert::testing::LoadTestFileModel("one_mul.tflite");

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

}  // namespace
