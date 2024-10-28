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
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/graph_tools.h"
#include "tensorflow/lite/experimental/litert/test/common.h"

namespace {

TEST(Subgraph, SimpleModel) {
  auto litert_model = litert::testing::LoadTestFileModel("one_mul.tflite");

  ASSERT_RESULT_OK_ASSIGN(auto litert_subgraph,
                          ::graph_tools::GetSubgraph(litert_model.get()));

  litert::Subgraph subgraph(litert_subgraph);
  ASSERT_EQ(subgraph.Inputs().size(), 2);
  ASSERT_EQ(subgraph.Outputs().size(), 1);
  ASSERT_EQ(subgraph.Ops().size(), 1);
}

}  // namespace
