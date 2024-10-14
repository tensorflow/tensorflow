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

#include "tensorflow/lite/experimental/lrt/cc/lite_rt_op.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/lrt/c/lite_rt_op_code.h"
#include "tensorflow/lite/experimental/lrt/core/graph_tools.h"
#include "tensorflow/lite/experimental/lrt/test/common.h"

namespace {

using ::lrt::LrtOpManager;

TEST(TestLrtOp, SimpleSupportedOp) {
  auto model = lrt::testing::LoadTestFileModel("one_mul.tflite");
  ASSERT_RESULT_OK_ASSIGN(auto subgraph,
                          ::graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, ::graph_tools::GetSubgraphOps(subgraph));

  LrtOpManager::Unique op;
  ASSERT_STATUS_OK(LrtOpManager::MakeFromOp(ops[0], op));

  EXPECT_EQ(op->Code(), kLrtOpCodeTflMul);
  EXPECT_EQ(op->Op(), ops[0]);
}

}  // namespace
