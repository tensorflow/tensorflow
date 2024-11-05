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
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/graph_tools.h"
#include "tensorflow/lite/experimental/litert/test/common.h"

namespace {

TEST(Op, SimpleSupportedOp) {
  auto model = litert::testing::LoadTestFileModel("one_mul.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph.ok());

  auto ops = subgraph->Ops();

  auto op = litert::Op(ops[0]);
  EXPECT_EQ(op.Code(), kLiteRtOpCodeTflMul);
  EXPECT_EQ(op.Get(), ops[0]);
}

}  // namespace
