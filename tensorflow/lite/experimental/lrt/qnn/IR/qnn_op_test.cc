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

#include "tensorflow/lite/experimental/lrt/qnn/IR/qnn_op.h"

#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "third_party/qairt/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/core/graph_tools.h"
#include "tensorflow/lite/experimental/lrt/test/common.h"

namespace {

TEST(TestInitQnnOp, BuildDefaultOp) {
  Qnn_OpConfig_t op = qnn::BuildDefaultOp();
  ASSERT_EQ(op.version, QNN_OPCONFIG_VERSION_1);
}

TEST(TestLegalizeOp, SimpleSupportedOp) {
  auto model = lrt::testing::LoadTestFileModel("one_mul.tflite");
  ASSERT_RESULT_OK_ASSIGN(auto subgraph,
                          ::graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, ::graph_tools::GetSubgraphOps(subgraph));

  Qnn_OpConfig_t qnn_op = qnn::BuildDefaultOp();
  ASSERT_STATUS_OK(qnn::LegalizeOp(ops[0], qnn_op));

  EXPECT_TRUE(absl::StrContains(qnn_op.v1.name, "mul"));
  EXPECT_STREQ(qnn_op.v1.packageName, "qti.aisw");
  EXPECT_STREQ(qnn_op.v1.typeName, "ElementWiseMultiply");

  EXPECT_EQ(qnn_op.v1.numOfInputs, 0);
  EXPECT_EQ(qnn_op.v1.numOfOutputs, 0);
  EXPECT_EQ(qnn_op.v1.numOfParams, 0);

  qnn::ResetOp(qnn_op);
}

TEST(TestLegalizeOp, UnsupportedOp) {
  auto model = lrt::testing::LoadTestFileModel("simple_floor_mod_op.tflite");
  ASSERT_RESULT_OK_ASSIGN(auto subgraph,
                          ::graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, ::graph_tools::GetSubgraphOps(subgraph));

  Qnn_OpConfig_t qnn_op = qnn::BuildDefaultOp();
  ASSERT_STATUS_HAS_CODE(qnn::LegalizeOp(ops[0], qnn_op),
                         kLrtStatusErrorUnsupported);

  qnn::ResetOp(qnn_op);
}

}  // namespace
