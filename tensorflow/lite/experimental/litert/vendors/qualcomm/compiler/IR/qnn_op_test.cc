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

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_op.h"

#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model_predicates.h"
#include "tensorflow/lite/experimental/litert/test/common.h"

namespace {

TEST(TestInitQnnOp, BuildDefaultOp) {
  Qnn_OpConfig_t op = litert::qnn::BuildDefaultOp();
  ASSERT_EQ(op.version, QNN_OPCONFIG_VERSION_1);
}

TEST(TestLegalizeOp, SimpleSupportedOp) {
  auto model = litert::testing::LoadTestFileModel("one_mul.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);
  auto ops = subgraph->Ops();

  Qnn_OpConfig_t qnn_op = litert::qnn::BuildDefaultOp();
  LITERT_ASSERT_STATUS_OK(litert::qnn::LegalizeOp(ops.front().Get(), qnn_op));

  EXPECT_TRUE(absl::StrContains(qnn_op.v1.name, "mul"));
  EXPECT_STREQ(qnn_op.v1.packageName, "qti.aisw");
  EXPECT_STREQ(qnn_op.v1.typeName, "ElementWiseMultiply");

  EXPECT_EQ(qnn_op.v1.numOfInputs, 0);
  EXPECT_EQ(qnn_op.v1.numOfOutputs, 0);
  EXPECT_EQ(qnn_op.v1.numOfParams, 0);

  litert::qnn::ResetOp(qnn_op);
}

TEST(TestLegalizeOp, UnsupportedOp) {
  auto model = litert::testing::LoadTestFileModel("simple_floor_mod_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);
  auto ops = subgraph->Ops();

  Qnn_OpConfig_t qnn_op = litert::qnn::BuildDefaultOp();
  LITERT_ASSERT_STATUS_HAS_CODE(
      litert::qnn::LegalizeOp(ops.front().Get(), qnn_op),
      kLiteRtStatusErrorUnsupported);

  litert::qnn::ResetOp(qnn_op);
}

}  // namespace
