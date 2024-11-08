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

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_tensor.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/core/graph_tools.h"
#include "tensorflow/lite/experimental/litert/test/common.h"

namespace {

TEST(TestInitQnnTensor, BuildDefaultTensor) {
  Qnn_Tensor_t tensor = litert::qnn::BuildDefaultTensor();
  ASSERT_EQ(tensor.version, QNN_TENSOR_VERSION_2);
  EXPECT_EQ(tensor.v2.dataFormat, QNN_TENSOR_DATA_FORMAT_DENSE);
  EXPECT_EQ(tensor.v2.rank, 0);
  EXPECT_EQ(tensor.v2.dimensions, nullptr);
  EXPECT_EQ(tensor.v2.id, 0);
}

TEST(TestInitQnnTensor, BuildDefaultTensorWithId) {
  Qnn_Tensor_t tensor = litert::qnn::BuildDefaultTensor(2);
  ASSERT_EQ(tensor.version, QNN_TENSOR_VERSION_2);
  EXPECT_EQ(tensor.v2.dataFormat, QNN_TENSOR_DATA_FORMAT_DENSE);
  EXPECT_EQ(tensor.v2.rank, 0);
  EXPECT_EQ(tensor.v2.dimensions, nullptr);
  EXPECT_EQ(tensor.v2.id, 2);
}

TEST(TestInitQnnTensor, BuildDefaultInputTensor) {
  Qnn_Tensor_t tensor = litert::qnn::BuildInputTensor();
  ASSERT_EQ(tensor.version, QNN_TENSOR_VERSION_2);
  EXPECT_EQ(tensor.v2.type, QNN_TENSOR_TYPE_APP_WRITE);
  EXPECT_EQ(tensor.v2.memType, QNN_TENSORMEMTYPE_RAW);
  EXPECT_EQ(tensor.v2.clientBuf.dataSize, 0);
}

TEST(TestInitQnnTensor, SetInputTensor) {
  Qnn_Tensor_t tensor = litert::qnn::BuildDefaultTensor();
  litert::qnn::SetInputTensorAttrs(tensor);
  ASSERT_EQ(tensor.version, QNN_TENSOR_VERSION_2);
  EXPECT_EQ(tensor.v2.type, QNN_TENSOR_TYPE_APP_WRITE);
  EXPECT_EQ(tensor.v2.memType, QNN_TENSORMEMTYPE_RAW);
  EXPECT_EQ(tensor.v2.clientBuf.dataSize, 0);
}

TEST(TestInitQnnTensor, BuildDefaultOutputTensor) {
  Qnn_Tensor_t tensor = litert::qnn::BuildOutputTensor();
  ASSERT_EQ(tensor.version, QNN_TENSOR_VERSION_2);
  EXPECT_EQ(tensor.v2.type, QNN_TENSOR_TYPE_APP_READ);
}

TEST(TestInitQnnTensor, SetOutputTensor) {
  Qnn_Tensor_t tensor = litert::qnn::BuildDefaultTensor();
  litert::qnn::SetOutputTensorAttrs(tensor);
  ASSERT_EQ(tensor.version, QNN_TENSOR_VERSION_2);
  EXPECT_EQ(tensor.v2.type, QNN_TENSOR_TYPE_APP_READ);
}

TEST(TestInitQnnTensor, MoveToId) {
  Qnn_Tensor_t tensor = litert::qnn::BuildDefaultTensor(2);

  litert::qnn::SetOutputTensorAttrs(tensor);
  ASSERT_EQ(tensor.version, QNN_TENSOR_VERSION_2);
  EXPECT_EQ(tensor.v2.type, QNN_TENSOR_TYPE_APP_READ);

  EXPECT_EQ(litert::qnn::MoveToId(tensor), 2);
  EXPECT_EQ(tensor.v2.id, 2);
  EXPECT_EQ(tensor.v2.type, QNN_TENSOR_TYPE_UNDEFINED);
}

TEST(TestLegalizeTensor, SimpleSupportedTensorSubgraphInput) {
  auto model = litert::testing::LoadTestFileModel("one_mul.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph.ok());
  auto outputs = subgraph->Outputs();

  auto qnn_tensor = litert::qnn::BuildDefaultTensor();
  auto output_tensor = litert::Tensor(outputs[0]);
  LITERT_ASSERT_STATUS_OK(
      litert::qnn::LegalizeTensor(output_tensor, qnn_tensor));

  ASSERT_EQ(qnn_tensor.version, QNN_TENSOR_VERSION_2);
  EXPECT_EQ(qnn_tensor.v2.dataType, QNN_DATATYPE_FLOAT_32);
  EXPECT_EQ(qnn_tensor.v2.type, QNN_TENSOR_TYPE_APP_READ);

  ASSERT_EQ(qnn_tensor.v2.rank, 2);
  ASSERT_NE(qnn_tensor.v2.dimensions, nullptr);
  EXPECT_THAT(absl::MakeConstSpan(qnn_tensor.v2.dimensions, 2),
              ::testing::ElementsAreArray({2, 2}));

  litert::qnn::ResetTensor(qnn_tensor);
}

TEST(TestLegalizeTensor, SimpleSupportedTensor) {
  auto model = litert::testing::LoadTestFileModel("simple_multi_op.tflite");

  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph.ok());
  auto ops = subgraph->Ops();
  auto op_outs = litert::Op(ops[1]).Outputs();

  auto qnn_tensor = litert::qnn::BuildDefaultTensor();
  auto op_out = litert::Tensor(op_outs[0]);
  LITERT_ASSERT_STATUS_OK(litert::qnn::LegalizeTensor(op_out, qnn_tensor));

  ASSERT_EQ(qnn_tensor.version, QNN_TENSOR_VERSION_2);
  EXPECT_EQ(qnn_tensor.v2.dataType, QNN_DATATYPE_FLOAT_32);
  EXPECT_EQ(qnn_tensor.v2.type, QNN_TENSOR_TYPE_NATIVE);

  ASSERT_EQ(qnn_tensor.v2.rank, 2);
  ASSERT_NE(qnn_tensor.v2.dimensions, nullptr);
  EXPECT_THAT(absl::MakeConstSpan(qnn_tensor.v2.dimensions, 2),
              ::testing::ElementsAreArray({2, 2}));

  litert::qnn::ResetTensor(qnn_tensor);
}

}  // namespace
