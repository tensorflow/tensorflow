// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(RoundRobinGenerateMasksOpTest, ShapeFn) {
  // RoundRobinTrimmer(max_sequence_length, N * input_values, N * row_splits)
  //     -> [N * masks]
  ShapeInferenceTestOp op("TFText>RoundRobinGenerateMasks");
  auto set_op = [&op](int n) {
    std::vector<NodeDefBuilder::NodeOut> inputs, input_splits;
    for (int i = 0; i < n; i++) inputs.emplace_back("a", 0, DT_INT32);
    for (int i = 0; i < n; i++) input_splits.emplace_back("b", 0, DT_INT64);
    TF_ASSERT_OK(NodeDefBuilder("test", "TFText>RoundRobinGenerateMasks")
                     .Input({"max_seq", 0, DT_INT32})
                     .Input(inputs)
                     .Input(input_splits)
                     .Attr("N", n)
                     .Attr("Tsplits", DT_INT64)
                     .Finalize(&op.node_def));
  };
  set_op(1);
  INFER_OK(op, "?;[?];[?]", "[?]");
  INFER_OK(op, "[];[?];[?]", "[?]");
  INFER_OK(op, "?;[3];[?]", "[3]");
  INFER_OK(op, "?;[?];[2]", "[?]");
  INFER_ERROR("Shape must be rank 1", op, "?;[];[?]");
  INFER_ERROR("Shape must be rank 1", op, "?;[1,2];[?]");
  INFER_ERROR("Shape must be rank 1", op, "?;[?,?];[?]");
  INFER_ERROR("Shape must be rank 1", op, "?;[?];[]");
  INFER_ERROR("Shape must be rank 1", op, "?;[?];[1,2]");
  INFER_ERROR("Shape must be rank 1", op, "?;[?];[?,?]");
  INFER_ERROR("Shape must be a scalar", op, "[?];[?];[?]");
  set_op(2);
  INFER_OK(op, "?;[?];[?];[?];[?]", "[?];[?]");
  INFER_OK(op, "?;[3];[2];[?];[?]", "[3];[2]");
  INFER_OK(op, "?;[?];[?];[5];[6]", "[?];[?]");
  INFER_ERROR("Shape must be rank 1", op, "?;[];[?];[?];[?]");
  INFER_ERROR("Shape must be rank 1", op, "?;[?];[?,?];[?];[?]");
  INFER_ERROR("Shape must be rank 1", op, "?;[?];[?];[?,?];[?]");
  INFER_ERROR("Shape must be rank 1", op, "?;[?];[?];[?,?];[]");
  INFER_ERROR("Wrong number of inputs passed", op, "?;[?];[?]");
  set_op(0);
  INFER_ERROR("Shape inference should have returned error", op, "?");
}

TEST(RoundRobinTrimOpTest, ShapeFn) {
  // RoundRobinTrimmer(max_sequence_length, N * input_values, N * row_splits)
  //     -> [N * output_values, N * row_splits]
  ShapeInferenceTestOp op("TFText>RoundRobinTrim");
  auto set_op = [&op](int n) {
    std::vector<NodeDefBuilder::NodeOut> inputs, input_splits;
    for (int i = 0; i < n; i++) inputs.emplace_back("a", 0, DT_INT32);
    for (int i = 0; i < n; i++) input_splits.emplace_back("b", 0, DT_INT64);
    TF_ASSERT_OK(NodeDefBuilder("test", "TFText>RoundRobinTrim")
                     .Input({"max_seq", 0, DT_INT32})
                     .Input(inputs)
                     .Input(input_splits)
                     .Attr("N", n)
                     .Attr("Tsplits", DT_INT64)
                     .Finalize(&op.node_def));
  };
  set_op(1);
  INFER_OK(op, "?;[?];[?]", "[?];[?]");
  INFER_OK(op, "[];[?];[?]", "[?];[?]");
  INFER_OK(op, "?;[3];[?]", "[?];[?]");
  INFER_OK(op, "?;[?];[2]", "[?];[2]");
  INFER_ERROR("Shape must be rank 1", op, "?;[];[?]");
  INFER_ERROR("Shape must be rank 1", op, "?;[1,2];[?]");
  INFER_ERROR("Shape must be rank 1", op, "?;[?,?];[?]");
  INFER_ERROR("Shape must be rank 1", op, "?;[?];[]");
  INFER_ERROR("Shape must be rank 1", op, "?;[?];[1,2]");
  INFER_ERROR("Shape must be rank 1", op, "?;[?];[?,?]");
  INFER_ERROR("Shape must be a scalar", op, "[?];[?];[?]");
  set_op(2);
  INFER_OK(op, "?;[?];[?];[?];[?]", "[?];[?];[?];[?]");
  INFER_OK(op, "?;[3];[2];[?];[?]", "[?];[?];[?];[?]");
  INFER_OK(op, "?;[?];[?];[5];[6]", "[?];[?];[5];[6]");
  INFER_ERROR("Shape must be rank 1", op, "?;[];[?];[?];[?]");
  INFER_ERROR("Shape must be rank 1", op, "?;[?];[?,?];[?];[?]");
  INFER_ERROR("Shape must be rank 1", op, "?;[?];[?];[?,?];[?]");
  INFER_ERROR("Shape must be rank 1", op, "?;[?];[?];[?,?];[]");
  INFER_ERROR("Wrong number of inputs passed", op, "?;[?];[?]");
  set_op(0);
  INFER_ERROR("Shape inference should have returned error", op, "?");
}

}  // namespace
}  // namespace tensorflow
