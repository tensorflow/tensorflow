/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tsl/platform/bfloat16.h"

namespace tensorflow {

namespace {

absl::Status RunGraph(const Graph& graph,
                      const std::vector<std::string>& output_tensor_names,
                      const std::vector<std::string>& target_tensor_names,
                      std::vector<Tensor>* output_tensors) {
  GraphDef graph_def;
  graph.ToGraphDef(&graph_def);
  SessionOptions session_options;
  std::unique_ptr<Session> session(NewSession(session_options));
  TF_RETURN_IF_ERROR(session->Create(std::move(graph_def)));
  RunOptions run_options;
  return session->Run(run_options, /*inputs=*/{}, output_tensor_names,
                      target_tensor_names, output_tensors,
                      /*run_metadata=*/nullptr);
}

TEST(TPUDummyInputOpTest, Basic) {
  Graph graph(OpRegistry::Global());
  Node* tpu_dummy_input = nullptr;
  DataType data_type = DT_FLOAT;
  const TensorShape input_shape({2, 2});
  TF_ASSERT_OK(NodeBuilder(graph.NewName("tpu_dummy_input"), "TPUDummyInput")
                   .Attr("dtype", data_type)
                   .Attr("shape", input_shape)
                   .Finalize(&graph, &tpu_dummy_input));

  std::vector<Tensor> output_tensors;
  TF_EXPECT_OK(RunGraph(graph,
                        /*output_tensor_names=*/{tpu_dummy_input->name()},
                        /*target_tensor_names=*/{}, &output_tensors));

  ASSERT_EQ(output_tensors.size(), 1);
  test::ExpectTensorEqual<float>(
      output_tensors[0],
      test::AsTensor<float>({0.0, 0.0, 0.0, 0.0}, TensorShape({2, 2})));
}

TEST(TPUDummyInputOpTest, BasicBfloat16) {
  Graph graph(OpRegistry::Global());
  Node* tpu_dummy_input = nullptr;
  DataType data_type = DT_BFLOAT16;
  const TensorShape input_shape({2, 2});
  TF_ASSERT_OK(NodeBuilder(graph.NewName("tpu_dummy_input"), "TPUDummyInput")
                   .Attr("dtype", data_type)
                   .Attr("shape", input_shape)
                   .Finalize(&graph, &tpu_dummy_input));

  std::vector<Tensor> output_tensors;
  TF_EXPECT_OK(RunGraph(graph,
                        /*output_tensor_names=*/{tpu_dummy_input->name()},
                        /*target_tensor_names=*/{}, &output_tensors));

  ASSERT_EQ(output_tensors.size(), 1);
  test::ExpectTensorEqual<tsl::bfloat16>(
      output_tensors[0],
      test::AsTensor<tsl::bfloat16>({tsl::bfloat16(0.0), tsl::bfloat16(0.0),
                                     tsl::bfloat16(0.0), tsl::bfloat16(0.0)},
                                    TensorShape({2, 2})));
}

}  // namespace

}  // namespace tensorflow
