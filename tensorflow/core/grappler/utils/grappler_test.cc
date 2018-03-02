/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/utils/grappler_test.h"
#include <memory>
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace grappler {

std::vector<Tensor> GrapplerTest::EvaluateNodes(
    const GraphDef& graph, const std::vector<string>& node_names) {
  SessionOptions options;
  std::unique_ptr<tensorflow::Session> session(NewSession(options));
  TF_CHECK_OK(session->Create(graph));
  RunOptions run_options;
  std::vector<Tensor> output_tensors;
  TF_CHECK_OK(session->Run(run_options, {}, node_names, node_names,
                           &output_tensors, nullptr));
  TF_CHECK_OK(session->Close());
  return output_tensors;
}

std::vector<Tensor> GrapplerTest::EvaluateFetchNodes(const GrapplerItem& item) {
  SessionOptions options;
  std::unique_ptr<tensorflow::Session> session(NewSession(options));
  TF_CHECK_OK(session->Create(item.graph));
  RunOptions run_options;
  if (!item.init_ops.empty()) {
    std::vector<Tensor> dummy;
    TF_CHECK_OK(
        session->Run(run_options, {}, {}, item.init_ops, &dummy, nullptr));
  }
  std::vector<Tensor> output_tensors;
  TF_CHECK_OK(session->Run(run_options, item.feed, item.fetch, {},
                           &output_tensors, nullptr));
  TF_CHECK_OK(session->Close());
  return output_tensors;
}

void GrapplerTest::AddNode(const string& name, const string& op,
                           const std::vector<string>& inputs, GraphDef* graph) {
  auto* node = graph->add_node();
  node->set_name(name);
  node->set_op(op);
  for (const auto& input : inputs) {
    node->add_input(input);
  }
}

void GrapplerTest::CompareGraphs(GraphDef want, GraphDef got) {
  auto comparator = [](const NodeDef& n1, const NodeDef& n2) -> bool {
    return n1.name() < n2.name();
  };
  std::sort(want.mutable_node()->begin(), want.mutable_node()->end(),
            comparator);
  std::sort(got.mutable_node()->begin(), got.mutable_node()->end(), comparator);

  for (int i = 0; i < want.node_size(); ++i) {
    std::sort(want.mutable_node(i)->mutable_input()->begin(),
              want.mutable_node(i)->mutable_input()->end());
  }
  for (int i = 0; i < got.node_size(); ++i) {
    std::sort(got.mutable_node(i)->mutable_input()->begin(),
              got.mutable_node(i)->mutable_input()->end());
  }

  ASSERT_EQ(want.node_size(), got.node_size());
  for (int i = 0; i < want.node_size(); ++i) {
    EXPECT_EQ(want.node(i).op(), got.node(i).op());
    EXPECT_EQ(want.node(i).name(), got.node(i).name());
    ASSERT_EQ(want.node(i).input_size(), got.node(i).input_size());
    for (int j = 0; j < want.node(i).input_size(); ++j) {
      EXPECT_TRUE(IsSameInput(want.node(i).input(j), got.node(i).input(j)));
    }
  }
}

}  // namespace grappler
}  // namespace tensorflow
