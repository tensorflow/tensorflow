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
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace grappler {

GrapplerTest::GrapplerTest() {
  // Turn off all the automatic optimizations to ensure that we run the graph
  // exactly as it is given to us. This ensures that we can compare the results
  // before and after manual optimization, without any of the automatic
  // optimizations interfering in the comparison.
  RewriterConfig* cfg =
      options_.config.mutable_graph_options()->mutable_rewrite_options();
  // TODO(rmlarsen): Add utility to generate config w/ all optimizers turned
  // off.
  cfg->set_arithmetic_optimization(RewriterConfig::OFF);
  cfg->set_constant_folding(RewriterConfig::OFF);
  cfg->set_debug_stripper(RewriterConfig::OFF);
  cfg->set_dependency_optimization(RewriterConfig::OFF);
  cfg->set_function_optimization(RewriterConfig::OFF);
  cfg->set_layout_optimizer(RewriterConfig::OFF);
  cfg->set_loop_optimization(RewriterConfig::OFF);
  cfg->set_pin_to_host_optimization(RewriterConfig::OFF);
}

std::vector<Tensor> GrapplerTest::EvaluateNodes(
    const GraphDef& graph, const std::vector<string>& node_names) const {
  return EvaluateNodes(graph, node_names, {});
}

std::vector<Tensor> GrapplerTest::EvaluateNodes(
    const GraphDef& graph, const std::vector<string>& node_names,
    const std::vector<std::pair<string, Tensor>>& inputs) const {
  std::unique_ptr<tensorflow::Session> session(NewSession(options_));
  TF_CHECK_OK(session->Create(graph));
  RunOptions run_options;
  std::vector<Tensor> output_tensors;
  TF_CHECK_OK(session->Run(run_options, inputs, node_names, node_names,
                           &output_tensors, nullptr));
  TF_CHECK_OK(session->Close());
  return output_tensors;
}

std::vector<Tensor> GrapplerTest::EvaluateFetchNodes(
    const GrapplerItem& item) const {
  std::unique_ptr<tensorflow::Session> session(NewSession(options_));
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

NodeDef* GrapplerTest::AddNode(
    const string& name, const string& op, const std::vector<string>& inputs,
    const std::vector<std::pair<string, AttrValue>>& attributes,
    GraphDef* graph) const {
  NodeDef* node = graph->add_node();
  node->set_name(name);
  node->set_op(op);
  for (const string& input : inputs) {
    node->add_input(input);
  }
  for (auto attr : attributes) {
    (*node->mutable_attr())[attr.first] = attr.second;
  }
  return node;
}

void GrapplerTest::CompareGraphs(GraphDef want, GraphDef got) const {
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
    EXPECT_EQ(want.node(i).device(), got.node(i).device());

    ASSERT_EQ(want.node(i).input_size(), got.node(i).input_size());
    for (int j = 0; j < want.node(i).input_size(); ++j) {
      const TensorId want_tensor = ParseTensorName(want.node(i).input(j));
      const TensorId got_tensor = ParseTensorName(got.node(i).input(j));
      EXPECT_EQ(want_tensor.ToString(), got_tensor.ToString());
    }
  }
}

bool GrapplerTest::IsNodesDirectlyConnected(const NodeMap& node_map,
                                            const string& src,
                                            const string& dst, int position) {
  const NodeDef* src_node = node_map.GetNode(src);
  const NodeDef* dst_node = node_map.GetNode(dst);
  EXPECT_TRUE(src_node != nullptr) << src << " node not found";
  EXPECT_TRUE(dst_node != nullptr) << dst << " node not found";
  return src_node && dst_node && dst_node->input(position) == src_node->name();
}

int GrapplerTest::CountOpNodes(const GraphDef& graph, const string& op) {
  return std::count_if(graph.node().begin(), graph.node().end(),
                       [&op](const NodeDef& node) { return node.op() == op; });
}

}  // namespace grappler
}  // namespace tensorflow
