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

#include "absl/algorithm/container.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace grappler {

namespace {
void CompareGraphNodes(protobuf::RepeatedPtrField<NodeDef>* want,
                       protobuf::RepeatedPtrField<NodeDef>* got) {
  auto comparator = [](const NodeDef& n1, const NodeDef& n2) -> bool {
    return n1.name() < n2.name();
  };

  std::sort(want->begin(), want->end(), comparator);
  std::sort(got->begin(), got->end(), comparator);

  ASSERT_EQ(want->size(), got->size());

  for (int i = 0; i < want->size(); ++i) {
    NodeDef& want_node = (*want)[i];
    NodeDef& got_node = (*got)[i];

    EXPECT_EQ(want_node.op(), got_node.op());
    EXPECT_EQ(want_node.name(), got_node.name());
    EXPECT_EQ(want_node.device(), got_node.device());
    ASSERT_EQ(want_node.input_size(), got_node.input_size())
        << "want_node =\n"
        << want_node.DebugString() << "\ngot_node =\n"
        << got_node.DebugString();

    // Order of control dependencies doesn't matter, so we sort them first.
    const auto is_control = [](const string& input) -> bool {
      return ParseTensorName(input).index() < 0;
    };

    auto want_inputs = want_node.mutable_input();
    auto got_inputs = got_node.mutable_input();
    std::sort(absl::c_find_if(*want_inputs, is_control), want_inputs->end());
    std::sort(absl::c_find_if(*got_inputs, is_control), got_inputs->end());

    for (int j = 0; j < want_node.input_size(); ++j) {
      const TensorId want_tensor = ParseTensorName(want_node.input(j));
      const TensorId got_tensor = ParseTensorName(got_node.input(j));
      EXPECT_EQ(want_tensor.ToString(), got_tensor.ToString());
    }
  }
}

void SetAllOptimizers(RewriterConfig* cfg, RewriterConfig::Toggle value) {
  cfg->set_arithmetic_optimization(value);
  cfg->set_auto_mixed_precision(value);
  cfg->set_auto_mixed_precision_bfloat16(value);
  cfg->set_common_subgraph_elimination(value);
  cfg->set_constant_folding(value);
  cfg->set_debug_stripper(value);
  cfg->set_dependency_optimization(value);
  cfg->set_function_optimization(value);
  cfg->set_implementation_selector(value);
  cfg->set_layout_optimizer(value);
  cfg->set_loop_optimization(value);
  cfg->set_pin_to_host_optimization(value);
  cfg->set_remapping(value);
  cfg->set_scoped_allocator_optimization(value);
  cfg->set_shape_optimization(value);
}
}  // namespace

GrapplerTest::GrapplerTest() {
  // Turn off all the automatic optimizations to ensure that we run the graph
  // exactly as it is given to us. This ensures that we can compare the
  // results before and after manual optimization, without any of the
  // automatic optimizations interfering in the comparison.
  DisableAllOptimizers();
}

void GrapplerTest::DisableAllOptimizers() {
  SetAllOptimizers(
      options_.config.mutable_graph_options()->mutable_rewrite_options(),
      RewriterConfig::OFF);
}

void GrapplerTest::EnableAllOptimizers() {
  SetAllOptimizers(
      options_.config.mutable_graph_options()->mutable_rewrite_options(),
      RewriterConfig::ON);
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
  CompareGraphNodes(want.mutable_node(), got.mutable_node());
}

void GrapplerTest::CompareFunctions(FunctionDef want, FunctionDef got) const {
  CompareGraphNodes(want.mutable_node_def(), got.mutable_node_def());
}

void GrapplerTest::CompareNodes(const NodeDef& want, const NodeDef& got) const {
  EXPECT_EQ(want.name(), got.name());
  EXPECT_EQ(want.op(), got.op());

  std::vector<string> want_inputs(want.input().begin(), want.input().end());
  std::vector<string> got_inputs(got.input().begin(), got.input().end());
  EXPECT_EQ(want_inputs, got_inputs);

  const auto attr_name = [](const std::pair<const string, AttrValue>& attr) {
    return attr.first;
  };

  std::vector<string> want_attrs;
  std::vector<string> got_attrs;
  absl::c_transform(want.attr(), std::back_inserter(want_attrs), attr_name);
  absl::c_transform(got.attr(), std::back_inserter(got_attrs), attr_name);
  absl::c_sort(want_attrs);
  absl::c_sort(got_attrs);
  EXPECT_EQ(want_attrs, got_attrs);

  for (const string& attr : want_attrs) {
    EXPECT_TRUE(AreAttrValuesEqual(want.attr().at(attr), got.attr().at(attr)));
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
