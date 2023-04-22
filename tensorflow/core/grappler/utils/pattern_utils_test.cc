/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/utils/pattern_utils.h"

#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace grappler {
namespace utils {
namespace {

using ::tensorflow::ops::Placeholder;

void GetMatMulBiasAddGeluGraph(GraphDef* graph,
                               bool add_external_dependent = false) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto input_shape = ops::Placeholder::Shape({8, 32});
  auto weight_shape = ops::Placeholder::Shape({32, 64});
  auto bias_shape = ops::Placeholder::Shape({64});

  auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
  auto weight = Placeholder(s.WithOpName("weight"), DT_FLOAT, weight_shape);
  auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

  auto matmul = ops::MatMul(s.WithOpName("matmul"), input, weight);
  auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), matmul, bias);
  if (add_external_dependent) {
    auto external_dependent =
        ops::Identity(s.WithOpName("external_dependent"), bias_add);
  }
  // Gelu with smaller ops
  auto one_over_square_root_two =
      ops::Const(s.WithOpName("one_over_square_root_two"), {0.707f}, {});
  auto bias_add_times_const = ops::Mul(s.WithOpName("bias_add_times_const"),
                                       bias_add, one_over_square_root_two);
  auto erf = ops::Erf(s.WithOpName("erf"), bias_add_times_const);
  auto one = ops::Const(s.WithOpName("one"), {1.0f}, {});
  auto erf_plus_one = ops::AddV2(s.WithOpName("erf_plus_one"), erf, one);
  auto one_half = ops::Const(s.WithOpName("one_half"), {0.5f}, {});
  auto one_half_times_erf_plus_one = ops::Mul(
      s.WithOpName("one_half_times_erf_plus_one"), one_half, erf_plus_one);
  auto gelu =
      ops::Mul(s.WithOpName("gelu"), one_half_times_erf_plus_one, bias_add);
  auto fetch = ops::Identity(s.WithOpName("fetch"), gelu);

  TF_ASSERT_OK(s.ToGraphDef(graph));
}

OpTypePattern GetMatMulBiasAddGeluPattern() {
  // Although labels are arbitrary, for the convenience of check they are
  // prefixed with "my_" to the orginal node names in the global graph.
  // clang-format off
  OpTypePattern pattern_syntax{"Mul", "my_gelu", NodeStatus::kReplace,
    {
      {"Mul", "my_one_half_times_erf_plus_one", NodeStatus::kRemove,
        {
          {"Const", "my_one_half", NodeStatus::kRemain},
          {"AddV2", "my_erf_plus_one", NodeStatus::kRemove,
            {
              {"Erf", "my_erf", NodeStatus::kRemove,
                {
                  {"Mul", "my_bias_add_times_const", NodeStatus::kRemove,
                    {
                      {"BiasAdd", "my_bias_add", NodeStatus::kRemove},
                      {"Const", "my_one_over_square_root_two", NodeStatus::kRemain}
                    }
                  }
                }
              },
              {"Const", "my_one", NodeStatus::kRemain}
            }
          }
        }
      },
      {"BiasAdd", "my_bias_add", NodeStatus::kRemove,
        {
          {"MatMul", "my_matmul", NodeStatus::kRemove},
          {"*", "my_bias", NodeStatus::kRemain}
        }
      }
    }
  };  // clang-format on

  return pattern_syntax;
}

class PatternMatcherTest : public ::testing::Test {
 protected:
  struct NodeConfig {
    NodeConfig(string name, string op, std::vector<string> inputs)
        : name(std::move(name)), op(std::move(op)), inputs(std::move(inputs)) {}

    string name;
    string op;
    std::vector<string> inputs;
  };

  static GraphDef CreateGraph(const std::vector<NodeConfig>& nodes) {
    GraphDef graph;

    for (const NodeConfig& node : nodes) {
      NodeDef node_def;
      node_def.set_name(node.name);
      node_def.set_op(node.op);
      for (const string& input : node.inputs) {
        node_def.add_input(input);
      }
      *graph.add_node() = std::move(node_def);
    }

    return graph;
  }
};

TEST_F(PatternMatcherTest, Tree) {
  // A Data flow graph. Data flows from top to bottom. Here A, B, C, D, and E
  // are ops.
  //
  //     Input graph              Subgraph for pattern matcher
  //
  //         A                          C   D
  //         |                           \ /
  //         B                            E
  //        /
  //       C   D
  //        \ /
  //         E
  //
  // E is the root of pattern syntax as shown below that the pattern matcher
  // would match.
  //  {"E", "my_e", NodeStatus::kReplace,
  //    {
  //      {"C", "my_c", NodeStatus::kRemove}
  //      {"D", "my_d", NodeStatus::kRemove}
  //    }
  //  }

  ::tensorflow::Status status;
  GraphDef graph = CreateGraph({{"e", "E", {"c", "d"}},
                                {"c", "C", {"b"}},
                                {"d", "D", {}},
                                {"b", "B", {"a"}},
                                {"a", "A", {}}});
  // clang-format off
  OpTypePattern pattern{"E", "my_e", NodeStatus::kReplace,
    {
      {"C", "my_c", NodeStatus::kRemove},
      {"D", "my_d", NodeStatus::kRemove}
    }
  };  // clang-format on

  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);
  TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  auto root_node_view = graph_view.GetNode("e");

  SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(&graph_view);
  std::map<string, int> matched_nodes_map;  // label to node index map
  std::set<int> remove_node_indices;
  bool found_match = graph_matcher.GetMatchedNodes(
      pattern, root_node_view, &matched_nodes_map, &remove_node_indices);

  EXPECT_TRUE(found_match);
  EXPECT_FALSE(matched_nodes_map.empty());
  EXPECT_FALSE(remove_node_indices.empty());

  bool all_indices_matched = true;
  for (auto it = matched_nodes_map.begin(); it != matched_nodes_map.begin();
       it++) {
    auto label = str_util::StripPrefix(it->first, "my_");
    int matched_node_idx = it->second;
    int expected_node_idx = graph_view.GetNode(label)->node_index();
    if (matched_node_idx != expected_node_idx) {
      all_indices_matched = false;
      break;
    }
  }
  EXPECT_TRUE(all_indices_matched);
}

TEST_F(PatternMatcherTest, DAG) {
  // A Data flow graph. Data flows from top to bottom. Here A, B, C, D, and E
  // are ops.
  //
  //     Input graph              Subgraph for pattern matcher
  //
  //         A
  //         |                           B
  //         B                          / \
  //        / \                        C   D
  //       C   D                        \ /
  //        \ /                          E
  //         E
  //
  // E is the root of pattern syntax as shown below that the pattern matcher
  // would match.
  //  {"E", "my_e", NodeStatus::kReplace,
  //    {
  //      {"C", "my_c", NodeStatus::kRemove,
  //        {
  //          {"B", "my_b", NodeStatus::kRemove}
  //        }
  //      },
  //      {"D", "my_d", NodeStatus::kRemove,
  //        {
  //          {"B", "my_b", NodeStatus::kRemove}
  //        }
  //      }
  //    }
  //  }

  ::tensorflow::Status status;
  GraphDef graph = CreateGraph({{"e", "E", {"c", "d"}},
                                {"c", "C", {"b"}},
                                {"d", "D", {"b"}},
                                {"b", "B", {"a"}},
                                {"a", "A", {}}});
  // clang-format off
  OpTypePattern pattern{"E", "my_e", NodeStatus::kReplace,
    {
      {"C", "my_c", NodeStatus::kRemove,
        {
          {"B", "my_b", NodeStatus::kRemove}
        }
      },
      {"D", "my_d", NodeStatus::kRemove,
        {
          {"B", "my_b", NodeStatus::kRemove}
        }
      }
    }
  };  // clang-format on

  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);
  TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  auto root_node_view = graph_view.GetNode("e");

  SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(&graph_view);
  std::map<string, int> matched_nodes_map;  // label to node index map
  std::set<int> remove_node_indices;
  bool found_match = graph_matcher.GetMatchedNodes(
      pattern, root_node_view, &matched_nodes_map, &remove_node_indices);

  EXPECT_TRUE(found_match);
  EXPECT_FALSE(matched_nodes_map.empty());
  EXPECT_FALSE(remove_node_indices.empty());

  bool all_indices_matched = true;
  for (auto it = matched_nodes_map.begin(); it != matched_nodes_map.begin();
       it++) {
    auto label = str_util::StripPrefix(it->first, "my_");
    int matched_node_idx = it->second;
    int expected_node_idx = graph_view.GetNode(label)->node_index();
    if (matched_node_idx != expected_node_idx) {
      all_indices_matched = false;
      break;
    }
  }
  EXPECT_TRUE(all_indices_matched);
}

// Pattern should not be matched if any of candidate remove nodes has external
// dependent.
TEST_F(PatternMatcherTest, DAGExternalDependent) {
  // A Data flow graph. Data flows from top to bottom. Here A, B, C, D, E, and F
  // are ops.
  //
  //     Input graph              Subgraph for pattern matcher
  //
  //         A
  //         |                           B
  //         B                          / \
  //        / \                        C   D
  //       C   D                        \ /
  //        \ / \                        E
  //         E   F
  //
  // E is the root of pattern syntax as shown below that the pattern matcher
  // would match. Note D is a candidate for remove node as mentioned in the
  // syntax. So Pattern matcher should not find a match.
  //  {"E", "my_e", NodeStatus::Replace,
  //    {
  //      {"C", "my_c", NodeStatus::kRemove,
  //        {
  //          {"B", "my_b", NodeStatus::kRemove}
  //        }
  //      },
  //      {"D", "my_d", NodeStatus::kRemove,
  //        {
  //          {"B", "my_b", NodeStatus::kRemove}
  //        }
  //      }
  //    }
  //  }

  ::tensorflow::Status status;
  GraphDef graph = CreateGraph({{"f", "F", {"d"}},
                                {"e", "E", {"c", "d"}},
                                {"c", "C", {"b"}},
                                {"d", "D", {"b"}},
                                {"b", "B", {"a"}},
                                {"a", "A", {}}});
  // clang-format off
  OpTypePattern pattern{"E", "my_e", NodeStatus::kReplace,
    {
      {"C", "my_c", NodeStatus::kRemove,
        {
          {"B", "my_b", NodeStatus::kRemove}
        }
      },
      {"D", "my_d", NodeStatus::kRemove,
        {
          {"B", "my_b", NodeStatus::kRemove}
        }
      }
    }
  };  // clang-format on

  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);
  TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  auto root_node_view = graph_view.GetNode("e");

  SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(&graph_view);
  std::map<string, int> matched_nodes_map;  // label to node index map
  std::set<int> remove_node_indices;
  bool found_match = graph_matcher.GetMatchedNodes(
      pattern, root_node_view, &matched_nodes_map, &remove_node_indices);

  EXPECT_FALSE(found_match);
  EXPECT_TRUE(matched_nodes_map.empty());
  EXPECT_TRUE(remove_node_indices.empty());
}

TEST_F(PatternMatcherTest, MatMulBiasAddGelu) {
  ::tensorflow::Status status;
  GraphDef graph;
  GetMatMulBiasAddGeluGraph(&graph);
  OpTypePattern pattern = GetMatMulBiasAddGeluPattern();
  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);
  TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  auto root_node_view = graph_view.GetNode("gelu");

  SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(&graph_view);
  std::map<string, int> matched_nodes_map;  // label to node index map
  std::set<int> remove_node_indices;
  bool found_match = graph_matcher.GetMatchedNodes(
      pattern, root_node_view, &matched_nodes_map, &remove_node_indices);

  EXPECT_TRUE(found_match);
  EXPECT_FALSE(matched_nodes_map.empty());
  EXPECT_FALSE(remove_node_indices.empty());

  bool all_indices_matched = true;
  for (auto it = matched_nodes_map.begin(); it != matched_nodes_map.begin();
       it++) {
    auto label = str_util::StripPrefix(it->first, "my_");
    int matched_node_idx = it->second;
    int expected_node_idx = graph_view.GetNode(label)->node_index();
    if (matched_node_idx != expected_node_idx) {
      all_indices_matched = false;
      break;
    }
  }
  EXPECT_TRUE(all_indices_matched);
}

// Pattern should not be matched if any of candidate remove nodes has external
// dependent.
TEST_F(PatternMatcherTest, MatMulBiasAddGeluExternalDependent) {
  ::tensorflow::Status status;
  GraphDef graph;
  GetMatMulBiasAddGeluGraph(&graph, /*add_external_dependent=*/true);
  OpTypePattern pattern = GetMatMulBiasAddGeluPattern();
  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);
  TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  auto root_node_view = graph_view.GetNode("gelu");

  SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(&graph_view);
  std::map<string, int> matched_nodes_map;  // label to node index map
  std::set<int> remove_node_indices;
  bool found_match = graph_matcher.GetMatchedNodes(
      pattern, root_node_view, &matched_nodes_map, &remove_node_indices);

  EXPECT_FALSE(found_match);
  EXPECT_TRUE(matched_nodes_map.empty());
  EXPECT_TRUE(remove_node_indices.empty());
}

TEST_F(PatternMatcherTest, MatMulBiasAddGeluMutation) {
  ::tensorflow::Status status;
  GraphDef graph;
  GetMatMulBiasAddGeluGraph(&graph);
  OpTypePattern pattern = GetMatMulBiasAddGeluPattern();
  MutableGraphView graph_view(&graph, &status);
  TF_ASSERT_OK(status);
  TF_EXPECT_OK(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  auto root_node_view = graph_view.GetNode("gelu");

  SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(&graph_view);
  std::map<string, int> matched_nodes_map;  // label to node index map
  std::set<int> remove_node_indices;
  bool found_match = graph_matcher.GetMatchedNodes(
      pattern, root_node_view, &matched_nodes_map, &remove_node_indices);
  EXPECT_TRUE(found_match);
  EXPECT_FALSE(matched_nodes_map.empty());
  EXPECT_FALSE(remove_node_indices.empty());

  // Before mutation number of nodes.
  int num_nodes_before = graph_view.NumNodes();
  // Before mutation node_names of the remove candidate nodes.
  std::vector<string> remove_node_names;
  for (auto const& node_idx : remove_node_indices) {
    remove_node_names.push_back(graph_view.GetNode(node_idx)->GetName());
  }

  Mutation* mutation = graph_view.GetMutationBuilder();
  // Replace with fused op.
  NodeDef fused_node;
  fused_node.set_name("gelu");
  fused_node.set_op("_FusedMatMul");
  fused_node.add_input(graph_view.GetNode("matmul")->node()->input(0));
  fused_node.add_input(graph_view.GetNode("matmul")->node()->input(1));
  fused_node.add_input(graph_view.GetNode("bias_add")->node()->input(1));
  mutation->AddNode(std::move(fused_node), &status);
  TF_ASSERT_OK(status);
  TF_EXPECT_OK(mutation->Apply());
  // Remove nodes that are marked as NodeStatus::kRemove.
  for (auto const& node_idx : remove_node_indices) {
    mutation->RemoveNode(graph_view.GetNode(node_idx));
  }
  TF_EXPECT_OK(mutation->Apply());

  // After mutation number of nodes.
  int num_nodes_after = graph_view.NumNodes();
  EXPECT_EQ(num_nodes_before - remove_node_indices.size(), num_nodes_after);

  bool remove_nodes_deleted = true;
  for (auto const& node_name : remove_node_names) {
    if (graph_view.GetNode(node_name) != nullptr) {
      remove_nodes_deleted = false;
      break;
    }
  }
  EXPECT_TRUE(remove_nodes_deleted);

  bool replace_node_exist = graph_view.HasNode("gelu") ? true : false;
  EXPECT_TRUE(replace_node_exist);
}

}  // namespace
}  // namespace utils
}  // namespace grappler
}  // namespace tensorflow
