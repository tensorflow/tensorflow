/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tools/graph_transforms/transform_utils.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace graph_transforms {

class TransformUtilsTest : public ::testing::Test {
 protected:
  void TestMapNamesToNodes() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 100;

    Tensor a_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_data, 1.0f);
    Output a_const = Const(root.WithOpName("a"), Input::Initializer(a_data));

    Tensor b_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&b_data, 1.0f);
    Output b_const = Const(root.WithOpName("b"), Input::Initializer(b_data));

    Output add = Add(root.WithOpName("add"), a_const, b_const);

    Output placeholder = Placeholder(root.WithOpName("placeholder"), DT_FLOAT);

    Output mul = Mul(root.WithOpName("output"), add, placeholder);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));
    std::map<string, const NodeDef*> node_map;
    MapNamesToNodes(graph_def, &node_map);

    EXPECT_EQ(1, node_map.count("a"));
    EXPECT_EQ(1, node_map.count("b"));
    EXPECT_EQ(1, node_map.count("add"));
    EXPECT_EQ(1, node_map.count("placeholder"));
    EXPECT_EQ(1, node_map.count("output"));
    EXPECT_EQ(0, node_map.count("no_such_node"));
  }

  void TestMapNodesToOutputs() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 100;

    Tensor a_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_data, 1.0f);
    Output a_const = Const(root.WithOpName("a"), Input::Initializer(a_data));

    Tensor b_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&b_data, 1.0f);
    Output b_const = Const(root.WithOpName("b"), Input::Initializer(b_data));

    Output add = Add(root.WithOpName("add"), a_const, b_const);

    Output placeholder = Placeholder(root.WithOpName("placeholder"), DT_FLOAT);

    Output mul = Mul(root.WithOpName("output"), add, placeholder);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));

    std::map<string, std::vector<const NodeDef*>> outputs_map;
    MapNodesToOutputs(graph_def, &outputs_map);

    EXPECT_EQ(1, outputs_map.count("a"));
    EXPECT_EQ(1, outputs_map["a"].size());
    EXPECT_EQ("add", outputs_map["a"][0]->name());

    EXPECT_EQ(1, outputs_map.count("b"));
    EXPECT_EQ(1, outputs_map["b"].size());
    EXPECT_EQ("add", outputs_map["b"][0]->name());

    EXPECT_EQ(1, outputs_map.count("add"));
    EXPECT_EQ(1, outputs_map["add"].size());
    EXPECT_EQ("output", outputs_map["add"][0]->name());

    EXPECT_EQ(1, outputs_map.count("placeholder"));
    EXPECT_EQ(1, outputs_map["placeholder"].size());
    EXPECT_EQ("output", outputs_map["placeholder"][0]->name());

    EXPECT_EQ(0, outputs_map.count("output"));
    EXPECT_EQ(0, outputs_map.count("no_such_node"));
  }

  void TestNodeNamePartsFromInput() {
    string prefix;
    string node_name;
    string suffix;

    NodeNamePartsFromInput("some_node_name", &prefix, &node_name, &suffix);
    EXPECT_EQ("", prefix);
    EXPECT_EQ("some_node_name", node_name);
    EXPECT_EQ("", suffix);

    NodeNamePartsFromInput("some_node_name/with/slashes", &prefix, &node_name,
                           &suffix);
    EXPECT_EQ("", prefix);
    EXPECT_EQ("some_node_name/with/slashes", node_name);
    EXPECT_EQ("", suffix);

    NodeNamePartsFromInput("some_node_name:0", &prefix, &node_name, &suffix);
    EXPECT_EQ("", prefix);
    EXPECT_EQ("some_node_name", node_name);
    EXPECT_EQ(":0", suffix);

    NodeNamePartsFromInput("^some_node_name", &prefix, &node_name, &suffix);
    EXPECT_EQ("^", prefix);
    EXPECT_EQ("some_node_name", node_name);
    EXPECT_EQ("", suffix);

    NodeNamePartsFromInput("^some_node_name:99", &prefix, &node_name, &suffix);
    EXPECT_EQ("^", prefix);
    EXPECT_EQ("some_node_name", node_name);
    EXPECT_EQ(":99", suffix);
  }

  void TestNodeNameFromInput() {
    EXPECT_EQ("node_name", NodeNameFromInput("node_name"));
    EXPECT_EQ("node_name", NodeNameFromInput("node_name:0"));
    EXPECT_EQ("node_name", NodeNameFromInput("^node_name"));
    EXPECT_EQ("node_name", NodeNameFromInput("^node_name:42"));
  }

  void TestCanonicalInputName() {
    EXPECT_EQ("node_name:0", CanonicalInputName("node_name"));
    EXPECT_EQ("node_name:0", CanonicalInputName("node_name:0"));
    EXPECT_EQ("^node_name:0", CanonicalInputName("^node_name"));
    EXPECT_EQ("^node_name:42", CanonicalInputName("^node_name:42"));
  }

  void TestAddNodeInput() {
    NodeDef node;
    AddNodeInput("foo", &node);
    EXPECT_EQ("foo", node.input(0));
  }

  void TestCopyNodeAttr() {
    NodeDef node;
    auto mutable_attr = node.mutable_attr();
    (*mutable_attr)["foo"].set_i(3);

    NodeDef copied_node;
    CopyNodeAttr(node, "foo", "bar", &copied_node);
    EXPECT_EQ(3, copied_node.attr().at("bar").i());
  }

  void TestSetNodeAttr() {
    NodeDef node;
    int32_t value_i = 32;
    SetNodeAttr("foo", value_i, &node);
    EXPECT_EQ(32, node.attr().at("foo").i());
    string value_s = "some_value";
    SetNodeAttr("bar", value_s, &node);
    EXPECT_EQ("some_value", node.attr().at("bar").s());
  }

  void TestSetNodeTensorAttr() {
    NodeDef node;
    SetNodeTensorAttr<int32>("foo", {3, 1}, {1, 2, 3}, &node);
    TensorProto tensor_proto = node.attr().at("foo").tensor();
    Tensor tensor;
    CHECK(tensor.FromProto(tensor_proto));
    EXPECT_EQ(DT_INT32, tensor.dtype());
    EXPECT_EQ(3, tensor.shape().dim_size(0));
    EXPECT_EQ(1, tensor.shape().dim_size(1));
    EXPECT_EQ(1, tensor.flat<int32>()(0));
    EXPECT_EQ(2, tensor.flat<int32>()(1));
    EXPECT_EQ(3, tensor.flat<int32>()(2));
  }

  void TestSetNodeTensorAttrWithTensor() {
    NodeDef node;
    Tensor input_tensor(DT_INT32, {4, 5});
    test::FillIota<int32>(&input_tensor, 1);
    SetNodeTensorAttr<int32>("foo", input_tensor, &node);
    TensorProto tensor_proto = node.attr().at("foo").tensor();
    Tensor tensor;
    CHECK(tensor.FromProto(tensor_proto));
    test::ExpectTensorEqual<int32>(input_tensor, tensor);
  }

  void TestGetNodeTensorAttr() {
    NodeDef node;
    Tensor input_tensor(DT_INT32, {4, 5});
    test::FillIota<int32>(&input_tensor, 1);
    TensorProto tensor_proto;
    input_tensor.AsProtoTensorContent(&tensor_proto);
    SetNodeAttr("foo", tensor_proto, &node);
    Tensor result = GetNodeTensorAttr(node, "foo");
    test::ExpectTensorEqual<int32>(input_tensor, result);
  }

  void TestFilterGraphDef() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 100;

    Tensor a_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_data, 1.0f);
    Output a_const = Const(root.WithOpName("a"), Input::Initializer(a_data));

    Tensor b_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&b_data, 1.0f);
    Output b_const = Const(root.WithOpName("b"), Input::Initializer(b_data));

    Output add = Add(root.WithOpName("add"), a_const, b_const);

    Output placeholder = Placeholder(root.WithOpName("placeholder"), DT_FLOAT);

    Output mul = Mul(root.WithOpName("output"), add, placeholder);

    Output remove_me = Add(root.WithOpName("remove_me"), mul, add);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));

    GraphDef result_graph_def;
    FilterGraphDef(
        graph_def,
        [](const NodeDef& node) { return (node.name() != "remove_me"); },
        &result_graph_def);

    std::map<string, const NodeDef*> node_map;
    MapNamesToNodes(result_graph_def, &node_map);
    EXPECT_EQ(1, node_map.count("a"));
    EXPECT_EQ(1, node_map.count("b"));
    EXPECT_EQ(1, node_map.count("add"));
    EXPECT_EQ(1, node_map.count("placeholder"));
    EXPECT_EQ(1, node_map.count("output"));
    EXPECT_EQ(0, node_map.count("remove_me"));
  }

  void TestRemoveAttributes() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Output placeholder = Placeholder(root.WithOpName("placeholder"), DT_FLOAT);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));

    GraphDef result_graph_def;
    RemoveAttributes(graph_def, {"dtype"}, &result_graph_def);

    std::map<string, const NodeDef*> node_map;
    MapNamesToNodes(result_graph_def, &node_map);
    const NodeDef* removed_placeholder = node_map["placeholder"];
    EXPECT_EQ(nullptr,
              tensorflow::AttrSlice(*removed_placeholder).Find("dtype"));
  }

  void TestGetOpTypeMatches() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 100;

    Tensor a_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_data, 1.0f);
    Output a_const = Const(root.WithOpName("a"), Input::Initializer(a_data));

    Tensor b_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&b_data, 1.0f);
    Output b_const = Const(root.WithOpName("b"), Input::Initializer(b_data));

    Output add = Add(root.WithOpName("add"), a_const, b_const);

    Output placeholder = Placeholder(root.WithOpName("placeholder"), DT_FLOAT);

    Output mul = Mul(root.WithOpName("output"), add, placeholder);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));

    GraphMatcher matcher(graph_def);

    std::vector<NodeMatch> const_matches;
    TF_ASSERT_OK(matcher.GetOpTypeMatches({"Const"}, &const_matches));
    EXPECT_EQ(2, const_matches.size());
    for (const NodeMatch& match : const_matches) {
      EXPECT_EQ("Const", match.node.op());
      EXPECT_TRUE(("a" == match.node.name()) || ("b" == match.node.name()))
          << "match.node.name()=" << match.node.name();
    }

    std::vector<NodeMatch> add_matches;
    TF_ASSERT_OK(matcher.GetOpTypeMatches({"Add"}, &add_matches));
    EXPECT_EQ(1, add_matches.size());
    EXPECT_EQ("Add", add_matches[0].node.op());
    EXPECT_EQ("add", add_matches[0].node.name());

    std::vector<NodeMatch> add_child_matches;
    TF_ASSERT_OK(matcher.GetOpTypeMatches({"Add", {{"Const"}, {"Const"}}},
                                          &add_child_matches));
    EXPECT_EQ(1, add_child_matches.size());
    EXPECT_EQ("Add", add_child_matches[0].node.op());
    EXPECT_EQ("add", add_child_matches[0].node.name());
    EXPECT_EQ(2, add_child_matches[0].inputs.size());
    for (const NodeMatch& match : add_child_matches[0].inputs) {
      EXPECT_EQ("Const", match.node.op());
      EXPECT_TRUE(("a" == match.node.name()) || ("b" == match.node.name()))
          << "match.node.name()=" << match.node.name();
    }

    std::vector<NodeMatch> no_such_matches;
    TF_ASSERT_OK(matcher.GetOpTypeMatches({"NoSuch"}, &no_such_matches));
    EXPECT_EQ(0, no_such_matches.size());

    std::vector<NodeMatch> all_matches;
    TF_ASSERT_OK(matcher.GetOpTypeMatches(
        {"Mul", {{"Add", {{"Const"}, {"Const"}}}, {"Placeholder"}}},
        &all_matches));
    EXPECT_EQ(1, all_matches.size());
    EXPECT_EQ("Mul", all_matches[0].node.op());
    EXPECT_EQ("output", all_matches[0].node.name());
    EXPECT_EQ(2, all_matches[0].inputs.size());
    EXPECT_EQ("Add", all_matches[0].inputs[0].node.op());
    EXPECT_EQ("add", all_matches[0].inputs[0].node.name());
    EXPECT_EQ(2, all_matches[0].inputs[0].inputs.size());
    EXPECT_EQ("Const", all_matches[0].inputs[0].inputs[0].node.op());
    EXPECT_EQ("a", all_matches[0].inputs[0].inputs[0].node.name());
    EXPECT_EQ(0, all_matches[0].inputs[0].inputs[0].inputs.size());
    EXPECT_EQ("Const", all_matches[0].inputs[0].inputs[1].node.op());
    EXPECT_EQ("b", all_matches[0].inputs[0].inputs[1].node.name());
    EXPECT_EQ(0, all_matches[0].inputs[0].inputs[1].inputs.size());
    EXPECT_EQ("Placeholder", all_matches[0].inputs[1].node.op());
    EXPECT_EQ("placeholder", all_matches[0].inputs[1].node.name());
    EXPECT_EQ(0, all_matches[0].inputs[1].inputs.size());

    std::vector<NodeMatch> wildcard_matches;
    TF_ASSERT_OK(
        matcher.GetOpTypeMatches({"*", {{"*"}, {"*"}}}, &wildcard_matches));
    EXPECT_EQ(1, wildcard_matches.size());
    EXPECT_EQ("Add", wildcard_matches[0].node.op());
    EXPECT_EQ("Const", wildcard_matches[0].inputs[0].node.op());
    EXPECT_EQ("a", wildcard_matches[0].inputs[0].node.name());
    EXPECT_EQ("Const", wildcard_matches[0].inputs[1].node.op());
    EXPECT_EQ("b", wildcard_matches[0].inputs[1].node.name());

    std::vector<NodeMatch> or_matches;
    TF_ASSERT_OK(matcher.GetOpTypeMatches({"Add|Mul"}, &or_matches));
    EXPECT_EQ(2, or_matches.size());
    EXPECT_EQ("Add", or_matches[0].node.op());
    EXPECT_EQ("add", or_matches[0].node.name());
    EXPECT_EQ("Mul", or_matches[1].node.op());
    EXPECT_EQ("output", or_matches[1].node.name());
  }

  void TestGetOpTypeMatchesDAG() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 100;

    Tensor a_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_data, 1.0f);
    Output a_const = Const(root.WithOpName("a"), Input::Initializer(a_data));

    Output add = Add(root.WithOpName("add"), a_const, a_const);

    Output placeholder = Placeholder(root.WithOpName("placeholder"), DT_FLOAT);

    Output mul = Mul(root.WithOpName("output"), add, placeholder);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));

    GraphMatcher matcher(graph_def);

    std::vector<NodeMatch> add_matches;
    TF_ASSERT_OK(matcher.GetOpTypeMatches({"Add", {{"Const"}, {"Const"}}},
                                          &add_matches));
    EXPECT_EQ(1, add_matches.size());
    EXPECT_EQ("Add", add_matches[0].node.op());
    EXPECT_EQ("add", add_matches[0].node.name());
    EXPECT_EQ("Const", add_matches[0].inputs[0].node.op());
    EXPECT_EQ("a", add_matches[0].inputs[0].node.name());
    EXPECT_EQ("Const", add_matches[0].inputs[1].node.op());
    EXPECT_EQ("a", add_matches[0].inputs[1].node.name());
  }

  void TestReplaceMatchingOpTypes() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 10;

    Tensor a_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_data, 1.0f);
    Output a_const = Const(root.WithOpName("a"), Input::Initializer(a_data));

    Tensor b_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&b_data, 1.0f);
    Output b_const = Const(root.WithOpName("b"), Input::Initializer(b_data));

    Output add = Add(root.WithOpName("add"), a_const, b_const);

    Output placeholder = Placeholder(root.WithOpName("placeholder"), DT_FLOAT);

    Output mul = Mul(root.WithOpName("output"), add, placeholder);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));

    GraphDef replaced_graph_def;
    TF_ASSERT_OK(ReplaceMatchingOpTypes(
        graph_def, {"*"},
        [](const NodeMatch& match, const std::set<string>& input_nodes,
           const std::set<string>& output_nodes,
           std::vector<NodeDef>* new_nodes) {
          NodeDef original_copy;
          original_copy = match.node;
          const string original_name = match.node.name();
          original_copy.set_name(original_name + "_before_identity");
          new_nodes->push_back(original_copy);

          NodeDef identity_node;
          identity_node.set_op("Identity");
          identity_node.set_name(original_name);
          *(identity_node.mutable_input()->Add()) = original_copy.name();
          new_nodes->push_back(identity_node);

          return OkStatus();
        },
        {}, &replaced_graph_def));

    EXPECT_EQ(10, replaced_graph_def.node_size());
    for (const NodeDef& node : replaced_graph_def.node()) {
      if (node.name() == "output") {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ("output_before_identity", node.input(0));
      } else if (node.name() == "output_before_identity") {
        EXPECT_EQ("Mul", node.op());
        EXPECT_EQ("add", node.input(0));
        EXPECT_EQ("placeholder", node.input(1));
      } else if (node.name() == "placeholder") {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ("placeholder_before_identity", node.input(0));
      } else if (node.name() == "placeholder_before_identity") {
        EXPECT_EQ("Placeholder", node.op());
      } else if (node.name() == "add") {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ("add_before_identity", node.input(0));
      } else if (node.name() == "add_before_identity") {
        EXPECT_EQ("Add", node.op());
        EXPECT_EQ("a", node.input(0));
        EXPECT_EQ("b", node.input(1));
      } else if (node.name() == "a") {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ("a_before_identity", node.input(0));
      } else if (node.name() == "a_before_identity") {
        EXPECT_EQ("Const", node.op());
      } else if (node.name() == "b") {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ("b_before_identity", node.input(0));
      } else if (node.name() == "b_before_identity") {
        EXPECT_EQ("Const", node.op());
      } else {
        EXPECT_EQ(true, false) << "Unexpected node name found: " << node.name();
      }
    }
  }

  void TestMatchedNodesAsArray() {
    NodeMatch fourth;
    fourth.node.set_name("fourth");

    NodeMatch second;
    second.node.set_name("second");
    second.inputs.push_back(fourth);

    NodeMatch third;
    third.node.set_name("third");
    third.inputs.push_back(fourth);

    NodeMatch first;
    first.node.set_name("first");
    first.inputs.push_back(second);
    first.inputs.push_back(third);

    std::vector<NodeDef> result;
    MatchedNodesAsArray(first, &result);

    EXPECT_EQ(4, result.size());
    EXPECT_EQ("first", result[0].name());
    EXPECT_EQ("second", result[1].name());
    EXPECT_EQ("third", result[2].name());
    EXPECT_EQ("fourth", result[3].name());
  }

  void TestRenameNodeInputs() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 10;

    Tensor a_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_data, 1.0f);
    Output a_const = Const(root.WithOpName("a"), Input::Initializer(a_data));

    Tensor b_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&b_data, 1.0f);
    Output b_const = Const(root.WithOpName("b"), Input::Initializer(b_data));

    Output add = Add(root.WithOpName("add"), a_const, a_const);

    Output placeholder = Placeholder(root.WithOpName("placeholder"), DT_FLOAT);

    Output mul = Mul(root.WithOpName("output"), add, placeholder);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));

    GraphDef renamed_graph_def;
    TF_ASSERT_OK(RenameNodeInputs(graph_def, {{"a", "b"}},
                                  std::unordered_set<string>(),
                                  &renamed_graph_def));

    std::map<string, const NodeDef*> node_map;
    MapNamesToNodes(renamed_graph_def, &node_map);
    EXPECT_EQ("b", node_map.at("add")->input(0));
    EXPECT_EQ("b", node_map.at("add")->input(1));
  }

  void TestRenameNodeInputsWithRedirects() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 10;

    Tensor a_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_data, 1.0f);
    Output a_const = Const(root.WithOpName("a"), Input::Initializer(a_data));

    Tensor b_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&b_data, 1.0f);
    Output b_const = Const(root.WithOpName("b"), Input::Initializer(b_data));

    Tensor c_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&c_data, 1.0f);
    Output c_const = Const(root.WithOpName("c"), Input::Initializer(c_data));

    Output add = Add(root.WithOpName("add"), a_const, b_const);

    Output placeholder = Placeholder(root.WithOpName("placeholder"), DT_FLOAT);

    Output mul = Mul(root.WithOpName("output"), add, placeholder);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));

    GraphDef renamed_graph_def;
    TF_ASSERT_OK(RenameNodeInputs(
        graph_def, {{"a", "f"}, {"f", "e"}, {"e", "d"}, {"d", "c"}},
        std::unordered_set<string>(), &renamed_graph_def));

    std::map<string, const NodeDef*> node_map;
    MapNamesToNodes(renamed_graph_def, &node_map);
    EXPECT_EQ("c", node_map.at("add")->input(0));
    EXPECT_EQ("b", node_map.at("add")->input(1));
  }

  void TestRenameNodeInputsWithCycle() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 10;

    Tensor a_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_data, 1.0f);
    Output a_const = Const(root.WithOpName("a"), Input::Initializer(a_data));

    Tensor b_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&b_data, 1.0f);
    Output b_const = Const(root.WithOpName("b"), Input::Initializer(b_data));

    Tensor c_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&c_data, 1.0f);
    Output c_const = Const(root.WithOpName("c"), Input::Initializer(c_data));

    Output add = Add(root.WithOpName("add"), a_const, b_const);

    Output placeholder = Placeholder(root.WithOpName("placeholder"), DT_FLOAT);

    Output mul = Mul(root.WithOpName("output"), add, placeholder);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));

    GraphDef renamed_graph_def;
    Status rename_status =
        RenameNodeInputs(graph_def, {{"a", "d"}, {"d", "a"}},
                         std::unordered_set<string>(), &renamed_graph_def);
    EXPECT_FALSE(rename_status.ok());
  }

  void TestRenameNodeInputsWithWildcard() {
    auto root = tensorflow::Scope::DisabledShapeInferenceScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 10;

    Tensor a_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_data, 1.0f);
    Output a_const = Const(root.WithOpName("a"), Input::Initializer(a_data));

    QuantizeV2 quantize_a(root.WithOpName("quantize_a"), a_const, a_const,
                          a_const, DT_QUINT8,
                          QuantizeV2::Attrs().Mode("MIN_FIRST"));

    Tensor b_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&b_data, 1.0f);
    Output b_const = Const(root.WithOpName("b"), Input::Initializer(b_data));

    QuantizeV2 quantize_b(root.WithOpName("quantize_b"), b_const, b_const,
                          b_const, DT_QUINT8,
                          QuantizeV2::Attrs().Mode("MIN_FIRST"));

    Output add = Add(root.WithOpName("add"), quantize_a.output_min,
                     quantize_a.output_max);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));

    GraphDef renamed_graph_def;
    TF_ASSERT_OK(RenameNodeInputs(graph_def, {{"quantize_a:*", "quantize_b"}},
                                  std::unordered_set<string>(),
                                  &renamed_graph_def));

    std::map<string, const NodeDef*> node_map;
    MapNamesToNodes(renamed_graph_def, &node_map);
    EXPECT_EQ("quantize_b:1", node_map.at("add")->input(0));
    EXPECT_EQ("quantize_b:2", node_map.at("add")->input(1));
  }

  void TestRenameNodeInputsWithIgnores() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 10;

    Tensor a_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_data, 1.0f);
    Output a_const = Const(root.WithOpName("a"), Input::Initializer(a_data));

    Tensor b_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&b_data, 1.0f);
    Output b_const = Const(root.WithOpName("b"), Input::Initializer(b_data));

    Output add = Add(root.WithOpName("add"), a_const, a_const);

    Output add2 = Add(root.WithOpName("add2"), a_const, a_const);

    Output placeholder = Placeholder(root.WithOpName("placeholder"), DT_FLOAT);

    Output mul = Mul(root.WithOpName("mul"), add, placeholder);

    Output mul2 = Mul(root.WithOpName("output"), mul, add2);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));

    GraphDef renamed_graph_def;
    TF_ASSERT_OK(RenameNodeInputs(graph_def, {{"a", "b"}}, {"add2"},
                                  &renamed_graph_def));

    std::map<string, const NodeDef*> node_map;
    MapNamesToNodes(renamed_graph_def, &node_map);
    EXPECT_EQ("b", node_map.at("add")->input(0));
    EXPECT_EQ("b", node_map.at("add")->input(1));
    EXPECT_EQ("a", node_map.at("add2")->input(0));
    EXPECT_EQ("a", node_map.at("add2")->input(1));
  }

  void TestFindInvalidInputs() {
    GraphDef graph_def;

    NodeDef* mul_node = graph_def.mutable_node()->Add();
    mul_node->set_op("Mul");
    mul_node->set_name("mul_node");
    *(mul_node->mutable_input()->Add()) = "add_node1";
    *(mul_node->mutable_input()->Add()) = "add_node2:0";
    *(mul_node->mutable_input()->Add()) = "^const_node1:0";

    NodeDef* add_node1 = graph_def.mutable_node()->Add();
    add_node1->set_op("Add");
    add_node1->set_name("add_node1");
    *(add_node1->mutable_input()->Add()) = "missing_input1";
    *(add_node1->mutable_input()->Add()) = "const_node1:0";
    *(add_node1->mutable_input()->Add()) = "missing_input2";

    NodeDef* add_node2 = graph_def.mutable_node()->Add();
    add_node2->set_op("Add");
    add_node2->set_name("add_node2");
    *(add_node2->mutable_input()->Add()) = "missing_input3";
    *(add_node2->mutable_input()->Add()) = "const_node1:0";
    *(add_node2->mutable_input()->Add()) = "^const_node2";

    NodeDef* const_node1 = graph_def.mutable_node()->Add();
    const_node1->set_op("Const");
    const_node1->set_name("const_node1");

    NodeDef* const_node2 = graph_def.mutable_node()->Add();
    const_node2->set_op("Const");
    const_node2->set_name("const_node2");

    std::vector<std::pair<string, string>> invalid_inputs;
    FindInvalidInputs(graph_def, &invalid_inputs);
    EXPECT_EQ(3, invalid_inputs.size());
    for (const std::pair<string, string>& invalid_input : invalid_inputs) {
      EXPECT_TRUE((invalid_input.first == "add_node1") ||
                  (invalid_input.first == "add_node2"));
      if (invalid_input.first == "add_node1") {
        EXPECT_TRUE((invalid_input.second == "missing_input1") ||
                    (invalid_input.second == "missing_input2"))
            << invalid_input.second;
      } else if (invalid_input.first == "add_node2") {
        EXPECT_EQ("missing_input3", invalid_input.second);
      }
    }
  }

  void TestIsGraphValid() {
    GraphDef invalid_graph_def;

    NodeDef* mul_node = invalid_graph_def.mutable_node()->Add();
    mul_node->set_op("Mul");
    mul_node->set_name("mul_node");
    *(mul_node->mutable_input()->Add()) = "add_node1";
    *(mul_node->mutable_input()->Add()) = "add_node2:0";
    *(mul_node->mutable_input()->Add()) = "^const_node1:0";

    NodeDef* add_node1 = invalid_graph_def.mutable_node()->Add();
    add_node1->set_op("Add");
    add_node1->set_name("add_node1");
    *(add_node1->mutable_input()->Add()) = "missing_input1";
    *(add_node1->mutable_input()->Add()) = "const_node1:0";
    *(add_node1->mutable_input()->Add()) = "missing_input2";

    NodeDef* add_node2 = invalid_graph_def.mutable_node()->Add();
    add_node2->set_op("Add");
    add_node2->set_name("add_node2");
    *(add_node2->mutable_input()->Add()) = "missing_input3";
    *(add_node2->mutable_input()->Add()) = "const_node1:0";
    *(add_node2->mutable_input()->Add()) = "^const_node2";

    NodeDef* const_node1 = invalid_graph_def.mutable_node()->Add();
    const_node1->set_op("Const");
    const_node1->set_name("const_node1");

    NodeDef* const_node2 = invalid_graph_def.mutable_node()->Add();
    const_node2->set_op("Const");
    const_node2->set_name("const_node2");

    EXPECT_FALSE(IsGraphValid(invalid_graph_def).ok());

    GraphDef valid_graph_def;

    NodeDef* const_node3 = valid_graph_def.mutable_node()->Add();
    const_node3->set_op("Const");
    const_node3->set_name("const_node2");

    EXPECT_TRUE(IsGraphValid(valid_graph_def).ok());
  }

  void TestGetInOutTypes() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 20;

    Tensor float_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&float_data, 1.0f);
    Output float_const =
        Const(root.WithOpName("float_const"), Input::Initializer(float_data));

    Tensor int_data(DT_INT32, TensorShape({width}));
    test::FillIota<int32>(&int_data, 1);
    Output int_const =
        Const(root.WithOpName("int_const"), Input::Initializer(int_data));

    Output float_relu = Relu(root.WithOpName("float_relu"), float_const);

    Output int_relu = Relu(root.WithOpName("int_relu"), int_const);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));

    std::map<string, const NodeDef*> node_map;
    MapNamesToNodes(graph_def, &node_map);

    const NodeDef* float_const_def = node_map.at("float_const");
    DataTypeVector float_const_inputs;
    DataTypeVector float_const_outputs;
    TF_EXPECT_OK(GetInOutTypes(*float_const_def, &float_const_inputs,
                               &float_const_outputs));
    ASSERT_EQ(0, float_const_inputs.size());
    ASSERT_EQ(1, float_const_outputs.size());
    EXPECT_EQ(DT_FLOAT, float_const_outputs[0]);

    const NodeDef* int_const_def = node_map.at("int_const");
    DataTypeVector int_const_inputs;
    DataTypeVector int_const_outputs;
    TF_EXPECT_OK(
        GetInOutTypes(*int_const_def, &int_const_inputs, &int_const_outputs));
    ASSERT_EQ(0, int_const_inputs.size());
    ASSERT_EQ(1, int_const_outputs.size());
    EXPECT_EQ(DT_INT32, int_const_outputs[0]);

    const NodeDef* float_relu_def = node_map.at("float_relu");
    DataTypeVector float_relu_inputs;
    DataTypeVector float_relu_outputs;
    TF_EXPECT_OK(GetInOutTypes(*float_relu_def, &float_relu_inputs,
                               &float_relu_outputs));
    ASSERT_EQ(1, float_relu_inputs.size());
    EXPECT_EQ(DT_FLOAT, float_relu_inputs[0]);
    ASSERT_EQ(1, float_relu_outputs.size());
    EXPECT_EQ(DT_FLOAT, float_relu_outputs[0]);

    const NodeDef* int_relu_def = node_map.at("int_relu");
    DataTypeVector int_relu_inputs;
    DataTypeVector int_relu_outputs;
    TF_EXPECT_OK(
        GetInOutTypes(*int_relu_def, &int_relu_inputs, &int_relu_outputs));
    ASSERT_EQ(1, int_relu_inputs.size());
    EXPECT_EQ(DT_INT32, int_relu_inputs[0]);
    ASSERT_EQ(1, int_relu_outputs.size());
    EXPECT_EQ(DT_INT32, int_relu_outputs[0]);
  }

  void TestCopyOriginalMatch() {
    NodeDef a;
    a.set_op("Relu");
    a.set_name("a");
    AddNodeInput("b", &a);

    NodeDef b;
    b.set_op("Const");
    b.set_name("b");

    NodeMatch b_match;
    b_match.node = b;

    NodeMatch a_match;
    a_match.node = a;
    a_match.inputs.push_back(b_match);

    std::vector<NodeDef> new_nodes;
    CopyOriginalMatch(a_match, &new_nodes);
    EXPECT_EQ(2, new_nodes.size());
    EXPECT_EQ("a", new_nodes[0].name());
    EXPECT_EQ("Relu", new_nodes[0].op());
    EXPECT_EQ("b", new_nodes[1].name());
    EXPECT_EQ("Const", new_nodes[1].op());
  }

  void TestHashNodeDef() {
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 10;

    auto a_root = tensorflow::Scope::NewRootScope();
    Tensor a_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_data, 1.0f);
    Output a_const = Const(a_root.WithOpName("a"), Input::Initializer(a_data));
    GraphDef a_graph_def;
    TF_ASSERT_OK(a_root.ToGraphDef(&a_graph_def));
    const NodeDef& a_node_def = a_graph_def.node(0);

    auto b_root = tensorflow::Scope::NewRootScope();
    Tensor b_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&b_data, 1.0f);
    Output b_const = Const(b_root.WithOpName("a"), Input::Initializer(b_data));
    GraphDef b_graph_def;
    TF_ASSERT_OK(b_root.ToGraphDef(&b_graph_def));
    const NodeDef& b_node_def = b_graph_def.node(0);

    auto c_root = tensorflow::Scope::NewRootScope();
    Tensor c_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&c_data, 2.0f);
    Output c_const = Const(c_root.WithOpName("a"), Input::Initializer(c_data));
    GraphDef c_graph_def;
    TF_ASSERT_OK(c_root.ToGraphDef(&c_graph_def));
    const NodeDef& c_node_def = c_graph_def.node(0);

    auto d_root = tensorflow::Scope::NewRootScope();
    Tensor d_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&d_data, 1.0f);
    Output d_const = Const(d_root.WithOpName("d"), Input::Initializer(d_data));
    GraphDef d_graph_def;
    TF_ASSERT_OK(d_root.ToGraphDef(&d_graph_def));
    const NodeDef& d_node_def = d_graph_def.node(0);

    auto e_root = tensorflow::Scope::NewRootScope();
    Tensor e_data(DT_INT32, TensorShape({width}));
    test::FillIota<int32>(&e_data, 1);
    Output e_const = Const(e_root.WithOpName("a"), Input::Initializer(e_data));
    GraphDef e_graph_def;
    TF_ASSERT_OK(e_root.ToGraphDef(&e_graph_def));
    const NodeDef& e_node_def = e_graph_def.node(0);

    auto f_root = tensorflow::Scope::NewRootScope();
    Tensor f_data(DT_FLOAT, TensorShape({width - 1}));
    test::FillIota<float>(&f_data, 1.0f);
    Output f_const = Const(f_root.WithOpName("a"), Input::Initializer(f_data));
    GraphDef f_graph_def;
    TF_ASSERT_OK(f_root.ToGraphDef(&f_graph_def));
    const NodeDef& f_node_def = f_graph_def.node(0);

    auto g_root = tensorflow::Scope::NewRootScope();
    Tensor g_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&g_data, 1);
    Output g_const = Const(g_root.WithOpName("a").WithDevice("some_device"),
                           Input::Initializer(g_data));
    GraphDef g_graph_def;
    TF_ASSERT_OK(g_root.ToGraphDef(&g_graph_def));
    const NodeDef& g_node_def = g_graph_def.node(0);

    NodeDef relu1_node_def;
    relu1_node_def.set_op("Relu");
    relu1_node_def.set_name("a");
    relu1_node_def.add_input("foo");

    NodeDef relu2_node_def;
    relu2_node_def.set_op("Relu");
    relu2_node_def.set_name("a");
    relu2_node_def.add_input("bar");

    EXPECT_EQ(HashNodeDef(a_node_def), HashNodeDef(b_node_def));
    EXPECT_NE(HashNodeDef(a_node_def), HashNodeDef(c_node_def));
    EXPECT_NE(HashNodeDef(a_node_def), HashNodeDef(d_node_def));
    EXPECT_NE(HashNodeDef(a_node_def), HashNodeDef(e_node_def));
    EXPECT_NE(HashNodeDef(a_node_def), HashNodeDef(f_node_def));
    EXPECT_NE(HashNodeDef(a_node_def), HashNodeDef(g_node_def));
    EXPECT_NE(HashNodeDef(a_node_def), HashNodeDef(relu1_node_def));
    EXPECT_NE(HashNodeDef(relu1_node_def), HashNodeDef(relu2_node_def));
  }

  void TestCountParameters() {
    TransformFuncContext context;
    context.params.insert({"foo", {"a", "b"}});
    context.params.insert({"bar", {"c"}});
    EXPECT_EQ(2, context.CountParameters("foo"));
    EXPECT_EQ(1, context.CountParameters("bar"));
    EXPECT_EQ(0, context.CountParameters("not_present"));
  }

  void TestGetOneStringParameter() {
    TransformFuncContext context;
    context.params.insert({"foo", {"a", "b"}});
    context.params.insert({"bar", {"c"}});
    string value;
    TF_EXPECT_OK(context.GetOneStringParameter("bar", "d", &value));
    EXPECT_EQ("c", value);
    EXPECT_FALSE(context.GetOneStringParameter("foo", "d", &value).ok());
    TF_EXPECT_OK(context.GetOneStringParameter("not_present", "d", &value));
    EXPECT_EQ("d", value);
  }

  void TestGetOneInt32Parameter() {
    TransformFuncContext context;
    context.params.insert({"foo", {"10", "20"}});
    context.params.insert({"bar", {"-23"}});
    context.params.insert({"not_a_number", {"not_numerical"}});
    context.params.insert({"float", {"-23.232323"}});
    int32_t value;
    TF_EXPECT_OK(context.GetOneInt32Parameter("bar", 0, &value));
    EXPECT_EQ(-23, value);
    EXPECT_FALSE(context.GetOneInt32Parameter("foo", 0, &value).ok());
    TF_EXPECT_OK(context.GetOneInt32Parameter("not_present", 10, &value));
    EXPECT_EQ(10, value);
    EXPECT_FALSE(context.GetOneInt32Parameter("not_a_number", 0, &value).ok());
    EXPECT_FALSE(context.GetOneInt32Parameter("float", 0, &value).ok());
  }

  void TestGetOneInt64Parameter() {
    TransformFuncContext context;
    context.params.insert({"foo", {"10", "20"}});
    context.params.insert({"bar", {"-23"}});
    context.params.insert({"not_a_number", {"not_numerical"}});
    context.params.insert({"float", {"-23.232323"}});
    int64_t value;
    TF_EXPECT_OK(context.GetOneInt64Parameter("bar", 0, &value));
    EXPECT_EQ(-23, value);
    EXPECT_FALSE(context.GetOneInt64Parameter("foo", 0, &value).ok());
    TF_EXPECT_OK(context.GetOneInt64Parameter("not_present", 10, &value));
    EXPECT_EQ(10, value);
    EXPECT_FALSE(context.GetOneInt64Parameter("not_a_number", 0, &value).ok());
    EXPECT_FALSE(context.GetOneInt64Parameter("float", 0, &value).ok());
  }

  void TestGetOneFloatParameter() {
    TransformFuncContext context;
    context.params.insert({"foo", {"10.0", "20.0"}});
    context.params.insert({"bar", {"-23.2323"}});
    context.params.insert({"not_a_number", {"not_numerical"}});
    float value;
    TF_EXPECT_OK(context.GetOneFloatParameter("bar", 0, &value));
    EXPECT_NEAR(-23.2323f, value, 1e-5f);
    EXPECT_FALSE(context.GetOneFloatParameter("foo", 0, &value).ok());
    TF_EXPECT_OK(context.GetOneFloatParameter("not_present", 10.5f, &value));
    EXPECT_NEAR(10.5f, value, 1e-5f);
    EXPECT_FALSE(context.GetOneFloatParameter("not_a_number", 0, &value).ok());
  }

  void TestGetOneBoolParameter() {
    TransformFuncContext context;
    context.params.insert({"foo", {"true", "false"}});
    context.params.insert({"true", {"true"}});
    context.params.insert({"false", {"false"}});
    context.params.insert({"one", {"1"}});
    context.params.insert({"zero", {"0"}});
    context.params.insert({"not_a_bool", {"not_boolean"}});

    bool value;
    EXPECT_FALSE(context.GetOneBoolParameter("foo", 0, &value).ok());

    value = false;
    TF_EXPECT_OK(context.GetOneBoolParameter("true", false, &value));
    EXPECT_TRUE(value);

    value = true;
    TF_EXPECT_OK(context.GetOneBoolParameter("false", true, &value));
    EXPECT_FALSE(value);

    value = false;
    TF_EXPECT_OK(context.GetOneBoolParameter("one", false, &value));
    EXPECT_TRUE(value);

    value = true;
    TF_EXPECT_OK(context.GetOneBoolParameter("zero", true, &value));
    EXPECT_FALSE(value);

    EXPECT_FALSE(context.GetOneBoolParameter("not_a_bool", false, &value).ok());

    value = false;
    TF_EXPECT_OK(context.GetOneBoolParameter("not_present", true, &value));
    EXPECT_TRUE(value);
  }
};

TEST_F(TransformUtilsTest, TestMapNamesToNodes) { TestMapNamesToNodes(); }

TEST_F(TransformUtilsTest, TestMapNodesToOutputs) { TestMapNodesToOutputs(); }

TEST_F(TransformUtilsTest, TestNodeNamePartsFromInput) {
  TestNodeNamePartsFromInput();
}

TEST_F(TransformUtilsTest, TestCanonicalInputName) { TestCanonicalInputName(); }

TEST_F(TransformUtilsTest, TestAddNodeInput) { TestAddNodeInput(); }

TEST_F(TransformUtilsTest, TestCopyNodeAttr) { TestCopyNodeAttr(); }

TEST_F(TransformUtilsTest, TestSetNodeAttr) { TestSetNodeAttr(); }

TEST_F(TransformUtilsTest, TestSetNodeTensorAttr) { TestSetNodeTensorAttr(); }

TEST_F(TransformUtilsTest, TestSetNodeTensorAttrWithTensor) {
  TestSetNodeTensorAttrWithTensor();
}

TEST_F(TransformUtilsTest, TestGetNodeTensorAttr) { TestGetNodeTensorAttr(); }

TEST_F(TransformUtilsTest, TestNodeNameFromInput) { TestNodeNameFromInput(); }

TEST_F(TransformUtilsTest, TestFilterGraphDef) { TestFilterGraphDef(); }

TEST_F(TransformUtilsTest, TestRemoveAttributes) { TestRemoveAttributes(); }

TEST_F(TransformUtilsTest, TestGetOpTypeMatches) { TestGetOpTypeMatches(); }

TEST_F(TransformUtilsTest, TestGetOpTypeMatchesDAG) {
  TestGetOpTypeMatchesDAG();
}

TEST_F(TransformUtilsTest, TestReplaceMatchingOpTypes) {
  TestReplaceMatchingOpTypes();
}

TEST_F(TransformUtilsTest, TestMatchedNodesAsArray) {
  TestMatchedNodesAsArray();
}

TEST_F(TransformUtilsTest, TestRenameNodeInputs) { TestRenameNodeInputs(); }

TEST_F(TransformUtilsTest, TestRenameNodeInputsWithRedirects) {
  TestRenameNodeInputsWithRedirects();
}

TEST_F(TransformUtilsTest, TestRenameNodeInputsWithCycle) {
  TestRenameNodeInputsWithCycle();
}

TEST_F(TransformUtilsTest, TestRenameNodeInputsWithWildcard) {
  TestRenameNodeInputsWithWildcard();
}

TEST_F(TransformUtilsTest, TestRenameNodeInputsWithIgnores) {
  TestRenameNodeInputsWithIgnores();
}

TEST_F(TransformUtilsTest, TestFindInvalidInputs) { TestFindInvalidInputs(); }

TEST_F(TransformUtilsTest, TestIsGraphValid) { TestIsGraphValid(); }

TEST_F(TransformUtilsTest, TestGetInOutTypes) { TestGetInOutTypes(); }

TEST_F(TransformUtilsTest, TestCopyOriginalMatch) { TestCopyOriginalMatch(); }

TEST_F(TransformUtilsTest, TestHashNodeDef) { TestHashNodeDef(); }

TEST_F(TransformUtilsTest, TestCountParameters) { TestCountParameters(); }

TEST_F(TransformUtilsTest, TestGetOneStringParameter) {
  TestGetOneStringParameter();
}

TEST_F(TransformUtilsTest, TestGetOneInt32Parameter) {
  TestGetOneInt32Parameter();
}

TEST_F(TransformUtilsTest, TestGetOneInt64Parameter) {
  TestGetOneInt64Parameter();
}

TEST_F(TransformUtilsTest, TestGetOneFloatParameter) {
  TestGetOneFloatParameter();
}

TEST_F(TransformUtilsTest, TestGetOneBoolParameter) {
  TestGetOneBoolParameter();
}

}  // namespace graph_transforms
}  // namespace tensorflow
