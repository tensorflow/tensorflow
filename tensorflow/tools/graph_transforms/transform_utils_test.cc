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
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"

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
};

TEST_F(TransformUtilsTest, TestMapNamesToNodes) { TestMapNamesToNodes(); }

TEST_F(TransformUtilsTest, TestNodeNamePartsFromInput) {
  TestNodeNamePartsFromInput();
}

TEST_F(TransformUtilsTest, TestNodeNameFromInput) { TestNodeNameFromInput(); }

TEST_F(TransformUtilsTest, TestFilterGraphDef) { TestFilterGraphDef(); }

TEST_F(TransformUtilsTest, TestRemoveAttributes) { TestRemoveAttributes(); }

}  // namespace graph_transforms
}  // namespace tensorflow
